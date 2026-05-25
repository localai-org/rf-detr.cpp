# rt-detr.cpp Heads + End-to-End Detect Implementation Plan (Plan 6c of 11)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the final pieces of the forward pipeline — class head (per-query linear → class logits) and bbox head (3-layer MLP with sigmoid → normalized cxcywh boxes) — then implement image preprocessing (resize + ImageNet normalize) and replace `rfdetr_detect`'s NOT_IMPLEMENTED stub with the real forward+postprocess path. End state: `rfdetr-cli detect` on any PNG/JPEG produces real JSON detections (probably empty due to random-weight nonsense scores, but the pipeline runs end-to-end and the existing `test_cli_integration` confirms it). 95 parity checkpoints total (91 from 6a/6b + class.logits + bbox.pred + 2 model.{output_class,output_bbox} wrappers).

**Architecture:** Two new TUs — `src/heads.{cpp,hpp}` (class + bbox heads) and `src/rfdetr_model.{cpp,hpp}` (the glue function `rfdetr_model_forward(ctx, m, input) → { class_logits, bbox_pred }`). Image preprocessing lands in `src/image_io.cpp` (extending the existing module) since it's image processing — no new TU needed. `rfdetr_detect` in `src/rfdetr.cpp` gets rewritten to build the graph, set the preprocessed input, compute, extract logits/boxes, and call the existing `rfdetr_select_detections` from `src/postprocess.cpp` (built in Plan 1 Task 10, finally exercised for real).

**Tech Stack:** Same as Plans 1-6b (C++17, ggml CPU backend, numpy + gguf). One new vendored dep: `stb_image_resize.h` for image resizing (Plan 1 already vendors stb_image / stb_image_write / stb_truetype; adding the resizer keeps the pattern).

---

## Scope decisions

- **Heads compute per-query, not per-token.** Class and bbox heads consume `decoder.output` which is `(model_dim, num_queries)`. Output is `(num_classes, num_queries)` for class and `(4, num_queries)` for bbox. No CLS handling — decoder queries are already the right granularity.
- **Bbox head: 3-layer MLP with ReLU + final sigmoid.** Sigmoid is essential because postprocess expects `cxcywh ∈ [0, 1]`. Bbox MLP uses ReLU (not GELU) — matches common DETR implementations. Verify against PyTorch reference in Plan 7.
- **Class head: single linear, no activation.** Raw logits go to `rfdetr_select_detections` which applies sigmoid + top-k + threshold.
- **Image preprocessing: vendor `stb_image_resize.h`** for the resize. ImageNet normalize uses `preprocess.mean` and `preprocess.std` from GGUF metadata (loaded in Plan 2 into `Config::preprocess_mean[3]` and `preprocess_std[3]`).
- **`rfdetr_model_forward` returns a `ForwardOutput` struct** with `ggml_tensor*` for class_logits and bbox_pred. `rfdetr_detect` copies their data out and calls `rfdetr_select_detections`.
- **End-to-end test verifies JSON shape, not contents.** Random weights produce garbage detection scores. Existing `test_cli_integration` already passes for empty detections; after Plan 6c the JSON's `detections` array may have 0 or more entries depending on whether any score exceeds the default 0.5 threshold. Update the assertions to check the JSON is well-formed but not the specific contents.
- **No CLI flag changes.** `rfdetr-cli detect --model X --input Y --output Z` already works (Plan 1); Plan 6c just changes what `detect` actually computes internally.
- **Test rename**: `test_parity_backbone.cpp` → `test_parity_full_forward.cpp` at the end (in Task 6), since the test now exercises the full forward pipeline including heads.

---

## File map (created or modified in this plan)

```
rt-detr.cpp/
├── docs/
│   └── parity.md                       # MODIFY — add heads + model.* checkpoint rows
├── scripts/
│   └── gen_numpy_baseline.py           # MODIFY — add heads_forward + final model output
├── src/
│   ├── heads.{cpp,hpp}                 # NEW — class + bbox heads
│   ├── rfdetr_model.{cpp,hpp}          # NEW — full pipeline glue: backbone → ... → heads
│   ├── image_io.{cpp,hpp}              # MODIFY — add preprocess() (resize + normalize)
│   ├── rfdetr.cpp                      # MODIFY — rewrite rfdetr_detect to use the full pipeline
│   └── (other src files unchanged)
├── third_party/stb/
│   └── stb_image_resize.h              # NEW — vendored (single header)
├── tests/
│   ├── CMakeLists.txt                  # MODIFY — link new src files; test rename
│   ├── test_parity_full_forward.cpp    # RENAMED from test_parity_backbone.cpp; +heads + model checkpoints
│   ├── test_cli_integration.cpp        # MODIFY — assert JSON shape is well-formed regardless of detections
│   └── test_capi.cpp                   # MODIFY — assert rfdetr_detect now succeeds (RFDETR_OK), even if zero detections
└── README.md                           # MODIFY — Plan 6c status
```

---

### Task 1: Class head + bbox head

Implement both heads + numpy reference + parity. Class head is one linear; bbox head is a 3-layer MLP with ReLU + sigmoid. New checkpoints: `heads.class.logits`, `heads.bbox.fc1.output`, `heads.bbox.fc2.output`, `heads.bbox.fc3.output`, `heads.bbox.pred` (post-sigmoid).

**Files:**
- Create: `src/heads.hpp`, `src/heads.cpp`
- Modify: `CMakeLists.txt` — add `src/heads.cpp` to `RFDETR_SOURCES`
- Modify: `scripts/gen_numpy_baseline.py` — add `class_head_forward`, `bbox_head_forward`
- Modify: `tests/test_parity_backbone.cpp` — call heads after decoder; add 5 checkpoints
- Modify: `docs/parity.md` — new rows

### Step 1: Write `src/heads.hpp`

```cpp
#ifndef RFDETR_HEADS_HPP
#define RFDETR_HEADS_HPP

#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace rfdetr {

/* Class head: linear projection from per-query embedding to per-class logits.
 *
 *   Input  ne = (model_dim, num_queries, 1, 1)
 *   Output ne = (num_classes, num_queries, 1, 1)
 *
 * No activation — postprocess applies sigmoid + top-k + threshold.
 *
 * Publishes "heads.class.logits". */
ggml_tensor* class_head_forward(ggml_context* ctx, const Model& m,
                                ggml_tensor* decoder_out);

/* Bbox head: 3-layer MLP with ReLU between layers and sigmoid at the end.
 *
 *   fc1: model_dim → model_dim, ReLU
 *   fc2: model_dim → model_dim, ReLU
 *   fc3: model_dim → 4 (cx, cy, w, h)
 *   sigmoid → ne = (4, num_queries, 1, 1) in [0, 1]
 *
 * Publishes "heads.bbox.fc1.output", "heads.bbox.fc2.output",
 * "heads.bbox.fc3.output", "heads.bbox.pred". */
ggml_tensor* bbox_head_forward(ggml_context* ctx, const Model& m,
                               ggml_tensor* decoder_out);

}  // namespace rfdetr

#endif
```

### Step 2: Write `src/heads.cpp`

```cpp
#include "heads.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <string>

namespace rfdetr {

namespace {

ggml_tensor* get_tensor(const Model& m, const std::string& name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "heads: missing tensor '%s'", name.c_str());
        return nullptr;
    }
    return it->second;
}

ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x,
                    ggml_tensor* W, ggml_tensor* b) {
    /* Wb @ x + bias; matches existing ops::mlp convention */
    ggml_tensor* y = ggml_mul_mat(ctx, W, x);
    y = ggml_add(ctx, y, b);
    return y;
}

}  // namespace

ggml_tensor* class_head_forward(ggml_context* ctx, const Model& m,
                                ggml_tensor* decoder_out) {
    ggml_tensor* W = get_tensor(m, "heads.class.fc.weight");
    ggml_tensor* b = get_tensor(m, "heads.class.fc.bias");
    if (!W || !b) return nullptr;

    /* W ne = (model_dim, num_classes); decoder_out ne = (model_dim, num_queries)
     * mul_mat → ne = (num_classes, num_queries) */
    ggml_tensor* logits = linear(ctx, decoder_out, W, b);
    publish("heads.class.logits", logits);
    return logits;
}

ggml_tensor* bbox_head_forward(ggml_context* ctx, const Model& m,
                               ggml_tensor* decoder_out) {
    ggml_tensor* W1 = get_tensor(m, "heads.bbox.fc1.weight");
    ggml_tensor* b1 = get_tensor(m, "heads.bbox.fc1.bias");
    ggml_tensor* W2 = get_tensor(m, "heads.bbox.fc2.weight");
    ggml_tensor* b2 = get_tensor(m, "heads.bbox.fc2.bias");
    ggml_tensor* W3 = get_tensor(m, "heads.bbox.fc3.weight");
    ggml_tensor* b3 = get_tensor(m, "heads.bbox.fc3.bias");
    if (!W1 || !b1 || !W2 || !b2 || !W3 || !b3) return nullptr;

    /* fc1 → ReLU */
    ggml_tensor* h = linear(ctx, decoder_out, W1, b1);
    h = ggml_relu(ctx, h);
    publish("heads.bbox.fc1.output", h);

    /* fc2 → ReLU */
    h = linear(ctx, h, W2, b2);
    h = ggml_relu(ctx, h);
    publish("heads.bbox.fc2.output", h);

    /* fc3 (no activation here — sigmoid applied next) */
    h = linear(ctx, h, W3, b3);
    publish("heads.bbox.fc3.output", h);

    /* Sigmoid → (4, num_queries) in [0, 1] */
    ggml_tensor* pred = ggml_sigmoid(ctx, h);
    publish("heads.bbox.pred", pred);
    return pred;
}

}  // namespace rfdetr
```

**Verify ggml API**: confirm `ggml_relu` and `ggml_sigmoid` exist in v0.13.0. If `ggml_sigmoid` doesn't exist, the equivalent is `1 / (1 + exp(-x))` which can be composed via `ggml_neg`, `ggml_exp`, `ggml_add` (const), `ggml_div`. Or check for `ggml_silu`/etc as alternatives. Read `third_party/ggml/include/ggml.h` for the actual list.

### Step 3: Add `src/heads.cpp` to CMakeLists.txt

Append to `RFDETR_SOURCES`.

### Step 4: Extend numpy reference

In `scripts/gen_numpy_baseline.py`, add the head functions after `decoder_layer`:

```python
def class_head_forward(cfg, tensors, decoder_out):
    """Class head: single linear. decoder_out: (num_queries, model_dim).
    Returns: dict with heads.class.logits = (num_queries, num_classes)."""
    W = tensors["heads.class.fc.weight"]  # (num_classes, model_dim)
    b = tensors["heads.class.fc.bias"]    # (num_classes,)
    logits = decoder_out @ W.T + b        # (num_queries, num_classes)
    return {"heads.class.logits": logits.copy()}, logits


def bbox_head_forward(cfg, tensors, decoder_out):
    """Bbox head: 3-layer MLP with ReLU + sigmoid.
    Returns: (dict, bbox_pred) where bbox_pred is (num_queries, 4) in [0,1]."""
    W1 = tensors["heads.bbox.fc1.weight"]; b1 = tensors["heads.bbox.fc1.bias"]
    W2 = tensors["heads.bbox.fc2.weight"]; b2 = tensors["heads.bbox.fc2.bias"]
    W3 = tensors["heads.bbox.fc3.weight"]; b3 = tensors["heads.bbox.fc3.bias"]

    h = decoder_out @ W1.T + b1
    h = np.maximum(h, 0)  # ReLU
    out_dict = {"heads.bbox.fc1.output": h.copy()}

    h = h @ W2.T + b2
    h = np.maximum(h, 0)
    out_dict["heads.bbox.fc2.output"] = h.copy()

    h = h @ W3.T + b3
    out_dict["heads.bbox.fc3.output"] = h.copy()

    pred = 1.0 / (1.0 + np.exp(-h))  # sigmoid; (num_queries, 4)
    out_dict["heads.bbox.pred"] = pred.copy()
    return out_dict, pred
```

Wire into `forward()` after the decoder loop:

```python
    cls_out, class_logits = class_head_forward(cfg, tensors, q)
    out.update(cls_out)
    bbox_out, bbox_pred = bbox_head_forward(cfg, tensors, q)
    out.update(bbox_out)
```

### Step 5: Wire heads into C++ test

In `tests/test_parity_backbone.cpp`, add `#include "heads.hpp"` at top.

After the `decoder_forward` call:

```cpp
ggml_tensor* class_logits = rfdetr::class_head_forward(gctx, *m, dec);
RFDETR_ASSERT(class_logits != nullptr);
ggml_tensor* bbox_pred = rfdetr::bbox_head_forward(gctx, *m, dec);
RFDETR_ASSERT(bbox_pred != nullptr);

/* Keep both alive in the graph */
ggml_build_forward_expand(graph, class_logits);
ggml_build_forward_expand(graph, bbox_pred);
```

In `build_tolerances`, append:

```cpp
tol["heads.class.logits"]    = {1e-5f, 1e-4f};
tol["heads.bbox.fc1.output"] = {1e-5f, 1e-4f};
tol["heads.bbox.fc2.output"] = {1e-5f, 1e-4f};
tol["heads.bbox.fc3.output"] = {1e-5f, 1e-4f};
tol["heads.bbox.pred"]       = {1e-5f, 1e-4f};
```

### Step 6: Update docs/parity.md

Add 5 rows.

### Step 7: Build + run

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build -R test_parity_backbone --output-on-failure 2>&1 | tail -15
```

Expected: PASS. Class head is parity-proven (linear projection just like the projector). Bbox head's only new operators are `ggml_relu` and `ggml_sigmoid`; if either has a numerical convention mismatch with numpy (very unlikely for these elementwise ops), this is where it surfaces.

### Step 8: Commit

```bash
git add src/heads.hpp src/heads.cpp CMakeLists.txt \
        scripts/gen_numpy_baseline.py tests/test_parity_backbone.cpp docs/parity.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(heads): class + bbox heads

Class head: single linear projection (model_dim → num_classes) producing
raw logits. Bbox head: 3-layer MLP with ReLU between layers and sigmoid
at the end, producing normalized cxcywh in [0, 1] per query.

5 new parity checkpoints (heads.class.logits, heads.bbox.{fc1,fc2,fc3}.output,
heads.bbox.pred) all green at {1e-5, 1e-4} tolerance.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `rfdetr_model_forward` — full-pipeline glue

Wrap the entire forward pipeline (backbone → projector → encoder → decoder → heads) in a single `rfdetr_model_forward(ctx, m, input) → { class_logits, bbox_pred }` entry point. Subsequent tasks (`rfdetr_detect`) call this one function instead of duplicating the stitching.

**Files:**
- Create: `src/rfdetr_model.hpp`, `src/rfdetr_model.cpp`
- Modify: `CMakeLists.txt` — add `src/rfdetr_model.cpp`
- Modify: `tests/test_parity_backbone.cpp` — call `rfdetr_model_forward` instead of the chain

### Step 1: Write `src/rfdetr_model.hpp`

```cpp
#ifndef RFDETR_MODEL_HPP
#define RFDETR_MODEL_HPP

#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace rfdetr {

/* Result of the full forward pass.
 *
 *   class_logits ne = (num_classes, num_queries, 1, 1) — raw, pre-sigmoid
 *   bbox_pred    ne = (4, num_queries, 1, 1)            — post-sigmoid, in [0, 1] */
struct ForwardOutput {
    ggml_tensor* class_logits = nullptr;
    ggml_tensor* bbox_pred    = nullptr;
};

/* Run the full forward pipeline: backbone → projector → encoder → decoder → heads.
 *
 *   input ne = (W, H, 3, 1) F32 — already mean/std normalized in NCHW-equivalent layout
 *
 * Publishes every named checkpoint from the sub-pipelines plus
 * "model.class_logits" and "model.bbox_pred" wrappers. */
ForwardOutput rfdetr_model_forward(ggml_context* ctx, const Model& m,
                                   ggml_tensor* input);

}  // namespace rfdetr

#endif
```

### Step 2: Write `src/rfdetr_model.cpp`

```cpp
#include "rfdetr_model.hpp"
#include "dinov2.hpp"
#include "projector.hpp"
#include "encoder.hpp"
#include "decoder.hpp"
#include "heads.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

namespace rfdetr {

ForwardOutput rfdetr_model_forward(ggml_context* ctx, const Model& m,
                                   ggml_tensor* input) {
    ForwardOutput out;

    /* 1. Backbone — produces BackboneOutput { final, multi_scale[4] } */
    BackboneOutput bb = dinov2_forward(ctx, m, input);
    if (!bb.final) {
        rfdetr_logf(RFDETR_LOG_ERROR, "rfdetr_model_forward: backbone failed");
        return out;
    }

    /* 2. Projector — consumes multi_scale, produces (model_dim, 4*N) */
    ggml_tensor* projected = projector_forward(ctx, m, bb);
    if (!projected) return out;

    /* 3. Encoder — 3 layers on projector output */
    ggml_tensor* enc = encoder_forward(ctx, m, projected);
    if (!enc) return out;

    /* 4. Decoder — 3 layers with cross-attention against encoder; queries loaded inside */
    ggml_tensor* dec = decoder_forward(ctx, m, enc);
    if (!dec) return out;

    /* 5. Heads */
    out.class_logits = class_head_forward(ctx, m, dec);
    out.bbox_pred    = bbox_head_forward(ctx, m, dec);

    if (out.class_logits) publish("model.class_logits", out.class_logits);
    if (out.bbox_pred)    publish("model.bbox_pred",    out.bbox_pred);

    return out;
}

}  // namespace rfdetr
```

### Step 3: Add `src/rfdetr_model.cpp` to CMakeLists.txt

### Step 4: Simplify the parity test

In `tests/test_parity_backbone.cpp`, replace the chain of explicit `dinov2_forward → projector_forward → encoder_forward → decoder_forward → class_head_forward → bbox_head_forward` calls with a single `rfdetr_model_forward` call:

```cpp
#include "rfdetr_model.hpp"

// ... in main, replace the chain with:
rfdetr::ForwardOutput out = rfdetr::rfdetr_model_forward(gctx, *m, input);
RFDETR_ASSERT(out.class_logits != nullptr);
RFDETR_ASSERT(out.bbox_pred    != nullptr);

ggml_build_forward_expand(graph, out.class_logits);
ggml_build_forward_expand(graph, out.bbox_pred);
```

Add tolerances for the two model.* checkpoints:

```cpp
tol["model.class_logits"] = {1e-5f, 1e-4f};
tol["model.bbox_pred"]    = {1e-5f, 1e-4f};
```

(These should match `heads.class.logits` and `heads.bbox.pred` exactly since they're the same tensors with different publish names.)

### Step 5: Update numpy

In `scripts/gen_numpy_baseline.py::forward()`, after the bbox head publishes, add:

```python
    out["model.class_logits"] = class_logits.copy()
    out["model.bbox_pred"] = bbox_pred.copy()
```

### Step 6: Build + run

Expected: still green; all 97 checkpoints (91 + 5 heads + 2 model.* wrappers — but model.* are duplicates so effectively 96 unique) match.

### Step 7: Commit

```bash
git add src/rfdetr_model.hpp src/rfdetr_model.cpp CMakeLists.txt \
        scripts/gen_numpy_baseline.py tests/test_parity_backbone.cpp docs/parity.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(model): rfdetr_model_forward — full forward pipeline glue

Wraps backbone → projector → encoder → decoder → heads into a single
entry point returning ForwardOutput { class_logits, bbox_pred }.
rfdetr_detect (Task 4) calls this; the parity test simplifies to one
call instead of stitching six pipeline stages by hand.

Two new "model.*" wrapper checkpoints duplicate heads' outputs under
their final-pipeline names; useful for downstream callers that want a
stable name regardless of which head produced them.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Image preprocessing — resize + ImageNet normalize

Add an `rfdetr_preprocess()` function that takes an `rfdetr_image*` (uint8 HWC RGB at arbitrary resolution) and produces a F32 tensor in the ggml `(W, H, 3, 1)` layout expected by `dinov2_patch_embed`. Vendors `stb_image_resize.h` for the resize.

**Files:**
- Create: `third_party/stb/stb_image_resize.h` (vendored)
- Modify: `src/image_io.hpp` — declare `rfdetr_preprocess`
- Modify: `src/image_io.cpp` — implement (uses STB_IMAGE_RESIZE_IMPLEMENTATION)
- Modify: `tests/test_image_io.cpp` — add a preprocess test

### Step 1: Vendor `stb_image_resize.h`

```bash
curl -fsSL https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h \
    -o third_party/stb/stb_image_resize.h
sha256sum third_party/stb/stb_image_resize.h
```

(stb's current resize header is `stb_image_resize2.h`; we vendor it as `stb_image_resize.h` to keep the include path stable. If the upstream renames or version changes, document the actual sha256 in the commit message.)

### Step 2: Declare `rfdetr_preprocess` in `src/image_io.hpp`

Add inside the existing `extern "C"` block:

```c
/* Preprocess an image for model input:
 *   1. Resize to (target_w, target_h) using high-quality bilinear interpolation
 *   2. Convert uint8 RGB → float32 in [0, 1]
 *   3. Apply ImageNet normalization: (pixel - mean) / std (per channel)
 *   4. Output in (W, H, 3, 1) ggml layout, ready to ggml_backend_tensor_set
 *
 * The output buffer is allocated by the function; caller frees with std::free.
 *
 * Returns RFDETR_OK and fills *out_data + *out_w + *out_h on success.
 * Returns RFDETR_ERR_* on error. */
rfdetr_status rfdetr_preprocess(const rfdetr_image* img,
                                int target_w, int target_h,
                                const float mean[3], const float std_[3],
                                float** out_data, int* out_w, int* out_h);
```

### Step 3: Implement in `src/image_io.cpp`

Near the top, add:

```cpp
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
```

Append the implementation:

```cpp
extern "C" rfdetr_status rfdetr_preprocess(const rfdetr_image* img,
                                           int target_w, int target_h,
                                           const float mean[3], const float std_[3],
                                           float** out_data, int* out_w, int* out_h) {
    if (!img || !out_data || !out_w || !out_h || !mean || !std_) {
        return RFDETR_ERR_INVALID_ARG;
    }
    if (target_w <= 0 || target_h <= 0) return RFDETR_ERR_INVALID_ARG;

    /* 1. Resize via stb_image_resize. Input is uint8 RGB packed HWC. */
    std::vector<uint8_t> resized((size_t)target_w * target_h * 3);
    if (!stbir_resize_uint8_linear(img->rgb.data(), img->width, img->height, 0,
                                   resized.data(), target_w, target_h, 0,
                                   STBIR_RGB)) {
        rfdetr_logf(RFDETR_LOG_ERROR, "rfdetr_preprocess: stbir_resize failed");
        return RFDETR_ERR_IO;
    }

    /* 2-3. Build the output F32 buffer in ggml ne = (W, H, 3, 1) layout.
     *
     * ggml's column-major: ne[0]=W is fastest-varying, ne[1]=H, ne[2]=C=3, ne[3]=N=1.
     * Memory order: for each (n, c, h, w) the offset is n*CHW + c*HW + h*W + w.
     * That's NCHW row-major, which is also what we get if we interpret ne[0]=W
     * as the fastest axis (it is).
     *
     * Input pixels are HWC (uint8): for each (h, w, c) offset = h*W*3 + w*3 + c.
     * Output: F32 NCHW (n=0): offset = c*HW + h*W + w. We need to transpose. */
    const size_t n_elems = (size_t)target_w * target_h * 3;
    float* buf = (float*)std::malloc(n_elems * sizeof(float));
    if (!buf) return RFDETR_ERR_OUT_OF_MEMORY;

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < target_h; ++h) {
            for (int w = 0; w < target_w; ++w) {
                uint8_t px = resized[(h * target_w + w) * 3 + c];
                float v = (float)px / 255.0f;
                v = (v - mean[c]) / std_[c];
                /* NCHW layout: c*HW + h*W + w */
                buf[c * target_h * target_w + h * target_w + w] = v;
            }
        }
    }

    *out_data = buf;
    *out_w = target_w;
    *out_h = target_h;
    return RFDETR_OK;
}
```

**Verify `stb_image_resize2.h` API**: the function name (`stbir_resize_uint8_linear`) and pixel layout enum (`STBIR_RGB`) are from stb_image_resize2's current API. The original stb_image_resize.h had different function names (`stbir_resize_uint8_srgb`, etc.). If the vendored header uses different names, adapt. Document in your report.

### Step 4: Add a preprocess test

In `tests/test_image_io.cpp`, append a small test:

```cpp
/* Preprocess test: load cats.png (16×16), resize to 56×56, normalize, verify shape */
{
    rfdetr_image* img = rfdetr_image_load_file((fixtures + "/cats.png").c_str(), nullptr);
    RFDETR_ASSERT(img != nullptr);

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_[3] = {0.229f, 0.224f, 0.225f};

    float* data = nullptr;
    int w = 0, h = 0;
    rfdetr_status st = rfdetr_preprocess(img, 56, 56, mean, std_, &data, &w, &h);
    RFDETR_ASSERT_EQ_INT(st, RFDETR_OK);
    RFDETR_ASSERT_EQ_INT(w, 56);
    RFDETR_ASSERT_EQ_INT(h, 56);
    RFDETR_ASSERT(data != nullptr);

    /* Sanity check: values should be in normalized range (roughly [-3, 3] for ImageNet-style data) */
    bool all_zero = true;
    for (int i = 0; i < 56 * 56 * 3; ++i) {
        if (data[i] != 0.0f) { all_zero = false; break; }
    }
    RFDETR_ASSERT(!all_zero);  /* preprocess shouldn't zero everything */

    std::free(data);
    rfdetr_image_free(img);
}
```

### Step 5: Build + run

```bash
rm -rf build
cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=OFF 2>&1 | tail -3
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build --output-on-failure
```

Expected: 9/9 pass including the extended test_image_io. If `stbir_resize_uint8_linear` doesn't exist with that exact name in the vendored header, the build will fail and you'll need to find the right symbol.

### Step 6: Commit

```bash
git add third_party/stb/stb_image_resize.h \
        src/image_io.hpp src/image_io.cpp tests/test_image_io.cpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(image_io): rfdetr_preprocess — resize + ImageNet normalize

Adds image preprocessing for model input: high-quality bilinear resize via
stb_image_resize, then uint8→F32 normalize via ImageNet mean/std, output
in ggml's (W, H, 3, 1) NCHW-equivalent layout. Vendored
stb_image_resize.h (single header, public domain).

Plan 6c Task 4 wires this into rfdetr_detect.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Wire `rfdetr_detect` end-to-end

Replace the NOT_IMPLEMENTED stub in `src/rfdetr.cpp::rfdetr_detect` with the real path: preprocess image → build forward graph → compute → extract logits + boxes → call `rfdetr_select_detections` → return populated detection array.

**Files:**
- Modify: `src/rfdetr.cpp` — rewrite `rfdetr_detect`
- Modify: `tests/test_capi.cpp` — change the NOT_IMPLEMENTED assertion to RFDETR_OK
- Modify: `tests/test_cli_integration.cpp` — relax detection-content assertions (random weights produce nonsense)

### Step 1: Rewrite `rfdetr_detect`

Currently:

```cpp
extern "C" rfdetr_status rfdetr_detect(rfdetr_context* ctx,
                                       const rfdetr_image* img,
                                       const rfdetr_detect_params* params,
                                       rfdetr_detection** out_detections,
                                       size_t* out_n) {
    if (out_detections) *out_detections = nullptr;
    if (out_n)          *out_n = 0;
    if (!ctx || !img) return RFDETR_ERR_INVALID_ARG;
    return RFDETR_ERR_NOT_IMPLEMENTED;
}
```

Replace with:

```cpp
#include "rfdetr_model.hpp"
#include "postprocess.hpp"  /* rfdetr_select_detections from Plan 1 Task 10 */
#include "image_io.hpp"
#include "ggml-backend.h"

extern "C" rfdetr_status rfdetr_detect(rfdetr_context* ctx,
                                       const rfdetr_image* img,
                                       const rfdetr_detect_params* params,
                                       rfdetr_detection** out_detections,
                                       size_t* out_n) {
    if (out_detections) *out_detections = nullptr;
    if (out_n)          *out_n = 0;
    if (!ctx || !img || !params) return RFDETR_ERR_INVALID_ARG;
    if (!ctx->model || !ctx->backend) return RFDETR_ERR_INVALID_ARG;

    const rfdetr::Config& cfg = ctx->model->config;
    const int img_size = (int)cfg.image_size;

    /* 1. Preprocess image to (img_size, img_size, 3, 1) F32 */
    float* px_data = nullptr;
    int px_w = 0, px_h = 0;
    rfdetr_status pp_st = rfdetr_preprocess(img, img_size, img_size,
                                            cfg.preprocess_mean, cfg.preprocess_std,
                                            &px_data, &px_w, &px_h);
    if (pp_st != RFDETR_OK) {
        rfdetr_logf(RFDETR_LOG_ERROR, "rfdetr_detect: preprocess failed");
        return pp_st;
    }

    /* 2. Build ggml graph */
    ggml_init_params ip{};
    ip.mem_size = 64 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context* gctx = ggml_init(ip);
    if (!gctx) { std::free(px_data); return RFDETR_ERR_OUT_OF_MEMORY; }

    ggml_tensor* input = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, px_w, px_h, 3, 1);
    ggml_set_name(input, "input");

    rfdetr::ForwardOutput fout = rfdetr::rfdetr_model_forward(gctx, *ctx->model, input);
    if (!fout.class_logits || !fout.bbox_pred) {
        ggml_free(gctx);
        std::free(px_data);
        rfdetr_logf(RFDETR_LOG_ERROR, "rfdetr_detect: model forward failed");
        return RFDETR_ERR_INFERENCE;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, ctx->backend);
    if (!buf) {
        ggml_free(gctx);
        std::free(px_data);
        return RFDETR_ERR_OUT_OF_MEMORY;
    }

    /* 3. Set input data and compute */
    ggml_backend_tensor_set(input, px_data, 0, (size_t)px_w * px_h * 3 * sizeof(float));
    std::free(px_data); px_data = nullptr;

    ggml_cgraph* graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, fout.class_logits);
    ggml_build_forward_expand(graph, fout.bbox_pred);

    auto status = ggml_backend_graph_compute(ctx->backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(gctx);
        return RFDETR_ERR_INFERENCE;
    }
    ggml_backend_synchronize(ctx->backend);

    /* 4. Copy logits + boxes back to host */
    const size_t n_logits = ggml_nelements(fout.class_logits);
    const size_t n_boxes  = ggml_nelements(fout.bbox_pred);
    std::vector<float> logits_buf(n_logits);
    std::vector<float> boxes_buf(n_boxes);
    ggml_backend_tensor_get(fout.class_logits, logits_buf.data(), 0, n_logits * sizeof(float));
    ggml_backend_tensor_get(fout.bbox_pred,    boxes_buf.data(),  0, n_boxes  * sizeof(float));

    /* 5. Call existing postprocess. Note: logits ne = (num_classes, num_queries)
     *    but rfdetr_select_detections expects flat (num_queries * num_classes)
     *    row-major (query, class). The ggml column-major layout where
     *    ne[0]=num_classes, ne[1]=num_queries already gives us a memory layout
     *    where the fastest-varying axis is classes — i.e. row-major (query, class).
     *    Same for boxes: ne[0]=4, ne[1]=num_queries → row-major (query, 4). */
    const size_t num_queries = (size_t)fout.class_logits->ne[1];
    const size_t num_classes = (size_t)fout.class_logits->ne[0];

    rfdetr_select_detections(logits_buf.data(), boxes_buf.data(),
                             num_queries, num_classes,
                             params->threshold, params->top_k,
                             params->class_filter, params->class_filter_len,
                             rfdetr_image_width(img), rfdetr_image_height(img),
                             out_detections, out_n);

    /* 6. Attach class names from the loaded config (best-effort) */
    if (out_detections && *out_detections) {
        const auto& names = ctx->model->config.class_names;
        for (size_t i = 0; i < *out_n; ++i) {
            uint32_t cid = (*out_detections)[i].class_id;
            if (cid < names.size()) {
                (*out_detections)[i].class_name = names[cid].c_str();
            }
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    return RFDETR_OK;
}
```

Add include `#include <vector>` if not already in the file.

### Step 2: Update `tests/test_capi.cpp`

Find the assertion:

```cpp
RFDETR_ASSERT_EQ_INT(det_st, RFDETR_ERR_NOT_IMPLEMENTED);
RFDETR_ASSERT_EQ_INT(n, 0);
RFDETR_ASSERT(dets == nullptr);
```

Replace with:

```cpp
RFDETR_ASSERT_EQ_INT(det_st, RFDETR_OK);
/* Random weights → garbage scores → typically zero detections above 0.5 threshold,
 * but the call must succeed regardless. */
RFDETR_ASSERT(n >= 0);  /* trivially true; documents intent */
/* If n > 0, the detection array must be non-null and freeable. */
if (n > 0) {
    RFDETR_ASSERT(dets != nullptr);
    rfdetr_detections_free(dets, n);
}
```

### Step 3: Update `tests/test_cli_integration.cpp`

The existing test asserts the JSON contains `"detections": ["` and `"width": 16`. These still hold. The only change needed: the test image is 16×16 but the model expects `image_size=56` (fixture). The CLI's `cmd_detect` already passes the original image to `rfdetr_detect`, which preprocesses it internally. So the JSON's `"image": {"width": 16, "height": 16}` will reflect the ORIGINAL image size — which is what we want for downstream coordinates.

Test should already pass without changes. Verify.

### Step 4: Build + run

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build --output-on-failure
```

Expected: all 10 tests pass (assuming CLI=ON). `test_capi` now exercises the real `rfdetr_detect` path; `test_cli_integration` runs the full end-to-end pipeline through the CLI binary.

If `test_cli_integration` fails because the JSON contains detection entries with `class_name` pointers that look wrong in the JSON output: the CLI's JSON writer in `examples/cli/main.cpp` currently doesn't output `class_name`. Verify and adjust if needed.

### Step 5: Commit

```bash
git add src/rfdetr.cpp tests/test_capi.cpp tests/test_cli_integration.cpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(api): rfdetr_detect end-to-end — preprocess + forward + postprocess

Replaces the NOT_IMPLEMENTED stub. Now: preprocesses the input image
(resize + ImageNet normalize), builds the full forward graph via
rfdetr_model_forward, sets the preprocessed data, runs ggml compute,
copies class_logits and bbox_pred back to host, calls Plan 1's
rfdetr_select_detections to filter/sort/top-k/project boxes, and returns
the populated detection array.

With random-weight fixture, detections are mostly empty (nonsense scores
rarely exceed 0.5 threshold) — the pipeline runs end-to-end and the API
contract holds. test_capi and test_cli_integration updated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Test rename + final smoke + README

`test_parity_backbone.cpp` no longer just covers the backbone — it covers the full forward pipeline. Rename to `test_parity_full_forward.cpp`. Final triple-config smoke + README.

**Files:**
- Rename: `tests/test_parity_backbone.cpp` → `tests/test_parity_full_forward.cpp`
- Modify: `tests/CMakeLists.txt` — update target name + custom_target name
- Modify: `tests/test_parity_full_forward.cpp` — update file-header comment
- Modify: `README.md` — Plan 6c status

### Step 1: Rename via git mv

```bash
git mv tests/test_parity_backbone.cpp tests/test_parity_full_forward.cpp
```

### Step 2: Update tests/CMakeLists.txt

Find these lines:
- `rfdetr_add_test(test_parity_backbone)`
- `add_dependencies(test_parity_backbone ...)`
- Custom command OUTPUT `baseline_backbone.gguf` (if it was renamed in Plan 4)
- `add_custom_target(rfdetr_baseline_backbone ...)`

Replace `backbone` with `full_forward` in test target names. Keep `baseline_backbone.gguf` filename as-is to avoid yet another rename (the file name doesn't need to match the test target name).

### Step 3: Update file-header comment

`tests/test_parity_full_forward.cpp` first line is something like:

```cpp
/* C++ forward pass through the full DINOv2 backbone (patch_embed +
 * CLS/pos_embed + 12 blocks + final norm + multi-scale taps) vs numpy
 * baseline. */
```

Replace with:

```cpp
/* C++ full forward pass (backbone + projector + encoder + decoder + heads) vs
 * numpy baseline. All 97 parity checkpoints across the pipeline. */
```

### Step 4: Triple-config smoke

```bash
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=OFF -DRFDETR_BUILD_CLI=OFF 2>&1 | tail -3 && cmake --build build -j 2>&1 | tail -3
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=OFF 2>&1 | tail -3 && cmake --build build -j 2>&1 | tail -3 && ctest --test-dir build --output-on-failure 2>&1 | tail -3
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=ON 2>&1 | tail -3 && cmake --build build -j 2>&1 | tail -5 && ctest --test-dir build --output-on-failure
```

Expected: all green; 10/10 tests with CLI=ON.

### Step 5: Update README.md

```markdown
## Status

**End-to-end detection (Plan 6c) complete.** `rfdetr-cli detect` now runs
the full RF-DETR forward pipeline — image preprocessing (resize + ImageNet
normalize), DINOv2 backbone (windowed + global attention), multi-scale
projector, 3-layer encoder, 3-layer decoder (300 queries), class + bbox
heads — and emits real JSON detections via Plan 1's postprocess. With the
synthesized random-weight fixture, detection scores rarely exceed the 0.5
threshold (output is typically empty), but the C++ pipeline is fully
exercised end-to-end. 97 parity checkpoints all green at 1e-5 absolute
tolerance against the numpy reference. Ten tests pass on a clean build.

The Python conversion script body (`scripts/convert_rfdetr_to_gguf.py`) is
still deferred — see Plan 2 Task 3. Once a real upstream `rfdetr-base`
checkpoint is converted to GGUF, `rfdetr-cli detect` will produce
meaningful detections.

Plan 7 swaps the numpy reference for a torch+rfdetr baseline (verifying
against the real model). Plan 8 adds Q8_0 quantization. Plan 9 adds the
nano/small/medium/large variants.
```

### Step 6: Commit

```bash
git add tests/test_parity_full_forward.cpp tests/CMakeLists.txt README.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
chore: rename test_parity_backbone → test_parity_full_forward; Plan 6c README

Test covers the full forward pipeline now (backbone + projector + encoder
+ decoder + heads), not just the backbone. README Status updated to mark
Plan 6c (end-to-end detect) complete.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- §3 Architecture (heads + end-to-end forward) — covered.
- §6 API — `rfdetr_detect` no longer NOT_IMPLEMENTED.
- §8 Tests — 97 parity checkpoints + end-to-end CLI integration.
- §9 Parity — full pipeline documented.

**Risk areas:**
1. **`ggml_sigmoid` availability** — if not in v0.13.0, compose manually. Most likely to surface in Task 1.
2. **`stb_image_resize.h` API** — current upstream is `stb_image_resize2.h` with different function names. Adapt to whichever you vendor.
3. **NCHW vs ggml-style input layout** — preprocess produces NCHW row-major; ggml's `ggml_new_tensor_4d(W, H, 3, 1)` interprets that same memory as `(W, H, 3, 1)` column-major-style (W fastest). For a (1, 3, H, W) NCHW buffer, the memory order is `n*CHW + c*HW + h*W + w`, which has `w` fastest. That matches ggml ne=(W, H, C, N) where ne[0]=W is fastest. ✓
4. **`rfdetr_select_detections` data layout** — expects logits in `(num_queries * num_classes)` row-major with classes fastest. ggml ne=(num_classes, num_queries) puts classes at ne[0]=fastest → matches. ✓
5. **CLI JSON output** — `cmd_detect` in `examples/cli/main.cpp` currently iterates `dets[i]` and outputs the fields; it doesn't output `class_name`. If anyone wants class_name in JSON, that's a separate change (defer to follow-up polish).

---

## Next plans

After this plan lands:

- **Plan 7** — Replace numpy with PyTorch+rfdetr reference. Verifies the convention choices made in Plans 3-6 against the real upstream model. Tightens or loosens tolerances based on torch vs numpy precision differences.
- **Plan 8** — Quantization (Q8_0). `rfdetr-quantize` binary. Weight-only quant; activations stay F32.
- **Plan 9** — Variants nano/small/medium/large. Per-variant config files, conversion script extensions, additional parity bundles.
