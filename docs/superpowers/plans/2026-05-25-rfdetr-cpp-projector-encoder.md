# rt-detr.cpp Projector + Encoder Implementation Plan (Plan 6a of 11)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `dinov2_forward` to return a structured `BackboneOutput` exposing both the final post-norm tensor and the 4 multi-scale features. Implement the multi-scale projector (per-level linear projection + level embedding + concat into a single token sequence) and the transformer encoder (3 layers of self-attn + FFN, pre-LN). Add numpy reference and parity checkpoints for each new component. End state: parity-green C++ pipeline through `projector.output` and `encoder.output`, plus all per-encoder-layer intermediates, ready for Plan 6b's decoder to consume.

**Architecture:** Two new TUs — `src/projector.{cpp,hpp}` and `src/encoder.{cpp,hpp}`. `dinov2_forward` API changes to return `BackboneOutput` (multi_scale features collected explicitly instead of leaking only through the trace callback). The encoder reuses the existing `layer_norm`/`mha`/`mlp` helpers from `dinov2.cpp` — those get promoted from the anonymous namespace into a shared `src/transformer_ops.{cpp,hpp}` so encoder/decoder/heads can call them. Numpy reference grows two functions (`projector_forward`, `encoder_forward`) and the parity test's checkpoint table extends.

**Tech Stack:** Same as Plans 1-5 (C++17, ggml CPU backend, numpy + gguf). No new dependencies.

---

## Scope decisions

- **`BackboneOutput` struct first commit** — Plan 4 and Plan 5 reviewers both recommended this. Refactor before adding new modules so projector consumes the structured features directly.
- **Promote `layer_norm`/`mha`/`mlp` out of `dinov2.cpp`'s anonymous namespace** into a new `src/transformer_ops.{cpp,hpp}`. Encoder/decoder/heads need them. Three small files become a shared utility module.
- **Multi-scale concat first, then encoder** — the encoder consumes a single token sequence built from the 4 multi-scale features concatenated along the token axis. After projection, each level becomes `(model_dim, N_level)` and they concatenate to `(model_dim, sum_of_N)`. Add level embeddings (one per level) to distinguish them.
- **Encoder is 3 layers of pre-LN self-attention + FFN** — same shape family as DINOv2 blocks but operating on `encoder.model_dim` (= 64 in the fixture) and over the concatenated multi-scale token sequence. No window attention here.
- **No positional embedding on encoder input** — DETR-style encoders apply spatial positional encoding inside attention via Q/K key modulation, but rfdetr's variant simplifies this by relying on the multi-scale level embeddings + the backbone's positional embedding flowing through. Plan 7 (real PyTorch baseline) will verify the exact convention.
- **Multi-scale token count check.** Fixture image_size=56, patch=14 → backbone produces (4×4 + 1 CLS) = 17 tokens per multi-scale level. We strip CLS before projection (multi-scale features are spatial; CLS is a separate non-spatial slot). So per-level N=16. Concat 4 levels = 64 tokens to the encoder.
- **Parity checkpoints added:** `projector.level{0..3}.output`, `projector.concat.output`, `encoder.layer{0..2}.{norm1,attn,mlp,output}`, `encoder.output`. = 4 + 1 + 12 + 1 = 18 new checkpoints (total 73).

---

## File map (created or modified in this plan)

```
rt-detr.cpp/
├── docs/
│   └── parity.md                       # MODIFY — add encoder + projector checkpoint rows
├── scripts/
│   └── gen_numpy_baseline.py           # MODIFY — add projector_forward, encoder_forward
├── src/
│   ├── transformer_ops.{cpp,hpp}       # NEW — layer_norm, mha, mlp promoted from dinov2.cpp
│   ├── projector.{cpp,hpp}             # NEW — multi-scale projector
│   ├── encoder.{cpp,hpp}               # NEW — 3-layer transformer encoder
│   └── dinov2.{cpp,hpp}                # MODIFY — BackboneOutput struct return type;
│                                       #          delete layer_norm/mha/mlp duplicates
├── tests/
│   ├── CMakeLists.txt                  # MODIFY — link new src files
│   └── test_parity_backbone.cpp        # MODIFY — extend checkpoints; rename to test_parity_full_forward?
│                                       #          (leave name for now; rename when Plan 6c adds decoder/heads)
└── README.md                           # MODIFY — Plan 6a status
```

---

### Task 1: `BackboneOutput` struct + extract `transformer_ops`

Two refactors in one commit: change `dinov2_forward` return type to `BackboneOutput`, and move `layer_norm`/`mha`/`mlp` from `dinov2.cpp`'s anonymous namespace into a new `src/transformer_ops.{cpp,hpp}` so subsequent tasks can call them from projector / encoder code.

**Files:**
- Create: `src/transformer_ops.hpp`, `src/transformer_ops.cpp`
- Modify: `src/dinov2.{cpp,hpp}` — declare `BackboneOutput`; return it from `dinov2_forward`; delete the local helpers (now in transformer_ops)
- Modify: `CMakeLists.txt` — add `src/transformer_ops.cpp` to `RFDETR_SOURCES`
- Modify: `tests/test_parity_backbone.cpp` — call site adjusts to consume `BackboneOutput`

### Step 1: Write `src/transformer_ops.hpp`

```cpp
#ifndef RFDETR_TRANSFORMER_OPS_HPP
#define RFDETR_TRANSFORMER_OPS_HPP

struct ggml_context;
struct ggml_tensor;

namespace rfdetr::ops {

/* LayerNorm with affine: y = (x - mean) / sqrt(var + eps) * weight + bias.
 * Operates on the last axis (ggml_norm normalizes along ne[0]). eps = 1e-5
 * (PyTorch torch.nn.LayerNorm default). */
ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* weight, ggml_tensor* bias);

/* Multi-head self-attention with packed QKV projection (global, no windowing).
 *
 *   Input x ne = (dim, N, 1, 1)
 *   Wqkv ne   = (dim, 3*dim)
 *   bqkv ne   = (3*dim,)
 *   Wproj ne  = (dim, dim)
 *   bproj ne  = (dim,)
 *
 * Output ne = (dim, N, 1, 1). */
ggml_tensor* mha(ggml_context* ctx, ggml_tensor* x,
                 ggml_tensor* Wqkv, ggml_tensor* bqkv,
                 ggml_tensor* Wproj, ggml_tensor* bproj,
                 int n_heads);

/* Position-wise feed-forward network: x -> fc1 -> erf-GELU -> fc2.
 *
 *   W1 ne = (dim, ffn_dim); b1 ne = (ffn_dim,)
 *   W2 ne = (ffn_dim, dim); b2 ne = (dim,)
 *
 * Output ne = same as input. */
ggml_tensor* mlp(ggml_context* ctx, ggml_tensor* x,
                 ggml_tensor* W1, ggml_tensor* b1,
                 ggml_tensor* W2, ggml_tensor* b2);

}  // namespace rfdetr::ops

#endif
```

### Step 2: Write `src/transformer_ops.cpp`

Copy the implementations of `layer_norm`, `mha`, `mlp` from `src/dinov2.cpp`'s anonymous namespace into this file. Wrap in `namespace rfdetr::ops`. Each function's body is unchanged; only the namespace changes.

```cpp
#include "transformer_ops.hpp"

#include "ggml.h"

#include <cmath>

namespace rfdetr::ops {

ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* weight, ggml_tensor* bias) {
    constexpr float eps = 1e-5f;
    ggml_tensor* y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, weight);
    y = ggml_add(ctx, y, bias);
    return y;
}

ggml_tensor* mha(ggml_context* ctx, ggml_tensor* x,
                 ggml_tensor* Wqkv, ggml_tensor* bqkv,
                 ggml_tensor* Wproj, ggml_tensor* bproj,
                 int n_heads) {
    /* Paste the body verbatim from dinov2.cpp's mha() — including the comments. */
}

ggml_tensor* mlp(ggml_context* ctx, ggml_tensor* x,
                 ggml_tensor* W1, ggml_tensor* b1,
                 ggml_tensor* W2, ggml_tensor* b2) {
    /* Paste the body verbatim from dinov2.cpp's mlp(). */
}

}  // namespace rfdetr::ops
```

### Step 3: Update `src/dinov2.hpp`

Add the `BackboneOutput` struct above `dinov2_forward`'s declaration:

```cpp
#include <array>

namespace rfdetr {

/* Outputs of the DINOv2 backbone.
 *
 * - `final`: post-norm tensor ne=(dim, N+1) where N+1 is patches+CLS.
 * - `multi_scale`: 4 tap features at the indices in backbone.multi_scale_layers.
 *   Each tap is the block output BEFORE the final norm — ne=(dim, N+1).
 *   The projector strips CLS and projects to encoder.model_dim. */
struct BackboneOutput {
    ggml_tensor* final = nullptr;
    std::array<ggml_tensor*, 4> multi_scale{nullptr, nullptr, nullptr, nullptr};
};

BackboneOutput dinov2_forward(ggml_context* ctx, const Model& m,
                              ggml_tensor* input);

}  // namespace rfdetr
```

(Update the existing `dinov2_forward` declaration to return `BackboneOutput`. Drop the comment about "multi-scale not returned explicitly".)

### Step 4: Update `src/dinov2.cpp`

Two changes:
1. **Delete the anonymous-namespace `layer_norm`, `mha`, `mlp`** (now in transformer_ops). Add `#include "transformer_ops.hpp"` and replace each call to `layer_norm(...)` etc. inside `mha_window` and `dinov2_block` and `dinov2_final_norm` with `ops::layer_norm(...)`, `ops::mha(...)`, `ops::mlp(...)`.
2. **Change `dinov2_forward` return type** to `BackboneOutput`. Inside the function, collect the multi-scale features into a `std::array<ggml_tensor*, 4>` as you publish them, and return both in the struct.

Concretely, replace the existing forward loop:

```cpp
const auto& ms = m.config.backbone.multi_scale_layers;
auto find_ms_level = [&](uint32_t block_i) -> int { /* ... */ };

BackboneOutput out;
for (uint32_t i = 0; i < m.config.backbone.depth; ++i) {
    t = dinov2_block(ctx, m, t, (int)i);
    if (!t) return out;  /* nulls inside the struct */
    int level = find_ms_level(i);
    if (level >= 0 && level < (int)out.multi_scale.size()) {
        publish("backbone.multiscale.level" + std::to_string(level), t);
        out.multi_scale[level] = t;  /* hold a pointer to the tensor for projector use */
    }
}
out.final = dinov2_final_norm(ctx, m, t);
return out;
```

### Step 5: Update `CMakeLists.txt`

Add `src/transformer_ops.cpp` to `RFDETR_SOURCES`:

```cmake
set(RFDETR_SOURCES
    src/common.cpp
    src/image_io.cpp
    src/postprocess.cpp
    src/visualize.cpp
    src/cli.cpp
    src/model_loader.cpp
    src/rfdetr.cpp
    src/rfdetr_capi.cpp
    src/trace.cpp
    src/backend.cpp
    src/dinov2.cpp
    src/transformer_ops.cpp
)
```

### Step 6: Update `tests/test_parity_backbone.cpp`

Find the line `ggml_tensor* t = rfdetr::dinov2_forward(gctx, *m, input);`. Change to:

```cpp
rfdetr::BackboneOutput bb = rfdetr::dinov2_forward(gctx, *m, input);
RFDETR_ASSERT(bb.final != nullptr);
ggml_tensor* t = bb.final;
```

For now we just consume the final tensor; the multi_scale array is unused by the test (subsequent tasks will use it). Verify all 55 existing checkpoints still pass.

### Step 7: Build and run

```bash
rm -rf build
cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=OFF 2>&1 | tail -3
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure
```

Expected: 9/9 pass; test_parity_backbone unchanged (55 checkpoints, same max_abs values as before).

### Step 8: Commit

```bash
git add CMakeLists.txt src/transformer_ops.hpp src/transformer_ops.cpp \
        src/dinov2.hpp src/dinov2.cpp tests/test_parity_backbone.cpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
refactor: BackboneOutput struct + extract transformer_ops module

Plan 4 and Plan 5 reviewers both recommended returning the multi-scale
features from dinov2_forward explicitly instead of leaking them only via
the trace callback. New BackboneOutput { final, multi_scale[4] }; Plan 6a's
projector (next task) consumes both.

Same commit promotes layer_norm/mha/mlp out of dinov2.cpp's anonymous
namespace into a shared src/transformer_ops.{cpp,hpp} so encoder/decoder/
heads (Plans 6a-c) can call them without duplication. Bodies are unchanged.

55 parity checkpoints still green; this is a pure refactor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Projector — declarations and per-level math

The projector takes each of the 4 multi-scale features (each `(dim, N+1)` = `(64, 17)` in the fixture), strips CLS, projects to encoder.model_dim, adds a per-level embedding, and concatenates along the token axis into `(encoder.model_dim, 4 * N)` = `(64, 64)`.

Note: in the fixture `encoder.model_dim = 64` (Plan 4 shrink) which happens to equal `backbone.dim`. In production both can differ.

**Files:**
- Create: `src/projector.hpp`, `src/projector.cpp`
- Modify: `CMakeLists.txt` — add `src/projector.cpp` to `RFDETR_SOURCES`
- Modify: `scripts/gen_numpy_baseline.py` — add `projector_forward()`
- Modify: `tests/test_parity_backbone.cpp` — call projector after backbone; add 5 new checkpoints
- Modify: `docs/parity.md` — new rows

### Step 1: Write `src/projector.hpp`

```cpp
#ifndef RFDETR_PROJECTOR_HPP
#define RFDETR_PROJECTOR_HPP

#include "dinov2.hpp"
#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace rfdetr {

/* Multi-scale projector:
 *   For each level j in 0..3:
 *     1. Strip CLS from backbone multi-scale feature j (keep patches only)
 *     2. Linear project: y = Wj @ patches + bj
 *        Wj ne = (backbone.dim, encoder.model_dim)
 *     3. Add level embedding: y += level_embed[j]
 *   Concatenate the 4 projected sequences along the token axis.
 *
 * Output ne = (encoder.model_dim, 4 * N_patches, 1, 1)
 *
 * Publishes:
 *   projector.level{0..3}.output  - per-level pre-concat tensor (model_dim, N_patches)
 *   projector.concat.output       - final concatenated tensor */
ggml_tensor* projector_forward(ggml_context* ctx, const Model& m,
                               const BackboneOutput& bb);

}  // namespace rfdetr

#endif
```

### Step 2: Write `src/projector.cpp`

```cpp
#include "projector.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <string>

namespace rfdetr {

ggml_tensor* projector_forward(ggml_context* ctx, const Model& m,
                               const BackboneOutput& bb) {
    const int n_levels = (int)m.config.backbone.multi_scale_layers.size();
    if (n_levels <= 0 || n_levels > (int)bb.multi_scale.size()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "projector_forward: invalid n_levels %d", n_levels);
        return nullptr;
    }

    auto get_tensor = [&](const std::string& name) -> ggml_tensor* {
        auto it = m.tensors.find(name);
        if (it == m.tensors.end()) {
            rfdetr_logf(RFDETR_LOG_ERROR, "projector_forward: missing tensor '%s'", name.c_str());
            return nullptr;
        }
        return it->second;
    };

    ggml_tensor* level_embed = get_tensor("projector.level_embed");
    if (!level_embed) return nullptr;
    /* level_embed ne in the fixture: (encoder.model_dim, n_levels) — one
     * encoder.model_dim vector per level. */

    ggml_tensor* projected[4] = {nullptr, nullptr, nullptr, nullptr};

    for (int j = 0; j < n_levels; ++j) {
        ggml_tensor* feat = bb.multi_scale[j];
        if (!feat) {
            rfdetr_logf(RFDETR_LOG_ERROR, "projector_forward: missing multi_scale[%d]", j);
            return nullptr;
        }
        /* feat ne = (backbone.dim, N+1, 1, 1). Strip CLS (axis 1, position 0). */
        const int dim = (int)feat->ne[0];
        const int N1  = (int)feat->ne[1];
        const int N   = N1 - 1;
        ggml_tensor* patches = ggml_view_2d(ctx, feat, dim, N,
                                            feat->nb[1],
                                            /* offset bytes */ feat->nb[1]);
        patches = ggml_cont(ctx, patches);
        /* patches ne = (backbone.dim, N) */

        /* Linear projection: Wj @ patches + bj
         * Wj ne = (backbone.dim, encoder.model_dim)
         * patches ne = (backbone.dim, N)
         * mul_mat result ne = (encoder.model_dim, N) */
        std::string p = "projector.level" + std::to_string(j) + ".";
        ggml_tensor* W = get_tensor(p + "weight");
        ggml_tensor* b = get_tensor(p + "bias");
        if (!W || !b) return nullptr;

        ggml_tensor* y = ggml_mul_mat(ctx, W, patches);
        y = ggml_add(ctx, y, b);
        /* y ne = (encoder.model_dim, N) */

        /* Add per-level embedding: select column j of level_embed and add to every
         * token in y. level_embed ne = (encoder.model_dim, n_levels). */
        const int model_dim = (int)y->ne[0];
        ggml_tensor* le_j = ggml_view_2d(ctx, level_embed,
                                         model_dim, 1,
                                         level_embed->nb[1],
                                         /* offset */ (size_t)j * level_embed->nb[1]);
        le_j = ggml_cont(ctx, le_j);
        /* le_j ne = (model_dim, 1) — broadcasts over (model_dim, N) via ggml_add. */
        y = ggml_add(ctx, y, le_j);

        publish("projector.level" + std::to_string(j) + ".output", y);
        projected[j] = y;
    }

    /* Concat all 4 levels along axis 1: each (model_dim, N) → (model_dim, 4*N). */
    ggml_tensor* out = projected[0];
    for (int j = 1; j < n_levels; ++j) {
        out = ggml_concat(ctx, out, projected[j], /*dim*/ 1);
    }

    publish("projector.concat.output", out);
    return out;
}

}  // namespace rfdetr
```

### Step 3: Update CMakeLists.txt

Add `src/projector.cpp` to `RFDETR_SOURCES`.

### Step 4: Extend numpy reference

In `scripts/gen_numpy_baseline.py`, add a `projector_forward(cfg, tensors, bb_multi_scale)` function:

```python
def projector_forward(cfg, tensors, multi_scale):
    """Multi-scale projector.

    Args:
        multi_scale: list of 4 numpy arrays, each (N+1, backbone.dim)
                     where token 0 is CLS, tokens 1..N are patches.

    Returns: dict with intermediate tensors:
        - projector.level{0..3}.output: per-level (N, model_dim)
        - projector.concat.output:       (4*N, model_dim)
    """
    out = {}
    n_levels = len(cfg["bb_multi_scale_layers"])
    level_embed = tensors["projector.level_embed"]  # (n_levels, model_dim) in numpy convention

    projected = []
    for j in range(n_levels):
        feat = multi_scale[j]                       # (N+1, backbone.dim)
        patches = feat[1:, :]                       # strip CLS → (N, backbone.dim)

        Wj = tensors[f"projector.level{j}.weight"]  # (model_dim, backbone.dim)
        bj = tensors[f"projector.level{j}.bias"]    # (model_dim,)
        y = patches @ Wj.T + bj                     # (N, model_dim)

        y = y + level_embed[j]                      # add level-j embedding (broadcast over N)

        out[f"projector.level{j}.output"] = y.copy()
        projected.append(y)

    concat = np.concatenate(projected, axis=0)      # (4*N, model_dim)
    out["projector.concat.output"] = concat.copy()
    return out, concat
```

Wire it into `forward()` after the backbone block loop and final norm:

```python
    # ... existing backbone code populates out["backbone.*"] and a final tensor x ...

    # Collect the multi-scale features from `out` (they were stashed during the block loop):
    multi_scale = [out[f"backbone.multiscale.level{j}"]
                   for j in range(len(cfg["bb_multi_scale_layers"]))]

    proj_out, proj_concat = projector_forward(cfg, tensors, multi_scale)
    out.update(proj_out)
    # proj_concat will feed the encoder in Task 4
```

### Step 5: Wire projector into the C++ test

In `tests/test_parity_backbone.cpp`, after the `dinov2_forward` call:

```cpp
rfdetr::BackboneOutput bb = rfdetr::dinov2_forward(gctx, *m, input);
RFDETR_ASSERT(bb.final != nullptr);

ggml_tensor* projected = rfdetr::projector_forward(gctx, *m, bb);
RFDETR_ASSERT(projected != nullptr);
```

Add `#include "projector.hpp"` at top.

In `build_tolerances`, append:

```cpp
for (int j = 0; j < (int)cfg.backbone.multi_scale_layers.size(); ++j) {
    tol["projector.level" + std::to_string(j) + ".output"] = {1e-5f, 1e-4f};
}
tol["projector.concat.output"] = {1e-5f, 1e-4f};
```

### Step 6: docs/parity.md

Add rows for the 5 new checkpoints.

### Step 7: Build + run; expect possible parity debug

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build -R test_parity_backbone --output-on-failure
```

Most likely failure mode: `projector.level0.output` differs because of `level_embed` layout — fixture writes it as `(encoder.model_dim, n_levels)` in ggml ne, which numpy reads as `(n_levels, model_dim)`. The slicing `level_embed[j]` (numpy) vs `ggml_view_2d` offset (C++) must match.

If parity fails at projector.level0.output, the likely culprits in order:
1. CLS-strip offset mismatch (numpy `feat[1:]` vs C++ `ggml_view_2d` with `feat->nb[1]` offset — verify byte arithmetic)
2. Linear projection: numpy `patches @ Wj.T + bj` assumes `Wj` is `(model_dim, backbone.dim)`; ggml_mul_mat with `Wj` ne=`(backbone.dim, model_dim)` produces the same math via transposition
3. Level embedding indexing: numpy `level_embed[j]` selects row j; C++ view selects column-j-of-`(model_dim, n_levels)`. Match the conventions.

Iterate until green.

### Step 8: Commit

```bash
git add src/projector.hpp src/projector.cpp CMakeLists.txt \
        scripts/gen_numpy_baseline.py tests/test_parity_backbone.cpp docs/parity.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(projector): multi-scale projector with level embeddings

For each of the 4 backbone multi-scale features: strip CLS, linear project
to encoder.model_dim, add per-level embedding, then concatenate all 4
projected sequences along the token axis. Output ne = (model_dim, 4*N).

Numpy reference matches; 5 new parity checkpoints
(projector.level{0..3}.output, projector.concat.output) all green at
{1e-5, 1e-4} tolerance.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Encoder — single layer + numpy + parity

The encoder is 3 layers of (LayerNorm → MHA → residual → LayerNorm → MLP → residual), operating on `encoder.model_dim` over the concatenated 4*N tokens from the projector. Same structural family as DINOv2 blocks but with different weight names and `encoder.ffn_dim` for the MLP.

**Files:**
- Create: `src/encoder.hpp`, `src/encoder.cpp`
- Modify: `CMakeLists.txt` — add `src/encoder.cpp`
- Modify: `scripts/gen_numpy_baseline.py` — add `encoder_layer()` function
- Modify: `tests/test_parity_backbone.cpp` — call one encoder layer; add 4 checkpoints

For Task 3 we land ONE layer (block index 0). Task 4 extends to all 3 layers.

### Step 1: Write `src/encoder.hpp`

```cpp
#ifndef RFDETR_ENCODER_HPP
#define RFDETR_ENCODER_HPP

#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace rfdetr {

/* One transformer encoder layer:
 *   x = x + attn(norm1(x))   // self-attention
 *   x = x + mlp(norm2(x))    // feed-forward
 *
 * Input/output ne = (encoder.model_dim, N_tokens, 1, 1) where N_tokens =
 * 4 * N_patches from the projector.
 *
 * Publishes:
 *   encoder.layer{idx}.norm1.output
 *   encoder.layer{idx}.attn.output
 *   encoder.layer{idx}.mlp.output
 *   encoder.layer{idx}.output */
ggml_tensor* encoder_layer(ggml_context* ctx, const Model& m,
                           ggml_tensor* x, int layer_idx);

/* Run all encoder.layers transformer encoder layers in sequence.
 *
 * Publishes everything encoder_layer publishes, plus:
 *   encoder.output  - final encoder output (post last layer, no extra norm) */
ggml_tensor* encoder_forward(ggml_context* ctx, const Model& m,
                             ggml_tensor* x);

}  // namespace rfdetr

#endif
```

### Step 2: Write `src/encoder.cpp` with just `encoder_layer` for now

```cpp
#include "encoder.hpp"
#include "transformer_ops.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <string>

namespace rfdetr {

ggml_tensor* encoder_layer(ggml_context* ctx, const Model& m,
                           ggml_tensor* x, int layer_idx) {
    const std::string p = "encoder.layers." + std::to_string(layer_idx) + ".";
    auto get = [&](const char* suffix) -> ggml_tensor* {
        auto it = m.tensors.find(p + suffix);
        if (it == m.tensors.end()) {
            rfdetr_logf(RFDETR_LOG_ERROR, "encoder_layer: missing tensor '%s'",
                        (p + suffix).c_str());
            return nullptr;
        }
        return it->second;
    };

    ggml_tensor* n1w  = get("norm1.weight");
    ggml_tensor* n1b  = get("norm1.bias");
    ggml_tensor* qkvW = get("self_attn.qkv.weight");
    ggml_tensor* qkvB = get("self_attn.qkv.bias");
    ggml_tensor* prW  = get("self_attn.out.weight");
    ggml_tensor* prB  = get("self_attn.out.bias");
    ggml_tensor* n2w  = get("norm2.weight");
    ggml_tensor* n2b  = get("norm2.bias");
    ggml_tensor* f1W  = get("ffn.fc1.weight");
    ggml_tensor* f1B  = get("ffn.fc1.bias");
    ggml_tensor* f2W  = get("ffn.fc2.weight");
    ggml_tensor* f2B  = get("ffn.fc2.bias");
    if (!n1w || !n1b || !qkvW || !qkvB || !prW || !prB ||
        !n2w || !n2b || !f1W || !f1B || !f2W || !f2B) {
        return nullptr;
    }

    const std::string pub = "encoder.layer" + std::to_string(layer_idx) + ".";

    /* x = x + attn(norm1(x)) */
    ggml_tensor* y = ops::layer_norm(ctx, x, n1w, n1b);
    publish(pub + "norm1.output", y);
    y = ops::mha(ctx, y, qkvW, qkvB, prW, prB, (int)m.config.encoder.heads);
    publish(pub + "attn.output", y);
    x = ggml_add(ctx, x, y);

    /* x = x + mlp(norm2(x)) */
    y = ops::layer_norm(ctx, x, n2w, n2b);
    ggml_tensor* mlp_out = ops::mlp(ctx, y, f1W, f1B, f2W, f2B);
    publish(pub + "mlp.output", mlp_out);
    x = ggml_add(ctx, x, mlp_out);

    publish(pub + "output", x);
    return x;
}

/* encoder_forward lands in Task 4. */

}  // namespace rfdetr
```

### Step 3: Update CMakeLists.txt

Add `src/encoder.cpp` to `RFDETR_SOURCES`.

### Step 4: Extend numpy reference

In `scripts/gen_numpy_baseline.py`, add `encoder_layer(cfg, tensors, x, i)`:

```python
def encoder_layer(cfg, tensors, x, i):
    """One encoder layer. x: (N_tokens, model_dim). Returns: same shape."""
    p = f"encoder.layers.{i}."
    pub = f"encoder.layer{i}."
    out = {}

    n1 = layer_norm(x, tensors[p + "norm1.weight"], tensors[p + "norm1.bias"])
    out[pub + "norm1.output"] = n1.copy()
    y = mha(n1,
            tensors[p + "self_attn.qkv.weight"], tensors[p + "self_attn.qkv.bias"],
            tensors[p + "self_attn.out.weight"], tensors[p + "self_attn.out.bias"],
            cfg["enc_heads"])
    out[pub + "attn.output"] = y.copy()
    x = x + y

    n2 = layer_norm(x, tensors[p + "norm2.weight"], tensors[p + "norm2.bias"])
    z = mlp(n2,
            tensors[p + "ffn.fc1.weight"], tensors[p + "ffn.fc1.bias"],
            tensors[p + "ffn.fc2.weight"], tensors[p + "ffn.fc2.bias"])
    out[pub + "mlp.output"] = z.copy()
    x = x + z

    out[pub + "output"] = x.copy()
    return out, x
```

Note: `cfg["enc_heads"]` needs to be in the cfg dict. Add to `read_model()`:

```python
"enc_heads": _u32(reader, "rfdetr.encoder.heads"),
```

(Verify the GGUF metadata key spelling matches the fixture writer.)

Wire one encoder layer call into `forward()` after the projector:

```python
    enc_out, x_enc = encoder_layer(cfg, tensors, proj_concat, 0)
    out.update(enc_out)
```

### Step 5: Wire one encoder layer into the C++ test

In `tests/test_parity_backbone.cpp`, after the projector call:

```cpp
ggml_tensor* enc = rfdetr::encoder_layer(gctx, *m, projected, /*layer_idx*/ 0);
RFDETR_ASSERT(enc != nullptr);
```

Add `#include "encoder.hpp"`.

Add tolerances for the 4 new checkpoints:

```cpp
for (int i = 0; i < 1; ++i) {  // just layer 0 for Task 3
    std::string p = "encoder.layer" + std::to_string(i) + ".";
    tol[p + "norm1.output"] = {1e-5f, 1e-4f};
    tol[p + "attn.output"]  = {1e-5f, 1e-4f};
    tol[p + "mlp.output"]   = {1e-5f, 1e-4f};
    tol[p + "output"]       = {1e-5f, 1e-4f};
}
```

### Step 6: Build + run

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build -R test_parity_backbone --output-on-failure
```

Expected: PASS — same op shapes as DINOv2 blocks (which already work), just different weight names.

### Step 7: Commit

```bash
git add src/encoder.hpp src/encoder.cpp CMakeLists.txt \
        scripts/gen_numpy_baseline.py tests/test_parity_backbone.cpp docs/parity.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(encoder): single transformer encoder layer

Same shape family as DINOv2 block — LN → MHA → residual → LN → MLP →
residual — operating on encoder.model_dim over the concatenated multi-scale
tokens from the projector. Uses ops::layer_norm / ops::mha / ops::mlp
from the shared transformer_ops module.

Numpy reference matches; 4 new parity checkpoints (encoder.layer0.*)
all green at {1e-5, 1e-4} tolerance.

Task 4 loops over all 3 encoder layers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Encoder — loop all layers + `encoder_forward` wrapper

Extend the single layer to all 3 layers in both C++ and numpy, and wrap in `encoder_forward`. Final encoder output published as `encoder.output`.

**Files:**
- Modify: `src/encoder.cpp` — implement `encoder_forward`
- Modify: `scripts/gen_numpy_baseline.py` — loop encoder layers
- Modify: `tests/test_parity_backbone.cpp` — call `encoder_forward`; extend tolerance loop

### Step 1: Implement `encoder_forward` in `src/encoder.cpp`

Append:

```cpp
ggml_tensor* encoder_forward(ggml_context* ctx, const Model& m,
                             ggml_tensor* x) {
    for (uint32_t i = 0; i < m.config.encoder.layers; ++i) {
        x = encoder_layer(ctx, m, x, (int)i);
        if (!x) return nullptr;
    }
    publish("encoder.output", x);
    return x;
}
```

### Step 2: numpy loop

In `gen_numpy_baseline.py`, replace the single `encoder_layer(..., 0)` call with a loop:

```python
    x_enc = proj_concat
    for i in range(cfg["enc_layers"]):
        enc_out, x_enc = encoder_layer(cfg, tensors, x_enc, i)
        out.update(enc_out)
    out["encoder.output"] = x_enc.copy()
```

`cfg["enc_layers"]` needs to be in cfg — add to `read_model()`:

```python
"enc_layers": _u32(reader, "rfdetr.encoder.layers"),
```

### Step 3: C++ test calls `encoder_forward`

Replace the single-layer call in `tests/test_parity_backbone.cpp`:

```cpp
ggml_tensor* enc = rfdetr::encoder_forward(gctx, *m, projected);
RFDETR_ASSERT(enc != nullptr);
```

Extend the tolerance loop to cover all `cfg.encoder.layers`:

```cpp
for (uint32_t i = 0; i < cfg.encoder.layers; ++i) {
    std::string p = "encoder.layer" + std::to_string(i) + ".";
    tol[p + "norm1.output"] = {1e-5f, 1e-4f};
    tol[p + "attn.output"]  = {1e-5f, 1e-4f};
    tol[p + "mlp.output"]   = {1e-5f, 1e-4f};
    tol[p + "output"]       = {1e-5f, 1e-4f};
}
tol["encoder.output"] = {1e-5f, 1e-4f};
```

### Step 4: Build + run

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build -R test_parity_backbone --output-on-failure
```

Expected: PASS. The 4 additional encoder checkpoints (layer1, layer2, plus encoder.output) all match.

### Step 5: Commit

```bash
git add src/encoder.cpp scripts/gen_numpy_baseline.py tests/test_parity_backbone.cpp docs/parity.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(encoder): loop all 3 encoder layers + encoder_forward wrapper

Extends the single-layer impl from Task 3 to all cfg.encoder.layers (= 3 for
base). encoder_forward publishes encoder.output after the last layer.

13 total encoder checkpoints (4 per layer × 3 + encoder.output) all green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Final smoke + README

**Files:**
- Modify: `docs/parity.md`
- Modify: `README.md`

### Step 1: Clean rebuild across configurations

Same triple-check as prior plans:

```bash
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=OFF -DRFDETR_BUILD_CLI=OFF 2>&1 | tail -3 && cmake --build build -j 2>&1 | tail -3
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=OFF 2>&1 | tail -3 && cmake --build build -j 2>&1 | tail -3 && ctest --test-dir build --output-on-failure 2>&1 | tail -3
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=ON 2>&1 | tail -3 && cmake --build build -j 2>&1 | tail -3 && ctest --test-dir build --output-on-failure 2>&1 | tail -3
```

Expected: all three green; 10/10 tests with CLI=ON.

### Step 2: Update README.md

```markdown
## Status

**Projector + encoder (Plan 6a) complete.** The forward pipeline now runs:
DINOv2 backbone → multi-scale projector (4 levels, level embeddings, concat)
→ 3-layer transformer encoder. 73 parity checkpoints all green at 1e-5
absolute tolerance against the numpy reference. Ten tests pass on a clean
build.

`dinov2_forward` now returns a `BackboneOutput { final, multi_scale[4] }`
so downstream consumers can access multi-scale features directly without
the trace-callback workaround. layer_norm/mha/mlp moved out of
`dinov2.cpp`'s anonymous namespace into shared `transformer_ops` so the
encoder, decoder (Plan 6b), and heads (Plan 6c) can call them.

Plan 6b adds the decoder (queries + self-attn + cross-attn + 3 layers).
Plan 6c wires the heads and end-to-end detect.

The Python conversion script body is still deferred (see Plan 2 Task 3).
The C++ side uses a synthesized F32 GGUF fixture for tests.
```

### Step 3: Commit

```bash
git add README.md docs/parity.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
docs: mark projector+encoder plan (Plan 6a) complete

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- §3 Architecture (projector + encoder) — covered. Decoder/heads in 6b/6c.
- §6 API — `rfdetr_detect` still NOT_IMPLEMENTED; Plan 6c wires it.
- §8 Tests — `test_parity_backbone` (will likely be renamed in Plan 6c) covers backbone + projector + encoder.
- §9 Parity workflow — 73 checkpoints; documented in `docs/parity.md`.
- §10 Build — `transformer_ops`, `projector`, `encoder` added to RFDETR_SOURCES.

**Risk areas:**
1. **`level_embed` layout** — fixture writes as `(model_dim, n_levels)` in ggml ne; numpy reads as `(n_levels, model_dim)`. The C++ `ggml_view_2d` column-slice and numpy's `level_embed[j]` row-slice must produce the same vector. Most likely first-failure-mode in Task 2.
2. **CLS stripping** — backbone features include CLS at position 0; projector strips it. Get the `feat->nb[1]` offset right in C++ to match numpy's `feat[1:]`.
3. **`backbone.dim` vs `encoder.model_dim`** — fixture has them both as 64. Production has different values. The projector code uses `feat->ne[0]` for input dim and `W->ne[1]` (or similar) for output dim, so it should generalize. Verify by inspection.
4. **Encoder layer shape compatibility** — uses `ops::mha` which expects `(dim, N)` input. Projector output is `(model_dim, 4*N)` so the shape lines up. Verify in Task 3.

---

## Next plans

After this plan lands:

- **Plan 6b** — Decoder: learnable queries, 3 layers of (self-attn(queries) + cross-attn(queries → encoder) + FFN), per-layer norms. Numpy + parity.
- **Plan 6c** — Class + bbox heads, `rfdetr_model_forward`, wire `rfdetr_detect` end-to-end (CLI `detect` returns real JSON detections, even if zeros from random weights).
- **Plan 7** — Real PyTorch baseline replacing numpy.
- **Plan 8** — Quantization.
- **Plan 9** — Variants nano/small/medium/large.
