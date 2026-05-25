# rt-detr.cpp — RF-DETR in ggml — Design

**Date:** 2026-05-25
**Author:** mudler@localai.io
**Status:** Approved (initial spec)

## 1. Goal

A ggml-based C++ inference implementation of Roboflow's [RF-DETR](https://github.com/roboflow/rf-detr) real-time object detection model. Project layout, build system, and distribution model mirror [vibevoice.cpp](https://github.com/mudler/vibevoice.cpp). llama.cpp's conventions (GGUF, backend abstraction, quantize tooling) are the broader reference.

Naming note: directory is `rt-detr.cpp`; library, headers, binaries, and code use the **`rfdetr`** prefix to be honest about the actual model.

## 2. Scope (v1)

In scope:

- All five upstream variants — **nano, small, base, medium, large** — driven from a single graph builder parameterized by a `rfdetr_config` loaded from GGUF metadata.
- Inference only. No training, fine-tuning, or backbone pretraining.
- Image input (JPEG/PNG via `stb_image`), JSON detections out, optional annotated PNG out.
- F16 default; Q8_0 via separate `rfdetr-quantize` binary (matrix-ops only; norms/embeddings stay F16).
- Backends: CPU, CUDA, Metal, Vulkan, HIPBLAS — forwarded to ggml via CMake cache vars (same pattern as vibevoice).
- Dual C API: opaque-pointer (`rfdetr.h`) + flat dlopen-friendly ABI (`rfdetr_capi.h`) for LocalAI/purego embedding.
- Numerical-parity test harness against the upstream PyTorch implementation.

Out of scope (v1):

- Video / streaming input (no ffmpeg dependency).
- ONNX or torchscript paths.
- Training, ONNX export, or fine-tuning utilities.
- Server binary (vibevoice has one optional; we defer).
- Tracking, classification heads other than the upstream detection head.
- Quantization beyond F16 + Q8_0. Q4_K and mixed profiles are a follow-up once Q8_0 parity is solid.

## 3. Architecture

RF-DETR is **LW-DETR + DINOv2 backbone + multi-scale features**. Crucially, it uses vanilla multi-head attention (not deformable attention); multi-scale is achieved by projecting backbone features at multiple resolutions. All required ops are already in ggml — no custom kernels.

Forward graph:

```
image
  → preprocess: resize to config.image_size, ImageNet mean/std normalize, HWC→CHW
  → DINOv2 backbone: patch_embed (conv2d) → N transformer blocks (mix of windowed + global attention)
  → multi-scale feature extraction: tap selected backbone layers
  → projector: per-scale 1x1 conv / linear projection to model_dim; add level + positional embeddings
  → encoder: M layers of (self-attention + FFN), pre-LN
  → decoder: K layers of (self-attention over queries + cross-attention to encoder tokens + FFN)
  → class head: per-query MLP → num_classes logits
  → bbox head: per-query MLP → (cx, cy, w, h) normalized
  → postprocess: sigmoid(class) → top-k → threshold → cxcywh→xyxy → denormalize to original image
```

Variant differences live entirely in `rfdetr_config`:

- backbone width, depth, num heads, window size, multi-scale layer indices
- projector levels and channels per level
- encoder layer count, decoder layer count, model_dim, ffn_dim
- num_queries, num_classes, image_size
- normalization stats, class names

One graph builder consumes the config; nothing else branches on variant.

## 4. Project layout

```
rt-detr.cpp/
├── CMakeLists.txt
├── README.md
├── LICENSE                       # Apache-2.0 (matches roboflow/rf-detr)
├── .gitignore  .gitmodules
├── third_party/
│   ├── ggml/                     # submodule
│   └── stb/                      # vendored stb_image.h, stb_image_write.h, stb_truetype.h
├── include/
│   ├── rfdetr.h                  # opaque-pointer C API
│   └── rfdetr_capi.h             # flat dlopen ABI
├── src/
│   ├── common.{cpp,hpp}          # logging, error codes, span helpers
│   ├── backend.{cpp,hpp}         # ggml_backend selection
│   ├── model_loader.{cpp,hpp}    # GGUF parse, variant detection, weight binding
│   ├── image_io.{cpp,hpp}        # stb_image load/save, preprocess
│   ├── dinov2.{cpp,hpp}          # DINOv2 ViT backbone (windowed + global attention)
│   ├── projector.{cpp,hpp}       # multi-scale feature projector
│   ├── encoder.{cpp,hpp}         # transformer encoder
│   ├── decoder.{cpp,hpp}         # transformer decoder
│   ├── heads.{cpp,hpp}           # class + bbox MLP heads
│   ├── rfdetr_model.{cpp,hpp}    # ties the graph together; forward pass
│   ├── trace.{cpp,hpp}           # named-checkpoint callback plumbing (parity harness hook)
│   ├── postprocess.{cpp,hpp}     # sigmoid → top-k → threshold → cxcywh→xyxy
│   ├── visualize.{cpp,hpp}       # draw bboxes + labels onto RGBA buffer
│   ├── rfdetr.cpp                # public C API impl
│   └── rfdetr_capi.cpp           # flat C ABI impl
├── examples/
│   ├── cli/                      # rfdetr-cli (detect, info, bench, compare)
│   └── quantize/                 # rfdetr-quantize
├── tests/
│   ├── CMakeLists.txt
│   ├── fixtures/                 # small images, mini-model, golden tensors
│   ├── parity/                   # baseline bundles, tolerance config
│   ├── test_image_io.cpp
│   ├── test_postprocess.cpp
│   ├── test_model_loader.cpp
│   ├── test_inference.cpp
│   ├── test_parity.cpp
│   ├── test_capi.cpp
│   └── test_capi_flat.cpp
├── scripts/
│   ├── convert_rfdetr_to_gguf.py # HF/PyTorch checkpoint → GGUF (all variants)
│   ├── run_rfdetr_baseline.py    # run upstream RF-DETR, dump detections + intermediate activations
│   ├── quantize_gguf.py          # optional Python helper (parallels rfdetr-quantize)
│   └── requirements.txt
└── docs/
    ├── architecture.md
    ├── conversion.md             # tensor-name map, variant detection rules
    ├── variants.md               # per-variant parameter tables
    ├── parity.md                 # baseline workflow, tolerance interpretation
    └── superpowers/specs/2026-05-25-rfdetr-cpp-design.md   # this doc
```

## 5. Weight format and conversion

- One GGUF per (variant × quantization): `rfdetr-base-f16.gguf`, `rfdetr-nano-q8_0.gguf`, etc.
- GGUF metadata keys (all under the `rfdetr.` namespace):
  - `rfdetr.variant` ∈ {nano, small, base, medium, large}
  - `rfdetr.image_size`, `rfdetr.num_queries`, `rfdetr.num_classes`
  - `rfdetr.class_names` (string array)
  - `rfdetr.backbone.{dim,depth,heads,window_size,multi_scale_layers}`
  - `rfdetr.encoder.{layers,model_dim,ffn_dim,heads}`
  - `rfdetr.decoder.{layers,model_dim,ffn_dim,heads}`
  - `rfdetr.preprocess.{mean,std}`
- `scripts/convert_rfdetr_to_gguf.py` consumes a HuggingFace `roboflow/rf-detr-*` checkpoint and emits GGUF. Tensor name mapping documented in `docs/conversion.md`.
- `rfdetr-quantize input.gguf output.gguf Q8_0` quantizes matmul weights to Q8_0, leaves norms/embeddings F16. Same convention as llama.cpp's quantize tool.

## 6. Public API

`include/rfdetr.h` — opaque-pointer C API:

```c
typedef struct rfdetr_context rfdetr_context;
typedef struct rfdetr_image   rfdetr_image;

typedef struct {
    uint32_t class_id;
    const char* class_name;   /* borrowed from context lifetime */
    float score;
    float x1, y1, x2, y2;     /* pixel coords on the original (pre-resize) image */
} rfdetr_detection;

typedef struct {
    const char* model_path;
    int n_threads;
    rfdetr_log_cb log_cb;
    void* log_user_data;
    /* backend selection flags forwarded to ggml */
} rfdetr_params;

typedef struct {
    float threshold;        /* default 0.5 */
    uint32_t top_k;         /* default 300 */
    uint32_t* class_filter; /* optional allowlist; NULL = all classes */
    size_t    class_filter_len;
} rfdetr_detect_params;

rfdetr_context* rfdetr_init(const rfdetr_params*);
void            rfdetr_free(rfdetr_context*);

rfdetr_image* rfdetr_image_load_file(const char* path);
rfdetr_image* rfdetr_image_load_buffer(const uint8_t* bytes, size_t len);
void          rfdetr_image_free(rfdetr_image*);

int rfdetr_detect(rfdetr_context*, const rfdetr_image*,
                  const rfdetr_detect_params*,
                  rfdetr_detection** out, size_t* n_out);
void rfdetr_detections_free(rfdetr_detection*, size_t);

int rfdetr_render(const rfdetr_image*, const rfdetr_detection*, size_t,
                  const char* out_path);

void rfdetr_set_log_callback(rfdetr_log_cb cb, void* user_data);
```

Return codes: 0 = success, negative = error. Error enum in `rfdetr.h`.

`include/rfdetr_capi.h` — flat dlopen-friendly ABI for purego:

- `rfdetr_capi_load(model_path) → handle`
- `rfdetr_capi_detect_path(handle, image_path, threshold, top_k) → JSON-encoded detections (caller-freed)`
- `rfdetr_capi_detect_buffer(handle, bytes, len, threshold, top_k) → JSON string`
- `rfdetr_capi_unload(handle)`
- No opaque pointers in this header; handles are `uintptr_t`.

## 7. CLI

```
rfdetr-cli detect --model rfdetr-base-f16.gguf --input image.jpg \
                  --output detections.json [--annotated out.png] \
                  [--threshold 0.5] [--topk 300] [--classes 0,1,16]

rfdetr-cli info   --model rfdetr-base-f16.gguf
    # prints: variant, image_size, num_classes, num_queries, param count, backend

rfdetr-cli bench  --model … --input … [--iters 50] [--warmup 5]
    # prints: ms/iter, p50/p95/p99, throughput

rfdetr-cli compare --baseline tests/parity/bundles/base/ --image …
    # runs C++ inference and diffs each captured checkpoint vs the baseline bundle.
    # prints a per-checkpoint table: max-abs-err, mean-abs-err, divergence layer.

rfdetr-quantize input.gguf output.gguf Q8_0
```

## 8. Tests

ctest-driven. Mirrors vibevoice's env-var pattern for model paths in E2E tests.

1. **Unit tests** (no model required):
   - `test_image_io`: PNG/JPEG round-trip, preprocess math.
   - `test_postprocess`: cxcywh→xyxy, sigmoid+top-k, threshold filtering, class allowlist.
   - `test_model_loader`: GGUF metadata parsing, variant auto-detect, missing-tensor errors.

2. **C-API smoke** (`test_capi`, `test_capi_flat`): load → detect → free; flat ABI mirror.

3. **E2E inference** (`test_inference`, env-gated by `RFDETR_TEST_MODEL_PATH`): detect on a COCO sample image, assert known objects present above threshold.

4. **Numerical parity** (`test_parity`, env-gated by `RFDETR_PARITY_BUNDLE`): load a precomputed baseline bundle, run the C++ model with the trace callback active, diff each named checkpoint against the baseline under per-layer tolerances declared in `tests/parity/tolerances.yaml`.

## 9. Baseline parity workflow

The official PyTorch `roboflow/rf-detr` is the ground truth. We capture its behavior on fixed inputs and assert ours matches.

Pieces:

- **`scripts/run_rfdetr_baseline.py`** — runs upstream RF-DETR on `(variant, image)` and writes a *bundle directory*:
  - `detections.json` — final boxes/scores/classes (end-to-end ground truth).
  - `activations.gguf` — named intermediate tensors at well-defined checkpoints. We use GGUF (not safetensors) so the C++ side reads them with the existing loader; no new dep. Checkpoint names mirror the names we publish on the C++ side via `trace_callback`.
  - `meta.json` — variant, image size, preprocess stats, seed, torch version, rfdetr pip version.

  Upstream is consumed via `pip install rfdetr` pinned in `scripts/requirements.txt`. Bundle generation runs on a developer host with PyTorch; CI does not need PyTorch.

- **Named checkpoints** captured at:
  - `preprocess.input` — model-space input tensor
  - `backbone.layer{i}` — selected ViT block outputs
  - `backbone.multiscale.level{j}` — per-scale features
  - `projector.level{j}` — post-projection features
  - `encoder.layer{i}` — encoder outputs
  - `decoder.layer{i}` — decoder outputs
  - `heads.class_logits`, `heads.bbox_pred`
  - `postprocess.scores`, `postprocess.boxes`

- **`tests/parity/`** — checked-in bundles (one per variant once parity for that variant is green). `base` ships first; the other four follow incrementally. `tolerances.yaml` defines atol/rtol per checkpoint.

- **`tests/test_parity.cpp`** — loads the bundle, hooks `trace_callback`, asserts `allclose` per checkpoint; asserts final detection IoU ≥ threshold and score-delta ≤ threshold.

- **`make baseline-regen VARIANT=base IMAGE=tests/fixtures/cats.jpg`** — Make target wrapping `scripts/run_rfdetr_baseline.py`. Regenerates the bundle. Output is what CI consumes.

- **`rfdetr-cli compare`** — developer-facing diff command (see CLI section). The day-to-day tool during bring-up.

- **`docs/parity.md`** — workflow doc: how to add a checkpoint, how to interpret a diff, when to widen a tolerance versus when to debug.

### Trace callback design

```c
typedef void (*rfdetr_trace_cb)(const char* name,
                                const float* data, size_t n_elements,
                                const int64_t* shape, size_t n_dims,
                                void* user_data);

void rfdetr_set_trace_callback(rfdetr_context*, rfdetr_trace_cb, void* user_data);
```

Production inference does not register a callback → publish sites compile to no-ops via inline check. The parity harness registers a callback that captures named tensors.

### Parity rollout order

1. `base` variant first — smallest path to a working harness end-to-end.
2. Then `small`, `nano`, `medium`, `large` — same code, new fixtures only.

## 10. Build

```
git clone --recursive https://…/rt-detr.cpp
cmake -B build -DRFDETR_BUILD_TESTS=ON \
              [-DRFDETR_GGML_CUDA=ON | -DRFDETR_GGML_METAL=ON \
               | -DRFDETR_GGML_VULKAN=ON | -DRFDETR_GGML_HIPBLAS=ON]
cmake --build build -j
```

Outputs in `build/bin/`. Backend flags forwarded to ggml's cache variables, exactly as vibevoice does. Static library `librfdetr` by default; `-DRFDETR_SHARED=ON` produces a shared library for purego embedding.

## 11. Dependencies

- `third_party/ggml` (git submodule) — inference engine.
- `third_party/stb` (vendored, header-only) — `stb_image.h`, `stb_image_write.h`, `stb_truetype.h` (label rendering on annotated output).
- A small open-license font for label rendering (DejaVu Sans or similar), embedded as a generated C header so the binary has no external font dependency. Lives under `third_party/fonts/` with its license alongside.
- Python (scripts only, not at build/run time): `torch`, `rfdetr`, `gguf`, `numpy`, `pillow`, `pyyaml`. Pinned in `scripts/requirements.txt`.

No external C/C++ deps beyond ggml + stb.

## 12. Known risks and mitigations

1. **DINOv2 windowed attention** — RF-DETR's backbone mixes windowed and global attention blocks. Implementation is reshape + standard attention + reverse-reshape; indexing is fiddly but a closed problem. Mitigation: parity tests at `backbone.layer{i}` catch this layer-by-layer.

2. **Multi-scale plumbing** — getting strides, level embeddings, and positional encodings exactly right across scales is the most likely source of subtle bugs. Mitigation: per-scale parity checkpoints (`backbone.multiscale.level{j}`, `projector.level{j}`).

3. **PyTorch numerical parity** — LayerNorm epsilon, attention scale, GELU exact vs approximate, upsample mode, RoPE conventions if used. Mitigation: every layer has a parity checkpoint; tolerances start tight and widen only with justification.

4. **All-variants-at-once** — user picked all 5 for v1. Mitigation: variant-agnostic graph + per-variant config; conversion script handles all 5; **bring-up order is base → small → nano → medium → large**, with parity-green required before moving to the next.

5. **Conversion script drift** — upstream may rename tensors between releases. Mitigation: `convert_rfdetr_to_gguf.py` pins the upstream `rfdetr` pip version; `tests/test_model_loader` asserts the expected tensor set is present after conversion.

## 13. Out of scope explicitly

For clarity on what we are *not* doing in v1:

- No training, distillation, or fine-tuning code.
- No video decode / streaming / webcam capture.
- No detection-tracker (ByteTrack, SORT, etc.).
- No HTTP server. The flat C ABI is the embedding surface; LocalAI provides the HTTP layer.
- No Python bindings beyond the conversion/baseline scripts.
- No additional quantization formats beyond F16 + Q8_0 in v1.

## 14. Success criteria

v1 is done when:

1. All five variants load and run inference on CPU and at least one GPU backend.
2. `rfdetr-cli detect` produces JSON detections and an annotated PNG indistinguishable by eye from upstream output on a fixture image.
3. `test_parity` passes for the `base` variant under tight tolerances.
4. `rfdetr-quantize` produces a working Q8_0 GGUF. On a checked-in 20-image fixture set, Q8_0 detections match F16 within: mean score delta ≤ 0.02 per matched detection and ≥ 95% box IoU on the top-K detections. (No external COCO ground-truth needed; we compare quantized output to F16 output, not to upstream labels.)
5. The flat C ABI is callable from a smoke-test purego program (mirrors vibevoice's coverage).
