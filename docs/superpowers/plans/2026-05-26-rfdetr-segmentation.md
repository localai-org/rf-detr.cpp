# Segmentation Variant Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add support for RF-DETR-Seg variants (SegNano, SegSmall, SegMedium, SegLarge, SegXLarge, Seg2XLarge) to rfdetr.cpp — including the segmentation head, mask postprocessing, CLI mask output, and the architectural differences (patch_size=12, num_windows=1, num_queries=100) that distinguish seg models from detection.

**Architecture:** Seg models add a `SegmentationHead` after the existing detection backbone+transformer pipeline. The head takes (a) the backbone's spatial features and (b) the decoder's per-layer query features, and produces a per-query mask via a query-conditioned einsum. We implement this as a new module `src/segmentation.{hpp,cpp}` that hooks into `rfdetr_model_forward` after the decoder, optionally enabled by a model-config flag in the GGUF metadata. New ops needed: depthwise Conv2d, bilinear 2D upsample (CPU side is fine for now). CLI gains a `--masks <out_dir>` flag that dumps per-detection PNG masks (or COCO RLE-encoded JSON).

**Tech Stack:** Python (converter extension), C++ (new modules, CLI), ggml (forward), stb_image_write (PNG mask output — already vendored).

---

## Architectural deltas vs detection variants

From `RFDETRSegNano()` config inspection (verified):

| Field | Detection (Base) | Segmentation (Nano) |
|---|---|---|
| `patch_size` | 14 | **12** |
| `num_windows` | 4 (so 4×4=16 windows) | **1** (no windowing — global attention everywhere) |
| `resolution` | 560 | 312 (scales with patch_size) |
| `num_queries` | 300 | **100** |
| `num_select` | 300 | 100 |
| `num_classes` | 91 | **90** |
| `dec_layers` | 3 | 4 |
| `out_feature_indexes` | [2, 5, 8, 11] | [3, 6, 9, 12] (off-by-one; 4 taps including layer 12 = final) |
| `positional_encoding_size` | 37 | 26 |
| `segmentation_head` | false | **true** — adds the mask head |
| `mask_downsample_ratio` | n/a | 4 |

Seg-specific module (`segmentation_head`, 36 tensors):
- `blocks.{0..3}` — 4 × DepthwiseConvBlock (each: depthwise Conv2d 3×3 + LayerNorm + pointwise Linear 256→256 + GELU)
- `spatial_features_proj` — Conv2d 1×1 (256→256, bottleneck projection)
- `query_features_block` — MLPBlock (LN + Linear 256→1024 + GELU + Linear 1024→256)
- `query_features_proj` — Linear 256→256
- `bias` — scalar (broadcast added to all mask logits)

Forward (from rfdetr/models/heads/segmentation.py):

```python
target_h, target_w = image_h // 4, image_w // 4
spatial = F.interpolate(spatial_features, (target_h, target_w), mode="bilinear", align_corners=False)
masks = []
for block, qf in zip(self.blocks, query_features_per_decoder_layer):
    spatial = block(spatial)                                # depthwise + norm + pointwise + GELU
    spatial_proj = self.spatial_features_proj(spatial)      # Conv2d 1x1
    qf = self.query_features_proj(self.query_features_block(qf))  # MLP + Linear
    masks.append(torch.einsum("bchw,bnc->bnhw", spatial_proj, qf) + self.bias)
# Inference uses masks[-1] (last decoder layer).
```

The `einsum("bchw,bnc->bnhw", spatial, qf)` is a per-query channel-dot — implementable as a batched matmul after reshape: `(N_queries, C) × (C, H*W) → (N_queries, H*W)`.

## File Structure

- Modify: `scripts/convert_rfdetr_to_gguf.py` — accept seg variants; emit `segmentation_head.*` tensors + metadata flag
- Modify: `src/model_loader.{cpp,hpp}` — load seg config (mask_downsample_ratio, has_segmentation_head); load seg head tensors
- Create: `src/segmentation.{hpp,cpp}` — segmentation head forward
- Modify: `src/rfdetr_model.{cpp,hpp}` — wire seg head after decoder; return masks alongside detections
- Modify: `include/rfdetr.h` — add mask data to public detection struct (optional, nullable)
- Modify: `src/rfdetr.cpp` — postprocess masks (threshold, upsample to image size)
- Modify: `examples/cli/main.cpp::cmd_detect` — `--masks <out_dir>` flag; write per-detection PNG masks via stb_image_write
- Create: `tests/test_parity_segmentation.cpp` — parity test against torch baseline
- Modify: `scripts/gen_torch_baseline.py` — hook seg head + register checkpoint tensors
- Modify: `BENCHMARK.md`, `README.md` — seg variants section

---

## Task 1: Investigate seg model in Python; extend baseline generator

**Files:**
- Modify: `scripts/gen_torch_baseline.py`

- [ ] **Step 1: Pick a seg variant for baseline** 

Use SegNano (smallest, fastest to debug). The baseline script currently uses RFDETRBase; we need a parallel codepath or a `--variant seg-nano` switch.

- [ ] **Step 2: Add segmentation checkpoint hooks**

Extend `scripts/gen_torch_baseline.py` to support seg models. New checkpoints to capture:
- `seg.spatial_features.input` — projector output before bilinear resize to (H/4, W/4)
- `seg.spatial_features.resized` — after bilinear resize
- `seg.block.{0..3}.spatial_out` — output of each DepthwiseConvBlock
- `seg.block.{0..3}.spatial_proj` — after spatial_features_proj 1×1
- `seg.block.{0..3}.qf_proj` — after query_features_block + query_features_proj
- `seg.masks.{0..3}` — per-decoder-layer mask tensor (B, N_queries, H/4, W/4)
- `seg.masks.final` — masks[-1] (the inference output)

Add a `--seg` flag to the baseline script that loads `RFDETRSegNano()` instead of `RFDETRBase()` and registers hooks on the seg head modules.

- [ ] **Step 3: Run + write the baseline bundle**

```bash
.venv/bin/python scripts/gen_torch_baseline.py --seg \
    --output tests/fixtures/baseline_torch_seg.gguf
ls -lh tests/fixtures/baseline_torch_seg.gguf
```

Expected: a separate baseline bundle file with seg-specific tensors.

- [ ] **Step 4: Commit**

```bash
git add scripts/gen_torch_baseline.py
git commit -m "feat(parity): seg-variant baseline bundle generator"
```

---

## Task 2: Extend converter for seg models

**Files:**
- Modify: `scripts/convert_rfdetr_to_gguf.py`

- [ ] **Step 1: Add `--variant` choices for seg variants**

Extend the `--variant` flag (added in the detection-variants plan, or add it here if that plan hasn't run yet):

```python
parser.add_argument(
    "--variant",
    choices=[
        "nano", "small", "base", "medium", "large",
        "seg-nano", "seg-small", "seg-medium", "seg-large",
        "seg-xlarge", "seg-2xlarge",
    ],
    default="base",
)
```

- [ ] **Step 2: Dispatch to the right seg class**

```python
from rfdetr import (
    RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge,
    RFDETRSegXLarge, RFDETRSeg2XLarge,
)
_VARIANT_CLASSES.update({
    "seg-nano": RFDETRSegNano, "seg-small": RFDETRSegSmall,
    "seg-medium": RFDETRSegMedium, "seg-large": RFDETRSegLarge,
    "seg-xlarge": RFDETRSegXLarge, "seg-2xlarge": RFDETRSeg2XLarge,
})
```

- [ ] **Step 3: Emit seg-specific metadata**

```python
is_seg = args.variant.startswith("seg-")
writer.add_bool("rfdetr.has_segmentation_head", is_seg)
if is_seg:
    writer.add_uint32("rfdetr.mask_downsample_ratio", m.model_config.mask_downsample_ratio)
    # patch_size, num_windows, num_queries already written by existing converter logic
    # — verify they're not hardcoded
```

- [ ] **Step 4: Emit segmentation_head tensors**

Add a section that iterates `inner.segmentation_head.named_parameters()` and writes each tensor under `segmentation_head.<name>`:

```python
if is_seg:
    for name, param in inner.segmentation_head.named_parameters():
        writer.add_tensor(f"segmentation_head.{name}", param.detach().cpu().numpy())
```

Use the same `should_quantize_q8_0` heuristic for these new tensors when quantizing.

- [ ] **Step 5: Convert SegNano**

```bash
.venv/bin/python scripts/convert_rfdetr_to_gguf.py --variant seg-nano \
    --output models/rfdetr-seg-nano-f32.gguf
ls -lh models/rfdetr-seg-nano-f32.gguf
```

Expected: GGUF file with seg metadata + 36 extra seg head tensors.

- [ ] **Step 6: Verify via gguf-py inspector**

```bash
.venv/bin/python -c "
import gguf
r = gguf.GGUFReader('models/rfdetr-seg-nano-f32.gguf')
print('seg head tensors:')
for t in r.tensors:
    if 'segmentation' in t.name:
        print(f'  {t.name}: {t.shape}')
" | head -20
```

Expected: 36 segmentation_head.* tensors with correct shapes.

- [ ] **Step 7: Commit**

```bash
git add scripts/convert_rfdetr_to_gguf.py
git commit -m "feat(convert): RF-DETR-Seg variant conversion + segmentation_head tensors"
```

---

## Task 3: Extend loader for seg config and tensors

**Files:**
- Modify: `src/model_loader.{cpp,hpp}`

- [ ] **Step 1: Add seg fields to Config**

In `src/model_loader.hpp`:

```cpp
struct Config {
    // ... existing fields ...
    bool has_segmentation_head = false;
    uint32_t mask_downsample_ratio = 4;
};
```

In `src/model_loader.cpp::load_config`, after the existing reads:

```cpp
int64_t kid;
if ((kid = gguf_find_key(g, "rfdetr.has_segmentation_head")) >= 0) {
    cfg.has_segmentation_head = gguf_get_val_bool(g, kid);
}
if (cfg.has_segmentation_head) {
    if ((kid = gguf_find_key(g, "rfdetr.mask_downsample_ratio")) >= 0) {
        cfg.mask_downsample_ratio = gguf_get_val_u32(g, kid);
    }
}
```

- [ ] **Step 2: Add seg tensors to expected_tensor_names()**

Find the function that enumerates all expected tensor names. If `cfg.has_segmentation_head`, append:

```cpp
if (cfg.has_segmentation_head) {
    for (int b = 0; b < 4; ++b) {
        names.push_back("segmentation_head.blocks." + std::to_string(b) + ".dwconv.weight");
        names.push_back("segmentation_head.blocks." + std::to_string(b) + ".dwconv.bias");
        names.push_back("segmentation_head.blocks." + std::to_string(b) + ".norm.weight");
        names.push_back("segmentation_head.blocks." + std::to_string(b) + ".norm.bias");
        names.push_back("segmentation_head.blocks." + std::to_string(b) + ".pwconv1.weight");
        names.push_back("segmentation_head.blocks." + std::to_string(b) + ".pwconv1.bias");
    }
    names.push_back("segmentation_head.spatial_features_proj.weight");
    names.push_back("segmentation_head.spatial_features_proj.bias");
    names.push_back("segmentation_head.query_features_block.norm_in.weight");
    names.push_back("segmentation_head.query_features_block.norm_in.bias");
    names.push_back("segmentation_head.query_features_block.layers.0.weight");
    names.push_back("segmentation_head.query_features_block.layers.0.bias");
    names.push_back("segmentation_head.query_features_block.layers.2.weight");
    names.push_back("segmentation_head.query_features_block.layers.2.bias");
    names.push_back("segmentation_head.query_features_proj.weight");
    names.push_back("segmentation_head.query_features_proj.bias");
    names.push_back("segmentation_head.bias");
}
```

(Verify exact tensor names against gguf-py inspection from Task 2 Step 6.)

- [ ] **Step 3: Load the tensors into the model's Tensors struct**

Find where the existing tensors are loaded into a per-module struct (e.g., `Model::seg`). Add a `Seg` sub-struct in the appropriate header and populate it during load.

- [ ] **Step 4: Verify load via rfdetr-cli info**

```bash
cmake --build build -j 2>&1 | tail -3
build/bin/rfdetr-cli info --model models/rfdetr-seg-nano-f32.gguf
```

Expected: `info` shows `has_segmentation_head=true`, n_tensors includes the 36 extra seg tensors, no load errors.

- [ ] **Step 5: Commit**

```bash
git add src/model_loader.cpp src/model_loader.hpp
git commit -m "feat(loader): load segmentation_head config + tensors for RF-DETR-Seg models"
```

---

## Task 4: Implement segmentation head forward (isolated, against baseline)

**Files:**
- Create: `src/segmentation.hpp`
- Create: `src/segmentation.cpp`

- [ ] **Step 1: Define the API**

`src/segmentation.hpp`:

```cpp
#pragma once
#include "ggml.h"
#include "model_loader.hpp"
#include <vector>

namespace rfdetr {

struct SegOutput {
    ggml_tensor* masks;  // shape (H/4, W/4, num_queries, 1) — last decoder layer only at inference
};

// Build the segmentation head graph nodes onto an existing ggml context.
// Inputs:
//   spatial_features: (W, H, C=256, 1) — projector output, BEFORE resize
//   query_features:   (256, num_queries, 1) — final decoder layer output (post-norm)
//   image_h, image_w: input image dimensions
SegOutput segmentation_forward(
    ggml_context* ctx,
    const Model::Seg& weights,
    ggml_tensor* spatial_features,
    ggml_tensor* query_features,
    int image_h,
    int image_w,
    int mask_downsample_ratio);

}  // namespace rfdetr
```

- [ ] **Step 2: Implement DepthwiseConvBlock helper**

In `src/segmentation.cpp`, write a helper `depthwise_conv_block`:

```cpp
static ggml_tensor* depthwise_conv_block(
    ggml_context* ctx,
    ggml_tensor* x,                  // (W, H, C, 1)
    ggml_tensor* dwconv_w,           // (3, 3, 1, C) — depthwise: per-channel single kernel
    ggml_tensor* dwconv_b,           // (C,)
    ggml_tensor* norm_w,             // (C,)
    ggml_tensor* norm_b,             // (C,)
    ggml_tensor* pwconv_w,           // (C, C) — linear pointwise
    ggml_tensor* pwconv_b)           // (C,)
{
    // 1. Depthwise conv: ggml_conv_2d_dw (if available) or manual loop
    // 2. Add bias (broadcast across spatial)
    // 3. Permute to (C, W*H, 1) for LayerNorm across C
    // 4. ggml_norm + mul(norm_w) + add(norm_b)
    // 5. Linear pwconv: ggml_mul_mat(pwconv_w, x_flat) + add(pwconv_b)
    // 6. GELU
    // 7. Permute back to (W, H, C, 1) for next block input
    return x;
}
```

**Important**: ggml has `ggml_conv_2d_dw` for depthwise convolutions. Verify by `grep -n "conv_2d_dw" third_party/ggml/include/ggml.h`. If not available, implement as a `ggml_im2col` + per-channel `mul`.

- [ ] **Step 3: Implement MLPBlock helper**

```cpp
static ggml_tensor* mlp_block(
    ggml_context* ctx,
    ggml_tensor* x,           // (C=256, N=num_queries, 1)
    ggml_tensor* norm_w,      // (C,)
    ggml_tensor* norm_b,      // (C,)
    ggml_tensor* l0_w,        // (256, 1024)
    ggml_tensor* l0_b,        // (1024,)
    ggml_tensor* l2_w,        // (1024, 256)
    ggml_tensor* l2_b)        // (256,)
{
    // LayerNorm + Linear + GELU + Linear
    return x;
}
```

- [ ] **Step 4: Implement `segmentation_forward`**

Stub all four blocks (only the last is used at inference but we run all four for parity verification):

```cpp
SegOutput segmentation_forward(
    ggml_context* ctx,
    const Model::Seg& w,
    ggml_tensor* spatial_features,
    ggml_tensor* query_features,
    int image_h, int image_w,
    int mask_downsample_ratio)
{
    int target_h = image_h / mask_downsample_ratio;
    int target_w = image_w / mask_downsample_ratio;
    // 1. Bilinear interpolate spatial_features to (target_w, target_h)
    //    ggml_upscale_ext OR manual via interp (verify ggml API)
    ggml_tensor* spatial = /* upsample */ spatial_features;
    rfdetr_trace(spatial, "seg.spatial_features.resized");

    ggml_tensor* qf = query_features;  // (256, num_queries, 1)

    ggml_tensor* masks_final = nullptr;
    for (int b = 0; b < 4; ++b) {
        spatial = depthwise_conv_block(ctx, spatial,
            w.blocks[b].dwconv_w, w.blocks[b].dwconv_b,
            w.blocks[b].norm_w, w.blocks[b].norm_b,
            w.blocks[b].pwconv_w, w.blocks[b].pwconv_b);
        rfdetr_trace(spatial, ("seg.block." + std::to_string(b) + ".spatial_out").c_str());

        ggml_tensor* spatial_proj = /* Conv2d 1x1 spatial_features_proj */;
        ggml_tensor* qf_proj = mlp_block(ctx, qf, ...);
        qf_proj = ggml_mul_mat(ctx, w.query_features_proj_w, qf_proj);
        // (+ bias)

        // Einsum bchw, bnc -> bnhw:
        //   reshape spatial_proj from (W, H, C, 1) to (C, W*H, 1)
        //   matmul: (N=num_queries, C) @ (C, W*H) -> (N, W*H)
        //   reshape to (W, H, N, 1)
        ggml_tensor* spatial_flat = ggml_reshape_3d(ctx, spatial_proj, 256, target_w * target_h, 1);
        ggml_tensor* masks = ggml_mul_mat(ctx, spatial_flat, qf_proj);  // -> (W*H, N, 1)
        masks = ggml_add(ctx, masks, w.bias);  // broadcast scalar bias
        masks = ggml_reshape_4d(ctx, masks, target_w, target_h, /*N=*/qf_proj->ne[1], 1);
        rfdetr_trace(masks, ("seg.masks." + std::to_string(b)).c_str());
        if (b == 3) masks_final = masks;
    }
    rfdetr_trace(masks_final, "seg.masks.final");
    return {masks_final};
}

}  // namespace rfdetr
```

- [ ] **Step 5: Wire into CMake**

`src/CMakeLists.txt` — add `segmentation.cpp` to the rfdetr library target.

- [ ] **Step 6: Build (compile-only check)**

```bash
cmake --build build -j 2>&1 | tail -10
```

Fix any compile errors before moving on.

- [ ] **Step 7: Commit**

```bash
git add src/segmentation.cpp src/segmentation.hpp src/CMakeLists.txt
git commit -m "feat(segmentation): seg head forward (depthwise blocks + MLP + einsum)"
```

---

## Task 5: Wire seg head into rfdetr_model_forward

**Files:**
- Modify: `src/rfdetr_model.{cpp,hpp}`

- [ ] **Step 1: Extend `ModelOutput` to carry masks (when enabled)**

In `src/rfdetr_model.hpp`:

```cpp
struct ModelOutput {
    // ... existing fields ...
    std::vector<std::vector<float>> masks;  // optional; empty if no seg head. Each inner vector is (H_mask × W_mask) for one query.
    int mask_h = 0, mask_w = 0;
};
```

- [ ] **Step 2: Conditional seg forward in `rfdetr_model_forward`**

In `src/rfdetr_model.cpp`, after the decoder + heads section:

```cpp
if (m.config.has_segmentation_head) {
    SegOutput seg = segmentation_forward(
        ctxA, m.seg,
        projector_output,        // (W, H, C, 1) — the projector spatial output
        decoder_norm_output,     // (256, N, 1)
        image_h, image_w,
        m.config.mask_downsample_ratio);
    // Add seg.masks to the compute graph
    ggml_build_forward_expand(graphA, seg.masks);
    // After compute, copy back to ModelOutput.masks
}
```

- [ ] **Step 3: Build + sanity-check**

```bash
cmake --build build -j 2>&1 | tail -5
build/bin/rfdetr-cli detect --model models/rfdetr-seg-nano-f32.gguf \
    --input /tmp/coco_sample.jpg --threshold 0.5 --threads 8 \
    --output /tmp/seg_det.json
.venv/bin/python -c "import json; d=json.load(open('/tmp/seg_det.json')); print(f'detections: {len(d[\"detections\"])}'); print('keys:', d.keys())"
```

Expected: runs without crash, produces detections. Masks aren't yet in JSON output (that's Task 7).

- [ ] **Step 4: Commit**

```bash
git add src/rfdetr_model.cpp src/rfdetr_model.hpp
git commit -m "feat(forward): wire segmentation head when has_segmentation_head=true"
```

---

## Task 6: Parity test against torch seg baseline

**Files:**
- Create: `tests/test_parity_segmentation.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the parity test**

Pattern follows `tests/test_parity_decoder.cpp`:
1. Load `models/rfdetr-seg-nano-f32.gguf`
2. Load `tests/fixtures/baseline_torch_seg.gguf`
3. Use baseline's `parity.preprocess.input` as input
4. Run full forward with seg head
5. Diff captured trace tensors against baseline's `parity.seg.*`:
   - `seg.spatial_features.resized`
   - `seg.block.{0..3}.spatial_out`
   - `seg.masks.final`
6. Tolerance: target 1e-3 max_abs; if drift compounds expect ≤ 1e-2

- [ ] **Step 2: Register test**

`tests/CMakeLists.txt`:

```cmake
rfdetr_add_test(test_parity_segmentation)
```

- [ ] **Step 3: Build + run**

```bash
cmake --build build -j 2>&1 | tail -3
ctest --test-dir build -R test_parity_segmentation --output-on-failure
```

Expected: passes. If a checkpoint drifts wildly, debug starting from the first divergent one (typically `seg.spatial_features.resized` if the bilinear upsample is wrong, then per-block if the depthwise conv is off).

- [ ] **Step 4: Commit**

```bash
git add tests/test_parity_segmentation.cpp tests/CMakeLists.txt
git commit -m "test(parity): segmentation head vs torch baseline"
```

---

## Task 7: CLI `--masks` output

**Files:**
- Modify: `examples/cli/main.cpp` (cmd_detect)
- Modify: `include/rfdetr.h` (expose masks via detection struct)
- Modify: `src/rfdetr.cpp` (postprocess: threshold + upsample masks to original image size)

- [ ] **Step 1: Extend `rfdetr_detection` to carry mask data**

In `include/rfdetr.h`:

```c
typedef struct {
    int class_id;
    float score;
    float bbox[4];  // x1, y1, x2, y2
    // Optional: present only if model has segmentation_head
    int mask_width;
    int mask_height;
    const uint8_t* mask;  // packed binary mask (0 or 1), row-major, size = mask_w * mask_h
} rfdetr_detection;
```

The mask buffer is owned by the rfdetr_context (lives until next detect call). Document ownership clearly.

- [ ] **Step 2: Postprocess masks in `rfdetr_detect`**

In `src/rfdetr.cpp`:
- After top-K detection filter, also gather the corresponding mask rows from ModelOutput.masks
- For each surviving detection: upsample its mask from (H/4, W/4) to original image size via bilinear (CPU side)
- Threshold at 0.5 (or expose via `rfdetr_detect_params`)
- Store as packed uint8 buffer
- Wire pointer into `rfdetr_detection.mask`

- [ ] **Step 3: Add `--masks <dir>` CLI flag**

In `examples/cli/main.cpp::cmd_detect`:

```cpp
// Parse --masks <dir>
std::string masks_dir;
// ... add to arg parser ...

// After detect:
if (!masks_dir.empty()) {
    for (int i = 0; i < n_detections; ++i) {
        const auto& d = detections[i];
        if (!d.mask) continue;
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/det_%03d_class%d_score%.2f.png",
                      masks_dir.c_str(), i, d.class_id, d.score);
        // Convert binary mask to grayscale + use stbi_write_png
        std::vector<uint8_t> img(d.mask_width * d.mask_height);
        for (int p = 0; p < d.mask_width * d.mask_height; ++p) {
            img[p] = d.mask[p] ? 255 : 0;
        }
        stbi_write_png(path, d.mask_width, d.mask_height, 1, img.data(), d.mask_width);
    }
    std::fprintf(stderr, "wrote %d mask PNGs to %s\n", n_detections, masks_dir.c_str());
}
```

- [ ] **Step 4: Build + test**

```bash
cmake --build build -j 2>&1 | tail -3
mkdir -p /tmp/seg_masks
build/bin/rfdetr-cli detect --model models/rfdetr-seg-nano-f32.gguf \
    --input /tmp/coco_sample.jpg --threshold 0.5 \
    --masks /tmp/seg_masks --output /tmp/seg_det.json --threads 8
ls /tmp/seg_masks/
```

Expected: PNG files per detection. Visually inspect at least 2 to confirm the masks look correct (use an image viewer or `display`).

- [ ] **Step 5: Compare against PyTorch's seg output**

```bash
.venv/bin/python -c "
from rfdetr import RFDETRSegNano
m = RFDETRSegNano()
result = m.predict('/tmp/coco_sample.jpg', threshold=0.5)
print(result)
# Save masks as PNG for comparison
"
```

Verify that C++ masks are pixel-comparable to PyTorch's. Differences should be sub-pixel-shaped (binary mask boundary may shift by 1-2 px due to FP rounding, but the silhouettes should match).

- [ ] **Step 6: Commit**

```bash
git add examples/cli/main.cpp src/rfdetr.cpp include/rfdetr.h
git commit -m "feat(cli): --masks output for segmentation models"
```

---

## Task 8: Bench + plot seg variants

**Files:**
- Modify: `scripts/bench_community.py`
- Modify: `scripts/plot_community.py`
- Modify: `BENCHMARK.md`, `README.md`

- [ ] **Step 1: Convert remaining seg variants (one-time)**

```bash
for v in seg-nano seg-small seg-medium seg-large seg-xlarge seg-2xlarge; do
    .venv/bin/python scripts/convert_rfdetr_to_gguf.py --variant $v \
        --output models/rfdetr-${v}-f32.gguf
done
```

- [ ] **Step 2: Bench seg variants**

Extend bench_community.py with a `--seg` flag that runs through the 6 seg variants on the same image set, capturing latency + mask IoU vs PyTorch.

- [ ] **Step 3: Generate seg plot**

In plot_community.py, add `plot_seg_variants` — bar chart of latency × seg variant, plus per-variant mask IoU bar.

- [ ] **Step 4: Update BENCHMARK.md + README.md**

Add a Segmentation section with the variant table, plot, and a sample mask visualization.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_community.py scripts/plot_community.py BENCHMARK.md README.md benchmarks/plots/seg_*.{png,svg}
git commit -m "feat(bench): segmentation variants benchmark + plots"
```

---

## Self-Review

- **Spec coverage**: converter handles seg variants ✓, loader handles seg head ✓, seg forward implemented ✓, parity tested ✓, CLI outputs masks ✓, bench + plots ✓, docs ✓.
- **Placeholders**: a few "verify exact tensor names" / "verify ggml API" prompts in Task 4 Steps 2-4 — these are intentional verification points the implementer must hit; not lazy placeholders. The skeleton code is concrete; the implementer fills in the right ggml calls after verifying against the headers.
- **Type consistency**: `Model::Seg` struct referenced in Task 4 must be defined first in Task 3 — verify order. `rfdetr_detection.mask` field added in Task 7 must be reflected in any public-API consumer.

**Open questions to verify at execution time:**
1. Does ggml have `ggml_conv_2d_dw`? If yes, use it. If not, implement via `ggml_im2col` + per-channel multiply.
2. Does ggml have a bilinear `ggml_upscale_ext` that takes (target_h, target_w) directly? Or do we need to express bilinear via `ggml_pool_2d` + interpolation? Check `third_party/ggml/include/ggml.h`.
3. The `num_windows=1` config means the seg backbone has NO windowing. Our windowed-DinoV2 implementation (`src/dinov2.cpp`) treats `num_windows=1` as "all global attention" — verify this is the case (it should already work). If not, add a fast path.
4. The `patch_size=12` config differs from Base's 14. Our `src/dinov2.cpp` reads patch_size from metadata, so it should work — but the bicubic pos_embed resampling math depends on the patch grid. Verify the seg model's `pos_embed` shape matches what our `bicubic_resample_patch_grid` expects.
5. The `out_feature_indexes=[3,6,9,12]` (HF stage indices) maps to post-layer outputs [2,5,8,11] — same as Base. Verify our existing multiscale tap logic handles this correctly.
