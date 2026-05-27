# Fine-tuning Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document and verify the train-in-Python / infer-in-C++ workflow for fine-tuning RF-DETR on custom datasets and serving with rfdetr.cpp.

**Architecture:** We do NOT build a training pipeline in C++/ggml. Training in PyTorch + serving in optimized C++ is the established pattern (llama.cpp, vLLM, mlx-lm, whisper.cpp all follow it). Our job is (a) verify our existing converter handles fine-tuned checkpoints with arbitrary `num_classes`, (b) provide a clear README walkthrough that goes end-to-end from "I have a custom dataset" → "I have a deployable GGUF I can serve with `rfdetr-cli detect`".

**Tech Stack:** PyTorch (rfdetr 1.7.0 in `.venv` for training), Python converter (`scripts/convert_rfdetr_to_gguf.py`), C++ CLI (`rfdetr-cli`).

---

## File Structure

- Modify: `scripts/convert_rfdetr_to_gguf.py` — verify it doesn't hardcode `num_classes=91`; load it from the checkpoint's config
- Create: `docs/finetuning.md` — full walkthrough (dataset format → fine-tune command → convert → inference)
- Create: `tests/test_custom_classes.cpp` — load a checkpoint with `num_classes != 91`, verify the loader reports it correctly
- Modify: `README.md` — add a "Fine-tuning" section pointing at `docs/finetuning.md`
- Create: `scripts/build_custom_checkpoint.py` — generates a tiny "fake fine-tune" checkpoint with `num_classes=5` for testing (small synthetic dataset; runs in seconds)

---

## Task 1: Verify the converter reads num_classes from the checkpoint

**Files:**
- Modify: `scripts/convert_rfdetr_to_gguf.py`

- [ ] **Step 1: Read the converter for num_classes handling**

Open `scripts/convert_rfdetr_to_gguf.py`. Search for `num_classes` or `91` (a magic number that would indicate hardcoding). Document what you find.

If `num_classes` is read from `m.model_config.num_classes` (rfdetr's runtime config), the converter is already class-count-aware. Move on to Task 2.

If `num_classes` is hardcoded to 91 anywhere, replace it with `m.model_config.num_classes`.

- [ ] **Step 2: Verify class_embed tensor shape is derived, not hardcoded**

The converter writes the `class_embed.weight` tensor. Its shape must be `(num_classes, 256)`. Verify the script uses `tensor.shape` (from PyTorch) rather than a hardcoded value.

The expected lines look like:

```python
class_embed_weight = inner.class_embed.weight  # shape: (num_classes, 256)
writer.add_tensor("class_embed.weight", class_embed_weight.numpy())
```

This is already correct if PyTorch's tensor shape is the source of truth. If there's any reshape with `(91, 256)`, that's a bug to fix.

- [ ] **Step 3: Write a verification print at conversion time**

Add (near where num_classes is written to metadata):

```python
print(f"[convert] num_classes={num_classes}, class_embed shape={tuple(class_embed_weight.shape)}")
```

This makes any future hardcoding stand out immediately.

- [ ] **Step 4: Commit (or skip if no changes were needed)**

```bash
git add scripts/convert_rfdetr_to_gguf.py
git commit -m "fix(convert): ensure num_classes is read from checkpoint, not hardcoded"
```

If no changes, document the verification in commit-message-only form (no commit needed).

---

## Task 2: Generate a custom-classes test checkpoint

**Files:**
- Create: `scripts/build_custom_checkpoint.py`

- [ ] **Step 1: Write the helper**

```python
#!/usr/bin/env python3
"""scripts/build_custom_checkpoint.py — produces a tiny RF-DETR-Base checkpoint
with num_classes=5, suitable for testing the fine-tuning conversion path.

This is NOT a real fine-tune; it just resizes the class_embed head to 5 classes
and saves the checkpoint. Use it to validate that our GGUF converter + C++ loader
handle custom class counts correctly.
"""
import argparse
import torch
from rfdetr import RFDETRBase


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, help="Output .pth path")
    p.add_argument("--num-classes", type=int, default=5)
    args = p.parse_args()

    m = RFDETRBase()
    inner = m.model.model

    # Replace class_embed with a fresh head sized to N classes
    old = inner.class_embed
    new = torch.nn.Linear(old.in_features, args.num_classes, bias=True)
    torch.nn.init.zeros_(new.weight)
    torch.nn.init.zeros_(new.bias)
    inner.class_embed = new
    # Same for two-stage init's enc_out_class_embed (13 groups)
    for g in inner.transformer.enc_out_class_embed:
        g.weight.data = torch.zeros(args.num_classes, g.weight.shape[1])
        g.bias.data = torch.zeros(args.num_classes)

    # Save in rfdetr's checkpoint format (state_dict only; rfdetr does the wrapping)
    sd = m.model.model.state_dict()
    torch.save({"model": sd, "args": {"num_classes": args.num_classes}}, args.output)
    print(f"wrote {args.output} with num_classes={args.num_classes}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
.venv/bin/python scripts/build_custom_checkpoint.py --output /tmp/custom5.pth --num-classes 5
ls -lh /tmp/custom5.pth
```

Expected: small .pth file (~120 MB — same as full Base since only the head differs).

- [ ] **Step 3: Commit**

```bash
git add scripts/build_custom_checkpoint.py
git commit -m "scripts: helper to build a custom-num-classes test checkpoint"
```

---

## Task 3: Convert the custom checkpoint + verify

**Files:**
- None modified (verification step)

- [ ] **Step 1: Add a `--checkpoint` flag to the converter (if not present)**

The converter currently downloads + loads from rfdetr's pretrained URL via `RFDETRBase()`. To test fine-tuned checkpoints, we need a way to load from a local `.pth`.

In `scripts/convert_rfdetr_to_gguf.py`, add:

```python
parser.add_argument(
    "--checkpoint",
    default=None,
    help="Optional local .pth checkpoint to load instead of the pretrained download.",
)
```

After instantiating the model:

```python
if args.checkpoint:
    sd = torch.load(args.checkpoint, map_location="cpu")
    state = sd.get("model", sd)
    if "args" in sd and "num_classes" in sd["args"]:
        # Resize class_embed before loading state_dict
        nc = sd["args"]["num_classes"]
        inner = m.model.model
        old = inner.class_embed
        inner.class_embed = torch.nn.Linear(old.in_features, nc, bias=True)
        for g in inner.transformer.enc_out_class_embed:
            g.weight.data = torch.zeros(nc, g.weight.shape[1])
            g.bias.data = torch.zeros(nc)
        # Update model_config so downstream code reads the right count
        m.model_config.num_classes = nc
    m.model.model.load_state_dict(state, strict=False)
    print(f"[convert] loaded custom checkpoint: {args.checkpoint}")
```

- [ ] **Step 2: Convert the custom checkpoint**

```bash
.venv/bin/python scripts/convert_rfdetr_to_gguf.py \
    --checkpoint /tmp/custom5.pth \
    --output /tmp/custom5.gguf
ls -lh /tmp/custom5.gguf
```

Expected: GGUF file slightly smaller than base (5-class vs 91-class head).

- [ ] **Step 3: Verify via rfdetr-cli info**

```bash
build/bin/rfdetr-cli info --model /tmp/custom5.gguf | grep -E "num_classes|variant"
```

Expected: `num_classes=5`.

- [ ] **Step 4: Run detection (will produce zeros since the head is zero-initialized)**

```bash
build/bin/rfdetr-cli detect --model /tmp/custom5.gguf \
    --input /tmp/coco_sample.jpg --threshold 0.0 --topk 5 \
    --output /tmp/custom5_detect.json
.venv/bin/python -c "import json; d=json.load(open('/tmp/custom5_detect.json')); print(f'detections: {len(d[\"detections\"])}'); print('first:', d['detections'][:2])"
```

Expected: gets back N detections (because threshold=0), all with low class IDs in [0..4]. This verifies the loader handles non-91-class models end-to-end.

- [ ] **Step 5: Commit**

```bash
git add scripts/convert_rfdetr_to_gguf.py
git commit -m "feat(convert): --checkpoint flag for fine-tuned local checkpoints"
```

---

## Task 4: Add a C++ test for custom class counts

**Files:**
- Create: `tests/test_custom_classes.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the test**

```cpp
/* tests/test_custom_classes.cpp — verify the C-API correctly reports
 * non-default num_classes from a fine-tuned-style checkpoint. Skips
 * gracefully if /tmp/custom5.gguf is not present.
 */
#include "rfdetr.h"
#include "test_assert.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <string>

static bool file_exists(const std::string& p) {
    struct stat st; return ::stat(p.c_str(), &st) == 0;
}

int main() {
    const std::string path = "/tmp/custom5.gguf";
    if (!file_exists(path)) {
        std::fprintf(stderr, "[test_custom_classes] SKIP: generate %s via "
                     "scripts/build_custom_checkpoint.py + convert\n", path.c_str());
        return 0;
    }
    rfdetr_init_params p{};
    p.model_path = path.c_str();
    p.n_threads = 4;
    rfdetr_context* ctx = nullptr;
    rfdetr_status st = rfdetr_init(&p, &ctx);
    RFDETR_ASSERT(st == RFDETR_OK);
    rfdetr_model_info info{};
    st = rfdetr_get_model_info(ctx, &info);
    RFDETR_ASSERT(st == RFDETR_OK);
    RFDETR_ASSERT(info.num_classes == 5);
    std::fprintf(stderr, "[test_custom_classes] OK — num_classes=%d (expected 5)\n",
                 info.num_classes);
    rfdetr_free(ctx);
    return 0;
}
```

- [ ] **Step 2: Register in CMake**

In `tests/CMakeLists.txt`:

```cmake
rfdetr_add_test(test_custom_classes)
```

- [ ] **Step 3: Build + run**

```bash
cmake --build build -j 2>&1 | tail -3
ctest --test-dir build -R test_custom_classes --output-on-failure
```

Expected: passes (since /tmp/custom5.gguf was generated in Task 3) OR skips gracefully.

- [ ] **Step 4: Commit**

```bash
git add tests/test_custom_classes.cpp tests/CMakeLists.txt
git commit -m "test: verify loader reports correct num_classes for fine-tuned checkpoints"
```

---

## Task 5: Write the fine-tuning walkthrough doc

**Files:**
- Create: `docs/finetuning.md`

- [ ] **Step 1: Write the doc**

Create `docs/finetuning.md`:

```markdown
# Fine-tuning RF-DETR for rfdetr.cpp

The rfdetr.cpp runtime is inference-only. To fine-tune RF-DETR on a custom
dataset, use the upstream **rfdetr** Python library, then convert the
resulting checkpoint to GGUF for fast C++ inference.

## End-to-end workflow

### 1. Prepare your dataset (COCO format)

rfdetr expects COCO-format annotations:

\```
my_dataset/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── train/
│   └── *.jpg
└── val/
    └── *.jpg
\```

See the [rfdetr training docs](https://github.com/roboflow/rf-detr#training) for
dataset prep details.

### 2. Fine-tune in Python

\```python
from rfdetr import RFDETRBase

model = RFDETRBase()  # or RFDETRNano, RFDETRSmall, etc.
model.train(
    dataset_dir="my_dataset",
    num_classes=5,            # your custom class count
    output_dir="runs/my_train",
    epochs=50,
    batch_size=4,
    lr=1e-4,
)
\```

Output: `runs/my_train/checkpoint_best_total.pth` (or similar).

### 3. Convert the fine-tuned checkpoint to GGUF

\```bash
.venv/bin/python scripts/convert_rfdetr_to_gguf.py \
    --checkpoint runs/my_train/checkpoint_best_total.pth \
    --variant base \
    --output models/my_finetune.gguf
\```

The converter reads `num_classes` from the checkpoint metadata and resizes
the class head accordingly.

### 4. (Optional) Quantize for smaller deployment

\```bash
build/bin/rfdetr-cli quantize models/my_finetune.gguf models/my_finetune-q8_0.gguf q8_0
\```

Q8_0 gives 3.1× smaller files with no measurable accuracy loss on the COCO
benchmark — see [BENCHMARK.md](../BENCHMARK.md). For your custom dataset,
verify accuracy on the val set before deploying quantized.

### 5. Run inference with rfdetr-cli

\```bash
build/bin/rfdetr-cli detect \
    --model models/my_finetune-q8_0.gguf \
    --input some_image.jpg \
    --output detections.json \
    --threshold 0.5
\```

## Class names

The converter does not store class names in the GGUF (rfdetr's Python pipeline
keeps them in a separate JSON). To use class names in your application, pass
them alongside the GGUF (e.g., a `classes.txt` file you read in your own
serving code).

## Verifying accuracy parity vs PyTorch

Before deploying a fine-tuned model, compare detections from PyTorch and from
rfdetr.cpp on a sample image:

\```bash
# PyTorch reference
.venv/bin/python -c "
from rfdetr import RFDETRBase
import torch
m = RFDETRBase()
sd = torch.load('runs/my_train/checkpoint_best_total.pth', map_location='cpu')
m.model.model.load_state_dict(sd['model'], strict=False)
print(m.predict('test_image.jpg', threshold=0.5))
"

# rfdetr.cpp
build/bin/rfdetr-cli detect --model models/my_finetune.gguf \\
    --input test_image.jpg --threshold 0.5 --output cpp_dets.json
\```

Detections should match to sub-pixel precision (rfdetr.cpp's parity guarantee
on the upstream pretrained model carries forward to fine-tunes).

## What we do NOT support

- **Training in C++** — use rfdetr Python.
- **ONNX export** — rfdetr Python has an ONNX exporter; we don't ingest ONNX.
- **Class name embedding in GGUF** — pass class names separately in your app.
- **Dataset preprocessing in C++** — rfdetr.cpp expects a JPEG/PNG path and
  does standard ImageNet normalize internally; data augmentation happens
  during training in Python.
```

- [ ] **Step 2: Commit**

```bash
git add docs/finetuning.md
git commit -m "docs: fine-tuning walkthrough (train in Python, serve in C++)"
```

---

## Task 6: Link from README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add Fine-tuning section to README**

Near the top (after the Performance section, before deep tech), add:

```markdown
## Fine-tuning

rfdetr.cpp is inference-only. To fine-tune on a custom dataset, train with
the upstream [rfdetr](https://github.com/roboflow/rf-detr) Python library,
then convert the checkpoint to GGUF:

\```bash
.venv/bin/python scripts/convert_rfdetr_to_gguf.py \
    --checkpoint runs/my_train/checkpoint.pth \
    --output models/my_finetune.gguf
\```

See [docs/finetuning.md](docs/finetuning.md) for the end-to-end walkthrough.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(README): link to fine-tuning walkthrough"
```

---

## Self-Review

- **Spec coverage**: verify converter handles custom num_classes ✓, generate test checkpoint ✓, convert it ✓, C++ test ✓, walkthrough doc ✓, README link ✓.
- **Placeholders**: none.
- **Type consistency**: `rfdetr_model_info::num_classes` referenced in Task 4 must match the actual struct field — verify before writing.

**Open questions to verify at execution time:**
1. rfdetr's `model.train()` API — does it accept `num_classes` directly, or is it set via `model_config`? Check the docstring / signature in `.venv/lib/.../rfdetr/main.py`. Update Task 5's example accordingly.
2. The fine-tuned checkpoint format — `sd["model"]` is the standard but some training paths save as `sd["state_dict"]` or just the raw dict. The Task 3 converter logic uses `sd.get("model", sd)` to handle both — verify this is robust against the actual rfdetr output format.
