# Fine-tuning RF-DETR for rf-detr.cpp

rf-detr.cpp is inference-only. To fine-tune RF-DETR on a custom dataset,
train with the upstream [rfdetr](https://github.com/roboflow/rf-detr) Python
library (PyTorch Lightning under the hood), then convert the resulting
checkpoint to GGUF for fast C++ inference.

This mirrors the established pattern used by llama.cpp, whisper.cpp, and
vLLM: PyTorch owns training; an optimized C++/ggml runtime owns serving.

## End-to-end workflow

### 1. Prepare your dataset (COCO format)

rfdetr expects [COCO-format](https://cocodataset.org/#format-data) annotations:

```
my_dataset/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/            (optional)
    ├── _annotations.coco.json
    └── *.jpg
```

The default `dataset_file="roboflow"` mode expects the per-split layout above
(annotations inline in each split directory). See the
[rfdetr training docs](https://github.com/roboflow/rf-detr#training) for
alternative layouts (`coco`, `o365`, `yolo`).

The class count is auto-detected from the train split's COCO `categories`
list; you don't pass `num_classes` to `train()`.

### 2. Fine-tune in Python

```python
from rfdetr import RFDETRBase  # or RFDETRNano / RFDETRSmall / RFDETRMedium / RFDETRLarge

model = RFDETRBase()
model.train(
    dataset_dir="my_dataset",
    output_dir="runs/my_train",
    epochs=50,
    batch_size=4,
    lr=1e-4,
)
```

`train()` is a thin wrapper around `rfdetr.training.build_trainer()`
(PyTorch Lightning). See
[TrainConfig in rfdetr/config.py](https://github.com/roboflow/rf-detr/blob/main/rfdetr/config.py)
for the full hyperparameter list (`lr_encoder`, `weight_decay`,
`grad_accum_steps`, `ema_decay`, `warmup_epochs`, `multi_scale`, ...).

Output: a series of `runs/my_train/checkpoint*.pth` files; the metric-best
one is typically named `checkpoint_best_total.pth` or `checkpoint_best_ema.pth`.

Each checkpoint is a `dict` with:

* `"model"`: the underlying `LWDETR` `state_dict`
* `"args"`: training config (includes `num_classes`)
* `"model_name"`: e.g. `"RFDETRBase"`

### 3. Convert the fine-tuned checkpoint to GGUF

```bash
.venv/bin/python scripts/convert_rfdetr_to_gguf.py \
    --checkpoint runs/my_train/checkpoint_best_total.pth \
    --variant base \
    --dtype f32 \
    --output models/my_finetune-f32.gguf
```

What the converter does on `--checkpoint`:

1. `torch.load()` the `.pth` (weights_only=False; required because rfdetr
   embeds an `argparse.Namespace` in legacy checkpoints).
2. Reads the actual head size from `checkpoint["model"]["class_embed.bias"].shape[0]`.
3. Constructs `RFDETRBase()` and calls `reinitialize_detection_head(head_size)`
   so the classification head (shared `class_embed` + 13 `enc_out_class_embed`
   groups) matches the checkpoint shape.
4. `load_state_dict(strict=False)` (mask_token is the one legitimate "unused"
   entry, training-only).
5. Writes GGUF with `rfdetr.num_classes = head_size` and a synthesized
   `class_<i>` placeholder list as `rfdetr.class_names`.

Variant must match the variant you trained on (`base`/`nano`/`small`/
`medium`/`large`).

#### Class names

The converter writes placeholder `class_0`, `class_1`, ... entries to
`rfdetr.class_names`. To use human-readable names in your application, ship a
parallel `classes.txt` (or similar) alongside the GGUF and look them up in
your serving code. (Upstream rfdetr does not standardize class-name storage
in the `.pth` checkpoint; embedding them would be guessing a schema.)

### 4. (Optional) Quantize for smaller deployment

```bash
build/bin/rfdetr-cli quantize \
    models/my_finetune-f32.gguf \
    models/my_finetune-q8_0.gguf \
    q8_0
```

Q8_0 gives about 3.1x smaller files with no measurable accuracy loss on the
COCO benchmark; see [BENCHMARK.md](../BENCHMARK.md). For Q4 variants there are
caveats documented in the same doc. For your custom dataset, validate
accuracy on the val set before deploying quantized.

K-quants (`q4_K` / `q5_K` / `q6_K`) require the C++ quantizer (the Python
`gguf` package doesn't expose them).

### 5. Run inference with rfdetr-cli

```bash
build/bin/rfdetr-cli detect \
    --model models/my_finetune-q8_0.gguf \
    --input some_image.jpg \
    --output detections.json \
    --threshold 0.5
```

Or use the C-API directly from your application; see
[`include/rfdetr.h`](../include/rfdetr.h) and the
[CLI source](../examples/cli/main.cpp) as a reference integration.

## Quick smoke-test (no real training needed)

If you just want to verify the conversion path against a non-91-class
checkpoint, the repo ships a helper that builds a synthetic checkpoint by
resizing the pretrained Base head to N classes:

```bash
# Build a 5-class synthetic checkpoint
.venv/bin/python scripts/build_custom_checkpoint.py \
    --output /tmp/custom5.pth --num-classes 5

# Convert it
.venv/bin/python scripts/convert_rfdetr_to_gguf.py \
    --checkpoint /tmp/custom5.pth \
    --output /tmp/custom5.gguf \
    --dtype f32

# Verify the loader reports num_classes=5
build/bin/rfdetr-cli info --model /tmp/custom5.gguf
# variant:      base
# image_size:   560
# num_classes:  5
# num_queries:  300
# n_tensors:    486
```

The C++ test `tests/test_custom_classes.cpp` automates this check (it skips
gracefully if `/tmp/custom5.gguf` is not present).

## Verifying accuracy parity vs PyTorch

Before deploying a fine-tuned model, compare detections from PyTorch and
from rf-detr.cpp on the same image:

```bash
# PyTorch reference
.venv/bin/python -c "
from rfdetr import RFDETR
m = RFDETR.from_checkpoint('runs/my_train/checkpoint_best_total.pth')
print(m.predict('test_image.jpg', threshold=0.5))
"

# rf-detr.cpp
build/bin/rfdetr-cli detect \
    --model models/my_finetune-f32.gguf \
    --input test_image.jpg --threshold 0.5 \
    --output cpp_dets.json
```

Detections should match to sub-pixel precision. rf-detr.cpp's parity
guarantee on the upstream pretrained model (see
[`docs/parity.md`](parity.md)) carries forward to fine-tunes because the
forward pass is shape-agnostic over `num_classes`.

## What we do NOT support

- Training in C++. Train with rfdetr Python, then convert.
- ONNX export. rfdetr Python has an ONNX exporter; we don't ingest ONNX.
- Class name embedding in GGUF. Ship class names separately with your app
  (see "Class names" above).
- Dataset preprocessing in C++. rf-detr.cpp accepts a JPEG/PNG path and
  applies the standard ImageNet normalize internally; data augmentation
  happens during training in Python.
- Segmentation fine-tunes are not yet supported by the converter (detection
  only today).
