# rt-detr.cpp Loader Implementation Plan (Plan 2 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert a real RF-DETR `base` PyTorch checkpoint to GGUF format via a Python script, load and inspect that GGUF from C++ (variant detection, config parsing, tensor inventory validation), wire the public `rfdetr_init`/`rfdetr_free` and `info` CLI subcommand to use the loader, and implement the flat C ABI (`rfdetr_capi_*`) on top of the same. No inference yet — `detect` continues to emit empty detections; Plan 3 builds the forward graph.

**Architecture:** Two new translation units in `src/`: `model_loader.{cpp,hpp}` owns GGUF parsing + variant config + tensor-name registry; `rfdetr.cpp` owns the public `rfdetr_context` lifecycle and now constructs a real model on `rfdetr_init`. `rfdetr_capi.cpp` is a thin shim over the opaque-pointer API. The Python script `scripts/convert_rfdetr_to_gguf.py` calls into the upstream `rfdetr` PyPI package to read the checkpoint, then writes a GGUF using the canonical `gguf` Python library. For tests we generate a tiny *synthesized* GGUF in C++ via ggml's gguf-writer so unit tests run without PyTorch in CI.

**Tech Stack:** C++17, ggml (already linked from Plan 1, gguf API), Python 3.9+ (conversion script: `torch`, `rfdetr`, `gguf`, `numpy`), CMake ≥ 3.14, ctest.

Label rendering (`stb_truetype` + embedded font) is **deferred** to a follow-up plan; the C API's `class_name` field is populated from GGUF metadata (Plan 2) but not yet rendered onto annotated PNGs.

---

## Scope decisions (one-line each)

- **Variant scope:** only `base`. The loader code is variant-agnostic — config keys drive everything — but the conversion script and parity test fixture target `base`. Plan 4 adds nano/small/medium/large.
- **Weight loading:** parse metadata + tensor *descriptors* only. Tensor data is not yet copied into a ggml backend buffer (that's Plan 3, where it's needed for inference). For Plan 2 we validate the tensor set is present.
- **Conversion script:** consumes a HuggingFace checkpoint via `rfdetr` PyPI package; emits F16. Quantization is Plan 4.
- **Test fixture:** a synthesized minimal GGUF built at test-build time by a small C++ generator that uses ggml's gguf-writer API. Real checkpoint conversion is exercised only by the developer running `make convert` (or the upstream baseline workflow); CI doesn't need PyTorch.
- **`detect` path:** still returns empty detections. `rfdetr_detect` now needs a loaded context but otherwise behaves as in Plan 1.

---

## File map (created or modified in this plan)

```
rt-detr.cpp/
├── docs/
│   ├── conversion.md            # NEW — tensor name map, metadata schema
│   └── variants.md              # NEW — parameter table (just `base` for now)
├── scripts/
│   ├── requirements.txt         # NEW — pinned Python deps
│   └── convert_rfdetr_to_gguf.py  # NEW — HF → GGUF for `base`
├── src/
│   ├── model_loader.hpp         # NEW — rfdetr_config struct, model_loader API
│   ├── model_loader.cpp         # NEW — GGUF parse + variant detect + tensor inventory
│   ├── rfdetr.cpp               # NEW — rfdetr_context lifecycle, rfdetr_init / rfdetr_free / rfdetr_detect
│   ├── rfdetr_capi.cpp          # NEW — flat C ABI shim
│   └── (existing files unchanged)
├── tests/
│   ├── CMakeLists.txt           # MODIFY — register new tests + the gguf fixture generator
│   ├── fixtures/
│   │   └── gen_model_gguf.cpp   # NEW — synthesizes a minimal rfdetr-base GGUF for tests
│   ├── test_model_loader.cpp    # NEW
│   ├── test_capi.cpp            # NEW — opaque-API smoke
│   └── test_capi_flat.cpp       # NEW — flat-ABI smoke
├── examples/cli/main.cpp        # MODIFY — implement `info` subcommand
├── CMakeLists.txt               # MODIFY — add new sources to librfdetr; remove rfdetr-cli's manual src/ include and let it pick up via target_include of librfdetr
└── README.md                    # MODIFY — Plan 2 status
```

---

### Task 1: Document the GGUF metadata schema

Lock the contract first. The conversion script writes these keys; the C++ loader reads them. Both sides reference this doc.

**Files:**
- Create: `docs/conversion.md`
- Create: `docs/variants.md`

- [ ] **Step 1: Write `docs/conversion.md`**

```markdown
# rt-detr.cpp GGUF Conversion

This document is the *contract* between `scripts/convert_rfdetr_to_gguf.py` and
`src/model_loader.cpp`. Both sides reference the same keys and tensor names.
Changes here require updating both sides and bumping `rfdetr.format.version`.

## Format version

Key: `rfdetr.format.version` (string)
Current: `"1"`

Bumped whenever the metadata schema or tensor-name convention changes
incompatibly. The loader refuses any value other than `"1"`.

## Metadata keys

All under the `rfdetr.` namespace.

| Key                          | Type      | Description                                  |
|------------------------------|-----------|----------------------------------------------|
| `rfdetr.format.version`      | string    | `"1"` (see above).                           |
| `rfdetr.variant`             | string    | One of `nano|small|base|medium|large`.       |
| `rfdetr.image_size`          | uint32    | Square input side, e.g. 560.                 |
| `rfdetr.num_queries`         | uint32    | Detection queries (e.g. 300).                |
| `rfdetr.num_classes`         | uint32    | Output classes (e.g. 80 for COCO).           |
| `rfdetr.class_names`         | string[]  | One per class, length `num_classes`.         |
| `rfdetr.preprocess.mean`     | float32[3]| Per-channel mean (ImageNet: 0.485, 0.456, 0.406). |
| `rfdetr.preprocess.std`      | float32[3]| Per-channel std  (ImageNet: 0.229, 0.224, 0.225). |
| `rfdetr.backbone.dim`        | uint32    | Backbone model dim.                          |
| `rfdetr.backbone.depth`      | uint32    | Number of backbone blocks.                   |
| `rfdetr.backbone.heads`      | uint32    | Backbone attention heads.                    |
| `rfdetr.backbone.window_size`| uint32    | Windowed-attention window (0 = global only). |
| `rfdetr.backbone.multi_scale_layers` | uint32[] | Backbone layer indices tapped for multi-scale features. |
| `rfdetr.encoder.layers`      | uint32    | Encoder layer count.                         |
| `rfdetr.encoder.model_dim`   | uint32    | Encoder/decoder hidden dim.                  |
| `rfdetr.encoder.ffn_dim`     | uint32    | Encoder FFN hidden dim.                      |
| `rfdetr.encoder.heads`       | uint32    | Encoder attention heads.                     |
| `rfdetr.decoder.layers`      | uint32    | Decoder layer count.                         |
| `rfdetr.decoder.model_dim`   | uint32    | (Usually equals encoder.model_dim.)          |
| `rfdetr.decoder.ffn_dim`     | uint32    | Decoder FFN hidden dim.                      |
| `rfdetr.decoder.heads`       | uint32    | Decoder attention heads.                     |

## Tensor naming

Tensor names follow the C++ runtime's expectations. The conversion script maps
PyTorch state_dict keys → these names. Names use `.` separators and zero-based
indices.

**Backbone (DINOv2 ViT):**

| Name pattern                              | Shape (example)         | Source PyTorch key (rfdetr-base) |
|-------------------------------------------|-------------------------|-----------------------------------|
| `backbone.patch_embed.weight`             | `[dim, 3, 14, 14]`      | `backbone.patch_embed.proj.weight` |
| `backbone.patch_embed.bias`               | `[dim]`                 | `backbone.patch_embed.proj.bias`   |
| `backbone.pos_embed`                      | `[1, n_tokens, dim]`    | `backbone.pos_embed`               |
| `backbone.cls_token`                      | `[1, 1, dim]`           | `backbone.cls_token`               |
| `backbone.blocks.{i}.norm1.{weight,bias}` | `[dim]` each            | `backbone.blocks.{i}.norm1.{weight,bias}` |
| `backbone.blocks.{i}.attn.qkv.{weight,bias}` | `[3*dim, dim]`, `[3*dim]` | `backbone.blocks.{i}.attn.qkv.{weight,bias}` |
| `backbone.blocks.{i}.attn.proj.{weight,bias}` | `[dim, dim]`, `[dim]` | `backbone.blocks.{i}.attn.proj.{weight,bias}` |
| `backbone.blocks.{i}.norm2.{weight,bias}` | `[dim]` each            | `backbone.blocks.{i}.norm2.{weight,bias}` |
| `backbone.blocks.{i}.mlp.fc1.{weight,bias}` | `[ffn, dim]`, `[ffn]` | `backbone.blocks.{i}.mlp.fc1.{weight,bias}` |
| `backbone.blocks.{i}.mlp.fc2.{weight,bias}` | `[dim, ffn]`, `[dim]` | `backbone.blocks.{i}.mlp.fc2.{weight,bias}` |
| `backbone.norm.{weight,bias}`             | `[dim]` each            | `backbone.norm.{weight,bias}`     |

**Projector (multi-scale):**

| Name pattern                            | Shape (example)        |
|-----------------------------------------|------------------------|
| `projector.level{j}.weight`             | `[model_dim, dim]`     |
| `projector.level{j}.bias`               | `[model_dim]`          |
| `projector.level_embed`                 | `[n_levels, model_dim]`|

**Transformer encoder:**

| Name pattern                                  | Shape (example)                |
|-----------------------------------------------|--------------------------------|
| `encoder.layers.{i}.self_attn.qkv.{w,b}`      | `[3*model_dim, model_dim]`, `[3*model_dim]` |
| `encoder.layers.{i}.self_attn.out.{w,b}`      | `[model_dim, model_dim]`, `[model_dim]`     |
| `encoder.layers.{i}.norm1.{weight,bias}`      | `[model_dim]` each             |
| `encoder.layers.{i}.ffn.fc1.{weight,bias}`    | `[ffn_dim, model_dim]`, `[ffn_dim]`         |
| `encoder.layers.{i}.ffn.fc2.{weight,bias}`    | `[model_dim, ffn_dim]`, `[model_dim]`       |
| `encoder.layers.{i}.norm2.{weight,bias}`      | `[model_dim]` each             |

**Transformer decoder:**

| Name pattern                                  | Shape (example)                |
|-----------------------------------------------|--------------------------------|
| `decoder.queries`                             | `[num_queries, model_dim]`     |
| `decoder.layers.{i}.self_attn.qkv.{w,b}`      | `[3*model_dim, model_dim]`, `[3*model_dim]` |
| `decoder.layers.{i}.self_attn.out.{w,b}`      | `[model_dim, model_dim]`, `[model_dim]`     |
| `decoder.layers.{i}.norm1.{weight,bias}`      | `[model_dim]` each             |
| `decoder.layers.{i}.cross_attn.q.{w,b}`       | `[model_dim, model_dim]`, `[model_dim]`     |
| `decoder.layers.{i}.cross_attn.kv.{w,b}`      | `[2*model_dim, model_dim]`, `[2*model_dim]` |
| `decoder.layers.{i}.cross_attn.out.{w,b}`     | `[model_dim, model_dim]`, `[model_dim]`     |
| `decoder.layers.{i}.norm2.{weight,bias}`      | `[model_dim]` each             |
| `decoder.layers.{i}.ffn.fc1.{weight,bias}`    | `[ffn_dim, model_dim]`, `[ffn_dim]`         |
| `decoder.layers.{i}.ffn.fc2.{weight,bias}`    | `[model_dim, ffn_dim]`, `[model_dim]`       |
| `decoder.layers.{i}.norm3.{weight,bias}`      | `[model_dim]` each             |

**Heads:**

| Name pattern                  | Shape (example)                  |
|-------------------------------|----------------------------------|
| `heads.class.fc.{w,b}`        | `[num_classes, model_dim]`, `[num_classes]` |
| `heads.bbox.fc1.{w,b}`        | `[model_dim, model_dim]`, `[model_dim]`     |
| `heads.bbox.fc2.{w,b}`        | `[model_dim, model_dim]`, `[model_dim]`     |
| `heads.bbox.fc3.{w,b}`        | `[4, model_dim]`, `[4]`                     |

## Discovery workflow

The PyTorch tensor names above are *expected* for the rfdetr-base release at
the version pinned in `scripts/requirements.txt`. Upstream renames are possible.
The conversion script's first task is to enumerate `state_dict().keys()`,
diff against the expected set, and refuse to convert if there are missing or
unmapped keys. Bringing up a new variant or upstream version starts by running
`python scripts/convert_rfdetr_to_gguf.py --dry-run` and reading the diff.
```

- [ ] **Step 2: Write `docs/variants.md`**

```markdown
# RF-DETR Variant Parameters

This document tabulates the per-variant configuration that gets stamped into
GGUF metadata. Plan 2 only ships `base`; Plan 4 adds the others.

## base

| Parameter                | Value |
|--------------------------|-------|
| `image_size`             | 560   |
| `num_queries`            | 300   |
| `num_classes`            | 80 (COCO) |
| `backbone.dim`           | 768   |
| `backbone.depth`         | 12    |
| `backbone.heads`         | 12    |
| `backbone.window_size`   | 14    |
| `backbone.multi_scale_layers` | `[2, 5, 8, 11]` |
| `encoder.layers`         | 3     |
| `encoder.model_dim`      | 256   |
| `encoder.ffn_dim`        | 2048  |
| `encoder.heads`          | 8     |
| `decoder.layers`         | 3     |
| `decoder.model_dim`      | 256   |
| `decoder.ffn_dim`        | 2048  |
| `decoder.heads`          | 8     |

Values reflect the upstream `rfdetr-base` checkpoint at the pinned version of
the `rfdetr` PyPI package. The conversion script reads them dynamically from
the loaded model; the loader treats them as opaque metadata.

## nano / small / medium / large

Filled in by Plan 4 when those variants land.
```

- [ ] **Step 3: Commit**

```bash
git add docs/conversion.md docs/variants.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
docs: document GGUF metadata schema and variant parameters

Locks the contract between scripts/convert_rfdetr_to_gguf.py and
src/model_loader.cpp before either is written. Format version "1"; all
metadata keys under the rfdetr.* namespace; tensor naming convention
documented per layer family.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Python deps + skeleton conversion script

**Files:**
- Create: `scripts/requirements.txt`
- Create: `scripts/convert_rfdetr_to_gguf.py` (skeleton; tensor mapping in Task 3)

- [ ] **Step 1: Write `scripts/requirements.txt`**

```
# Pinned at last known-good commit at time of writing.
# Bump deliberately; re-run `python scripts/convert_rfdetr_to_gguf.py --dry-run`
# to confirm the tensor name map still matches.
torch==2.5.1
rfdetr==0.4.0
gguf==0.10.0
numpy==1.26.4
pillow==10.4.0
```

(If `rfdetr==0.4.0` is not the latest at implementation time, pick the
current stable and document it.)

- [ ] **Step 2: Write `scripts/convert_rfdetr_to_gguf.py` skeleton**

```python
#!/usr/bin/env python3
"""Convert an upstream rfdetr PyTorch checkpoint to GGUF.

Usage:
    python scripts/convert_rfdetr_to_gguf.py --variant base \
        --output rfdetr-base-f16.gguf [--dtype f16|f32] [--dry-run]

Format version: "1" (see docs/conversion.md).
"""

import argparse
import sys
from pathlib import Path


FORMAT_VERSION = "1"

# Per-variant config that gets stamped into GGUF metadata.
# Pulled into the script so we don't depend on the rfdetr package exposing it
# (it might be private). Verified against the pinned rfdetr version.
VARIANTS = {
    "base": {
        "image_size":      560,
        "num_queries":     300,
        "num_classes":     80,
        "backbone": {
            "dim":               768,
            "depth":             12,
            "heads":             12,
            "window_size":       14,
            "multi_scale_layers": [2, 5, 8, 11],
        },
        "encoder": {"layers": 3, "model_dim": 256, "ffn_dim": 2048, "heads": 8},
        "decoder": {"layers": 3, "model_dim": 256, "ffn_dim": 2048, "heads": 8},
    },
    # nano / small / medium / large added in Plan 4
}

# COCO 80-class names; rfdetr ships with COCO pretraining.
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=sorted(VARIANTS.keys()), default="base")
    p.add_argument("--output", required=False,
                   help="Output GGUF path. Default: rfdetr-<variant>-<dtype>.gguf")
    p.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    p.add_argument("--checkpoint",
                   help="Optional path to a local rfdetr .pth. Default: download via rfdetr pkg.")
    p.add_argument("--dry-run", action="store_true",
                   help="Load model + print tensor diff, but do not write GGUF.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.output is None:
        args.output = f"rfdetr-{args.variant}-{args.dtype}.gguf"

    # Lazy imports so --help works without torch / rfdetr installed
    import torch
    try:
        from rfdetr import RFDETRBase  # Task 3 imports the correct class per variant
    except ImportError:
        print("error: 'rfdetr' package not installed. pip install -r scripts/requirements.txt",
              file=sys.stderr)
        return 2

    print(f"Loading rfdetr-{args.variant} ...", file=sys.stderr)
    # Task 3 fills in:
    # - load model (handles --checkpoint vs auto-download)
    # - enumerate state_dict
    # - build the tensor name map
    # - validate (no missing / no unmapped)
    # - if --dry-run: print diff and return
    # - else: open GGUF writer, write metadata + tensors, close

    raise NotImplementedError("Task 3 fills in the conversion body")


if __name__ == "__main__":
    sys.exit(main())
```

Make it executable:
```bash
chmod +x scripts/convert_rfdetr_to_gguf.py
```

- [ ] **Step 3: Smoke-check --help works**

```bash
python scripts/convert_rfdetr_to_gguf.py --help
```

Expected: argparse prints usage and exits 0. (Lazy import means torch/rfdetr
don't need to be installed for `--help`.)

- [ ] **Step 4: Commit**

```bash
git add scripts/requirements.txt scripts/convert_rfdetr_to_gguf.py
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(convert): skeleton conversion script with variant config + COCO classes

CLI surface is final; the tensor-name mapping body is filled in by the next
task. requirements.txt pins torch/rfdetr/gguf/numpy/pillow. --help works
without the heavy deps installed (lazy imports).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Implement conversion body

This is the only Python-heavy task. It requires a working `pip install -r scripts/requirements.txt` on the developer host. CI does not run this.

**Files:**
- Modify: `scripts/convert_rfdetr_to_gguf.py` (replace the `raise NotImplementedError`)

- [ ] **Step 1: Install deps in a venv**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

If installation fails (e.g. CUDA wheel issues, rfdetr version unavailable),
STOP and report BLOCKED with the exact error. Do not work around it by
downgrading silently.

- [ ] **Step 2: Probe the rfdetr state_dict**

Write a small one-off script to print the keys:

```python
# scratch_probe.py — NOT checked in
from rfdetr import RFDETRBase
m = RFDETRBase()
for k, v in m.model.state_dict().items():
    print(f"{k}\t{tuple(v.shape)}\t{v.dtype}")
```

```bash
python scratch_probe.py > /tmp/rfdetr_keys.txt
wc -l /tmp/rfdetr_keys.txt
head -30 /tmp/rfdetr_keys.txt
```

Compare the keys against the expected names in `docs/conversion.md`. If the
actual API exposes the underlying torch model under a different attribute
(e.g. `m.model.model` or `m._model`), discover the right access path before
writing the conversion. The exact attribute path is one of the unknowns this
task discovers.

- [ ] **Step 3: Build the tensor-name map**

In `convert_rfdetr_to_gguf.py`, add a `TENSOR_NAME_MAP` dict at module scope.
Keys are GGUF names per `docs/conversion.md`; values are either:
- a string (direct rename of the PyTorch state_dict key), or
- a callable `(state_dict) -> torch.Tensor` for composite mappings (e.g. if
  rfdetr stores Q, K, V separately and we want them packed as `qkv.weight`).

Generate the map programmatically for repeated layer blocks (avoid 12×8 manual
entries). Example:

```python
def build_tensor_name_map(variant_cfg):
    m = {}
    bb_depth = variant_cfg["backbone"]["depth"]
    enc_layers = variant_cfg["encoder"]["layers"]
    dec_layers = variant_cfg["decoder"]["layers"]

    # patch embed (PyTorch: backbone.patch_embed.proj.weight; GGUF: backbone.patch_embed.weight)
    m["backbone.patch_embed.weight"] = "backbone.patch_embed.proj.weight"
    m["backbone.patch_embed.bias"]   = "backbone.patch_embed.proj.bias"
    m["backbone.pos_embed"]          = "backbone.pos_embed"
    m["backbone.cls_token"]          = "backbone.cls_token"

    for i in range(bb_depth):
        for layer_norm in ("norm1", "norm2"):
            for suffix in ("weight", "bias"):
                key = f"backbone.blocks.{i}.{layer_norm}.{suffix}"
                m[key] = key  # identity rename
        m[f"backbone.blocks.{i}.attn.qkv.weight"] = f"backbone.blocks.{i}.attn.qkv.weight"
        m[f"backbone.blocks.{i}.attn.qkv.bias"]   = f"backbone.blocks.{i}.attn.qkv.bias"
        m[f"backbone.blocks.{i}.attn.proj.weight"] = f"backbone.blocks.{i}.attn.proj.weight"
        m[f"backbone.blocks.{i}.attn.proj.bias"]   = f"backbone.blocks.{i}.attn.proj.bias"
        m[f"backbone.blocks.{i}.mlp.fc1.weight"] = f"backbone.blocks.{i}.mlp.fc1.weight"
        m[f"backbone.blocks.{i}.mlp.fc1.bias"]   = f"backbone.blocks.{i}.mlp.fc1.bias"
        m[f"backbone.blocks.{i}.mlp.fc2.weight"] = f"backbone.blocks.{i}.mlp.fc2.weight"
        m[f"backbone.blocks.{i}.mlp.fc2.bias"]   = f"backbone.blocks.{i}.mlp.fc2.bias"

    m["backbone.norm.weight"] = "backbone.norm.weight"
    m["backbone.norm.bias"]   = "backbone.norm.bias"

    # Projector / encoder / decoder / heads — fill in based on the keys probed
    # in Step 2. The exact PyTorch keys for these layers are discovered, not
    # assumed: rfdetr's transformer / projector / heads may use different
    # internal names than the GGUF spec. If discovery shows the source names
    # differ, write the mapping accordingly.

    # ... (continue per docs/conversion.md tensor table)

    return m
```

If the probe in Step 2 shows the rfdetr backbone uses different keys (e.g.
`backbone.0.blocks.{i}...` or some other wrapper), update the mapping and
note the deviation in a comment.

- [ ] **Step 4: Validate the map**

Add a `validate(state_dict, name_map)` function that:
- For each value in `name_map`, asserts the source key exists in `state_dict`
- For each unused source key, prints a warning
- Returns `(missing_targets, unused_sources)`

```python
def validate(state_dict, name_map):
    missing = []
    used = set()
    for gguf_name, src in name_map.items():
        if isinstance(src, str):
            if src not in state_dict:
                missing.append((gguf_name, src))
            else:
                used.add(src)
        else:
            # callable — we cannot statically check sources, but the lambda
            # is expected to consume named tensors itself
            pass
    unused = sorted(set(state_dict.keys()) - used)
    return missing, unused
```

- [ ] **Step 5: Implement the writer**

```python
def write_gguf(path, variant_cfg, name_map, state_dict, dtype):
    import gguf
    import numpy as np

    writer = gguf.GGUFWriter(path, arch="rfdetr")
    writer.add_string("rfdetr.format.version", FORMAT_VERSION)
    writer.add_string("rfdetr.variant", variant_cfg["__variant_name"])
    writer.add_uint32("rfdetr.image_size", variant_cfg["image_size"])
    writer.add_uint32("rfdetr.num_queries", variant_cfg["num_queries"])
    writer.add_uint32("rfdetr.num_classes", variant_cfg["num_classes"])
    writer.add_array("rfdetr.class_names", COCO_CLASSES)
    writer.add_array("rfdetr.preprocess.mean",
                     np.array([0.485, 0.456, 0.406], dtype=np.float32).tolist())
    writer.add_array("rfdetr.preprocess.std",
                     np.array([0.229, 0.224, 0.225], dtype=np.float32).tolist())

    bb = variant_cfg["backbone"]
    writer.add_uint32("rfdetr.backbone.dim",         bb["dim"])
    writer.add_uint32("rfdetr.backbone.depth",       bb["depth"])
    writer.add_uint32("rfdetr.backbone.heads",       bb["heads"])
    writer.add_uint32("rfdetr.backbone.window_size", bb["window_size"])
    writer.add_array("rfdetr.backbone.multi_scale_layers",
                     [int(x) for x in bb["multi_scale_layers"]])

    for stage in ("encoder", "decoder"):
        cfg = variant_cfg[stage]
        writer.add_uint32(f"rfdetr.{stage}.layers",    cfg["layers"])
        writer.add_uint32(f"rfdetr.{stage}.model_dim", cfg["model_dim"])
        writer.add_uint32(f"rfdetr.{stage}.ffn_dim",   cfg["ffn_dim"])
        writer.add_uint32(f"rfdetr.{stage}.heads",     cfg["heads"])

    np_dtype = np.float16 if dtype == "f16" else np.float32
    for gguf_name, src in name_map.items():
        if isinstance(src, str):
            t = state_dict[src].detach().cpu().to(torch.float32).numpy().astype(np_dtype)
        else:
            t = src(state_dict).detach().cpu().to(torch.float32).numpy().astype(np_dtype)
        writer.add_tensor(gguf_name, t)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
```

- [ ] **Step 6: Wire `main()` to call the pieces**

Replace the `raise NotImplementedError` in `main()` with:

```python
    import torch
    cfg = dict(VARIANTS[args.variant])
    cfg["__variant_name"] = args.variant

    if args.checkpoint:
        # Load state_dict directly from file
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
    else:
        if args.variant == "base":
            model = RFDETRBase()
        else:
            print(f"variant {args.variant} not wired in conversion script yet (Plan 4)",
                  file=sys.stderr)
            return 2
        state_dict = model.model.state_dict()  # discovered access path; adjust per Step 2

    name_map = build_tensor_name_map(cfg)
    missing, unused = validate(state_dict, name_map)

    if missing:
        print(f"\nERROR: {len(missing)} mapped tensors not found in state_dict:", file=sys.stderr)
        for gguf_name, src in missing[:20]:
            print(f"  {gguf_name}  <-  {src}  (source missing)", file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more", file=sys.stderr)
        return 3

    if unused:
        print(f"\nWARNING: {len(unused)} state_dict keys not mapped to GGUF (will be dropped):",
              file=sys.stderr)
        for k in unused[:20]:
            print(f"  {k}", file=sys.stderr)
        if len(unused) > 20:
            print(f"  ... and {len(unused) - 20} more", file=sys.stderr)

    if args.dry_run:
        print(f"\nDry run: would write {len(name_map)} tensors to {args.output}", file=sys.stderr)
        return 0

    write_gguf(args.output, cfg, name_map, state_dict, args.dtype)
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0
```

- [ ] **Step 7: Smoke-test (dry-run first, then real conversion)**

```bash
python scripts/convert_rfdetr_to_gguf.py --variant base --dry-run
```

Expected: validates the tensor set, prints any unused keys, exits 0.
If `missing` is non-empty, update `build_tensor_name_map` per the probe.

```bash
python scripts/convert_rfdetr_to_gguf.py --variant base \
    --output /tmp/rfdetr-base-f16.gguf
ls -lh /tmp/rfdetr-base-f16.gguf
```

Expected: GGUF written, file size in the tens-to-hundreds of MB range for
F16 base.

- [ ] **Step 8: Commit**

```bash
git add scripts/convert_rfdetr_to_gguf.py
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(convert): implement HF -> GGUF body for rf-detr base variant

Builds the tensor name map (programmatic for repeated layer blocks),
validates against the state_dict, writes GGUF metadata + F16 tensors via
the gguf PyPI package. --dry-run validates without writing. Other
variants (nano/small/medium/large) wired in Plan 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: model_loader.hpp — rfdetr_config struct and loader API

**Files:**
- Create: `src/model_loader.hpp`

- [ ] **Step 1: Write `src/model_loader.hpp`**

```cpp
#ifndef RFDETR_MODEL_LOADER_HPP
#define RFDETR_MODEL_LOADER_HPP

#include "rfdetr.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;

namespace rfdetr {

/* Configuration loaded from GGUF metadata. Plan 3's graph builder consumes
 * this; Plan 2's loader only populates it. */
struct Config {
    std::string variant;          // "base" / "nano" / ...
    uint32_t image_size  = 0;
    uint32_t num_queries = 0;
    uint32_t num_classes = 0;
    std::vector<std::string> class_names;
    float preprocess_mean[3] = {0, 0, 0};
    float preprocess_std[3]  = {1, 1, 1};

    struct {
        uint32_t dim                 = 0;
        uint32_t depth               = 0;
        uint32_t heads               = 0;
        uint32_t window_size         = 0;
        std::vector<uint32_t> multi_scale_layers;
    } backbone;

    struct {
        uint32_t layers    = 0;
        uint32_t model_dim = 0;
        uint32_t ffn_dim   = 0;
        uint32_t heads     = 0;
    } encoder, decoder;
};

/* Loaded model. Plan 2 populates `config` and `tensors` (ggml_tensor
 * descriptors from gguf_init_from_file — data is NOT loaded into a backend
 * buffer yet). Plan 3 will add a backend buffer + actual data load. */
struct Model {
    Config config;
    ::gguf_context* gguf  = nullptr;
    ::ggml_context* meta  = nullptr;  // ggml_context produced by gguf_init_from_file
    std::unordered_map<std::string, ::ggml_tensor*> tensors;
};

/* Load a model from a GGUF file at `path`. Returns nullptr on error and sets
 * `*out_status`. Caller owns the returned pointer; free with
 * `rfdetr::model_free`. */
Model* model_load(const std::string& path, rfdetr_status* out_status);

/* Free a model returned by `model_load`. Releases the gguf_context and
 * underlying ggml_context. */
void model_free(Model* m);

/* Validate that the expected tensor set for the config's variant is present
 * in `m->tensors`. Returns RFDETR_OK if all expected tensors are present,
 * RFDETR_ERR_MODEL_LOAD with a logged error otherwise. */
rfdetr_status model_validate_tensors(const Model& m);

/* Build the list of *expected* tensor names for a given variant config. Used
 * by both `model_validate_tensors` and the test-fixture generator. */
std::vector<std::string> expected_tensor_names(const Config& cfg);

}  // namespace rfdetr

#endif
```

- [ ] **Step 2: Commit**

```bash
git add src/model_loader.hpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(loader): declare model_loader API (Config + Model + load/free/validate)

Header-only declarations. Implementation lands in the next tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: model_loader.cpp — read metadata into Config

TDD: write a unit test that loads a synthesized GGUF and asserts the Config fields. Failure first, then impl.

**Files:**
- Modify: `tests/CMakeLists.txt` (register test_model_loader and the fixture generator)
- Create: `tests/fixtures/gen_model_gguf.cpp` (synthesized minimal GGUF)
- Create: `tests/test_model_loader.cpp`
- Replace placeholder: `src/model_loader.cpp` (created now if not yet)
- Modify: `CMakeLists.txt` (add `src/model_loader.cpp` to RFDETR_SOURCES)

- [ ] **Step 1: Add `src/model_loader.cpp` to RFDETR_SOURCES**

In `CMakeLists.txt`, find the `RFDETR_SOURCES` block and add `src/model_loader.cpp`:

```cmake
set(RFDETR_SOURCES
    src/common.cpp
    src/image_io.cpp
    src/postprocess.cpp
    src/visualize.cpp
    src/cli.cpp
    src/model_loader.cpp
)
```

(Tasks 9/12 add `src/rfdetr.cpp` and `src/rfdetr_capi.cpp` to the same list.)

- [ ] **Step 2: Create placeholder `src/model_loader.cpp`**

```cpp
/* Implemented incrementally in Task 5 and Task 6. */
```

This is to keep the build green between tasks; the real content lands later in this task.

- [ ] **Step 3: Write the synthesized-GGUF fixture generator**

`tests/fixtures/gen_model_gguf.cpp`:

```cpp
/* Synthesizes a minimal rfdetr-base GGUF for unit tests. No PyTorch needed.
 * Writes ALL metadata keys the loader will read, plus the FULL expected
 * tensor set (zero-initialized F16 tensors of the right shapes). Result is
 * a few MB. */

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct VariantCfg {
    const char*   name             = "base";
    uint32_t      image_size       = 560;
    uint32_t      num_queries      = 300;
    uint32_t      num_classes      = 80;
    uint32_t      bb_dim           = 768;
    uint32_t      bb_depth         = 12;
    uint32_t      bb_heads         = 12;
    uint32_t      bb_window        = 14;
    std::vector<int32_t> bb_ms_layers = {2, 5, 8, 11};
    uint32_t      enc_layers       = 3;
    uint32_t      enc_model_dim    = 256;
    uint32_t      enc_ffn_dim      = 2048;
    uint32_t      enc_heads        = 8;
    uint32_t      dec_layers       = 3;
    uint32_t      dec_model_dim    = 256;
    uint32_t      dec_ffn_dim      = 2048;
    uint32_t      dec_heads        = 8;
};

const char* COCO_CLASSES[] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
};
const size_t COCO_N = sizeof(COCO_CLASSES) / sizeof(COCO_CLASSES[0]);

ggml_tensor* make_tensor(ggml_context* ctx, const char* name,
                         ggml_type type, int64_t d0, int64_t d1 = 1,
                         int64_t d2 = 1, int64_t d3 = 1) {
    ggml_tensor* t;
    if (d3 > 1)      t = ggml_new_tensor_4d(ctx, type, d0, d1, d2, d3);
    else if (d2 > 1) t = ggml_new_tensor_3d(ctx, type, d0, d1, d2);
    else if (d1 > 1) t = ggml_new_tensor_2d(ctx, type, d0, d1);
    else             t = ggml_new_tensor_1d(ctx, type, d0);
    ggml_set_name(t, name);
    return t;
}

void add_all_tensors(ggml_context* ctx, std::vector<ggml_tensor*>& out,
                     const VariantCfg& v) {
    const ggml_type F = GGML_TYPE_F16;

    // Backbone
    out.push_back(make_tensor(ctx, "backbone.patch_embed.weight", F, 14, 14, 3, v.bb_dim));
    out.push_back(make_tensor(ctx, "backbone.patch_embed.bias",   F, v.bb_dim));
    // pos_embed: (560/14)^2 + 1 = 1601 tokens for image_size=560, patch=14
    int n_patches = (v.image_size / 14) * (v.image_size / 14);
    out.push_back(make_tensor(ctx, "backbone.pos_embed", F, v.bb_dim, n_patches + 1));
    out.push_back(make_tensor(ctx, "backbone.cls_token", F, v.bb_dim));

    for (uint32_t i = 0; i < v.bb_depth; ++i) {
        std::string p = "backbone.blocks." + std::to_string(i) + ".";
        out.push_back(make_tensor(ctx, (p + "norm1.weight").c_str(), F, v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "norm1.bias").c_str(),   F, v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "attn.qkv.weight").c_str(),  F, v.bb_dim, 3 * v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "attn.qkv.bias").c_str(),    F, 3 * v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "attn.proj.weight").c_str(), F, v.bb_dim, v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "attn.proj.bias").c_str(),   F, v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "norm2.weight").c_str(), F, v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "norm2.bias").c_str(),   F, v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "mlp.fc1.weight").c_str(), F, v.bb_dim, 4 * v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "mlp.fc1.bias").c_str(),   F, 4 * v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "mlp.fc2.weight").c_str(), F, 4 * v.bb_dim, v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "mlp.fc2.bias").c_str(),   F, v.bb_dim));
    }
    out.push_back(make_tensor(ctx, "backbone.norm.weight", F, v.bb_dim));
    out.push_back(make_tensor(ctx, "backbone.norm.bias",   F, v.bb_dim));

    // Projector
    const size_t n_levels = v.bb_ms_layers.size();
    for (size_t j = 0; j < n_levels; ++j) {
        std::string p = "projector.level" + std::to_string(j) + ".";
        out.push_back(make_tensor(ctx, (p + "weight").c_str(), F, v.bb_dim, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "bias").c_str(),   F, v.enc_model_dim));
    }
    out.push_back(make_tensor(ctx, "projector.level_embed", F, v.enc_model_dim, (int64_t)n_levels));

    // Encoder
    for (uint32_t i = 0; i < v.enc_layers; ++i) {
        std::string p = "encoder.layers." + std::to_string(i) + ".";
        out.push_back(make_tensor(ctx, (p + "self_attn.qkv.weight").c_str(), F, v.enc_model_dim, 3 * v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "self_attn.qkv.bias").c_str(),   F, 3 * v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "self_attn.out.weight").c_str(), F, v.enc_model_dim, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "self_attn.out.bias").c_str(),   F, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm1.weight").c_str(), F, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm1.bias").c_str(),   F, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc1.weight").c_str(), F, v.enc_model_dim, v.enc_ffn_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc1.bias").c_str(),   F, v.enc_ffn_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc2.weight").c_str(), F, v.enc_ffn_dim, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc2.bias").c_str(),   F, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm2.weight").c_str(), F, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm2.bias").c_str(),   F, v.enc_model_dim));
    }

    // Decoder
    out.push_back(make_tensor(ctx, "decoder.queries", F, v.dec_model_dim, v.num_queries));
    for (uint32_t i = 0; i < v.dec_layers; ++i) {
        std::string p = "decoder.layers." + std::to_string(i) + ".";
        out.push_back(make_tensor(ctx, (p + "self_attn.qkv.weight").c_str(),  F, v.dec_model_dim, 3 * v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "self_attn.qkv.bias").c_str(),    F, 3 * v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "self_attn.out.weight").c_str(),  F, v.dec_model_dim, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "self_attn.out.bias").c_str(),    F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm1.weight").c_str(), F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm1.bias").c_str(),   F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "cross_attn.q.weight").c_str(),  F, v.dec_model_dim, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "cross_attn.q.bias").c_str(),    F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "cross_attn.kv.weight").c_str(), F, v.dec_model_dim, 2 * v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "cross_attn.kv.bias").c_str(),   F, 2 * v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "cross_attn.out.weight").c_str(),F, v.dec_model_dim, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "cross_attn.out.bias").c_str(),  F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm2.weight").c_str(), F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm2.bias").c_str(),   F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc1.weight").c_str(), F, v.dec_model_dim, v.dec_ffn_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc1.bias").c_str(),   F, v.dec_ffn_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc2.weight").c_str(), F, v.dec_ffn_dim, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc2.bias").c_str(),   F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm3.weight").c_str(), F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm3.bias").c_str(),   F, v.dec_model_dim));
    }

    // Heads
    out.push_back(make_tensor(ctx, "heads.class.fc.weight", F, v.dec_model_dim, v.num_classes));
    out.push_back(make_tensor(ctx, "heads.class.fc.bias",   F, v.num_classes));
    out.push_back(make_tensor(ctx, "heads.bbox.fc1.weight", F, v.dec_model_dim, v.dec_model_dim));
    out.push_back(make_tensor(ctx, "heads.bbox.fc1.bias",   F, v.dec_model_dim));
    out.push_back(make_tensor(ctx, "heads.bbox.fc2.weight", F, v.dec_model_dim, v.dec_model_dim));
    out.push_back(make_tensor(ctx, "heads.bbox.fc2.bias",   F, v.dec_model_dim));
    out.push_back(make_tensor(ctx, "heads.bbox.fc3.weight", F, v.dec_model_dim, 4));
    out.push_back(make_tensor(ctx, "heads.bbox.fc3.bias",   F, 4));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gen_model_gguf <out_path>\n");
        return 1;
    }
    const char* out_path = argv[1];
    VariantCfg v;

    // Estimate context size: rough upper bound = #tensors * 4KB overhead + sum of bytes.
    // Use a generous fixed size for the metadata-only context (no data alloc).
    ggml_init_params iparams{};
    iparams.mem_size   = 256 * 1024 * 1024;  // 256MB scratch
    iparams.mem_buffer = nullptr;
    iparams.no_alloc   = false;  // We allocate F16 placeholder data so gguf writer can read it
    ggml_context* ctx = ggml_init(iparams);
    if (!ctx) { std::fprintf(stderr, "ggml_init failed\n"); return 2; }

    std::vector<ggml_tensor*> tensors;
    tensors.reserve(600);
    add_all_tensors(ctx, tensors, v);

    // Zero-init every tensor's data (already happens because no_alloc=false +
    // ggml mallocs zero memory? — be defensive and memset).
    for (auto* t : tensors) {
        std::memset(t->data, 0, ggml_nbytes(t));
    }

    gguf_context* gguf = gguf_init_empty();
    gguf_set_kv(gguf, ctx);  // not strictly required; metadata is below

    gguf_set_val_str(gguf, "general.architecture",    "rfdetr");
    gguf_set_val_str(gguf, "rfdetr.format.version",   "1");
    gguf_set_val_str(gguf, "rfdetr.variant",          v.name);
    gguf_set_val_u32(gguf, "rfdetr.image_size",       v.image_size);
    gguf_set_val_u32(gguf, "rfdetr.num_queries",      v.num_queries);
    gguf_set_val_u32(gguf, "rfdetr.num_classes",      v.num_classes);
    gguf_set_arr_str(gguf, "rfdetr.class_names",      COCO_CLASSES, COCO_N);

    float mean[3] = {0.485f, 0.456f, 0.406f};
    float stdv[3] = {0.229f, 0.224f, 0.225f};
    gguf_set_arr_data(gguf, "rfdetr.preprocess.mean", GGUF_TYPE_FLOAT32, mean, 3);
    gguf_set_arr_data(gguf, "rfdetr.preprocess.std",  GGUF_TYPE_FLOAT32, stdv, 3);

    gguf_set_val_u32(gguf, "rfdetr.backbone.dim",         v.bb_dim);
    gguf_set_val_u32(gguf, "rfdetr.backbone.depth",       v.bb_depth);
    gguf_set_val_u32(gguf, "rfdetr.backbone.heads",       v.bb_heads);
    gguf_set_val_u32(gguf, "rfdetr.backbone.window_size", v.bb_window);
    gguf_set_arr_data(gguf, "rfdetr.backbone.multi_scale_layers",
                      GGUF_TYPE_INT32, v.bb_ms_layers.data(), v.bb_ms_layers.size());

    gguf_set_val_u32(gguf, "rfdetr.encoder.layers",    v.enc_layers);
    gguf_set_val_u32(gguf, "rfdetr.encoder.model_dim", v.enc_model_dim);
    gguf_set_val_u32(gguf, "rfdetr.encoder.ffn_dim",   v.enc_ffn_dim);
    gguf_set_val_u32(gguf, "rfdetr.encoder.heads",     v.enc_heads);
    gguf_set_val_u32(gguf, "rfdetr.decoder.layers",    v.dec_layers);
    gguf_set_val_u32(gguf, "rfdetr.decoder.model_dim", v.dec_model_dim);
    gguf_set_val_u32(gguf, "rfdetr.decoder.ffn_dim",   v.dec_ffn_dim);
    gguf_set_val_u32(gguf, "rfdetr.decoder.heads",     v.dec_heads);

    for (auto* t : tensors) gguf_add_tensor(gguf, t);

    if (!gguf_write_to_file(gguf, out_path, /*only_meta*/ false)) {
        std::fprintf(stderr, "gguf_write_to_file failed for %s\n", out_path);
        gguf_free(gguf);
        ggml_free(ctx);
        return 3;
    }

    gguf_free(gguf);
    ggml_free(ctx);
    std::printf("wrote %s\n", out_path);
    return 0;
}
```

(Note: ggml's gguf API surface has historically shifted. If the exact
function names above don't match the pinned ggml v0.13.0, look at
`third_party/ggml/include/gguf.h` and `third_party/ggml/examples/gpt-2/main.cpp`
for the current spelling. The structure is right; the function names may need
adjusting. Document the actual calls used in your report.)

- [ ] **Step 4: Update `tests/CMakeLists.txt` to build the generator and register the new test**

Append:

```cmake
# Synthesized model GGUF for tests (no PyTorch needed)
add_executable(gen_model_gguf fixtures/gen_model_gguf.cpp)
target_link_libraries(gen_model_gguf PRIVATE ggml)
add_custom_command(
    OUTPUT  ${RFDETR_TEST_FIXTURES}/model_base.gguf
    COMMAND gen_model_gguf ${RFDETR_TEST_FIXTURES}/model_base.gguf
    DEPENDS gen_model_gguf
    VERBATIM
)
add_custom_target(rfdetr_model_fixture ALL DEPENDS ${RFDETR_TEST_FIXTURES}/model_base.gguf)

rfdetr_add_test(test_model_loader)
add_dependencies(test_model_loader rfdetr_model_fixture)
```

Also add `tests/fixtures/*.gguf` to `.gitignore` (parallel to the `*.png`
rule):

```
tests/fixtures/*.gguf
```

- [ ] **Step 5: Write the failing test `tests/test_model_loader.cpp`**

```cpp
#include "test_assert.hpp"
#include "model_loader.hpp"
#include "rfdetr.h"
#include <string>

int main() {
    const std::string fixtures = RFDETR_TEST_FIXTURES;
    const std::string path = fixtures + "/model_base.gguf";

    rfdetr_status st;
    rfdetr::Model* m = rfdetr::model_load(path, &st);
    RFDETR_ASSERT(m != nullptr);
    RFDETR_ASSERT_EQ_INT(st, RFDETR_OK);

    // Config
    RFDETR_ASSERT_STR_EQ(m->config.variant.c_str(), "base");
    RFDETR_ASSERT_EQ_INT(m->config.image_size,  560);
    RFDETR_ASSERT_EQ_INT(m->config.num_queries, 300);
    RFDETR_ASSERT_EQ_INT(m->config.num_classes, 80);
    RFDETR_ASSERT_EQ_INT(m->config.class_names.size(), 80);
    RFDETR_ASSERT_STR_EQ(m->config.class_names[0].c_str(),  "person");
    RFDETR_ASSERT_STR_EQ(m->config.class_names[79].c_str(), "toothbrush");

    // Preprocess
    RFDETR_ASSERT_NEAR(m->config.preprocess_mean[0], 0.485f, 1e-4);
    RFDETR_ASSERT_NEAR(m->config.preprocess_std[2],  0.225f, 1e-4);

    // Backbone / encoder / decoder
    RFDETR_ASSERT_EQ_INT(m->config.backbone.dim,         768);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.depth,       12);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.heads,       12);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.window_size, 14);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.multi_scale_layers.size(), 4);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.multi_scale_layers[0], 2);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.multi_scale_layers[3], 11);
    RFDETR_ASSERT_EQ_INT(m->config.encoder.layers,    3);
    RFDETR_ASSERT_EQ_INT(m->config.encoder.model_dim, 256);
    RFDETR_ASSERT_EQ_INT(m->config.decoder.heads,     8);

    // Tensors present (sample a few)
    RFDETR_ASSERT(m->tensors.count("backbone.patch_embed.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("backbone.blocks.0.attn.qkv.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("backbone.blocks.11.mlp.fc2.bias") == 1);
    RFDETR_ASSERT(m->tensors.count("decoder.queries") == 1);
    RFDETR_ASSERT(m->tensors.count("heads.class.fc.weight") == 1);

    // Validation passes
    rfdetr_status v = rfdetr::model_validate_tensors(*m);
    RFDETR_ASSERT_EQ_INT(v, RFDETR_OK);

    // Bad path -> error
    rfdetr_status st2;
    rfdetr::Model* bad = rfdetr::model_load("/no/such/path.gguf", &st2);
    RFDETR_ASSERT(bad == nullptr);
    RFDETR_ASSERT(st2 == RFDETR_ERR_FILE_NOT_FOUND || st2 == RFDETR_ERR_MODEL_FORMAT);

    rfdetr::model_free(m);
    return 0;
}
```

- [ ] **Step 6: Run to confirm failure**

```bash
rm -rf build
cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=OFF
cmake --build build --target test_model_loader -j 2>&1 | tail -25
```

Expected: build fails. Either the generator can't build because gguf API
calls need adjusting, or test_model_loader can't link because `rfdetr::model_load`
is undefined.

If the generator fails to build: look at `third_party/ggml/include/gguf.h`
and adjust the API calls in `gen_model_gguf.cpp`. ggml v0.13.0's gguf
writer surface is documented there. STOP and report if you can't make the
generator compile.

- [ ] **Step 7: Implement `src/model_loader.cpp`**

```cpp
#include "model_loader.hpp"
#include "common.hpp"
#include "rfdetr.h"

#include "ggml.h"
#include "gguf.h"

#include <cstring>
#include <fstream>
#include <new>
#include <string>
#include <vector>

namespace rfdetr {

namespace {

const char* kFormatVersion = "1";

bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.is_open();
}

template <typename T>
bool get_u32(gguf_context* g, const char* key, T& out) {
    const int kid = gguf_find_key(g, key);
    if (kid < 0) return false;
    out = (T)gguf_get_val_u32(g, kid);
    return true;
}

bool get_str(gguf_context* g, const char* key, std::string& out) {
    const int kid = gguf_find_key(g, key);
    if (kid < 0) return false;
    out = gguf_get_val_str(g, kid);
    return true;
}

bool get_f32_array(gguf_context* g, const char* key, float* out, size_t n) {
    const int kid = gguf_find_key(g, key);
    if (kid < 0) return false;
    if ((size_t)gguf_get_arr_n(g, kid) != n) return false;
    const float* data = (const float*)gguf_get_arr_data(g, kid);
    std::memcpy(out, data, n * sizeof(float));
    return true;
}

bool get_i32_array(gguf_context* g, const char* key, std::vector<uint32_t>& out) {
    const int kid = gguf_find_key(g, key);
    if (kid < 0) return false;
    size_t n = gguf_get_arr_n(g, kid);
    const int32_t* data = (const int32_t*)gguf_get_arr_data(g, kid);
    out.resize(n);
    for (size_t i = 0; i < n; ++i) out[i] = (uint32_t)data[i];
    return true;
}

bool get_str_array(gguf_context* g, const char* key, std::vector<std::string>& out) {
    const int kid = gguf_find_key(g, key);
    if (kid < 0) return false;
    size_t n = gguf_get_arr_n(g, kid);
    out.clear();
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.emplace_back(gguf_get_arr_str(g, kid, i));
    }
    return true;
}

}  // namespace

Model* model_load(const std::string& path, rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) { if (out_status) *out_status = s; };

    if (!file_exists(path)) {
        rfdetr_logf(RFDETR_LOG_ERROR, "model_load: file not found '%s'", path.c_str());
        set(RFDETR_ERR_FILE_NOT_FOUND);
        return nullptr;
    }

    gguf_init_params gp{};
    gp.no_alloc = true;  // We don't load tensor data in Plan 2
    gp.ctx      = nullptr;

    ggml_context* gctx = nullptr;
    gguf_init_params init_with_ctx{ /* no_alloc */ true, /* ctx */ &gctx };
    gguf_context* gguf = gguf_init_from_file(path.c_str(), init_with_ctx);
    if (!gguf) {
        rfdetr_logf(RFDETR_LOG_ERROR, "model_load: gguf_init_from_file failed for '%s'", path.c_str());
        set(RFDETR_ERR_MODEL_FORMAT);
        return nullptr;
    }

    auto fail = [&](rfdetr_status s, const char* msg) -> Model* {
        rfdetr_logf(RFDETR_LOG_ERROR, "model_load: %s", msg);
        gguf_free(gguf);
        if (gctx) ggml_free(gctx);
        set(s);
        return nullptr;
    };

    // Format version
    std::string fmt;
    if (!get_str(gguf, "rfdetr.format.version", fmt) || fmt != kFormatVersion) {
        return fail(RFDETR_ERR_MODEL_FORMAT, "unsupported rfdetr.format.version");
    }

    Model* m = new (std::nothrow) Model();
    if (!m) return fail(RFDETR_ERR_OUT_OF_MEMORY, "alloc Model");

    m->gguf = gguf;
    m->meta = gctx;

    auto& c = m->config;
    if (!get_str(gguf, "rfdetr.variant",     c.variant))     return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.variant missing");
    if (!get_u32(gguf, "rfdetr.image_size",  c.image_size))  return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.image_size missing");
    if (!get_u32(gguf, "rfdetr.num_queries", c.num_queries)) return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.num_queries missing");
    if (!get_u32(gguf, "rfdetr.num_classes", c.num_classes)) return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.num_classes missing");
    if (!get_str_array(gguf, "rfdetr.class_names", c.class_names))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.class_names missing");
    if (c.class_names.size() != c.num_classes)
        return fail(RFDETR_ERR_MODEL_FORMAT, "class_names length != num_classes");

    if (!get_f32_array(gguf, "rfdetr.preprocess.mean", c.preprocess_mean, 3))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.preprocess.mean missing or wrong shape");
    if (!get_f32_array(gguf, "rfdetr.preprocess.std", c.preprocess_std, 3))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.preprocess.std missing or wrong shape");

    if (!get_u32(gguf, "rfdetr.backbone.dim",         c.backbone.dim)         ||
        !get_u32(gguf, "rfdetr.backbone.depth",       c.backbone.depth)       ||
        !get_u32(gguf, "rfdetr.backbone.heads",       c.backbone.heads)       ||
        !get_u32(gguf, "rfdetr.backbone.window_size", c.backbone.window_size))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.backbone.* incomplete");
    if (!get_i32_array(gguf, "rfdetr.backbone.multi_scale_layers", c.backbone.multi_scale_layers))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.backbone.multi_scale_layers missing");

    if (!get_u32(gguf, "rfdetr.encoder.layers",    c.encoder.layers)    ||
        !get_u32(gguf, "rfdetr.encoder.model_dim", c.encoder.model_dim) ||
        !get_u32(gguf, "rfdetr.encoder.ffn_dim",   c.encoder.ffn_dim)   ||
        !get_u32(gguf, "rfdetr.encoder.heads",     c.encoder.heads))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.encoder.* incomplete");

    if (!get_u32(gguf, "rfdetr.decoder.layers",    c.decoder.layers)    ||
        !get_u32(gguf, "rfdetr.decoder.model_dim", c.decoder.model_dim) ||
        !get_u32(gguf, "rfdetr.decoder.ffn_dim",   c.decoder.ffn_dim)   ||
        !get_u32(gguf, "rfdetr.decoder.heads",     c.decoder.heads))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.decoder.* incomplete");

    // Tensor inventory (descriptors only — data not loaded)
    const int n_tensors = gguf_get_n_tensors(gguf);
    m->tensors.reserve(n_tensors);
    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gguf, i);
        ggml_tensor* t = ggml_get_tensor(gctx, name);
        if (!t) return fail(RFDETR_ERR_MODEL_LOAD, "ggml_get_tensor failed");
        m->tensors.emplace(name, t);
    }

    set(RFDETR_OK);
    return m;
}

void model_free(Model* m) {
    if (!m) return;
    if (m->gguf) gguf_free(m->gguf);
    if (m->meta) ggml_free(m->meta);
    delete m;
}

}  // namespace rfdetr
```

(Task 6 adds `model_validate_tensors` and `expected_tensor_names`.)

- [ ] **Step 8: Add stubs so the test links**

The test calls `rfdetr::model_validate_tensors`. Add a temporary stub at the
bottom of `model_loader.cpp`:

```cpp
namespace rfdetr {

rfdetr_status model_validate_tensors(const Model& /*m*/) {
    /* Filled in by Task 6. */
    return RFDETR_OK;
}

std::vector<std::string> expected_tensor_names(const Config& /*cfg*/) {
    return {};
}

}  // namespace rfdetr
```

- [ ] **Step 9: Build and run**

```bash
cmake --build build --target test_model_loader -j
ctest --test-dir build -R test_model_loader --output-on-failure
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add CMakeLists.txt tests/CMakeLists.txt tests/fixtures/gen_model_gguf.cpp tests/test_model_loader.cpp src/model_loader.cpp .gitignore
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(loader): parse GGUF metadata into Config; tensor inventory; tests

Adds tests/fixtures/gen_model_gguf — a small ggml-based binary that
writes a synthesized rfdetr-base GGUF for unit testing (no PyTorch
needed in CI). model_loader::model_load parses every metadata key per
docs/conversion.md, populates Config, and registers each tensor name -->
ggml_tensor* descriptor (no data load — Plan 3 adds that).
model_validate_tensors and expected_tensor_names are temporary stubs
filled in by the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: model_loader — validate the expected tensor set

TDD: extend `test_model_loader` to assert validation catches *missing* tensors. To set up the failure case we'll write a separate test program that synthesizes an incomplete GGUF and exercises the negative path.

**Files:**
- Modify: `src/model_loader.cpp` (implement `expected_tensor_names` and `model_validate_tensors` properly)
- Modify: `tests/test_model_loader.cpp` (add negative-path assertions)
- Modify: `tests/fixtures/gen_model_gguf.cpp` to accept an optional `--missing <tensor_name>` flag that omits one tensor

- [ ] **Step 1: Extend `gen_model_gguf.cpp` to accept `--missing <name>`**

In `main()`, parse a second argument as the name to skip when adding tensors:

```cpp
int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gen_model_gguf <out_path> [--missing <tensor_name>]\n");
        return 1;
    }
    const char* out_path = argv[1];
    const char* skip_name = nullptr;
    for (int i = 2; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--missing") == 0) {
            skip_name = argv[i + 1];
        }
    }
    // ... rest of body unchanged, but:
    // after add_all_tensors() builds the local `tensors` vector, filter out
    // the one matching skip_name *before* the gguf_add_tensor loop:
    if (skip_name) {
        auto it = std::remove_if(tensors.begin(), tensors.end(),
            [&](ggml_tensor* t) { return std::string(ggml_get_name(t)) == skip_name; });
        tensors.erase(it, tensors.end());
    }
    // ... gguf_add_tensor loop and write_to_file unchanged
}
```

Add `#include <algorithm>` at the top.

- [ ] **Step 2: Update `tests/CMakeLists.txt` to generate a second fixture**

Append (after the existing rfdetr_model_fixture target):

```cmake
add_custom_command(
    OUTPUT  ${RFDETR_TEST_FIXTURES}/model_base_missing.gguf
    COMMAND gen_model_gguf ${RFDETR_TEST_FIXTURES}/model_base_missing.gguf --missing backbone.norm.weight
    DEPENDS gen_model_gguf
    VERBATIM
)
add_custom_target(rfdetr_model_fixture_missing ALL DEPENDS ${RFDETR_TEST_FIXTURES}/model_base_missing.gguf)
add_dependencies(test_model_loader rfdetr_model_fixture_missing)
```

- [ ] **Step 3: Extend `test_model_loader.cpp`**

Add before `return 0;`:

```cpp
    // ---- Missing tensor -> validation fails ----
    {
        std::string bad_path = fixtures + "/model_base_missing.gguf";
        rfdetr_status st_;
        rfdetr::Model* bm = rfdetr::model_load(bad_path, &st_);
        RFDETR_ASSERT(bm != nullptr);
        RFDETR_ASSERT_EQ_INT(st_, RFDETR_OK);

        rfdetr_status v = rfdetr::model_validate_tensors(*bm);
        RFDETR_ASSERT_EQ_INT(v, RFDETR_ERR_MODEL_LOAD);

        rfdetr::model_free(bm);
    }

    // ---- expected_tensor_names produces the right count for base ----
    {
        rfdetr_status st_;
        rfdetr::Model* good = rfdetr::model_load(path, &st_);
        std::vector<std::string> expected = rfdetr::expected_tensor_names(good->config);
        // Loader counted all tensor names; expected list should match.
        RFDETR_ASSERT_EQ_INT(expected.size(), good->tensors.size());
        for (const auto& n : expected) {
            RFDETR_ASSERT(good->tensors.count(n) == 1);
        }
        rfdetr::model_free(good);
    }
```

- [ ] **Step 4: Run to confirm failure**

```bash
cmake --build build --target test_model_loader -j
ctest --test-dir build -R test_model_loader --output-on-failure
```

Expected: failure. With the stub `model_validate_tensors` always returning
`RFDETR_OK`, the missing-tensor assertion fails. The `expected_tensor_names`
stub returns an empty list, which won't equal `good->tensors.size()` → also
fails.

- [ ] **Step 5: Implement `expected_tensor_names` and `model_validate_tensors`**

Replace the stubs at the bottom of `src/model_loader.cpp` with:

```cpp
namespace rfdetr {

std::vector<std::string> expected_tensor_names(const Config& cfg) {
    std::vector<std::string> names;

    // Backbone
    names.emplace_back("backbone.patch_embed.weight");
    names.emplace_back("backbone.patch_embed.bias");
    names.emplace_back("backbone.pos_embed");
    names.emplace_back("backbone.cls_token");
    for (uint32_t i = 0; i < cfg.backbone.depth; ++i) {
        std::string p = "backbone.blocks." + std::to_string(i) + ".";
        names.emplace_back(p + "norm1.weight");
        names.emplace_back(p + "norm1.bias");
        names.emplace_back(p + "attn.qkv.weight");
        names.emplace_back(p + "attn.qkv.bias");
        names.emplace_back(p + "attn.proj.weight");
        names.emplace_back(p + "attn.proj.bias");
        names.emplace_back(p + "norm2.weight");
        names.emplace_back(p + "norm2.bias");
        names.emplace_back(p + "mlp.fc1.weight");
        names.emplace_back(p + "mlp.fc1.bias");
        names.emplace_back(p + "mlp.fc2.weight");
        names.emplace_back(p + "mlp.fc2.bias");
    }
    names.emplace_back("backbone.norm.weight");
    names.emplace_back("backbone.norm.bias");

    // Projector
    for (size_t j = 0; j < cfg.backbone.multi_scale_layers.size(); ++j) {
        std::string p = "projector.level" + std::to_string(j) + ".";
        names.emplace_back(p + "weight");
        names.emplace_back(p + "bias");
    }
    names.emplace_back("projector.level_embed");

    // Encoder
    for (uint32_t i = 0; i < cfg.encoder.layers; ++i) {
        std::string p = "encoder.layers." + std::to_string(i) + ".";
        names.emplace_back(p + "self_attn.qkv.weight");
        names.emplace_back(p + "self_attn.qkv.bias");
        names.emplace_back(p + "self_attn.out.weight");
        names.emplace_back(p + "self_attn.out.bias");
        names.emplace_back(p + "norm1.weight");
        names.emplace_back(p + "norm1.bias");
        names.emplace_back(p + "ffn.fc1.weight");
        names.emplace_back(p + "ffn.fc1.bias");
        names.emplace_back(p + "ffn.fc2.weight");
        names.emplace_back(p + "ffn.fc2.bias");
        names.emplace_back(p + "norm2.weight");
        names.emplace_back(p + "norm2.bias");
    }

    // Decoder
    names.emplace_back("decoder.queries");
    for (uint32_t i = 0; i < cfg.decoder.layers; ++i) {
        std::string p = "decoder.layers." + std::to_string(i) + ".";
        names.emplace_back(p + "self_attn.qkv.weight");
        names.emplace_back(p + "self_attn.qkv.bias");
        names.emplace_back(p + "self_attn.out.weight");
        names.emplace_back(p + "self_attn.out.bias");
        names.emplace_back(p + "norm1.weight");
        names.emplace_back(p + "norm1.bias");
        names.emplace_back(p + "cross_attn.q.weight");
        names.emplace_back(p + "cross_attn.q.bias");
        names.emplace_back(p + "cross_attn.kv.weight");
        names.emplace_back(p + "cross_attn.kv.bias");
        names.emplace_back(p + "cross_attn.out.weight");
        names.emplace_back(p + "cross_attn.out.bias");
        names.emplace_back(p + "norm2.weight");
        names.emplace_back(p + "norm2.bias");
        names.emplace_back(p + "ffn.fc1.weight");
        names.emplace_back(p + "ffn.fc1.bias");
        names.emplace_back(p + "ffn.fc2.weight");
        names.emplace_back(p + "ffn.fc2.bias");
        names.emplace_back(p + "norm3.weight");
        names.emplace_back(p + "norm3.bias");
    }

    // Heads
    names.emplace_back("heads.class.fc.weight");
    names.emplace_back("heads.class.fc.bias");
    names.emplace_back("heads.bbox.fc1.weight");
    names.emplace_back("heads.bbox.fc1.bias");
    names.emplace_back("heads.bbox.fc2.weight");
    names.emplace_back("heads.bbox.fc2.bias");
    names.emplace_back("heads.bbox.fc3.weight");
    names.emplace_back("heads.bbox.fc3.bias");

    return names;
}

rfdetr_status model_validate_tensors(const Model& m) {
    const auto expected = expected_tensor_names(m.config);
    std::vector<std::string> missing;
    for (const auto& n : expected) {
        if (m.tensors.find(n) == m.tensors.end()) {
            missing.push_back(n);
        }
    }
    if (!missing.empty()) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "model_validate_tensors: %zu missing tensor(s); first: '%s'",
                    missing.size(), missing.front().c_str());
        return RFDETR_ERR_MODEL_LOAD;
    }
    return RFDETR_OK;
}

}  // namespace rfdetr
```

- [ ] **Step 6: Run to confirm pass**

```bash
cmake --build build --target test_model_loader -j
ctest --test-dir build -R test_model_loader --output-on-failure
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/gen_model_gguf.cpp tests/CMakeLists.txt tests/test_model_loader.cpp src/model_loader.cpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(loader): expected_tensor_names + model_validate_tensors

expected_tensor_names builds the variant-driven list of required tensor
names (backbone, projector, encoder, decoder, heads). model_validate_tensors
returns RFDETR_ERR_MODEL_LOAD if any are missing, logging the first one.
Test fixture extended with --missing to generate a deliberately-broken GGUF
for the negative-path assertion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: rfdetr.cpp — opaque-pointer API (rfdetr_init, rfdetr_free)

**Files:**
- Create: `src/rfdetr.cpp`
- Modify: `CMakeLists.txt` (add to RFDETR_SOURCES)
- Create: `tests/test_capi.cpp`
- Modify: `tests/CMakeLists.txt` (register test_capi)

- [ ] **Step 1: Add `src/rfdetr.cpp` to `RFDETR_SOURCES`**

In `CMakeLists.txt`:

```cmake
set(RFDETR_SOURCES
    src/common.cpp
    src/image_io.cpp
    src/postprocess.cpp
    src/visualize.cpp
    src/cli.cpp
    src/model_loader.cpp
    src/rfdetr.cpp
)
```

- [ ] **Step 2: Write the failing test `tests/test_capi.cpp`**

```cpp
#include "test_assert.hpp"
#include "rfdetr.h"
#include <string>

int main() {
    const std::string fixtures = RFDETR_TEST_FIXTURES;
    const std::string model_path = fixtures + "/model_base.gguf";

    // Successful init
    rfdetr_params p{};
    p.model_path = model_path.c_str();
    p.n_threads  = 1;

    rfdetr_status st;
    rfdetr_context* ctx = rfdetr_init(&p, &st);
    RFDETR_ASSERT(ctx != nullptr);
    RFDETR_ASSERT_EQ_INT(st, RFDETR_OK);

    rfdetr_free(ctx);

    // Bad path -> nullptr, status set
    rfdetr_params p2{};
    p2.model_path = "/no/such/model.gguf";
    p2.n_threads  = 1;
    rfdetr_status st2;
    rfdetr_context* ctx2 = rfdetr_init(&p2, &st2);
    RFDETR_ASSERT(ctx2 == nullptr);
    RFDETR_ASSERT(st2 != RFDETR_OK);

    // Null params -> nullptr, RFDETR_ERR_INVALID_ARG
    rfdetr_status st3;
    rfdetr_context* ctx3 = rfdetr_init(nullptr, &st3);
    RFDETR_ASSERT(ctx3 == nullptr);
    RFDETR_ASSERT_EQ_INT(st3, RFDETR_ERR_INVALID_ARG);

    // rfdetr_free(nullptr) must not crash
    rfdetr_free(nullptr);

    // rfdetr_detect still returns NOT_IMPLEMENTED (Plan 3 wires inference)
    rfdetr_context* ctx4 = rfdetr_init(&p, &st);
    rfdetr_image*   img  = rfdetr_image_load_file((fixtures + "/cats.png").c_str(), nullptr);
    RFDETR_ASSERT(ctx4 && img);
    rfdetr_detect_params dp{};
    dp.threshold = 0.5f; dp.top_k = 300;
    rfdetr_detection* dets = nullptr;
    size_t n = 0;
    rfdetr_status det_st = rfdetr_detect(ctx4, img, &dp, &dets, &n);
    RFDETR_ASSERT_EQ_INT(det_st, RFDETR_ERR_NOT_IMPLEMENTED);
    RFDETR_ASSERT_EQ_INT(n, 0);
    RFDETR_ASSERT(dets == nullptr);

    rfdetr_image_free(img);
    rfdetr_free(ctx4);
    return 0;
}
```

- [ ] **Step 3: Register `test_capi` in `tests/CMakeLists.txt`**

Append:

```cmake
rfdetr_add_test(test_capi)
add_dependencies(test_capi rfdetr_model_fixture rfdetr_fixtures)
```

- [ ] **Step 4: Confirm failure**

```bash
cmake -B build -DRFDETR_BUILD_TESTS=ON
cmake --build build --target test_capi -j 2>&1 | tail -15
```

Expected: link errors — `rfdetr_init`, `rfdetr_free`, `rfdetr_detect` undefined.

- [ ] **Step 5: Write `src/rfdetr.cpp`**

```cpp
#include "rfdetr.h"
#include "common.hpp"
#include "model_loader.hpp"

#include <new>
#include <string>

/* Opaque struct — defined here so external callers can only hold pointers. */
struct rfdetr_context {
    rfdetr::Model* model = nullptr;
    int            n_threads = 1;
};

extern "C" rfdetr_context* rfdetr_init(const rfdetr_params* params, rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) { if (out_status) *out_status = s; };

    if (!params || !params->model_path) {
        set(RFDETR_ERR_INVALID_ARG);
        return nullptr;
    }

    /* The params struct also carries a logging callback; install it now so
     * the model_load errors are surfaced through the caller's channel. */
    if (params->log_cb) {
        rfdetr_set_log_callback(params->log_cb, params->log_user_data);
    }

    rfdetr_status load_st;
    rfdetr::Model* m = rfdetr::model_load(params->model_path, &load_st);
    if (!m) {
        set(load_st);
        return nullptr;
    }

    rfdetr_status v = rfdetr::model_validate_tensors(*m);
    if (v != RFDETR_OK) {
        rfdetr::model_free(m);
        set(v);
        return nullptr;
    }

    auto* ctx = new (std::nothrow) rfdetr_context();
    if (!ctx) {
        rfdetr::model_free(m);
        set(RFDETR_ERR_OUT_OF_MEMORY);
        return nullptr;
    }
    ctx->model     = m;
    ctx->n_threads = params->n_threads > 0 ? params->n_threads : 1;

    rfdetr_logf(RFDETR_LOG_INFO, "rfdetr_init: loaded variant=%s, num_classes=%u, num_queries=%u",
                m->config.variant.c_str(),
                m->config.num_classes,
                m->config.num_queries);

    set(RFDETR_OK);
    return ctx;
}

extern "C" void rfdetr_free(rfdetr_context* ctx) {
    if (!ctx) return;
    rfdetr::model_free(ctx->model);
    delete ctx;
}

extern "C" rfdetr_status rfdetr_detect(rfdetr_context* ctx,
                                       const rfdetr_image* img,
                                       const rfdetr_detect_params* /*params*/,
                                       rfdetr_detection** out_detections,
                                       size_t* out_n) {
    if (out_detections) *out_detections = nullptr;
    if (out_n)          *out_n = 0;
    if (!ctx || !img) return RFDETR_ERR_INVALID_ARG;

    /* Plan 3 wires the forward graph. For now: signal that the model loaded
     * successfully but inference isn't implemented. */
    return RFDETR_ERR_NOT_IMPLEMENTED;
}

/* Accessors used by `info` subcommand. */
extern "C" {

const char* rfdetr_context_variant(const rfdetr_context* ctx) {
    return (ctx && ctx->model) ? ctx->model->config.variant.c_str() : "";
}
uint32_t rfdetr_context_image_size(const rfdetr_context* ctx) {
    return ctx ? ctx->model->config.image_size : 0;
}
uint32_t rfdetr_context_num_queries(const rfdetr_context* ctx) {
    return ctx ? ctx->model->config.num_queries : 0;
}
uint32_t rfdetr_context_num_classes(const rfdetr_context* ctx) {
    return ctx ? ctx->model->config.num_classes : 0;
}
size_t rfdetr_context_n_tensors(const rfdetr_context* ctx) {
    return ctx ? ctx->model->tensors.size() : 0;
}

}  // extern "C"
```

- [ ] **Step 6: Build and run**

```bash
cmake --build build --target test_capi -j
ctest --test-dir build -R test_capi --output-on-failure
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt src/rfdetr.cpp tests/test_capi.cpp tests/CMakeLists.txt
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(api): implement rfdetr_init / rfdetr_free + info-accessor functions

rfdetr_init now opens a GGUF, validates the tensor set, and constructs an
rfdetr_context owning the loaded Model. rfdetr_detect remains
NOT_IMPLEMENTED until Plan 3 wires the forward graph. Five small
context accessors (variant, image_size, num_queries, num_classes,
n_tensors) feed the info subcommand and the flat C ABI.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: `info` subcommand wiring + integration test

**Files:**
- Modify: `examples/cli/main.cpp` (implement `cmd_info`)
- Modify: `include/rfdetr.h` (declare the new accessor functions added in Task 7)
- Modify: `tests/test_cli_integration.cpp` (add a block that runs `info` on the fixture)

- [ ] **Step 1: Declare the accessors in `include/rfdetr.h`**

After the existing `rfdetr_set_log_callback` line, add:

```c
/* Accessors for loaded context (for `info`-style introspection). */
const char* rfdetr_context_variant(const rfdetr_context* ctx);
uint32_t    rfdetr_context_image_size(const rfdetr_context* ctx);
uint32_t    rfdetr_context_num_queries(const rfdetr_context* ctx);
uint32_t    rfdetr_context_num_classes(const rfdetr_context* ctx);
size_t      rfdetr_context_n_tensors(const rfdetr_context* ctx);
```

(Place inside the existing `extern "C"` block.)

- [ ] **Step 2: Implement `cmd_info` in `examples/cli/main.cpp`**

Add this function above `main()`:

```cpp
static int cmd_info(const rfdetr_cli::InfoArgs& a) {
    rfdetr_params p{};
    p.model_path = a.model.c_str();
    p.n_threads  = 1;

    rfdetr_status st;
    rfdetr_context* ctx = rfdetr_init(&p, &st);
    if (!ctx) {
        std::fprintf(stderr, "rfdetr_init failed: %s\n", rfdetr_status_str(st));
        return 2;
    }

    std::printf("variant:      %s\n", rfdetr_context_variant(ctx));
    std::printf("image_size:   %u\n", rfdetr_context_image_size(ctx));
    std::printf("num_classes:  %u\n", rfdetr_context_num_classes(ctx));
    std::printf("num_queries:  %u\n", rfdetr_context_num_queries(ctx));
    std::printf("n_tensors:    %zu\n", rfdetr_context_n_tensors(ctx));

    rfdetr_free(ctx);
    return 0;
}
```

And in `main()`'s switch, replace the `Info` arm:

```cpp
        case rfdetr_cli::Subcommand::Info:
            return cmd_info(r.info);
```

(Leave Bench and Compare in the "not implemented" arm.)

- [ ] **Step 3: Extend `tests/test_cli_integration.cpp`**

Add before `return 0;`:

```cpp
    // ---- info on synthesized model ----
    {
        std::string cmd2 = std::string(RFDETR_CLI_BINARY) +
                           " info --model " + fixtures + "/model_base.gguf > " +
                           fixtures + "/generated/info_out.txt 2>&1";
        int rc2 = std::system(cmd2.c_str());
        RFDETR_ASSERT_EQ_INT(WEXITSTATUS(rc2), 0);

        std::string info_body = read_file(fixtures + "/generated/info_out.txt");
        RFDETR_ASSERT(info_body.find("variant:      base")   != std::string::npos);
        RFDETR_ASSERT(info_body.find("image_size:   560")    != std::string::npos);
        RFDETR_ASSERT(info_body.find("num_classes:  80")     != std::string::npos);
        RFDETR_ASSERT(info_body.find("num_queries:  300")    != std::string::npos);
    }
```

- [ ] **Step 4: Make `test_cli_integration` depend on the model fixture**

In `tests/CMakeLists.txt`, find the `if(TARGET rfdetr-cli) ... endif()` block
from Plan 1 and add a dependency on `rfdetr_model_fixture`:

```cmake
if(TARGET rfdetr-cli)
    rfdetr_add_test(test_cli_integration)
    target_compile_definitions(test_cli_integration PRIVATE
        RFDETR_CLI_BINARY="$<TARGET_FILE:rfdetr-cli>")
    add_dependencies(test_cli_integration rfdetr-cli rfdetr_model_fixture)
endif()
```

- [ ] **Step 5: Build and run**

```bash
rm -rf build
cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: all tests pass, including the info integration check.

- [ ] **Step 6: Commit**

```bash
git add include/rfdetr.h examples/cli/main.cpp tests/test_cli_integration.cpp tests/CMakeLists.txt
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(cli): implement info subcommand; integration test on fixture model

rfdetr-cli info loads a GGUF, prints variant/image_size/num_classes/
num_queries/n_tensors, and exits 0. The integration test exercises it
against the synthesized base fixture.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Flat C ABI implementation (`rfdetr_capi_*`)

**Files:**
- Create: `src/rfdetr_capi.cpp`
- Modify: `CMakeLists.txt` (add to RFDETR_SOURCES)
- Create: `tests/test_capi_flat.cpp`
- Modify: `tests/CMakeLists.txt` (register test_capi_flat)

- [ ] **Step 1: Add `src/rfdetr_capi.cpp` to `RFDETR_SOURCES`**

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
)
```

- [ ] **Step 2: Write the failing test `tests/test_capi_flat.cpp`**

```cpp
#include "test_assert.hpp"
#include "rfdetr_capi.h"
#include <cstdlib>
#include <cstring>
#include <string>

int main() {
    const std::string fixtures = RFDETR_TEST_FIXTURES;
    const std::string model_path = fixtures + "/model_base.gguf";

    // Load
    rfdetr_handle_t h = 0;
    int rc = rfdetr_capi_load(model_path.c_str(), 1, &h);
    RFDETR_ASSERT_EQ_INT(rc, 0);
    RFDETR_ASSERT(h != 0);

    // Detect — returns JSON envelope, even with empty detections (NOT_IMPLEMENTED
    // inference path). The flat ABI must convert the C-side status to a JSON
    // response: status field plus the empty detections array.
    char* json = nullptr;
    int dr = rfdetr_capi_detect_path(h, (fixtures + "/cats.png").c_str(),
                                     0.5f, 300, &json);
    RFDETR_ASSERT_EQ_INT(dr, 0);   /* 0 = handled (even if NOT_IMPLEMENTED internally) */
    RFDETR_ASSERT(json != nullptr);
    std::string body(json);
    rfdetr_capi_free_string(json);

    RFDETR_ASSERT(body.find("\"status\"")     != std::string::npos);
    RFDETR_ASSERT(body.find("\"detections\"") != std::string::npos);
    RFDETR_ASSERT(body.find("\"width\"")      != std::string::npos);
    RFDETR_ASSERT(body.find("\"height\"")     != std::string::npos);

    // Bad image path -> non-zero rc
    char* json2 = nullptr;
    int dr2 = rfdetr_capi_detect_path(h, "/no/such/image.png", 0.5f, 300, &json2);
    RFDETR_ASSERT(dr2 != 0);
    if (json2) rfdetr_capi_free_string(json2);

    // Unload
    int ur = rfdetr_capi_unload(h);
    RFDETR_ASSERT_EQ_INT(ur, 0);

    // Re-unload zero handle is a no-op (0)
    int ur2 = rfdetr_capi_unload(0);
    RFDETR_ASSERT_EQ_INT(ur2, 0);

    // Bad path on load
    rfdetr_handle_t h2 = 0;
    int br = rfdetr_capi_load("/no/such/file.gguf", 1, &h2);
    RFDETR_ASSERT(br != 0);
    RFDETR_ASSERT(h2 == 0);

    return 0;
}
```

- [ ] **Step 3: Register `test_capi_flat` in `tests/CMakeLists.txt`**

Append:

```cmake
rfdetr_add_test(test_capi_flat)
add_dependencies(test_capi_flat rfdetr_model_fixture rfdetr_fixtures)
```

- [ ] **Step 4: Confirm failure**

```bash
cmake --build build --target test_capi_flat -j 2>&1 | tail -15
```

Expected: link errors — `rfdetr_capi_load`, `rfdetr_capi_detect_path`,
`rfdetr_capi_unload`, `rfdetr_capi_free_string` undefined.

- [ ] **Step 5: Write `src/rfdetr_capi.cpp`**

```cpp
#include "rfdetr_capi.h"
#include "rfdetr.h"
#include "common.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

namespace {

/* Serialize an rfdetr_context plus the result of a detect attempt into a
 * JSON envelope. The envelope always includes a `status` field so callers
 * can distinguish "model loaded but inference NYI" from "decode failure". */
std::string make_json(int img_w, int img_h, const char* status,
                      const rfdetr_detection* dets, size_t n) {
    std::ostringstream o;
    o << "{\"status\":\"" << status << "\","
      << "\"image\":{\"width\":" << img_w << ",\"height\":" << img_h << "},"
      << "\"detections\":[";
    for (size_t i = 0; i < n; ++i) {
        if (i) o << ",";
        o << "{\"class_id\":" << dets[i].class_id
          << ",\"score\":" << dets[i].score
          << ",\"x1\":" << dets[i].x1
          << ",\"y1\":" << dets[i].y1
          << ",\"x2\":" << dets[i].x2
          << ",\"y2\":" << dets[i].y2
          << "}";
    }
    o << "]}";
    return o.str();
}

char* dup_to_c(const std::string& s) {
    char* buf = (char*)std::malloc(s.size() + 1);
    if (!buf) return nullptr;
    std::memcpy(buf, s.data(), s.size());
    buf[s.size()] = '\0';
    return buf;
}

}  // namespace

extern "C" int rfdetr_capi_load(const char* model_path, int n_threads, rfdetr_handle_t* out_handle) {
    if (!model_path || !out_handle) return -1;
    *out_handle = 0;

    rfdetr_params p{};
    p.model_path = model_path;
    p.n_threads  = n_threads > 0 ? n_threads : 1;

    rfdetr_status st;
    rfdetr_context* ctx = rfdetr_init(&p, &st);
    if (!ctx) return (int)st;

    *out_handle = (rfdetr_handle_t)ctx;
    return 0;
}

extern "C" int rfdetr_capi_unload(rfdetr_handle_t handle) {
    if (handle == 0) return 0;
    rfdetr_free((rfdetr_context*)handle);
    return 0;
}

static int capi_detect_common(rfdetr_handle_t handle, rfdetr_image* img,
                              float threshold, uint32_t top_k, char** out_json) {
    auto* ctx = (rfdetr_context*)handle;
    if (!ctx || !img || !out_json) return -1;
    *out_json = nullptr;

    rfdetr_detect_params dp{};
    dp.threshold = threshold;
    dp.top_k     = top_k;

    rfdetr_detection* dets = nullptr;
    size_t n = 0;
    rfdetr_status st = rfdetr_detect(ctx, img, &dp, &dets, &n);

    const char* status = (st == RFDETR_OK)                  ? "ok"
                       : (st == RFDETR_ERR_NOT_IMPLEMENTED) ? "not_implemented"
                       :                                       rfdetr_status_str(st);

    std::string json = make_json(rfdetr_image_width(img), rfdetr_image_height(img),
                                 status, dets, n);
    rfdetr_detections_free(dets, n);

    *out_json = dup_to_c(json);
    if (!*out_json) return RFDETR_ERR_OUT_OF_MEMORY;
    return 0;
}

extern "C" int rfdetr_capi_detect_path(rfdetr_handle_t handle, const char* image_path,
                                       float threshold, uint32_t top_k, char** out_json) {
    if (!handle || !image_path || !out_json) return -1;
    *out_json = nullptr;

    rfdetr_status load_st;
    rfdetr_image* img = rfdetr_image_load_file(image_path, &load_st);
    if (!img) return (int)load_st;

    int rc = capi_detect_common(handle, img, threshold, top_k, out_json);
    rfdetr_image_free(img);
    return rc;
}

extern "C" int rfdetr_capi_detect_buffer(rfdetr_handle_t handle,
                                         const uint8_t* bytes, size_t len,
                                         float threshold, uint32_t top_k,
                                         char** out_json) {
    if (!handle || !bytes || len == 0 || !out_json) return -1;
    *out_json = nullptr;

    rfdetr_status load_st;
    rfdetr_image* img = rfdetr_image_load_buffer(bytes, len, &load_st);
    if (!img) return (int)load_st;

    int rc = capi_detect_common(handle, img, threshold, top_k, out_json);
    rfdetr_image_free(img);
    return rc;
}

extern "C" void rfdetr_capi_free_string(char* s) {
    std::free(s);
}
```

- [ ] **Step 6: Build and run**

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: all tests pass, including `test_capi_flat`.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt src/rfdetr_capi.cpp tests/test_capi_flat.cpp tests/CMakeLists.txt
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(capi): implement flat dlopen ABI (load / detect_path / detect_buffer / unload)

Thin layer over the opaque-pointer API. JSON envelope always includes a
status field so the (current) NOT_IMPLEMENTED detect path is
distinguishable from real errors when consumed by purego / dlopen.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Final clean rebuild + README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Clean rebuild + full test run**

```bash
rm -rf build
cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=ON
cmake --build build -j 2>&1 | tail -20
ctest --test-dir build --output-on-failure
```

Expected: all 9 tests pass (`test_common`, `test_image_io`, `test_postprocess`,
`test_visualize`, `test_cli_smoke`, `test_cli_integration`, `test_model_loader`,
`test_capi`, `test_capi_flat`).

Capture this output for the report.

- [ ] **Step 2: Verify configure works for the three combos**

```bash
# tests off, cli off — library-only build (e.g. for purego embedding)
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=OFF -DRFDETR_BUILD_CLI=OFF 2>&1 | tail -5
cmake --build build -j 2>&1 | tail -5

# tests on, cli off — same as above (no rfdetr-cli) but with tests
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=OFF 2>&1 | tail -5
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -3

# both on
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=ON
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -3
```

Expected: all configurations succeed; tests pass in both `tests=ON` builds.

- [ ] **Step 3: Update `README.md` Status section**

Replace the existing Status paragraph with:

```markdown
## Status

**Loader (Plan 2) complete.** The repo can convert an upstream `rfdetr-base`
PyTorch checkpoint to GGUF, load it from C++ (variant detection, config
parsing, tensor inventory validation), and report it via `rfdetr-cli info`.
The flat dlopen ABI is wired through to a JSON envelope. Nine tests pass on
a clean build. Inference (`detect`) still returns `not_implemented` —
Plan 3 builds the forward graph and parity harness.
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
docs: mark loader plan (Plan 2) complete in README

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage** against `docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md`:

- §3 Architecture (forward graph) — N/A for Plan 2 (Plan 3).
- §4 Project layout — Plan 2 adds `docs/conversion.md`, `docs/variants.md`,
  `scripts/`, `src/model_loader.{cpp,hpp}`, `src/rfdetr.cpp`, `src/rfdetr_capi.cpp`.
  `src/dinov2`, `src/projector`, `src/encoder`, `src/decoder`, `src/heads`, `src/rfdetr_model`,
  `src/trace`, `src/postprocess` (extra ops) are Plan 3.
- §5 Weight format and conversion — fully covered: format version, metadata
  keys, tensor naming, conversion script for `base`.
- §6 API — `rfdetr_init`/`rfdetr_free` implemented; `rfdetr_detect` stub
  returns NOT_IMPLEMENTED (Plan 3 wires it); `rfdetr_image_*` and
  `rfdetr_render` unchanged from Plan 1; `rfdetr_capi_*` all implemented.
- §7 CLI — `info` now works; `detect` runs (returns 0, NOT_IMPLEMENTED-style
  empty JSON); `bench` and `compare` still return 99 — Plan 3.
- §8 Tests — Plan 2 adds `test_model_loader`, `test_capi`, `test_capi_flat`.
- §9 Parity workflow — Plan 3.
- §10 Build — `rfdetr.cpp`, `rfdetr_capi.cpp`, `model_loader.cpp` added to
  RFDETR_SOURCES.
- §11 Dependencies — Python deps pinned in `scripts/requirements.txt`. No
  new C++ deps.

**Placeholder scan:** all `Implemented in a later task` strings refer to
later tasks **within this plan** (Task 5's stubs for `model_validate_tensors`/
`expected_tensor_names` are filled in by Task 6, then committed in Task 6).
No "TBD"/"fill in later"/"TODO" without an explicit follow-up task.

**Type consistency:**
- `rfdetr::Config` vs `rfdetr_params` — these are distinct; `Config` is the
  loaded-model config (internal), `rfdetr_params` is the init-time public
  param struct. Names don't conflict.
- `rfdetr_status` is the public C enum; `model_load` / `model_validate_tensors`
  return it directly. ✓
- `rfdetr::Model::tensors` is `std::unordered_map<std::string, ggml_tensor*>`.
  `expected_tensor_names` returns `std::vector<std::string>`. Both refer to
  the same tensor-name strings declared in `docs/conversion.md`. ✓
- The five context accessors are declared in `rfdetr.h` (Task 8 Step 1)
  *after* they're defined in `rfdetr.cpp` (Task 7). The declarations are
  needed before the CLI can use them — Task 8 Step 1 lands first. ✓

**Risk areas worth flagging to the executor:**

1. **ggml gguf API drift**: ggml v0.13.0 might use slightly different
   function names than the ones in Task 5 Step 3 (`gguf_set_val_u32`, etc.).
   The implementer must check `third_party/ggml/include/gguf.h` and adapt.
   The structure is correct; only the spelling may differ.

2. **rfdetr PyPI access path**: Task 3 Step 2 discovers whether
   `m.model.state_dict()` is the right access path. If rfdetr exposes the
   torch model differently, the discovery step finds the right one.

3. **Conversion script CI exclusion**: Plan 2 deliberately *does not*
   require Python to run tests. The synthesized GGUF fixture is the test
   surface. A developer-host conversion only matters at Plan 3 parity time.

If issues #1-2 prevent progress, the executor should report BLOCKED with the
specific symbol or attribute that's missing — not work around it silently.

---

## Next plan

After this plan lands:

- **Plan 3** — DINOv2 backbone, projector, encoder, decoder, heads,
  `rfdetr_model.cpp` graph build, trace callback,
  `scripts/run_rfdetr_baseline.py`, `test_parity` against the `base` variant.
- **Plan 2.5 (optional polish)** — font embedding header generator + label
  rendering in `visualize.cpp` (deferred from Plan 2).
- **Plan 4** — `rfdetr-quantize` binary, small/nano/medium/large bring-up.
