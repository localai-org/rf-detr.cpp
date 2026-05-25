#!/usr/bin/env python3
"""Convert an upstream rfdetr PyTorch checkpoint to GGUF.

Usage:
    python scripts/convert_rfdetr_to_gguf.py --variant base \
        --output rfdetr-base-f16.gguf [--dtype f16|f32] [--dry-run]

Format version: "1" (see docs/conversion.md).

Status
------
**BLOCKED** at the upstream-architecture probe step. The actual rfdetr-base
state_dict (rfdetr 1.7.0) does not match the schema this project's C++ runtime
was designed against. The conversion cannot be completed without redesigning
the C++ runtime to match the real model. See `BLOCKED` section below for the
diff between expected and actual.

What this script does today
---------------------------
* Loads `RFDETRBase()` (downloads ~355 MB to ~/.roboflow on first run).
* Extracts the underlying nn.Module via `m.model.model` (type `LWDETR`).
* Probes its state_dict (487 entries) and prints a structural mismatch report
  against the C++ runtime's expected tensor set.
* Returns exit code 90 (blocked, architectural mismatch).
* `--dry-run` is honored: it prints the same report and returns 90.

To complete the conversion, the project needs to either:

  (A) Re-implement the C++ runtime (`src/dinov2.cpp`, `src/encoder.cpp`,
      `src/decoder.cpp`, `src/projector.cpp`, `src/heads.cpp`,
      `src/model_loader.cpp`, `tests/fixtures/gen_model_gguf.cpp`,
      `docs/conversion.md`) to match the real rfdetr architecture, then
      finish this script's `build_tensor_name_map` and `write_gguf`.

  (B) Adopt a different upstream that DOES match the current C++ runtime
      schema (would need to find one — the schema seems aspirational rather
      than tied to a specific public release).

The first ~30 lines of `build_tensor_name_map` below capture what *can* be
mapped (a handful of backbone names) for reference when (A) is pursued.

BLOCKED — structural diffs at a glance
--------------------------------------
| concept                  | runtime expects                              | rfdetr-base actual                                      |
| ------------------------ | -------------------------------------------- | ------------------------------------------------------- |
| backbone dim             | 768                                          | 384 (DINOv2-small, `dinov2_windowed_small`)             |
| backbone FFN             | (4*dim = 3072 implied)                       | 1536                                                    |
| backbone attn            | packed `attn.qkv.{w,b}` [3*dim, dim]         | separate `attention.attention.{query,key,value}.{w,b}`  |
| layer scale              | none                                         | `layer_scale1.lambda1`, `layer_scale2.lambda1` per blk  |
| cls_token shape          | [1, 1, dim]                                  | [1, 1, 384]                                             |
| pos_embed shape          | [1, n_tokens, dim]                           | [1, 1370, 384]                                          |
| backbone.norm            | `backbone.norm.{w,b}`                        | `backbone.0.encoder.encoder.layernorm.{w,b}`            |
| projector                | 4 linear levels + level_embed                | 1 `MultiScaleProjector` (conv blocks + BN, P4 only)     |
| encoder                  | 3-layer transformer (qkv + ffn + 2 norms)    | none — replaced by `enc_output` ModuleList of 13 linears + 13 LayerNorms (two-stage init) |
| decoder self_attn        | packed `qkv.{w,b}` + `out.{w,b}`             | `in_proj_weight/bias` + `out_proj.{w,b}` (PyTorch MHA)  |
| decoder cross_attn       | standard MHA: `q + kv + out`                 | **deformable**: `sampling_offsets`, `attention_weights`, `value_proj`, `output_proj` |
| heads.class              | `heads.class.fc.{w,b}`, [num_classes, model] | `class_embed.{w,b}` (Linear) + 13 `enc_out_class_embed` |
| heads.bbox               | `heads.bbox.fc1/fc2/fc3.{w,b}`               | `bbox_embed.layers.{0,1,2}.{w,b}` (MLP)                 |
| num_classes              | 80 (COCO)                                    | 91 (COCO + background) per `class_embed.weight = (91,256)` |
| num_queries              | 300                                          | 300 (active), but `refpoint_embed.weight = (3900, 4)` (`num_queries * group_detr=13`) |

Of the ~264 tensor names produced by `expected_tensor_names(cfg)` in
`src/model_loader.cpp`, at minimum:
  - 36 encoder tensors (3 layers x 12) have NO source — encoder doesn't exist
  - ~12 projector tensors have NO source with matching shape
  - 18 decoder cross_attn tensors have NO source — algorithm differs
  - 8 head tensors have NO source with matching shape (num_classes 80 vs 91,
    head names completely different)

Total: ~74 structurally-unbridgeable expected tensors. This vastly exceeds
the 50-missing threshold in the task brief's BLOCKED criteria.
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


# Subset of the expected tensor names this script CAN currently map back to
# rfdetr-base's state_dict (with renames or concatenation). Listed in case
# someone redesigns the runtime later and wants a head start.
#
# Format: GGUF_NAME -> ("rename"|"concat3", source_key(s), reshape_or_None)
#
# This is NOT used by main() at the moment — it just documents the partial map.
PARTIAL_BACKBONE_MAP = {
    # patch_embed: rename only (shape matches, but only when bb_dim=384)
    "backbone.patch_embed.weight":
        ("rename", "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight", None),
    "backbone.patch_embed.bias":
        ("rename", "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.bias", None),
    "backbone.cls_token":
        # source is (1, 1, 384); spec stores (1, 1, dim) — fine if dim=384
        ("rename", "backbone.0.encoder.encoder.embeddings.cls_token", None),
    "backbone.pos_embed":
        # source (1, 1370, 384); spec expects (1, n_tokens, dim).
        # n_tokens for 560/14 = 40^2 + 1 cls = 1601. 1370 = 37^2 + 1 — implies
        # positional_encoding_size=37 (pretrain res). Loader currently sizes
        # by image_size so the lengths disagree (1601 vs 1370). Real port
        # must either interpolate or use a smaller image_size.
        ("rename", "backbone.0.encoder.encoder.embeddings.position_embeddings", None),
    "backbone.norm.weight":
        ("rename", "backbone.0.encoder.encoder.layernorm.weight", None),
    "backbone.norm.bias":
        ("rename", "backbone.0.encoder.encoder.layernorm.bias", None),
    # Per-layer (i = 0..11):
    #   norm1/norm2: rename from backbone.0.encoder.encoder.encoder.layer.{i}.norm{1,2}.{w,b}
    #   attn.qkv.weight: torch.cat([q.w, k.w, v.w], 0) from
    #       backbone.0.encoder.encoder.encoder.layer.{i}.attention.attention.{query,key,value}.weight
    #   attn.qkv.bias: torch.cat([q.b, k.b, v.b], 0) similarly
    #   attn.proj.{w,b}: rename from .attention.output.dense.{w,b}
    #   mlp.fc1/fc2.{w,b}: rename from .mlp.fc1/fc2.{w,b} (NB: fc dim = 1536 = 4*384)
    #   layer_scale1/2.lambda1: NO target in current spec; runtime missing the gamma multiply
}


def main() -> int:
    args = parse_args()
    if args.output is None:
        args.output = f"rfdetr-{args.variant}-{args.dtype}.gguf"

    # Lazy imports so --help works without torch / rfdetr installed
    try:
        import torch  # noqa: F401
        from rfdetr import RFDETRBase
    except ImportError as e:
        print(f"error: missing dependency ({e}). pip install -r scripts/requirements.txt",
              file=sys.stderr)
        return 2

    print(f"Loading rfdetr-{args.variant} ...", file=sys.stderr)

    # NOTE: RFDETRBase() will download ~355 MB to ~/.roboflow on first run.
    # --checkpoint is not yet wired (rfdetr 1.7.0 reads pretrain_weights from
    # the model config; would need a custom constructor path).
    if args.checkpoint:
        print(f"warning: --checkpoint ignored; rfdetr 1.7.0 uses model_config.pretrain_weights "
              f"(currently '{args.checkpoint}' is unused).", file=sys.stderr)

    rfdetr_model = RFDETRBase()
    inner = rfdetr_model.model.model  # LWDETR (rfdetr.models.lwdetr)
    sd = inner.state_dict()

    print(f"\n[probe] inner module: {type(inner).__name__}", file=sys.stderr)
    print(f"[probe] state_dict entries: {len(sd)}", file=sys.stderr)
    print(f"[probe] top-level prefixes: {sorted(set(k.split('.')[0] for k in sd.keys()))}",
          file=sys.stderr)

    # Architecture mismatch summary (see module docstring).
    print("", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print("BLOCKED: rfdetr-base state_dict does not match this project's C++", file=sys.stderr)
    print("         runtime schema. Conversion cannot be completed.", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print("", file=sys.stderr)
    print("Key gaps (see this script's module docstring for the full table):",
          file=sys.stderr)
    print("  - backbone dim 384 (actual) vs 768 (spec) — DINOv2-small not -base",
          file=sys.stderr)
    print("  - separate q/k/v in backbone, not packed qkv", file=sys.stderr)
    print("  - layer_scale (lambda) terms have no place in the spec", file=sys.stderr)
    print("  - NO transformer encoder in the model (spec has 3-layer encoder)",
          file=sys.stderr)
    print("  - decoder cross-attn is DEFORMABLE, not standard MHA", file=sys.stderr)
    print("  - heads use different names + shapes (91 classes vs 80)", file=sys.stderr)
    print("  - projector is conv-based with BN, not 4 linear levels", file=sys.stderr)
    print("", file=sys.stderr)
    print("Source state_dict tensor counts by top-level prefix:", file=sys.stderr)
    prefix_counts = {}
    for k in sd.keys():
        prefix_counts[k.split(".")[0]] = prefix_counts.get(k.split(".")[0], 0) + 1
    for p, n in sorted(prefix_counts.items()):
        print(f"  {p:18s}: {n}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Next step: redesign the C++ runtime (src/*.cpp, model_loader.cpp,",
          file=sys.stderr)
    print("docs/conversion.md, tests/fixtures/gen_model_gguf.cpp) against the",
          file=sys.stderr)
    print("real architecture, then come back to finish this script.", file=sys.stderr)

    if args.dry_run:
        print("\n(--dry-run) Not writing GGUF.", file=sys.stderr)
    return 90  # blocked, architectural mismatch


if __name__ == "__main__":
    sys.exit(main())
