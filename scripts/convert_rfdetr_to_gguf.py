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
    # Conversion body deferred. To complete, see Plan 2 Task 3 in
    # docs/superpowers/plans/2026-05-25-rfdetr-cpp-loader.md. Steps:
    # - load model (handles --checkpoint vs auto-download)
    # - enumerate state_dict
    # - build the tensor name map
    # - validate (no missing / no unmapped)
    # - if --dry-run: print diff and return
    # - else: open GGUF writer, write metadata + tensors, close
    #
    # The C++ loader (src/model_loader.cpp) is exercised by a synthesized
    # GGUF fixture (tests/fixtures/gen_model_gguf.cpp), so this script is not
    # on the critical path for the C++ test suite. Completing it requires
    # installing rfdetr + torch (~GBs) and probing the upstream state_dict
    # to reconcile the spec against the actual rfdetr 1.x tensor names.

    print("error: conversion body not yet implemented (deferred from Plan 2 Task 3).",
          file=sys.stderr)
    print("       See docs/superpowers/plans/2026-05-25-rfdetr-cpp-loader.md", file=sys.stderr)
    return 99


if __name__ == "__main__":
    sys.exit(main())
