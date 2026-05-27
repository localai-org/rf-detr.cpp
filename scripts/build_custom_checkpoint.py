#!/usr/bin/env python3
"""scripts/build_custom_checkpoint.py — produces a tiny RF-DETR checkpoint
with a custom num_classes, suitable for testing the fine-tuning conversion
path end-to-end.

This is NOT a real fine-tune; it loads the pretrained variant (default Base),
resizes the classification head via reinitialize_detection_head(), and saves
in rfdetr's standard checkpoint format ({"model": state_dict, "args": {...}}).
Use it to validate that the GGUF converter + C++ loader handle non-91 class
counts correctly.

Usage:
    .venv/bin/python scripts/build_custom_checkpoint.py \\
        --output /tmp/custom5.pth --num-classes 5

The output checkpoint can then be converted with:
    .venv/bin/python scripts/convert_rfdetr_to_gguf.py \\
        --checkpoint /tmp/custom5.pth --output /tmp/custom5.gguf
"""
import argparse
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, help="Output .pth path")
    p.add_argument("--num-classes", type=int, default=5,
                   help="Target class count for the resized head")
    p.add_argument("--variant", choices=["nano", "small", "base", "medium", "large"],
                   default="base", help="Which RF-DETR variant to base the checkpoint on")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import torch
        from rfdetr import (
            RFDETRBase, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge,
        )
    except ImportError as e:
        print(f"error: missing dependency ({e}). pip install -r scripts/requirements.txt",
              file=sys.stderr)
        return 2

    _VARIANT_CLASSES = {
        "nano":   RFDETRNano,
        "small":  RFDETRSmall,
        "base":   RFDETRBase,
        "medium": RFDETRMedium,
        "large":  RFDETRLarge,
    }

    print(f"Loading rfdetr-{args.variant} (pretrained) ...", file=sys.stderr)
    m = _VARIANT_CLASSES[args.variant]()
    inner = m.model.model

    print(f"Resizing classification head: {inner.class_embed.out_features} -> "
          f"{args.num_classes}", file=sys.stderr)
    # reinitialize_detection_head replaces both class_embed and
    # transformer.enc_out_class_embed (13 groups) when two_stage=True. This
    # is the same path the upstream library uses for fine-tuning.
    inner.reinitialize_detection_head(args.num_classes)

    sd = inner.state_dict()
    # Sanity-check the head shapes after the resize.
    assert sd["class_embed.weight"].shape == (args.num_classes, 256), \
        f"unexpected class_embed.weight shape: {tuple(sd['class_embed.weight'].shape)}"
    assert sd["class_embed.bias"].shape == (args.num_classes,), \
        f"unexpected class_embed.bias shape: {tuple(sd['class_embed.bias'].shape)}"

    ckpt = {
        "model": sd,
        "args": {"num_classes": args.num_classes},
        "model_name": _VARIANT_CLASSES[args.variant].__name__,
    }
    torch.save(ckpt, args.output)
    print(f"wrote {args.output} (num_classes={args.num_classes}, "
          f"variant={args.variant})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
