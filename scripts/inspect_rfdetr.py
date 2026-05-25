#!/usr/bin/env python3
"""Dump rfdetr-base architecture to stdout.

Idempotent introspection helper. Loads RFDETRBase() (uses cached weights at
~/.roboflow/models/rf-detr-base.pth on subsequent runs) and prints a structured
view of:

  - inner LWDETR top-level children
  - full state_dict listing grouped by top-level prefix
  - DinoV2 backbone deep-dive (modules + HF config: hidden_size, num_hidden_layers,
    num_attention_heads, patch_size, image_size, window_block_indexes,
    num_register_tokens, num_windows, out_features/out_indices)
  - MultiScaleProjector deep-dive (C2f / ConvX with LayerNorm-as-BN)
  - Transformer deep-dive (decoder layers, two-stage enc_output/enc_output_norm/
    enc_out_class_embed/enc_out_bbox_embed, ref_point_head, MSDeformAttn params)
  - decoder layer 0 detail (cross_attn n_levels / n_heads / n_points)
  - heads (class_embed Linear, bbox_embed MLP)
  - tgt_embed and refpoint_embed (note: rfdetr-base uses `query_feat` + `refpoint_embed`)
  - preprocessing means/stds and resolution from the RFDETRBase wrapper

Output drives Plan 7's schema rewrite and Plans 8-12's per-module redesigns.

Usage:
  python3 scripts/inspect_rfdetr.py > docs/rfdetr_arch.txt
"""

from __future__ import annotations

from collections import defaultdict

import torch
from rfdetr import RFDETRBase


SECTION = "=" * 80


def _dump_module(mod, indent: int = 2, max_depth: int = 10) -> None:
    if max_depth <= 0:
        print(f"{' ' * indent}... (max depth)")
        return
    for name, child in mod.named_children():
        ctype = type(child).__name__
        extra = ""
        for attr in (
            "in_features",
            "out_features",
            "embed_dim",
            "num_heads",
            "num_attention_heads",
            "kernel_size",
            "stride",
            "padding",
            "normalized_shape",
            "elementwise_affine",
        ):
            if hasattr(child, attr):
                v = getattr(child, attr)
                if not isinstance(v, torch.nn.Module):
                    extra += f" {attr}={v}"
        print(f"{' ' * indent}{name}: {ctype}{extra}")
        if list(child.named_children()):
            _dump_module(child, indent + 2, max_depth - 1)


def _print_scalar_attrs(label: str, obj, attrs: list[str]) -> None:
    for a in attrs:
        if hasattr(obj, a):
            v = getattr(obj, a)
            if isinstance(v, (int, float, str, bool, tuple, list)):
                print(f"  {label}.{a} = {v!r}")


def main() -> None:
    m = RFDETRBase()
    inner = m.model.model  # LWDETR

    # 1. Top-level structure
    print(SECTION)
    print("TOP-LEVEL nn.Modules under inner (LWDETR):")
    for name, mod in inner.named_children():
        print(f"  {name}: {type(mod).__name__}")

    # 2. Full state_dict by prefix
    print("\n" + SECTION)
    print("STATE_DICT BY PREFIX:")
    sd = inner.state_dict()
    by_pfx: dict[str, list[tuple[str, tuple[int, ...], str]]] = defaultdict(list)
    for k, v in sd.items():
        by_pfx[k.split(".")[0]].append((k, tuple(v.shape), str(v.dtype)))
    for pfx in sorted(by_pfx.keys()):
        entries = by_pfx[pfx]
        print(f"\n--- prefix: {pfx} ({len(entries)} tensors) ---")
        for k, shape, dt in entries:
            print(f"  {k} {shape} {dt}")

    # 3. Backbone deep-dive
    print("\n" + SECTION)
    print("BACKBONE DEEP-DIVE:")
    bb_joiner = inner.backbone
    print(f"  backbone (joiner) type: {type(bb_joiner).__name__}")
    print(f"  backbone[0] type: {type(bb_joiner[0]).__name__}")
    bb = bb_joiner[0]
    _dump_module(bb)

    # 3b. DinoV2 wrapper + HF config
    enc = bb.encoder  # DinoV2 wrapper
    print("\n  bb.encoder (DinoV2 wrapper):")
    _print_scalar_attrs("    enc", enc, ["num_windows", "patch_size", "shape"])

    hf = enc.encoder  # HF WindowedDinov2WithRegistersBackbone
    print(f"\n  bb.encoder.encoder type: {type(hf).__name__}")
    if hasattr(hf, "config"):
        cfg = hf.config
        d = cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg)
        print("  HF config (selected keys):")
        for k in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "mlp_ratio",
            "hidden_act",
            "layer_norm_eps",
            "image_size",
            "patch_size",
            "qkv_bias",
            "layerscale_value",
            "num_register_tokens",
            "num_windows",
            "window_block_indexes",
            "out_features",
            "out_indices",
            "apply_layernorm",
            "reshape_hidden_states",
        ):
            if k in d:
                print(f"    {k} = {d[k]!r}")

    # 4. Projector (top-level inner attr or under bb)
    print("\n" + SECTION)
    print("PROJECTOR DEEP-DIVE:")
    for attr in ("projector", "input_proj", "neck"):
        if hasattr(bb, attr):
            print(f"  bb.{attr} type: {type(getattr(bb, attr)).__name__}")
            _dump_module(getattr(bb, attr), indent=4)
            break
        if hasattr(inner, attr):
            print(f"  inner.{attr} type: {type(getattr(inner, attr)).__name__}")
            _dump_module(getattr(inner, attr), indent=4)
            break

    # 5. Transformer
    print("\n" + SECTION)
    print("TRANSFORMER DEEP-DIVE:")
    tr = inner.transformer
    print(f"  type: {type(tr).__name__}")
    print("  named_children:")
    _dump_module(tr, indent=4)
    _print_scalar_attrs(
        "  tr",
        tr,
        [
            "num_queries",
            "group_detr",
            "two_stage",
            "num_levels",
            "num_classes",
            "d_model",
            "nhead",
            "dim_feedforward",
            "num_decoder_layers",
        ],
    )

    # 6. Decoder layer 0 + cross_attn (MSDeformAttn) params
    if hasattr(tr, "decoder") and hasattr(tr.decoder, "layers"):
        print("\n" + SECTION)
        print("DECODER LAYER 0 DEEP-DIVE:")
        l0 = tr.decoder.layers[0]
        print(f"  type: {type(l0).__name__}")
        _dump_module(l0, indent=4)
        if hasattr(l0, "cross_attn"):
            ca = l0.cross_attn
            print("\n  Decoder layer 0 cross_attn details:")
            print(f"    cross_attn type: {type(ca).__name__}")
            _print_scalar_attrs(
                "    cross_attn",
                ca,
                ["n_levels", "n_heads", "n_points", "d_model", "im2col_step"],
            )

    # 7. Heads
    print("\n" + SECTION)
    print("HEADS DEEP-DIVE:")
    for attr in ("class_embed", "bbox_embed"):
        if hasattr(inner, attr):
            v = getattr(inner, attr)
            print(f"  inner.{attr}: {type(v).__name__}")
            if isinstance(v, torch.nn.Module):
                _dump_module(v, indent=4)

    # 8. Query / refpoint embed (note: rfdetr-base uses `query_feat`, not `tgt_embed`)
    print("\n" + SECTION)
    print("QUERY/REFPOINT EMBEDS:")
    for attr in ("tgt_embed", "query_feat", "refpoint_embed"):
        for owner_name, owner in (("inner", inner), ("tr", tr)):
            if hasattr(owner, attr):
                v = getattr(owner, attr)
                print(f"  {owner_name}.{attr}: {type(v).__name__}")
                if hasattr(v, "weight"):
                    print(f"    weight shape: {tuple(v.weight.shape)}")

    # 9. Preprocessing
    print("\n" + SECTION)
    print("PREPROCESSING:")
    for a in ("means", "stds", "size", "class_names"):
        if hasattr(m, a):
            v = getattr(m, a)
            if isinstance(v, (list, tuple)) and len(v) > 10:
                print(f"  m.{a} = [{len(v)} entries] (first 5: {list(v)[:5]})")
            else:
                print(f"  m.{a} = {v!r}")
    if hasattr(m.model, "resolution"):
        print(f"  m.model.resolution = {m.model.resolution!r}")

    # 10. Summary
    print("\n" + SECTION)
    print(f"TOTAL TENSORS IN state_dict: {len(sd)}")
    print("TOP-LEVEL PREFIX COUNTS:")
    for pfx, entries in sorted(by_pfx.items(), key=lambda x: -len(x[1])):
        print(f"  {pfx}: {len(entries)}")


if __name__ == "__main__":
    main()
