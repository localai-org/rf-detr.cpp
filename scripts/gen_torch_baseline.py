#!/usr/bin/env python3
"""Generate a torch parity baseline by running real rfdetr-base on a deterministic
synthetic input and capturing intermediate tensors at named checkpoints. Writes a
baseline-bundle GGUF that Plans 8-12 use as the parity reference while rewriting
the C++ numerical modules.

Format: same as Plan 3's numpy baseline (gguf with `parity.<name>` tensors plus
`parity.format.version = "2"`, `parity.reference = "torch"`, `parity.input_shape`).
Format version bumped to "2" because the checkpoint set is different from Plan 3's
v1 numpy baseline.

Usage:
    python3 scripts/gen_torch_baseline.py \
        --output tests/fixtures/baseline_torch.gguf \
        [--input-seed 7] [--image-size 560]

Requires .venv/ with rfdetr 1.7.0, torch 2.5.1, gguf, numpy installed.
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict

import numpy as np
import torch

try:
    import gguf
except ImportError:
    print("error: 'gguf' package not installed. pip install gguf", file=sys.stderr)
    sys.exit(2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        required=True,
        help="Output baseline GGUF path.",
    )
    p.add_argument(
        "--input-seed",
        type=int,
        default=7,
        help="Seed for the deterministic synthetic input image.",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="Side length of square input (default depends on model: 560 for "
             "rfdetr-base, otherwise the native resolution of --seg-variant).",
    )
    p.add_argument(
        "--seg-variant",
        choices=["nano", "small", "medium", "large", "xlarge", "2xlarge"],
        default=None,
        help="Capture a segmentation variant instead of RFDETRBase. Adds "
             "segmentation_head hook captures. Omit for rfdetr-base.",
    )
    return p.parse_args()


def make_input(seed: int, image_size: int) -> torch.Tensor:
    """Build a deterministic (1, 3, H, W) F32 NCHW tensor that's already mean/std
    normalized. The forward path uses this as the actual model input — we do NOT
    pass it through rfdetr's preprocessor.
    """
    rng = np.random.default_rng(seed)
    # ImageNet-normalized inputs are roughly N(0, ~1). Use that distribution directly.
    arr = rng.standard_normal((1, 3, image_size, image_size)).astype(np.float32)
    return torch.from_numpy(arr)


def _to_tensor(x):
    """Extract a torch.Tensor from a possibly-tuple/list/dict hook output."""
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)) and len(x) > 0 and torch.is_tensor(x[0]):
        return x[0]
    return None


def install_hooks(inner, captured: "OrderedDict[str, torch.Tensor]") -> list:
    """Install forward hooks on the relevant modules to capture intermediate tensors.

    inner: the LWDETR module (m.model.model from RFDETRBase()).
    captured: dict that hooks populate {name: torch.Tensor}.

    Returns the list of registered hooks (caller removes them after forward).
    """
    hooks = []

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------
    # backbone is a Joiner; backbone[0] is the wrapper containing
    # encoder (DinoV2) + projector.
    bb_wrapper = inner.backbone[0]
    dinov2_wrap = bb_wrapper.encoder  # rfdetr DinoV2 wrapper
    hf = dinov2_wrap.encoder  # HF WindowedDinov2WithRegistersBackbone

    # patch embed / CLS + pos_embed addition: the embeddings module's output
    # is the (B, N+1+R, C) sequence fed into the transformer blocks.
    def embed_hook(_mod, _inp, out):
        t = _to_tensor(out)
        if t is not None:
            captured["backbone.patch_embed.output"] = t.detach().clone()
    hooks.append(hf.embeddings.register_forward_hook(embed_hook))

    # Each transformer block's output (12 layers for dinov2_small)
    for i, layer in enumerate(hf.encoder.layer):
        def make_hook(idx):
            def hook(_mod, _inp, out):
                t = _to_tensor(out)
                if t is not None:
                    captured[f"backbone.block.{idx}.output"] = t.detach().clone()
            return hook
        hooks.append(layer.register_forward_hook(make_hook(i)))

    # Final backbone layernorm (when applied — may or may not run depending on
    # apply_layernorm in HF config; hook it regardless).
    if hasattr(hf, "layernorm"):
        def bb_norm_hook(_mod, _inp, out):
            t = _to_tensor(out)
            if t is not None:
                captured["backbone.norm.output"] = t.detach().clone()
        hooks.append(hf.layernorm.register_forward_hook(bb_norm_hook))

    # The DinoV2 wrapper's forward returns a list of feature maps (one per
    # out_feature_index). Capture each.
    def dinov2_wrap_hook(_mod, _inp, out):
        if isinstance(out, (list, tuple)):
            for k, v in enumerate(out):
                if torch.is_tensor(v):
                    captured[f"backbone.multiscale.level{k}"] = v.detach().clone()
    hooks.append(dinov2_wrap.register_forward_hook(dinov2_wrap_hook))

    # ------------------------------------------------------------------
    # Projector (MultiScaleProjector)
    # ------------------------------------------------------------------
    projector = bb_wrapper.projector

    def projector_hook(_mod, _inp, out):
        # projector returns a list of feature maps.
        if isinstance(out, (list, tuple)):
            for k, v in enumerate(out):
                if torch.is_tensor(v):
                    captured[f"projector.output.level{k}"] = v.detach().clone()
            # Also publish the (single-level for rfdetr-base) flat alias.
            if len(out) >= 1 and torch.is_tensor(out[0]):
                captured["projector.output"] = out[0].detach().clone()
        elif torch.is_tensor(out):
            captured["projector.output"] = out.detach().clone()
    hooks.append(projector.register_forward_hook(projector_hook))

    # Sub-stages inside the projector: stages[0] is Sequential(C2f, LayerNorm).
    try:
        c2f = projector.stages[0][0]  # C2f
        final_norm = projector.stages[0][1]  # LayerNorm

        def cv1_hook(_mod, _inp, out):
            if torch.is_tensor(out):
                captured["projector.cv1.output"] = out.detach().clone()

        def cv2_hook(_mod, _inp, out):
            if torch.is_tensor(out):
                captured["projector.cv2.output"] = out.detach().clone()

        def final_norm_hook(_mod, _inp, out):
            if torch.is_tensor(out):
                captured["projector.final_norm.output"] = out.detach().clone()

        hooks.append(c2f.cv1.register_forward_hook(cv1_hook))
        hooks.append(c2f.cv2.register_forward_hook(cv2_hook))
        hooks.append(final_norm.register_forward_hook(final_norm_hook))

        # Per-bottleneck output + internal cv1/cv2 (Plan 9 debug aid).
        for j, bn in enumerate(c2f.m):
            def make_bn_hook(idx):
                def hook(_mod, _inp, out):
                    if torch.is_tensor(out):
                        captured[f"projector.bottleneck.{idx}.output"] = out.detach().clone()
                return hook
            hooks.append(bn.register_forward_hook(make_bn_hook(j)))
            def make_inner_hook(idx, sub):
                def hook(_mod, _inp, out):
                    if torch.is_tensor(out):
                        captured[f"projector.bottleneck.{idx}.{sub}.output"] = out.detach().clone()
                return hook
            hooks.append(bn.cv1.register_forward_hook(make_inner_hook(j, "cv1")))
            hooks.append(bn.cv2.register_forward_hook(make_inner_hook(j, "cv2")))
    except (AttributeError, IndexError, TypeError) as e:
        print(f"# warning: projector sub-stage hooks failed: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Transformer / two-stage / decoder
    # ------------------------------------------------------------------
    tr = inner.transformer

    # enc_output_norm[0]: projected memory tokens for group 0 (the inference group).
    def enc_output_norm_hook(_mod, _inp, out):
        if torch.is_tensor(out):
            captured["two_stage.enc_output_norm.output"] = out.detach().clone()
    hooks.append(tr.enc_output_norm[0].register_forward_hook(enc_output_norm_hook))

    # Capture the inputs to the decoder (post two-stage init): query_feat and the
    # initial refpoints. Hook the *first* decoder layer's pre-call to grab the
    # arguments that flow in. The decoder layer's forward receives (tgt, memory,
    # ...) positionally; we want tgt and the reference_points kwarg.
    if hasattr(tr.decoder, "layers") and len(tr.decoder.layers) > 0:
        def make_dec_input_hook():
            captured_flag = {"done": False}
            def hook(_mod, inp, _out):
                # forward_pre_hook would be cleaner, but we want the actual inputs
                # post pre-hooks. Just stash once.
                if captured_flag["done"]:
                    return
                if isinstance(inp, (list, tuple)) and len(inp) > 0 and torch.is_tensor(inp[0]):
                    captured["decoder.input.tgt"] = inp[0].detach().clone()
                captured_flag["done"] = True
            return hook
        hooks.append(tr.decoder.layers[0].register_forward_hook(make_dec_input_hook()))

        # Use forward_pre_hook to grab the kwargs (reference_points) that the
        # TransformerDecoderLayer.forward receives.
        def dec_layer0_prehook(_mod, args, kwargs):
            if isinstance(args, tuple) and len(args) >= 1 and torch.is_tensor(args[0]):
                captured["decoder.input.tgt"] = args[0].detach().clone()
            rp = kwargs.get("reference_points")
            if torch.is_tensor(rp):
                captured["decoder.input.reference_points"] = rp.detach().clone()
            qp = kwargs.get("query_pos")
            if torch.is_tensor(qp):
                captured["decoder.input.query_pos"] = qp.detach().clone()
            qse = kwargs.get("query_sine_embed")
            if torch.is_tensor(qse):
                captured["decoder.input.query_sine_embed"] = qse.detach().clone()
            return None
        hooks.append(tr.decoder.layers[0].register_forward_pre_hook(
            dec_layer0_prehook, with_kwargs=True))

    # enc_out_class_embed[0]: class proposals (group 0).
    # In LWDETR.forward, the class proposals are computed AFTER transformer.forward
    # returns. But internally to transformer.forward, enc_out_class_embed[0] is also
    # invoked for top-K proposal selection.
    def enc_out_class_hook(_mod, _inp, out):
        if torch.is_tensor(out):
            # Last-write-wins: the LAST invocation is the one used downstream.
            captured["two_stage.enc_out_class.output"] = out.detach().clone()
    hooks.append(tr.enc_out_class_embed[0].register_forward_hook(enc_out_class_hook))

    # Decoder layers — each layer's output (full layer including all three norms +
    # residuals + FFN). Decoder forward returns (output_tensor,) per layer.
    if hasattr(tr, "decoder") and hasattr(tr.decoder, "layers"):
        for i, layer in enumerate(tr.decoder.layers):
            def make_dec_hook(idx):
                def hook(_mod, _inp, out):
                    t = _to_tensor(out)
                    if t is not None:
                        captured[f"decoder.layer.{idx}.output"] = t.detach().clone()
                return hook
            hooks.append(layer.register_forward_hook(make_dec_hook(i)))

            # Sub-module hooks for granular parity debugging.
            def make_sa_hook(idx):
                def hook(_mod, _inp, out):
                    # nn.MultiheadAttention returns (out, attn_weights)
                    t = _to_tensor(out)
                    if t is not None:
                        captured[f"decoder.layer.{idx}.self_attn.output"] = t.detach().clone()
                return hook
            hooks.append(layer.self_attn.register_forward_hook(make_sa_hook(i)))

            def make_ca_hook(idx):
                def hook(_mod, _inp, out):
                    t = _to_tensor(out)
                    if t is not None:
                        captured[f"decoder.layer.{idx}.cross_attn.output"] = t.detach().clone()
                return hook
            hooks.append(layer.cross_attn.register_forward_hook(make_ca_hook(i)))

            def make_n1_hook(idx):
                def hook(_mod, _inp, out):
                    t = _to_tensor(out)
                    if t is not None:
                        captured[f"decoder.layer.{idx}.norm1.output"] = t.detach().clone()
                return hook
            hooks.append(layer.norm1.register_forward_hook(make_n1_hook(i)))

            def make_n2_hook(idx):
                def hook(_mod, _inp, out):
                    t = _to_tensor(out)
                    if t is not None:
                        captured[f"decoder.layer.{idx}.norm2.output"] = t.detach().clone()
                return hook
            hooks.append(layer.norm2.register_forward_hook(make_n2_hook(i)))

            def make_l1_hook(idx):
                def hook(_mod, _inp, out):
                    t = _to_tensor(out)
                    if t is not None:
                        captured[f"decoder.layer.{idx}.linear1.output"] = t.detach().clone()
                return hook
            hooks.append(layer.linear1.register_forward_hook(make_l1_hook(i)))

            def make_l2_hook(idx):
                def hook(_mod, _inp, out):
                    t = _to_tensor(out)
                    if t is not None:
                        captured[f"decoder.layer.{idx}.linear2.output"] = t.detach().clone()
                return hook
            hooks.append(layer.linear2.register_forward_hook(make_l2_hook(i)))

        # Final decoder norm
        if hasattr(tr.decoder, "norm"):
            def dec_norm_hook(_mod, _inp, out):
                t = _to_tensor(out)
                if t is not None:
                    captured["decoder.norm.output"] = t.detach().clone()
            hooks.append(tr.decoder.norm.register_forward_hook(dec_norm_hook))

        # ref_point_head: takes sine-cosine pos embed → query_pos. Capture both
        # input and output of the MLP — its input is the gen_sineembed_for_position
        # of the (cx,cy,w,h) refpoints, useful for verifying the sine embed
        # implementation. The MLP's input is its first positional argument.
        if hasattr(tr.decoder, "ref_point_head"):
            def rph_hook(_mod, inp, out):
                if isinstance(inp, (list, tuple)) and len(inp) > 0 and torch.is_tensor(inp[0]):
                    captured["decoder.ref_point_head.input"] = inp[0].detach().clone()
                t = _to_tensor(out)
                if t is not None:
                    captured["decoder.ref_point_head.output"] = t.detach().clone()
            hooks.append(tr.decoder.ref_point_head.register_forward_hook(rph_hook))

    # ------------------------------------------------------------------
    # Heads (top-level on inner). LWDETR applies bbox_embed and class_embed on
    # the stacked decoder outputs (all layers) — the hooked tensor will be the
    # full (n_layers, B, num_queries, *) result. That's still useful as parity;
    # downstream tests can index the last layer.
    # ------------------------------------------------------------------
    def class_hook(_mod, _inp, out):
        if torch.is_tensor(out):
            captured["heads.class_logits"] = out.detach().clone()

    def bbox_hook(_mod, _inp, out):
        if torch.is_tensor(out):
            captured["heads.bbox_pred"] = out.detach().clone()

    hooks.append(inner.class_embed.register_forward_hook(class_hook))
    hooks.append(inner.bbox_embed.register_forward_hook(bbox_hook))

    return hooks


def install_seg_decoder_intermediate_hook(inner, captured) -> list:
    """For seg models the decoder needs to expose per-layer post-norm
    outputs (the seg head iterates each layer's normalized output as the
    `query_features` stream). The default install_hooks already captures
    each `decoder.layer.{i}.output` (pre-final-norm), but the seg head sees
    the FINAL norm applied per-layer.

    We monkey-patch the TransformerDecoder.forward to grab `intermediate` —
    the list of per-layer normed outputs.
    """
    hooks = []
    dec = inner.transformer.decoder
    orig_fwd = dec.forward

    def wrapped(*args, **kwargs):
        out = orig_fwd(*args, **kwargs)
        # When `return_intermediate=True` the decoder returns
        # [stacked_intermediate, stacked_refpoints]. Each stacked tensor is
        # (n_layers, NQ, B, C) — we squeeze to per-layer captures.
        if isinstance(out, (list, tuple)) and len(out) >= 1:
            hs = out[0]
            if torch.is_tensor(hs) and hs.dim() >= 1:
                for i in range(hs.shape[0]):
                    captured[f"decoder.layer.{i}.post_norm"] = hs[i].detach().clone()
        return out

    dec.forward = wrapped

    class _Restore:
        def __init__(self, mod, orig):
            self.mod = mod
            self.orig = orig

        def remove(self):
            self.mod.forward = self.orig

    hooks.append(_Restore(dec, orig_fwd))
    return hooks


def install_seg_hooks(inner, captured: "OrderedDict[str, torch.Tensor]") -> list:
    """Install hooks specific to the SegmentationHead module.

    inner: the LWDETR module. inner.segmentation_head is the SegmentationHead.

    The seg head forward iterates `zip(self.blocks, query_features)` where
    `query_features` is the stacked decoder output (n_layers, B, NQ, C). For
    each layer:
      spatial_features = block(spatial_features)
      spatial_proj     = spatial_features_proj(spatial_features)
      qf               = query_features_block(qf); qf = query_features_proj(qf)
      mask             = einsum("bchw,bnc->bnhw", spatial_proj, qf) + bias

    We capture:
      seg.spatial_features.input    — features[0].tensors (projector output)
      seg.spatial_features.resized  — after F.interpolate
      seg.block.{i}.spatial_out     — after block(spatial_features)
      seg.block.{i}.spatial_proj    — after spatial_features_proj (1x1 conv)
      seg.block.{i}.qf_proj         — after query_features_block + proj
      seg.masks.{i}                 — per-layer mask logits
      seg.masks.final               — masks[-1] (inference output)
    """
    hooks = []
    sh = inner.segmentation_head

    # Wrap the seg head's forward to intercept resized + per-block intermediates.
    # NOTE: LWDETR.forward calls the seg head TWICE when two_stage=True:
    #   1. With the stacked decoder outputs `hs` (skip_blocks=False)
    #   2. With `[hs_enc]` and skip_blocks=True (for the encoder branch)
    #
    # The inference output `pred_masks` is `outputs_masks[-1]` from CALL #1,
    # NOT call #2 — so we only capture intermediates from the first call,
    # and never overwrite them from the skip_blocks=True path.
    orig_forward = sh.forward
    state = {"call_count": 0}

    def traced_forward(spatial_features, query_features, image_size, skip_blocks=False):
        import torch as _t
        import torch.nn.functional as _F

        state["call_count"] += 1
        is_first_call = state["call_count"] == 1

        if not skip_blocks and is_first_call:
            captured["seg.spatial_features.input"] = spatial_features.detach().clone()
        target_size = (image_size[0] // sh.downsample_ratio,
                       image_size[1] // sh.downsample_ratio)
        sp = _F.interpolate(spatial_features, size=target_size,
                            mode="bilinear", align_corners=False)
        if not skip_blocks and is_first_call:
            captured["seg.spatial_features.resized"] = sp.detach().clone()

        mask_logits = []
        if not skip_blocks:
            for i, (block, qf) in enumerate(zip(sh.blocks, query_features)):
                sp = block(sp)
                sp_proj = sh.spatial_features_proj(sp)
                qf_p = sh.query_features_proj(sh.query_features_block(qf))
                mk = _t.einsum("bchw,bnc->bnhw", sp_proj, qf_p) + sh.bias
                if is_first_call:
                    captured[f"seg.block.{i}.spatial_out"] = sp.detach().clone()
                    captured[f"seg.block.{i}.spatial_proj"] = sp_proj.detach().clone()
                    captured[f"seg.block.{i}.qf_proj"] = qf_p.detach().clone()
                    captured[f"seg.masks.{i}"] = mk.detach().clone()
                mask_logits.append(mk)
        else:
            assert len(query_features) == 1
            qf_p = sh.query_features_proj(sh.query_features_block(query_features[0]))
            mk = _t.einsum("bchw,bnc->bnhw", sp, qf_p) + sh.bias
            mask_logits.append(mk)

        if mask_logits and not skip_blocks and is_first_call:
            captured["seg.masks.final"] = mask_logits[-1].detach().clone()
        return mask_logits

    # Install by monkey-patching forward; the caller's "hooks" list owns a
    # restore-thunk.
    sh.forward = traced_forward

    class _RestoreFwd:
        def __init__(self, mod, orig):
            self.mod = mod
            self.orig = orig

        def remove(self):
            self.mod.forward = self.orig

    hooks.append(_RestoreFwd(sh, orig_forward))
    return hooks


def run_forward(inner, x: torch.Tensor):
    """Run LWDETR forward with a NestedTensor wrapping `x`. Returns the model out."""
    from rfdetr.utilities.tensors import NestedTensor

    mask = torch.zeros(x.shape[0], x.shape[2], x.shape[3], dtype=torch.bool)
    nt = NestedTensor(x, mask)
    with torch.no_grad():
        return inner(nt)


def write_baseline(
    path: str,
    captured: "OrderedDict[str, torch.Tensor]",
    input_shape: tuple,
) -> int:
    """Write the captured tensors to a baseline GGUF in Plan-3 v2 format.

    Returns the number of tensors written (excluding any non-tensor entries).
    """
    writer = gguf.GGUFWriter(path, arch="rfdetr-parity")
    writer.add_string("parity.format.version", "2")
    writer.add_string("parity.reference", "torch")
    writer.add_array("parity.input_shape", [int(s) for s in input_shape])

    written = 0
    for name in sorted(captured.keys()):
        t = captured[name]
        if not torch.is_tensor(t):
            print(f"# skip non-tensor capture {name!r}: {type(t).__name__}",
                  file=sys.stderr)
            continue
        arr = t.cpu().to(torch.float32).contiguous().numpy()
        writer.add_tensor(f"parity.{name}", arr)
        written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return written


# Native inference resolution per seg variant. Must match the table in
# scripts/convert_rfdetr_to_gguf.py:160-201.
SEG_VARIANTS = {
    "nano":    ("RFDETRSegNano",    312),
    "small":   ("RFDETRSegSmall",   384),
    "medium":  ("RFDETRSegMedium",  432),
    "large":   ("RFDETRSegLarge",   504),
    "xlarge":  ("RFDETRSegXLarge",  624),
    "2xlarge": ("RFDETRSeg2XLarge", 768),
}


def main() -> int:
    args = parse_args()

    is_seg = args.seg_variant is not None

    # Default image size depends on model variant.
    image_size = args.image_size
    if image_size <= 0:
        image_size = SEG_VARIANTS[args.seg_variant][1] if is_seg else 560

    if is_seg:
        cls_name = SEG_VARIANTS[args.seg_variant][0]
        print(f"Loading rfdetr-seg-{args.seg_variant} ({cls_name})...", file=sys.stderr)
        import rfdetr
        m = getattr(rfdetr, cls_name)()
    else:
        print("Loading rfdetr-base...", file=sys.stderr)
        from rfdetr import RFDETRBase
        m = RFDETRBase()
    m.model.model.eval()
    inner = m.model.model  # LWDETR

    x = make_input(args.input_seed, image_size)
    print(f"Input shape: {tuple(x.shape)}", file=sys.stderr)

    captured: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    captured["preprocess.input"] = x.detach().clone()

    hooks = install_hooks(inner, captured)
    if is_seg:
        hooks.extend(install_seg_decoder_intermediate_hook(inner, captured))
        hooks.extend(install_seg_hooks(inner, captured))
    print(f"Installed {len(hooks)} forward hooks.", file=sys.stderr)

    print("Running forward...", file=sys.stderr)
    try:
        run_forward(inner, x)
    finally:
        for h in hooks:
            h.remove()
    print(f"Forward done; captured {len(captured)} tensors.", file=sys.stderr)

    print(f"Writing baseline to {args.output}...", file=sys.stderr)
    n_written = write_baseline(args.output, captured, x.shape)

    # Print captured names so the run is auditable.
    print("Captured checkpoints:", file=sys.stderr)
    for name in sorted(captured.keys()):
        t = captured[name]
        if torch.is_tensor(t):
            print(f"  {name}: {tuple(t.shape)} {t.dtype}", file=sys.stderr)
        else:
            print(f"  {name}: <{type(t).__name__}>", file=sys.stderr)
    print(f"Wrote {n_written} tensors to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
