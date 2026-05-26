#!/usr/bin/env python3
"""Convert an upstream rfdetr PyTorch checkpoint to GGUF (format v2).

Usage:
    python scripts/convert_rfdetr_to_gguf.py --variant base \
        --output rfdetr-base-f16.gguf \
        [--dtype f16|f32|q4_0|q4_1|q5_0|q5_1|q8_0] [--dry-run]

When --dtype is one of the quantized formats, only 2D weight tensors with both
dims >= 64 are quantized; LayerNorm params, biases, embeddings, layer-scale
gammas and conv kernels stay F32. This keeps the loader simple while still
compressing the ~30 MB of linear-projection weights that dominate model size.

Quant format notes (block sizes / compression ratios on rfdetr-base's ~30 MB
of quantizable linear weights):
  * q8_0 — 32-element blocks, F16 scale + 32 int8 — ~3.1x model compression
  * q5_1 — 32-element blocks, F16 scale + F16 min + 32 5-bit nibbles — ~4.1x
  * q5_0 — 32-element blocks, F16 scale + 32 5-bit nibbles — ~4.5x
  * q4_1 — 32-element blocks, F16 scale + F16 min + 32 4-bit nibbles — ~5.0x
  * q4_0 — 32-element blocks, F16 scale + 32 4-bit nibbles — ~5.6x

K-quants (q4_K, q5_K, q6_K) are not exposed by the `gguf` Python package
(NotImplementedError in gguf.quants.quantize_blocks). For K-quants, use
the C++ quantizer instead — it links ggml directly and supports the full
ggml type set:

    build/bin/rfdetr-cli quantize models/rfdetr-base-f32.gguf \
        models/rfdetr-base-q4_K.gguf q4_K

Q8_0 output from this Python converter is byte-for-byte identical to the
C++ output (no quant-rounding ambiguity). Q4_0 / Q5_0 outputs match in all
but a handful of nibbles across the ~30 MB tensor blob — the Python path
uses the gguf package's "FMA-cursed" reference whereas the C++ path uses
ggml's own ref kernel (which is what runs at inference time). See
BENCHMARK.md → "What about 4-bit?" for the full comparison.

Format version: "2" (see docs/conversion.md).

What this script does
---------------------
1. Loads `RFDETRBase()` (downloads ~355 MB to ~/.roboflow on first run).
2. Extracts the underlying nn.Module via `m.model.model` (LWDETR).
3. Builds a {gguf_tensor_name -> source_state_dict_key} map from the live
   state_dict, slicing the multi-group query tensors down to group 0 and
   skipping training-only entries (mask_token).
4. Validates that every expected v2 tensor has a source.
5. Writes the GGUF with metadata matching docs/conversion.md.

PyTorch -> GGUF shape conventions
---------------------------------
- Linear weight (out, in) -> ne (in, out)              (numpy as-is, gguf reverses)
- Conv2d weight (out, in, kh, kw) -> ne (kw, kh, in, out)
- cls_token (1, 1, dim) -> squeezed to (dim,)
- pos_embed (1, n_tokens, dim) -> squeezed to (n_tokens, dim) -> ne (dim, n_tokens)
- 1D LayerNorm/bias -> (dim,)
"""

import argparse
import sys
from pathlib import Path


FORMAT_VERSION = "2"

# Per-variant config that gets stamped into GGUF metadata.
# All 5 detection variants share the same DINOv2-small backbone (dim=384,
# depth=12, heads=6, ffn_dim=1536) and the same decoder model_dim (256),
# num_queries (300), num_classes (91), and group_detr (13). They differ in:
#   * image_size & patch_size (controls inference grid: image_size/patch_size)
#   * num_windows (Base uses 4, all others use 2)
#   * global_attn_indices (0-indexed block ids running global attention)
#   * out_feature_indices (1-indexed HF stage names; stage k = layer k-1 output)
#   * pos_embed_train_size (DinoV2 pretrain grid size; pos_embed has
#     pos_embed_train_size^2 + 1 tokens)
#   * decoder.layers (2/3/3/4/4 for Nano/Small/Base/Medium/Large)
def _make_variant_cfg(*, image_size, patch_size, num_windows,
                       global_attn_indices, out_feature_indices,
                       pos_embed_train_size, dec_layers):
    # `num_classes` here is the COCO default. The actual value written to
    # GGUF is derived from the loaded model's class_embed.weight.shape[0]
    # (see main()), so fine-tuned checkpoints with a custom class count
    # are stamped correctly without any code change.
    return {
        "image_size":             image_size,
        "patch_size":             patch_size,
        "num_queries":            300,
        "group_detr":             13,
        "num_classes":            91,
        "backbone": {
            "dim":                384,
            "depth":              12,
            "heads":              6,
            "ffn_dim":            1536,
            "num_windows":        num_windows,
            "global_attn_indices": global_attn_indices,
            "out_feature_indices": out_feature_indices,
            "pos_embed_train_size": pos_embed_train_size,
        },
        "projector": {
            "in_dim":             1536,
            "out_dim":            256,
            "bottleneck_dim":     128,
            "n_bottlenecks":      3,
        },
        "decoder": {
            "layers":             dec_layers,
            "model_dim":          256,
            "ffn_dim":            2048,
            "self_attn_heads":    8,
            "cross_attn_heads":   16,
            "cross_attn_n_levels": 1,
            "cross_attn_n_points": 2,
        },
        "two_stage": {
            "n_groups":           13,
        },
    }


VARIANTS = {
    "nano": _make_variant_cfg(
        image_size=384, patch_size=16, num_windows=2,
        global_attn_indices=[3, 6, 9],
        out_feature_indices=[3, 6, 9, 12],
        pos_embed_train_size=24, dec_layers=2,
    ),
    "small": _make_variant_cfg(
        image_size=512, patch_size=16, num_windows=2,
        global_attn_indices=[3, 6, 9],
        out_feature_indices=[3, 6, 9, 12],
        pos_embed_train_size=32, dec_layers=3,
    ),
    "base": _make_variant_cfg(
        image_size=560, patch_size=14, num_windows=4,
        global_attn_indices=[2, 5, 8, 11],
        out_feature_indices=[2, 5, 8, 11],
        pos_embed_train_size=37, dec_layers=3,
    ),
    "medium": _make_variant_cfg(
        image_size=576, patch_size=16, num_windows=2,
        global_attn_indices=[3, 6, 9],
        out_feature_indices=[3, 6, 9, 12],
        pos_embed_train_size=36, dec_layers=4,
    ),
    "large": _make_variant_cfg(
        image_size=704, patch_size=16, num_windows=2,
        global_attn_indices=[3, 6, 9],
        out_feature_indices=[3, 6, 9, 12],
        pos_embed_train_size=44, dec_layers=4,
    ),
}

# 91-slot COCO logit table for the `class_embed` head. The 80 COCO names sit
# at their canonical 1..90 index positions; reserved IDs are "" placeholders.
# (Slot 0 is conventionally "background" in COCO but rfdetr keeps it in the
# 91-wide logit set; left empty for clarity.)
COCO_CLASS_NAMES = [""] * 91
_COCO_91 = [
    (1, "person"), (2, "bicycle"), (3, "car"), (4, "motorcycle"),
    (5, "airplane"), (6, "bus"), (7, "train"), (8, "truck"), (9, "boat"),
    (10, "traffic light"), (11, "fire hydrant"), (13, "stop sign"),
    (14, "parking meter"), (15, "bench"), (16, "bird"), (17, "cat"),
    (18, "dog"), (19, "horse"), (20, "sheep"), (21, "cow"), (22, "elephant"),
    (23, "bear"), (24, "zebra"), (25, "giraffe"), (27, "backpack"),
    (28, "umbrella"), (31, "handbag"), (32, "tie"), (33, "suitcase"),
    (34, "frisbee"), (35, "skis"), (36, "snowboard"), (37, "sports ball"),
    (38, "kite"), (39, "baseball bat"), (40, "baseball glove"),
    (41, "skateboard"), (42, "surfboard"), (43, "tennis racket"),
    (44, "bottle"), (46, "wine glass"), (47, "cup"), (48, "fork"),
    (49, "knife"), (50, "spoon"), (51, "bowl"), (52, "banana"),
    (53, "apple"), (54, "sandwich"), (55, "orange"), (56, "broccoli"),
    (57, "carrot"), (58, "hot dog"), (59, "pizza"), (60, "donut"),
    (61, "cake"), (62, "chair"), (63, "couch"), (64, "potted plant"),
    (65, "bed"), (67, "dining table"), (70, "toilet"), (72, "tv"),
    (73, "laptop"), (74, "mouse"), (75, "remote"), (76, "keyboard"),
    (77, "cell phone"), (78, "microwave"), (79, "oven"), (80, "toaster"),
    (81, "sink"), (82, "refrigerator"), (84, "book"), (85, "clock"),
    (86, "vase"), (87, "scissors"), (88, "teddy bear"), (89, "hair drier"),
    (90, "toothbrush"),
]
for _idx, _name in _COCO_91:
    COCO_CLASS_NAMES[_idx] = _name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=sorted(VARIANTS.keys()), default="base")
    p.add_argument("--output", required=False,
                   help="Output GGUF path. Default: rfdetr-<variant>-<dtype>.gguf")
    p.add_argument("--dtype",
                   choices=["f16", "f32", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
                   default="f16")
    p.add_argument("--checkpoint",
                   help="Optional path to a local rfdetr .pth. Default: download via rfdetr pkg.")
    p.add_argument("--dry-run", action="store_true",
                   help="Load model + validate name map, but do not write GGUF.")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Tensor name map: GGUF name -> (source_state_dict_key, transform)
#
# transform is either:
#   None                     -> use the tensor as-is
#   "slice_first_300"        -> tensor[:300]      (group-0 slice of query tensors)
#   "squeeze_cls"            -> tensor.reshape(-1) (drops leading 1,1)
#   "squeeze_pos"            -> tensor[0]         (drops leading 1 to keep (n_tokens, dim))
# ----------------------------------------------------------------------------

def build_tensor_name_map(variant_cfg):
    """Returns ordered dict: gguf_name -> (source_key, transform_tag)."""
    bb_depth   = variant_cfg["backbone"]["depth"]
    n_bn       = variant_cfg["projector"]["n_bottlenecks"]
    n_groups   = variant_cfg["two_stage"]["n_groups"]
    dec_layers = variant_cfg["decoder"]["layers"]

    BB = "backbone.0.encoder.encoder."   # DINOv2 prefix inside Joiner
    PJ = "backbone.0.projector.stages.0." # single-stage projector
    TF = "transformer."                   # two-stage + decoder root

    m = {}

    # ---- Backbone embeddings (4) ----
    m["backbone.patch_embed.weight"] = (BB + "embeddings.patch_embeddings.projection.weight", None)
    m["backbone.patch_embed.bias"]   = (BB + "embeddings.patch_embeddings.projection.bias", None)
    m["backbone.cls_token"]          = (BB + "embeddings.cls_token", "squeeze_cls")
    m["backbone.pos_embed"]          = (BB + "embeddings.position_embeddings", "squeeze_pos")

    # ---- Backbone blocks ----
    for i in range(bb_depth):
        src = BB + f"encoder.layer.{i}."
        dst = f"backbone.blocks.{i}."
        m[dst + "norm1.weight"]    = (src + "norm1.weight", None)
        m[dst + "norm1.bias"]      = (src + "norm1.bias",   None)
        m[dst + "attn.q.weight"]   = (src + "attention.attention.query.weight", None)
        m[dst + "attn.q.bias"]     = (src + "attention.attention.query.bias",   None)
        m[dst + "attn.k.weight"]   = (src + "attention.attention.key.weight",   None)
        m[dst + "attn.k.bias"]     = (src + "attention.attention.key.bias",     None)
        m[dst + "attn.v.weight"]   = (src + "attention.attention.value.weight", None)
        m[dst + "attn.v.bias"]     = (src + "attention.attention.value.bias",   None)
        m[dst + "attn.proj.weight"] = (src + "attention.output.dense.weight",   None)
        m[dst + "attn.proj.bias"]   = (src + "attention.output.dense.bias",     None)
        m[dst + "layer_scale1"]    = (src + "layer_scale1.lambda1", None)
        m[dst + "norm2.weight"]    = (src + "norm2.weight", None)
        m[dst + "norm2.bias"]      = (src + "norm2.bias",   None)
        m[dst + "mlp.fc1.weight"]  = (src + "mlp.fc1.weight", None)
        m[dst + "mlp.fc1.bias"]    = (src + "mlp.fc1.bias",   None)
        m[dst + "mlp.fc2.weight"]  = (src + "mlp.fc2.weight", None)
        m[dst + "mlp.fc2.bias"]    = (src + "mlp.fc2.bias",   None)
        m[dst + "layer_scale2"]    = (src + "layer_scale2.lambda1", None)

    # ---- Backbone final norm ----
    m["backbone.norm.weight"] = (BB + "layernorm.weight", None)
    m["backbone.norm.bias"]   = (BB + "layernorm.bias",   None)

    # ---- Projector (single P4 C2f) ----
    # Note: upstream .bn.{weight,bias} are LayerNorm params (no running stats);
    # we rename to .norm. on the GGUF side.
    m["projector.cv1.conv.weight"] = (PJ + "0.cv1.conv.weight", None)
    m["projector.cv1.norm.weight"] = (PJ + "0.cv1.bn.weight",   None)
    m["projector.cv1.norm.bias"]   = (PJ + "0.cv1.bn.bias",     None)
    m["projector.cv2.conv.weight"] = (PJ + "0.cv2.conv.weight", None)
    m["projector.cv2.norm.weight"] = (PJ + "0.cv2.bn.weight",   None)
    m["projector.cv2.norm.bias"]   = (PJ + "0.cv2.bn.bias",     None)
    for j in range(n_bn):
        sj = PJ + f"0.m.{j}."
        dj = f"projector.bottleneck.{j}."
        m[dj + "cv1.conv.weight"] = (sj + "cv1.conv.weight", None)
        m[dj + "cv1.norm.weight"] = (sj + "cv1.bn.weight",   None)
        m[dj + "cv1.norm.bias"]   = (sj + "cv1.bn.bias",     None)
        m[dj + "cv2.conv.weight"] = (sj + "cv2.conv.weight", None)
        m[dj + "cv2.norm.weight"] = (sj + "cv2.bn.weight",   None)
        m[dj + "cv2.norm.bias"]   = (sj + "cv2.bn.bias",     None)
    m["projector.final_norm.weight"] = (PJ + "1.weight", None)
    m["projector.final_norm.bias"]   = (PJ + "1.bias",   None)

    # ---- Two-stage groups ----
    for g in range(n_groups):
        gs = str(g)
        m[f"two_stage.enc_output.{gs}.weight"]      = (TF + f"enc_output.{gs}.weight", None)
        m[f"two_stage.enc_output.{gs}.bias"]        = (TF + f"enc_output.{gs}.bias",   None)
        m[f"two_stage.enc_output_norm.{gs}.weight"] = (TF + f"enc_output_norm.{gs}.weight", None)
        m[f"two_stage.enc_output_norm.{gs}.bias"]   = (TF + f"enc_output_norm.{gs}.bias",   None)
        m[f"two_stage.enc_out_class_embed.{gs}.weight"] = (TF + f"enc_out_class_embed.{gs}.weight", None)
        m[f"two_stage.enc_out_class_embed.{gs}.bias"]   = (TF + f"enc_out_class_embed.{gs}.bias",   None)
        for j in range(3):
            m[f"two_stage.enc_out_bbox_embed.{gs}.layers.{j}.weight"] = (
                TF + f"enc_out_bbox_embed.{gs}.layers.{j}.weight", None)
            m[f"two_stage.enc_out_bbox_embed.{gs}.layers.{j}.bias"] = (
                TF + f"enc_out_bbox_embed.{gs}.layers.{j}.bias",   None)

    # ---- Decoder queries (slice to group 0) ----
    m["decoder.queries.feat"]      = ("query_feat.weight",      "slice_first_300")
    m["decoder.queries.refpoints"] = ("refpoint_embed.weight",  "slice_first_300")

    # ---- Decoder ref_point_head ----
    m["decoder.ref_point_head.layers.0.weight"] = (TF + "decoder.ref_point_head.layers.0.weight", None)
    m["decoder.ref_point_head.layers.0.bias"]   = (TF + "decoder.ref_point_head.layers.0.bias",   None)
    m["decoder.ref_point_head.layers.1.weight"] = (TF + "decoder.ref_point_head.layers.1.weight", None)
    m["decoder.ref_point_head.layers.1.bias"]   = (TF + "decoder.ref_point_head.layers.1.bias",   None)

    # ---- Decoder layers ----
    for i in range(dec_layers):
        src = TF + f"decoder.layers.{i}."
        dst = f"decoder.layers.{i}."
        m[dst + "self_attn.in_proj.weight"]  = (src + "self_attn.in_proj_weight", None)
        m[dst + "self_attn.in_proj.bias"]    = (src + "self_attn.in_proj_bias",   None)
        m[dst + "self_attn.out_proj.weight"] = (src + "self_attn.out_proj.weight", None)
        m[dst + "self_attn.out_proj.bias"]   = (src + "self_attn.out_proj.bias",   None)
        m[dst + "norm1.weight"] = (src + "norm1.weight", None)
        m[dst + "norm1.bias"]   = (src + "norm1.bias",   None)
        m[dst + "cross_attn.sampling_offsets.weight"]  = (src + "cross_attn.sampling_offsets.weight",  None)
        m[dst + "cross_attn.sampling_offsets.bias"]    = (src + "cross_attn.sampling_offsets.bias",    None)
        m[dst + "cross_attn.attention_weights.weight"] = (src + "cross_attn.attention_weights.weight", None)
        m[dst + "cross_attn.attention_weights.bias"]   = (src + "cross_attn.attention_weights.bias",   None)
        m[dst + "cross_attn.value_proj.weight"]        = (src + "cross_attn.value_proj.weight",        None)
        m[dst + "cross_attn.value_proj.bias"]          = (src + "cross_attn.value_proj.bias",          None)
        m[dst + "cross_attn.output_proj.weight"]       = (src + "cross_attn.output_proj.weight",       None)
        m[dst + "cross_attn.output_proj.bias"]         = (src + "cross_attn.output_proj.bias",         None)
        m[dst + "norm2.weight"] = (src + "norm2.weight", None)
        m[dst + "norm2.bias"]   = (src + "norm2.bias",   None)
        m[dst + "linear1.weight"] = (src + "linear1.weight", None)
        m[dst + "linear1.bias"]   = (src + "linear1.bias",   None)
        m[dst + "linear2.weight"] = (src + "linear2.weight", None)
        m[dst + "linear2.bias"]   = (src + "linear2.bias",   None)
        m[dst + "norm3.weight"] = (src + "norm3.weight", None)
        m[dst + "norm3.bias"]   = (src + "norm3.bias",   None)

    # ---- Decoder final norm ----
    m["decoder.norm.weight"] = (TF + "decoder.norm.weight", None)
    m["decoder.norm.bias"]   = (TF + "decoder.norm.bias",   None)

    # ---- Heads (shared) ----
    m["heads.class_embed.weight"]         = ("class_embed.weight", None)
    m["heads.class_embed.bias"]           = ("class_embed.bias",   None)
    m["heads.bbox_embed.layers.0.weight"] = ("bbox_embed.layers.0.weight", None)
    m["heads.bbox_embed.layers.0.bias"]   = ("bbox_embed.layers.0.bias",   None)
    m["heads.bbox_embed.layers.1.weight"] = ("bbox_embed.layers.1.weight", None)
    m["heads.bbox_embed.layers.1.bias"]   = ("bbox_embed.layers.1.bias",   None)
    m["heads.bbox_embed.layers.2.weight"] = ("bbox_embed.layers.2.weight", None)
    m["heads.bbox_embed.layers.2.bias"]   = ("bbox_embed.layers.2.bias",   None)

    return m


def validate(state_dict, name_map):
    """Return (missing_sources, unused_keys)."""
    sd_keys = set(state_dict.keys())
    missing = []
    used = set()
    for gguf_name, (src_key, _t) in name_map.items():
        if src_key not in sd_keys:
            missing.append((gguf_name, src_key))
        else:
            used.add(src_key)
    unused = sorted(sd_keys - used)
    return missing, unused


def apply_transform(tensor, transform_tag):
    import numpy as np
    if transform_tag is None:
        return tensor
    if transform_tag == "slice_first_300":
        return tensor[:300]
    if transform_tag == "squeeze_cls":
        # (1, 1, dim) -> (dim,)
        return tensor.reshape(-1)
    if transform_tag == "squeeze_pos":
        # (1, n_tokens, dim) -> (n_tokens, dim)
        return tensor[0]
    raise ValueError(f"unknown transform tag: {transform_tag}")


def should_quantize(gguf_name, arr):
    """Block-quant heuristic: only 2D weights with both dims >= 64.

    Excludes:
      * LayerNorm weights/biases and all biases (1D)
      * Conv kernels (4D)
      * Embeddings (cls_token, pos_embed, query_feat, refpoints) — 2D but used
        as raw lookups, not as multiplicands in `mul_mat`
      * Tiny linear weights (e.g. bbox head's final 256->4 layer) — small enough
        that quantizing them barely saves bytes but hurts precision

    Applies uniformly to q4_0 / q4_1 / q5_0 / q5_1 / q8_0 — all use the same
    32-element block layout and the same exclusion rules.
    """
    if not gguf_name.endswith(".weight"):
        return False
    if arr.ndim != 2:
        return False
    if arr.shape[0] < 64 or arr.shape[1] < 64:
        return False
    # Embeddings: 2D but not used in mul_mat — they're indexed/broadcast
    skiplist = {
        "backbone.pos_embed",
        "decoder.queries.feat",
        "decoder.queries.refpoints",
    }
    if gguf_name in skiplist:
        return False
    # 32-element blocks require the innermost dim divisible by 32. In numpy
    # row-major view, shape[-1] is the innermost; gguf reverses, so the
    # logical innermost dim on disk corresponds to the first numpy axis.
    # In practice all rfdetr 2D weights pass this check; guard anyway.
    if arr.shape[-1] % 32 != 0:
        return False
    return True


# Backward-compatibility alias (older imports may reference this).
should_quantize_q8_0 = should_quantize


# Mapping from --dtype string -> gguf.GGMLQuantizationType. Populated lazily
# inside write_gguf so importing the module doesn't require gguf installed.
QUANT_DTYPES = {"q4_0", "q4_1", "q5_0", "q5_1", "q8_0"}


def write_gguf(out_path, variant_name, variant_cfg, name_map, state_dict,
               dtype_str, num_classes=None, class_names=None):
    """Write the GGUF.

    num_classes / class_names default to the COCO-91 baked into variant_cfg
    (preserves behavior for pretrained models). For fine-tuned checkpoints
    main() derives `num_classes` from class_embed.weight.shape[0] and passes
    a class-count-matched class_names list (auto-generated "class_<idx>" if
    no names are available).
    """
    import numpy as np
    import gguf as gguf_mod

    # Per-dtype mapping table:
    #   default_np_dtype / default_ggml_dtype apply to *unquantized* tensors
    #   (biases, norms, embeddings, conv kernels, etc.). For quantized modes,
    #   those stay F32 — only 2D linear weights get the block quantizer.
    quant_dtype_map = {
        "q4_0": gguf_mod.GGMLQuantizationType.Q4_0,
        "q4_1": gguf_mod.GGMLQuantizationType.Q4_1,
        "q5_0": gguf_mod.GGMLQuantizationType.Q5_0,
        "q5_1": gguf_mod.GGMLQuantizationType.Q5_1,
        "q8_0": gguf_mod.GGMLQuantizationType.Q8_0,
    }

    if dtype_str == "f16":
        default_np_dtype = np.float16
        default_ggml_dtype = gguf_mod.GGMLQuantizationType.F16
        quant_ggml_dtype = None
    elif dtype_str == "f32":
        default_np_dtype = np.float32
        default_ggml_dtype = gguf_mod.GGMLQuantizationType.F32
        quant_ggml_dtype = None
    elif dtype_str in QUANT_DTYPES:
        default_np_dtype = np.float32
        default_ggml_dtype = gguf_mod.GGMLQuantizationType.F32
        quant_ggml_dtype = quant_dtype_map[dtype_str]
    else:
        raise ValueError(f"unknown --dtype: {dtype_str}")

    writer = gguf_mod.GGUFWriter(out_path, arch="rfdetr")

    # Resolve num_classes / class_names: caller-supplied overrides win,
    # else fall back to the variant_cfg COCO-91 defaults.
    effective_num_classes = num_classes if num_classes is not None else variant_cfg["num_classes"]
    if class_names is None:
        if effective_num_classes == 91:
            effective_class_names = COCO_CLASS_NAMES
        else:
            # Custom-class checkpoint with no name table — synthesize placeholders.
            effective_class_names = [f"class_{i}" for i in range(effective_num_classes)]
    else:
        if len(class_names) != effective_num_classes:
            raise ValueError(f"class_names length ({len(class_names)}) != "
                             f"num_classes ({effective_num_classes})")
        effective_class_names = list(class_names)

    # --- Top-level metadata ---
    writer.add_string("rfdetr.format.version", FORMAT_VERSION)
    writer.add_string("rfdetr.variant",         variant_name)
    writer.add_uint32("rfdetr.image_size",      variant_cfg["image_size"])
    writer.add_uint32("rfdetr.patch_size",      variant_cfg["patch_size"])
    writer.add_uint32("rfdetr.num_queries",     variant_cfg["num_queries"])
    writer.add_uint32("rfdetr.group_detr",      variant_cfg["group_detr"])
    writer.add_uint32("rfdetr.num_classes",     effective_num_classes)

    # class_names: N-slot string array, where N == num_classes.
    # gguf wants a Sequence[str].
    writer.add_array("rfdetr.class_names", effective_class_names)

    # preprocess
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    writer.add_array("rfdetr.preprocess.mean", mean.tolist())
    writer.add_array("rfdetr.preprocess.std",  std.tolist())

    # backbone
    bb = variant_cfg["backbone"]
    writer.add_uint32("rfdetr.backbone.dim",                  bb["dim"])
    writer.add_uint32("rfdetr.backbone.depth",                bb["depth"])
    writer.add_uint32("rfdetr.backbone.heads",                bb["heads"])
    writer.add_uint32("rfdetr.backbone.ffn_dim",              bb["ffn_dim"])
    writer.add_uint32("rfdetr.backbone.num_windows",          bb["num_windows"])
    writer.add_array ("rfdetr.backbone.global_attn_indices",  bb["global_attn_indices"])
    writer.add_array ("rfdetr.backbone.out_feature_indices",  bb["out_feature_indices"])
    writer.add_uint32("rfdetr.backbone.pos_embed_train_size", bb["pos_embed_train_size"])

    # projector
    pj = variant_cfg["projector"]
    writer.add_uint32("rfdetr.projector.in_dim",         pj["in_dim"])
    writer.add_uint32("rfdetr.projector.out_dim",        pj["out_dim"])
    writer.add_uint32("rfdetr.projector.bottleneck_dim", pj["bottleneck_dim"])
    writer.add_uint32("rfdetr.projector.n_bottlenecks",  pj["n_bottlenecks"])

    # decoder
    dc = variant_cfg["decoder"]
    writer.add_uint32("rfdetr.decoder.layers",              dc["layers"])
    writer.add_uint32("rfdetr.decoder.model_dim",           dc["model_dim"])
    writer.add_uint32("rfdetr.decoder.ffn_dim",             dc["ffn_dim"])
    writer.add_uint32("rfdetr.decoder.self_attn_heads",     dc["self_attn_heads"])
    writer.add_uint32("rfdetr.decoder.cross_attn_heads",    dc["cross_attn_heads"])
    writer.add_uint32("rfdetr.decoder.cross_attn_n_levels", dc["cross_attn_n_levels"])
    writer.add_uint32("rfdetr.decoder.cross_attn_n_points", dc["cross_attn_n_points"])

    # two-stage
    writer.add_uint32("rfdetr.two_stage.n_groups", variant_cfg["two_stage"]["n_groups"])

    # --- Tensors ---
    n_written = 0
    n_quantized = 0
    for gguf_name, (src_key, transform) in name_map.items():
        t = state_dict[src_key]
        # torch -> numpy
        arr = t.detach().cpu().numpy()
        arr = apply_transform(arr, transform)
        arr = np.ascontiguousarray(arr)

        if quant_ggml_dtype is not None and should_quantize(gguf_name, arr):
            # Pre-quantize to block-quant bytes. The gguf writer infers the
            # logical F32 shape from the uint8 byte shape via
            # quant_shape_from_byte_shape.
            arr_f32 = arr.astype(np.float32, copy=False)
            q_bytes = gguf_mod.quants.quantize(arr_f32, quant_ggml_dtype)
            writer.add_tensor(
                gguf_name, q_bytes,
                raw_dtype=quant_ggml_dtype,
            )
            n_quantized += 1
        else:
            arr = arr.astype(default_np_dtype, copy=False)
            writer.add_tensor(gguf_name, arr, raw_dtype=default_ggml_dtype)
        n_written += 1

    if quant_ggml_dtype is not None:
        print(f"[quantize] {dtype_str} tensors: {n_quantized}/{n_written}",
              file=sys.stderr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()

    return n_written


def main() -> int:
    args = parse_args()
    if args.output is None:
        args.output = f"rfdetr-{args.variant}-{args.dtype}.gguf"

    try:
        import torch  # noqa: F401
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

    print(f"Loading rfdetr-{args.variant} ...", file=sys.stderr)

    if args.checkpoint:
        # Load a local fine-tuned checkpoint. The rfdetr saver writes
        # {"model": state_dict, "args": {"num_classes": N, ...}, ...}.
        #
        # Convention reminder: rfdetr's "num_classes" is the *logical* class
        # count without the background slot, but class_embed.bias.shape[0]
        # equals num_classes + 1 (background occupies slot 0). We read the
        # raw head size from the checkpoint tensor (truth source) and resize
        # the model's classification head to match BEFORE load_state_dict so
        # the shapes line up.
        print(f"[checkpoint] loading {args.checkpoint}", file=sys.stderr)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            print("error: checkpoint must be a dict with a 'model' state_dict key.",
                  file=sys.stderr)
            return 4
        state = ckpt["model"]
        if "class_embed.bias" not in state:
            print("error: checkpoint state_dict lacks class_embed.bias — cannot resize head.",
                  file=sys.stderr)
            return 4
        head_size = int(state["class_embed.bias"].shape[0])
        # Construct the model with default num_classes, then resize the head
        # to match the checkpoint's actual head_size. This avoids the +1
        # background-slot ambiguity around RFDETRBase(num_classes=N).
        rfdetr_model = _VARIANT_CLASSES[args.variant]()
        rfdetr_model.model.model.reinitialize_detection_head(head_size)
        rfdetr_model.model.model.load_state_dict(state, strict=False)
        print(f"[checkpoint] loaded head_size={head_size}", file=sys.stderr)
    else:
        rfdetr_model = _VARIANT_CLASSES[args.variant]()

    inner = rfdetr_model.model.model
    sd = inner.state_dict()

    print(f"[probe] inner module: {type(inner).__name__}", file=sys.stderr)
    print(f"[probe] state_dict entries: {len(sd)}", file=sys.stderr)

    variant_cfg = VARIANTS[args.variant]
    name_map    = build_tensor_name_map(variant_cfg)

    # Derive num_classes from the live class_embed weight (truth source).
    # This works for both the COCO pretrained models (91) and fine-tunes.
    class_embed_w = sd.get("class_embed.weight")
    if class_embed_w is None:
        print("error: class_embed.weight missing from state_dict", file=sys.stderr)
        return 5
    actual_num_classes = int(class_embed_w.shape[0])
    print(f"[convert] num_classes={actual_num_classes}, "
          f"class_embed shape={tuple(class_embed_w.shape)}", file=sys.stderr)

    missing, unused = validate(sd, name_map)
    expected_count  = len(name_map)
    print(f"[validate] expected v2 tensors: {expected_count}", file=sys.stderr)
    print(f"[validate] missing (source key not in state_dict): {len(missing)}", file=sys.stderr)
    if missing:
        print("MISSING (will block conversion):", file=sys.stderr)
        for gname, skey in missing[:40]:
            print(f"  {gname:60s} <- {skey}", file=sys.stderr)
        if len(missing) > 40:
            print(f"  ... and {len(missing) - 40} more", file=sys.stderr)
        return 3
    print(f"[validate] unused state_dict keys (dropped): {len(unused)}", file=sys.stderr)
    if unused:
        for k in unused[:10]:
            print(f"  - {k}", file=sys.stderr)
        if len(unused) > 10:
            print(f"  ... and {len(unused) - 10} more", file=sys.stderr)

    if args.dry_run:
        print("\n(--dry-run) Not writing GGUF.", file=sys.stderr)
        return 0

    print(f"\nWriting GGUF -> {args.output} (dtype={args.dtype}) ...", file=sys.stderr)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_written = write_gguf(args.output, args.variant, variant_cfg, name_map, sd,
                           args.dtype, num_classes=actual_num_classes)
    print(f"[done] wrote {n_written} tensors to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
