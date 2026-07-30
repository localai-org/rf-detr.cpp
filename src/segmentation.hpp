#ifndef RFDETR_SEGMENTATION_HPP
#define RFDETR_SEGMENTATION_HPP

#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace rfdetr {

/* SegmentationHead forward graph builder.
 *
 * Mirrors `rfdetr.models.heads.segmentation.SegmentationHead.forward` with
 * `skip_blocks=False`:
 *
 *   target = (image_h / r, image_w / r), r = mask_downsample_ratio
 *   spatial = F.interpolate(spatial_features, target, mode="bilinear",
 *                            align_corners=False)
 *   for each (block, qf) in zip(self.blocks, query_features_per_layer):
 *     spatial = block(spatial)
 *     spatial_proj = spatial_features_proj(spatial)     # Conv2d 1x1
 *     qf = query_features_proj(query_features_block(qf))
 *     mask = einsum("bchw,bnc->bnhw", spatial_proj, qf) + bias
 *
 * Inference uses masks[-1] (the last decoder layer's mask). For parity
 * verification we also publish each intermediate via the trace mechanism:
 *   seg.spatial_features.resized
 *   seg.block.{0..n_layers-1}.spatial_out
 *   seg.block.{0..n_layers-1}.spatial_proj
 *   seg.block.{0..n_layers-1}.qf_proj
 *   seg.masks.{0..n_layers-1}
 *   seg.masks.final
 *
 * Inputs (ggml memory layout):
 *   spatial_features    ne = (W, H, C=256, 1)         — projector output BEFORE resize
 *   query_features_per_layer  ne = (C=256, NQ, 1) per layer; pass `n_layers`
 *                             tensors via the array. SegmentationHead.__init__
 *                             constructs exactly one DepthwiseConvBlock per
 *                             decoder layer, so the block count always equals
 *                             n_layers — 4 for nano/small, 5 for medium/large,
 *                             6 for xlarge/2xlarge. If you only have the last
 *                             layer post-norm tensor, pass it n_layers times to
 *                             reproduce the export-time (skip_blocks=False)
 *                             behaviour, but **the resulting masks[-1] will
 *                             NOT match the inference output** because the
 *                             spatial features get processed by n_layers
 *                             different query streams in the real forward.
 *                             For parity pass exactly n_layers per-layer
 *                             tensors.
 *
 * Output (ggml memory layout):
 *   masks_final          ne = (W_mask, H_mask, NQ, 1)  — raw logits
 *
 * Returns nullptr on failure (also logs via rfdetr_logf). */
ggml_tensor* segmentation_forward(
    ggml_context* ctx,
    const Model& m,
    ggml_tensor* spatial_features,
    ggml_tensor* const* query_features_per_layer,
    int n_layers,
    int image_h, int image_w,
    int mask_downsample_ratio);

}  // namespace rfdetr

#endif
