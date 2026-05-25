#ifndef RFDETR_DINOV2_HPP
#define RFDETR_DINOV2_HPP

#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace rfdetr {

/* Build the patch_embed forward graph node.
 *
 * Input:  `input` — (1, H, W, 3) F32 image already mean/std normalized
 * Output: a token tensor (1, N_patches, dim) F32, where
 *           N_patches = (H / 14) * (W / 14)
 *
 * Publishes "backbone.patch_embed.output" via the trace callback. */
ggml_tensor* dinov2_patch_embed(ggml_context* ctx, const Model& m,
                                ggml_tensor* input);

/* Build one DINOv2 transformer block (pre-LN style):
 *
 *   x = x + attn(norm1(x))
 *   x = x + mlp(norm2(x))
 *
 * Input/output shape: (1, N_patches, dim) F32.
 *
 * Publishes:
 *   "backbone.block.{idx}.norm1.output"
 *   "backbone.block.{idx}.attn.output"
 *   "backbone.block.{idx}.mlp.output"
 *   "backbone.block.{idx}.output"
 *
 * Plan 3 uses global self-attention (window_size ignored). Plan 4 adds
 * window-attention switching. */
ggml_tensor* dinov2_block(ggml_context* ctx, const Model& m,
                          ggml_tensor* x, int block_idx);

}  // namespace rfdetr

#endif
