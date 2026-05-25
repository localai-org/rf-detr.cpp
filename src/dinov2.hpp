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

/* Concatenate the learnable CLS token to the front of the patch tokens and
 * add the positional embedding.
 *
 * Input:  `tokens` — (dim, N_patches, 1, 1) F32
 * Output: a tensor of shape (dim, N_patches + 1, 1, 1) F32 with CLS at
 *         index 0 and positional offsets added to every position.
 *
 * Publishes "backbone.cls_pos_embed.output" via the trace callback. */
ggml_tensor* dinov2_add_cls_and_pos_embed(ggml_context* ctx, const Model& m,
                                          ggml_tensor* tokens);

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

/* Apply the final backbone LayerNorm (after the last block).
 *
 * Input/output: (dim, N+1, 1, 1) F32.
 *
 * Publishes "backbone.norm.output" via the trace callback. */
ggml_tensor* dinov2_final_norm(ggml_context* ctx, const Model& m,
                               ggml_tensor* x);

}  // namespace rfdetr

#endif
