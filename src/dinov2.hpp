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

/* Run the full DINOv2 backbone: patch_embed → CLS+pos_embed → N blocks →
 * final_norm. Publishes every per-block, multi-scale, and final checkpoint
 * via the trace callback. Returns the final post-norm tensor (dim, N+1, 1, 1)
 * F32.
 *
 * Multi-scale features are NOT returned explicitly. Plan 6's projector can
 * either re-run the backbone with its own trace callback, or this API can
 * grow a structured return type. Plan 4 keeps the API minimal. */
ggml_tensor* dinov2_forward(ggml_context* ctx, const Model& m,
                            ggml_tensor* input);

}  // namespace rfdetr

#endif
