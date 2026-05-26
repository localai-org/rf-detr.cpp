#ifndef RFDETR_DINOV2_HPP
#define RFDETR_DINOV2_HPP

#include "model_loader.hpp"

#include <array>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace rfdetr {

/* Outputs of the windowed DINOv2 backbone.
 *
 * Each `multi_scale[j]` is a spatial feature map produced by tapping the
 * windowed block output at out_feature_indices[j], applying the final
 * LayerNorm, stripping the per-window CLS token, un-windowing, and reshaping
 * to image-grid form.
 *
 * Shapes (ggml ne convention — ne[0] is the fastest axis):
 *   multi_scale[j]: (W_patches, H_patches, dim, 1)
 *                 = (40, 40, 384, 1) for rfdetr-base @ 560
 *
 * `final` is published for parity but is the same data the projector reads via
 * multi_scale[3] (last tap). It is the LayerNorm of the last block's output
 * in windowed token form: ne = (dim, tokens_per_window, n_windows, 1)
 *                       = (384, 101, 16, 1) for rfdetr-base. */
struct BackboneOutput {
    ggml_tensor* final = nullptr;
    std::array<ggml_tensor*, 4> multi_scale{nullptr, nullptr, nullptr, nullptr};
};

/* Run the windowed DINOv2 backbone for rfdetr-base.
 *
 *   input ne = (W, H, 3, 1) F32 — already mean/std normalized
 *
 * Pipeline:
 *   1. patch_embed (Conv2d k=14 s=14)               → (W_p, H_p, dim, 1)
 *   2. flatten / transpose to tokens                 → (dim, N_patches, 1, 1)
 *   3. prepend CLS                                   → (dim, N_patches+1, 1, 1)
 *   4. add bicubic-interpolated pos_embed
 *   5. window-partition (CLS broadcast)              → (dim, T, n_windows, 1)
 *      T = tokens_per_window = (W_p/num_windows)^2 + 1
 *   6. for each block i in [0, depth):
 *      6a. norm1, attention (windowed if i in window_block_indexes else global)
 *      6b. * layer_scale1, + residual
 *      6c. norm2, mlp, * layer_scale2, + residual
 *      Block output ne = (dim, T, n_windows, 1).
 *      If i in out_feature_indices:
 *        - apply final LayerNorm
 *        - strip CLS
 *        - un-window to (W_patches, H_patches, dim, 1)
 *        - stash as multi_scale[level]
 *   7. final LayerNorm of last block → BackboneOutput.final
 *
 * Trace callbacks published:
 *   backbone.patch_embed.output           — post window-partition embeddings
 *                                           ne = (dim, T, n_windows, 1)
 *   backbone.block.{i}.output (i = 0..11) — windowed block output
 *                                           ne = (dim, T, n_windows, 1)
 *   backbone.norm.output                  — final LN of last block's output
 *                                           ne = (dim, T, n_windows, 1)
 *   backbone.multiscale.level{0..3}       — image-grid feature maps
 *                                           ne = (W_p, H_p, dim, 1)
 */
BackboneOutput dinov2_forward(ggml_context* ctx, const Model& m,
                              ggml_tensor* input);

}  // namespace rfdetr

#endif
