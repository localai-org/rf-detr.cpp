#ifndef RFDETR_ENCODER_HPP
#define RFDETR_ENCODER_HPP

#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace rfdetr {

/* One transformer encoder layer (pre-LN):
 *   x = x + attn(norm1(x))   // self-attention
 *   x = x + mlp(norm2(x))    // feed-forward
 *
 * Input/output ne = (encoder.model_dim, N_tokens, 1, 1) where N_tokens =
 * 4 * N_patches from the projector.
 *
 * Publishes:
 *   encoder.layer{idx}.norm1.output
 *   encoder.layer{idx}.attn.output
 *   encoder.layer{idx}.mlp.output
 *   encoder.layer{idx}.output */
ggml_tensor* encoder_layer(ggml_context* ctx, const Model& m,
                           ggml_tensor* x, int layer_idx);

/* encoder_forward declared here; implementation lands in Task 4. */
ggml_tensor* encoder_forward(ggml_context* ctx, const Model& m,
                             ggml_tensor* x);

}  // namespace rfdetr

#endif
