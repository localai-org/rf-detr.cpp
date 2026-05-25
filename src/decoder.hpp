#ifndef RFDETR_DECODER_HPP
#define RFDETR_DECODER_HPP

#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace rfdetr {

/* One DETR-style transformer decoder layer (pre-LN):
 *   q = q + self_attn(norm1(q))
 *   q = q + cross_attn(norm2(q), encoder_out)
 *   q = q + mlp(norm3(q))
 *
 * Input:
 *   q ne           = (model_dim, num_queries, 1, 1)
 *   encoder_out ne = (model_dim, n_enc_tokens, 1, 1)
 *
 * Output: same shape as q.
 *
 * Publishes:
 *   decoder.layer{idx}.self_attn.output
 *   decoder.layer{idx}.cross_attn.output
 *   decoder.layer{idx}.mlp.output
 *   decoder.layer{idx}.output */
ggml_tensor* decoder_layer(ggml_context* ctx, const Model& m,
                           ggml_tensor* q, ggml_tensor* encoder_out,
                           int layer_idx);

/* decoder_forward declared here; implementation lands in Task 3. */
ggml_tensor* decoder_forward(ggml_context* ctx, const Model& m,
                             ggml_tensor* encoder_out);

}  // namespace rfdetr

#endif
