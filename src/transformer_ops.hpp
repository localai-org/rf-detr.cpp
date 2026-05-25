#ifndef RFDETR_TRANSFORMER_OPS_HPP
#define RFDETR_TRANSFORMER_OPS_HPP

struct ggml_context;
struct ggml_tensor;

namespace rfdetr::ops {

/* LayerNorm with affine: y = (x - mean) / sqrt(var + eps) * weight + bias.
 * Operates on the last axis (ggml_norm normalizes along ne[0]). eps = 1e-5
 * (PyTorch torch.nn.LayerNorm default). */
ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* weight, ggml_tensor* bias);

/* Multi-head self-attention with packed QKV projection (global, no windowing).
 *
 *   Input x ne = (dim, N, 1, 1)
 *   Wqkv ne   = (dim, 3*dim)
 *   bqkv ne   = (3*dim,)
 *   Wproj ne  = (dim, dim)
 *   bproj ne  = (dim,)
 *
 * Output ne = (dim, N, 1, 1). */
ggml_tensor* mha(ggml_context* ctx, ggml_tensor* x,
                 ggml_tensor* Wqkv, ggml_tensor* bqkv,
                 ggml_tensor* Wproj, ggml_tensor* bproj,
                 int n_heads);

/* Position-wise feed-forward network: x -> fc1 -> erf-GELU -> fc2.
 *
 *   W1 ne = (dim, ffn_dim); b1 ne = (ffn_dim,)
 *   W2 ne = (ffn_dim, dim); b2 ne = (dim,)
 *
 * Output ne = same as input. */
ggml_tensor* mlp(ggml_context* ctx, ggml_tensor* x,
                 ggml_tensor* W1, ggml_tensor* b1,
                 ggml_tensor* W2, ggml_tensor* b2);

}  // namespace rfdetr::ops

#endif
