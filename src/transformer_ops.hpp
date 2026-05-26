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

}  // namespace rfdetr::ops

#endif
