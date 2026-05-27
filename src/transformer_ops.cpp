#include "transformer_ops.hpp"

#include "ggml.h"

namespace rfdetr::ops {

/* LayerNorm: x_normalized = (x - mean) / sqrt(var + eps) * weight + bias.
 *
 * ggml_norm normalizes along ne[0] (the contiguous / "row" axis), which is
 * exactly the channel/feature dim for our (dim, N, 1, 1) layout. The weight
 * and bias are 1-D tensors of size `dim` (ne = (dim, 1, 1, 1)) and broadcast
 * over the token axis via ggml_can_repeat.
 *
 * eps = 1e-5 matches PyTorch's default torch.nn.LayerNorm. */
ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* weight, ggml_tensor* bias) {
    constexpr float eps = 1e-5f;
    ggml_tensor* y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, weight);
    y = ggml_add(ctx, y, bias);
    return y;
}

}  // namespace rfdetr::ops
