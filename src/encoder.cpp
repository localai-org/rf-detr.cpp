#include "encoder.hpp"
#include "transformer_ops.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <string>

namespace rfdetr {

ggml_tensor* encoder_layer(ggml_context* ctx, const Model& m,
                           ggml_tensor* x, int layer_idx) {
    const std::string p = "encoder.layers." + std::to_string(layer_idx) + ".";
    auto get = [&](const char* suffix) -> ggml_tensor* {
        auto it = m.tensors.find(p + suffix);
        if (it == m.tensors.end()) {
            rfdetr_logf(RFDETR_LOG_ERROR, "encoder_layer: missing tensor '%s'",
                        (p + suffix).c_str());
            return nullptr;
        }
        return it->second;
    };

    ggml_tensor* n1w  = get("norm1.weight");
    ggml_tensor* n1b  = get("norm1.bias");
    ggml_tensor* qkvW = get("self_attn.qkv.weight");
    ggml_tensor* qkvB = get("self_attn.qkv.bias");
    ggml_tensor* prW  = get("self_attn.out.weight");
    ggml_tensor* prB  = get("self_attn.out.bias");
    ggml_tensor* n2w  = get("norm2.weight");
    ggml_tensor* n2b  = get("norm2.bias");
    ggml_tensor* f1W  = get("ffn.fc1.weight");
    ggml_tensor* f1B  = get("ffn.fc1.bias");
    ggml_tensor* f2W  = get("ffn.fc2.weight");
    ggml_tensor* f2B  = get("ffn.fc2.bias");
    if (!n1w || !n1b || !qkvW || !qkvB || !prW || !prB ||
        !n2w || !n2b || !f1W || !f1B || !f2W || !f2B) {
        return nullptr;
    }

    const std::string pub = "encoder.layer" + std::to_string(layer_idx) + ".";

    /* x = x + attn(norm1(x)) */
    ggml_tensor* y = ops::layer_norm(ctx, x, n1w, n1b);
    publish(pub + "norm1.output", y);
    y = ops::mha(ctx, y, qkvW, qkvB, prW, prB, (int)m.config.encoder.heads);
    publish(pub + "attn.output", y);
    x = ggml_add(ctx, x, y);

    /* x = x + mlp(norm2(x)) */
    y = ops::layer_norm(ctx, x, n2w, n2b);
    ggml_tensor* mlp_out = ops::mlp(ctx, y, f1W, f1B, f2W, f2B);
    publish(pub + "mlp.output", mlp_out);
    x = ggml_add(ctx, x, mlp_out);

    publish(pub + "output", x);
    return x;
}

ggml_tensor* encoder_forward(ggml_context* ctx, const Model& m,
                             ggml_tensor* x) {
    for (uint32_t i = 0; i < m.config.encoder.layers; ++i) {
        x = encoder_layer(ctx, m, x, (int)i);
        if (!x) return nullptr;
    }
    publish("encoder.output", x);
    return x;
}

}  // namespace rfdetr
