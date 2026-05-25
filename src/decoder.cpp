#include "decoder.hpp"
#include "transformer_ops.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <string>

namespace rfdetr {

ggml_tensor* decoder_layer(ggml_context* ctx, const Model& m,
                           ggml_tensor* q, ggml_tensor* encoder_out,
                           int layer_idx) {
    const std::string p = "decoder.layers." + std::to_string(layer_idx) + ".";
    auto get = [&](const char* suffix) -> ggml_tensor* {
        auto it = m.tensors.find(p + suffix);
        if (it == m.tensors.end()) {
            rfdetr_logf(RFDETR_LOG_ERROR, "decoder_layer: missing tensor '%s'",
                        (p + suffix).c_str());
            return nullptr;
        }
        return it->second;
    };

    /* Self-attention weights (packed QKV like encoder) */
    ggml_tensor* n1w   = get("norm1.weight");
    ggml_tensor* n1b   = get("norm1.bias");
    ggml_tensor* sqkvW = get("self_attn.qkv.weight");
    ggml_tensor* sqkvB = get("self_attn.qkv.bias");
    ggml_tensor* soW   = get("self_attn.out.weight");
    ggml_tensor* soB   = get("self_attn.out.bias");

    /* Cross-attention weights (separate Q vs packed KV) */
    ggml_tensor* n2w   = get("norm2.weight");
    ggml_tensor* n2b   = get("norm2.bias");
    ggml_tensor* cqW   = get("cross_attn.q.weight");
    ggml_tensor* cqB   = get("cross_attn.q.bias");
    ggml_tensor* ckvW  = get("cross_attn.kv.weight");
    ggml_tensor* ckvB  = get("cross_attn.kv.bias");
    ggml_tensor* coW   = get("cross_attn.out.weight");
    ggml_tensor* coB   = get("cross_attn.out.bias");

    /* MLP weights */
    ggml_tensor* n3w   = get("norm3.weight");
    ggml_tensor* n3b   = get("norm3.bias");
    ggml_tensor* f1W   = get("ffn.fc1.weight");
    ggml_tensor* f1B   = get("ffn.fc1.bias");
    ggml_tensor* f2W   = get("ffn.fc2.weight");
    ggml_tensor* f2B   = get("ffn.fc2.bias");

    if (!n1w || !n1b || !sqkvW || !sqkvB || !soW || !soB ||
        !n2w || !n2b || !cqW || !cqB || !ckvW || !ckvB || !coW || !coB ||
        !n3w || !n3b || !f1W || !f1B || !f2W || !f2B) {
        return nullptr;
    }

    const std::string pub = "decoder.layer" + std::to_string(layer_idx) + ".";

    /* q = q + self_attn(norm1(q)) */
    ggml_tensor* y = ops::layer_norm(ctx, q, n1w, n1b);
    y = ops::mha(ctx, y, sqkvW, sqkvB, soW, soB, (int)m.config.decoder.heads);
    publish(pub + "self_attn.output", y);
    q = ggml_add(ctx, q, y);

    /* q = q + cross_attn(norm2(q), encoder_out) */
    y = ops::layer_norm(ctx, q, n2w, n2b);
    y = ops::cross_attn(ctx, y, encoder_out,
                        cqW, cqB, ckvW, ckvB, coW, coB,
                        (int)m.config.decoder.heads);
    publish(pub + "cross_attn.output", y);
    q = ggml_add(ctx, q, y);

    /* q = q + mlp(norm3(q)) */
    y = ops::layer_norm(ctx, q, n3w, n3b);
    ggml_tensor* mlp_out = ops::mlp(ctx, y, f1W, f1B, f2W, f2B);
    publish(pub + "mlp.output", mlp_out);
    q = ggml_add(ctx, q, mlp_out);

    publish(pub + "output", q);
    return q;
}

/* Stub — Task 3 implements. */
ggml_tensor* decoder_forward(ggml_context* /*ctx*/, const Model& /*m*/,
                             ggml_tensor* /*encoder_out*/) {
    return nullptr;
}

}  // namespace rfdetr
