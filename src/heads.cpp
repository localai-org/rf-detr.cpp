#include "heads.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <string>

namespace rfdetr {

namespace {

ggml_tensor* fetch(const Model& m, const std::string& name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "heads: missing tensor '%s'", name.c_str());
        return nullptr;
    }
    return it->second;
}

ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x,
                    ggml_tensor* W, ggml_tensor* b) {
    ggml_tensor* y = ggml_mul_mat(ctx, W, x);
    y = ggml_add(ctx, y, b);
    return y;
}

}  // namespace

ggml_tensor* class_head_forward(ggml_context* ctx, const Model& m,
                                ggml_tensor* decoder_out) {
    ggml_tensor* W = fetch(m, "heads.class_embed.weight");
    ggml_tensor* b = fetch(m, "heads.class_embed.bias");
    if (!W || !b) return nullptr;

    ggml_tensor* logits = linear(ctx, decoder_out, W, b);
    publish("heads.class_logits", logits);
    return logits;
}

ggml_tensor* bbox_head_forward(ggml_context* ctx, const Model& m,
                               ggml_tensor* decoder_out) {
    ggml_tensor* W0 = fetch(m, "heads.bbox_embed.layers.0.weight");
    ggml_tensor* b0 = fetch(m, "heads.bbox_embed.layers.0.bias");
    ggml_tensor* W1 = fetch(m, "heads.bbox_embed.layers.1.weight");
    ggml_tensor* b1 = fetch(m, "heads.bbox_embed.layers.1.bias");
    ggml_tensor* W2 = fetch(m, "heads.bbox_embed.layers.2.weight");
    ggml_tensor* b2 = fetch(m, "heads.bbox_embed.layers.2.bias");
    if (!W0 || !b0 || !W1 || !b1 || !W2 || !b2) return nullptr;

    ggml_tensor* h = linear(ctx, decoder_out, W0, b0);
    h = ggml_relu(ctx, h);
    h = linear(ctx, h, W1, b1);
    h = ggml_relu(ctx, h);
    ggml_tensor* delta = linear(ctx, h, W2, b2);
    publish("heads.bbox_pred", delta);
    return delta;
}

}  // namespace rfdetr
