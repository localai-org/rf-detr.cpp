#include "heads.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <string>

namespace rfdetr {

namespace {

ggml_tensor* get_tensor(const Model& m, const std::string& name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "heads: missing tensor '%s'", name.c_str());
        return nullptr;
    }
    return it->second;
}

ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x,
                    ggml_tensor* W, ggml_tensor* b) {
    /* W @ x + bias */
    ggml_tensor* y = ggml_mul_mat(ctx, W, x);
    y = ggml_add(ctx, y, b);
    return y;
}

}  // namespace

ggml_tensor* class_head_forward(ggml_context* ctx, const Model& m,
                                ggml_tensor* decoder_out) {
    ggml_tensor* W = get_tensor(m, "heads.class.fc.weight");
    ggml_tensor* b = get_tensor(m, "heads.class.fc.bias");
    if (!W || !b) return nullptr;

    ggml_tensor* logits = linear(ctx, decoder_out, W, b);
    publish("heads.class.logits", logits);
    return logits;
}

ggml_tensor* bbox_head_forward(ggml_context* ctx, const Model& m,
                               ggml_tensor* decoder_out) {
    ggml_tensor* W1 = get_tensor(m, "heads.bbox.fc1.weight");
    ggml_tensor* b1 = get_tensor(m, "heads.bbox.fc1.bias");
    ggml_tensor* W2 = get_tensor(m, "heads.bbox.fc2.weight");
    ggml_tensor* b2 = get_tensor(m, "heads.bbox.fc2.bias");
    ggml_tensor* W3 = get_tensor(m, "heads.bbox.fc3.weight");
    ggml_tensor* b3 = get_tensor(m, "heads.bbox.fc3.bias");
    if (!W1 || !b1 || !W2 || !b2 || !W3 || !b3) return nullptr;

    ggml_tensor* h = linear(ctx, decoder_out, W1, b1);
    h = ggml_relu(ctx, h);
    publish("heads.bbox.fc1.output", h);

    h = linear(ctx, h, W2, b2);
    h = ggml_relu(ctx, h);
    publish("heads.bbox.fc2.output", h);

    h = linear(ctx, h, W3, b3);
    publish("heads.bbox.fc3.output", h);

    /* ggml v0.13.0 ships ggml_sigmoid (verified in ggml.h:1150). */
    ggml_tensor* pred = ggml_sigmoid(ctx, h);
    publish("heads.bbox.pred", pred);
    return pred;
}

}  // namespace rfdetr
