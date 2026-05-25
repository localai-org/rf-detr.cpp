#include "trace.hpp"

#include "ggml.h"

#include <cstring>
#include <stdexcept>

namespace rfdetr {

namespace {
thread_local trace_cb tls_cb;  // thread-local so concurrent contexts don't clash
}

void set_trace_callback(trace_cb cb) {
    tls_cb = std::move(cb);
}

void publish(const std::string& name, const ggml_tensor* t) {
    if (tls_cb) {
        tls_cb(name, t);
    }
}

std::vector<float> copy_tensor_to_f32(const ggml_tensor* t) {
    const size_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)t->data, out.data(), n);
    } else {
        // Generic path via ggml's quantize/dequantize would land here.
        // Plan 3 only ever publishes F32/F16; throw to surface unsupported.
        throw std::runtime_error("copy_tensor_to_f32: unsupported tensor type");
    }
    return out;
}

std::vector<int64_t> tensor_shape(const ggml_tensor* t) {
    std::vector<int64_t> s(GGML_MAX_DIMS);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) s[i] = t->ne[i];
    return s;
}

}  // namespace rfdetr
