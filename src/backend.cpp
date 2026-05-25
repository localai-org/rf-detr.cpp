#include "backend.hpp"
#include "common.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

namespace rfdetr {

ggml_backend_t init_backend(int n_threads, rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) { if (out_status) *out_status = s; };

    ggml_backend_t b = ggml_backend_cpu_init();
    if (!b) {
        rfdetr_logf(RFDETR_LOG_ERROR, "ggml_backend_cpu_init returned null");
        set(RFDETR_ERR_INFERENCE);
        return nullptr;
    }
    if (n_threads > 0) {
        ggml_backend_cpu_set_n_threads(b, n_threads);
    }
    set(RFDETR_OK);
    return b;
}

void free_backend(ggml_backend_t b) {
    if (b) ggml_backend_free(b);
}

bool is_cpu(ggml_backend_t b) {
    if (!b) return false;
    return ggml_backend_is_cpu(b);
}

}  // namespace rfdetr
