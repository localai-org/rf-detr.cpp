#include "rfdetr.h"
#include "backend.hpp"
#include "common.hpp"
#include "model_loader.hpp"

#include <new>
#include <string>

/* Opaque struct — defined here so external callers can only hold pointers. */
struct rfdetr_context {
    rfdetr::Model* model     = nullptr;
    ggml_backend_t backend   = nullptr;
    int            n_threads = 1;
};

extern "C" rfdetr_context* rfdetr_init(const rfdetr_params* params, rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) { if (out_status) *out_status = s; };

    if (!params || !params->model_path) {
        set(RFDETR_ERR_INVALID_ARG);
        return nullptr;
    }

    /* The params struct also carries a logging callback; install it now so
     * the model_load errors are surfaced through the caller's channel. */
    if (params->log_cb) {
        rfdetr_set_log_callback(params->log_cb, params->log_user_data);
    }

    rfdetr_status load_st;
    rfdetr::Model* m = rfdetr::model_load(params->model_path, &load_st);
    if (!m) {
        set(load_st);
        return nullptr;
    }

    rfdetr_status v = rfdetr::model_validate_tensors(*m);
    if (v != RFDETR_OK) {
        rfdetr::model_free(m);
        set(v);
        return nullptr;
    }

    rfdetr_status bk_st;
    ggml_backend_t backend = rfdetr::init_backend(
        params->n_threads > 0 ? params->n_threads : 1, &bk_st);
    if (!backend) {
        rfdetr::model_free(m);
        set(bk_st);
        return nullptr;
    }

    rfdetr_status rw_st = rfdetr::model_realize_weights(*m, backend);
    if (rw_st != RFDETR_OK) {
        rfdetr::free_backend(backend);
        rfdetr::model_free(m);
        set(rw_st);
        return nullptr;
    }

    auto* ctx = new (std::nothrow) rfdetr_context();
    if (!ctx) {
        rfdetr::free_backend(backend);
        rfdetr::model_free(m);
        set(RFDETR_ERR_OUT_OF_MEMORY);
        return nullptr;
    }
    ctx->model     = m;
    ctx->backend   = backend;
    ctx->n_threads = params->n_threads > 0 ? params->n_threads : 1;

    rfdetr_logf(RFDETR_LOG_INFO, "rfdetr_init: loaded variant=%s, num_classes=%u, num_queries=%u",
                m->config.variant.c_str(),
                m->config.num_classes,
                m->config.num_queries);

    set(RFDETR_OK);
    return ctx;
}

extern "C" void rfdetr_free(rfdetr_context* ctx) {
    if (!ctx) return;
    rfdetr::model_free(ctx->model);
    rfdetr::free_backend(ctx->backend);
    delete ctx;
}

extern "C" rfdetr_status rfdetr_detect(rfdetr_context* ctx,
                                       const rfdetr_image* img,
                                       const rfdetr_detect_params* /*params*/,
                                       rfdetr_detection** out_detections,
                                       size_t* out_n) {
    if (out_detections) *out_detections = nullptr;
    if (out_n)          *out_n = 0;
    if (!ctx || !img) return RFDETR_ERR_INVALID_ARG;

    /* Plan 3 wires the forward graph. For now: signal that the model loaded
     * successfully but inference isn't implemented. */
    return RFDETR_ERR_NOT_IMPLEMENTED;
}

/* Accessors used by `info` subcommand. */
extern "C" {

const char* rfdetr_context_variant(const rfdetr_context* ctx) {
    return (ctx && ctx->model) ? ctx->model->config.variant.c_str() : "";
}
uint32_t rfdetr_context_image_size(const rfdetr_context* ctx) {
    return (ctx && ctx->model) ? ctx->model->config.image_size : 0;
}
uint32_t rfdetr_context_num_queries(const rfdetr_context* ctx) {
    return (ctx && ctx->model) ? ctx->model->config.num_queries : 0;
}
uint32_t rfdetr_context_num_classes(const rfdetr_context* ctx) {
    return (ctx && ctx->model) ? ctx->model->config.num_classes : 0;
}
size_t rfdetr_context_n_tensors(const rfdetr_context* ctx) {
    return (ctx && ctx->model) ? ctx->model->tensors.size() : 0;
}

}  // extern "C"
