#include "rfdetr.h"
#include "backend.hpp"
#include "common.hpp"
#include "model_loader.hpp"
#include "rfdetr_model.hpp"
#include "postprocess.hpp"
#include "image_io.hpp"
#include "ggml-backend.h"

#include <cstdlib>
#include <new>
#include <string>
#include <vector>

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
                                       const rfdetr_detect_params* params,
                                       rfdetr_detection** out_detections,
                                       size_t* out_n) {
    if (out_detections) *out_detections = nullptr;
    if (out_n)          *out_n = 0;
    if (!ctx || !img || !params || !out_detections || !out_n) return RFDETR_ERR_INVALID_ARG;
    if (!ctx->model || !ctx->backend) return RFDETR_ERR_INVALID_ARG;

    const rfdetr::Config& cfg = ctx->model->config;
    const int img_size = (int)cfg.image_size;

    /* 1. Preprocess image to F32 (img_size × img_size × 3 × 1) */
    float* px_data = nullptr;
    int px_w = 0, px_h = 0;
    rfdetr_status pp_st = rfdetr_preprocess(img, img_size, img_size,
                                            cfg.preprocess_mean, cfg.preprocess_std,
                                            &px_data, &px_w, &px_h);
    if (pp_st != RFDETR_OK) {
        rfdetr_logf(RFDETR_LOG_ERROR, "rfdetr_detect: preprocess failed");
        return pp_st;
    }

    /* 2. Full forward (2 graphs internally: backbone+projector+two_stage,
     *    then CPU top-K + decoder + heads). Returns host-side outputs. */
    rfdetr::ForwardOutput fout = rfdetr::rfdetr_model_forward(
        *ctx->model, px_data, px_w, ctx->backend);
    std::free(px_data);
    px_data = nullptr;
    if (fout.class_logits.empty() || fout.bbox_cxcywh.empty()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "rfdetr_detect: model forward failed");
        return RFDETR_ERR_INFERENCE;
    }

    /* 3. Postprocess. class_logits is laid out (NC, NQ) column-major in
     * ggml-land; the host vector preserves that order with NC fastest-varying,
     * i.e. row-major (query, class) — what rfdetr_select_detections expects. */
    rfdetr_select_detections(fout.class_logits.data(), fout.bbox_cxcywh.data(),
                             (size_t)fout.num_queries, (size_t)fout.num_classes,
                             params->threshold, params->top_k,
                             params->class_filter, params->class_filter_len,
                             rfdetr_image_width(img), rfdetr_image_height(img),
                             out_detections, out_n);

    /* 4. Attach class names from the loaded config (best-effort) */
    if (out_detections && *out_detections) {
        const auto& names = ctx->model->config.class_names;
        for (size_t i = 0; i < *out_n; ++i) {
            uint32_t cid = (*out_detections)[i].class_id;
            if (cid < names.size()) {
                (*out_detections)[i].class_name = names[cid].c_str();
            }
        }
    }

    return RFDETR_OK;
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
