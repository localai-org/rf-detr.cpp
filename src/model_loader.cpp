#include "model_loader.hpp"
#include "common.hpp"
#include "rfdetr.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <new>
#include <string>
#include <vector>

namespace rfdetr {

namespace {

const char* kFormatVersion = "1";

bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.is_open();
}

template <typename T>
bool get_u32(gguf_context* g, const char* key, T& out) {
    const int64_t kid = gguf_find_key(g, key);
    if (kid < 0) return false;
    out = (T)gguf_get_val_u32(g, kid);
    return true;
}

bool get_str(gguf_context* g, const char* key, std::string& out) {
    const int64_t kid = gguf_find_key(g, key);
    if (kid < 0) return false;
    out = gguf_get_val_str(g, kid);
    return true;
}

bool get_f32_array(gguf_context* g, const char* key, float* out, size_t n) {
    const int64_t kid = gguf_find_key(g, key);
    if (kid < 0) return false;
    if ((size_t)gguf_get_arr_n(g, kid) != n) return false;
    const float* data = (const float*)gguf_get_arr_data(g, kid);
    std::memcpy(out, data, n * sizeof(float));
    return true;
}

bool get_i32_array(gguf_context* g, const char* key, std::vector<uint32_t>& out) {
    const int64_t kid = gguf_find_key(g, key);
    if (kid < 0) return false;
    size_t n = gguf_get_arr_n(g, kid);
    const int32_t* data = (const int32_t*)gguf_get_arr_data(g, kid);
    out.resize(n);
    for (size_t i = 0; i < n; ++i) out[i] = (uint32_t)data[i];
    return true;
}

bool get_str_array(gguf_context* g, const char* key, std::vector<std::string>& out) {
    const int64_t kid = gguf_find_key(g, key);
    if (kid < 0) return false;
    size_t n = gguf_get_arr_n(g, kid);
    out.clear();
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.emplace_back(gguf_get_arr_str(g, kid, i));
    }
    return true;
}

}  // namespace

Model* model_load(const std::string& path, rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) { if (out_status) *out_status = s; };

    if (!file_exists(path)) {
        rfdetr_logf(RFDETR_LOG_ERROR, "model_load: file not found '%s'", path.c_str());
        set(RFDETR_ERR_FILE_NOT_FOUND);
        return nullptr;
    }

    ggml_context* gctx = nullptr;
    gguf_init_params init_params{ /* no_alloc */ true, /* ctx */ &gctx };
    gguf_context* gguf = gguf_init_from_file(path.c_str(), init_params);
    if (!gguf) {
        rfdetr_logf(RFDETR_LOG_ERROR, "model_load: gguf_init_from_file failed for '%s'", path.c_str());
        set(RFDETR_ERR_MODEL_FORMAT);
        return nullptr;
    }

    auto fail = [&](rfdetr_status s, const char* msg) -> Model* {
        rfdetr_logf(RFDETR_LOG_ERROR, "model_load: %s", msg);
        gguf_free(gguf);
        if (gctx) ggml_free(gctx);
        set(s);
        return nullptr;
    };

    // Format version
    std::string fmt;
    if (!get_str(gguf, "rfdetr.format.version", fmt) || fmt != kFormatVersion) {
        return fail(RFDETR_ERR_MODEL_FORMAT, "unsupported rfdetr.format.version");
    }

    Model* m = new (std::nothrow) Model();
    if (!m) return fail(RFDETR_ERR_OUT_OF_MEMORY, "alloc Model");

    m->gguf = gguf;
    m->meta = gctx;
    m->path = path;

    auto& c = m->config;
    if (!get_str(gguf, "rfdetr.variant",     c.variant))     return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.variant missing");
    if (!get_u32(gguf, "rfdetr.image_size",  c.image_size))  return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.image_size missing");
    if (!get_u32(gguf, "rfdetr.num_queries", c.num_queries)) return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.num_queries missing");
    if (!get_u32(gguf, "rfdetr.num_classes", c.num_classes)) return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.num_classes missing");
    if (!get_str_array(gguf, "rfdetr.class_names", c.class_names))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.class_names missing");
    if (c.class_names.size() != c.num_classes)
        return fail(RFDETR_ERR_MODEL_FORMAT, "class_names length != num_classes");

    if (!get_f32_array(gguf, "rfdetr.preprocess.mean", c.preprocess_mean, 3))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.preprocess.mean missing or wrong shape");
    if (!get_f32_array(gguf, "rfdetr.preprocess.std", c.preprocess_std, 3))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.preprocess.std missing or wrong shape");

    if (!get_u32(gguf, "rfdetr.backbone.dim",         c.backbone.dim)         ||
        !get_u32(gguf, "rfdetr.backbone.depth",       c.backbone.depth)       ||
        !get_u32(gguf, "rfdetr.backbone.heads",       c.backbone.heads)       ||
        !get_u32(gguf, "rfdetr.backbone.window_size", c.backbone.window_size))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.backbone.* incomplete");
    if (!get_i32_array(gguf, "rfdetr.backbone.multi_scale_layers", c.backbone.multi_scale_layers))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.backbone.multi_scale_layers missing");

    if (!get_u32(gguf, "rfdetr.encoder.layers",    c.encoder.layers)    ||
        !get_u32(gguf, "rfdetr.encoder.model_dim", c.encoder.model_dim) ||
        !get_u32(gguf, "rfdetr.encoder.ffn_dim",   c.encoder.ffn_dim)   ||
        !get_u32(gguf, "rfdetr.encoder.heads",     c.encoder.heads))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.encoder.* incomplete");

    if (!get_u32(gguf, "rfdetr.decoder.layers",    c.decoder.layers)    ||
        !get_u32(gguf, "rfdetr.decoder.model_dim", c.decoder.model_dim) ||
        !get_u32(gguf, "rfdetr.decoder.ffn_dim",   c.decoder.ffn_dim)   ||
        !get_u32(gguf, "rfdetr.decoder.heads",     c.decoder.heads))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.decoder.* incomplete");

    // Tensor inventory (descriptors only — data not loaded)
    const int64_t n_tensors = gguf_get_n_tensors(gguf);
    m->tensors.reserve(n_tensors);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gguf, i);
        ggml_tensor* t = ggml_get_tensor(gctx, name);
        if (!t) return fail(RFDETR_ERR_MODEL_LOAD, "ggml_get_tensor failed");
        m->tensors.emplace(name, t);
    }

    set(RFDETR_OK);
    return m;
}

void model_free(Model* m) {
    if (!m) return;
    if (m->weights) ggml_backend_buffer_free(m->weights);
    if (m->gguf) gguf_free(m->gguf);
    if (m->meta) ggml_free(m->meta);
    delete m;
}

rfdetr_status model_realize_weights(Model& m, ggml_backend_t backend) {
    if (m.weights) return RFDETR_OK;

    if (!backend) {
        rfdetr_logf(RFDETR_LOG_ERROR, "model_realize_weights: null backend");
        return RFDETR_ERR_INVALID_ARG;
    }
    if (m.path.empty()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "model_realize_weights: model has no stashed path");
        return RFDETR_ERR_MODEL_LOAD;
    }

    /* Allocate a buffer big enough for every tensor in m.meta, on the
     * supplied backend. */
    m.weights = ggml_backend_alloc_ctx_tensors(m.meta, backend);
    if (!m.weights) {
        rfdetr_logf(RFDETR_LOG_ERROR, "model_realize_weights: backend alloc failed");
        return RFDETR_ERR_MODEL_LOAD;
    }

    FILE* fp = std::fopen(m.path.c_str(), "rb");
    if (!fp) {
        rfdetr_logf(RFDETR_LOG_ERROR, "model_realize_weights: open failed: %s", m.path.c_str());
        ggml_backend_buffer_free(m.weights);
        m.weights = nullptr;
        return RFDETR_ERR_MODEL_LOAD;
    }

    const int64_t n_tensors = gguf_get_n_tensors(m.gguf);
    const size_t data_offset = gguf_get_data_offset(m.gguf);
    std::vector<uint8_t> buf;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(m.gguf, i);
        const size_t offset = data_offset + gguf_get_tensor_offset(m.gguf, i);
        ggml_tensor* t = ggml_get_tensor(m.meta, name);
        if (!t) {
            rfdetr_logf(RFDETR_LOG_ERROR, "model_realize_weights: tensor '%s' missing in ctx", name);
            std::fclose(fp);
            ggml_backend_buffer_free(m.weights);
            m.weights = nullptr;
            return RFDETR_ERR_MODEL_LOAD;
        }
        const size_t nbytes = ggml_nbytes(t);
        buf.resize(nbytes);

        if (std::fseek(fp, (long)offset, SEEK_SET) != 0 ||
            std::fread(buf.data(), 1, nbytes, fp) != nbytes) {
            rfdetr_logf(RFDETR_LOG_ERROR, "model_realize_weights: read failed for '%s'", name);
            std::fclose(fp);
            ggml_backend_buffer_free(m.weights);
            m.weights = nullptr;
            return RFDETR_ERR_MODEL_LOAD;
        }

        ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
    }

    std::fclose(fp);
    return RFDETR_OK;
}

std::vector<std::string> expected_tensor_names(const Config& cfg) {
    std::vector<std::string> names;

    // Backbone
    names.emplace_back("backbone.patch_embed.weight");
    names.emplace_back("backbone.patch_embed.bias");
    names.emplace_back("backbone.pos_embed");
    names.emplace_back("backbone.cls_token");
    for (uint32_t i = 0; i < cfg.backbone.depth; ++i) {
        std::string p = "backbone.blocks." + std::to_string(i) + ".";
        names.emplace_back(p + "norm1.weight");
        names.emplace_back(p + "norm1.bias");
        names.emplace_back(p + "attn.qkv.weight");
        names.emplace_back(p + "attn.qkv.bias");
        names.emplace_back(p + "attn.proj.weight");
        names.emplace_back(p + "attn.proj.bias");
        names.emplace_back(p + "norm2.weight");
        names.emplace_back(p + "norm2.bias");
        names.emplace_back(p + "mlp.fc1.weight");
        names.emplace_back(p + "mlp.fc1.bias");
        names.emplace_back(p + "mlp.fc2.weight");
        names.emplace_back(p + "mlp.fc2.bias");
    }
    names.emplace_back("backbone.norm.weight");
    names.emplace_back("backbone.norm.bias");

    // Projector
    for (size_t j = 0; j < cfg.backbone.multi_scale_layers.size(); ++j) {
        std::string p = "projector.level" + std::to_string(j) + ".";
        names.emplace_back(p + "weight");
        names.emplace_back(p + "bias");
    }
    names.emplace_back("projector.level_embed");

    // Encoder
    for (uint32_t i = 0; i < cfg.encoder.layers; ++i) {
        std::string p = "encoder.layers." + std::to_string(i) + ".";
        names.emplace_back(p + "self_attn.qkv.weight");
        names.emplace_back(p + "self_attn.qkv.bias");
        names.emplace_back(p + "self_attn.out.weight");
        names.emplace_back(p + "self_attn.out.bias");
        names.emplace_back(p + "norm1.weight");
        names.emplace_back(p + "norm1.bias");
        names.emplace_back(p + "ffn.fc1.weight");
        names.emplace_back(p + "ffn.fc1.bias");
        names.emplace_back(p + "ffn.fc2.weight");
        names.emplace_back(p + "ffn.fc2.bias");
        names.emplace_back(p + "norm2.weight");
        names.emplace_back(p + "norm2.bias");
    }

    // Decoder
    names.emplace_back("decoder.queries");
    for (uint32_t i = 0; i < cfg.decoder.layers; ++i) {
        std::string p = "decoder.layers." + std::to_string(i) + ".";
        names.emplace_back(p + "self_attn.qkv.weight");
        names.emplace_back(p + "self_attn.qkv.bias");
        names.emplace_back(p + "self_attn.out.weight");
        names.emplace_back(p + "self_attn.out.bias");
        names.emplace_back(p + "norm1.weight");
        names.emplace_back(p + "norm1.bias");
        names.emplace_back(p + "cross_attn.q.weight");
        names.emplace_back(p + "cross_attn.q.bias");
        names.emplace_back(p + "cross_attn.kv.weight");
        names.emplace_back(p + "cross_attn.kv.bias");
        names.emplace_back(p + "cross_attn.out.weight");
        names.emplace_back(p + "cross_attn.out.bias");
        names.emplace_back(p + "norm2.weight");
        names.emplace_back(p + "norm2.bias");
        names.emplace_back(p + "ffn.fc1.weight");
        names.emplace_back(p + "ffn.fc1.bias");
        names.emplace_back(p + "ffn.fc2.weight");
        names.emplace_back(p + "ffn.fc2.bias");
        names.emplace_back(p + "norm3.weight");
        names.emplace_back(p + "norm3.bias");
    }

    // Heads
    names.emplace_back("heads.class.fc.weight");
    names.emplace_back("heads.class.fc.bias");
    names.emplace_back("heads.bbox.fc1.weight");
    names.emplace_back("heads.bbox.fc1.bias");
    names.emplace_back("heads.bbox.fc2.weight");
    names.emplace_back("heads.bbox.fc2.bias");
    names.emplace_back("heads.bbox.fc3.weight");
    names.emplace_back("heads.bbox.fc3.bias");

    return names;
}

rfdetr_status model_validate_tensors(const Model& m) {
    const auto expected = expected_tensor_names(m.config);
    std::vector<std::string> missing;
    for (const auto& n : expected) {
        if (m.tensors.find(n) == m.tensors.end()) {
            missing.push_back(n);
        }
    }
    if (!missing.empty()) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "model_validate_tensors: %zu missing tensor(s); first: '%s'",
                    missing.size(), missing.front().c_str());
        return RFDETR_ERR_MODEL_LOAD;
    }
    return RFDETR_OK;
}

}  // namespace rfdetr
