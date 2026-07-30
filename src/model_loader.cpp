#include "model_loader.hpp"
#include "common.hpp"
#include "rfdetr.h"
#include "two_stage.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <new>
#include <string>
#include <vector>

namespace rfdetr {

namespace {

const char* kFormatVersion = "2";

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

bool get_bool(gguf_context* g, const char* key, bool& out) {
    const int64_t kid = gguf_find_key(g, key);
    if (kid < 0) return false;
    out = gguf_get_val_bool(g, kid);
    return true;
}

/* Bicubic-resample a (dim, src*src) patch position embedding (row-major
 * patch grid, dim-fastest) onto a (dim, dst*dst) grid. Matches PyTorch's
 * F.interpolate(mode='bicubic', align_corners=False, antialias=True) up
 * to float rounding.
 *
 * IMPORTANT: HF DinoV2's interpolate_pos_encoding uses antialias=True
 * (see modeling_dinov2_with_registers.py). PyTorch's antialias path differs
 * from its non-AA path in several ways even when upsampling (src < dst):
 *
 *  1. The bicubic kernel uses a = -0.5 (Keys, PIL-compatible) instead of the
 *     a = -0.75 used by the non-AA path. See:
 *     https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cpu/UpSampleKernel.cpp#L1338
 *     (HelperInterpCubic::aa_filter): "We are using -0.5 for bicubic,
 *     antialiasing=true (compatibility with PIL) and using -0.75 for bicubic,
 *     antialiasing=false (compatibility with Opencv)".
 *
 *  2. Center convention is `scale * (i + 0.5)` (not `scale * (i + 0.5) - 0.5`).
 *     See _compute_indices_min_size_weights_aa at L752 in the same file.
 *
 *  3. Weights are normalized to sum to 1 across the support window.
 *
 *  4. Support window is `2.0 * max(scale, 1.0)`; for upsampling (scale<1)
 *     this is 2.0 and the kernel argument is `(j + xmin - center + 0.5)`
 *     (no inv_scale division).
 *
 *  5. Indices are computed via `xmin = (int)(center - support + 0.5)`,
 *     `xmax = (int)(center + support + 0.5)`, BOTH clamped to [0, src_size].
 *     Out-of-range source pixels do NOT contribute (no per-tap edge clamp).
 *
 * Kernel:
 *   K(x) = ((A+2)|x| - (A+3))|x|^2 + 1                for |x| < 1
 *   K(x) = ((A|x| - 5A)|x| + 8A)|x| - 4A              for 1 <= |x| < 2
 *   K(x) = 0                                          otherwise
 * with A = -0.5 for antialias=True. */
void bicubic_resample_patch_grid(const float* src, int src_side, int dim,
                                 float* dst, int dst_side) {
    constexpr float A = -0.5f;  // antialias=True path uses Keys, not Catmull-Rom
    auto kernel = [](float x) -> float {
        const float ax = std::fabs(x);
        if (ax < 1.0f) {
            return ((A + 2.0f) * ax - (A + 3.0f)) * ax * ax + 1.0f;
        }
        if (ax < 2.0f) {
            return ((A * ax - 5.0f * A) * ax + 8.0f * A) * ax - 4.0f * A;
        }
        return 0.0f;
    };

    /* Precompute per-output-coordinate variable-tap weights for x and y axes.
     * The max possible support count is ceil(2*max(scale,1))*2 + 1, but for
     * upsample (scale<1) the support is 2.0 → at most 4 taps. We size for
     * upsample only here because rfdetr-base always upsamples (37 → 40). */
    auto fill_taps = [&](int dst_n, int src_n,
                         std::vector<float>& w_out, std::vector<int>& idx_out,
                         std::vector<int>& size_out) {
        const float scale_d = (float)src_n / (float)dst_n;
        /* PyTorch's AA path uses `double` internally for indices/weights,
         * but the final cast happens via `static_cast<int64_t>(x + 0.5)`
         * which is equivalent to floor(x + 0.5) for non-negative x — i.e.
         * rounding to nearest. We mimic that. */
        const float invscale = scale_d >= 1.0f ? 1.0f / scale_d : 1.0f;
        const float support  = scale_d >= 1.0f ? 2.0f * scale_d : 2.0f;
        const int   max_taps = (int)std::ceil(support) * 2 + 1;
        w_out.assign((size_t)dst_n * (size_t)max_taps, 0.0f);
        idx_out.assign((size_t)dst_n * (size_t)max_taps, 0);
        size_out.assign(dst_n, 0);
        for (int o = 0; o < dst_n; ++o) {
            const float center = scale_d * ((float)o + 0.5f);
            const int xmin_raw = (int)(center - support + 0.5f);
            const int xmax_raw = (int)(center + support + 0.5f);
            const int xmin = std::max(xmin_raw, 0);
            const int xmax = std::min(xmax_raw, src_n);
            const int xsize_unclipped = xmax - xmin;
            const int xsize = std::min(std::max(xsize_unclipped, 0), max_taps);
            float total = 0.0f;
            for (int j = 0; j < xsize; ++j) {
                const float w = kernel(((float)j + (float)xmin - center + 0.5f) * invscale);
                w_out[(size_t)o * (size_t)max_taps + (size_t)j] = w;
                idx_out[(size_t)o * (size_t)max_taps + (size_t)j] = xmin + j;
                total += w;
            }
            if (total != 0.0f) {
                for (int j = 0; j < xsize; ++j) {
                    w_out[(size_t)o * (size_t)max_taps + (size_t)j] /= total;
                }
            }
            size_out[o] = xsize;
        }
        return max_taps;
    };

    std::vector<float> wx, wy;
    std::vector<int>   ix, iy;
    std::vector<int>   sx, sy;
    const float scale_d = (float)src_side / (float)dst_side;
    const float support_d = scale_d >= 1.0f ? 2.0f * scale_d : 2.0f;
    const int   max_taps = (int)std::ceil(support_d) * 2 + 1;
    fill_taps(dst_side, src_side, wx, ix, sx);
    fill_taps(dst_side, src_side, wy, iy, sy);

    /* For each (oy, ox) output pixel: sum_{ky, kx} wy[ky] * wx[kx] * src[iy[ky], ix[kx], d]
     * for every d. Loop ordering keeps the dim accumulator hot. */
    for (int oy = 0; oy < dst_side; ++oy) {
        for (int ox = 0; ox < dst_side; ++ox) {
            float* dst_row = dst + ((size_t)oy * dst_side + (size_t)ox) * dim;
            for (int d = 0; d < dim; ++d) dst_row[d] = 0.0f;
            const int ny = sy[oy];
            const int nx = sx[ox];
            for (int ky = 0; ky < ny; ++ky) {
                const int sy_i = iy[(size_t)oy * (size_t)max_taps + (size_t)ky];
                const float w_y = wy[(size_t)oy * (size_t)max_taps + (size_t)ky];
                if (w_y == 0.0f) continue;
                for (int kx = 0; kx < nx; ++kx) {
                    const int sx_i = ix[(size_t)ox * (size_t)max_taps + (size_t)kx];
                    const float w_x = wx[(size_t)ox * (size_t)max_taps + (size_t)kx];
                    if (w_x == 0.0f) continue;
                    const float w = w_x * w_y;
                    const float* sp = src + ((size_t)sy_i * src_side + (size_t)sx_i) * dim;
                    for (int d = 0; d < dim; ++d) {
                        dst_row[d] += w * sp[d];
                    }
                }
            }
        }
    }
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
    if (!get_u32(gguf, "rfdetr.patch_size",  c.patch_size))  return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.patch_size missing");
    if (!get_u32(gguf, "rfdetr.num_queries", c.num_queries)) return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.num_queries missing");
    if (!get_u32(gguf, "rfdetr.group_detr",  c.group_detr))  return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.group_detr missing");
    if (!get_u32(gguf, "rfdetr.num_classes", c.num_classes)) return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.num_classes missing");
    if (!get_str_array(gguf, "rfdetr.class_names", c.class_names))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.class_names missing");
    if (c.class_names.size() != c.num_classes)
        return fail(RFDETR_ERR_MODEL_FORMAT, "class_names length != num_classes");

    if (!get_f32_array(gguf, "rfdetr.preprocess.mean", c.preprocess_mean, 3))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.preprocess.mean missing or wrong shape");
    if (!get_f32_array(gguf, "rfdetr.preprocess.std", c.preprocess_std, 3))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.preprocess.std missing or wrong shape");

    if (!get_u32(gguf, "rfdetr.backbone.dim",                   c.backbone.dim)         ||
        !get_u32(gguf, "rfdetr.backbone.depth",                 c.backbone.depth)       ||
        !get_u32(gguf, "rfdetr.backbone.heads",                 c.backbone.heads)       ||
        !get_u32(gguf, "rfdetr.backbone.ffn_dim",               c.backbone.ffn_dim)     ||
        !get_u32(gguf, "rfdetr.backbone.num_windows",           c.backbone.num_windows) ||
        !get_u32(gguf, "rfdetr.backbone.pos_embed_train_size",  c.backbone.pos_embed_train_size))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.backbone.* incomplete");
    if (!get_i32_array(gguf, "rfdetr.backbone.global_attn_indices", c.backbone.global_attn_indices))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.backbone.global_attn_indices missing");
    if (!get_i32_array(gguf, "rfdetr.backbone.out_feature_indices", c.backbone.out_feature_indices))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.backbone.out_feature_indices missing");

    if (!get_u32(gguf, "rfdetr.projector.in_dim",         c.projector.in_dim)         ||
        !get_u32(gguf, "rfdetr.projector.out_dim",        c.projector.out_dim)        ||
        !get_u32(gguf, "rfdetr.projector.bottleneck_dim", c.projector.bottleneck_dim) ||
        !get_u32(gguf, "rfdetr.projector.n_bottlenecks",  c.projector.n_bottlenecks))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.projector.* incomplete");

    if (!get_u32(gguf, "rfdetr.decoder.layers",              c.decoder.layers)              ||
        !get_u32(gguf, "rfdetr.decoder.model_dim",           c.decoder.model_dim)           ||
        !get_u32(gguf, "rfdetr.decoder.ffn_dim",             c.decoder.ffn_dim)             ||
        !get_u32(gguf, "rfdetr.decoder.self_attn_heads",     c.decoder.self_attn_heads)     ||
        !get_u32(gguf, "rfdetr.decoder.cross_attn_heads",    c.decoder.cross_attn_heads)    ||
        !get_u32(gguf, "rfdetr.decoder.cross_attn_n_levels", c.decoder.cross_attn_n_levels) ||
        !get_u32(gguf, "rfdetr.decoder.cross_attn_n_points", c.decoder.cross_attn_n_points))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.decoder.* incomplete");

    if (!get_u32(gguf, "rfdetr.two_stage.n_groups", c.two_stage.n_groups))
        return fail(RFDETR_ERR_MODEL_FORMAT, "rfdetr.two_stage.n_groups missing");

    /* Segmentation head (optional). Pretrained detection variants converted
     * before the seg field was added simply lack the key — that's not an
     * error, just leaves has_segmentation_head=false. */
    if (!get_bool(gguf, "rfdetr.has_segmentation_head", c.has_segmentation_head)) {
        c.has_segmentation_head = false;
    }
    if (c.has_segmentation_head) {
        if (!get_u32(gguf, "rfdetr.mask_downsample_ratio", c.mask_downsample_ratio)) {
            return fail(RFDETR_ERR_MODEL_FORMAT,
                        "rfdetr.mask_downsample_ratio missing for seg model");
        }
        if (c.mask_downsample_ratio == 0) {
            return fail(RFDETR_ERR_MODEL_FORMAT,
                        "rfdetr.mask_downsample_ratio must be > 0");
        }
    }

    // Tensor inventory (descriptors only — data not loaded)
    const int64_t n_tensors = gguf_get_n_tensors(gguf);
    m->tensors.reserve(n_tensors);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gguf, i);
        ggml_tensor* t = ggml_get_tensor(gctx, name);
        if (!t) return fail(RFDETR_ERR_MODEL_LOAD, "ggml_get_tensor failed");
        m->tensors.emplace(name, t);
    }

    /* Pre-allocate a slot for the bicubic-interpolated pos_embed in a
     * separate ggml_context (gguf's meta ctx has a tight memory pool sized
     * for the loaded tensor descriptors only). */
    const int64_t inf_patches_per_side =
        (int64_t)(c.image_size / c.patch_size);
    const int64_t inf_tokens = inf_patches_per_side * inf_patches_per_side + 1;
    {
        ggml_init_params ep{};
        ep.mem_size   = ggml_tensor_overhead() * 8;
        ep.mem_buffer = nullptr;
        ep.no_alloc   = true;
        m->extras_ctx = ggml_init(ep);
        if (!m->extras_ctx) return fail(RFDETR_ERR_MODEL_LOAD, "alloc extras_ctx");
    }
    ggml_tensor* pe_interp = ggml_new_tensor_2d(
        m->extras_ctx, GGML_TYPE_F32,
        (int64_t)c.backbone.dim, inf_tokens);
    if (!pe_interp) return fail(RFDETR_ERR_MODEL_LOAD, "alloc pe_interp");
    ggml_set_name(pe_interp, "backbone.pos_embed.interp");
    m->backbone_pos_embed_interp = pe_interp;

    /* Two-stage proposal grid: (4, inf_side * inf_side). Constant — depends
     * only on the spatial grid (40x40 for rfdetr-base at 560 px). */
    const int64_t inf_n = inf_patches_per_side * inf_patches_per_side;
    ggml_tensor* prop = ggml_new_tensor_2d(m->extras_ctx, GGML_TYPE_F32,
                                            /*ne0*/ 4, /*ne1*/ inf_n);
    if (!prop) return fail(RFDETR_ERR_MODEL_LOAD, "alloc proposals_grid");
    ggml_set_name(prop, "two_stage.proposals.grid");
    m->proposals_grid = prop;

    set(RFDETR_OK);
    return m;
}

void model_free(Model* m) {
    if (!m) return;
    if (m->extras_buf) ggml_backend_buffer_free(m->extras_buf);
    if (m->extras_ctx) ggml_free(m->extras_ctx);
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

    /* Alloc the extras_ctx backend buffer if not already done. */
    if (!m.extras_buf) {
        m.extras_buf = ggml_backend_alloc_ctx_tensors(m.extras_ctx, backend);
        if (!m.extras_buf) {
            rfdetr_logf(RFDETR_LOG_ERROR,
                        "model_realize_weights: extras buffer alloc failed");
            return RFDETR_ERR_MODEL_LOAD;
        }
    }

    /* Compute the bicubic-interpolated pos_embed for the inference image
     * size. Layout: stored pos_embed ne = (dim, n_train_tokens) with
     * n_train_tokens = 1 (CLS) + train_side^2. Output ne = (dim, n_inf_tokens)
     * with n_inf_tokens = 1 + inf_side^2.
     *
     * Memory: the pos_embed tensor is dim-fastest (column-major). For each
     * token index t, embedding[d, t] lives at offset t*dim + d. So treating
     * the patch grid as `(dim, side*side)` is equivalent to (side, side, dim)
     * in row-major / dim-fastest. */
    {
        auto it_pe = m.tensors.find("backbone.pos_embed");
        if (it_pe == m.tensors.end() || !m.backbone_pos_embed_interp) {
            rfdetr_logf(RFDETR_LOG_ERROR,
                        "model_realize_weights: missing backbone.pos_embed slot");
            return RFDETR_ERR_MODEL_LOAD;
        }
        ggml_tensor* pe = it_pe->second;
        const int dim       = (int)m.config.backbone.dim;
        const int train_side = (int)m.config.backbone.pos_embed_train_size;
        const int inf_side   = (int)(m.config.image_size / m.config.patch_size);
        const int n_train_tokens = train_side * train_side + 1;
        const int n_inf_tokens   = inf_side   * inf_side   + 1;

        if ((int64_t)pe->ne[0] != dim || (int64_t)pe->ne[1] != n_train_tokens) {
            rfdetr_logf(RFDETR_LOG_ERROR,
                        "model_realize_weights: pos_embed shape mismatch "
                        "ne=(%lld, %lld), expected (%d, %d)",
                        (long long)pe->ne[0], (long long)pe->ne[1],
                        dim, n_train_tokens);
            return RFDETR_ERR_MODEL_LOAD;
        }

        /* pos_embed may be stored as F32 or F16 in the GGUF. Read the raw
         * bytes (size = ggml_nbytes(pe), which accounts for the dtype) and
         * convert F16 -> F32 in-place when needed. Without this conversion,
         * reading dim*n_train_tokens*sizeof(float) bytes from an F16 tensor
         * overruns its backing buffer and trips ggml's bounds check. */
        std::vector<float> pe_raw((size_t)dim * (size_t)n_train_tokens);
        if (pe->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(pe, pe_raw.data(), 0, ggml_nbytes(pe));
        } else if (pe->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> pe_half((size_t)dim * (size_t)n_train_tokens);
            ggml_backend_tensor_get(pe, pe_half.data(), 0, ggml_nbytes(pe));
            ggml_fp16_to_fp32_row(pe_half.data(), pe_raw.data(), pe_raw.size());
        } else {
            rfdetr_logf(RFDETR_LOG_ERROR,
                        "model_realize_weights: pos_embed has unsupported dtype %d "
                        "(expected F32 or F16)",
                        (int)pe->type);
            return RFDETR_ERR_MODEL_LOAD;
        }

        std::vector<float> pe_out((size_t)dim * (size_t)n_inf_tokens);
        /* Copy CLS pos embedding (token 0) unchanged. */
        std::memcpy(pe_out.data(), pe_raw.data(),
                    (size_t)dim * sizeof(float));

        if (train_side == inf_side) {
            std::memcpy(pe_out.data() + (size_t)dim, pe_raw.data() + (size_t)dim,
                        (size_t)dim * train_side * train_side * sizeof(float));
        } else {
            bicubic_resample_patch_grid(
                pe_raw.data() + (size_t)dim,   /* skip CLS row */
                train_side, dim,
                pe_out.data() + (size_t)dim,
                inf_side);
        }

        ggml_backend_tensor_set(m.backbone_pos_embed_interp, pe_out.data(),
                                0, pe_out.size() * sizeof(float));
    }

    /* Populate the two-stage proposal grid. wh = 0.05 * 2^lvl, lvl=0 for the
     * single P4 level rfdetr-base uses. */
    if (m.proposals_grid) {
        const int inf_side =
            (int)(m.config.image_size / m.config.patch_size);
        const size_t n = (size_t)inf_side * (size_t)inf_side;
        std::vector<float> grid(n * 4);
        compute_proposal_grid(inf_side, inf_side, /*wh_value*/ 0.05f,
                              grid.data());
        ggml_backend_tensor_set(m.proposals_grid, grid.data(),
                                0, grid.size() * sizeof(float));
    }

    return RFDETR_OK;
}

std::vector<std::string> expected_tensor_names(const Config& cfg) {
    std::vector<std::string> names;

    // --- Backbone embeddings (4) ---
    names.emplace_back("backbone.patch_embed.weight");
    names.emplace_back("backbone.patch_embed.bias");
    names.emplace_back("backbone.cls_token");
    names.emplace_back("backbone.pos_embed");

    // --- Backbone blocks (18 each) ---
    for (uint32_t i = 0; i < cfg.backbone.depth; ++i) {
        std::string p = "backbone.blocks." + std::to_string(i) + ".";
        names.emplace_back(p + "norm1.weight");
        names.emplace_back(p + "norm1.bias");
        names.emplace_back(p + "attn.q.weight");
        names.emplace_back(p + "attn.q.bias");
        names.emplace_back(p + "attn.k.weight");
        names.emplace_back(p + "attn.k.bias");
        names.emplace_back(p + "attn.v.weight");
        names.emplace_back(p + "attn.v.bias");
        names.emplace_back(p + "attn.proj.weight");
        names.emplace_back(p + "attn.proj.bias");
        names.emplace_back(p + "layer_scale1");
        names.emplace_back(p + "norm2.weight");
        names.emplace_back(p + "norm2.bias");
        names.emplace_back(p + "mlp.fc1.weight");
        names.emplace_back(p + "mlp.fc1.bias");
        names.emplace_back(p + "mlp.fc2.weight");
        names.emplace_back(p + "mlp.fc2.bias");
        names.emplace_back(p + "layer_scale2");
    }
    // --- Backbone final norm (2) ---
    names.emplace_back("backbone.norm.weight");
    names.emplace_back("backbone.norm.bias");

    // --- Projector (single P4 C2f) ---
    names.emplace_back("projector.cv1.conv.weight");
    names.emplace_back("projector.cv1.norm.weight");
    names.emplace_back("projector.cv1.norm.bias");
    names.emplace_back("projector.cv2.conv.weight");
    names.emplace_back("projector.cv2.norm.weight");
    names.emplace_back("projector.cv2.norm.bias");
    for (uint32_t j = 0; j < cfg.projector.n_bottlenecks; ++j) {
        std::string p = "projector.bottleneck." + std::to_string(j) + ".";
        names.emplace_back(p + "cv1.conv.weight");
        names.emplace_back(p + "cv1.norm.weight");
        names.emplace_back(p + "cv1.norm.bias");
        names.emplace_back(p + "cv2.conv.weight");
        names.emplace_back(p + "cv2.norm.weight");
        names.emplace_back(p + "cv2.norm.bias");
    }
    names.emplace_back("projector.final_norm.weight");
    names.emplace_back("projector.final_norm.bias");

    // --- Two-stage groups (12 each) ---
    for (uint32_t g = 0; g < cfg.two_stage.n_groups; ++g) {
        const std::string gi = std::to_string(g);
        names.emplace_back("two_stage.enc_output." + gi + ".weight");
        names.emplace_back("two_stage.enc_output." + gi + ".bias");
        names.emplace_back("two_stage.enc_output_norm." + gi + ".weight");
        names.emplace_back("two_stage.enc_output_norm." + gi + ".bias");
        names.emplace_back("two_stage.enc_out_class_embed." + gi + ".weight");
        names.emplace_back("two_stage.enc_out_class_embed." + gi + ".bias");
        for (int j = 0; j < 3; ++j) {
            const std::string ji = std::to_string(j);
            names.emplace_back("two_stage.enc_out_bbox_embed." + gi + ".layers." + ji + ".weight");
            names.emplace_back("two_stage.enc_out_bbox_embed." + gi + ".layers." + ji + ".bias");
        }
    }

    // --- Decoder queries (group 0 slice) ---
    names.emplace_back("decoder.queries.feat");
    names.emplace_back("decoder.queries.refpoints");

    // --- Decoder ref_point_head (2-layer MLP) ---
    names.emplace_back("decoder.ref_point_head.layers.0.weight");
    names.emplace_back("decoder.ref_point_head.layers.0.bias");
    names.emplace_back("decoder.ref_point_head.layers.1.weight");
    names.emplace_back("decoder.ref_point_head.layers.1.bias");

    // --- Decoder layers (22 each) ---
    for (uint32_t i = 0; i < cfg.decoder.layers; ++i) {
        std::string p = "decoder.layers." + std::to_string(i) + ".";
        names.emplace_back(p + "self_attn.in_proj.weight");
        names.emplace_back(p + "self_attn.in_proj.bias");
        names.emplace_back(p + "self_attn.out_proj.weight");
        names.emplace_back(p + "self_attn.out_proj.bias");
        names.emplace_back(p + "norm1.weight");
        names.emplace_back(p + "norm1.bias");
        names.emplace_back(p + "cross_attn.sampling_offsets.weight");
        names.emplace_back(p + "cross_attn.sampling_offsets.bias");
        names.emplace_back(p + "cross_attn.attention_weights.weight");
        names.emplace_back(p + "cross_attn.attention_weights.bias");
        names.emplace_back(p + "cross_attn.value_proj.weight");
        names.emplace_back(p + "cross_attn.value_proj.bias");
        names.emplace_back(p + "cross_attn.output_proj.weight");
        names.emplace_back(p + "cross_attn.output_proj.bias");
        names.emplace_back(p + "norm2.weight");
        names.emplace_back(p + "norm2.bias");
        names.emplace_back(p + "linear1.weight");
        names.emplace_back(p + "linear1.bias");
        names.emplace_back(p + "linear2.weight");
        names.emplace_back(p + "linear2.bias");
        names.emplace_back(p + "norm3.weight");
        names.emplace_back(p + "norm3.bias");
    }

    // --- Decoder final norm ---
    names.emplace_back("decoder.norm.weight");
    names.emplace_back("decoder.norm.bias");

    // --- Heads (shared single instances) ---
    names.emplace_back("heads.class_embed.weight");
    names.emplace_back("heads.class_embed.bias");
    for (int j = 0; j < 3; ++j) {
        const std::string ji = std::to_string(j);
        names.emplace_back("heads.bbox_embed.layers." + ji + ".weight");
        names.emplace_back("heads.bbox_embed.layers." + ji + ".bias");
    }

    // --- Segmentation head (RFDETRSeg* variants only) ---
    if (cfg.has_segmentation_head) {
        // SegmentationHead builds one DepthwiseConvBlock per decoder layer
        // (rfdetr.models.heads.segmentation.SegmentationHead): 4 for
        // nano/small, 5 for medium/large, 6 for xlarge/2xlarge.
        for (uint32_t b = 0; b < cfg.decoder.layers; ++b) {
            const std::string p = "segmentation_head.blocks." + std::to_string(b) + ".";
            names.emplace_back(p + "dwconv.weight");
            names.emplace_back(p + "dwconv.bias");
            names.emplace_back(p + "norm.weight");
            names.emplace_back(p + "norm.bias");
            names.emplace_back(p + "pwconv1.weight");
            names.emplace_back(p + "pwconv1.bias");
        }
        names.emplace_back("segmentation_head.spatial_features_proj.weight");
        names.emplace_back("segmentation_head.spatial_features_proj.bias");
        names.emplace_back("segmentation_head.query_features_block.norm_in.weight");
        names.emplace_back("segmentation_head.query_features_block.norm_in.bias");
        names.emplace_back("segmentation_head.query_features_block.layers.0.weight");
        names.emplace_back("segmentation_head.query_features_block.layers.0.bias");
        names.emplace_back("segmentation_head.query_features_block.layers.2.weight");
        names.emplace_back("segmentation_head.query_features_block.layers.2.bias");
        names.emplace_back("segmentation_head.query_features_proj.weight");
        names.emplace_back("segmentation_head.query_features_proj.bias");
        names.emplace_back("segmentation_head.bias");
    }

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
