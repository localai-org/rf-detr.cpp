#include "rfdetr.h"
#include "cli.hpp"
#include "image_io.hpp"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <vector>

static void default_log_cb(rfdetr_log_level lvl, const char* msg, void* /*ud*/) {
    const char* tag = "?";
    switch (lvl) {
        case RFDETR_LOG_DEBUG: tag = "DEBUG"; break;
        case RFDETR_LOG_INFO:  tag = "INFO";  break;
        case RFDETR_LOG_WARN:  tag = "WARN";  break;
        case RFDETR_LOG_ERROR: tag = "ERROR"; break;
    }
    std::fprintf(stderr, "[%s] %s\n", tag, msg);
}

/* Resolve --threads N:
 *   - N > 0          → use N
 *   - N == 0 (auto)  → use std::thread::hardware_concurrency() (>=1)
 *   - N < 0          → clamped to 1
 */
static int resolve_n_threads(int requested) {
    if (requested > 0) return requested;
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 1;
    return (int)hc;
}

static int cmd_detect(const rfdetr_cli::DetectArgs& a) {
    /* 1. Initialize model context */
    rfdetr_params params{};
    params.model_path = a.model.c_str();
    params.n_threads  = resolve_n_threads(a.n_threads);

    rfdetr_status init_st;
    rfdetr_context* ctx = rfdetr_init(&params, &init_st);
    if (!ctx) {
        std::fprintf(stderr, "rfdetr_init failed: %s\n",
                     rfdetr_status_str(init_st));
        return 2;
    }

    /* 2. Load input image */
    rfdetr_status load_st;
    rfdetr_image* img = rfdetr_image_load_file(a.input.c_str(), &load_st);
    if (!img) {
        std::fprintf(stderr, "failed to load image '%s': %s\n",
                     a.input.c_str(), rfdetr_status_str(load_st));
        rfdetr_free(ctx);
        return 3;
    }

    /* 3. Build detect params from CLI args */
    rfdetr_detect_params dp{};
    dp.threshold        = a.threshold;
    dp.top_k            = a.top_k;
    dp.class_filter     = a.classes.empty() ? nullptr : a.classes.data();
    dp.class_filter_len = a.classes.size();

    /* 4. Run detection */
    rfdetr_detection* dets = nullptr;
    size_t n = 0;
    rfdetr_status det_st = rfdetr_detect(ctx, img, &dp, &dets, &n);
    if (det_st != RFDETR_OK) {
        std::fprintf(stderr, "rfdetr_detect failed: %s\n",
                     rfdetr_status_str(det_st));
        rfdetr_image_free(img);
        rfdetr_free(ctx);
        return 4;
    }

    /* 5. Write JSON output */
    std::ofstream out(a.output);
    if (!out.is_open()) {
        std::fprintf(stderr, "failed to open '%s' for writing\n", a.output.c_str());
        rfdetr_detections_free(dets, n);
        rfdetr_image_free(img);
        rfdetr_free(ctx);
        return 5;
    }
    out << "{\n";
    out << "  \"image\": {\"width\": " << rfdetr_image_width(img)
        << ", \"height\": " << rfdetr_image_height(img) << "},\n";
    out << "  \"detections\": [";
    for (size_t i = 0; i < n; ++i) {
        out << (i ? ",\n    " : "\n    ");
        out << "{"
            << "\"class_id\": " << dets[i].class_id
            << ", \"class_name\": \""
            << (dets[i].class_name ? dets[i].class_name : "")
            << "\""
            << ", \"score\": " << dets[i].score
            << ", \"bbox\": ["
            << dets[i].x1 << ", " << dets[i].y1 << ", "
            << dets[i].x2 << ", " << dets[i].y2
            << "]";
        if (dets[i].mask && dets[i].mask_width > 0 && dets[i].mask_height > 0) {
            out << ", \"mask_width\": " << dets[i].mask_width
                << ", \"mask_height\": " << dets[i].mask_height;
        }
        out << "}";
    }
    if (n > 0) out << "\n  ";
    out << "]\n}\n";
    out.close();

    /* 6. Optional annotated PNG */
    if (!a.annotated.empty()) {
        rfdetr_status render_st = rfdetr_render(img, dets, n, a.annotated.c_str());
        if (render_st != RFDETR_OK) {
            std::fprintf(stderr, "rfdetr_render failed: %s\n",
                         rfdetr_status_str(render_st));
        }
    }

    /* 7. Optional per-detection mask PNGs (seg models only). */
    if (!a.masks_dir.empty()) {
        /* Create the masks directory if it doesn't exist. */
        struct stat st_buf;
        if (::stat(a.masks_dir.c_str(), &st_buf) != 0) {
            if (::mkdir(a.masks_dir.c_str(), 0755) != 0) {
                std::fprintf(stderr, "failed to create masks dir '%s'\n",
                             a.masks_dir.c_str());
            }
        }
        size_t n_written = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!dets[i].mask || dets[i].mask_width <= 0 || dets[i].mask_height <= 0) {
                continue;
            }
            char path[1024];
            std::snprintf(path, sizeof(path),
                          "%s/det_%03zu_class%u_score%02d.png",
                          a.masks_dir.c_str(),
                          i, dets[i].class_id,
                          (int)(dets[i].score * 100.0f));
            rfdetr_status wst = rfdetr_write_gray_png(
                path, dets[i].mask, dets[i].mask_width, dets[i].mask_height);
            if (wst == RFDETR_OK) ++n_written;
        }
        std::fprintf(stderr, "wrote %zu mask PNGs to %s\n",
                     n_written, a.masks_dir.c_str());
    }

    /* 8. Cleanup */
    rfdetr_detections_free(dets, n);
    rfdetr_image_free(img);
    rfdetr_free(ctx);
    return 0;
}

static int cmd_bench(const rfdetr_cli::BenchArgs& a) {
    /* 1. Initialize model context (load happens once). */
    rfdetr_params params{};
    params.model_path = a.model.c_str();
    params.n_threads  = resolve_n_threads(a.n_threads);

    using clock = std::chrono::steady_clock;
    auto t_load_start = clock::now();
    rfdetr_status init_st;
    rfdetr_context* ctx = rfdetr_init(&params, &init_st);
    auto t_load_end = clock::now();
    if (!ctx) {
        std::fprintf(stderr, "rfdetr_init failed: %s\n",
                     rfdetr_status_str(init_st));
        return 2;
    }
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    /* 2. Load input image once. */
    rfdetr_status load_st;
    rfdetr_image* img = rfdetr_image_load_file(a.input.c_str(), &load_st);
    if (!img) {
        std::fprintf(stderr, "failed to load image '%s': %s\n",
                     a.input.c_str(), rfdetr_status_str(load_st));
        rfdetr_free(ctx);
        return 3;
    }

    rfdetr_detect_params dp{};
    dp.threshold = 0.5f;
    dp.top_k     = 300;

    const int warmup = std::max(0, a.warmup);
    const int iters  = std::max(1, a.iters);

    std::printf("model:     %s\n", a.model.c_str());
    std::printf("image:     %s (%dx%d)\n", a.input.c_str(),
                rfdetr_image_width(img), rfdetr_image_height(img));
    std::printf("threads:   %d\n", params.n_threads);
    std::printf("load_ms:   %.2f\n", load_ms);
    std::printf("warmup:    %d\n", warmup);
    std::printf("iters:     %d\n", iters);
    std::fflush(stdout);

    /* 3. Warmup. */
    for (int i = 0; i < warmup; ++i) {
        rfdetr_detection* dets = nullptr;
        size_t n = 0;
        rfdetr_status st = rfdetr_detect(ctx, img, &dp, &dets, &n);
        if (st != RFDETR_OK) {
            std::fprintf(stderr, "warmup %d: rfdetr_detect failed: %s\n", i,
                         rfdetr_status_str(st));
            rfdetr_image_free(img);
            rfdetr_free(ctx);
            return 4;
        }
        rfdetr_detections_free(dets, n);
    }

    /* 4. Timed iterations. */
    std::vector<double> ms_per_iter;
    ms_per_iter.reserve((size_t)iters);
    size_t last_n = 0;
    for (int i = 0; i < iters; ++i) {
        rfdetr_detection* dets = nullptr;
        size_t n = 0;
        auto t0 = clock::now();
        rfdetr_status st = rfdetr_detect(ctx, img, &dp, &dets, &n);
        auto t1 = clock::now();
        if (st != RFDETR_OK) {
            std::fprintf(stderr, "iter %d: rfdetr_detect failed: %s\n", i,
                         rfdetr_status_str(st));
            rfdetr_image_free(img);
            rfdetr_free(ctx);
            return 4;
        }
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ms_per_iter.push_back(ms);
        last_n = n;
        rfdetr_detections_free(dets, n);
    }

    /* 5. Aggregate. */
    std::vector<double> sorted = ms_per_iter;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0.0;
    for (double v : ms_per_iter) sum += v;
    double mean   = sum / (double)ms_per_iter.size();
    double minv   = sorted.front();
    double maxv   = sorted.back();
    double median = sorted[sorted.size() / 2];

    std::printf("detections: %zu\n", last_n);
    std::printf("min_ms:    %.2f\n", minv);
    std::printf("median_ms: %.2f\n", median);
    std::printf("mean_ms:   %.2f\n", mean);
    std::printf("max_ms:    %.2f\n", maxv);

    rfdetr_image_free(img);
    rfdetr_free(ctx);
    return 0;
}

/* ---- Quantize subcommand ---------------------------------------------------
 *
 * Reads an input rfdetr GGUF, copies metadata KV pairs unchanged, and either
 * copies or quantizes each tensor into a new output GGUF.
 *
 * The `should_quantize` heuristic mirrors scripts/convert_rfdetr_to_gguf.py:
 *   - tensor name must end with ".weight"
 *   - exactly 2 dims
 *   - both dims >= 64
 *   - innermost dim divisible by 32 (legacy quants) or 256 (K-quants)
 *   - not in the embedding skiplist (pos_embed / decoder query embeddings,
 *     which are not used as mul_mat multiplicands)
 *
 * Legacy quants (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) produce byte-for-byte identical
 * tensor data as the Python converter when the input is the matching F32 GGUF.
 * K-quants are only available here (the Python `gguf` package's quantize_blocks
 * raises NotImplementedError for them).
 * --------------------------------------------------------------------------- */

namespace {

bool quant_file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

struct DtypeInfo {
    const char* name;
    ggml_type   type;
};

static const DtypeInfo kDtypeTable[] = {
    {"f32",  GGML_TYPE_F32},
    {"f16",  GGML_TYPE_F16},
    {"q4_0", GGML_TYPE_Q4_0},
    {"q4_1", GGML_TYPE_Q4_1},
    {"q5_0", GGML_TYPE_Q5_0},
    {"q5_1", GGML_TYPE_Q5_1},
    {"q8_0", GGML_TYPE_Q8_0},
    {"q4_k", GGML_TYPE_Q4_K},
    {"q5_k", GGML_TYPE_Q5_K},
    {"q6_k", GGML_TYPE_Q6_K},
};

bool parse_dtype(const std::string& s_in, ggml_type& out) {
    std::string s = s_in;
    for (auto& c : s) {
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    }
    for (const auto& e : kDtypeTable) {
        if (s == e.name) { out = e.type; return true; }
    }
    return false;
}

const char* type_name(ggml_type t) {
    return ggml_type_name(t);
}

bool tensor_in_skiplist(const char* name) {
    /* Embeddings: 2D but indexed/broadcast, never used in mul_mat. Match
     * scripts/convert_rfdetr_to_gguf.py:should_quantize. */
    static const char* kSkip[] = {
        "backbone.pos_embed",
        "decoder.queries.feat",
        "decoder.queries.refpoints",
    };
    for (const char* s : kSkip) {
        if (std::strcmp(name, s) == 0) return true;
    }
    return false;
}

bool should_quantize_tensor(const char* name, const ggml_tensor* t, ggml_type target) {
    /* Only quantize 2D `.weight` tensors. */
    const size_t nlen = std::strlen(name);
    static const char kWeightSuffix[] = ".weight";
    const size_t wlen = sizeof(kWeightSuffix) - 1;
    if (nlen < wlen || std::strcmp(name + nlen - wlen, kWeightSuffix) != 0) return false;

    if (ggml_n_dims(t) != 2) return false;
    if (t->ne[0] < 64 || t->ne[1] < 64) return false;
    if (tensor_in_skiplist(name)) return false;

    /* The innermost row dimension (ne[0]) must be a multiple of the target
     * type's block size. For legacy 32-element quants this is always true on
     * rfdetr-base (all 2D weights have ne[0] % 32 == 0). For K-quants the
     * block size is 256 — a few tensors with ne[0] = 128 (decoder's 128-dim
     * MLP halves) will skip and stay F32. */
    const int64_t blck = ggml_blck_size(target);
    if (t->ne[0] % blck != 0) return false;
    return true;
}

/* Dequantize an arbitrary ggml type to F32 using the type traits table.
 * Returns true on success. */
bool dequantize_to_f32(const ggml_tensor* t, std::vector<float>& out) {
    const int64_t n = ggml_nelements(t);
    out.assign((size_t)n, 0.0f);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, (size_t)n * sizeof(float));
        return true;
    }
    if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t*>(t->data),
                              out.data(), n);
        return true;
    }
    const ggml_type_traits* tr = ggml_get_type_traits(t->type);
    if (!tr || !tr->to_float) return false;
    tr->to_float(t->data, out.data(), n);
    return true;
}

}  // namespace

static int cmd_quantize(const rfdetr_cli::QuantizeArgs& a) {
    ggml_type target;
    if (!parse_dtype(a.type, target)) {
        std::fprintf(stderr,
                     "quantize: unknown dtype '%s' (expected one of: "
                     "f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q4_K, q5_K, q6_K)\n",
                     a.type.c_str());
        return 2;
    }
    if (ggml_quantize_requires_imatrix(target)) {
        std::fprintf(stderr,
                     "quantize: dtype '%s' requires an importance matrix "
                     "(imatrix). Not supported by this CLI.\n",
                     a.type.c_str());
        return 2;
    }

    if (!quant_file_exists(a.input)) {
        std::fprintf(stderr, "quantize: input file not found: %s\n", a.input.c_str());
        return 3;
    }

    /* 1. Open input GGUF. no_alloc=false so the gguf_init owns a ggml_context
     *    with tensor data malloc'd inside it. We need to read the bytes. */
    ggml_context* in_ctx = nullptr;
    gguf_init_params ip{};
    ip.no_alloc = false;
    ip.ctx      = &in_ctx;
    gguf_context* in_gguf = gguf_init_from_file(a.input.c_str(), ip);
    if (!in_gguf) {
        std::fprintf(stderr, "quantize: failed to open '%s'\n", a.input.c_str());
        return 4;
    }

    /* 2. Build output gguf + a scratch ggml_context to hold the rewritten
     *    tensor descriptors. The scratch ctx is sized to hold (n_tensors)
     *    descriptors plus the underlying buffers we manage ourselves. */
    const int64_t n_tensors = gguf_get_n_tensors(in_gguf);
    gguf_context* out_gguf = gguf_init_empty();
    if (!out_gguf) {
        std::fprintf(stderr, "quantize: gguf_init_empty failed\n");
        gguf_free(in_gguf);
        ggml_free(in_ctx);
        return 5;
    }

    ggml_init_params ep{};
    ep.mem_size = ggml_tensor_overhead() * (size_t)(n_tensors + 8);
    ep.mem_buffer = nullptr;
    ep.no_alloc = true;
    ggml_context* out_ctx = ggml_init(ep);
    if (!out_ctx) {
        std::fprintf(stderr, "quantize: ggml_init for out_ctx failed\n");
        gguf_free(out_gguf);
        gguf_free(in_gguf);
        ggml_free(in_ctx);
        return 6;
    }

    /* 3. Copy KV pairs (metadata) verbatim. gguf_set_kv preserves insertion
     *    order from the source, so subsequent code that depends on key order
     *    is unaffected. */
    gguf_set_kv(out_gguf, in_gguf);

    /* 4. Walk every tensor. Per-tensor buffer ownership: we own a vector of
     *    `std::vector<uint8_t>` that backs each new tensor's data ptr; they
     *    live until after gguf_write_to_file completes. */
    std::vector<std::vector<uint8_t>> tensor_data_owners;
    tensor_data_owners.reserve((size_t)n_tensors);

    ggml_quantize_init(target);

    int n_quantized = 0;
    int n_kept_f32  = 0;
    int n_kept_f16  = 0;
    int n_kept_other = 0;
    int n_kquant_fallback = 0;
    int n_kquant_fallback_q8 = 0;
    size_t in_total_bytes  = 0;
    size_t out_total_bytes = 0;

    /* For K-quants, fall back to Q8_0 (still much smaller than F32) for
     * tensors whose ne[0] isn't a multiple of 256 but IS a multiple of 32.
     * On rfdetr-base this covers the dim-384 backbone weights, keeping the
     * compression close to legacy Q8_0 instead of leaking 60 large tensors
     * out as F32. */
    const bool is_kquant_target =
        (target == GGML_TYPE_Q4_K || target == GGML_TYPE_Q5_K ||
         target == GGML_TYPE_Q6_K);
    if (is_kquant_target) {
        ggml_quantize_init(GGML_TYPE_Q8_0);
    }

    std::vector<float> f32_buf;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(in_gguf, i);
        ggml_tensor* src = ggml_get_tensor(in_ctx, name);
        if (!src || !src->data) {
            std::fprintf(stderr, "quantize: tensor '%s' has no data\n", name);
            ggml_free(out_ctx);
            gguf_free(out_gguf);
            gguf_free(in_gguf);
            ggml_free(in_ctx);
            return 7;
        }

        const size_t src_nbytes = ggml_nbytes(src);
        in_total_bytes += src_nbytes;

        bool want_quant = should_quantize_tensor(name, src, target);
        ggml_type used_type = target;

        /* K-quant fallback: if the row size isn't a multiple of 256 but IS a
         * multiple of 32, quantize to Q8_0 instead of leaking the tensor out
         * as F32. The legacy heuristic still applies (must be 2D weight, both
         * dims >= 64, not in embedding skiplist). */
        if (!want_quant && is_kquant_target) {
            const size_t nlen = std::strlen(name);
            if (nlen >= 7 && std::strcmp(name + nlen - 7, ".weight") == 0 &&
                ggml_n_dims(src) == 2 &&
                src->ne[0] >= 64 && src->ne[1] >= 64 &&
                !tensor_in_skiplist(name) &&
                src->ne[0] % ggml_blck_size(GGML_TYPE_Q8_0) == 0) {
                used_type = GGML_TYPE_Q8_0;
                want_quant = true;
                ++n_kquant_fallback_q8;
            }
        }

        ggml_tensor* dst = nullptr;
        std::vector<uint8_t> dst_bytes;

        if (want_quant) {
            /* Quantize: dequant src to F32 first (handles F32 passthrough and
             * F16 source files), then call ggml_quantize_chunk. */
            if (!dequantize_to_f32(src, f32_buf)) {
                std::fprintf(stderr,
                             "quantize: cannot dequantize tensor '%s' (type=%s)\n",
                             name, type_name(src->type));
                ggml_free(out_ctx);
                gguf_free(out_gguf);
                gguf_free(in_gguf);
                ggml_free(in_ctx);
                return 8;
            }

            const int64_t n_per_row = src->ne[0];
            const int64_t nrows     = (int64_t)(ggml_nelements(src) / n_per_row);
            const size_t  qbytes    = ggml_row_size(used_type, n_per_row) * (size_t)nrows;
            dst_bytes.resize(qbytes);
            const size_t actually =
                ggml_quantize_chunk(used_type, f32_buf.data(), dst_bytes.data(),
                                    /*start=*/0, nrows, n_per_row,
                                    /*imatrix=*/nullptr);
            if (actually != qbytes) {
                std::fprintf(stderr,
                             "quantize: ggml_quantize_chunk size mismatch for '%s': "
                             "got %zu, expected %zu\n",
                             name, actually, qbytes);
                ggml_free(out_ctx);
                gguf_free(out_gguf);
                gguf_free(in_gguf);
                ggml_free(in_ctx);
                return 9;
            }

            const int64_t ne_dims[GGML_MAX_DIMS] = {src->ne[0], src->ne[1], src->ne[2], src->ne[3]};
            dst = ggml_new_tensor(out_ctx, used_type, ggml_n_dims(src), ne_dims);
            ggml_set_name(dst, name);
            if (used_type == target) {
                ++n_quantized;
            }
        } else {
            /* Copy as-is: same type, same shape, same bytes.
             *
             * Edge case: target is a K-quant but the source tensor would have
             * passed the legacy heuristic except for the row-size constraint
             * (ne[0] % 256 != 0). Log it once. */
            if (ggml_is_quantized(target)) {
                const size_t nlen = std::strlen(name);
                if (nlen >= 7 && std::strcmp(name + nlen - 7, ".weight") == 0 &&
                    ggml_n_dims(src) == 2 &&
                    src->ne[0] >= 64 && src->ne[1] >= 64 &&
                    !tensor_in_skiplist(name) &&
                    src->ne[0] % ggml_blck_size(target) != 0) {
                    std::fprintf(stderr,
                                 "  [fallback] %s: ne[0]=%lld not divisible by "
                                 "blck_size(%s)=%d — keeping as F32\n",
                                 name, (long long)src->ne[0], type_name(target),
                                 (int)ggml_blck_size(target));
                    ++n_kquant_fallback;
                }
            }

            dst_bytes.assign((const uint8_t*)src->data,
                             (const uint8_t*)src->data + src_nbytes);
            const int64_t ne_dims[GGML_MAX_DIMS] = {src->ne[0], src->ne[1], src->ne[2], src->ne[3]};
            dst = ggml_new_tensor(out_ctx, src->type, ggml_n_dims(src), ne_dims);
            ggml_set_name(dst, name);
            if (src->type == GGML_TYPE_F32)      ++n_kept_f32;
            else if (src->type == GGML_TYPE_F16) ++n_kept_f16;
            else                                 ++n_kept_other;
        }

        /* Wire data pointer (gguf writer reads from dst->data during write). */
        tensor_data_owners.emplace_back(std::move(dst_bytes));
        dst->data = tensor_data_owners.back().data();

        gguf_add_tensor(out_gguf, dst);
        out_total_bytes += ggml_nbytes(dst);
    }

    /* 5. Write the output file. */
    if (!gguf_write_to_file(out_gguf, a.output.c_str(), /*only_meta=*/false)) {
        std::fprintf(stderr, "quantize: gguf_write_to_file failed for '%s'\n",
                     a.output.c_str());
        ggml_free(out_ctx);
        gguf_free(out_gguf);
        gguf_free(in_gguf);
        ggml_free(in_ctx);
        return 10;
    }

    /* 6. Summary. */
    std::printf("input:        %s (%.2f MB on disk)\n",
                a.input.c_str(),
                (double)in_total_bytes / (1024.0 * 1024.0));
    std::printf("output:       %s\n", a.output.c_str());
    std::printf("type:         %s\n", type_name(target));
    std::printf("tensors:      %lld total\n", (long long)n_tensors);
    std::printf("  quantized:  %d  -> %s\n", n_quantized, type_name(target));
    if (n_kquant_fallback_q8 > 0)
        std::printf("  k-quant Q8_0 fallback (row != 256x): %d  -> q8_0\n",
                    n_kquant_fallback_q8);
    std::printf("  kept_f32:   %d\n", n_kept_f32);
    if (n_kept_f16 > 0)
        std::printf("  kept_f16:   %d\n", n_kept_f16);
    if (n_kept_other > 0)
        std::printf("  kept_other: %d\n", n_kept_other);
    if (n_kquant_fallback > 0)
        std::printf("  k-quant fallbacks (kept F32): %d\n", n_kquant_fallback);

    struct stat ost;
    if (::stat(a.output.c_str(), &ost) == 0) {
        std::printf("size_in:      %.2f MB (tensor data)\n",
                    (double)in_total_bytes / (1024.0 * 1024.0));
        std::printf("size_out:     %.2f MB (file on disk, incl. meta)\n",
                    (double)ost.st_size / (1024.0 * 1024.0));
        if (in_total_bytes > 0) {
            std::printf("compression:  %.2fx\n",
                        (double)in_total_bytes / (double)out_total_bytes);
        }
    }

    ggml_free(out_ctx);
    gguf_free(out_gguf);
    gguf_free(in_gguf);
    ggml_free(in_ctx);
    return 0;
}

static int cmd_info(const rfdetr_cli::InfoArgs& a) {
    rfdetr_params p{};
    p.model_path = a.model.c_str();
    p.n_threads  = resolve_n_threads(a.n_threads);

    rfdetr_status st;
    rfdetr_context* ctx = rfdetr_init(&p, &st);
    if (!ctx) {
        std::fprintf(stderr, "rfdetr_init failed: %s\n", rfdetr_status_str(st));
        return 2;
    }

    std::printf("variant:      %s\n", rfdetr_context_variant(ctx));
    std::printf("image_size:   %u\n", rfdetr_context_image_size(ctx));
    std::printf("num_classes:  %u\n", rfdetr_context_num_classes(ctx));
    std::printf("num_queries:  %u\n", rfdetr_context_num_queries(ctx));
    std::printf("n_tensors:    %zu\n", rfdetr_context_n_tensors(ctx));

    rfdetr_free(ctx);
    return 0;
}

int main(int argc, char** argv) {
    rfdetr_set_log_callback(default_log_cb, nullptr);

    auto r = rfdetr_cli::parse(argc, argv);

    if (!r.error.empty()) {
        std::fprintf(stderr, "error: %s\n\n", r.error.c_str());
        rfdetr_cli::print_help();
        return 1;
    }

    switch (r.sub) {
        case rfdetr_cli::Subcommand::Help:
            rfdetr_cli::print_help();
            return 0;
        case rfdetr_cli::Subcommand::Detect:
            return cmd_detect(r.detect);
        case rfdetr_cli::Subcommand::Info:
            return cmd_info(r.info);
        case rfdetr_cli::Subcommand::Bench:
            return cmd_bench(r.bench);
        case rfdetr_cli::Subcommand::Quantize:
            return cmd_quantize(r.quantize);
        case rfdetr_cli::Subcommand::Compare:
            std::fprintf(stderr, "this subcommand is not yet implemented (see Plan 3)\n");
            return 99;
        case rfdetr_cli::Subcommand::None:
            rfdetr_cli::print_help();
            return 1;
    }
    return 1;
}
