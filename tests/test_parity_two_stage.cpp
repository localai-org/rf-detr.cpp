/* Plan 10: C++ two-stage init module vs torch parity baseline.
 *
 * Module-isolated test: feeds the torch baseline's `parity.projector.output`
 * directly into the C++ two-stage forward. Bypasses backbone + projector so
 * residual drift from earlier modules doesn't leak in.
 *
 * Checkpoints:
 *   two_stage.enc_output_norm.output  (256, 1600)  — direct compare
 *   two_stage.enc_out_class.output    (91, 300)    — CPU top-K gather, then
 *                                                   compare against baseline
 *
 * Top-K reconstruction: enc_out_class_embed[0] is a per-row Linear, so
 *   class_logits[topk_idx, :]  ==  enc_out_class_embed[0](memory[topk_idx, :])
 * — i.e. gathering rows from the pre-topk class logits is equivalent to
 * re-applying the class head to the gathered memory rows. We use that to
 * avoid a second graph compile.
 *
 * Target: max_abs ≤ 1e-4. Skips gracefully when model GGUF or baseline isn't
 * present. */
#include "test_assert.hpp"
#include "rfdetr.h"
#include "model_loader.hpp"
#include "backend.hpp"
#include "trace.hpp"
#include "two_stage.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <sys/stat.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

struct Baseline {
    std::unordered_map<std::string, std::vector<float>> tensors;
    std::unordered_map<std::string, std::vector<int64_t>> shapes;
};

Baseline load_baseline(const std::string& path) {
    Baseline b;

    ggml_context* ctx = nullptr;
    gguf_init_params p{};
    p.no_alloc = false;
    p.ctx      = &ctx;
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    RFDETR_ASSERT(g != nullptr);
    RFDETR_ASSERT(ctx != nullptr);

    const int64_t nt = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < nt; ++i) {
        const char* name = gguf_get_tensor_name(g, i);
        std::string n(name);
        const std::string prefix = "parity.";
        if (n.compare(0, prefix.size(), prefix) != 0) continue;
        std::string key = n.substr(prefix.size());

        ggml_tensor* t = ggml_get_tensor(ctx, name);
        RFDETR_ASSERT(t != nullptr);
        RFDETR_ASSERT_EQ_INT(t->type, GGML_TYPE_F32);
        const size_t nelem = ggml_nelements(t);
        std::vector<float> v(nelem);
        std::memcpy(v.data(), t->data, nelem * sizeof(float));

        std::vector<int64_t> shape;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) shape.push_back(t->ne[d]);
        b.shapes.emplace(key, std::move(shape));
        b.tensors.emplace(std::move(key), std::move(v));
    }

    gguf_free(g);
    ggml_free(ctx);
    return b;
}

struct DiffStats {
    float max_abs   = 0.0f;
    float mean_abs  = 0.0f;
    float rms       = 0.0f;
    size_t max_idx  = 0;
    float got_at_max  = 0.0f;
    float want_at_max = 0.0f;
};

DiffStats diff(const std::vector<float>& got, const std::vector<float>& want) {
    DiffStats s;
    double sum_abs = 0.0;
    double sum_sq  = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float d = std::fabs(got[i] - want[i]);
        sum_abs += d;
        sum_sq  += (double)d * (double)d;
        if (d > s.max_abs) {
            s.max_abs = d;
            s.max_idx = i;
            s.got_at_max = got[i];
            s.want_at_max = want[i];
        }
    }
    if (!got.empty()) {
        s.mean_abs = (float)(sum_abs / (double)got.size());
        s.rms      = (float)std::sqrt(sum_sq / (double)got.size());
    }
    return s;
}

}  // namespace

int main() {
    const std::string fixtures   = RFDETR_TEST_FIXTURES;
    const std::string base_path  = fixtures + "/baseline_torch.gguf";

    std::string model_path;
    for (const std::string& candidate : {
            fixtures + "/rfdetr-base-f32.gguf",
            fixtures + "/../../models/rfdetr-base-f32.gguf",
        }) {
        if (file_exists(candidate)) { model_path = candidate; break; }
    }
    if (model_path.empty()) {
        std::fprintf(stderr,
            "[test_parity_two_stage] SKIPPED: real rfdetr-base GGUF not found.\n");
        return 0;
    }
    if (!file_exists(base_path)) {
        std::fprintf(stderr,
            "[test_parity_two_stage] SKIPPED: baseline not present (%s).\n",
            base_path.c_str());
        return 0;
    }

    Baseline base = load_baseline(base_path);
    auto it_proj = base.tensors.find("projector.output");
    if (it_proj == base.tensors.end()) {
        std::fprintf(stderr,
            "[test_parity_two_stage] SKIPPED: baseline missing parity.projector.output\n");
        return 0;
    }
    const auto& proj_shape = base.shapes.at("projector.output");
    /* Expected: (W=40, H=40, C=256, 1) in ggml ne order. */
    RFDETR_ASSERT_EQ_INT(proj_shape[2], 256);

    rfdetr_status st = RFDETR_OK;
    rfdetr::Model* m = rfdetr::model_load(model_path, &st);
    RFDETR_ASSERT(m != nullptr);
    RFDETR_ASSERT_EQ_INT(st, RFDETR_OK);

    ggml_backend_t backend = rfdetr::init_backend(/*n_threads*/ 4, &st);
    RFDETR_ASSERT(backend != nullptr);
    RFDETR_ASSERT_EQ_INT(rfdetr::model_realize_weights(*m, backend), RFDETR_OK);

    /* Build the graph with projector_out as a fresh input. */
    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context* gctx = ggml_init(ip);
    RFDETR_ASSERT(gctx != nullptr);

    ggml_tensor* proj_in = ggml_new_tensor_4d(gctx, GGML_TYPE_F32,
                                              proj_shape[0], proj_shape[1],
                                              proj_shape[2], proj_shape[3]);
    ggml_set_name(proj_in, "projector.output.in");

    std::map<std::string, const ggml_tensor*> traced;
    rfdetr::set_trace_callback(
        [&](const std::string& name, const ggml_tensor* tt) {
            traced[name] = tt;
        });

    rfdetr::TwoStageOutput ts = rfdetr::two_stage_forward(gctx, *m, proj_in);
    RFDETR_ASSERT(ts.enc_output_norm_out != nullptr);
    RFDETR_ASSERT(ts.cls_all != nullptr);
    RFDETR_ASSERT(ts.bbox_all != nullptr);

    ggml_cgraph* graph = ggml_new_graph_custom(gctx, /*size*/ 8192, /*grads*/ false);
    ggml_build_forward_expand(graph, ts.enc_output_norm_out);
    ggml_build_forward_expand(graph, ts.cls_all);
    ggml_build_forward_expand(graph, ts.bbox_all);
    for (const auto& [_, tt] : traced) {
        ggml_build_forward_expand(graph, const_cast<ggml_tensor*>(tt));
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
    RFDETR_ASSERT(buf != nullptr);

    const auto& proj_data = it_proj->second;
    ggml_backend_tensor_set(proj_in, proj_data.data(), 0,
                            proj_data.size() * sizeof(float));

    auto status = ggml_backend_graph_compute(backend, graph);
    RFDETR_ASSERT(status == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    /* ---- Pull tensors back. ---- */
    auto got_enc_ln  = rfdetr::copy_tensor_to_f32(traced.at("two_stage.enc_output_norm.output"));
    auto got_cls_all = rfdetr::copy_tensor_to_f32(ts.cls_all);
    auto got_bbox    = rfdetr::copy_tensor_to_f32(ts.bbox_all);

    rfdetr::set_trace_callback(nullptr);

    const float kTol = 1e-4f;
    int n_fail = 0;

    /* 1. enc_output_norm.output: shape (256, 1600). */
    {
        auto itw = base.tensors.find("two_stage.enc_output_norm.output");
        RFDETR_ASSERT(itw != base.tensors.end());
        const auto& want = itw->second;
        if (got_enc_ln.size() != want.size()) {
            std::fprintf(stderr,
                "  [enc_output_norm.output] size mismatch C++=%zu torch=%zu — STRUCT FAIL\n",
                got_enc_ln.size(), want.size());
            ++n_fail;
        } else {
            DiffStats ds = diff(got_enc_ln, want);
            const char* tag = (ds.max_abs <= kTol) ? "OK" : "FAIL";
            std::fprintf(stderr,
                "  [enc_output_norm.output]      %-4s max_abs=%.4g mean_abs=%.4g rms=%.4g "
                "(got=%g want=%g at idx %zu)\n",
                tag, ds.max_abs, ds.mean_abs, ds.rms,
                ds.got_at_max, ds.want_at_max, ds.max_idx);
            if (ds.max_abs > kTol) ++n_fail;
        }
    }

    /* 2. Top-K: per-token max class score → top-300 indices descending. */
    const int64_t N         = 1600;
    const int64_t n_classes = 91;
    const int64_t num_queries = (int64_t)m->config.num_queries;
    RFDETR_ASSERT_EQ_INT((int64_t)got_cls_all.size(), N * n_classes);
    RFDETR_ASSERT_EQ_INT(num_queries, 300);

    std::vector<float> per_token_max(N);
    for (int64_t i = 0; i < N; ++i) {
        float mx = -std::numeric_limits<float>::infinity();
        const float* row = got_cls_all.data() + (size_t)i * n_classes;
        for (int64_t c = 0; c < n_classes; ++c) {
            if (row[c] > mx) mx = row[c];
        }
        per_token_max[i] = mx;
    }

    /* Descending argsort (stable to match torch.topk tie-breaking by smaller
     * index — torch.topk is deterministic on CPU but unordered ties don't
     * happen for floats in practice). */
    std::vector<int32_t> indices(N);
    for (int64_t i = 0; i < N; ++i) indices[i] = (int32_t)i;
    std::partial_sort(indices.begin(), indices.begin() + num_queries,
                      indices.end(),
                      [&](int32_t a, int32_t b) {
                          if (per_token_max[a] != per_token_max[b])
                              return per_token_max[a] > per_token_max[b];
                          return a < b;
                      });
    indices.resize(num_queries);

    /* 3. Gather rows from cls_all by top-K indices → got_cls_topk (91, 300). */
    std::vector<float> got_cls_topk((size_t)num_queries * (size_t)n_classes);
    for (int64_t k = 0; k < num_queries; ++k) {
        const int32_t src = indices[(size_t)k];
        std::memcpy(got_cls_topk.data() + (size_t)k * n_classes,
                    got_cls_all.data() + (size_t)src * n_classes,
                    n_classes * sizeof(float));
    }

    /* 4. Compare against baseline enc_out_class.output (91, 300). */
    {
        auto itw = base.tensors.find("two_stage.enc_out_class.output");
        RFDETR_ASSERT(itw != base.tensors.end());
        const auto& want = itw->second;
        if (got_cls_topk.size() != want.size()) {
            std::fprintf(stderr,
                "  [enc_out_class.output]        size mismatch C++=%zu torch=%zu — STRUCT FAIL\n",
                got_cls_topk.size(), want.size());
            ++n_fail;
        } else {
            DiffStats ds = diff(got_cls_topk, want);
            const char* tag = (ds.max_abs <= kTol) ? "OK" : "FAIL";
            std::fprintf(stderr,
                "  [enc_out_class.output]        %-4s max_abs=%.4g mean_abs=%.4g rms=%.4g "
                "(got=%g want=%g at idx %zu)\n",
                tag, ds.max_abs, ds.mean_abs, ds.rms,
                ds.got_at_max, ds.want_at_max, ds.max_idx);
            if (ds.max_abs > kTol) ++n_fail;
        }
    }

    /* 5. bbox top-K shape check: (4, 300). bbox_all is (4, 1600); gather → (4, 300). */
    RFDETR_ASSERT_EQ_INT((int64_t)got_bbox.size(), N * 4);
    std::vector<float> got_bbox_topk((size_t)num_queries * 4);
    for (int64_t k = 0; k < num_queries; ++k) {
        const int32_t src = indices[(size_t)k];
        std::memcpy(got_bbox_topk.data() + (size_t)k * 4,
                    got_bbox.data() + (size_t)src * 4,
                    4 * sizeof(float));
    }
    std::fprintf(stderr,
        "  [refpoints]                   shape=(4, %lld) — structural OK\n",
        (long long)num_queries);

    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    rfdetr::model_free(m);
    rfdetr::free_backend(backend);

    return n_fail == 0 ? 0 : 1;
}
