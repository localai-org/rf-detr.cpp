/* Plan 14: C++ heads vs torch parity baseline.
 *
 * Module-isolated test: feeds baseline `parity.decoder.norm.output` directly
 * into the C++ heads forward (class + bbox), bypassing upstream drift from
 * backbone / projector / decoder.
 *
 *   decoder_norm_out  <- baseline parity.decoder.norm.output    (256, 300, 1)
 *
 * Diffs vs baseline:
 *   heads.class_logits  → parity.heads.class_logits[..., -1]    (91, 300, 1)
 *   heads.bbox_pred     → parity.heads.bbox_pred[..., -1]       (4,  300, 1)
 *
 * The torch baseline stacks per-layer outputs along ne[3] = 3 (one per
 * decoder layer); only the last layer is consumed at inference (LWDETR uses
 * outputs_class[-1] / outputs_coord[-1]).
 *
 * Target: max_abs ≤ 1e-4. Skips gracefully if either the model GGUF or the
 * torch baseline is missing. */
#include "test_assert.hpp"
#include "rfdetr.h"
#include "model_loader.hpp"
#include "backend.hpp"
#include "trace.hpp"
#include "heads.hpp"

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
    size_t max_idx  = 0;
    float got_at_max  = 0.0f;
    float want_at_max = 0.0f;
};

DiffStats diff(const std::vector<float>& got, const std::vector<float>& want) {
    DiffStats s;
    double sum_abs = 0.0;
    const size_t n = std::min(got.size(), want.size());
    for (size_t i = 0; i < n; ++i) {
        const float d = std::fabs(got[i] - want[i]);
        sum_abs += d;
        if (d > s.max_abs) {
            s.max_abs = d;
            s.max_idx = i;
            s.got_at_max = got[i];
            s.want_at_max = want[i];
        }
    }
    if (n > 0) s.mean_abs = (float)(sum_abs / (double)n);
    return s;
}

void report(const char* name, const std::vector<float>& got,
            const std::vector<float>& want, float tol, int& n_fail) {
    if (got.size() != want.size()) {
        std::fprintf(stderr,
            "  [%-30s] size mismatch C++=%zu torch=%zu — STRUCT FAIL\n",
            name, got.size(), want.size());
        ++n_fail;
        return;
    }
    DiffStats ds = diff(got, want);
    const char* tag = (ds.max_abs <= tol) ? "OK" : "FAIL";
    std::fprintf(stderr,
        "  [%-30s] %-4s max_abs=%.4g mean_abs=%.4g (got=%g want=%g at idx %zu)\n",
        name, tag, ds.max_abs, ds.mean_abs,
        ds.got_at_max, ds.want_at_max, ds.max_idx);
    if (ds.max_abs > tol) ++n_fail;
}

/* Slice the last layer (ne[3]=2) from a 4D F32 baseline tensor stored as
 * column-major (ne0, ne1, ne2, ne3=N_layers). Returns a flat vector with
 * shape (ne0, ne1, ne2) for layer index `layer`. */
std::vector<float> slice_layer(const std::vector<float>& src,
                               const std::vector<int64_t>& shape,
                               int layer) {
    const int64_t n0 = shape[0];
    const int64_t n1 = shape[1];
    const int64_t n2 = shape[2];
    const int64_t n3 = shape[3];
    RFDETR_ASSERT(layer >= 0 && layer < (int)n3);
    const size_t per_layer = (size_t)n0 * (size_t)n1 * (size_t)n2;
    std::vector<float> out(per_layer);
    /* ne3 is the outermost axis in ggml column-major: stride = n0*n1*n2. */
    std::memcpy(out.data(), src.data() + (size_t)layer * per_layer,
                per_layer * sizeof(float));
    return out;
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
            "[test_parity_heads] SKIPPED: real rfdetr-base GGUF not found.\n");
        return 0;
    }
    if (!file_exists(base_path)) {
        std::fprintf(stderr,
            "[test_parity_heads] SKIPPED: baseline not present (%s).\n",
            base_path.c_str());
        return 0;
    }

    Baseline base = load_baseline(base_path);
    auto have = [&](const char* k) { return base.tensors.find(k) != base.tensors.end(); };
    if (!have("decoder.norm.output") ||
        !have("heads.class_logits") ||
        !have("heads.bbox_pred")) {
        std::fprintf(stderr,
            "[test_parity_heads] SKIPPED: baseline missing heads checkpoints "
            "(regenerate baseline_torch.gguf).\n");
        return 0;
    }

    rfdetr_status st = RFDETR_OK;
    rfdetr::Model* m = rfdetr::model_load(model_path, &st);
    RFDETR_ASSERT(m != nullptr);
    RFDETR_ASSERT_EQ_INT(st, RFDETR_OK);

    ggml_backend_t backend = rfdetr::init_backend(/*n_threads*/ 4, &st);
    RFDETR_ASSERT(backend != nullptr);
    RFDETR_ASSERT_EQ_INT(rfdetr::model_realize_weights(*m, backend), RFDETR_OK);

    const int dim = (int)m->config.decoder.model_dim;     // 256
    const int NQ  = (int)m->config.num_queries;           // 300
    const int NC  = (int)m->config.num_classes;           // 91

    /* Build the C++ heads graph. */
    ggml_init_params ip{};
    ip.mem_size   = 64 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context* gctx = ggml_init(ip);
    RFDETR_ASSERT(gctx != nullptr);

    ggml_tensor* dec_norm_in = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, dim, NQ, 1);
    ggml_set_name(dec_norm_in, "decoder.norm.output");

    std::map<std::string, const ggml_tensor*> traced;
    rfdetr::set_trace_callback(
        [&](const std::string& name, const ggml_tensor* tt) { traced[name] = tt; });

    ggml_tensor* cls_logits = rfdetr::class_head_forward(gctx, *m, dec_norm_in);
    RFDETR_ASSERT(cls_logits != nullptr);
    ggml_tensor* bbox_delta = rfdetr::bbox_head_forward(gctx, *m, dec_norm_in);
    RFDETR_ASSERT(bbox_delta != nullptr);

    RFDETR_ASSERT_EQ_INT(cls_logits->ne[0], NC);
    RFDETR_ASSERT_EQ_INT(cls_logits->ne[1], NQ);
    RFDETR_ASSERT_EQ_INT(bbox_delta->ne[0], 4);
    RFDETR_ASSERT_EQ_INT(bbox_delta->ne[1], NQ);

    ggml_cgraph* graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, cls_logits);
    ggml_build_forward_expand(graph, bbox_delta);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
    RFDETR_ASSERT(buf != nullptr);

    /* Populate input from the baseline. */
    const auto& dec_data = base.tensors.at("decoder.norm.output");
    RFDETR_ASSERT_EQ_INT((int64_t)dec_data.size(), (int64_t)dim * NQ);
    ggml_backend_tensor_set(dec_norm_in, dec_data.data(), 0,
                            dec_data.size() * sizeof(float));

    auto status = ggml_backend_graph_compute(backend, graph);
    RFDETR_ASSERT(status == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    rfdetr::set_trace_callback(nullptr);

    /* Compare with last-layer slice of baseline. */
    const float kTol = 1e-4f;
    int n_fail = 0;

    {
        const auto& want_full   = base.tensors.at("heads.class_logits");
        const auto& want_shape  = base.shapes.at("heads.class_logits");
        const int n_layers      = (int)want_shape[3];
        auto want_last = slice_layer(want_full, want_shape, n_layers - 1);
        auto got = rfdetr::copy_tensor_to_f32(cls_logits);
        report("heads.class_logits[-1]", got, want_last, kTol, n_fail);
    }
    {
        const auto& want_full   = base.tensors.at("heads.bbox_pred");
        const auto& want_shape  = base.shapes.at("heads.bbox_pred");
        const int n_layers      = (int)want_shape[3];
        auto want_last = slice_layer(want_full, want_shape, n_layers - 1);
        auto got = rfdetr::copy_tensor_to_f32(bbox_delta);
        report("heads.bbox_pred[-1]", got, want_last, kTol, n_fail);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    rfdetr::model_free(m);
    rfdetr::free_backend(backend);

    return n_fail == 0 ? 0 : 1;
}
