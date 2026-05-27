/* Plan 9: C++ MultiScaleProjector vs torch parity baseline.
 *
 * Two phases:
 *   Phase 1 — projector-only.  Feed the torch baseline's backbone multiscale
 *             outputs (parity.backbone.multiscale.level{0..3}) DIRECTLY into
 *             the C++ projector.  Bypasses the backbone so Plan 8 drift can't
 *             leak in.  Target: max_abs ≤ 1e-4 per checkpoint.
 *   Phase 2 — backbone + projector.  Real input → C++ backbone → C++
 *             projector → diff vs parity.projector.output.  Reports the
 *             "real" end-to-end drift for the next plans to track.
 *
 * Skips gracefully when either the model GGUF or the baseline isn't present. */
#include "test_assert.hpp"
#include "rfdetr.h"
#include "model_loader.hpp"
#include "backend.hpp"
#include "trace.hpp"
#include "dinov2.hpp"
#include "projector.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <sys/stat.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    std::vector<int64_t> input_shape;
    std::vector<float> input_data;
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

    int64_t kid = gguf_find_key(g, "parity.input_shape");
    RFDETR_ASSERT(kid >= 0);
    size_t n_shape = gguf_get_arr_n(g, kid);
    const int32_t* sd = (const int32_t*)gguf_get_arr_data(g, kid);
    for (size_t i = 0; i < n_shape; ++i) b.input_shape.push_back(sd[i]);

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

        if (key == "preprocess.input") {
            b.input_data = std::move(v);
        } else {
            b.shapes.emplace(key, std::move(shape));
            b.tensors.emplace(std::move(key), std::move(v));
        }
    }

    gguf_free(g);
    ggml_free(ctx);
    return b;
}

struct DiffStats {
    float max_abs   = 0.0f;
    float mean_abs  = 0.0f;
    float rms       = 0.0f;
    size_t n_over   = 0;
    size_t n_total  = 0;
    size_t max_idx  = 0;
    float got_at_max  = 0.0f;
    float want_at_max = 0.0f;
};

DiffStats diff(const std::vector<float>& got, const std::vector<float>& want,
               float atol) {
    DiffStats s;
    s.n_total = got.size();
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
        if (d > atol) ++s.n_over;
    }
    if (!got.empty()) {
        s.mean_abs = (float)(sum_abs / (double)got.size());
        s.rms      = (float)std::sqrt(sum_sq / (double)got.size());
    }
    return s;
}

const std::vector<std::string> kCheckpoints = {
    "projector.cv1.output",
    "projector.bottleneck.0.output",
    "projector.bottleneck.1.output",
    "projector.bottleneck.2.output",
    "projector.cv2.output",
    "projector.final_norm.output",
    "projector.output",
};

/* Phase 1: feed baseline backbone multiscale outputs straight into the C++
 * projector. Returns the number of structural failures (size mismatches /
 * missing tensors), separately from numeric drift. */
int run_phase1(rfdetr::Model* m, ggml_backend_t backend, const Baseline& base) {
    /* Build a graph with 4 fresh input tensors representing the multiscale
     * features, fed by the baseline data. */
    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context* gctx = ggml_init(ip);
    RFDETR_ASSERT(gctx != nullptr);

    rfdetr::BackboneOutput fake_bb;
    std::vector<int64_t> ms_shape;  // (W, H, C, 1)
    for (int j = 0; j < 4; ++j) {
        const std::string name = "backbone.multiscale.level" + std::to_string(j);
        auto it = base.tensors.find(name);
        if (it == base.tensors.end()) {
            std::fprintf(stderr, "[phase1] missing baseline %s\n", name.c_str());
            ggml_free(gctx);
            return 1;
        }
        const auto& sh = base.shapes.at(name);
        if (j == 0) ms_shape = sh;
        ggml_tensor* t = ggml_new_tensor_4d(gctx, GGML_TYPE_F32,
                                            sh[0], sh[1], sh[2], sh[3]);
        ggml_set_name(t, name.c_str());
        fake_bb.multi_scale[j] = t;
    }
    fake_bb.final = fake_bb.multi_scale[3];  // unused by projector but keep non-null

    std::map<std::string, const ggml_tensor*> traced;
    rfdetr::set_trace_callback(
        [&](const std::string& name, const ggml_tensor* tt) {
            traced[name] = tt;
        });

    ggml_tensor* projected = rfdetr::projector_forward(gctx, *m, fake_bb);
    RFDETR_ASSERT(projected != nullptr);

    ggml_cgraph* graph = ggml_new_graph_custom(gctx, /*size*/ 8192, /*grads*/ false);
    ggml_build_forward_expand(graph, projected);
    for (const auto& [_, tt] : traced) {
        ggml_build_forward_expand(graph, const_cast<ggml_tensor*>(tt));
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
    RFDETR_ASSERT(buf != nullptr);

    for (int j = 0; j < 4; ++j) {
        const std::string name = "backbone.multiscale.level" + std::to_string(j);
        const auto& v = base.tensors.at(name);
        ggml_backend_tensor_set(fake_bb.multi_scale[j], v.data(),
                                0, v.size() * sizeof(float));
    }

    auto status = ggml_backend_graph_compute(backend, graph);
    RFDETR_ASSERT(status == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::map<std::string, std::vector<float>> got;
    for (const auto& [name, tt] : traced) {
        got[name] = rfdetr::copy_tensor_to_f32(tt);
    }

    /* Tolerances. kTight is the ideal F32 parity target; kStruct is a sloppy
     * upper bound — anything OVER means structural bug, not drift. */
    const float kTight  = 1e-4f;
    const float kStruct = 5.0f;

    int n_tight  = 0;
    int n_loose  = 0;
    int n_struct = 0;
    int n_skip   = 0;
    int n_total  = 0;
    std::fprintf(stderr, "[phase1] projector-only (baseline backbone inputs):\n");
    for (const auto& name : kCheckpoints) {
        auto itw = base.tensors.find(name);
        if (itw == base.tensors.end()) {
            std::fprintf(stderr, "  [%s] no baseline tensor — SKIP\n", name.c_str());
            ++n_skip;
            continue;
        }
        auto itg = got.find(name);
        if (itg == got.end()) {
            std::fprintf(stderr, "  [%s] not published by C++ forward — STRUCT FAIL\n",
                         name.c_str());
            ++n_struct;
            continue;
        }
        if (itg->second.size() != itw->second.size()) {
            std::fprintf(stderr,
                "  [%s] size mismatch C++=%zu torch=%zu — STRUCT FAIL\n",
                name.c_str(), itg->second.size(), itw->second.size());
            ++n_struct;
            continue;
        }
        ++n_total;
        DiffStats ds = diff(itg->second, itw->second, kTight);
        const char* tag;
        if (ds.max_abs <= kTight) { tag = "OK";    ++n_tight; }
        else if (ds.max_abs <= kStruct) { tag = "DRIFT"; ++n_loose; }
        else                            { tag = "STRUCT"; ++n_struct; }
        std::fprintf(stderr,
            "  [%s] %-6s max_abs=%.4g mean_abs=%.4g rms=%.4g "
            "(got=%g want=%g at idx %zu)\n",
            name.c_str(), tag, ds.max_abs, ds.mean_abs, ds.rms,
            ds.got_at_max, ds.want_at_max, ds.max_idx);
    }
    std::fprintf(stderr,
        "[phase1] tight(<=%g)=%d  drift(<=%g)=%d  struct(>%g)=%d  skip=%d\n",
        (double)kTight, n_tight, (double)kStruct, n_loose,
        (double)kStruct, n_struct, n_skip);

    rfdetr::set_trace_callback(nullptr);
    ggml_backend_buffer_free(buf);
    ggml_free(gctx);

    /* Phase 1 must pass at tight tolerance — the projector is exercised in
     * isolation against ground-truth backbone outputs, so any drift here is
     * a bug, not accumulated FP32 round-off. */
    return n_struct + n_loose;
}

/* Phase 2: full backbone → projector pipeline. Reports drift only — not
 * asserted, since accumulated Plan 8 backbone drift propagates here. */
int run_phase2(rfdetr::Model* m, ggml_backend_t backend, const Baseline& base) {
    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context* gctx = ggml_init(ip);
    RFDETR_ASSERT(gctx != nullptr);

    const int64_t N = base.input_shape[0];
    const int64_t C = base.input_shape[1];
    const int64_t H = base.input_shape[2];
    const int64_t W = base.input_shape[3];
    ggml_tensor* input = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, W, H, C, N);
    ggml_set_name(input, "input");

    std::map<std::string, const ggml_tensor*> traced;
    rfdetr::set_trace_callback(
        [&](const std::string& name, const ggml_tensor* tt) {
            traced[name] = tt;
        });

    rfdetr::BackboneOutput bb = rfdetr::dinov2_forward(gctx, *m, input);
    RFDETR_ASSERT(bb.final != nullptr);
    ggml_tensor* projected = rfdetr::projector_forward(gctx, *m, bb);
    RFDETR_ASSERT(projected != nullptr);

    ggml_cgraph* graph = ggml_new_graph_custom(gctx, /*size*/ 16384, /*grads*/ false);
    ggml_build_forward_expand(graph, projected);
    for (const auto& [_, tt] : traced) {
        ggml_build_forward_expand(graph, const_cast<ggml_tensor*>(tt));
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
    RFDETR_ASSERT(buf != nullptr);
    ggml_backend_tensor_set(input, base.input_data.data(),
                            0, base.input_data.size() * sizeof(float));

    auto status = ggml_backend_graph_compute(backend, graph);
    RFDETR_ASSERT(status == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::map<std::string, std::vector<float>> got;
    for (const auto& [name, tt] : traced) {
        got[name] = rfdetr::copy_tensor_to_f32(tt);
    }

    std::fprintf(stderr, "[phase2] backbone + projector end-to-end:\n");
    int n_struct = 0;
    for (const auto& name : kCheckpoints) {
        auto itw = base.tensors.find(name);
        auto itg = got.find(name);
        if (itw == base.tensors.end() || itg == got.end()) {
            std::fprintf(stderr, "  [%s] missing — SKIP\n", name.c_str());
            continue;
        }
        if (itg->second.size() != itw->second.size()) {
            std::fprintf(stderr,
                "  [%s] size mismatch C++=%zu torch=%zu — STRUCT FAIL\n",
                name.c_str(), itg->second.size(), itw->second.size());
            ++n_struct;
            continue;
        }
        DiffStats ds = diff(itg->second, itw->second, 1e-4f);
        std::fprintf(stderr,
            "  [%s] max_abs=%.4g mean_abs=%.4g rms=%.4g "
            "(got=%g want=%g at idx %zu)\n",
            name.c_str(), ds.max_abs, ds.mean_abs, ds.rms,
            ds.got_at_max, ds.want_at_max, ds.max_idx);
    }

    rfdetr::set_trace_callback(nullptr);
    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    return n_struct;
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
            "[test_parity_projector] SKIPPED: real rfdetr-base GGUF not found.\n"
            "  Run: .venv/bin/python scripts/convert_rfdetr_to_gguf.py "
            "--variant base --dtype f32 --output models/rfdetr-base-f32.gguf\n");
        return 0;
    }
    if (!file_exists(base_path)) {
        std::fprintf(stderr,
            "[test_parity_projector] SKIPPED: baseline not present (%s).\n"
            "  Run: cmake --build build --target rfdetr_baseline_torch\n",
            base_path.c_str());
        return 0;
    }

    Baseline base = load_baseline(base_path);
    RFDETR_ASSERT(base.input_shape.size() == 4);
    RFDETR_ASSERT(!base.input_data.empty());

    rfdetr_status st = RFDETR_OK;
    rfdetr::Model* m = rfdetr::model_load(model_path, &st);
    RFDETR_ASSERT(m != nullptr);
    RFDETR_ASSERT_EQ_INT(st, RFDETR_OK);

    ggml_backend_t backend = rfdetr::init_backend(/*n_threads*/ 4, &st);
    RFDETR_ASSERT(backend != nullptr);
    RFDETR_ASSERT_EQ_INT(rfdetr::model_realize_weights(*m, backend), RFDETR_OK);

    int n_fail = run_phase1(m, backend, base);
    /* Phase 2 is informational — drift is expected to be larger than phase 1
     * because the C++ backbone's residual drift propagates through. */
    (void)run_phase2(m, backend, base);

    rfdetr::model_free(m);
    rfdetr::free_backend(backend);

    return n_fail == 0 ? 0 : 1;
}
