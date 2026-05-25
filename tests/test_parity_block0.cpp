/* C++ forward pass through patch_embed + backbone block 0 vs numpy baseline. */
#include "test_assert.hpp"
#include "rfdetr.h"
#include "model_loader.hpp"
#include "backend.hpp"
#include "trace.hpp"
#include "dinov2.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Tol { float atol; float rtol; };

/* Plan 4 switched the fixture to F32 weights, eliminating the F16
 * quantization noise floor (~4e-4 on patch_embed) that previously forced
 * loose 1e-3 tolerances. All checkpoints share a uniform tight bound
 * that's tight enough to catch real correctness bugs and loose enough to
 * absorb ggml's F32 vs numpy's float64 order-of-operations drift. Built
 * programmatically over `depth` blocks to avoid hand-listing 4*depth+2
 * checkpoints. */
std::map<std::string, Tol> build_tolerances(const rfdetr::Config& cfg) {
    std::map<std::string, Tol> tol;
    tol["backbone.patch_embed.output"]    = {1e-5f, 1e-4f};
    tol["backbone.cls_pos_embed.output"]  = {1e-5f, 1e-4f};
    for (uint32_t i = 0; i < cfg.backbone.depth; ++i) {
        std::string p = "backbone.block." + std::to_string(i) + ".";
        tol[p + "norm1.output"] = {1e-5f, 1e-4f};
        tol[p + "attn.output"]  = {1e-5f, 1e-4f};
        tol[p + "mlp.output"]   = {1e-5f, 1e-4f};
        tol[p + "output"]       = {1e-5f, 1e-4f};
    }
    tol["backbone.norm.output"] = {1e-5f, 1e-4f};
    for (size_t k = 0; k < cfg.backbone.multi_scale_layers.size(); ++k) {
        tol["backbone.multiscale.level" + std::to_string(k)] = {1e-5f, 1e-4f};
    }
    return tol;
}

struct Baseline {
    std::unordered_map<std::string, std::vector<float>> tensors;
    std::vector<int64_t> input_shape;
    /* preprocess.input stored separately so we know its raw bytes for
     * pushing into the ggml input tensor. */
    std::vector<float> input_data;
};

Baseline load_baseline(const std::string& path) {
    Baseline b;

    ggml_context* ctx = nullptr;
    gguf_init_params p{ /*no_alloc*/ false, /*ctx*/ &ctx };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    RFDETR_ASSERT(g != nullptr);

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
        size_t nelem = ggml_nelements(t);
        std::vector<float> v(nelem);
        std::memcpy(v.data(), t->data, nelem * sizeof(float));

        if (key == "preprocess.input") {
            b.input_data = v;
        } else {
            b.tensors.emplace(std::move(key), std::move(v));
        }
    }

    gguf_free(g);
    ggml_free(ctx);
    return b;
}

bool allclose(const std::string& name,
              const std::vector<float>& got,
              const std::vector<float>& want,
              float atol, float rtol) {
    if (got.size() != want.size()) {
        std::fprintf(stderr, "[%s] size mismatch: got=%zu want=%zu\n",
                     name.c_str(), got.size(), want.size());
        return false;
    }
    float max_abs = 0.0f;
    size_t max_idx = 0;
    double sum_abs = 0.0;
    size_t fail_count = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        float diff = std::fabs(got[i] - want[i]);
        float tol  = atol + rtol * std::fabs(want[i]);
        sum_abs += diff;
        if (diff > max_abs) { max_abs = diff; max_idx = i; }
        if (diff > tol) ++fail_count;
    }
    if (fail_count == 0) {
        std::fprintf(stderr, "[%s] OK (max_abs=%g, mean_abs=%g)\n",
                     name.c_str(), max_abs, sum_abs / got.size());
        return true;
    }
    std::fprintf(stderr,
                 "[%s] FAIL (max_abs=%g at idx %zu - got=%g want=%g) "
                 "atol=%g rtol=%g, %zu/%zu over tol\n",
                 name.c_str(), max_abs, max_idx,
                 got[max_idx], want[max_idx],
                 atol, rtol, fail_count, got.size());
    return false;
}

}  // namespace

int main() {
    const std::string fixtures = RFDETR_TEST_FIXTURES;
    const std::string model_path    = fixtures + "/model_base_seeded.gguf";
    const std::string baseline_path = fixtures + "/baseline_block0.gguf";

    Baseline base = load_baseline(baseline_path);
    RFDETR_ASSERT(base.input_shape.size() == 4);
    RFDETR_ASSERT(!base.input_data.empty());

    /* Baseline input is NCHW (1, 3, 56, 56). The C++ side wants ggml shape
     * (W, H, C, N) with ne[0]=W as the fastest-varying axis. Both layouts
     * use idx = n*CHW + c*HW + h*W + w (W fastest, then H, then C, then N),
     * so the raw float bytes are identical — no transpose needed. */

    /* Open model + backend + realize weights. */
    rfdetr_status st;
    rfdetr::Model* m = rfdetr::model_load(model_path, &st);
    RFDETR_ASSERT(m != nullptr);
    ggml_backend_t backend = rfdetr::init_backend(1, &st);
    RFDETR_ASSERT(backend != nullptr);
    RFDETR_ASSERT_EQ_INT(rfdetr::model_realize_weights(*m, backend), RFDETR_OK);

    /* Build a graph for patch_embed + block 0. */
    ggml_init_params ip{};
    ip.mem_size   = 64 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context* gctx = ggml_init(ip);
    RFDETR_ASSERT(gctx != nullptr);

    const int64_t N = base.input_shape[0];   // 1
    const int64_t C = base.input_shape[1];   // 3
    const int64_t H = base.input_shape[2];   // 56
    const int64_t W = base.input_shape[3];   // 56
    /* ggml convention: (W, H, C, N) */
    ggml_tensor* input = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, W, H, C, N);
    ggml_set_name(input, "input");

    /* Install the trace callback BEFORE building the graph: publish() is
     * called inline at graph-build time and we use it to record (name ->
     * ggml_tensor*) pairs. The tensor data is read AFTER compute. */
    std::map<std::string, const ggml_tensor*> traced;
    rfdetr::set_trace_callback(
        [&](const std::string& name, const ggml_tensor* tt) {
            traced[name] = tt;
        });

    ggml_tensor* t = rfdetr::dinov2_patch_embed(gctx, *m, input);
    RFDETR_ASSERT(t != nullptr);
    t = rfdetr::dinov2_add_cls_and_pos_embed(gctx, *m, t);
    RFDETR_ASSERT(t != nullptr);
    const auto& ms_layers = m->config.backbone.multi_scale_layers;
    auto find_ms_level = [&](uint32_t block_i) -> int {
        for (size_t k = 0; k < ms_layers.size(); ++k) {
            if (ms_layers[k] == block_i) return (int)k;
        }
        return -1;
    };

    for (uint32_t i = 0; i < m->config.backbone.depth; ++i) {
        t = rfdetr::dinov2_block(gctx, *m, t, (int)i);
        RFDETR_ASSERT(t != nullptr);
        /* Multi-scale tap: publish backbone output at selected layer indices */
        int level = find_ms_level(i);
        if (level >= 0) {
            rfdetr::publish("backbone.multiscale.level" + std::to_string(level), t);
        }
    }
    t = rfdetr::dinov2_final_norm(gctx, *m, t);
    RFDETR_ASSERT(t != nullptr);

    auto kTolerances = build_tolerances(m->config);

    /* Build the graph BEFORE allocating buffers — published nodes that are
     * intermediate views (e.g. permute results) must be reachable from the
     * graph or they'd be optimized away. ggml_build_forward_expand walks
     * back from `t` and pulls in everything; the published tensors are all
     * ancestors of `t`, so they're included. */
    ggml_cgraph* graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, t);

    /* Allocate compute buffers + set input + run. */
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
    RFDETR_ASSERT(buf != nullptr);
    ggml_backend_tensor_set(input, base.input_data.data(),
                            0, base.input_data.size() * sizeof(float));

    auto status = ggml_backend_graph_compute(backend, graph);
    RFDETR_ASSERT(status == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    /* Materialize traced tensors now that they're computed. */
    std::map<std::string, std::vector<float>> got;
    for (const auto& [name, tt] : traced) {
        got[name] = rfdetr::copy_tensor_to_f32(tt);
    }

    bool ok = true;
    for (const auto& [name, tol] : kTolerances) {
        auto g = got.find(name);
        auto w = base.tensors.find(name);
        if (g == got.end()) {
            std::fprintf(stderr, "[%s] FAIL: not published by C++ forward pass\n",
                         name.c_str());
            ok = false;
            continue;
        }
        if (w == base.tensors.end()) {
            std::fprintf(stderr, "[%s] FAIL: not present in baseline\n", name.c_str());
            ok = false;
            continue;
        }
        if (!allclose(name, g->second, w->second, tol.atol, tol.rtol)) {
            ok = false;
        }
    }

    rfdetr::set_trace_callback(nullptr);
    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    rfdetr::model_free(m);
    rfdetr::free_backend(backend);

    return ok ? 0 : 1;
}
