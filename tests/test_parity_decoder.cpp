/* Plan 11: C++ LWDETR decoder vs torch parity baseline.
 *
 * Module-isolated test: feeds baseline tensors directly into the C++ decoder
 * forward to bypass upstream drift.
 *
 *   memory     <- baseline parity.projector.output, flattened (W,H,C,1) → (C, W*H, 1)
 *   tgt        <- baseline parity.decoder.input.tgt        (256, 300, 1)
 *   refpoints  <- baseline parity.decoder.input.reference_points (squeezed L axis → (4, 300, 1))
 *   query_pos  <- C++ ref_point_head(sine_embed(refpoints)) — also verified
 *                 against baseline parity.decoder.ref_point_head.output
 *
 * Checkpoints diffed:
 *   decoder.ref_point_head.output         (256, 300, 1)
 *   decoder.layer.{0,1,2}.self_attn.output (256, 300, 1)
 *   decoder.layer.{0,1,2}.norm1.output    (256, 300, 1)
 *   decoder.layer.{0,1,2}.cross_attn.output (256, 300, 1)
 *   decoder.layer.{0,1,2}.norm2.output    (256, 300, 1)
 *   decoder.layer.{0,1,2}.linear1.output  (2048, 300, 1)
 *   decoder.layer.{0,1,2}.linear2.output  (256, 300, 1)
 *   decoder.layer.{0,1,2}.output          (256, 300, 1)
 *   decoder.norm.output                   (256, 300, 1)
 *
 * Target: max_abs ≤ 1e-3 (post-LN cascades amplify drift). Sub-module
 * checkpoints help localize which step dominates the gap.
 *
 * Skips gracefully when model GGUF or baseline isn't present. */
#include "test_assert.hpp"
#include "rfdetr.h"
#include "model_loader.hpp"
#include "backend.hpp"
#include "trace.hpp"
#include "decoder.hpp"

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
    const size_t n = std::min(got.size(), want.size());
    for (size_t i = 0; i < n; ++i) {
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
    if (n > 0) {
        s.mean_abs = (float)(sum_abs / (double)n);
        s.rms      = (float)std::sqrt(sum_sq / (double)n);
    }
    return s;
}

void report(const char* name, const std::vector<float>& got,
            const std::vector<float>& want, float tol, int& n_fail) {
    if (got.size() != want.size()) {
        std::fprintf(stderr,
            "  [%-38s] size mismatch C++=%zu torch=%zu — STRUCT FAIL\n",
            name, got.size(), want.size());
        ++n_fail;
        return;
    }
    DiffStats ds = diff(got, want);
    const char* tag = (ds.max_abs <= tol) ? "OK" : "FAIL";
    std::fprintf(stderr,
        "  [%-38s] %-4s max_abs=%.4g mean_abs=%.4g rms=%.4g (got=%g want=%g at idx %zu)\n",
        name, tag, ds.max_abs, ds.mean_abs, ds.rms,
        ds.got_at_max, ds.want_at_max, ds.max_idx);
    if (ds.max_abs > tol) ++n_fail;
}

/* Flatten projector.output (ne=(W,H,C,1)) → memory ne=(C, W*H, 1) with token
 * order t = h*W + w (h outer, w inner). The projector output is
 * dim-fastest-at-(w,h) i.e. memory layout (W,H,C,1) → projector_out[w + h*W + c*W*H].
 * After permute(1,2,0,3) we get (C,W,H,1) memory laid out as C-fastest then W then H,
 * which when reshape'd to (C, W*H) gives token t at byte offset c + (w+h*W)*C.
 *
 * For the baseline (which is column-major in torch (1, 256, 40, 40)): the data
 * in the baseline tensor is read as ne=(W=40, H=40, C=256, 1) by ggml's GGUF
 * loader, which preserves the original element order. So we need the same
 * permute+reshape that two_stage does to get (256, 1600, 1) with token order
 * t = h*W + w.
 *
 * This function does it on the CPU because the input here is a raw vector
 * (we already loaded the baseline into memory, not a ggml tensor). */
std::vector<float> projector_to_memory(const std::vector<float>& proj,
                                       int W, int H, int C) {
    std::vector<float> out((size_t)W * H * C);
    /* proj layout (ne=(W,H,C,1)): elt[w + h*W + c*W*H]. The torch source was
     * (1, C, H, W) with stride (C*H*W, H*W, W, 1). Reading element[b, c, h, w]
     * gives offset c*H*W + h*W + w. When GGUF round-trips, the byte order is
     * preserved, so ggml sees ne=(W, H, C, 1) — ne[0]=W is fastest (innermost
     * stride 1 element), then H (stride W), then C (stride H*W). So
     * proj[w + h*W + c*H*W] is the (c, h, w) value.
     *
     * Out layout: ne=(C, W*H, 1) — C fastest, then token t. Token order:
     * t = h*W + w (h outer, w inner). out[c + t*C] = proj[w + h*W + c*H*W]. */
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            const int t = h * W + w;
            for (int c = 0; c < C; ++c) {
                out[(size_t)c + (size_t)t * C] = proj[(size_t)w + (size_t)h * W + (size_t)c * W * H];
            }
        }
    }
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
            "[test_parity_decoder] SKIPPED: real rfdetr-base GGUF not found.\n");
        return 0;
    }
    if (!file_exists(base_path)) {
        std::fprintf(stderr,
            "[test_parity_decoder] SKIPPED: baseline not present (%s).\n",
            base_path.c_str());
        return 0;
    }

    Baseline base = load_baseline(base_path);
    /* Skip gracefully if the baseline doesn't include the new decoder inputs
     * (i.e. fixtures predate the Plan 11 baseline regen). */
    auto have = [&](const char* k) { return base.tensors.find(k) != base.tensors.end(); };
    if (!have("projector.output") ||
        !have("decoder.input.tgt") ||
        !have("decoder.input.reference_points")) {
        std::fprintf(stderr,
            "[test_parity_decoder] SKIPPED: baseline missing decoder inputs (regenerate baseline_torch.gguf).\n");
        return 0;
    }

    rfdetr_status st = RFDETR_OK;
    rfdetr::Model* m = rfdetr::model_load(model_path, &st);
    RFDETR_ASSERT(m != nullptr);
    RFDETR_ASSERT_EQ_INT(st, RFDETR_OK);

    ggml_backend_t backend = rfdetr::init_backend(/*n_threads*/ 4, &st);
    RFDETR_ASSERT(backend != nullptr);
    RFDETR_ASSERT_EQ_INT(rfdetr::model_realize_weights(*m, backend), RFDETR_OK);

    const int dim = (int)m->config.decoder.model_dim;  // 256
    const int NQ  = (int)m->config.num_queries;         // 300
    const int H   = (int)(m->config.image_size / m->config.patch_size);  // 40
    const int W   = H;
    const int N_in = H * W;  // 1600

    /* Flatten projector output. */
    const auto& proj_data = base.tensors.at("projector.output");
    const auto& proj_shape = base.shapes.at("projector.output");
    RFDETR_ASSERT_EQ_INT(proj_shape[0], W);
    RFDETR_ASSERT_EQ_INT(proj_shape[1], H);
    RFDETR_ASSERT_EQ_INT(proj_shape[2], dim);
    std::vector<float> memory_flat = projector_to_memory(proj_data, W, H, dim);

    /* Build the C++ decoder graph. */
    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context* gctx = ggml_init(ip);
    RFDETR_ASSERT(gctx != nullptr);

    ggml_tensor* tgt_in = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, dim, NQ, 1);
    ggml_set_name(tgt_in, "decoder.input.tgt");
    ggml_tensor* mem_in = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, dim, N_in, 1);
    ggml_set_name(mem_in, "decoder.input.memory");
    ggml_tensor* rp_in  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 4, NQ, 1);
    ggml_set_name(rp_in, "decoder.input.refpoints");
    ggml_tensor* qpos_in = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, dim, NQ, 1);
    ggml_set_name(qpos_in, "decoder.input.query_pos");

    /* Sine embed input + the ref_point_head MLP (in-graph). */
    const int d_half = dim / 2;  // 128
    ggml_tensor* sine_in = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 4 * d_half, NQ, 1);
    ggml_set_name(sine_in, "decoder.ref_point_head.input");

    auto get_w = [&](const std::string& name) -> ggml_tensor* {
        auto it = m->tensors.find(name);
        RFDETR_ASSERT(it != m->tensors.end());
        return it->second;
    };

    ggml_tensor* rph_w0 = get_w("decoder.ref_point_head.layers.0.weight");
    ggml_tensor* rph_b0 = get_w("decoder.ref_point_head.layers.0.bias");
    ggml_tensor* rph_w1 = get_w("decoder.ref_point_head.layers.1.weight");
    ggml_tensor* rph_b1 = get_w("decoder.ref_point_head.layers.1.bias");

    /* ref_point_head: 2-layer MLP per `MLP(2*d_model, d_model, d_model, 2)` —
     * Linear(512, 256) → ReLU → Linear(256, 256). */
    ggml_tensor* qpos_built = ggml_mul_mat(gctx, rph_w0, sine_in);
    qpos_built = ggml_add(gctx, qpos_built, rph_b0);
    qpos_built = ggml_relu(gctx, qpos_built);
    qpos_built = ggml_mul_mat(gctx, rph_w1, qpos_built);
    qpos_built = ggml_add(gctx, qpos_built, rph_b1);
    ggml_set_name(qpos_built, "decoder.ref_point_head.output");

    /* Stash trace tensors. */
    std::map<std::string, const ggml_tensor*> traced;
    rfdetr::set_trace_callback(
        [&](const std::string& name, const ggml_tensor* tt) {
            traced[name] = tt;
        });

    ggml_tensor* dec_out = rfdetr::decoder_forward(
        gctx, *m, tgt_in, mem_in, rp_in, qpos_built, H, W);
    RFDETR_ASSERT(dec_out != nullptr);

    ggml_cgraph* graph = ggml_new_graph_custom(gctx, /*size*/ 16384, /*grads*/ false);
    ggml_build_forward_expand(graph, dec_out);
    ggml_build_forward_expand(graph, qpos_built);
    for (const auto& [_, tt] : traced) {
        ggml_build_forward_expand(graph, const_cast<ggml_tensor*>(tt));
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
    RFDETR_ASSERT(buf != nullptr);

    /* Populate inputs. */
    const auto& tgt_data = base.tensors.at("decoder.input.tgt");
    RFDETR_ASSERT_EQ_INT((int64_t)tgt_data.size(), (int64_t)dim * NQ);
    ggml_backend_tensor_set(tgt_in, tgt_data.data(), 0, tgt_data.size() * sizeof(float));

    ggml_backend_tensor_set(mem_in, memory_flat.data(), 0,
                            memory_flat.size() * sizeof(float));

    /* reference_points is (4, 1, 300, 1) in baseline — squeeze L=1 to (4, 300, 1). */
    const auto& rp_raw = base.tensors.at("decoder.input.reference_points");
    RFDETR_ASSERT_EQ_INT((int64_t)rp_raw.size(), (int64_t)4 * NQ);
    ggml_backend_tensor_set(rp_in, rp_raw.data(), 0, rp_raw.size() * sizeof(float));

    /* Compute query_sine_embed on CPU from refpoints (using cx, cy, w, h). */
    std::vector<float> sine_data((size_t)4 * d_half * NQ);
    rfdetr::compute_query_sine_embed(rp_raw.data(), NQ, d_half, sine_data.data());
    ggml_backend_tensor_set(sine_in, sine_data.data(), 0,
                            sine_data.size() * sizeof(float));

    auto status = ggml_backend_graph_compute(backend, graph);
    RFDETR_ASSERT(status == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    rfdetr::set_trace_callback(nullptr);

    /* Compare. Sub-module + full-layer checkpoints all pass at 1e-3; in
     * practice every checkpoint we see is < 2e-4 with the deformable
     * cross-attention CPU kernel — the largest gaps are on tensors with
     * |value| ~ 400, so relative error is ~5e-7. */
    const float kTol = 5e-4f;
    int n_fail = 0;

    /* Sine embed vs baseline. */
    {
        const auto& want = base.tensors.at("decoder.ref_point_head.input");
        report("decoder.ref_point_head.input", sine_data, want, kTol, n_fail);
    }
    /* query_pos (ref_point_head output). */
    {
        auto got = rfdetr::copy_tensor_to_f32(qpos_built);
        const auto& want = base.tensors.at("decoder.ref_point_head.output");
        report("decoder.ref_point_head.output", got, want, kTol, n_fail);
    }

    auto compare_traced = [&](const char* trace_name, const char* baseline_key) {
        auto it = traced.find(trace_name);
        if (it == traced.end()) {
            std::fprintf(stderr, "  [%-38s] missing trace — STRUCT FAIL\n", trace_name);
            ++n_fail;
            return;
        }
        auto got = rfdetr::copy_tensor_to_f32(it->second);
        auto itw = base.tensors.find(baseline_key);
        if (itw == base.tensors.end()) {
            std::fprintf(stderr, "  [%-38s] baseline missing %s — SKIP\n",
                         trace_name, baseline_key);
            return;
        }
        report(trace_name, got, itw->second, kTol, n_fail);
    };

    for (int li = 0; li < (int)m->config.decoder.layers; ++li) {
        const std::string lp = "decoder.layer." + std::to_string(li) + ".";
        compare_traced((lp + "self_attn.output").c_str(),
                       (lp + "self_attn.output").c_str());
        compare_traced((lp + "norm1.output").c_str(),
                       (lp + "norm1.output").c_str());
        compare_traced((lp + "cross_attn.output").c_str(),
                       (lp + "cross_attn.output").c_str());
        compare_traced((lp + "norm2.output").c_str(),
                       (lp + "norm2.output").c_str());
        compare_traced((lp + "linear1.output").c_str(),
                       (lp + "linear1.output").c_str());
        compare_traced((lp + "linear2.output").c_str(),
                       (lp + "linear2.output").c_str());
        compare_traced((lp + "output").c_str(),
                       (lp + "output").c_str());
    }
    compare_traced("decoder.norm.output", "decoder.norm.output");

    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    rfdetr::model_free(m);
    rfdetr::free_backend(backend);

    return n_fail == 0 ? 0 : 1;
}
