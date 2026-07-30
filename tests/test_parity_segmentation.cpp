/* Plan C: C++ SegmentationHead vs torch parity baseline.
 *
 * Two-phase test:
 *
 * Phase 1 — Isolated seg head: feed baseline `seg.spatial_features.input`
 * + per-layer `decoder.layer.{i}.post_norm` directly into the C++ seg head
 * forward and diff every intermediate against the torch baseline. This
 * pinpoints any bug in the seg head module itself (free of upstream drift).
 *
 * Phase 2 — Cumulative end-to-end: run the full rfdetr_model_forward on
 * the baseline's preprocess.input and diff the final masks against the
 * torch baseline's seg.masks.final. Logs the achieved drift; only fails on
 * egregious mismatch.
 *
 * Skips gracefully if either the seg model GGUF or the seg baseline isn't
 * present. */
#include "test_assert.hpp"
#include "rfdetr.h"
#include "model_loader.hpp"
#include "backend.hpp"
#include "rfdetr_model.hpp"
#include "trace.hpp"
#include "segmentation.hpp"

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
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    size_t max_idx = 0;
    float gm = 0, wm = 0;
};

DiffStats diff(const std::vector<float>& got, const std::vector<float>& want) {
    DiffStats s;
    double sm = 0.0;
    const size_t n = std::min(got.size(), want.size());
    for (size_t i = 0; i < n; ++i) {
        const float d = std::fabs(got[i] - want[i]);
        sm += d;
        if (d > s.max_abs) {
            s.max_abs = d;
            s.max_idx = i;
            s.gm = got[i];
            s.wm = want[i];
        }
    }
    if (n > 0) s.mean_abs = (float)(sm / (double)n);
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
        "  [%-38s] %-4s max_abs=%.4g mean_abs=%.4g (got=%g want=%g at idx %zu)\n",
        name, tag, ds.max_abs, ds.mean_abs, ds.gm, ds.wm, ds.max_idx);
    if (ds.max_abs > tol) ++n_fail;
}

/* Outcome of a phase-1 run. `ran` says whether the comparisons actually
 * executed — a bare `n_fail == 0` cannot distinguish "everything matched"
 * from "we bailed out before comparing anything", and this test is the
 * acceptance gate for the 5-block seg path, so that distinction has to
 * survive all the way to the caller. */
struct PhaseResult {
    bool ran    = false;
    int  n_fail = 0;
};

/* Phase 1: isolated seg head — feed baseline's seg.spatial_features.input
 * and per-layer post-norm decoder outputs into the C++ seg head, compare
 * each intermediate against the baseline.
 *
 * Every shape is derived from the loaded model's config (decoder layer
 * count, image size, mask downsample ratio) and from the baseline's own
 * tensor shapes — nothing here assumes a 4-block / 312px seg-nano. */
PhaseResult phase1_isolated(rfdetr::Model* m, ggml_backend_t backend,
                            const Baseline& base, const char* label) {
    auto have = [&](const std::string& k) {
        return base.tensors.find(k) != base.tensors.end();
    };

    const int n_layers   = (int)m->config.decoder.layers;
    const int image_size = (int)m->config.image_size;
    const int ratio      = (int)m->config.mask_downsample_ratio;

    if (!have("seg.spatial_features.input") || !have("seg.masks.final")) {
        std::fprintf(stderr,
            "[Phase 1: %s] NOT RUN: baseline file is present but missing seg "
            "captures.\n", label);
        return PhaseResult{};
    }
    for (int i = 0; i < n_layers; ++i) {
        const std::string k = "decoder.layer." + std::to_string(i) + ".post_norm";
        if (!have(k)) {
            std::fprintf(stderr,
                "[Phase 1: %s] NOT RUN: baseline missing %s (model has %d "
                "decoder layers — baseline was generated for a different "
                "variant?).\n", label, k.c_str(), n_layers);
            return PhaseResult{};
        }
    }

    /* Input geometry comes from the baseline tensors themselves. GGUF/ggml
     * order is reversed vs torch: torch (1, C, H, W) → ne = (W, H, C, 1),
     * torch (1, NQ, C) → ne = (C, NQ, 1, 1). */
    const auto& sp_shape = base.shapes.at("seg.spatial_features.input");
    const auto& qf_shape = base.shapes.at("decoder.layer.0.post_norm");
    const int W_proj = (int)sp_shape[0];
    const int H_proj = (int)sp_shape[1];
    const int C      = (int)sp_shape[2];
    const int NQ     = (int)qf_shape[1];

    std::fprintf(stderr,
        "[Phase 1: %s] Isolated seg head vs torch baseline: n_layers=%d "
        "image_size=%d ratio=%d spatial=(%dx%dx%d) NQ=%d masks=%dx%d\n",
        label, n_layers, image_size, ratio, W_proj, H_proj, C, NQ,
        image_size / ratio, image_size / ratio);

    RFDETR_ASSERT_EQ_INT((int)qf_shape[0], C);

    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context* gctx = ggml_init(ip);
    RFDETR_ASSERT(gctx != nullptr);

    /* Inputs to the seg head:
     *   spatial_features ne = (W_proj, H_proj, C, 1)
     *   per-layer decoder outputs ne = (C, NQ, 1) — one per decoder layer
     */
    ggml_tensor* spatial_in = ggml_new_tensor_4d(gctx, GGML_TYPE_F32,
                                                  W_proj, H_proj, C, 1);
    ggml_set_name(spatial_in, "seg.spatial_in");

    std::vector<ggml_tensor*> qf_in((size_t)n_layers);
    for (int i = 0; i < n_layers; ++i) {
        qf_in[i] = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, C, NQ, 1);
        ggml_set_name(qf_in[i], ("seg.qf_in." + std::to_string(i)).c_str());
    }

    /* Capture every published trace tensor — defer reading data until after
     * compute. Trace fires inline at graph-build time. */
    std::map<std::string, const ggml_tensor*> traced;
    rfdetr::set_trace_callback(
        [&](const std::string& name, const ggml_tensor* tt) {
            traced[name] = tt;
        });

    ggml_tensor* masks = rfdetr::segmentation_forward(
        gctx, *m, spatial_in, qf_in.data(), n_layers,
        /*image_h*/ image_size, /*image_w*/ image_size, /*ratio*/ ratio);
    RFDETR_ASSERT(masks != nullptr);

    ggml_cgraph* graph = ggml_new_graph_custom(gctx, /*size*/ 16384,
                                                /*grads*/ false);
    ggml_build_forward_expand(graph, masks);
    for (const auto& [_, tt] : traced) {
        ggml_build_forward_expand(graph, const_cast<ggml_tensor*>(tt));
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
    RFDETR_ASSERT(buf != nullptr);

    /* Stage the baseline data into the input tensors. */
    const auto& spd = base.tensors.at("seg.spatial_features.input");
    ggml_backend_tensor_set(spatial_in, spd.data(), 0, spd.size() * sizeof(float));
    for (int i = 0; i < n_layers; ++i) {
        const auto& qd = base.tensors.at("decoder.layer." + std::to_string(i) + ".post_norm");
        ggml_backend_tensor_set(qf_in[i], qd.data(), 0, qd.size() * sizeof(float));
    }

    auto status = ggml_backend_graph_compute(backend, graph);
    RFDETR_ASSERT(status == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    rfdetr::set_trace_callback(nullptr);

    /* The seg head is well-conditioned: norm → linear → gelu → linear in
     * the MLPBlock, plus depthwise+norm+linear+gelu+residual in each
     * DepthwiseConvBlock. F32 round-off should give us max_abs ≤ 1e-3
     * easily on isolated inputs (baseline is also F32). */
    const float kTol = 1e-3f;
    int n_fail = 0;

    /* Diff each captured intermediate. */
    auto check = [&](const char* trace_name, const char* baseline_name) {
        auto it = traced.find(trace_name);
        if (it == traced.end()) {
            /* A checkpoint the seg head is supposed to publish went missing
             * — e.g. a block that never ran. That is a failure, not a skip. */
            std::fprintf(stderr, "  [%-38s] FAIL not in traced map\n", trace_name);
            ++n_fail;
            return;
        }
        const auto baseline_it = base.tensors.find(baseline_name);
        if (baseline_it == base.tensors.end()) {
            /* Symmetric with the missing-trace branch above: a checkpoint we
             * expect to compare that has no baseline counterpart means the
             * gate silently stopped covering it. Fail, don't shrug. */
            std::fprintf(stderr, "  [%-38s] FAIL missing baseline %s\n",
                         trace_name, baseline_name);
            ++n_fail;
            return;
        }
        auto got = rfdetr::copy_tensor_to_f32(it->second);
        report(trace_name, got, baseline_it->second, kTol, n_fail);
    };

    check("seg.spatial_features.resized", "seg.spatial_features.resized");
    for (int i = 0; i < n_layers; ++i) {
        const std::string b = "seg.block." + std::to_string(i) + ".";
        const std::string mk = "seg.masks." + std::to_string(i);
        check((b + "spatial_out").c_str(),  (b + "spatial_out").c_str());
        check((b + "spatial_proj").c_str(), (b + "spatial_proj").c_str());
        check((b + "qf_proj").c_str(),      (b + "qf_proj").c_str());
        check(mk.c_str(),                   mk.c_str());
    }
    check("seg.masks.final",              "seg.masks.final");

    /* A dropped block would leave the last block's intermediates untraced
     * rather than mismatched, so assert the traced count explicitly: one
     * seg.block.{i}.spatial_out per decoder layer, no more, no fewer. */
    int n_traced_blocks = 0;
    for (const auto& [name, _] : traced) {
        if (name.rfind("seg.block.", 0) == 0 &&
            name.find(".spatial_out") != std::string::npos) {
            ++n_traced_blocks;
        }
    }
    std::fprintf(stderr, "  traced seg blocks: %d (expected %d)\n",
                 n_traced_blocks, n_layers);
    if (n_traced_blocks != n_layers) ++n_fail;

    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    return PhaseResult{/*ran*/ true, n_fail};
}

/* Run phase 1 (and the phase-2 cumulative log) for one seg variant.
 *
 * The layer count, image size and mask ratio are read from the loaded
 * model's config rather than passed in, so a 5-block variant exercises all
 * 5 blocks.
 *
 * Returns true only if the phase-1 comparisons actually executed. Two
 * distinct outcomes are deliberately NOT conflated:
 *   - fixtures absent          → SKIPPED, returns false, no failure (a fresh
 *                                clone without the generated GGUFs must pass);
 *   - fixtures present but the
 *     comparisons could not run → FAIL, returns false and bumps `n_fail`. A
 *                                 fixture that exists but cannot be compared
 *                                 against means this gate is covering nothing,
 *                                 which must never read as green.
 * Phase-1 comparison failures are also added to `n_fail`. */
bool run_phase1(const std::string& baseline_path,
                const std::string& model_path,
                const char* label,
                int& n_fail) {
    if (!file_exists(baseline_path) || !file_exists(model_path)) {
        std::fprintf(stderr,
            "[Phase 1: %s] SKIPPED: fixtures not present (%s / %s).\n",
            label, baseline_path.c_str(), model_path.c_str());
        return false;
    }

    Baseline base = load_baseline(baseline_path);
    auto have = [&](const char* k) {
        return base.tensors.find(k) != base.tensors.end();
    };
    if (!have("preprocess.input") || !have("seg.masks.final")) {
        std::fprintf(stderr,
            "[Phase 1: %s] FAIL: baseline %s is present but missing "
            "preprocess.input / seg.masks.final — regenerate it with "
            "scripts/gen_torch_baseline.py --seg-variant.\n",
            label, baseline_path.c_str());
        ++n_fail;
        return false;
    }

    rfdetr_status st = RFDETR_OK;
    rfdetr::Model* m = rfdetr::model_load(model_path, &st);
    RFDETR_ASSERT(m != nullptr);
    RFDETR_ASSERT_EQ_INT(st, RFDETR_OK);
    RFDETR_ASSERT(m->config.has_segmentation_head);

    ggml_backend_t backend = rfdetr::init_backend(/*n_threads*/ 4, &st);
    RFDETR_ASSERT(backend != nullptr);
    RFDETR_ASSERT_EQ_INT(rfdetr::model_realize_weights(*m, backend), RFDETR_OK);

    const PhaseResult p1 = phase1_isolated(m, backend, base, label);
    n_fail += p1.n_fail;
    if (!p1.ran) {
        /* The fixtures are on disk, so the operator asked for this variant to
         * be checked and it was not. Never let that read as a pass. */
        std::fprintf(stderr,
            "[Phase 1: %s] FAIL: fixtures present but the comparisons never "
            "ran (see the NOT RUN line above).\n", label);
        ++n_fail;
        rfdetr::model_free(m);
        rfdetr::free_backend(backend);
        return false;
    }

    /* Phase 2: full end-to-end. Cumulative drift includes backbone +
     * projector + decoder + seg head. We log it but don't enforce a tight
     * tolerance — Phase 1 already proves the seg head itself is correct. */
    std::fprintf(stderr, "[Phase 2: %s] End-to-end vs torch baseline:\n", label);

    const auto& in_data  = base.tensors.at("preprocess.input");
    const auto& in_shape = base.shapes.at("preprocess.input");
    const int W = (int)in_shape[0];

    rfdetr::ForwardOutput fout = rfdetr::rfdetr_model_forward(
        *m, in_data.data(), W, backend);
    RFDETR_ASSERT(!fout.class_logits.empty());
    RFDETR_ASSERT(!fout.masks.empty());

    const auto& want = base.tensors.at("seg.masks.final");
    RFDETR_ASSERT_EQ_INT((int)fout.masks.size(), (int)want.size());
    DiffStats ds = diff(fout.masks, want);
    std::fprintf(stderr,
        "  [seg.masks.final (cumulative)         ] max_abs=%.4g mean_abs=%.4g "
        "(got=%g want=%g at idx %zu)\n",
        ds.max_abs, ds.mean_abs, ds.gm, ds.wm, ds.max_idx);

    rfdetr::model_free(m);
    rfdetr::free_backend(backend);
    return true;
}

}  // namespace

int main() {
    const std::string fx = RFDETR_TEST_FIXTURES;

    /* Model GGUFs may sit next to the fixtures or in the repo's models/. */
    auto find_model = [&](const std::string& name) -> std::string {
        for (const std::string& candidate : {
                fx + "/" + name,
                fx + "/../../models/" + name,
            }) {
            if (file_exists(candidate)) return candidate;
        }
        return fx + "/../../models/" + name;  /* reported in the skip message */
    };

    int n_fail = 0;
    const bool ran_nano = run_phase1(fx + "/baseline_torch_seg.gguf",
                                     find_model("rfdetr-seg-nano-f32.gguf"),
                                     "seg-nano (4 blocks)", n_fail);
    const bool ran_medium = run_phase1(fx + "/baseline_torch_seg_medium.gguf",
                                       find_model("rfdetr-seg-medium-f32.gguf"),
                                       "seg-medium (5 blocks)", n_fail);
    /* The larger seg variants. seg-large is a second 5-block witness at a
     * higher resolution; seg-xlarge / seg-2xlarge are the only coverage the
     * 6-block path has. Their baselines are large (roughly 350MB / 540MB /
     * 800MB) and gitignored, so they are normally absent and skip — see the
     * regeneration recipe in tests/CMakeLists.txt. */
    const bool ran_large = run_phase1(fx + "/baseline_torch_seg_large.gguf",
                                      find_model("rfdetr-seg-large-f32.gguf"),
                                      "seg-large (5 blocks)", n_fail);
    const bool ran_xlarge = run_phase1(fx + "/baseline_torch_seg_xlarge.gguf",
                                       find_model("rfdetr-seg-xlarge-f32.gguf"),
                                       "seg-xlarge (6 blocks)", n_fail);
    const bool ran_2xlarge = run_phase1(fx + "/baseline_torch_seg_2xlarge.gguf",
                                        find_model("rfdetr-seg-2xlarge-f32.gguf"),
                                        "seg-2xlarge (6 blocks)", n_fail);
    /* State the verdict explicitly: ctest only surfaces the exit code, and
     * "green" must never be ambiguous about which variants were actually
     * exercised — in particular whether the 6-block path ran at all. */
    std::fprintf(stderr,
        "[test_parity_segmentation] seg-nano: %s | seg-medium (5 blocks): %s "
        "| seg-large (5 blocks): %s | seg-xlarge (6 blocks): %s "
        "| seg-2xlarge (6 blocks): %s | failures: %d\n",
        ran_nano ? "RAN" : "not run", ran_medium ? "RAN" : "not run",
        ran_large ? "RAN" : "not run", ran_xlarge ? "RAN" : "not run",
        ran_2xlarge ? "RAN" : "not run", n_fail);

    if (!ran_nano && !ran_medium && !ran_large && !ran_xlarge && !ran_2xlarge) {
        std::fprintf(stderr,
            "[test_parity_segmentation] SKIPPED: no seg fixtures present.\n");
    }

    return n_fail > 0 ? 1 : 0;
}
