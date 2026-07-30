/* gen_encoder_output_proposals validity mask.
 *
 * Upstream (rfdetr 1.9.0, rfdetr/models/transformer.py:127-143):
 *
 *   output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
 *   ...
 *   else:                                     # unsigmoid = not bbox_reparam = False
 *       output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))
 *   output_memory = memory
 *   output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
 *
 * Both `output_proposals` AND `output_memory` are zeroed for tokens whose
 * proposal falls outside the open interval (0.01, 0.99). Since the centre is
 * cx = (w + 0.5) / S for a grid of side S, the condition can only fire when
 * S >= 50, where it masks exactly the one-cell border ring (4*S - 4 tokens).
 *
 * This test has two halves:
 *
 *  A. `compute_proposal_grid` arithmetic — pure host-side, no model needed.
 *     Checks the fired set is exactly the border ring at S = 50/52/64, and
 *     that S = 40/49 (every published detection variant and every seg variant
 *     below xlarge) is a strict no-op: proposals stay bit-identical to the
 *     analytic formula.
 *
 *  B. `two_stage_forward` graph behaviour — a synthetic 4-channel two-stage
 *     head over a 52x52 and a 40x40 grid. enc_output is the identity with a
 *     zero bias and enc_output_norm has unit weight plus a distinctive bias
 *     `b_ln`, so a token whose MEMORY was zeroed lands on exactly `b_ln`
 *     (LayerNorm of the zero vector) and nothing else does. That pins the
 *     mask to the memory path specifically, not just to the proposals.
 *
 * No PyTorch baseline fixture required; the rule is pure arithmetic. */
#include "test_assert.hpp"
#include "rfdetr.h"
#include "model_loader.hpp"
#include "backend.hpp"
#include "trace.hpp"
#include "two_stage.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void report(const char* what, bool ok) {
    std::fprintf(stderr, "  [%-44s] %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_fail;
}

bool is_border(int w, int h, int S) {
    return w == 0 || h == 0 || w == S - 1 || h == S - 1;
}

/* ---- Part A: compute_proposal_grid ------------------------------------- */

/* Returns the number of fully-zeroed proposal rows; fails if any zeroed row
 * is not on the border ring, or any border row is not zeroed when
 * `expect_masked` is set, or any surviving row deviates from the analytic
 * formula. */
void check_grid(int S, bool expect_masked) {
    const float wh = 0.05f;
    std::vector<float> grid((size_t)S * S * 4, -1.0f);
    rfdetr::compute_proposal_grid(S, S, wh, grid.data());

    const float fs = (float)S;
    int n_zeroed = 0;
    bool ring_exact = true;
    bool survivors_exact = true;

    for (int h = 0; h < S; ++h) {
        for (int w = 0; w < S; ++w) {
            const float* p = grid.data() + ((size_t)h * S + w) * 4;
            const bool zeroed = (p[0] == 0.0f && p[1] == 0.0f &&
                                 p[2] == 0.0f && p[3] == 0.0f);
            if (zeroed) {
                ++n_zeroed;
                if (!is_border(w, h, S)) ring_exact = false;
            } else {
                /* Untouched rows must be bit-identical to the formula. */
                if (p[0] != ((float)w + 0.5f) / fs) survivors_exact = false;
                if (p[1] != ((float)h + 0.5f) / fs) survivors_exact = false;
                if (p[2] != wh || p[3] != wh)       survivors_exact = false;
                if (expect_masked && is_border(w, h, S)) ring_exact = false;
            }
        }
    }

    const int want_zeroed = expect_masked ? (4 * S - 4) : 0;
    char label[96];
    std::snprintf(label, sizeof(label), "grid %d: %d zeroed (want %d)",
                  S, n_zeroed, want_zeroed);
    report(label, n_zeroed == want_zeroed);

    std::snprintf(label, sizeof(label), "grid %d: zeroed set == border ring", S);
    report(label, ring_exact);

    std::snprintf(label, sizeof(label), "grid %d: survivors bit-exact", S);
    report(label, survivors_exact);
}

/* ---- Part B: two_stage_forward ----------------------------------------- */

const int    kD    = 4;                 /* synthetic model_dim */
const int    kNC   = 2;                 /* synthetic num_classes */
const float  kBLn[kD] = {0.25f, -1.5f, 3.0f, 0.125f};  /* LayerNorm bias */

/* Deterministic, always-nonzero, never channel-constant input. */
float input_at(int w, int h, int c) {
    return 0.5f + 0.01f * (float)((w * 7 + h * 13 + c * 3) % 17);
}

ggml_tensor* make(ggml_context* ctx, rfdetr::Model& m, const char* name,
                  int64_t ne0, int64_t ne1) {
    ggml_tensor* t = ne1 > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1)
                             : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    RFDETR_ASSERT(t != nullptr);
    ggml_set_name(t, name);
    m.tensors[name] = t;
    return t;
}

void run_forward_case(int S, bool expect_masked) {
    rfdetr::Model m;
    m.config.decoder.model_dim = kD;
    m.config.num_classes       = kNC;

    rfdetr_status st = RFDETR_OK;
    ggml_backend_t backend = rfdetr::init_backend(/*n_threads*/ 2, &st);
    RFDETR_ASSERT(backend != nullptr);

    const int64_t N = (int64_t)S * S;

    /* --- synthetic weights + the proposal grid, in their own context --- */
    ggml_init_params wp{};
    wp.mem_size   = ggml_tensor_overhead() * 64;
    wp.mem_buffer = nullptr;
    wp.no_alloc   = true;
    ggml_context* wctx = ggml_init(wp);
    RFDETR_ASSERT(wctx != nullptr);

    ggml_tensor* W_eo  = make(wctx, m, "two_stage.enc_output.0.weight", kD, kD);
    ggml_tensor* b_eo  = make(wctx, m, "two_stage.enc_output.0.bias", kD, 0);
    ggml_tensor* w_ln  = make(wctx, m, "two_stage.enc_output_norm.0.weight", kD, 0);
    ggml_tensor* b_ln  = make(wctx, m, "two_stage.enc_output_norm.0.bias", kD, 0);
    ggml_tensor* W_cls = make(wctx, m, "two_stage.enc_out_class_embed.0.weight", kD, kNC);
    ggml_tensor* b_cls = make(wctx, m, "two_stage.enc_out_class_embed.0.bias", kNC, 0);
    ggml_tensor* W_b0  = make(wctx, m, "two_stage.enc_out_bbox_embed.0.layers.0.weight", kD, kD);
    ggml_tensor* b_b0  = make(wctx, m, "two_stage.enc_out_bbox_embed.0.layers.0.bias", kD, 0);
    ggml_tensor* W_b1  = make(wctx, m, "two_stage.enc_out_bbox_embed.0.layers.1.weight", kD, kD);
    ggml_tensor* b_b1  = make(wctx, m, "two_stage.enc_out_bbox_embed.0.layers.1.bias", kD, 0);
    ggml_tensor* W_b2  = make(wctx, m, "two_stage.enc_out_bbox_embed.0.layers.2.weight", kD, kD);
    ggml_tensor* b_b2  = make(wctx, m, "two_stage.enc_out_bbox_embed.0.layers.2.bias", kD, 0);

    ggml_tensor* prop = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 4, N);
    RFDETR_ASSERT(prop != nullptr);
    ggml_set_name(prop, "two_stage.proposals.grid");
    m.proposals_grid = prop;

    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    RFDETR_ASSERT(wbuf != nullptr);

    auto fill = [](ggml_tensor* t, const std::vector<float>& v) {
        RFDETR_ASSERT_EQ_INT((int64_t)v.size(), ggml_nelements(t));
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    };

    /* enc_output = identity, zero bias, so a zeroed memory row stays zero. */
    std::vector<float> ident((size_t)kD * kD, 0.0f);
    for (int i = 0; i < kD; ++i) ident[(size_t)i * kD + i] = 1.0f;
    fill(W_eo, ident);
    fill(b_eo, std::vector<float>(kD, 0.0f));
    fill(w_ln, std::vector<float>(kD, 1.0f));
    fill(b_ln, std::vector<float>(kBLn, kBLn + kD));

    /* Class + bbox heads: arbitrary but nonzero and non-degenerate. */
    std::vector<float> wc((size_t)kD * kNC);
    for (size_t i = 0; i < wc.size(); ++i) wc[i] = 0.1f + 0.05f * (float)i;
    fill(W_cls, wc);
    fill(b_cls, std::vector<float>(kNC, 0.3f));

    std::vector<float> wb((size_t)kD * kD);
    for (size_t i = 0; i < wb.size(); ++i) wb[i] = 0.07f + 0.011f * (float)i;
    fill(W_b0, wb);
    fill(W_b1, wb);
    fill(W_b2, wb);
    fill(b_b0, std::vector<float>(kD, 0.2f));
    fill(b_b1, std::vector<float>(kD, 0.2f));
    fill(b_b2, std::vector<float>(kD, 0.2f));

    std::vector<float> grid((size_t)N * 4);
    rfdetr::compute_proposal_grid(S, S, /*wh_value*/ 0.05f, grid.data());
    fill(prop, grid);

    /* --- graph --- */
    ggml_init_params gp{};
    gp.mem_size   = 64 * 1024 * 1024;
    gp.mem_buffer = nullptr;
    gp.no_alloc   = true;
    ggml_context* gctx = ggml_init(gp);
    RFDETR_ASSERT(gctx != nullptr);

    ggml_tensor* proj_in = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, S, S, kD, 1);
    ggml_set_name(proj_in, "projector.output.in");

    rfdetr::TwoStageOutput ts = rfdetr::two_stage_forward(gctx, m, proj_in);
    RFDETR_ASSERT(ts.enc_output_norm_out != nullptr);
    RFDETR_ASSERT(ts.bbox_all != nullptr);

    ggml_cgraph* graph = ggml_new_graph_custom(gctx, /*size*/ 8192, /*grads*/ false);
    ggml_build_forward_expand(graph, ts.enc_output_norm_out);
    ggml_build_forward_expand(graph, ts.bbox_all);

    ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, backend);
    RFDETR_ASSERT(gbuf != nullptr);

    std::vector<float> in((size_t)S * S * kD);
    for (int c = 0; c < kD; ++c)
        for (int h = 0; h < S; ++h)
            for (int w = 0; w < S; ++w)
                in[((size_t)c * S + h) * S + w] = input_at(w, h, c);
    ggml_backend_tensor_set(proj_in, in.data(), 0, in.size() * sizeof(float));

    RFDETR_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> mem  = rfdetr::copy_tensor_to_f32(ts.enc_output_norm_out);
    std::vector<float> bbox = rfdetr::copy_tensor_to_f32(ts.bbox_all);
    RFDETR_ASSERT_EQ_INT((int64_t)mem.size(), N * kD);
    RFDETR_ASSERT_EQ_INT((int64_t)bbox.size(), N * 4);

    int n_mem_masked  = 0;
    int n_bbox_zero   = 0;
    bool mem_ring     = true;
    bool bbox_ring    = true;

    for (int h = 0; h < S; ++h) {
        for (int w = 0; w < S; ++w) {
            const size_t t = (size_t)h * S + w;
            const bool border = is_border(w, h, S);

            /* LayerNorm(0) == b_ln exactly: only reachable if the MEMORY row
             * was zeroed before enc_output. */
            bool is_bln = true;
            for (int c = 0; c < kD; ++c)
                if (mem[t * kD + c] != kBLn[c]) { is_bln = false; break; }
            if (is_bln) {
                ++n_mem_masked;
                if (!border) mem_ring = false;
            } else if (expect_masked && border) {
                mem_ring = false;
            }

            bool bz = true;
            for (int c = 0; c < 4; ++c)
                if (bbox[t * 4 + c] != 0.0f) { bz = false; break; }
            if (bz) {
                ++n_bbox_zero;
                if (!border) bbox_ring = false;
            } else if (expect_masked && border) {
                bbox_ring = false;
            }
        }
    }

    const int want = expect_masked ? (4 * S - 4) : 0;
    char label[96];
    std::snprintf(label, sizeof(label),
                  "fwd %dx%d: memory masked %d (want %d)", S, S, n_mem_masked, want);
    report(label, n_mem_masked == want);
    std::snprintf(label, sizeof(label), "fwd %dx%d: masked memory == border ring", S, S);
    report(label, mem_ring);
    std::snprintf(label, sizeof(label),
                  "fwd %dx%d: bbox zeroed %d (want %d)", S, S, n_bbox_zero, want);
    report(label, n_bbox_zero == want);
    std::snprintf(label, sizeof(label), "fwd %dx%d: zeroed bbox == border ring", S, S);
    report(label, bbox_ring);

    ggml_backend_buffer_free(gbuf);
    ggml_backend_buffer_free(wbuf);
    ggml_free(gctx);
    ggml_free(wctx);
    rfdetr::free_backend(backend);
}

}  // namespace

int main() {
    std::fprintf(stderr, "[test_two_stage_mask] A. compute_proposal_grid\n");
    /* Below the S >= 50 threshold: the mask must be a strict no-op. 40 is
     * rfdetr-base / seg-nano territory, 49 is the last grid that survives. */
    check_grid(40, /*expect_masked*/ false);
    check_grid(49, /*expect_masked*/ false);
    /* At and above the threshold: exactly the one-cell border ring. */
    check_grid(50, /*expect_masked*/ true);
    check_grid(52, /*expect_masked*/ true);   /* seg-xlarge  — 204 tokens */
    check_grid(64, /*expect_masked*/ true);   /* seg-2xlarge — 252 tokens */

    std::fprintf(stderr, "[test_two_stage_mask] B. two_stage_forward\n");
    run_forward_case(40, /*expect_masked*/ false);
    run_forward_case(52, /*expect_masked*/ true);

    if (g_fail) {
        std::fprintf(stderr, "[test_two_stage_mask] %d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::fprintf(stderr, "[test_two_stage_mask] all checks passed\n");
    return 0;
}
