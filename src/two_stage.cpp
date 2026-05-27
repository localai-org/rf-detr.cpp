#include "two_stage.hpp"
#include "common.hpp"
#include "trace.hpp"
#include "transformer_ops.hpp"

#include "ggml.h"

#include <string>

namespace rfdetr {

namespace {

ggml_tensor* fetch(const Model& m, const std::string& name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "two_stage: missing tensor '%s'", name.c_str());
        return nullptr;
    }
    return it->second;
}

/* Apply a Linear layer (W: (in, out), b: (out,)) to a (in, N) tokens tensor.
 *   ggml_mul_mat(W, x): contracts ne[0]=in → (out, N) */
ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x,
                    ggml_tensor* W, ggml_tensor* b) {
    ggml_tensor* y = ggml_mul_mat(ctx, W, x);
    y = ggml_add(ctx, y, b);
    return y;
}

}  // namespace

void compute_proposal_grid(int width, int height, float wh_value, float* out) {
    /* Matches gen_encoder_output_proposals in
     * .venv/lib/.../rfdetr/models/transformer.py for a single feature level
     * with no padding mask. Per (h, w) the proposal is:
     *   cx = (w + 0.5) / valid_width
     *   cy = (h + 0.5) / valid_height
     *   wh = wh_value (constant 0.05 for lvl=0)
     * Layout: (4, W*H) F32, with token order (h, w) row-major (h outer, w inner),
     * matching torch's reshape(-1, H*W, 4) on a (H, W, 4) source. */
    const float fw = (float)width;
    const float fh = (float)height;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            const int token = h * width + w;
            float* p = out + (size_t)token * 4;
            p[0] = ((float)w + 0.5f) / fw;  // cx
            p[1] = ((float)h + 0.5f) / fh;  // cy
            p[2] = wh_value;                // w
            p[3] = wh_value;                // h
        }
    }
}

TwoStageOutput two_stage_forward(ggml_context* ctx, const Model& m,
                                 ggml_tensor* projector_out) {
    TwoStageOutput out;

    if (!projector_out) {
        rfdetr_logf(RFDETR_LOG_ERROR, "two_stage: null projector_out");
        return out;
    }
    const int64_t W = projector_out->ne[0];
    const int64_t H = projector_out->ne[1];
    const int64_t C = projector_out->ne[2];
    const int64_t B = projector_out->ne[3];
    if (B != 1) {
        rfdetr_logf(RFDETR_LOG_ERROR, "two_stage: batch != 1 (got %lld)", (long long)B);
        return out;
    }
    if (C != (int64_t)m.config.decoder.model_dim) {
        rfdetr_logf(RFDETR_LOG_ERROR, "two_stage: C %lld != decoder.model_dim %u",
                    (long long)C, m.config.decoder.model_dim);
        return out;
    }

    const int64_t N = W * H;

    /* 1. Flatten spatial. projector_out ne = (W, H, C, 1).
     *
     * We want tokens ne = (C, W*H, 1), with token index t = h*W + w (h outer,
     * w inner) — matching the (h, w) row-major order used by
     * gen_encoder_output_proposals.
     *
     * Permute (W, H, C, B) → (C, W, H, B) means result axis 0 ← src axis 2,
     * axis 1 ← src axis 0, axis 2 ← src axis 1. ggml_permute's args specify
     * WHERE each source axis ends up in the result, so the call is
     * permute(a, 1, 2, 0, 3): src axis 0 (W) → result axis 1, src axis 1 (H)
     * → result axis 2, src axis 2 (C) → result axis 0.
     *
     * After cont + reshape to (C, W*H, 1), the memory layout for token t is
     * (C contiguous), and t = w_idx + h_idx * W since W is the inner of the
     * permuted (W, H) pair. */
    ggml_tensor* tokens = ggml_cont(ctx, ggml_permute(ctx, projector_out, 1, 2, 0, 3));
    tokens = ggml_reshape_3d(ctx, tokens, C, N, 1);

    /* 2. enc_output[0]: Linear(256, 256) */
    ggml_tensor* W_eo = fetch(m, "two_stage.enc_output.0.weight");
    ggml_tensor* b_eo = fetch(m, "two_stage.enc_output.0.bias");
    if (!W_eo || !b_eo) return out;
    ggml_tensor* y = linear(ctx, tokens, W_eo, b_eo);

    /* 3. enc_output_norm[0]: LayerNorm(256) over channel dim (ne[0]). */
    ggml_tensor* w_ln = fetch(m, "two_stage.enc_output_norm.0.weight");
    ggml_tensor* b_ln = fetch(m, "two_stage.enc_output_norm.0.bias");
    if (!w_ln || !b_ln) return out;
    y = ops::layer_norm(ctx, y, w_ln, b_ln);
    publish("two_stage.enc_output_norm.output", y);
    out.enc_output_norm_out = y;

    /* 4. enc_out_class_embed[0]: Linear(256, num_classes=91). */
    ggml_tensor* W_cls = fetch(m, "two_stage.enc_out_class_embed.0.weight");
    ggml_tensor* b_cls = fetch(m, "two_stage.enc_out_class_embed.0.bias");
    if (!W_cls || !b_cls) return out;
    out.cls_all = linear(ctx, y, W_cls, b_cls);

    /* 5. enc_out_bbox_embed[0]: MLP(256 → 256 → 256 → 4), ReLU between layers
     * (rfdetr/models/math.py:MLP uses F.relu, NOT GELU). */
    ggml_tensor* W_b0 = fetch(m, "two_stage.enc_out_bbox_embed.0.layers.0.weight");
    ggml_tensor* b_b0 = fetch(m, "two_stage.enc_out_bbox_embed.0.layers.0.bias");
    ggml_tensor* W_b1 = fetch(m, "two_stage.enc_out_bbox_embed.0.layers.1.weight");
    ggml_tensor* b_b1 = fetch(m, "two_stage.enc_out_bbox_embed.0.layers.1.bias");
    ggml_tensor* W_b2 = fetch(m, "two_stage.enc_out_bbox_embed.0.layers.2.weight");
    ggml_tensor* b_b2 = fetch(m, "two_stage.enc_out_bbox_embed.0.layers.2.bias");
    if (!W_b0 || !b_b0 || !W_b1 || !b_b1 || !W_b2 || !b_b2) return out;

    ggml_tensor* h = linear(ctx, y, W_b0, b_b0);
    h = ggml_relu(ctx, h);
    h = linear(ctx, h, W_b1, b_b1);
    h = ggml_relu(ctx, h);
    ggml_tensor* bbox_delta = linear(ctx, h, W_b2, b_b2);  /* (4, N) */

    /* 6. bbox_reparam (=True for rfdetr-base): combine delta with proposals.
     *   cxcy = delta[:2] * proposals[2:] + proposals[:2]
     *   wh   = exp(delta[2:]) * proposals[2:]
     *
     * proposals lives in m.proposals_grid ((4, N) F32). */
    if (!m.proposals_grid) {
        rfdetr_logf(RFDETR_LOG_ERROR, "two_stage: missing proposals_grid (run model_realize_weights first)");
        return out;
    }
    if (m.proposals_grid->ne[0] != 4 || m.proposals_grid->ne[1] != N) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "two_stage: proposals_grid shape (%lld, %lld) != (4, %lld)",
                    (long long)m.proposals_grid->ne[0],
                    (long long)m.proposals_grid->ne[1],
                    (long long)N);
        return out;
    }

    /* Slice each (4, N) tensor along ne[0] into two (2, N) halves. */
    const size_t esz_d = ggml_element_size(bbox_delta);
    const size_t esz_p = ggml_element_size(m.proposals_grid);
    ggml_tensor* delta_xy = ggml_cont(ctx, ggml_view_2d(
        ctx, bbox_delta, /*ne0*/ 2, /*ne1*/ N,
        /*nb1*/ bbox_delta->nb[1], /*offset*/ 0 * esz_d));
    ggml_tensor* delta_wh = ggml_cont(ctx, ggml_view_2d(
        ctx, bbox_delta, /*ne0*/ 2, /*ne1*/ N,
        /*nb1*/ bbox_delta->nb[1], /*offset*/ 2 * esz_d));
    ggml_tensor* prop_xy = ggml_cont(ctx, ggml_view_2d(
        ctx, m.proposals_grid, /*ne0*/ 2, /*ne1*/ N,
        /*nb1*/ m.proposals_grid->nb[1], /*offset*/ 0 * esz_p));
    ggml_tensor* prop_wh = ggml_cont(ctx, ggml_view_2d(
        ctx, m.proposals_grid, /*ne0*/ 2, /*ne1*/ N,
        /*nb1*/ m.proposals_grid->nb[1], /*offset*/ 2 * esz_p));

    /* cxcy = delta_xy * prop_wh + prop_xy */
    ggml_tensor* cxcy = ggml_add(ctx, ggml_mul(ctx, delta_xy, prop_wh), prop_xy);
    /* wh = exp(delta_wh) * prop_wh */
    ggml_tensor* wh = ggml_mul(ctx, ggml_exp(ctx, delta_wh), prop_wh);

    /* Concat along ne[0]: (2, N) ++ (2, N) → (4, N). */
    out.bbox_all = ggml_concat(ctx, cxcy, wh, /*dim*/ 0);

    return out;
}

}  // namespace rfdetr
