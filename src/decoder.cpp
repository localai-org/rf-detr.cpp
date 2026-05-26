#include "decoder.hpp"
#include "transformer_ops.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace rfdetr {

namespace {

ggml_tensor* fetch(const Model& m, const std::string& name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "decoder: missing tensor '%s'", name.c_str());
        return nullptr;
    }
    return it->second;
}

ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x,
                    ggml_tensor* W, ggml_tensor* b) {
    ggml_tensor* y = ggml_mul_mat(ctx, W, x);
    y = ggml_add(ctx, y, b);
    return y;
}

/* Self-attention with packed in_proj weight (PyTorch nn.MultiheadAttention).
 *
 * in_proj_weight ne (in PyTorch) is (3*dim, dim) — laid out [Q|K|V] along the
 * output axis. In ggml, that's ne = (dim, 3*dim) and `mul_mat(W, x)` produces
 * (3*dim, N_q). We slice into 3 chunks of (dim, dim) along ne[1] to get Wq,
 * Wk, Wv; same for the (3*dim,) bias.
 *
 * Q,K share the same input (tgt + query_pos); V uses tgt only.
 *
 * Output ne = (dim, N_q). */
ggml_tensor* self_attn_packed(ggml_context* ctx,
                              ggml_tensor* qk_in, ggml_tensor* v_in,
                              ggml_tensor* W_in_proj, ggml_tensor* b_in_proj,
                              ggml_tensor* W_out_proj, ggml_tensor* b_out_proj,
                              int n_heads) {
    const int dim = (int)qk_in->ne[0];
    const int N_q = (int)qk_in->ne[1];
    const int head_dim = dim / n_heads;

    /* Slice in_proj weight: ne = (dim, 3*dim). Views into rows [0..dim),
     * [dim..2dim), [2dim..3dim) along ne[1]. Offset is row_index * row_stride. */
    const size_t W_row_size = W_in_proj->nb[1];
    ggml_tensor* Wq = ggml_view_2d(ctx, W_in_proj, dim, dim, W_row_size, (size_t)0       * W_row_size);
    ggml_tensor* Wk = ggml_view_2d(ctx, W_in_proj, dim, dim, W_row_size, (size_t)dim     * W_row_size);
    ggml_tensor* Wv = ggml_view_2d(ctx, W_in_proj, dim, dim, W_row_size, (size_t)(2*dim) * W_row_size);

    /* Materialize so mul_mat sees contiguous (dim, dim) tensors. */
    Wq = ggml_cont(ctx, Wq);
    Wk = ggml_cont(ctx, Wk);
    Wv = ggml_cont(ctx, Wv);

    /* Slice bias: ne = (3*dim,). */
    const size_t b_esz = ggml_element_size(b_in_proj);
    ggml_tensor* bq = ggml_view_1d(ctx, b_in_proj, dim, (size_t)0       * dim * b_esz);
    ggml_tensor* bk = ggml_view_1d(ctx, b_in_proj, dim, (size_t)1       * dim * b_esz);
    ggml_tensor* bv = ggml_view_1d(ctx, b_in_proj, dim, (size_t)2       * dim * b_esz);
    bq = ggml_cont(ctx, bq);
    bk = ggml_cont(ctx, bk);
    bv = ggml_cont(ctx, bv);

    /* Projections. */
    ggml_tensor* q = ggml_add(ctx, ggml_mul_mat(ctx, Wq, qk_in), bq);
    ggml_tensor* k = ggml_add(ctx, ggml_mul_mat(ctx, Wk, qk_in), bk);
    ggml_tensor* v = ggml_add(ctx, ggml_mul_mat(ctx, Wv, v_in),  bv);

    /* Reshape per-head. */
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, N_q);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, N_q);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, N_q);

    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    ggml_tensor* logits = ggml_mul_mat(ctx, k, q);
    logits = ggml_scale(ctx, logits, 1.0f / std::sqrt((float)head_dim));
    logits = ggml_soft_max(ctx, logits);

    ggml_tensor* v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    ggml_tensor* out = ggml_mul_mat(ctx, v_t, logits);

    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    out = ggml_reshape_2d(ctx, out, dim, N_q);

    out = ggml_mul_mat(ctx, W_out_proj, out);
    out = ggml_add(ctx, out, b_out_proj);
    return out;
}

/* MSDeformAttn bilinear-sampling + weighted-sum custom op.
 *
 * userdata layout (struct DefAttnArgs):
 *   H, W      spatial extent of memory
 *   n_heads, n_points
 *
 * Inputs (via dst->src[]):
 *   src[0]: value      ne = (head_dim, n_heads, H*W, 1)    — pre-permuted value
 *   src[1]: sampling_locations  ne = (2, n_points, n_heads, NQ)  xy fastest, [0,1]
 *   src[2]: attention_weights   ne = (n_points, n_heads, NQ)     softmaxed
 *
 * Output:
 *   dst ne = (head_dim * n_heads, NQ, 1, 1)    — packed (h, d) along ne[0]
 *     with d outer-of-h, i.e. flat[d + h*head_dim] for head h dim d. (Matches
 *     torch's transpose+reshape view at the end of ms_deform_attn_core_pytorch.)
 *
 * Single level (n_levels=1) — rfdetr-base only uses P4. */
struct DefAttnArgs {
    int H, W;
    int n_heads;
    int n_points;
    int head_dim;
};

void def_attn_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const DefAttnArgs* a = (const DefAttnArgs*)userdata;
    const int H = a->H, W = a->W;
    const int n_heads = a->n_heads;
    const int n_points = a->n_points;
    const int head_dim = a->head_dim;

    const ggml_tensor* value    = dst->src[0];
    const ggml_tensor* sloc     = dst->src[1];
    const ggml_tensor* aw       = dst->src[2];

    const int64_t NQ = dst->ne[1];
    const float* val_data  = (const float*)value->data;
    const float* sloc_data = (const float*)sloc->data;
    const float* aw_data   = (const float*)aw->data;
    float* dst_data        = (float*)dst->data;

    /* Strides (in elements, not bytes). */
    const size_t val_stride_h = (size_t)head_dim;                 // per-head stride along ne[1]
    const size_t val_stride_i = (size_t)head_dim * (size_t)n_heads; // per-token stride along ne[2]
    const size_t sloc_stride_p = 2;
    const size_t sloc_stride_h = (size_t)2 * (size_t)n_points;
    const size_t sloc_stride_q = (size_t)2 * (size_t)n_points * (size_t)n_heads;
    const size_t aw_stride_h = (size_t)n_points;
    const size_t aw_stride_q = (size_t)n_points * (size_t)n_heads;
    const size_t dst_stride_q = (size_t)head_dim * (size_t)n_heads;

    /* Parallelize over queries. */
    for (int64_t q = ith; q < NQ; q += nth) {
        float* out_q = dst_data + (size_t)q * dst_stride_q;
        for (int d = 0; d < head_dim * n_heads; ++d) out_q[d] = 0.0f;

        for (int h = 0; h < n_heads; ++h) {
            for (int p = 0; p < n_points; ++p) {
                const float sx = sloc_data[
                    (size_t)q * sloc_stride_q + (size_t)h * sloc_stride_h +
                    (size_t)p * sloc_stride_p + 0];
                const float sy = sloc_data[
                    (size_t)q * sloc_stride_q + (size_t)h * sloc_stride_h +
                    (size_t)p * sloc_stride_p + 1];
                const float w_attn = aw_data[
                    (size_t)q * aw_stride_q + (size_t)h * aw_stride_h + (size_t)p];

                /* sampling_locations in [0,1]; bilinear with align_corners=False:
                 *   ix = sx*W - 0.5,  iy = sy*H - 0.5
                 * Padding mode = zeros: corners outside image contribute 0. */
                const float ix = sx * (float)W - 0.5f;
                const float iy = sy * (float)H - 0.5f;
                const int ix0 = (int)std::floor(ix);
                const int iy0 = (int)std::floor(iy);
                const int ix1 = ix0 + 1;
                const int iy1 = iy0 + 1;
                const float wx1 = ix - (float)ix0;
                const float wy1 = iy - (float)iy0;
                const float wx0 = 1.0f - wx1;
                const float wy0 = 1.0f - wy1;

                auto safe_token = [&](int xi, int yi) -> int {
                    if (xi < 0 || xi >= W || yi < 0 || yi >= H) return -1;
                    return yi * W + xi;  /* h outer, w inner — matches token order */
                };
                const int t00 = safe_token(ix0, iy0);
                const int t10 = safe_token(ix1, iy0);
                const int t01 = safe_token(ix0, iy1);
                const int t11 = safe_token(ix1, iy1);

                const float* head_slot = val_data + (size_t)h * val_stride_h;
                for (int d = 0; d < head_dim; ++d) {
                    float s = 0.0f;
                    if (t00 >= 0) s += wx0 * wy0 * head_slot[(size_t)t00 * val_stride_i + d];
                    if (t10 >= 0) s += wx1 * wy0 * head_slot[(size_t)t10 * val_stride_i + d];
                    if (t01 >= 0) s += wx0 * wy1 * head_slot[(size_t)t01 * val_stride_i + d];
                    if (t11 >= 0) s += wx1 * wy1 * head_slot[(size_t)t11 * val_stride_i + d];
                    /* Output is packed (d + h*head_dim) — matches torch's
                     * reshape(B, n_heads*head_dim, NQ) after transpose. */
                    out_q[(size_t)h * (size_t)head_dim + (size_t)d] += w_attn * s;
                }
            }
        }
    }
}

/* Build the deformable cross-attention sub-graph.
 *
 * Inputs (graph nodes):
 *   query_in   ne = (dim, NQ)  — tgt + query_pos
 *   memory     ne = (dim, N_in)
 *   ref_xy     ne = (2, NQ)    — (cx, cy)
 *   ref_wh     ne = (2, NQ)    — (w, h)
 *   weights (from m.tensors): sampling_offsets_W/b, attention_weights_W/b,
 *           value_proj_W/b, output_proj_W/b
 *   H, W       memory spatial extent
 *
 * Output: ne = (dim, NQ). */
ggml_tensor* cross_attn_deformable(ggml_context* ctx,
                                   ggml_tensor* query_in, ggml_tensor* memory,
                                   ggml_tensor* ref_xy, ggml_tensor* ref_wh,
                                   ggml_tensor* W_so, ggml_tensor* b_so,
                                   ggml_tensor* W_aw, ggml_tensor* b_aw,
                                   ggml_tensor* W_vp, ggml_tensor* b_vp,
                                   ggml_tensor* W_op, ggml_tensor* b_op,
                                   int n_heads, int n_points,
                                   int H, int W,
                                   DefAttnArgs* args_userdata) {
    const int dim = (int)query_in->ne[0];
    const int NQ  = (int)query_in->ne[1];
    const int N_in = (int)memory->ne[1];
    const int head_dim = dim / n_heads;

    /* sampling_offsets: Linear(dim, n_heads*L*n_points*2). For L=1:
     *   raw ne = (n_heads*n_points*2, NQ).
     *   Reshape to (2, n_points, n_heads, NQ) (xy fastest, then p, then h, then q). */
    ggml_tensor* so_raw = linear(ctx, query_in, W_so, b_so);
    ggml_tensor* so = ggml_reshape_4d(ctx, so_raw, 2, n_points, n_heads, NQ);

    /* sampling_locations = ref_xy + (so / n_points) * ref_wh * 0.5
     * = ref_xy + so * (0.5 / n_points) * ref_wh.
     *
     * Use elementwise mul + broadcast. so ne = (2, n_points, n_heads, NQ);
     * ref_wh, ref_xy need to be (2, 1, 1, NQ) to broadcast. */
    ggml_tensor* ref_wh_4d = ggml_reshape_4d(ctx, ref_wh, 2, 1, 1, NQ);
    ggml_tensor* ref_xy_4d = ggml_reshape_4d(ctx, ref_xy, 2, 1, 1, NQ);

    ggml_tensor* so_scaled = ggml_scale(ctx, so, 0.5f / (float)n_points);
    ggml_tensor* so_wh = ggml_mul(ctx, so_scaled, ref_wh_4d);
    ggml_tensor* sloc = ggml_add(ctx, so_wh, ref_xy_4d);

    /* attention_weights: Linear(dim, n_heads*L*n_points). For L=1:
     *   raw ne = (n_heads*n_points, NQ).
     *   Reshape to (n_points, n_heads, NQ). Softmax over ne[0]=n_points. */
    ggml_tensor* aw_raw = linear(ctx, query_in, W_aw, b_aw);
    ggml_tensor* aw = ggml_reshape_3d(ctx, aw_raw, n_points, n_heads, NQ);
    aw = ggml_soft_max(ctx, aw);

    /* value: Linear(dim, dim) over memory. ne = (dim, N_in).
     *   Reshape to (head_dim, n_heads, N_in) — torch transpose+view equivalent. */
    ggml_tensor* val = linear(ctx, memory, W_vp, b_vp);
    val = ggml_reshape_3d(ctx, val, head_dim, n_heads, N_in);
    val = ggml_cont(ctx, val);

    /* Make sure the inputs to the custom op are contiguous (the op walks raw
     * pointers; non-contiguous views would silently produce wrong results). */
    sloc = ggml_cont(ctx, sloc);
    aw   = ggml_cont(ctx, aw);

    args_userdata->H = H;
    args_userdata->W = W;
    args_userdata->n_heads = n_heads;
    args_userdata->n_points = n_points;
    args_userdata->head_dim = head_dim;

    ggml_tensor* inputs[3] = { val, sloc, aw };
    ggml_tensor* sampled = ggml_custom_4d(
        ctx, GGML_TYPE_F32,
        /*ne0*/ dim, /*ne1*/ NQ, /*ne2*/ 1, /*ne3*/ 1,
        inputs, /*n_args*/ 3,
        def_attn_op,
        /*n_tasks*/ GGML_N_TASKS_MAX,
        (void*)args_userdata);

    /* Output projection. */
    ggml_tensor* out = ggml_mul_mat(ctx, W_op, sampled);
    out = ggml_add(ctx, out, b_op);
    return out;
}

ggml_tensor* decoder_layer(ggml_context* ctx, const Model& m,
                           ggml_tensor* tgt, ggml_tensor* memory,
                           ggml_tensor* query_pos,
                           ggml_tensor* ref_xy, ggml_tensor* ref_wh,
                           int layer_idx, int H, int W,
                           DefAttnArgs* args_userdata) {
    const std::string p = "decoder.layers." + std::to_string(layer_idx) + ".";
    auto get = [&](const char* suffix) -> ggml_tensor* {
        return fetch(m, p + suffix);
    };

    ggml_tensor* W_ip   = get("self_attn.in_proj.weight");
    ggml_tensor* b_ip   = get("self_attn.in_proj.bias");
    ggml_tensor* W_op   = get("self_attn.out_proj.weight");
    ggml_tensor* b_op   = get("self_attn.out_proj.bias");
    ggml_tensor* n1w    = get("norm1.weight");
    ggml_tensor* n1b    = get("norm1.bias");
    ggml_tensor* W_so   = get("cross_attn.sampling_offsets.weight");
    ggml_tensor* b_so   = get("cross_attn.sampling_offsets.bias");
    ggml_tensor* W_aw   = get("cross_attn.attention_weights.weight");
    ggml_tensor* b_aw   = get("cross_attn.attention_weights.bias");
    ggml_tensor* W_vp   = get("cross_attn.value_proj.weight");
    ggml_tensor* b_vp   = get("cross_attn.value_proj.bias");
    ggml_tensor* W_cop  = get("cross_attn.output_proj.weight");
    ggml_tensor* b_cop  = get("cross_attn.output_proj.bias");
    ggml_tensor* n2w    = get("norm2.weight");
    ggml_tensor* n2b    = get("norm2.bias");
    ggml_tensor* W_l1   = get("linear1.weight");
    ggml_tensor* b_l1   = get("linear1.bias");
    ggml_tensor* W_l2   = get("linear2.weight");
    ggml_tensor* b_l2   = get("linear2.bias");
    ggml_tensor* n3w    = get("norm3.weight");
    ggml_tensor* n3b    = get("norm3.bias");

    if (!W_ip || !b_ip || !W_op || !b_op || !n1w || !n1b ||
        !W_so || !b_so || !W_aw || !b_aw || !W_vp || !b_vp ||
        !W_cop || !b_cop || !n2w || !n2b ||
        !W_l1 || !b_l1 || !W_l2 || !b_l2 || !n3w || !n3b) {
        return nullptr;
    }

    const std::string pub = "decoder.layer." + std::to_string(layer_idx) + ".";

    const int sa_heads  = (int)m.config.decoder.self_attn_heads;
    const int ca_heads  = (int)m.config.decoder.cross_attn_heads;
    const int n_points  = (int)m.config.decoder.cross_attn_n_points;

    /* ---- Self-attention ---- */
    ggml_tensor* qk_input = ggml_add(ctx, tgt, query_pos);
    ggml_tensor* sa = self_attn_packed(ctx, qk_input, tgt,
                                       W_ip, b_ip, W_op, b_op, sa_heads);
    publish(pub + "self_attn.output", sa);

    /* Post-LN: tgt = norm1(tgt + sa) */
    ggml_tensor* x = ggml_add(ctx, tgt, sa);
    x = ops::layer_norm(ctx, x, n1w, n1b);
    publish(pub + "norm1.output", x);

    /* ---- Deformable cross-attention ---- */
    ggml_tensor* ca_query = ggml_add(ctx, x, query_pos);
    ggml_tensor* ca = cross_attn_deformable(
        ctx, ca_query, memory, ref_xy, ref_wh,
        W_so, b_so, W_aw, b_aw, W_vp, b_vp, W_cop, b_cop,
        ca_heads, n_points, H, W, args_userdata);
    publish(pub + "cross_attn.output", ca);

    /* Post-LN: tgt = norm2(tgt + ca) */
    x = ggml_add(ctx, x, ca);
    x = ops::layer_norm(ctx, x, n2w, n2b);
    publish(pub + "norm2.output", x);

    /* ---- FFN (linear1 → ReLU → linear2). Default activation in rfdetr is
     *      "relu" (see TransformerDecoderLayer.__init__). ---- */
    ggml_tensor* h = ggml_mul_mat(ctx, W_l1, x);
    h = ggml_add(ctx, h, b_l1);
    publish(pub + "linear1.output", h);
    h = ggml_relu(ctx, h);
    h = ggml_mul_mat(ctx, W_l2, h);
    h = ggml_add(ctx, h, b_l2);
    publish(pub + "linear2.output", h);

    /* Post-LN: tgt = norm3(tgt + ffn) */
    x = ggml_add(ctx, x, h);
    x = ops::layer_norm(ctx, x, n3w, n3b);
    publish(pub + "output", x);
    return x;
}

}  // namespace

void compute_query_sine_embed(const float* refpoints, int num_queries,
                              int d_half, float* out) {
    /* Matches gen_sineembed_for_position(pos_tensor, dim=128) for 4D refpoints
     * with concat order [pos_y, pos_x, pos_w, pos_h].
     *
     * refpoints layout: (4, NQ) F32 with [cx, cy, w, h] per query — so for
     * query q, refpoints[q*4 + k] is the k-th coord. ggml stores ne=(4, NQ)
     * column-major: offset = k + q*4, same as our linear layout here.
     *
     * Output layout: (4*d_half, NQ) F32 column-major. For query q the 512-dim
     * embed is at out[q*(4*d_half) : (q+1)*(4*d_half)] in concat order
     * [pos_y(d_half) | pos_x(d_half) | pos_w(d_half) | pos_h(d_half)]. */
    const float TWO_PI = 6.28318530717958647692f;
    std::vector<float> inv_dim_t((size_t)d_half);
    for (int k = 0; k < d_half; ++k) {
        /* dim_t = 10000 ** (2 * (k // 2) / d_half). Use logf+expf for accuracy. */
        const int kk = (k / 2) * 2;  /* 2*(k//2) */
        inv_dim_t[(size_t)k] = std::exp(-std::log(10000.0f) * (float)kk / (float)d_half);
    }
    /* Order of components in `out`: [y, x, w, h]. Input refpoints order is
     * [cx, cy, w, h] (the LWDETR convention). So component index -> refpoint
     * coord index: y→1, x→0, w→2, h→3. */
    const int comp_to_rp[4] = { 1, 0, 2, 3 };
    for (int q = 0; q < num_queries; ++q) {
        for (int c = 0; c < 4; ++c) {
            const float v = refpoints[q * 4 + comp_to_rp[c]] * TWO_PI;
            float* out_block = out + (size_t)q * 4 * d_half + (size_t)c * d_half;
            for (int k = 0; k < d_half; k += 2) {
                const float arg = v * inv_dim_t[(size_t)k];
                out_block[k]     = std::sin(arg);
                out_block[k + 1] = std::cos(arg);
            }
        }
    }
}

ggml_tensor* decoder_forward(ggml_context* ctx, const Model& m,
                             ggml_tensor* tgt,
                             ggml_tensor* memory,
                             ggml_tensor* refpoints,
                             ggml_tensor* query_pos,
                             int memory_H, int memory_W) {
    if (!tgt || !memory || !refpoints || !query_pos) {
        rfdetr_logf(RFDETR_LOG_ERROR, "decoder_forward: null input");
        return nullptr;
    }
    if (tgt->ne[0] != (int64_t)m.config.decoder.model_dim) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "decoder_forward: tgt ne[0]=%lld != model_dim=%u",
                    (long long)tgt->ne[0], m.config.decoder.model_dim);
        return nullptr;
    }
    if (refpoints->ne[0] != 4) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "decoder_forward: refpoints ne[0]=%lld != 4",
                    (long long)refpoints->ne[0]);
        return nullptr;
    }

    const int NQ = (int)tgt->ne[1];

    /* Slice refpoints (4, NQ) → ref_xy (2, NQ) and ref_wh (2, NQ). The slice
     * needs cont because downstream broadcast-multiply via reshape expects
     * a contiguous source. */
    const size_t rp_row_size = refpoints->nb[1];
    const size_t rp_esz      = ggml_element_size(refpoints);
    ggml_tensor* ref_xy = ggml_cont(ctx, ggml_view_2d(
        ctx, refpoints, /*ne0*/ 2, /*ne1*/ NQ,
        /*nb1*/ rp_row_size, /*offset*/ (size_t)0 * rp_esz));
    ggml_tensor* ref_wh = ggml_cont(ctx, ggml_view_2d(
        ctx, refpoints, /*ne0*/ 2, /*ne1*/ NQ,
        /*nb1*/ rp_row_size, /*offset*/ (size_t)2 * rp_esz));

    /* The DefAttnArgs userdata lives one per layer (3 layers). Store them in
     * static thread-local storage so their address remains valid for the life
     * of the graph compute.
     *
     * NOTE: this means decoder_forward is not safe to call concurrently from
     * the same thread before the previous graph completes — that's fine for
     * the single-threaded inference path. */
    static thread_local DefAttnArgs tls_args[8];  // 8 layers max; rfdetr-base uses 3

    ggml_tensor* x = tgt;
    for (uint32_t i = 0; i < m.config.decoder.layers; ++i) {
        x = decoder_layer(ctx, m, x, memory, query_pos, ref_xy, ref_wh,
                          (int)i, memory_H, memory_W, &tls_args[i]);
        if (!x) return nullptr;
    }

    ggml_tensor* nfw = fetch(m, "decoder.norm.weight");
    ggml_tensor* nfb = fetch(m, "decoder.norm.bias");
    if (!nfw || !nfb) return nullptr;
    x = ops::layer_norm(ctx, x, nfw, nfb);
    publish("decoder.norm.output", x);
    return x;
}

}  // namespace rfdetr
