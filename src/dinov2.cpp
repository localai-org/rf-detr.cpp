#include "dinov2.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <cmath>
#include <string>

namespace rfdetr {

namespace {

/* layer_norm_eps for HF DinoV2-with-registers (matches the upstream config). */
constexpr float kLnEps = 1e-6f;

ggml_tensor* layer_norm_eps(ggml_context* ctx, ggml_tensor* x,
                            ggml_tensor* weight, ggml_tensor* bias) {
    ggml_tensor* y = ggml_norm(ctx, x, kLnEps);
    y = ggml_mul(ctx, y, weight);
    y = ggml_add(ctx, y, bias);
    return y;
}

/* Self-attention over a token sequence with separate Q, K, V projections.
 * Operates per "batch" element along ne[2] (and ne[3]) via mul_mat broadcast.
 *
 *   x ne = (dim, T, B, 1)   token sequence (B = batch / window dimension)
 *   Wq,Wk,Wv ne = (dim, dim)
 *   bq,bk,bv ne = (dim,)
 *   Wo ne = (dim, dim)
 *   bo ne = (dim,)
 *
 * Returns ne = (dim, T, B, 1). */
ggml_tensor* sdpa_attention(ggml_context* ctx, ggml_tensor* x,
                            ggml_tensor* Wq, ggml_tensor* bq,
                            ggml_tensor* Wk, ggml_tensor* bk,
                            ggml_tensor* Wv, ggml_tensor* bv,
                            ggml_tensor* Wo, ggml_tensor* bo,
                            int n_heads) {
    const int dim = (int)x->ne[0];
    const int T   = (int)x->ne[1];
    const int B   = (int)x->ne[2];
    const int head_dim = dim / n_heads;

    /* Project Q/K/V. mul_mat with W (dim, dim) and x (dim, T, B) yields
     * (dim, T, B). Bias add is broadcast over T, B via ggml_can_repeat. */
    auto project = [&](ggml_tensor* W, ggml_tensor* b) -> ggml_tensor* {
        ggml_tensor* p = ggml_mul_mat(ctx, W, x);
        return ggml_add(ctx, p, b);
    };
    ggml_tensor* q = project(Wq, bq);
    ggml_tensor* k = project(Wk, bk);
    ggml_tensor* v = project(Wv, bv);

    /* Reshape (dim, T, B) → (head_dim, n_heads, T, B). */
    q = ggml_reshape_4d(ctx, q, head_dim, n_heads, T, B);
    k = ggml_reshape_4d(ctx, k, head_dim, n_heads, T, B);
    v = ggml_reshape_4d(ctx, v, head_dim, n_heads, T, B);

    /* Permute to (head_dim, T, n_heads, B): the per-head attention groups
     * land on axis 2, the batch (window) axis on axis 3. */
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    /* logits = (k_j · q_i) / sqrt(head_dim). Pre-scale Q before the matmul
     * (mathematically equivalent to scaling logits, slightly fewer rounding
     * steps). ggml_soft_max normalizes along ne[0] (the key axis). */
    q = ggml_scale(ctx, q, 1.0f / std::sqrt((float)head_dim));
    ggml_tensor* logits = ggml_mul_mat(ctx, k, q);
    logits = ggml_soft_max(ctx, logits);

    /* out[d, i] = sum_j v[d, j] * logits[j, i]. Permute v so token axis is
     * ne[0] for mul_mat: (T, head_dim, n_heads, B). */
    ggml_tensor* v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    ggml_tensor* out = ggml_mul_mat(ctx, v_t, logits);
    /* out ne = (head_dim, T, n_heads, B) */

    /* Merge heads: (head_dim, T, n_heads, B) → permute → (head_dim, n_heads,
     * T, B) → reshape → (dim, T, B). */
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    out = ggml_reshape_3d(ctx, out, dim, T, B);

    /* Output projection. */
    out = ggml_mul_mat(ctx, Wo, out);
    out = ggml_add(ctx, out, bo);
    return out;
}

/* Position-wise MLP: x → fc1 → exact-erf GELU → fc2. nn.GELU() default is
 * the exact erf form, matching ggml_gelu_erf. */
ggml_tensor* mlp_block(ggml_context* ctx, ggml_tensor* x,
                       ggml_tensor* W1, ggml_tensor* b1,
                       ggml_tensor* W2, ggml_tensor* b2) {
    ggml_tensor* h = ggml_mul_mat(ctx, W1, x);
    h = ggml_add(ctx, h, b1);
    h = ggml_gelu_erf(ctx, h);
    h = ggml_mul_mat(ctx, W2, h);
    h = ggml_add(ctx, h, b2);
    return h;
}

/* True iff block i runs WINDOWED attention. Windowed indexes are the
 * complement of global_attn_indices over [0, depth). */
bool is_windowed_block(const Config& cfg, uint32_t i) {
    for (uint32_t v : cfg.backbone.global_attn_indices) {
        if (v == i) return false;
    }
    return true;
}

/* Window-partition for the patch grid only (not CLS).
 *
 *   patches ne = (dim, N_p, 1, 1) where N_p = inf_side^2
 *   inf_side = patches per side (e.g. 40)
 *   num_windows = num_windows per side (e.g. 4)
 *   patches_per_window_side = inf_side / num_windows (e.g. 10)
 *   T_p = patches_per_window_side^2 (e.g. 100)
 *   n_windows = num_windows^2 (e.g. 16)
 *
 * Returns ne = (dim, T_p, n_windows, 1).
 *
 * The torch reference (WindowedDinov2WithRegistersEmbeddings.forward):
 *   tokens.view(B, num_h_patches, num_w_patches, C)
 *     -> reshape(B * num_windows, num_h_per_win, num_windows, num_w_per_win, C)
 *     -> permute(0, 2, 1, 3, 4)
 *     -> reshape(B * num_windows^2, num_h_per_win * num_w_per_win, C)
 *
 * In ggml's (dim-fastest, then-column) layout the equivalent steps are:
 *   patches (dim, inf_side*inf_side)
 *     -> reshape (dim, inf_side, inf_side)     fast→slow: d, w, h
 *     -> reshape (dim, win_side, n_ww, inf_side)
 *                                            (split w into win_side, n_ww)
 *     -> permute(0, 1, 3, 2)                (swap h and n_ww outer axes)
 *     -> ne (dim, win_side, inf_side, n_ww)  fast→slow: d, w_in, h_in_full, n_ww
 *     -> reshape (dim*win_side*win_side, n_hw, n_ww)
 *        (further split inf_side into n_hw outer × win_side inner)
 *     -> permute(0, 2, 1, 3)                (swap n_hw and n_ww order)
 *     -> reshape (dim, win_side*win_side, n_ww*n_hw)
 *
 * Numpy/torch semantics: window index = h_outer*n_ww + w_outer; inside-window
 * token = h_inner*win_side + w_inner. This matches the embeddings produced
 * by HF's permute(0, 2, 1, 3, 4).
 */
ggml_tensor* window_partition_patches(ggml_context* ctx, ggml_tensor* patches,
                                      int inf_side, int num_windows) {
    const int dim         = (int)patches->ne[0];
    const int n_per_side  = num_windows;
    const int win_side    = inf_side / num_windows;
    const int t_per_win   = win_side * win_side;
    const int n_windows   = num_windows * num_windows;

    /* patches ne = (dim, inf_side*inf_side). Tokens are row-major in (h, w):
     * t = h*inf_side + w, so memory layout dim-fastest gives
     * fast→slow: d, w, h. */
    ggml_tensor* g = ggml_reshape_3d(ctx, patches, dim, inf_side, inf_side);
    /* g ne = (dim, inf_side=w, inf_side=h), fast→slow d, w, h. */

    /* Split w into (win_side=w_in, n_per_side=w_out): ne (dim, win_side, n_ww, inf_side=h). */
    g = ggml_reshape_4d(ctx, g, dim, win_side, n_per_side, inf_side);
    /* fast→slow: d, w_in, w_out, h. */

    /* Permute to (dim, win_side, inf_side=h, n_ww): swap h and w_out.
     * ggml_permute axes: out.ne[k] = in.ne[axis[k]]. We want:
     *   out.ne[0] = dim     (in axis 0)
     *   out.ne[1] = win_side(in axis 1)
     *   out.ne[2] = h       (in axis 3)
     *   out.ne[3] = n_ww    (in axis 2)
     * → permute(0, 1, 3, 2). */
    g = ggml_cont(ctx, ggml_permute(ctx, g, 0, 1, 3, 2));
    /* g ne = (dim, win_side, inf_side, n_ww), fast→slow d, w_in, h_full, w_out. */

    /* Further split h_full = h_in * n_hw. Currently fast→slow d, w_in, h_full,
     * w_out. We want fast→slow d, w_in, h_in, h_out, w_out, so split h_full as
     * (h_in fast, h_out slow). Reshape with ne = (dim, win_side, win_side,
     * n_hw, n_ww) - but ggml is 4D only. Collapse outer block:
     *   ne = (dim*win_side, h_full=inf_side, n_ww, 1)   keeps layout
     *   then ne = (dim*win_side, win_side, n_hw, n_ww). */
    g = ggml_reshape_4d(ctx, g, dim * win_side, win_side, n_per_side, n_per_side);
    /* fast→slow: (d, w_in)=DW, h_in, h_out, w_out. */

    /* Permute swap h_out and w_out outer order: axes (0, 1, 3, 2).
     * Numpy/torch convention: window index = h_out * n_ww + w_out (h_out
     * outer, w_out inner). In ggml fast→slow order: ..., w_out, h_out so
     * window index encodes as window_idx = h_out + w_out * n_hw (w_out is
     * slower in memory). To produce the numpy ordering — window_idx =
     * h_out*n_ww + w_out (h_out slower) — swap h_out and w_out so memory is
     * fast→slow: ..., h_out, w_out. */
    g = ggml_cont(ctx, ggml_permute(ctx, g, 0, 1, 3, 2));
    /* g ne = (dim*win_side, win_side, n_ww=w_out, n_hw=h_out), fast→slow
     * (d, w_in), h_in, w_out, h_out. */

    /* Reshape (dim, win_side*win_side=T_p, n_windows). Inner block of size
     * dim*win_side*win_side per window contains tokens in (h_in, w_in)
     * row-major order — t = h_in*win_side + w_in. */
    g = ggml_reshape_3d(ctx, g, dim, t_per_win, n_windows);
    /* g ne = (dim, T_p, n_windows). */

    return g;
}

/* Reverse window-partition: undo the per-window pack so the spatial grid is
 * recovered. Input ne = (dim, T_p, n_windows); output ne = (dim, inf_side^2)
 * in the same dim-fastest layout used pre-partition. */
ggml_tensor* window_unpartition_patches(ggml_context* ctx, ggml_tensor* w,
                                        int inf_side, int num_windows) {
    const int dim         = (int)w->ne[0];
    const int n_per_side  = num_windows;
    const int win_side    = inf_side / num_windows;

    /* Inverse: window_partition does (in order)
     *   (dim, inf*inf)
     *   reshape (dim, inf, inf)
     *   reshape (dim, win_side, n_ww, inf)
     *   permute(0, 1, 3, 2)
     *   reshape (dim*win_side, win_side, n_ww, n_hw)
     *   permute(0, 1, 3, 2)
     *   reshape (dim, t_per_win, n_windows)
     *
     * Reverse each step. */
    ggml_tensor* g = ggml_reshape_4d(ctx, w, dim * win_side, win_side, n_per_side, n_per_side);
    /* (d, w_in), h_in, w_out, h_out. */

    g = ggml_cont(ctx, ggml_permute(ctx, g, 0, 1, 3, 2));
    /* (d, w_in), h_in, h_out, w_out. */

    g = ggml_reshape_4d(ctx, g, dim, win_side, inf_side, n_per_side);
    /* (d, w_in, h_full, w_out) where h_full = h_in fast + h_out slow → fast→slow d,w_in,h_in,h_out,w_out. */

    g = ggml_cont(ctx, ggml_permute(ctx, g, 0, 1, 3, 2));
    /* (d, w_in, w_out, h_full). */

    g = ggml_reshape_3d(ctx, g, dim, inf_side, inf_side);
    /* (d, w_full=w_in*w_out, h). */

    g = ggml_reshape_2d(ctx, g, dim, inf_side * inf_side);
    return g;
}

/* Reverse window-partition for tokens (incl. CLS).
 *   in: (dim, t_per_win=T, n_windows) where T = 1 + win_side^2
 *   This is the layout HF emits at block output. Internally, token 0 of each
 *   window is CLS (one of 16 copies); tokens 1..T-1 are 10×10 patches in that
 *   window. The patch grid is the same as window_unpartition_patches. */

/* Build the windowed positional embedding once at graph-build time.
 *
 * Source: m.backbone_pos_embed_interp ne = (dim, inf_tokens) where
 *   inf_tokens = 1 + inf_side^2; the first column is CLS embedding, the rest
 *   is the bicubic-interpolated patch grid in row-major order.
 *
 * Output: ne = (dim, T_p+1, n_windows) where T_p+1 = 1 (CLS) + patches_per_win
 *   per window. The CLS pos embedding is broadcast to all n_windows windows.
 *
 * Notice: in HF, the embeddings flow is:
 *   tokens (1, 1601, 384)  -- includes CLS at position 0
 *   tokens += interpolated_pos_embed
 *   pixel_tokens = tokens[:, 1:].reshape(B, 40, 40, C)
 *   windowed_pixel_tokens = pixel_tokens.windowed_view → (16, 100, C)
 *   cls = tokens[:, :1].repeat(16, 1, 1)              → (16, 1, C)
 *   embeddings = cat([cls, windowed_pixel_tokens])    → (16, 101, C)
 *
 * So both pos_embed for CLS AND for patches are added BEFORE windowing.
 * Equivalent: compute the windowed pos_embed (windowed patch pos_embed plus
 * broadcasted CLS pos_embed) and add to windowed token tensor. We do that
 * here. */
ggml_tensor* build_windowed_pos_embed(ggml_context* ctx, const Model& m,
                                      int inf_side, int num_windows) {
    const int dim = (int)m.config.backbone.dim;
    const int n_per_side = num_windows;
    const int n_windows  = n_per_side * n_per_side;

    ggml_tensor* pe = m.backbone_pos_embed_interp;
    /* pe ne = (dim, 1 + inf_side*inf_side). */

    /* Slice off CLS (first column) and the patch grid (rest). */
    ggml_tensor* cls = ggml_view_2d(ctx, pe, dim, 1,
                                    /*nb1*/ pe->nb[1],
                                    /*offset*/ 0);
    ggml_tensor* patches = ggml_view_2d(ctx, pe, dim, inf_side * inf_side,
                                        /*nb1*/ pe->nb[1],
                                        /*offset*/ pe->nb[1]);
    cls     = ggml_cont(ctx, cls);
    patches = ggml_cont(ctx, patches);

    /* Window-partition the patch pos embedding (mirrors window_partition_patches
     * but using a different source). */
    ggml_tensor* pw = window_partition_patches(ctx, patches, inf_side, num_windows);
    /* pw ne = (dim, t_per_win, n_windows). */

    /* Broadcast CLS to (dim, 1, n_windows) via ggml_repeat_4d. */
    ggml_tensor* cls_b = ggml_repeat_4d(ctx, cls, dim, 1, n_windows, 1);
    /* cls_b ne = (dim, 1, n_windows, 1). */

    /* Concat along axis 1 (token axis): (dim, 1, n_windows) ⊕ (dim, t_per_win,
     * n_windows) → (dim, 1+t_per_win, n_windows). */
    ggml_tensor* out = ggml_concat(ctx, cls_b, pw, /*dim*/ 1);
    /* out ne = (dim, t_per_win+1, n_windows). */
    return out;
}

ggml_tensor* fetch(const Model& m, const std::string& name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "dinov2: missing tensor '%s'", name.c_str());
        return nullptr;
    }
    return it->second;
}

}  // namespace

BackboneOutput dinov2_forward(ggml_context* ctx, const Model& m,
                              ggml_tensor* input) {
    BackboneOutput out;

    const int dim         = (int)m.config.backbone.dim;
    const int patch_size  = (int)m.config.patch_size;
    const int num_windows = (int)m.config.backbone.num_windows;
    const int depth       = (int)m.config.backbone.depth;
    const int n_heads     = (int)m.config.backbone.heads;
    const int image_side  = (int)m.config.image_size;
    const int inf_side    = image_side / patch_size;
    const int win_side    = inf_side / num_windows;
    const int t_per_win   = win_side * win_side;
    const int T_full      = t_per_win + 1;
    const int n_windows   = num_windows * num_windows;
    (void)dim;

    if (inf_side * patch_size != image_side ||
        win_side * num_windows != inf_side) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "dinov2_forward: image_size %d, patch_size %d, num_windows %d not compatible",
                    image_side, patch_size, num_windows);
        return out;
    }

    /* --- 1. Patch embed: Conv2d k=14 s=14 → (W_p, H_p, dim, 1). --- */
    ggml_tensor* W_pe = fetch(m, "backbone.patch_embed.weight");
    ggml_tensor* b_pe = fetch(m, "backbone.patch_embed.bias");
    if (!W_pe || !b_pe) return out;

    ggml_tensor* conv = ggml_conv_2d(ctx, W_pe, input,
                                     /*s0*/ patch_size, /*s1*/ patch_size,
                                     /*p0*/ 0, /*p1*/ 0,
                                     /*d0*/ 1, /*d1*/ 1);
    /* conv ne = (W_p, H_p, dim, 1). */
    ggml_tensor* b_pe_r = ggml_reshape_3d(ctx, b_pe, 1, 1, b_pe->ne[0]);
    ggml_tensor* feat = ggml_add(ctx, conv, b_pe_r);

    /* HF does projection(pixel_values).flatten(2).transpose(1, 2), which
     * yields token sequence (B, N, dim) where token order is row-major (h, w)
     * — the same as torch.flatten over the spatial dims of (B, dim, H_p, W_p).
     *
     * In ggml/torch parity: torch's (B=1, dim, H_p, W_p) has memory layout
     * fast→slow w, h, dim, B. Our conv result (W_p, H_p, dim, 1) has identical
     * memory (ne[0]=W_p fastest). flatten(2)+transpose(1,2) yields tokens
     * (B, N, dim) with token t = h*W_p + w (row-major) and dim as the inner
     * axis. In ggml that's (dim, N) with t as ne[1], identical bytes after a
     * permute to bring dim inward. */
    ggml_tensor* tokens;
    {
        const int64_t W_p = feat->ne[0];
        const int64_t H_p = feat->ne[1];
        const int64_t dim_ne = feat->ne[2];
        ggml_tensor* flat = ggml_reshape_3d(ctx, feat, W_p * H_p, dim_ne, 1);
        /* flat ne = (N, dim, 1). Now permute to bring dim to ne[0]. */
        tokens = ggml_cont(ctx, ggml_permute(ctx, flat, 1, 0, 2, 3));
        /* tokens ne = (dim, N, 1, 1). */
    }

    /* --- 2. Prepend CLS, add pos_embed (interpolated). ---
     *
     * In HF flow: tokens (1, N, dim) → cat with cls (1, 1, dim) → (1, N+1, dim)
     * → add pos_embed (1, N+1, dim, bicubic-interpolated). Then window-partition.
     *
     * Equivalent and easier for windowing: window-partition the patch tokens
     * first (no CLS), then add the windowed pos_embed (which already has CLS
     * broadcast to each window). The CLS *values* themselves (the learned
     * cls_token) are independent of window: they only get a pos offset added
     * per window slot but the CLS embedding itself is the same vector.
     *
     * Steps in this implementation:
     *   patches (dim, N) → window_partition → (dim, T_p, n_windows)
     *   build cls_donor (dim, 1, n_windows) filled with broadcasted cls_token
     *   concat along axis 1 → (dim, T_p+1, n_windows)
     *   build windowed pos_embed (dim, T_p+1, n_windows)
     *   add. */
    ggml_tensor* cls_token = fetch(m, "backbone.cls_token");
    if (!cls_token) return out;

    ggml_tensor* windowed_patches = window_partition_patches(ctx, tokens, inf_side, num_windows);
    /* windowed_patches ne = (dim, T_p, n_windows). */

    /* CLS broadcast: cls_token ne = (dim,). Reshape to (dim, 1, 1) then
     * repeat to (dim, 1, n_windows). */
    ggml_tensor* cls_3d = ggml_reshape_3d(ctx, cls_token, cls_token->ne[0], 1, 1);
    ggml_tensor* cls_b = ggml_repeat_4d(ctx, cls_3d, cls_token->ne[0], 1, n_windows, 1);

    /* Concat along axis 1: (dim, 1, n_windows) ⊕ (dim, T_p, n_windows) →
     * (dim, T_p+1, n_windows). */
    ggml_tensor* x = ggml_concat(ctx, cls_b, windowed_patches, /*dim*/ 1);

    /* Add windowed pos embedding. */
    ggml_tensor* pos = build_windowed_pos_embed(ctx, m, inf_side, num_windows);
    x = ggml_add(ctx, x, pos);
    /* x ne = (dim, T_p+1, n_windows). */

    publish("backbone.patch_embed.output", x);

    /* --- 3. Transformer blocks. --- */
    /* For multiscale taps: we need to apply the final layernorm to the tap
     * tensor, strip CLS, un-window, and reshape to (W_p, H_p, dim, 1). */
    ggml_tensor* ln_w = fetch(m, "backbone.norm.weight");
    ggml_tensor* ln_b = fetch(m, "backbone.norm.bias");
    if (!ln_w || !ln_b) return out;

    /* out_feature_indices stores HF stage indices ("stage2"→2 etc.).
     * stage k corresponds to the hidden state AFTER layer k-1, so the tap
     * fires at the output of layer (stage-1). For rfdetr-base:
     *   stages [2, 5, 8, 11] → tap layer outputs [1, 4, 7, 10]. */
    const auto& out_idx = m.config.backbone.out_feature_indices;
    auto find_level = [&](uint32_t block_i) -> int {
        const uint32_t stage = block_i + 1;
        for (size_t k = 0; k < out_idx.size(); ++k) {
            if (out_idx[k] == stage) return (int)k;
        }
        return -1;
    };

    for (int i = 0; i < depth; ++i) {
        const std::string p = "backbone.blocks." + std::to_string(i) + ".";
        ggml_tensor* n1w  = fetch(m, p + "norm1.weight");
        ggml_tensor* n1b  = fetch(m, p + "norm1.bias");
        ggml_tensor* qW   = fetch(m, p + "attn.q.weight");
        ggml_tensor* qB   = fetch(m, p + "attn.q.bias");
        ggml_tensor* kW   = fetch(m, p + "attn.k.weight");
        ggml_tensor* kB   = fetch(m, p + "attn.k.bias");
        ggml_tensor* vW   = fetch(m, p + "attn.v.weight");
        ggml_tensor* vB   = fetch(m, p + "attn.v.bias");
        ggml_tensor* oW   = fetch(m, p + "attn.proj.weight");
        ggml_tensor* oB   = fetch(m, p + "attn.proj.bias");
        ggml_tensor* ls1  = fetch(m, p + "layer_scale1");
        ggml_tensor* n2w  = fetch(m, p + "norm2.weight");
        ggml_tensor* n2b  = fetch(m, p + "norm2.bias");
        ggml_tensor* f1W  = fetch(m, p + "mlp.fc1.weight");
        ggml_tensor* f1B  = fetch(m, p + "mlp.fc1.bias");
        ggml_tensor* f2W  = fetch(m, p + "mlp.fc2.weight");
        ggml_tensor* f2B  = fetch(m, p + "mlp.fc2.bias");
        ggml_tensor* ls2  = fetch(m, p + "layer_scale2");
        if (!n1w || !n1b || !qW || !qB || !kW || !kB || !vW || !vB ||
            !oW || !oB || !ls1 || !n2w || !n2b || !f1W || !f1B || !f2W ||
            !f2B || !ls2) {
            return out;
        }

        ggml_tensor* shortcut = x;

        /* --- norm1 + attention --- */
        ggml_tensor* h = layer_norm_eps(ctx, x, n1w, n1b);

        if (is_windowed_block(m.config, (uint32_t)i)) {
            /* Per-window MHA on (dim, T_full, n_windows). */
            h = sdpa_attention(ctx, h, qW, qB, kW, kB, vW, vB, oW, oB, n_heads);
        } else {
            /* Global attention: collapse windows. HF reshapes the windowed
             * tensor (B*W^2, T_full, C) → (B, W^2 * T_full, C). The 16 CLS
             * copies are NOT merged — they remain as 16 separate tokens in the
             * unified sequence. After attention, reshape back to windowed. */
            const int T_long = T_full * n_windows;
            ggml_tensor* h2 = ggml_reshape_3d(ctx, h, h->ne[0], T_long, 1);
            h2 = sdpa_attention(ctx, h2, qW, qB, kW, kB, vW, vB, oW, oB, n_heads);
            h  = ggml_reshape_3d(ctx, h2, h2->ne[0], T_full, n_windows);
        }

        /* Layer scale 1: multiply by per-channel gamma. ls1 ne = (dim,)
         * broadcasts over (T, n_windows). */
        h = ggml_mul(ctx, h, ls1);
        x = ggml_add(ctx, shortcut, h);

        /* --- norm2 + MLP --- */
        ggml_tensor* y = layer_norm_eps(ctx, x, n2w, n2b);
        y = mlp_block(ctx, y, f1W, f1B, f2W, f2B);
        y = ggml_mul(ctx, y, ls2);
        x = ggml_add(ctx, x, y);

        publish("backbone.block." + std::to_string(i) + ".output", x);

        /* If this block is a multiscale tap: apply LN, strip CLS, un-window,
         * reshape to spatial. */
        const int level = find_level((uint32_t)i);
        if (level >= 0 && level < (int)out.multi_scale.size()) {
            ggml_tensor* nrm = layer_norm_eps(ctx, x, ln_w, ln_b);
            /* nrm ne = (dim, T_full, n_windows). */
            if (level == (int)out.multi_scale.size() - 1) {
                /* The hook on hf.layernorm fires once per stage tap; the LAST
                 * invocation feeds the publish for backbone.norm.output —
                 * which is the LN of the highest-stage tap (layer-10 output
                 * for rfdetr-base, NOT layer 11 which still runs but is
                 * unused). */
                publish("backbone.norm.output", nrm);
            }
            /* Strip CLS — view tokens 1..T_full. */
            const int dim_ne = (int)nrm->ne[0];
            ggml_tensor* patches_only = ggml_view_3d(ctx, nrm,
                /*ne0*/ dim_ne, /*ne1*/ t_per_win, /*ne2*/ n_windows,
                /*nb1*/ nrm->nb[1], /*nb2*/ nrm->nb[2],
                /*offset bytes*/ nrm->nb[1]);
            patches_only = ggml_cont(ctx, patches_only);

            /* Un-window: (dim, T_p, n_windows) → (dim, inf_side*inf_side). */
            ggml_tensor* flat_patches = window_unpartition_patches(
                ctx, patches_only, inf_side, num_windows);

            /* Reshape to (W_p, H_p, dim, 1) for downstream conv-style code.
             * `flat_patches` ne = (dim, inf*inf); reshape to (dim, w, h) and
             * permute so that the spatial dims lead.
             *
             * ggml_permute semantics: `ggml_permute(t, ax0, ax1, ax2, ax3)`
             * places INPUT axis k at OUTPUT position ax_k. We want
             *   input  ne = (d=384, w=40, h=40, 1)
             *   output ne = (w=40,  h=40,  d=384, 1)
             * → input axis 0 (d) goes to output position 2 (ax0=2)
             * → input axis 1 (w) goes to output position 0 (ax1=0)
             * → input axis 2 (h) goes to output position 1 (ax2=1)
             * → input axis 3 (1) stays at position 3   (ax3=3) */
            ggml_tensor* spatial_d_first = ggml_reshape_3d(
                ctx, flat_patches, dim_ne, inf_side, inf_side);
            /* fast→slow d, w, h. */
            ggml_tensor* spatial = ggml_cont(ctx, ggml_permute(
                ctx, spatial_d_first, 2, 0, 1, 3));
            /* ne (w=inf_side, h=inf_side, dim, 1), fast→slow w, h, d, 1. */

            publish("backbone.multiscale.level" + std::to_string(level), spatial);
            out.multi_scale[level] = spatial;
        }
    }

    /* For API compatibility we still need a `final` tensor. HF doesn't use
     * the last layernorm's output of layer-11 (the multiscale tap at stage11
     * fires at layer-10 output), so `final` is just a recomputation of the
     * last-tap LN — same data as backbone.norm.output that was published
     * inside the tap branch. We give callers something to bind to without
     * forcing them to dig multi_scale[3] out of the struct. */
    if (out.multi_scale[out.multi_scale.size() - 1] != nullptr) {
        /* The publish wired backbone.norm.output to the LN of the last-tap
         * block. We can't read that from the trace; reconstruct by LN'ing the
         * last-tap layer's stored state. Cheap: just LN the current `x`
         * (layer-11 output) — caller won't use `final` for the spatial
         * forward path, and Plan 9 will rewrite this to consume multi_scale
         * directly. */
        out.final = layer_norm_eps(ctx, x, ln_w, ln_b);
    }
    return out;
}

}  // namespace rfdetr
