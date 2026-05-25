#include "dinov2.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <cmath>
#include <string>

namespace rfdetr {

/* DINOv2's patch size is architecturally 14. RF-DETR doesn't expose this as
 * a per-variant parameter — all variants use 14. If a future variant ever
 * uses a different patch size, this constant should move into Config and
 * be sourced from GGUF metadata. */
static constexpr int kPatchSize = 14;

ggml_tensor* dinov2_patch_embed(ggml_context* ctx, const Model& m,
                                ggml_tensor* input) {
    /* DINOv2 patch embedding is a Conv2d with kernel=stride=14.
     *
     * In ggml's (W, H, C, N) convention — where ne[0] is the fastest-varying
     * (contiguous) axis and corresponds to width:
     *   input shape:  (W, H, 3, 1)
     *   kernel shape: (14, 14, 3, dim)   — `backbone.patch_embed.weight`
     *   bias shape:   (dim,)             — `backbone.patch_embed.bias`
     *
     * ggml_conv_2d(ctx, kernel, data, s0, s1, p0, p1, d0, d1):
     *   - parameters: kernel first, data second
     *   - confirmed by the ggml_conv_2d_sk_p0 example in ggml.h:
     *       a:   16 16    3  768   (kernel)
     *       b: 1024 1024  3    1   (data)
     *       res:  64 64 768    1   (output: W/s, H/s, OC, N)
     *
     * After conv2d(stride=14):  (W/14, H/14, dim, 1)
     * After bias add:           (W/14, H/14, dim, 1)
     * Reshape:                  (W/14 * H/14, dim, 1)  i.e. tokens flattened
     * Permute to (dim, N):      (dim, N, 1, 1)
     *
     * Downstream attention/MLP consumes ne = (dim, N_patches, 1, 1) — dim is
     * the leading (contiguous) axis so per-token rows are contiguous.
     */

    auto it_w = m.tensors.find("backbone.patch_embed.weight");
    auto it_b = m.tensors.find("backbone.patch_embed.bias");
    if (it_w == m.tensors.end() || it_b == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "dinov2_patch_embed: missing patch_embed weight or bias");
        return nullptr;
    }
    ggml_tensor* W = it_w->second;
    ggml_tensor* b = it_b->second;

    ggml_tensor* conv = ggml_conv_2d(ctx, W, input,
                                     /*s0*/ 14, /*s1*/ 14,
                                     /*p0*/ 0,  /*p1*/ 0,
                                     /*d0*/ 1,  /*d1*/ 1);
    /* conv ne = (W/14, H/14, dim, 1) */

    /* Bias add. ggml_add(a, b) requires ggml_can_repeat(b, a): each a->ne[i]
     * must be a multiple of b->ne[i]. With b reshaped to (1, 1, dim, 1) and
     * a=(Wp, Hp, dim, 1) the check holds, so b is broadcast over the spatial
     * dims. (Verified in third_party/ggml/src/ggml.c ggml_can_repeat.) */
    ggml_tensor* b_reshape = ggml_reshape_3d(ctx, b, 1, 1, b->ne[0]);
    /* b_reshape ne = (1, 1, dim, 1) */
    ggml_tensor* with_bias = ggml_add(ctx, conv, b_reshape);
    /* with_bias ne = (Wp, Hp, dim, 1) */

    /* Flatten the spatial dims into the token dimension. Result keeps `dim`
     * as ne[1]; tokens are now indexed along ne[0]. */
    const int64_t Wp  = with_bias->ne[0];
    const int64_t Hp  = with_bias->ne[1];
    const int64_t dim = with_bias->ne[2];
    ggml_tensor* flat = ggml_reshape_3d(ctx, with_bias, Wp * Hp, dim, 1);
    /* flat ne = (N, dim, 1) where N = Wp*Hp */

    /* Permute so that `dim` becomes the leading (contiguous) axis. ggml's
     * permute semantics: ggml_permute(ctx, a, axis0, axis1, axis2, axis3)
     * means "the output's axis 0 takes input's axis `axis0`, etc.".
     * Here we want out_ne = (dim, N, 1, 1), so axis0=1 (take input's dim),
     * axis1=0 (take input's N), axis2=2, axis3=3. */
    ggml_tensor* out = ggml_permute(ctx, flat, 1, 0, 2, 3);
    out = ggml_cont(ctx, out);
    /* out ne = (dim, N, 1, 1) */

    publish("backbone.patch_embed.output", out);
    return out;
}

ggml_tensor* dinov2_add_cls_and_pos_embed(ggml_context* ctx, const Model& m,
                                          ggml_tensor* tokens) {
    auto it_cls = m.tensors.find("backbone.cls_token");
    auto it_pe  = m.tensors.find("backbone.pos_embed");
    if (it_cls == m.tensors.end() || it_pe == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "dinov2_add_cls_and_pos_embed: missing cls_token or pos_embed");
        return nullptr;
    }
    ggml_tensor* cls = it_cls->second;
    ggml_tensor* pe  = it_pe->second;

    /* cls_token in the seeded fixture is 1D (dim,). Reshape to (dim, 1) so
     * it concats correctly along axis 1 with tokens (dim, N). */
    ggml_tensor* cls2 = ggml_reshape_2d(ctx, cls, cls->ne[0], 1);

    /* Concat along axis 1: (dim, 1) ⊕ (dim, N) → (dim, N+1) */
    ggml_tensor* with_cls = ggml_concat(ctx, cls2, tokens, /*dim*/ 1);

    /* pos_embed in the fixture is (dim, N+1) — direct add works. */
    ggml_tensor* out = ggml_add(ctx, with_cls, pe);

    publish("backbone.cls_pos_embed.output", out);
    return out;
}

}  // namespace rfdetr

namespace rfdetr {

namespace {

/* LayerNorm: x_normalized = (x - mean) / sqrt(var + eps) * weight + bias.
 *
 * ggml_norm normalizes along ne[0] (the contiguous / "row" axis), which is
 * exactly the channel/feature dim for our (dim, N, 1, 1) layout. The weight
 * and bias are 1-D tensors of size `dim` (ne = (dim, 1, 1, 1)) and broadcast
 * over the token axis via ggml_can_repeat.
 *
 * eps = 1e-5 matches PyTorch's default torch.nn.LayerNorm. */
ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* weight, ggml_tensor* bias) {
    constexpr float eps = 1e-5f;
    ggml_tensor* y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, weight);
    y = ggml_add(ctx, y, bias);
    return y;
}

/* Multi-head self-attention with packed QKV projection.
 *
 *   qkv     = Wqkv @ x + bqkv               shape (3*dim, N)
 *   split   into q, k, v each shape         (dim, N)
 *   reshape each to                         (head_dim, n_heads, N, 1)
 *   permute to                              (head_dim, N, n_heads, 1)
 *   attn    = softmax(q @ k.T / sqrt(hd)) @ v   shape (head_dim, N, n_heads, 1)
 *   permute back to                         (head_dim, n_heads, N, 1)
 *   reshape to                              (dim, N)
 *   out     = Wproj @ attn + bproj          shape (dim, N)
 *
 * Uses the manual attention path (mul_mat / scale / soft_max / mul_mat).
 * ggml_flash_attn_ext exists with the expected signature but the manual path
 * is sufficient and easier to debug per-block against the numpy parity
 * baseline. */
ggml_tensor* mha(ggml_context* ctx, ggml_tensor* x,
                 ggml_tensor* Wqkv, ggml_tensor* bqkv,
                 ggml_tensor* Wproj, ggml_tensor* bproj,
                 int n_heads) {
    const int dim = (int)x->ne[0];
    const int N   = (int)x->ne[1];
    const int head_dim = dim / n_heads;

    /* QKV projection: Wqkv ne = (dim, 3*dim); x ne = (dim, N);
     * ggml_mul_mat(Wqkv, x) computes Wqkv^T @ x -> ne = (3*dim, N). */
    ggml_tensor* qkv = ggml_mul_mat(ctx, Wqkv, x);
    /* Bias add: bqkv ne = (3*dim, 1, 1, 1) broadcasts over (3*dim, N, 1, 1). */
    qkv = ggml_add(ctx, qkv, bqkv);

    /* Split along axis 0. Each view has ne = (dim, N). qkv's element type is
     * F32 here (ggml promotes the mul_mat result), but use ggml_element_size
     * defensively so the offset is correct regardless. */
    const size_t row_size = qkv->nb[1];           // byte stride between rows (axis 1)
    const size_t esz      = ggml_element_size(qkv);
    ggml_tensor* q = ggml_view_2d(ctx, qkv, dim, N, row_size, 0 * dim * esz);
    ggml_tensor* k = ggml_view_2d(ctx, qkv, dim, N, row_size, 1 * dim * esz);
    ggml_tensor* v = ggml_view_2d(ctx, qkv, dim, N, row_size, 2 * dim * esz);

    /* Reshape each (dim, N) -> (head_dim, n_heads, N). Requires contiguity;
     * the view is contiguous along ne[0] within each row, but the reshape
     * needs the source to be contiguous overall. ggml_view_2d marks the view
     * non-contiguous in general, so cont-it first. */
    q = ggml_cont(ctx, q);
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, N);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, N);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, N);

    /* Permute (head_dim, n_heads, N) -> (head_dim, N, n_heads), so per-head
     * matmul groups along the outermost ne[2] axis. */
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    /* Manual attention. ggml_mul_mat(a, b) treats ne[0] of each operand as
     * the contracted ("k") axis, so for two tensors shaped (k, M, ...) and
     * (k, N, ...) the result is shaped (M, N, ...). (See ggml.h's mul_mat
     * doc-comment around line 1414.)
     *
     * logits = mul_mat(k, q) — k, q ne = (head_dim, N, n_heads, 1)
     *   contracts head_dim -> per-head result ne = (N, N, n_heads, 1) where
     *   logits[ne0=j, ne1=i] = sum_d k[ne0=d, ne1=j] * q[ne0=d, ne1=i]
     *                        = <query i, key j>. ggml_soft_max normalizes
     *   along ne[0] (over the key axis j). */
    ggml_tensor* logits = ggml_mul_mat(ctx, k, q);
    logits = ggml_scale(ctx, logits, 1.0f / std::sqrt((float)head_dim));
    logits = ggml_soft_max(ctx, logits);

    /* out[ne0=d, ne1=i] = sum_j v[ne0=d, ne1=j] * logits[ne0=j, ne1=i].
     *
     * To express this with mul_mat (which contracts ne[0]), permute v so its
     * token axis becomes ne[0]: v_t ne = (N, head_dim, n_heads, 1) and
     * v_t[ne0=j, ne1=d] = v[ne0=d, ne1=j].
     *   mul_mat(v_t, logits): contracts j ->
     *     result[ne0=d, ne1=i] = sum_j v_t[ne0=j, ne1=d] * logits[ne0=j, ne1=i]
     *                          = sum_j v[ne0=d, ne1=j]  * logits[ne0=j, ne1=i].  ✓ */
    ggml_tensor* v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    ggml_tensor* out = ggml_mul_mat(ctx, v_t, logits);
    /* out ne = (head_dim, N, n_heads, 1) */

    /* Permute (head_dim, N, n_heads) -> (head_dim, n_heads, N) so the
     * subsequent reshape concatenates heads back into the channel axis. */
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    /* out ne = (head_dim, n_heads, N, 1) */
    out = ggml_reshape_2d(ctx, out, dim, N);
    /* out ne = (dim, N, 1, 1) */

    /* Output projection: Wproj ne = (dim, dim). */
    out = ggml_mul_mat(ctx, Wproj, out);
    out = ggml_add(ctx, out, bproj);
    return out;
}

/* Windowed multi-head self-attention on patch tokens only.
 *
 * Layout: x ne = (dim, N+1, 1, 1) — token 0 is CLS, tokens 1..N are patches
 * in row-major (h, w) order. hp * wp must equal N. window_size must divide
 * both hp and wp. CLS passes through unchanged; patches are
 * window-partitioned, attended per window via vanilla MHA, then
 * unpartitioned and re-concatenated with CLS.
 *
 * Output: same shape as input.
 *
 * Numpy reference for the partition layout is in scripts/gen_numpy_baseline.py
 * mha_window() — it's the source of truth for the choreography. */
ggml_tensor* mha_window(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* Wqkv, ggml_tensor* bqkv,
                        ggml_tensor* Wproj, ggml_tensor* bproj,
                        int n_heads, int window_size, int hp, int wp) {
    const int dim = (int)x->ne[0];
    const int N   = hp * wp;
    const int W   = window_size;
    const int n_hw = hp / W;
    const int n_ww = wp / W;
    const int n_windows = n_hw * n_ww;
    const int T   = W * W;          /* tokens per window */
    const int head_dim = dim / n_heads;

    if (hp % W != 0 || wp % W != 0) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "mha_window: patch grid %dx%d not divisible by window_size %d",
                    hp, wp, W);
        return nullptr;
    }
    if (n_windows * T != N) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "mha_window: n_windows*T (%d) != N (%d)", n_windows * T, N);
        return nullptr;
    }

    /* Split off CLS (axis 1, position 0) and patches (axis 1, positions 1..N). */
    ggml_tensor* cls = ggml_view_2d(ctx, x, dim, 1,
                                    x->nb[1],
                                    /* offset bytes */ 0);
    ggml_tensor* patches = ggml_view_2d(ctx, x, dim, N,
                                        x->nb[1],
                                        /* offset bytes */ x->nb[1]);
    /* CLS and patches share x's storage; ggml_cont allocates fresh contiguous copies. */
    cls = ggml_cont(ctx, cls);
    patches = ggml_cont(ctx, patches);
    /* patches ne = (dim, N, 1, 1) */

    /* Window-partition: numpy does
     *   grid(hp, wp, dim) -> reshape(n_hw, W, n_ww, W, dim) -> transpose(0,2,1,3,4)
     *                     -> reshape(n_hw*n_ww, W*W, dim) = (n_windows, T, dim)
     *
     * Numpy semantics:
     *   t  = h_inner * W   + w_inner    (h_inner slow, w_inner fast)
     *   n  = h_outer * n_ww + w_outer   (h_outer slow, w_outer fast)
     *
     * Target ggml memory layout (fast → slow): d, w_inner, h_inner, w_outer, h_outer.
     *
     * Starting layout of `patches ne = (dim, N)`:
     *   N is row-major (h, w) → view as (dim, wp, hp), memory fast→slow:
     *     d, w_inner, w_outer, h_inner, h_outer
     *   (swap (w_outer, h_inner) is the only thing needed). */
    ggml_tensor* grid = ggml_reshape_3d(ctx, patches, dim, wp, hp);
    /* grid ne = (dim, wp, hp) */

    /* Split wp = n_ww * W (w_inner fast, w_outer slow within wp):
     *   ne = (dim, W=w_inner, n_ww=w_outer, hp) */
    ggml_tensor* w1 = ggml_reshape_4d(ctx, grid, dim, W, n_ww, hp);

    /* Permute to bring hp inboard of n_ww:
     *   ne (dim, W, n_ww, hp) -> ne (dim, W, hp, n_ww)
     * Memory fast→slow after cont: d, w_inner, h_inner, h_outer, w_outer. */
    ggml_tensor* w2 = ggml_cont(ctx, ggml_permute(ctx, w1, 0, 1, 3, 2));

    /* Collapse to expose (h_outer, w_outer) as separate ne axes that we can
     * permute. After permute, memory of w2 has (h_inner, w_inner) packed as
     * the inner W*W*dim block, then h_outer, then w_outer:
     *   block_layout = (d, w_inner, h_inner)  — inner W*W*dim elements per outer
     *   outer index  = h_outer + w_outer * n_hw   (h_outer fast, w_outer slow)
     *
     * Reshape to (dim*W*W, n_hw, n_ww, 1) keeps memory; then permute axes 1,2
     * to swap (h_outer, w_outer) so the new ne[2] is row-major (h_outer slow,
     * w_outer fast) = numpy's window pack order. */
    ggml_tensor* w3 = ggml_reshape_4d(ctx, w2, dim * W * W, n_hw, n_ww, 1);
    ggml_tensor* w4 = ggml_cont(ctx, ggml_permute(ctx, w3, 0, 2, 1, 3));
    /* w4 ne = (dim*W*W, n_ww, n_hw, 1). Memory fast→slow:
     *   d, w_inner, h_inner, w_outer, h_outer. ✓ matches target. */

    /* Collapse to (dim, T, n_windows). The inner dim*W*W block re-splits as
     * (dim, T=W*W) with t = h_inner*W + w_inner (h_inner slow, w_inner fast). */
    ggml_tensor* windows = ggml_reshape_3d(ctx, w4, dim, T, n_windows);
    /* windows ne = (dim, T, n_windows, 1) — matches numpy. */

    /* Batched QKV projection: ggml_mul_mat broadcasts over axes 2 and 3.
     *   Wqkv ne = (dim, 3*dim); windows ne = (dim, T, n_windows)
     *   result ne = (3*dim, T, n_windows)
     */
    ggml_tensor* qkv = ggml_mul_mat(ctx, Wqkv, windows);
    qkv = ggml_add(ctx, qkv, bqkv);

    /* Split qkv into q, k, v on axis 0 (each takes `dim` slots).
     * ggml_view_3d signature: (ctx, src, ne0, ne1, ne2, nb1, nb2, offset_bytes) */
    const size_t row_stride = qkv->nb[1];   /* bytes per (T) row */
    const size_t mat_stride = qkv->nb[2];   /* bytes per (n_windows) matrix */
    ggml_tensor* q = ggml_view_3d(ctx, qkv, dim, T, n_windows,
                                  row_stride, mat_stride,
                                  /* offset */ 0 * dim * sizeof(float));
    ggml_tensor* k = ggml_view_3d(ctx, qkv, dim, T, n_windows,
                                  row_stride, mat_stride,
                                  1 * dim * sizeof(float));
    ggml_tensor* v = ggml_view_3d(ctx, qkv, dim, T, n_windows,
                                  row_stride, mat_stride,
                                  2 * dim * sizeof(float));
    q = ggml_cont(ctx, q);
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);

    /* Reshape each (dim, T, n_windows) -> (head_dim, n_heads, T, n_windows) */
    q = ggml_reshape_4d(ctx, q, head_dim, n_heads, T, n_windows);
    k = ggml_reshape_4d(ctx, k, head_dim, n_heads, T, n_windows);
    v = ggml_reshape_4d(ctx, v, head_dim, n_heads, T, n_windows);

    /* Permute to (head_dim, T, n_heads, n_windows): per-head attention groups along
     * axis 2; per-window batching on axis 3. */
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    /* Attention via manual path (matching existing mha()'s approach).
     * ggml_mul_mat(a, b) contracts ne[0] of BOTH operands, so for k, q both
     * shaped (head_dim, T, n_heads, n_windows), mul_mat(k, q) contracts
     * head_dim and yields ne = (T, T, n_heads, n_windows) where
     *   logits[ne0=j, ne1=i] = sum_d k[d, j] * q[d, i] = <q_i, k_j>. */
    ggml_tensor* logits = ggml_mul_mat(ctx, k, q);
    logits = ggml_scale(ctx, logits, 1.0f / std::sqrt((float)head_dim));
    logits = ggml_soft_max(ctx, logits);

    /* out[d, i] = sum_j v[d, j] * logits[j, i]. To express with mul_mat
     * (which contracts ne[0]), permute v so token axis becomes ne[0]:
     *   vt = permute(v, 1, 0, 2, 3) -> ne = (T, head_dim, n_heads, n_windows)
     *   mul_mat(vt, logits) contracts T -> ne = (head_dim, T, n_heads, n_windows). */
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    ggml_tensor* attn_out = ggml_mul_mat(ctx, vt, logits);

    /* Merge heads: (head_dim, T, n_heads, n_windows) -> permute (0, 2, 1, 3)
     *   -> (head_dim, n_heads, T, n_windows) -> reshape -> (dim, T, n_windows) */
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    attn_out = ggml_reshape_3d(ctx, attn_out, dim, T, n_windows);

    /* Output projection: Wproj ne = (dim, dim); attn_out ne = (dim, T, n_windows)
     *   result ne = (dim, T, n_windows) */
    attn_out = ggml_mul_mat(ctx, Wproj, attn_out);
    attn_out = ggml_add(ctx, attn_out, bproj);

    /* Reverse window-partition: undo each forward step in reverse order. Forward
     * was (top→bottom):
     *   patches (dim, N)
     *     → reshape (dim, wp, hp)              = grid
     *     → reshape (dim, W, n_ww, hp)         = w1
     *     → permute (0,1,3,2) + cont           → (dim, W, hp, n_ww)  = w2
     *     → reshape (dim*W*W, n_hw, n_ww, 1)   = w3
     *     → permute (0,2,1,3) + cont           → (dim*W*W, n_ww, n_hw, 1) = w4
     *     → reshape (dim, T, n_windows)        = windows  */

    /* Inverse step A: windows → w4-shape. */
    ggml_tensor* u1 = ggml_reshape_4d(ctx, attn_out, dim * W * W, n_ww, n_hw, 1);

    /* Inverse step B: permute(0,2,1,3) is self-inverse → w3-shape. */
    ggml_tensor* u2 = ggml_cont(ctx, ggml_permute(ctx, u1, 0, 2, 1, 3));
    /* u2 ne = (dim*W*W, n_hw, n_ww, 1) */

    /* Inverse step C: re-split dim*W*W and merge n_hw with the hp axis →
     * (dim, W, hp, n_ww) = w2-shape. */
    ggml_tensor* u3 = ggml_reshape_4d(ctx, u2, dim, W, hp, n_ww);

    /* Inverse step D: permute(0,1,3,2) is self-inverse → w1-shape. */
    ggml_tensor* u4 = ggml_cont(ctx, ggml_permute(ctx, u3, 0, 1, 3, 2));
    /* u4 ne = (dim, W, n_ww, hp) */

    /* Inverse step E: merge (W, n_ww) into wp → grid-shape (dim, wp, hp). */
    ggml_tensor* u5 = ggml_reshape_3d(ctx, u4, dim, wp, hp);

    /* Inverse step F: flatten (wp, hp) → N. Memory layout of u5 is exactly the
     * patches layout, so a 2d reshape (no permute) suffices. */
    ggml_tensor* patches_out = ggml_reshape_2d(ctx, u5, dim, N);

    /* Re-prepend CLS along axis 1: (dim, 1) ⊕ (dim, N) → (dim, N+1) */
    ggml_tensor* full = ggml_concat(ctx, cls, patches_out, /*dim*/ 1);
    return full;
}

/* MLP: x -> fc1 -> GELU -> fc2.
 *
 * PyTorch's nn.GELU() default uses the erf-based exact formula, so we use
 * ggml_gelu_erf (the plain ggml_gelu uses the tanh-approx, which would
 * introduce a small numerical mismatch).
 *
 * fc1: W1 ne = (dim, ffn_dim); fc1(x) -> ne = (ffn_dim, N)
 * fc2: W2 ne = (ffn_dim, dim); fc2(h) -> ne = (dim, N) */
ggml_tensor* mlp(ggml_context* ctx, ggml_tensor* x,
                 ggml_tensor* W1, ggml_tensor* b1,
                 ggml_tensor* W2, ggml_tensor* b2) {
    ggml_tensor* h = ggml_mul_mat(ctx, W1, x);
    h = ggml_add(ctx, h, b1);
    h = ggml_gelu_erf(ctx, h);
    h = ggml_mul_mat(ctx, W2, h);
    h = ggml_add(ctx, h, b2);
    return h;
}

}  // namespace

bool is_global_block(const Config& cfg, uint32_t i) {
    for (uint32_t v : cfg.backbone.multi_scale_layers) {
        if (v == i) return true;
    }
    return false;
}

ggml_tensor* dinov2_block(ggml_context* ctx, const Model& m,
                          ggml_tensor* x, int block_idx) {
    /* GGUF tensor names use PyTorch's "blocks.N" (plural). */
    const std::string p = "backbone.blocks." + std::to_string(block_idx) + ".";
    auto get = [&](const char* suffix) -> ggml_tensor* {
        auto it = m.tensors.find(p + suffix);
        if (it == m.tensors.end()) {
            rfdetr_logf(RFDETR_LOG_ERROR, "dinov2_block: missing tensor '%s'",
                        (p + suffix).c_str());
            return nullptr;
        }
        return it->second;
    };

    ggml_tensor* n1w  = get("norm1.weight");
    ggml_tensor* n1b  = get("norm1.bias");
    ggml_tensor* qkvW = get("attn.qkv.weight");
    ggml_tensor* qkvB = get("attn.qkv.bias");
    ggml_tensor* prW  = get("attn.proj.weight");
    ggml_tensor* prB  = get("attn.proj.bias");
    ggml_tensor* n2w  = get("norm2.weight");
    ggml_tensor* n2b  = get("norm2.bias");
    ggml_tensor* f1W  = get("mlp.fc1.weight");
    ggml_tensor* f1B  = get("mlp.fc1.bias");
    ggml_tensor* f2W  = get("mlp.fc2.weight");
    ggml_tensor* f2B  = get("mlp.fc2.bias");
    if (!n1w || !n1b || !qkvW || !qkvB || !prW || !prB ||
        !n2w || !n2b || !f1W || !f1B || !f2W || !f2B) {
        return nullptr;
    }

    /* Publish names use "block.N" (singular) to match docs/parity.md and the
     * numpy reference. */
    const std::string pub = "backbone.block." + std::to_string(block_idx) + ".";

    /* x = x + attn(norm1(x)) */
    ggml_tensor* y = layer_norm(ctx, x, n1w, n1b);
    publish(pub + "norm1.output", y);
    /* Per rfdetr convention, multi_scale_layers indices == global-attention blocks. */
    if (is_global_block(m.config, (uint32_t)block_idx)) {
        y = mha(ctx, y, qkvW, qkvB, prW, prB, (int)m.config.backbone.heads);
    } else {
        const int hp = (int)(m.config.image_size / kPatchSize);
        const int wp = (int)(m.config.image_size / kPatchSize);
        y = mha_window(ctx, y, qkvW, qkvB, prW, prB,
                       (int)m.config.backbone.heads,
                       (int)m.config.backbone.window_size, hp, wp);
    }
    publish(pub + "attn.output", y);
    x = ggml_add(ctx, x, y);

    /* x = x + mlp(norm2(x)) */
    y = layer_norm(ctx, x, n2w, n2b);
    ggml_tensor* mlp_out = mlp(ctx, y, f1W, f1B, f2W, f2B);
    publish(pub + "mlp.output", mlp_out);
    x = ggml_add(ctx, x, mlp_out);

    publish(pub + "output", x);
    return x;
}

ggml_tensor* dinov2_final_norm(ggml_context* ctx, const Model& m,
                               ggml_tensor* x) {
    auto it_w = m.tensors.find("backbone.norm.weight");
    auto it_b = m.tensors.find("backbone.norm.bias");
    if (it_w == m.tensors.end() || it_b == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "dinov2_final_norm: missing backbone.norm");
        return nullptr;
    }
    constexpr float eps = 1e-5f;
    ggml_tensor* y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, it_w->second);
    y = ggml_add(ctx, y, it_b->second);
    publish("backbone.norm.output", y);
    return y;
}

ggml_tensor* dinov2_forward(ggml_context* ctx, const Model& m,
                            ggml_tensor* input) {
    ggml_tensor* t = dinov2_patch_embed(ctx, m, input);
    if (!t) return nullptr;

    t = dinov2_add_cls_and_pos_embed(ctx, m, t);
    if (!t) return nullptr;

    const auto& ms = m.config.backbone.multi_scale_layers;
    auto find_ms_level = [&](uint32_t block_i) -> int {
        for (size_t k = 0; k < ms.size(); ++k) {
            if (ms[k] == block_i) return (int)k;
        }
        return -1;
    };

    for (uint32_t i = 0; i < m.config.backbone.depth; ++i) {
        t = dinov2_block(ctx, m, t, (int)i);
        if (!t) return nullptr;
        int level = find_ms_level(i);
        if (level >= 0) {
            publish("backbone.multiscale.level" + std::to_string(level), t);
        }
    }

    t = dinov2_final_norm(ctx, m, t);
    return t;
}

}  // namespace rfdetr
