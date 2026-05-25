#include "transformer_ops.hpp"

#include "ggml.h"

#include <cmath>

namespace rfdetr::ops {

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

}  // namespace rfdetr::ops
