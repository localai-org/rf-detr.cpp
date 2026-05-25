#include "dinov2.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <cmath>
#include <string>

namespace rfdetr {

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

/* dinov2_block lands in Task 9. */

}  // namespace rfdetr
