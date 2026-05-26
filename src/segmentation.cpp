#include "segmentation.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <cmath>
#include <string>

namespace rfdetr {

namespace {

constexpr float kLnEps = 1e-6f;

ggml_tensor* fetch(const Model& m, const std::string& name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "segmentation: missing tensor '%s'", name.c_str());
        return nullptr;
    }
    return it->second;
}

/* Channel-axis LayerNorm for a spatial tensor (mirrors projector helper).
 *
 *   x ne = (W, H, C, B)  →  norm over C  →  ne = (W, H, C, B).
 *
 * ggml_norm normalizes over ne[0], so we permute to bring C to ne[0]. */
ggml_tensor* channel_layer_norm(ggml_context* ctx, ggml_tensor* x,
                                ggml_tensor* weight, ggml_tensor* bias) {
    /* (W, H, C, B) → (C, W, H, B). */
    ggml_tensor* y = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
    y = ggml_norm(ctx, y, kLnEps);
    y = ggml_mul(ctx, y, weight);
    y = ggml_add(ctx, y, bias);
    /* (C, W, H, B) → (W, H, C, B). */
    y = ggml_cont(ctx, ggml_permute(ctx, y, 2, 0, 1, 3));
    return y;
}

/* DepthwiseConvBlock forward (simplified ConvNeXt block).
 *
 *   x  ne = (W, H, C, B)
 *   dwconv weight ne = (KW=3, KH=3, 1, C)   — ggml depthwise-conv layout
 *   dwconv bias   ne = (C,)
 *   norm   weight ne = (C,)
 *   norm   bias   ne = (C,)
 *   pwconv1 weight ne = (C_in=256, C_out=256)
 *   pwconv1 bias  ne = (C,)
 *
 * PyTorch reference (DepthwiseConvBlock.forward):
 *   input = x
 *   x = self._depthwise_conv(x)                  # (N, C, H, W)
 *   x = x.permute(0, 2, 3, 1)                    # (N, H, W, C)
 *   x = self.norm(x)                             # LN over C
 *   x = self.pwconv1(x)                          # Linear C→C
 *   x = self.act(x)                              # GELU
 *   x = x.permute(0, 3, 1, 2)                    # (N, C, H, W)
 *   return x + input
 *
 * In ggml memory layout x is already (W, H, C, B), i.e. dim-fastest-on-W;
 * channel layer-norm operates on the C axis (axis 2) via the helper above,
 * and pwconv1 is a Linear that operates on C — we apply it as a mul_mat after
 * permuting C to ne[0]. */
ggml_tensor* depthwise_conv_block(ggml_context* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* dw_w, ggml_tensor* dw_b,
                                  ggml_tensor* n_w,  ggml_tensor* n_b,
                                  ggml_tensor* pw_w, ggml_tensor* pw_b) {
    ggml_tensor* shortcut = x;

    /* 1. Depthwise 3x3 conv (stride 1, padding 1).
     *    Kernel ne = (3, 3, 1, C), input ne = (W, H, C, B). */
    ggml_tensor* y = ggml_conv_2d_dw(ctx, dw_w, x,
                                     /*s0*/ 1, /*s1*/ 1,
                                     /*p0*/ 1, /*p1*/ 1,
                                     /*d0*/ 1, /*d1*/ 1);
    /* y ne = (W, H, C, B). Add per-channel bias via broadcast over (W, H). */
    {
        ggml_tensor* b_r = ggml_reshape_3d(ctx, dw_b, 1, 1, dw_b->ne[0]);
        y = ggml_add(ctx, y, b_r);
    }

    /* 2. LayerNorm over channel axis with affine (norm_w, norm_b). */
    y = channel_layer_norm(ctx, y, n_w, n_b);

    /* 3. Pointwise Linear (256 → 256) applied to channels.
     *    pwconv1.weight ne = (256, 256) — same axis ordering as a torch
     *    nn.Linear (in=256, out=256). For mul_mat we need C-fastest on the
     *    input, so permute to (C, W, H, B), mul_mat (which contracts axis 0),
     *    add bias (C,), then permute back to (W, H, C, B). */
    ggml_tensor* yc = ggml_cont(ctx, ggml_permute(ctx, y, 1, 2, 0, 3));
    /* yc ne = (C, W, H, B). */
    ggml_tensor* lin = ggml_mul_mat(ctx, pw_w, yc);
    lin = ggml_add(ctx, lin, pw_b);
    /* lin ne = (C, W, H, B). */

    /* 4. GELU (PyTorch uses nn.GELU which is the exact erf-based GELU). */
    lin = ggml_gelu_erf(ctx, lin);

    /* 5. Permute back to (W, H, C, B). */
    lin = ggml_cont(ctx, ggml_permute(ctx, lin, 2, 0, 1, 3));

    /* 6. Residual. */
    return ggml_add(ctx, lin, shortcut);
}

/* MLPBlock forward:
 *
 *   input = x
 *   x = norm_in(x)                       # LN over C
 *   x = layers.0(x)  = Linear(C, 4C)
 *   x = GELU
 *   x = layers.2(x)  = Linear(4C, C)
 *   return x + input
 *
 * Input ne = (C, NQ, 1)  (channel-first as decoder output is). The norm and
 * linears all operate on ne[0]=C directly — no permute needed. */
ggml_tensor* mlp_block(ggml_context* ctx, ggml_tensor* x,
                       ggml_tensor* n_w, ggml_tensor* n_b,
                       ggml_tensor* l0_w, ggml_tensor* l0_b,
                       ggml_tensor* l2_w, ggml_tensor* l2_b) {
    ggml_tensor* shortcut = x;
    ggml_tensor* y = ggml_norm(ctx, x, kLnEps);
    y = ggml_mul(ctx, y, n_w);
    y = ggml_add(ctx, y, n_b);
    /* Linear 256 → 1024 */
    y = ggml_mul_mat(ctx, l0_w, y);
    y = ggml_add(ctx, y, l0_b);
    y = ggml_gelu_erf(ctx, y);
    /* Linear 1024 → 256 */
    y = ggml_mul_mat(ctx, l2_w, y);
    y = ggml_add(ctx, y, l2_b);
    return ggml_add(ctx, y, shortcut);
}

/* spatial_features_proj is a Conv2d(C, C, kernel_size=1). For seg-nano the
 * bottleneck_ratio is 1 so out_channels == in_channels == 256. We apply it as
 * a per-pixel Linear because the kernel is 1x1 — that gives the same
 * arithmetic but is cheaper than ggml_conv_2d (and avoids the F32-only path).
 *
 *   weight ne = (1, 1, 256, 256)  →  squeezed view of shape (256, 256)
 *   x ne = (W, H, C, B)           →  Linear over C → (W, H, C', B)
 */
ggml_tensor* spatial_features_proj_1x1(ggml_context* ctx, ggml_tensor* x,
                                       ggml_tensor* w, ggml_tensor* b) {
    /* Build a 2D view of the 1x1 conv kernel.
     *   torch shape: (out=256, in=256, kh=1, kw=1)
     *   ggml ne   : (kw=1, kh=1, in=256, out=256)
     * Reshape to (in, out) — i.e. (256, 256). */
    ggml_tensor* w2d = ggml_reshape_2d(ctx, w, w->ne[2], w->ne[3]);

    /* Permute x to put C on ne[0]: (W, H, C, B) → (C, W, H, B). */
    ggml_tensor* xc = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
    ggml_tensor* y = ggml_mul_mat(ctx, w2d, xc);
    y = ggml_add(ctx, y, b);
    /* Permute back: (C', W, H, B) → (W, H, C', B). */
    return ggml_cont(ctx, ggml_permute(ctx, y, 2, 0, 1, 3));
}

}  // namespace

ggml_tensor* segmentation_forward(
    ggml_context* ctx,
    const Model& m,
    ggml_tensor* spatial_features,
    ggml_tensor* const* query_features_per_layer,
    int n_layers,
    int image_h, int image_w,
    int mask_downsample_ratio) {
    if (!spatial_features || !query_features_per_layer || n_layers <= 0 ||
        mask_downsample_ratio <= 0) {
        rfdetr_logf(RFDETR_LOG_ERROR, "segmentation_forward: invalid args");
        return nullptr;
    }
    if (n_layers != 4) {
        rfdetr_logf(RFDETR_LOG_WARN,
                    "segmentation_forward: n_layers=%d, but SegmentationHead "
                    "has exactly 4 blocks. Will iterate min(n_layers, 4).",
                    n_layers);
    }

    const int target_h = image_h / mask_downsample_ratio;
    const int target_w = image_w / mask_downsample_ratio;

    /* 1. Bilinear resize spatial_features (W_in, H_in, C, 1) →
     *    (target_w, target_h, C, 1). PyTorch passes `size=(target_h,
     *    target_w)` with mode="bilinear", align_corners=False — matches
     *    ggml_interpolate(GGML_SCALE_MODE_BILINEAR) when ALIGN_CORNERS is
     *    NOT set (default for F.interpolate). */
    ggml_tensor* spatial = ggml_interpolate(
        ctx, spatial_features,
        /*ne0*/ (int64_t)target_w,
        /*ne1*/ (int64_t)target_h,
        /*ne2*/ spatial_features->ne[2],
        /*ne3*/ spatial_features->ne[3],
        (uint32_t)GGML_SCALE_MODE_BILINEAR);
    publish("seg.spatial_features.resized", spatial);

    /* 2. Fetch all per-block weights up-front so we fail fast if any are
     *    missing. */
    struct BlockW {
        ggml_tensor *dw_w, *dw_b, *n_w, *n_b, *pw_w, *pw_b;
    } bw[4];
    for (int b = 0; b < 4; ++b) {
        const std::string p = "segmentation_head.blocks." + std::to_string(b) + ".";
        bw[b].dw_w = fetch(m, p + "dwconv.weight");
        bw[b].dw_b = fetch(m, p + "dwconv.bias");
        bw[b].n_w  = fetch(m, p + "norm.weight");
        bw[b].n_b  = fetch(m, p + "norm.bias");
        bw[b].pw_w = fetch(m, p + "pwconv1.weight");
        bw[b].pw_b = fetch(m, p + "pwconv1.bias");
        if (!bw[b].dw_w || !bw[b].dw_b || !bw[b].n_w || !bw[b].n_b ||
            !bw[b].pw_w || !bw[b].pw_b) {
            return nullptr;
        }
    }
    ggml_tensor* sf_proj_w = fetch(m, "segmentation_head.spatial_features_proj.weight");
    ggml_tensor* sf_proj_b = fetch(m, "segmentation_head.spatial_features_proj.bias");
    ggml_tensor* qf_n_w = fetch(m, "segmentation_head.query_features_block.norm_in.weight");
    ggml_tensor* qf_n_b = fetch(m, "segmentation_head.query_features_block.norm_in.bias");
    ggml_tensor* qf_l0_w = fetch(m, "segmentation_head.query_features_block.layers.0.weight");
    ggml_tensor* qf_l0_b = fetch(m, "segmentation_head.query_features_block.layers.0.bias");
    ggml_tensor* qf_l2_w = fetch(m, "segmentation_head.query_features_block.layers.2.weight");
    ggml_tensor* qf_l2_b = fetch(m, "segmentation_head.query_features_block.layers.2.bias");
    ggml_tensor* qf_proj_w = fetch(m, "segmentation_head.query_features_proj.weight");
    ggml_tensor* qf_proj_b = fetch(m, "segmentation_head.query_features_proj.bias");
    ggml_tensor* seg_bias = fetch(m, "segmentation_head.bias");
    if (!sf_proj_w || !sf_proj_b || !qf_n_w || !qf_n_b || !qf_l0_w || !qf_l0_b ||
        !qf_l2_w || !qf_l2_b || !qf_proj_w || !qf_proj_b || !seg_bias) {
        return nullptr;
    }

    /* 3. Iterate 4 blocks. The number of query streams is N (always 4 in
     *    practice). If fewer were provided we cap iteration. */
    const int n_iter = (n_layers < 4) ? n_layers : 4;
    ggml_tensor* masks_final = nullptr;
    for (int b = 0; b < n_iter; ++b) {
        spatial = depthwise_conv_block(ctx, spatial,
                                       bw[b].dw_w, bw[b].dw_b,
                                       bw[b].n_w,  bw[b].n_b,
                                       bw[b].pw_w, bw[b].pw_b);
        publish("seg.block." + std::to_string(b) + ".spatial_out", spatial);

        ggml_tensor* spatial_proj = spatial_features_proj_1x1(
            ctx, spatial, sf_proj_w, sf_proj_b);
        publish("seg.block." + std::to_string(b) + ".spatial_proj", spatial_proj);

        /* qf forward: query_features_block (MLP) then query_features_proj
         * (Linear 256→256). qf ne = (C, NQ, 1). */
        ggml_tensor* qf = query_features_per_layer[b];
        if (!qf) {
            rfdetr_logf(RFDETR_LOG_ERROR,
                        "segmentation_forward: null query_features[%d]", b);
            return nullptr;
        }
        ggml_tensor* qfm = mlp_block(ctx, qf,
                                     qf_n_w, qf_n_b,
                                     qf_l0_w, qf_l0_b,
                                     qf_l2_w, qf_l2_b);
        ggml_tensor* qf_proj = ggml_mul_mat(ctx, qf_proj_w, qfm);
        qf_proj = ggml_add(ctx, qf_proj, qf_proj_b);
        publish("seg.block." + std::to_string(b) + ".qf_proj", qf_proj);
        /* qf_proj ne = (C=256, NQ, 1). */

        /* 4. einsum("bchw,bnc->bnhw", spatial_proj, qf_proj) + bias.
         *
         * spatial_proj ne = (W, H, C, 1). We want to contract C.
         * Reshape spatial_proj to (W*H, C) by:
         *   permute (W,H,C,1) → (C,W,H,1) → reshape (C, W*H, 1)
         * Then matmul with qf_proj (C, NQ, 1):
         *   ggml_mul_mat(A, B) computes B^T A in numpy terms, treating
         *   ne[0] as the contracted axis. Both A and B have ne[0]=C, so
         *   the result is (NQ, W*H, 1) ... wait, let me get this right.
         *
         * ggml_mul_mat semantics: out[i, j] = sum_k a[k, i] * b[k, j]
         *   a ne = (ne00, ne01, ...)
         *   b ne = (ne10, ne11, ...) with ne10 == ne00 (contracted)
         *   out ne = (ne01, ne11, ...)
         *
         * Set a = spatial_proj_flat ne = (C, W*H, 1), and
         *     b = qf_proj ne = (C, NQ, 1).
         * out ne = (W*H, NQ, 1).
         *
         * Then reshape to (W, H, NQ, 1) — that's the bnhw layout in ggml. */
        const int64_t W = spatial_proj->ne[0];
        const int64_t H = spatial_proj->ne[1];
        const int64_t C = spatial_proj->ne[2];
        /* (W, H, C, 1) → (C, W, H, 1) → (C, W*H, 1). */
        ggml_tensor* sp_perm = ggml_cont(ctx, ggml_permute(ctx, spatial_proj, 1, 2, 0, 3));
        ggml_tensor* sp_flat = ggml_reshape_3d(ctx, sp_perm, C, W * H, 1);
        ggml_tensor* mask = ggml_mul_mat(ctx, sp_flat, qf_proj);
        /* mask ne = (W*H, NQ, 1). Add scalar bias (broadcast). */
        mask = ggml_add(ctx, mask, seg_bias);
        /* Reshape to (W, H, NQ, 1). */
        mask = ggml_reshape_4d(ctx, mask, W, H, qf_proj->ne[1], 1);
        publish("seg.masks." + std::to_string(b), mask);

        if (b == n_iter - 1) {
            masks_final = mask;
        }
    }

    publish("seg.masks.final", masks_final);
    return masks_final;
}

}  // namespace rfdetr
