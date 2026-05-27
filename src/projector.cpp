#include "projector.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <string>

namespace rfdetr {

namespace {

/* Channel-axis LayerNorm for a spatial tensor.
 *
 * Input  ne = (W, H, C, B)  (ggml dim-fastest order — channels at axis 2)
 * Output ne = (W, H, C, B), same layout.
 *
 * Mirrors rfdetr's `LayerNorm` in projector.py: permute(0,2,3,1) to put C last
 * (or first in ggml), F.layer_norm over C with weight + bias (C,), then permute
 * back. ggml_norm normalizes along ne[0], so we bring C to ne[0] for the norm
 * call and restore the original axis order after. */
constexpr float kLnEps = 1e-6f;

ggml_tensor* channel_layer_norm(ggml_context* ctx, ggml_tensor* x,
                                ggml_tensor* weight, ggml_tensor* bias) {
    /* (W, H, C, B) → (C, W, H, B). Input axis 0 (W) → out pos 1, axis 1 (H)
     * → out pos 2, axis 2 (C) → out pos 0, axis 3 (B) → out pos 3. */
    ggml_tensor* y = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
    y = ggml_norm(ctx, y, kLnEps);
    y = ggml_mul(ctx, y, weight);
    y = ggml_add(ctx, y, bias);
    /* (C, W, H, B) → (W, H, C, B). Inverse permute: input axis 0 (C) → out
     * pos 2, axis 1 (W) → out pos 0, axis 2 (H) → out pos 1. */
    y = ggml_cont(ctx, ggml_permute(ctx, y, 2, 0, 1, 3));
    return y;
}

/* ConvX = Conv2d (bias-free) → ChannelLN → SiLU.
 *
 *   x      ne = (W, H, IC, B)
 *   kernel ne = (KW, KH, IC, OC)
 *   norm_w/b ne = (OC,)
 *
 * stride 1, padding = kernel/2 (so output spatial = input spatial when k is
 * 1 or 3, which is all the ConvX uses we need for the projector). */
ggml_tensor* convx(ggml_context* ctx, ggml_tensor* x, ggml_tensor* kernel,
                   ggml_tensor* norm_w, ggml_tensor* norm_b) {
    const int kw = (int)kernel->ne[0];
    const int kh = (int)kernel->ne[1];
    const int p0 = kw / 2;
    const int p1 = kh / 2;
    ggml_tensor* y = ggml_conv_2d(ctx, kernel, x,
                                  /*s0*/ 1, /*s1*/ 1,
                                  /*p0*/ p0, /*p1*/ p1,
                                  /*d0*/ 1, /*d1*/ 1);
    y = channel_layer_norm(ctx, y, norm_w, norm_b);
    y = ggml_silu(ctx, y);
    return y;
}

/* Bottleneck: x → cv1 → cv2 (no residual).
 *
 * Real rfdetr instantiates C2f WITHOUT passing `shortcut=True`, so C2f's
 * default `shortcut=False` propagates into each Bottleneck. `Bottleneck.add`
 * is `shortcut and c1 == c2` = False, so forward returns plain cv2(cv1(x)). */
ggml_tensor* bottleneck(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* cv1_k, ggml_tensor* cv1_nw, ggml_tensor* cv1_nb,
                        ggml_tensor* cv2_k, ggml_tensor* cv2_nw, ggml_tensor* cv2_nb) {
    ggml_tensor* y = convx(ctx, x, cv1_k, cv1_nw, cv1_nb);
    y = convx(ctx, y, cv2_k, cv2_nw, cv2_nb);
    return y;
}

ggml_tensor* fetch(const Model& m, const std::string& name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "projector: missing tensor '%s'", name.c_str());
        return nullptr;
    }
    return it->second;
}

}  // namespace

ggml_tensor* projector_forward(ggml_context* ctx, const Model& m,
                               const BackboneOutput& bb) {
    const auto& pcfg = m.config.projector;
    const int n_levels = (int)m.config.backbone.out_feature_indices.size();
    if (n_levels <= 0 || n_levels > (int)bb.multi_scale.size()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "projector: invalid n_levels %d", n_levels);
        return nullptr;
    }
    for (int j = 0; j < n_levels; ++j) {
        if (!bb.multi_scale[j]) {
            rfdetr_logf(RFDETR_LOG_ERROR, "projector: missing multi_scale[%d]", j);
            return nullptr;
        }
    }

    /* 1. Sampling stage (single P4: identity per level) + channel concat.
     *
     * Each multiscale tensor: ne = (W, H, C_in, 1). After concat along axis 2,
     * ne = (W, H, n_levels * C_in, 1) = (40, 40, 1536, 1). */
    ggml_tensor* x = bb.multi_scale[0];
    for (int j = 1; j < n_levels; ++j) {
        x = ggml_concat(ctx, x, bb.multi_scale[j], /*dim*/ 2);
    }

    /* 2. C2f.cv1: ConvX(1536, 2*c, k=1). 2*c = out_dim (since c = out_dim/2). */
    ggml_tensor* cv1_k  = fetch(m, "projector.cv1.conv.weight");
    ggml_tensor* cv1_nw = fetch(m, "projector.cv1.norm.weight");
    ggml_tensor* cv1_nb = fetch(m, "projector.cv1.norm.bias");
    if (!cv1_k || !cv1_nw || !cv1_nb) return nullptr;

    ggml_tensor* h = convx(ctx, x, cv1_k, cv1_nw, cv1_nb);
    /* h ne = (W, H, 2c, 1) where c = out_dim/2. */
    publish("projector.cv1.output", h);

    /* 3. Split h channel-wise into y0, y1 (each (W, H, c, 1)). The C2f code:
     *     y = list(self.cv1(x).split((self.c, self.c), 1))
     *
     * For ggml: views along ne[2] with the right byte offset. nb[2] is the
     * per-channel stride (bytes per channel slice in W*H). */
    const int c = (int)(pcfg.out_dim / 2);  // bottleneck_dim = out_dim/2 = 128
    if ((int)h->ne[2] != 2 * c) {
        rfdetr_logf(RFDETR_LOG_ERROR,
            "projector: cv1 output channels %d != 2*c %d",
            (int)h->ne[2], 2 * c);
        return nullptr;
    }
    const int64_t W = h->ne[0];
    const int64_t H = h->ne[1];
    ggml_tensor* y0 = ggml_cont(ctx, ggml_view_4d(ctx, h, W, H, c, 1,
                                                  h->nb[1], h->nb[2], h->nb[3],
                                                  /*offset bytes*/ 0));
    ggml_tensor* y1 = ggml_cont(ctx, ggml_view_4d(ctx, h, W, H, c, 1,
                                                  h->nb[1], h->nb[2], h->nb[3],
                                                  /*offset bytes*/ (size_t)c * h->nb[2]));

    /* 4. Three bottlenecks applied sequentially to y1 (the C2f extends y with
     * `[y0, y1, m_0(y1), m_1(m_0(y1)), m_2(m_1(m_0(y1)))]`). */
    ggml_tensor* m_out[3] = {nullptr, nullptr, nullptr};
    ggml_tensor* cur = y1;
    for (uint32_t j = 0; j < pcfg.n_bottlenecks; ++j) {
        const std::string p = "projector.bottleneck." + std::to_string(j) + ".";
        ggml_tensor* b_cv1_k  = fetch(m, p + "cv1.conv.weight");
        ggml_tensor* b_cv1_nw = fetch(m, p + "cv1.norm.weight");
        ggml_tensor* b_cv1_nb = fetch(m, p + "cv1.norm.bias");
        ggml_tensor* b_cv2_k  = fetch(m, p + "cv2.conv.weight");
        ggml_tensor* b_cv2_nw = fetch(m, p + "cv2.norm.weight");
        ggml_tensor* b_cv2_nb = fetch(m, p + "cv2.norm.bias");
        if (!b_cv1_k || !b_cv1_nw || !b_cv1_nb || !b_cv2_k || !b_cv2_nw || !b_cv2_nb) {
            return nullptr;
        }
        cur = bottleneck(ctx, cur, b_cv1_k, b_cv1_nw, b_cv1_nb,
                                   b_cv2_k, b_cv2_nw, b_cv2_nb);
        if ((int)j < 3) m_out[j] = cur;
        publish("projector.bottleneck." + std::to_string(j) + ".output", cur);
    }

    /* 5. Concat [y0, y1, m_out...] along channel axis → (W, H, (2+n)*c, 1). */
    ggml_tensor* cat = ggml_concat(ctx, y0, y1, /*dim*/ 2);
    for (uint32_t j = 0; j < pcfg.n_bottlenecks; ++j) {
        cat = ggml_concat(ctx, cat, m_out[j], /*dim*/ 2);
    }
    /* cat ne = (W, H, (2 + n_bottlenecks) * c, 1) = (40, 40, 640, 1). */

    /* 6. C2f.cv2: ConvX((2+n)*c, out_dim, k=1). */
    ggml_tensor* cv2_k  = fetch(m, "projector.cv2.conv.weight");
    ggml_tensor* cv2_nw = fetch(m, "projector.cv2.norm.weight");
    ggml_tensor* cv2_nb = fetch(m, "projector.cv2.norm.bias");
    if (!cv2_k || !cv2_nw || !cv2_nb) return nullptr;

    ggml_tensor* out = convx(ctx, cat, cv2_k, cv2_nw, cv2_nb);
    /* out ne = (W, H, out_dim, 1) = (40, 40, 256, 1). */
    publish("projector.cv2.output", out);

    /* 7. Final LayerNorm over channel dim. */
    ggml_tensor* fn_w = fetch(m, "projector.final_norm.weight");
    ggml_tensor* fn_b = fetch(m, "projector.final_norm.bias");
    if (!fn_w || !fn_b) return nullptr;

    out = channel_layer_norm(ctx, out, fn_w, fn_b);
    publish("projector.final_norm.output", out);
    publish("projector.output", out);

    return out;
}

}  // namespace rfdetr
