#include "projector.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <string>

namespace rfdetr {

ggml_tensor* projector_forward(ggml_context* ctx, const Model& m,
                               const BackboneOutput& bb) {
    /* TODO Plan 10: rewrite projector against v2 schema (single-scale C2f).
     * For now use out_feature_indices.size() — same indices, renamed field. */
    const int n_levels = (int)m.config.backbone.out_feature_indices.size();
    if (n_levels <= 0 || n_levels > (int)bb.multi_scale.size()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "projector_forward: invalid n_levels %d", n_levels);
        return nullptr;
    }

    auto get_tensor = [&](const std::string& name) -> ggml_tensor* {
        auto it = m.tensors.find(name);
        if (it == m.tensors.end()) {
            rfdetr_logf(RFDETR_LOG_ERROR, "projector_forward: missing tensor '%s'", name.c_str());
            return nullptr;
        }
        return it->second;
    };

    ggml_tensor* level_embed = get_tensor("projector.level_embed");
    if (!level_embed) return nullptr;
    /* level_embed ne in the fixture: (encoder.model_dim, n_levels) — one
     * encoder.model_dim vector per level, columns are levels. */

    ggml_tensor* projected[4] = {nullptr, nullptr, nullptr, nullptr};

    for (int j = 0; j < n_levels; ++j) {
        ggml_tensor* feat = bb.multi_scale[j];
        if (!feat) {
            rfdetr_logf(RFDETR_LOG_ERROR, "projector_forward: missing multi_scale[%d]", j);
            return nullptr;
        }
        /* feat ne = (backbone.dim, N+1, 1, 1). Strip CLS (axis 1, position 0). */
        const int dim = (int)feat->ne[0];
        const int N1  = (int)feat->ne[1];
        const int N   = N1 - 1;
        ggml_tensor* patches = ggml_view_2d(ctx, feat, dim, N,
                                            feat->nb[1],
                                            /* offset bytes */ feat->nb[1]);
        patches = ggml_cont(ctx, patches);
        /* patches ne = (backbone.dim, N) */

        /* Linear projection: Wj @ patches + bj
         * Wj ne = (backbone.dim, encoder.model_dim) in ggml convention
         * patches ne = (backbone.dim, N)
         * mul_mat result ne = (encoder.model_dim, N) */
        std::string p = "projector.level" + std::to_string(j) + ".";
        ggml_tensor* W = get_tensor(p + "weight");
        ggml_tensor* b = get_tensor(p + "bias");
        if (!W || !b) return nullptr;

        ggml_tensor* y = ggml_mul_mat(ctx, W, patches);
        y = ggml_add(ctx, y, b);
        /* y ne = (encoder.model_dim, N) */

        /* Add per-level embedding: select column j of level_embed and add to every
         * token in y. level_embed ne = (model_dim, n_levels). Column j is the
         * vector for level j. */
        const int model_dim = (int)y->ne[0];
        ggml_tensor* le_j = ggml_view_2d(ctx, level_embed,
                                         model_dim, 1,
                                         level_embed->nb[1],
                                         /* offset */ (size_t)j * level_embed->nb[1]);
        le_j = ggml_cont(ctx, le_j);
        /* le_j ne = (model_dim, 1) — broadcasts over (model_dim, N) via ggml_add. */
        y = ggml_add(ctx, y, le_j);

        publish("projector.level" + std::to_string(j) + ".output", y);
        projected[j] = y;
    }

    /* Concat all 4 levels along axis 1: each (model_dim, N) → (model_dim, 4*N). */
    ggml_tensor* out = projected[0];
    for (int j = 1; j < n_levels; ++j) {
        out = ggml_concat(ctx, out, projected[j], /*dim*/ 1);
    }

    publish("projector.concat.output", out);
    return out;
}

}  // namespace rfdetr
