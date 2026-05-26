#include "rfdetr_model.hpp"
#include "dinov2.hpp"
#include "projector.hpp"
#include "two_stage.hpp"
#include "decoder.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

namespace rfdetr {

/* Plan 11 transitional state: backbone + projector + two_stage + decoder are
 * v2-schema; heads are still v1 and the top-K + decoder-input plumbing isn't
 * wired in graph yet. This whole-pipeline glue is being rebuilt in Plan 12.
 *
 * For now `rfdetr_model_forward` runs backbone+projector only and returns
 * an empty output. Tests use the per-module forwards directly. */
ForwardOutput rfdetr_model_forward(ggml_context* ctx, const Model& m,
                                   ggml_tensor* input) {
    ForwardOutput out;

    BackboneOutput bb = dinov2_forward(ctx, m, input);
    if (!bb.final) {
        rfdetr_logf(RFDETR_LOG_ERROR, "rfdetr_model_forward: backbone failed");
        return out;
    }

    ggml_tensor* projected = projector_forward(ctx, m, bb);
    if (!projected) return out;

    /* TODO Plan 12: two_stage top-K → decoder → heads → out.class_logits, out.bbox_pred. */
    (void)ctx;
    return out;
}

}  // namespace rfdetr
