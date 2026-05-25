#include "rfdetr_model.hpp"
#include "dinov2.hpp"
#include "projector.hpp"
#include "encoder.hpp"
#include "decoder.hpp"
#include "heads.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

namespace rfdetr {

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

    ggml_tensor* enc = encoder_forward(ctx, m, projected);
    if (!enc) return out;

    ggml_tensor* dec = decoder_forward(ctx, m, enc);
    if (!dec) return out;

    out.class_logits = class_head_forward(ctx, m, dec);
    out.bbox_pred    = bbox_head_forward(ctx, m, dec);

    if (out.class_logits) publish("model.class_logits", out.class_logits);
    if (out.bbox_pred)    publish("model.bbox_pred",    out.bbox_pred);

    return out;
}

}  // namespace rfdetr
