#ifndef RFDETR_MODEL_HPP
#define RFDETR_MODEL_HPP

#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace rfdetr {

/* Result of the full forward pass.
 *
 *   class_logits ne = (num_classes, num_queries, 1, 1) — raw, pre-sigmoid
 *   bbox_pred    ne = (4, num_queries, 1, 1)            — post-sigmoid, in [0, 1] */
struct ForwardOutput {
    ggml_tensor* class_logits = nullptr;
    ggml_tensor* bbox_pred    = nullptr;
};

/* Run the full forward pipeline: backbone → projector → encoder → decoder → heads.
 *
 *   input ne = (W, H, 3, 1) F32 — already mean/std normalized
 *
 * Publishes every named checkpoint from the sub-pipelines plus
 * "model.class_logits" and "model.bbox_pred" wrappers. */
ForwardOutput rfdetr_model_forward(ggml_context* ctx, const Model& m,
                                   ggml_tensor* input);

}  // namespace rfdetr

#endif
