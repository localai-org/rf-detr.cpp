#ifndef RFDETR_HEADS_HPP
#define RFDETR_HEADS_HPP

#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace rfdetr {

/* Class head: linear projection from per-query embedding to per-class logits.
 *
 *   Input  ne = (model_dim, num_queries, 1, 1)
 *   Output ne = (num_classes, num_queries, 1, 1)
 *
 * No activation — postprocess applies sigmoid + top-k + threshold.
 *
 * Publishes "heads.class.logits". */
ggml_tensor* class_head_forward(ggml_context* ctx, const Model& m,
                                ggml_tensor* decoder_out);

/* Bbox head: 3-layer MLP with ReLU between layers and sigmoid at the end.
 *
 *   fc1: model_dim → model_dim, ReLU
 *   fc2: model_dim → model_dim, ReLU
 *   fc3: model_dim → 4 (cx, cy, w, h)
 *   sigmoid → ne = (4, num_queries, 1, 1) in [0, 1]
 *
 * Publishes "heads.bbox.fc1.output", "heads.bbox.fc2.output",
 * "heads.bbox.fc3.output", "heads.bbox.pred". */
ggml_tensor* bbox_head_forward(ggml_context* ctx, const Model& m,
                               ggml_tensor* decoder_out);

}  // namespace rfdetr

#endif
