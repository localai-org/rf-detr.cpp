#ifndef RFDETR_HEADS_HPP
#define RFDETR_HEADS_HPP

#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace rfdetr {

/* Shared class_embed head: Linear(model_dim=256, num_classes=91).
 *
 * In rfdetr-base the same `class_embed` is applied to every decoder layer
 * output (training uses per-layer auxiliary losses; inference uses the last
 * layer only). This forward applies it to a single (256, num_queries, 1)
 * input from the final decoder.norm.
 *
 *   Input  ne = (256, NQ, 1)
 *   Output ne = (num_classes, NQ, 1) — raw logits (apply sigmoid in postproc)
 *
 * Publishes "heads.class_logits". */
ggml_tensor* class_head_forward(ggml_context* ctx, const Model& m,
                                ggml_tensor* decoder_out);

/* Shared bbox_embed head: 3-layer MLP (256 → 256 → 256 → 4) with ReLU between
 * layers, NO final activation. Output is a delta in (dcx, dcy, dlogw, dlogh)
 * space — the bbox_reparam formula combines it with the reference points to
 * yield the final (cx, cy, w, h) in [0, 1]. (Per rfdetr/models/lwdetr.py:230.)
 *
 *   Input  ne = (256, NQ, 1)
 *   Output ne = (4, NQ, 1)            — raw delta (NOT post-sigmoid)
 *
 * Publishes "heads.bbox_pred". */
ggml_tensor* bbox_head_forward(ggml_context* ctx, const Model& m,
                               ggml_tensor* decoder_out);

}  // namespace rfdetr

#endif
