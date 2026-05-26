#ifndef RFDETR_TWO_STAGE_HPP
#define RFDETR_TWO_STAGE_HPP

#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace rfdetr {

/* Two-stage init bridge between MultiScaleProjector and the decoder.
 *
 * For rfdetr-base, only group 0 is used at inference (group_detr=13 are
 * training-only parallel proposal heads). The two-stage head:
 *   1. flatten projector output (W, H, C, 1) → tokens (C, W*H, 1)
 *   2. enc_output[0]: Linear(256, 256) over tokens
 *   3. enc_output_norm[0]: LayerNorm(256) over channel dim
 *   4. enc_out_class_embed[0]: Linear(256, 91) → cls_all (91, W*H)
 *   5. enc_out_bbox_embed[0]: 3-layer MLP (256→256→256→4) → bbox_delta_all (4, W*H)
 *   6. bbox_reparam (=True for rfdetr-base):
 *        cxcy = bbox_delta[..., :2] * proposals[..., 2:] + proposals[..., :2]
 *        wh   = exp(bbox_delta[..., 2:]) * proposals[..., 2:]
 *      → bbox_all (4, W*H), absolute (cx, cy, w, h) in [0, 1]
 *
 * Proposals are constants depending only on the spatial grid (40x40 for
 * rfdetr-base) — pre-computed at model_realize_weights and stored in
 * `Model::extras_ctx` as a (4, 1600) F32 tensor.
 *
 * Top-K selection is performed by the caller (CPU-side reduction): per-token
 * max class score → argsort descending → first num_queries indices. Since
 * `enc_out_class_embed[0]` is a per-row Linear, gathering the top-K rows from
 * `cls_all` is equivalent to applying the class head to the top-K rows of
 * `enc_output_norm_out`.
 *
 * Trace callbacks published (matching baseline parity tensor names):
 *   two_stage.enc_output_norm.output  → (256, 1600, 1) F32
 *
 * Note: `two_stage.enc_out_class.output` (91, 300, 1) in the baseline is the
 * POST-top-K class output. The caller computes it by gathering rows from
 * `cls_all` per top-K indices.
 *
 * Returns nullptr fields on failure (missing tensor / bad shape). */
struct TwoStageOutput {
    ggml_tensor* enc_output_norm_out = nullptr;  // (256, 1600, 1) — graph node
    ggml_tensor* cls_all             = nullptr;  // (91, 1600, 1)
    ggml_tensor* bbox_all            = nullptr;  // (4, 1600, 1) — reparam'd (cx,cy,w,h)
};

TwoStageOutput two_stage_forward(ggml_context* ctx, const Model& m,
                                 ggml_tensor* projector_out);

/* Compute the constant proposal grid for the configured spatial extent.
 * Writes (4, W*H) F32 in `(cx, cy, w, h)` order per row.
 *
 * Used by model_realize_weights to populate the extras_ctx slot. Exposed
 * for tests. */
void compute_proposal_grid(int width, int height, float wh_value, float* out);

}  // namespace rfdetr

#endif
