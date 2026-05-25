#ifndef RFDETR_PROJECTOR_HPP
#define RFDETR_PROJECTOR_HPP

#include "dinov2.hpp"
#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace rfdetr {

/* Multi-scale projector:
 *   For each level j in 0..3:
 *     1. Strip CLS from backbone multi-scale feature j (keep patches only)
 *     2. Linear project: y = Wj @ patches + bj
 *        Wj ne = (backbone.dim, encoder.model_dim) in ggml convention
 *     3. Add level embedding: y += level_embed[j]
 *   Concatenate the 4 projected sequences along the token axis.
 *
 * Output ne = (encoder.model_dim, 4 * N_patches, 1, 1)
 *
 * Publishes:
 *   projector.level{0..3}.output  — per-level pre-concat tensor (model_dim, N_patches)
 *   projector.concat.output       — final concatenated tensor */
ggml_tensor* projector_forward(ggml_context* ctx, const Model& m,
                               const BackboneOutput& bb);

}  // namespace rfdetr

#endif
