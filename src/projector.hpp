#ifndef RFDETR_PROJECTOR_HPP
#define RFDETR_PROJECTOR_HPP

#include "dinov2.hpp"
#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace rfdetr {

/* MultiScaleProjector (rfdetr-base, single P4 stage).
 *
 * Consumes the 4 multiscale backbone feature maps (all (W, H, C, 1) = (40, 40,
 * 384, 1) for rfdetr-base at 560 px), concatenates them channel-wise into a
 * (40, 40, 1536, 1) tensor, runs C2f(1536 → 256, n=3) plus a final LayerNorm
 * over the channel dim. Returns spatial features (40, 40, 256, 1).
 *
 * Trace callbacks published:
 *   projector.cv1.output        — (40, 40, 256, 1) after C2f.cv1 (ConvX, k=1)
 *   projector.cv2.output        — (40, 40, 256, 1) after C2f.cv2 (ConvX, k=1)
 *   projector.final_norm.output — (40, 40, 256, 1) after channel-axis LN
 *   projector.output            — same tensor (final spatial output)
 *
 * Returns nullptr on failure (missing tensor / bad config). */
ggml_tensor* projector_forward(ggml_context* ctx, const Model& m,
                               const BackboneOutput& bb);

}  // namespace rfdetr

#endif
