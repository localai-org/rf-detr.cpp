#ifndef RFDETR_VISUALIZE_HPP
#define RFDETR_VISUALIZE_HPP

#include "rfdetr.h"
#include "image_io.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/* Draw a single detection's bounding box on the image (mutated in place).
 *
 * Behavior:
 *   - The rectangle is drawn in a deterministic per-class color (hash of
 *     det.class_id into a 20-color palette).
 *   - `thickness` acts as a lower bound; the renderer enforces a minimum
 *     visible stroke (currently 3 px) clamped to the box dimensions.
 *   - A filled label background is drawn above the box (or just inside the
 *     top edge if the box is at the image top), containing the formatted
 *     "<class_name> <score>" text in a contrasting color.
 *   - Box coords are in pixel space; out-of-bounds pixels are clipped.
 *
 * The C-ABI (signature) is unchanged from prior versions so callers do not
 * need to be recompiled. */
void rfdetr_visualize_draw_box(rfdetr_image* img, rfdetr_detection det, int thickness);

/* Blend a detection's binary mask into the image using the detection's
 * per-class color at the given alpha (0..1). No-op if the detection has no
 * mask (detection-only models). Mutates the image in place. Call this BEFORE
 * rfdetr_visualize_draw_box so the bbox stroke and label sit on top of the
 * tinted mask region. */
void rfdetr_visualize_overlay_mask(rfdetr_image* img, rfdetr_detection det, float alpha);

#ifdef __cplusplus
}
#endif

#endif
