#ifndef RFDETR_POSTPROCESS_HPP
#define RFDETR_POSTPROCESS_HPP

#include "rfdetr.h"
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

/* Convert a single normalized (cx, cy, w, h) box to pixel-space (x1, y1, x2, y2),
 * clamped to [0, img_w] x [0, img_h]. Input and output may not alias. */
void rfdetr_bbox_cxcywh_to_xyxy(const float in[4], int img_w, int img_h, float out[4]);

#ifdef __cplusplus
}
#endif

#endif
