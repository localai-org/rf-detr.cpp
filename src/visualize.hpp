#ifndef RFDETR_VISUALIZE_HPP
#define RFDETR_VISUALIZE_HPP

#include "rfdetr.h"
#include "image_io.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/* Draw a single detection's bounding box (red, hollow rectangle) on the
 * image. The image is mutated in place. Box coords are in pixel space;
 * out-of-bounds pixels are clipped. */
void rfdetr_visualize_draw_box(rfdetr_image* img, rfdetr_detection det, int thickness);

#ifdef __cplusplus
}
#endif

#endif
