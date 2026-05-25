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

/* Score selection.
 *
 * Inputs:
 *   class_logits     [num_queries * num_classes] — raw logits, row-major (query, class)
 *   bbox_cxcywh      [num_queries * 4]            — normalized predicted boxes
 *   num_queries, num_classes
 *   threshold        — drop predictions with sigmoid(logit) <= threshold
 *   top_k            — keep at most this many predictions (after threshold)
 *   class_filter     — optional allowlist of class ids; NULL = all classes
 *   class_filter_len — length of allowlist
 *   img_w, img_h     — original image dimensions, for xyxy projection
 *
 * Output:
 *   detections — vector cleared and populated by the function. class_name is
 *   set to nullptr; caller may attach names from the model's class list.
 */
void rfdetr_select_detections(const float* class_logits,
                              const float* bbox_cxcywh,
                              size_t num_queries, size_t num_classes,
                              float threshold, uint32_t top_k,
                              const uint32_t* class_filter, size_t class_filter_len,
                              int img_w, int img_h,
                              rfdetr_detection** out_detections, size_t* out_n);

#ifdef __cplusplus
}
#endif

#endif
