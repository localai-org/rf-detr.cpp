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
 *   top_k            — keep at most this many predictions (after threshold).
 *                      0 means "no cap" (keep all predictions that passed the
 *                      threshold). Public callers use rfdetr_detect_params.top_k
 *                      which defaults to 300; only the internal helper supports
 *                      0 == unlimited.
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

/* Score selection that additionally attaches a binary mask to each
 * surviving detection.
 *
 * Inputs (same as rfdetr_select_detections plus):
 *   masks_logits     [mask_w * mask_h * num_queries] — raw mask logits
 *                    (W, H, N) flat layout in the same order ggml produces
 *                    them (W fastest). May be nullptr if num_queries=0.
 *   mask_w, mask_h   spatial extent of `masks_logits` (e.g. 78x78 for seg-nano
 *                    at 312px input).
 *   mask_threshold   sigmoid(logit) > threshold → foreground. Default 0.5.
 *
 * For each surviving detection the mask is bilinearly upsampled to
 * (img_w, img_h) and thresholded; resulting bitmask is owned by the
 * returned `rfdetr_detection[]` array (lifetime tied to it; freed by
 * rfdetr_detections_free).
 */
void rfdetr_select_detections_with_masks(
    const float* class_logits,
    const float* bbox_cxcywh,
    const float* masks_logits, int mask_w, int mask_h, float mask_threshold,
    size_t num_queries, size_t num_classes,
    float threshold, uint32_t top_k,
    const uint32_t* class_filter, size_t class_filter_len,
    int img_w, int img_h,
    rfdetr_detection** out_detections, size_t* out_n);

#ifdef __cplusplus
}
#endif

#endif
