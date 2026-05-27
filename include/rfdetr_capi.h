#ifndef RFDETR_CAPI_H
#define RFDETR_CAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Flat handle. All functions return int status; 0 = ok, negative = error. */
typedef uintptr_t rfdetr_handle_t;

int rfdetr_capi_load(const char* model_path, int n_threads, rfdetr_handle_t* out_handle);
int rfdetr_capi_unload(rfdetr_handle_t handle);

/* Detect on an image file path. Output is a JSON string (caller frees with
 * rfdetr_capi_free_string). */
int rfdetr_capi_detect_path(rfdetr_handle_t handle,
                            const char* image_path,
                            float threshold, uint32_t top_k,
                            char** out_json);

/* Detect on an in-memory image buffer (encoded PNG/JPEG bytes). */
int rfdetr_capi_detect_buffer(rfdetr_handle_t handle,
                              const uint8_t* bytes, size_t len,
                              float threshold, uint32_t top_k,
                              char** out_json);

void rfdetr_capi_free_string(char* s);

/* --------------------------------------------------------------------------
 * Per-detection accessors (LocalAI / purego pattern, mirrors sam3.cpp).
 *
 * `rfdetr_capi_detect_path` / `_detect_buffer` persist the detection batch on
 * the handle. The following accessors read from that batch without
 * re-parsing the JSON envelope. Every detect call clears the previous batch.
 * -------------------------------------------------------------------------- */

/* Number of detections from the most recent detect call on `handle`.
 * Returns 0 if no detect call has been made or it returned no detections.
 * Returns -1 if `handle` is invalid (0/NULL). */
int rfdetr_capi_get_n_detections(rfdetr_handle_t handle);

/* Class id of detection `i` (0-indexed). Returns -1 if `i` is out of range
 * or `handle` is invalid. */
int rfdetr_capi_get_detection_class_id(rfdetr_handle_t handle, int i);

/* Bounding box of detection `i` in original-image pixel coordinates.
 * Writes 4 floats to `out_xyxy` (x1, y1, x2, y2). Returns 0 on success,
 * -1 on invalid handle/index/NULL out pointer. */
int rfdetr_capi_get_detection_box(rfdetr_handle_t handle, int i, float out_xyxy[4]);

/* Confidence score of detection `i`. Returns -1.0 on invalid handle/index. */
float rfdetr_capi_get_detection_score(rfdetr_handle_t handle, int i);

/* Class name of detection `i` (NUL-terminated UTF-8). Two-call sizing:
 *   - pass NULL / 0 to get the required buffer size (including NUL byte);
 *   - pass a buffer >= required size to write the string.
 * Returns the required size (>= 1) on success, -1 on invalid handle/index.
 */
int rfdetr_capi_get_detection_class_name(rfdetr_handle_t handle, int i,
                                         char* buf, int buf_size);

/* PNG-encoded binary segmentation mask of detection `i` (1 byte per pixel,
 * 0 = background, 255 = foreground, same dimensions as the source image).
 *
 * Two-call sizing:
 *   - pass NULL / 0 to get the required buffer size (encoded PNG byte length);
 *   - pass a buffer >= required size to write the encoded bytes.
 *
 * Returns the required size on success, 0 if this detection has no mask
 * (i.e., the model is a detection-only variant), or -1 on invalid
 * handle/index. */
int rfdetr_capi_get_detection_mask_png(rfdetr_handle_t handle, int i,
                                       unsigned char* buf, int buf_size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RFDETR_CAPI_H */
