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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RFDETR_CAPI_H */
