#ifndef RFDETR_H
#define RFDETR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes. 0 = success. */
typedef enum {
    RFDETR_OK                  =  0,
    RFDETR_ERR_INVALID_ARG     = -1,
    RFDETR_ERR_FILE_NOT_FOUND  = -2,
    RFDETR_ERR_IO              = -3,
    RFDETR_ERR_OUT_OF_MEMORY   = -4,
    RFDETR_ERR_DECODE          = -5,
    RFDETR_ERR_MODEL_FORMAT    = -6,   /* GGUF parse / variant detection */
    RFDETR_ERR_MODEL_LOAD      = -7,   /* tensor binding or inventory (e.g. missing tensors) */
    RFDETR_ERR_INFERENCE       = -8,
    RFDETR_ERR_NOT_IMPLEMENTED = -99
} rfdetr_status;

const char* rfdetr_status_str(rfdetr_status s);

/* Logging */
typedef enum {
    RFDETR_LOG_DEBUG = 0,
    RFDETR_LOG_INFO  = 1,
    RFDETR_LOG_WARN  = 2,
    RFDETR_LOG_ERROR = 3
} rfdetr_log_level;

typedef void (*rfdetr_log_cb)(rfdetr_log_level lvl, const char* msg, void* user_data);
void rfdetr_set_log_callback(rfdetr_log_cb cb, void* user_data);

/* Opaque types */
typedef struct rfdetr_context rfdetr_context;
typedef struct rfdetr_image   rfdetr_image;

/* Detections */
typedef struct {
    uint32_t    class_id;
    const char* class_name;  /* borrowed; lifetime tied to rfdetr_context */
    float       score;
    float       x1, y1, x2, y2;  /* pixel coords on the original image */
} rfdetr_detection;

/* Init / detect parameters (forward-declared; full impl in Plan 2/3) */
typedef struct {
    const char*   model_path;
    int           n_threads;
    rfdetr_log_cb log_cb;
    void*         log_user_data;
} rfdetr_params;

typedef struct {
    float           threshold;          /* default 0.5 */
    uint32_t        top_k;              /* default 300 */
    const uint32_t* class_filter;       /* optional allowlist */
    size_t          class_filter_len;
} rfdetr_detect_params;

/* Lifecycle (not yet implemented; returns RFDETR_ERR_NOT_IMPLEMENTED in this plan) */
rfdetr_context* rfdetr_init(const rfdetr_params* params, rfdetr_status* out_status);
void            rfdetr_free(rfdetr_context* ctx);

/* Accessors for loaded context (for `info`-style introspection). */
const char* rfdetr_context_variant(const rfdetr_context* ctx);
uint32_t    rfdetr_context_image_size(const rfdetr_context* ctx);
uint32_t    rfdetr_context_num_queries(const rfdetr_context* ctx);
uint32_t    rfdetr_context_num_classes(const rfdetr_context* ctx);
size_t      rfdetr_context_n_tensors(const rfdetr_context* ctx);

/* Image I/O (IMPLEMENTED in this plan) */
rfdetr_image* rfdetr_image_load_file(const char* path, rfdetr_status* out_status);
rfdetr_image* rfdetr_image_load_buffer(const uint8_t* bytes, size_t len, rfdetr_status* out_status);
void          rfdetr_image_free(rfdetr_image* img);
int           rfdetr_image_width(const rfdetr_image* img);
int           rfdetr_image_height(const rfdetr_image* img);

/* Detection (not yet implemented in this plan) */
rfdetr_status rfdetr_detect(rfdetr_context* ctx,
                            const rfdetr_image* img,
                            const rfdetr_detect_params* params,
                            rfdetr_detection** out_detections,
                            size_t* out_n);
void rfdetr_detections_free(rfdetr_detection* detections, size_t n);

/* Render annotated image (IMPLEMENTED in this plan, boxes only — labels in Plan 2) */
rfdetr_status rfdetr_render(const rfdetr_image* img,
                            const rfdetr_detection* detections, size_t n,
                            const char* out_path);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RFDETR_H */
