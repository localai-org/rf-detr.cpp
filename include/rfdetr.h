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

/* Detections.
 *
 * Mask fields (mask, mask_width, mask_height) are populated ONLY when the
 * model has a segmentation head (RFDETRSeg* variants). For detection-only
 * models they're set to {nullptr, 0, 0}.
 *
 * mask layout: row-major (y, x), 1 byte per pixel, 0=background, 255=foreground
 * (thresholded). mask_width and mask_height are the ORIGINAL image dimensions
 * (the seg-head output is upsampled to that resolution and then thresholded).
 *
 * Ownership: the mask buffer is allocated alongside the rfdetr_detection
 * array and freed by rfdetr_detections_free. Do not free `mask` separately. */
typedef struct {
    uint32_t    class_id;
    const char* class_name;  /* borrowed; lifetime tied to rfdetr_context */
    float       score;
    float       x1, y1, x2, y2;  /* pixel coords on the original image */
    const uint8_t* mask;     /* row-major (y, x) binary mask, or NULL */
    int         mask_width;  /* width in pixels of the mask, or 0 */
    int         mask_height; /* height in pixels of the mask, or 0 */
} rfdetr_detection;

/* Init / detect parameters (forward-declared; full impl in Plan 2/3) */
typedef struct {
    const char*   model_path;
    /* CPU thread count for the ggml backend.
     *   0 or negative → auto-detect (std::thread::hardware_concurrency(), clamped to 1).
     *   >0            → use that many threads. */
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

/* Wrap an existing 8-bit RGB pixel buffer as an rfdetr_image. The buffer is
 * COPIED into the returned image (no aliasing); the caller retains ownership
 * of `rgb` and may free it immediately after this call.
 *
 * Layout: `rgb` must be width*height*3 bytes in HWC row-major order
 * (R, G, B, R, G, B, ...) — the same layout produced by stb_image with
 * desired_channels=3 or by `ffmpeg -f rawvideo -pix_fmt rgb24 -`.
 *
 * Returns NULL on failure; the returned image must be freed with
 * rfdetr_image_free. */
rfdetr_image* rfdetr_image_from_rgb_buffer(const uint8_t* rgb,
                                           int width, int height,
                                           rfdetr_status* out_status);

void          rfdetr_image_free(rfdetr_image* img);
int           rfdetr_image_width(const rfdetr_image* img);
int           rfdetr_image_height(const rfdetr_image* img);

/* Pointer to the image's contiguous 8-bit RGB pixel buffer (HWC row-major,
 * width*height*3 bytes). Valid until the image is freed or mutated by a
 * future API call (e.g. an in-place annotation helper).
 *
 * Used by examples that need to push annotated frames back out to disk or to
 * an external encoder. Returns NULL if `img` is NULL. */
const uint8_t* rfdetr_image_rgb_data(const rfdetr_image* img);

/* Draw the bounding box + label for a single detection on `img` (mutated in
 * place). For segmentation models, you typically call
 * rfdetr_visualize_overlay_mask FIRST so the box and label sit on top of the
 * tinted mask region.
 *
 *   - The rectangle is drawn in a deterministic per-class color (palette of 20).
 *   - `thickness` is a lower bound; a minimum visible stroke (3 px) is enforced
 *     and clamped to the box dimensions.
 *   - A filled label background is drawn just above the box (or just inside
 *     the top edge if the box is at the image top), with "<class_name> <score>"
 *     text in a contrasting color.
 *   - Out-of-bounds pixels are clipped silently. */
void rfdetr_visualize_draw_box(rfdetr_image* img, rfdetr_detection det, int thickness);

/* Blend a detection's binary mask into `img` using the detection's per-class
 * color at the given `alpha` (0..1). No-op if the detection has no mask
 * (detection-only models). Mutates the image in place. */
void rfdetr_visualize_overlay_mask(rfdetr_image* img, rfdetr_detection det, float alpha);

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
