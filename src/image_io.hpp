#ifndef RFDETR_IMAGE_IO_HPP
#define RFDETR_IMAGE_IO_HPP

#include "rfdetr.h"

#include <cstdint>
#include <vector>

struct rfdetr_image {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<uint8_t> rgb;  /* HWC, row-major, 0..255 */
};

#ifdef __cplusplus
extern "C" {
#endif

const uint8_t* rfdetr_image_rgb_data(const rfdetr_image* img);

/* Preprocess an image for model input:
 *   1. Resize to (target_w, target_h). `bilinear_no_antialias=true` matches
 *      RF-DETR 1.9's float bilinear F.resize(..., antialias=False); false
 *      preserves the legacy stb filtering used by existing GGUFs.
 *   2. Convert uint8 RGB -> float32 in [0, 1]
 *   3. Apply ImageNet normalization: (pixel - mean) / std (per channel)
 *   4. Output in (W, H, 3, 1) ggml layout, NCHW row-major equivalent
 *
 * The output buffer is allocated by the function; caller frees with std::free.
 *
 * Returns RFDETR_OK and fills *out_data + *out_w + *out_h on success. */
rfdetr_status rfdetr_preprocess(const rfdetr_image* img,
                                int target_w, int target_h,
                                const float mean[3], const float std_[3],
                                bool bilinear_no_antialias,
                                float** out_data, int* out_w, int* out_h);

/* Write a single-channel uint8 (grayscale) buffer as a PNG. Returns
 * RFDETR_OK on success. `data` is row-major, size = width * height bytes. */
rfdetr_status rfdetr_write_gray_png(const char* path,
                                    const uint8_t* data,
                                    int width, int height);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
/* C++-only helper: PNG-encode a grayscale buffer into an in-memory vector.
 * Same input layout as rfdetr_write_gray_png (row-major, 1 byte per pixel).
 * Used by the flat C-API accessor for serving masks to LocalAI without
 * hitting disk. Returns true on success, false on encoding failure. */
bool rfdetr_encode_gray_png(const uint8_t* data, int width, int height,
                            std::vector<uint8_t>& out);
#endif

#endif
