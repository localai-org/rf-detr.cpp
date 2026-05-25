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
 *   1. Resize to (target_w, target_h) using high-quality bilinear interpolation
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
                                float** out_data, int* out_w, int* out_h);

#ifdef __cplusplus
}
#endif

#endif
