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

#ifdef __cplusplus
}
#endif

#endif
