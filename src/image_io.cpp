#include "image_io.hpp"
#include "common.hpp"
#include "visualize.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <vector>

extern "C" rfdetr_image* rfdetr_image_load_buffer(const uint8_t* bytes, size_t len, rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) { if (out_status) *out_status = s; };
    if (!bytes || len == 0) { set(RFDETR_ERR_INVALID_ARG); return nullptr; }

    int w = 0, h = 0, c = 0;
    uint8_t* px = stbi_load_from_memory(bytes, (int)len, &w, &h, &c, 3 /* force RGB */);
    if (!px) {
        rfdetr_logf(RFDETR_LOG_ERROR, "stbi_load_from_memory failed: %s", stbi_failure_reason());
        set(RFDETR_ERR_DECODE);
        return nullptr;
    }

    auto* img = new (std::nothrow) rfdetr_image();
    if (!img) { stbi_image_free(px); set(RFDETR_ERR_OUT_OF_MEMORY); return nullptr; }
    img->width    = w;
    img->height   = h;
    img->channels = 3;
    img->rgb.assign(px, px + (size_t)w * h * 3);
    stbi_image_free(px);
    set(RFDETR_OK);
    return img;
}

extern "C" rfdetr_image* rfdetr_image_load_file(const char* path, rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) { if (out_status) *out_status = s; };
    if (!path) { set(RFDETR_ERR_INVALID_ARG); return nullptr; }

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "image_load_file: cannot open '%s' (%s)", path, std::strerror(errno));
        set(RFDETR_ERR_FILE_NOT_FOUND);
        return nullptr;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.empty()) { set(RFDETR_ERR_IO); return nullptr; }

    return rfdetr_image_load_buffer(buf.data(), buf.size(), out_status);
}

extern "C" void rfdetr_image_free(rfdetr_image* img) {
    delete img;
}

extern "C" int rfdetr_image_width(const rfdetr_image* img) {
    return img ? img->width : 0;
}

extern "C" int rfdetr_image_height(const rfdetr_image* img) {
    return img ? img->height : 0;
}

extern "C" const uint8_t* rfdetr_image_rgb_data(const rfdetr_image* img) {
    return img ? img->rgb.data() : nullptr;
}

extern "C" rfdetr_status rfdetr_render(const rfdetr_image* img,
                                       const rfdetr_detection* detections, size_t n,
                                       const char* out_path) {
    if (!img || !out_path) return RFDETR_ERR_INVALID_ARG;

    /* Copy so we don't mutate the caller's image. */
    rfdetr_image copy = *img;
    for (size_t i = 0; i < n; ++i) {
        rfdetr_visualize_draw_box(&copy, detections[i], /*thickness*/ 2);
    }

    int w = copy.width, h = copy.height;
    if (!stbi_write_png(out_path, w, h, 3, copy.rgb.data(), w * 3)) {
        rfdetr_logf(RFDETR_LOG_ERROR, "stbi_write_png failed for '%s'", out_path);
        return RFDETR_ERR_IO;
    }
    return RFDETR_OK;
}
