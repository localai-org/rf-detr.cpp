#include "image_io.hpp"
#include "common.hpp"
#include "visualize.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
/* The vendored stb_image_resize2.h (renamed to stb_image_resize.h to match
 * the rest of the stb single-headers in third_party/stb/) self-includes its
 * own filename for re-entry from generated coder sections. Point it at the
 * actual filename so those internal includes resolve. */
#define STBIR__HEADER_FILENAME "stb_image_resize.h"
#include "stb_image_resize.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
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

    rfdetr_image* img = nullptr;
    try {
        img = new (std::nothrow) rfdetr_image();
        if (!img) { stbi_image_free(px); set(RFDETR_ERR_OUT_OF_MEMORY); return nullptr; }
        img->width    = w;
        img->height   = h;
        img->channels = 3;
        img->rgb.assign(px, px + (size_t)w * (size_t)h * 3);
    } catch (const std::bad_alloc&) {
        stbi_image_free(px);
        delete img;
        set(RFDETR_ERR_OUT_OF_MEMORY);
        return nullptr;
    }
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
    std::vector<uint8_t> buf;
    try {
        buf.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    } catch (const std::bad_alloc&) {
        set(RFDETR_ERR_OUT_OF_MEMORY);
        return nullptr;
    }
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
    rfdetr_image copy;
    try {
        copy = *img;  /* deep copy of pixel buffer */
    } catch (const std::bad_alloc&) {
        return RFDETR_ERR_OUT_OF_MEMORY;
    }
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

extern "C" rfdetr_status rfdetr_write_gray_png(const char* path,
                                               const uint8_t* data,
                                               int width, int height) {
    if (!path || !data || width <= 0 || height <= 0) return RFDETR_ERR_INVALID_ARG;
    if (!stbi_write_png(path, width, height, /*channels*/ 1, data, width)) {
        rfdetr_logf(RFDETR_LOG_ERROR, "stbi_write_png failed for '%s'", path);
        return RFDETR_ERR_IO;
    }
    return RFDETR_OK;
}

static void rfdetr_png_writer_callback(void* ctx, void* data, int size) {
    auto* v = static_cast<std::vector<uint8_t>*>(ctx);
    if (!v || !data || size <= 0) return;
    const auto* p = static_cast<const uint8_t*>(data);
    v->insert(v->end(), p, p + size);
}

bool rfdetr_encode_gray_png(const uint8_t* data, int width, int height,
                            std::vector<uint8_t>& out) {
    out.clear();
    if (!data || width <= 0 || height <= 0) return false;
    int rc = stbi_write_png_to_func(rfdetr_png_writer_callback, &out,
                                    width, height, /*channels*/ 1, data, width);
    return rc != 0 && !out.empty();
}

extern "C" rfdetr_status rfdetr_preprocess(const rfdetr_image* img,
                                           int target_w, int target_h,
                                           const float mean[3], const float std_[3],
                                           float** out_data, int* out_w, int* out_h) {
    if (!img || !out_data || !out_w || !out_h || !mean || !std_) {
        return RFDETR_ERR_INVALID_ARG;
    }
    if (target_w <= 0 || target_h <= 0) return RFDETR_ERR_INVALID_ARG;
    if (img->width <= 0 || img->height <= 0 || img->rgb.empty()) {
        return RFDETR_ERR_INVALID_ARG;
    }

    /* 1. Resize via stb_image_resize2 (linear-space bilinear). Input is uint8 RGB packed HWC. */
    std::vector<uint8_t> resized;
    try {
        resized.assign((size_t)target_w * (size_t)target_h * 3, 0);
    } catch (const std::bad_alloc&) {
        return RFDETR_ERR_OUT_OF_MEMORY;
    }
    if (!stbir_resize_uint8_linear(img->rgb.data(), img->width, img->height, 0,
                                   resized.data(), target_w, target_h, 0,
                                   STBIR_RGB)) {
        rfdetr_logf(RFDETR_LOG_ERROR, "rfdetr_preprocess: stbir_resize_uint8_linear failed");
        return RFDETR_ERR_IO;
    }

    /* 2-3. Allocate output F32 buffer and write NCHW row-major.
     *
     * ggml ne = (W, H, 3, 1): ne[0]=W fastest-varying. Memory order:
     *   offset(c, h, w) = c*H*W + h*W + w
     * That's NCHW row-major where w is fastest. */
    const size_t n_elems = (size_t)target_w * (size_t)target_h * 3;
    float* buf = (float*)std::malloc(n_elems * sizeof(float));
    if (!buf) return RFDETR_ERR_OUT_OF_MEMORY;

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < target_h; ++h) {
            for (int w = 0; w < target_w; ++w) {
                uint8_t px = resized[(size_t)(h * target_w + w) * 3 + c];
                float v = (float)px / 255.0f;
                v = (v - mean[c]) / std_[c];
                buf[(size_t)c * target_h * target_w + (size_t)h * target_w + w] = v;
            }
        }
    }

    *out_data = buf;
    *out_w = target_w;
    *out_h = target_h;
    return RFDETR_OK;
}
