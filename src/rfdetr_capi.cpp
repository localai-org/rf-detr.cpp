#include "rfdetr_capi.h"
#include "rfdetr.h"
#include "common.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

namespace {

/* Serialize an rfdetr_context plus the result of a detect attempt into a
 * JSON envelope. The envelope always includes a `status` field so callers
 * can distinguish "model loaded but inference NYI" from "decode failure". */
std::string make_json(int img_w, int img_h, const char* status,
                      const rfdetr_detection* dets, size_t n) {
    std::ostringstream o;
    o << "{\"status\":\"" << status << "\","
      << "\"image\":{\"width\":" << img_w << ",\"height\":" << img_h << "},"
      << "\"detections\":[";
    for (size_t i = 0; i < n; ++i) {
        if (i) o << ",";
        o << "{\"class_id\":" << dets[i].class_id
          << ",\"score\":" << dets[i].score
          << ",\"x1\":" << dets[i].x1
          << ",\"y1\":" << dets[i].y1
          << ",\"x2\":" << dets[i].x2
          << ",\"y2\":" << dets[i].y2
          << "}";
    }
    o << "]}";
    return o.str();
}

char* dup_to_c(const std::string& s) {
    char* buf = (char*)std::malloc(s.size() + 1);
    if (!buf) return nullptr;
    std::memcpy(buf, s.data(), s.size());
    buf[s.size()] = '\0';
    return buf;
}

}  // namespace

extern "C" int rfdetr_capi_load(const char* model_path, int n_threads, rfdetr_handle_t* out_handle) {
    if (!model_path || !out_handle) return -1;
    *out_handle = 0;

    rfdetr_params p{};
    p.model_path = model_path;
    p.n_threads  = n_threads > 0 ? n_threads : 1;

    rfdetr_status st;
    rfdetr_context* ctx = rfdetr_init(&p, &st);
    if (!ctx) return (int)st;

    *out_handle = (rfdetr_handle_t)ctx;
    return 0;
}

extern "C" int rfdetr_capi_unload(rfdetr_handle_t handle) {
    if (handle == 0) return 0;
    rfdetr_free((rfdetr_context*)handle);
    return 0;
}

static int capi_detect_common(rfdetr_handle_t handle, rfdetr_image* img,
                              float threshold, uint32_t top_k, char** out_json) {
    auto* ctx = (rfdetr_context*)handle;
    if (!ctx || !img || !out_json) return -1;
    *out_json = nullptr;

    rfdetr_detect_params dp{};
    dp.threshold = threshold;
    dp.top_k     = top_k;

    rfdetr_detection* dets = nullptr;
    size_t n = 0;
    rfdetr_status st = rfdetr_detect(ctx, img, &dp, &dets, &n);

    const char* status = (st == RFDETR_OK)                  ? "ok"
                       : (st == RFDETR_ERR_NOT_IMPLEMENTED) ? "not_implemented"
                       :                                       rfdetr_status_str(st);

    std::string json = make_json(rfdetr_image_width(img), rfdetr_image_height(img),
                                 status, dets, n);
    rfdetr_detections_free(dets, n);

    *out_json = dup_to_c(json);
    if (!*out_json) return RFDETR_ERR_OUT_OF_MEMORY;
    return 0;
}

extern "C" int rfdetr_capi_detect_path(rfdetr_handle_t handle, const char* image_path,
                                       float threshold, uint32_t top_k, char** out_json) {
    if (!handle || !image_path || !out_json) return -1;
    *out_json = nullptr;

    rfdetr_status load_st;
    rfdetr_image* img = rfdetr_image_load_file(image_path, &load_st);
    if (!img) return (int)load_st;

    int rc = capi_detect_common(handle, img, threshold, top_k, out_json);
    rfdetr_image_free(img);
    return rc;
}

extern "C" int rfdetr_capi_detect_buffer(rfdetr_handle_t handle,
                                         const uint8_t* bytes, size_t len,
                                         float threshold, uint32_t top_k,
                                         char** out_json) {
    if (!handle || !bytes || len == 0 || !out_json) return -1;
    *out_json = nullptr;

    rfdetr_status load_st;
    rfdetr_image* img = rfdetr_image_load_buffer(bytes, len, &load_st);
    if (!img) return (int)load_st;

    int rc = capi_detect_common(handle, img, threshold, top_k, out_json);
    rfdetr_image_free(img);
    return rc;
}

extern "C" void rfdetr_capi_free_string(char* s) {
    std::free(s);
}
