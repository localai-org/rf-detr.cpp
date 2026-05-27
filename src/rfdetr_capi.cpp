#include "rfdetr_capi.h"
#include "rfdetr.h"
#include "image_io.hpp"
#include "common.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <sstream>
#include <string>
#include <vector>

namespace {

/* Per-handle storage. The flat C-API previously stored only the
 * `rfdetr_context*` cast to `uintptr_t`. To support the accessor pattern used
 * by LocalAI's purego layer (number of detections + per-detection field
 * lookups + PNG-encoded masks), each `rfdetr_capi_load` now allocates this
 * wrapper. `rfdetr_capi_detect_path` / `_detect_buffer` populate
 * `last_detections` (and `last_json`) on every call; accessors read from
 * `last_detections`.
 *
 * Stale results from a prior detect call are cleared at the start of every
 * new detect call, so an accessor following a failed detect returns "no
 * detections" rather than leaking the previous batch. */
struct CapiHandle {
    rfdetr_context* ctx = nullptr;

    struct DetectionStore {
        int class_id = 0;
        std::string class_name;  // owned copy (rfdetr_detection.class_name is borrowed)
        float score = 0.0f;
        float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        std::vector<uint8_t> mask_png;  // empty for non-seg models / no mask
        int mask_w = 0;
        int mask_h = 0;
    };
    std::vector<DetectionStore> last_detections;
    std::string last_json;
};

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

    auto* h = new (std::nothrow) CapiHandle();
    if (!h) {
        rfdetr_free(ctx);
        return (int)RFDETR_ERR_OUT_OF_MEMORY;
    }
    h->ctx = ctx;

    *out_handle = (rfdetr_handle_t)h;
    return 0;
}

extern "C" int rfdetr_capi_unload(rfdetr_handle_t handle) {
    if (handle == 0) return 0;
    auto* h = (CapiHandle*)handle;
    if (h->ctx) rfdetr_free(h->ctx);
    delete h;
    return 0;
}

static int capi_detect_common(rfdetr_handle_t handle, rfdetr_image* img,
                              float threshold, uint32_t top_k, char** out_json) {
    auto* h = (CapiHandle*)handle;
    if (!h || !h->ctx || !img || !out_json) return -1;
    *out_json = nullptr;

    /* Clear any prior batch so accessors can never read stale results. */
    h->last_detections.clear();
    h->last_json.clear();

    rfdetr_detect_params dp{};
    dp.threshold = threshold;
    dp.top_k     = top_k;

    rfdetr_detection* dets = nullptr;
    size_t n = 0;
    rfdetr_status st = rfdetr_detect(h->ctx, img, &dp, &dets, &n);

    const char* status = (st == RFDETR_OK)                  ? "ok"
                       : (st == RFDETR_ERR_NOT_IMPLEMENTED) ? "not_implemented"
                       :                                       rfdetr_status_str(st);

    /* Persist the batch into the handle storage so accessors can read it.
     * Done before json serialization and dets-free so we can copy fields. */
    if (st == RFDETR_OK && dets && n > 0) {
        h->last_detections.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const rfdetr_detection& d = dets[i];
            CapiHandle::DetectionStore s;
            s.class_id   = (int)d.class_id;
            s.class_name = d.class_name ? d.class_name : "";
            s.score      = d.score;
            s.x1 = d.x1; s.y1 = d.y1; s.x2 = d.x2; s.y2 = d.y2;
            s.mask_w = d.mask_width;
            s.mask_h = d.mask_height;
            if (d.mask && d.mask_width > 0 && d.mask_height > 0) {
                rfdetr_encode_gray_png(d.mask, d.mask_width, d.mask_height, s.mask_png);
            }
            h->last_detections.push_back(std::move(s));
        }
    }

    std::string json = make_json(rfdetr_image_width(img), rfdetr_image_height(img),
                                 status, dets, n);
    rfdetr_detections_free(dets, n);

    h->last_json = json;
    *out_json = dup_to_c(json);
    if (!*out_json) return RFDETR_ERR_OUT_OF_MEMORY;
    return 0;
}

extern "C" int rfdetr_capi_detect_path(rfdetr_handle_t handle, const char* image_path,
                                       float threshold, uint32_t top_k, char** out_json) {
    if (!handle || !image_path || !out_json) return -1;
    *out_json = nullptr;

    /* Clear stale results before image load so an early-failure detect call
     * cannot leave the previous batch readable via the accessors. */
    if (auto* h = (CapiHandle*)handle) {
        h->last_detections.clear();
        h->last_json.clear();
    }

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

    if (auto* h = (CapiHandle*)handle) {
        h->last_detections.clear();
        h->last_json.clear();
    }

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

/* --------------------------------------------------------------------------
 * Accessor functions (LocalAI / purego pattern, mirrors sam3-cpp gosam3.h).
 *
 * Each call reads from the handle's `last_detections`, populated by the most
 * recent detect call. Returns sentinel values for invalid handle/index so the
 * Go side can branch on negative return codes.
 *
 * Two-call sizing pattern (class_name, mask_png): first call with NULL/0
 * returns the required buffer size; second call with a sufficient buffer
 * writes the bytes. Avoids any allocator coordination across the FFI boundary.
 * -------------------------------------------------------------------------- */

extern "C" int rfdetr_capi_get_n_detections(rfdetr_handle_t handle) {
    auto* h = (CapiHandle*)handle;
    if (!h) return -1;
    return (int)h->last_detections.size();
}

extern "C" int rfdetr_capi_get_detection_class_id(rfdetr_handle_t handle, int i) {
    auto* h = (CapiHandle*)handle;
    if (!h || i < 0 || (size_t)i >= h->last_detections.size()) return -1;
    return h->last_detections[i].class_id;
}

extern "C" int rfdetr_capi_get_detection_box(rfdetr_handle_t handle, int i, float out_xyxy[4]) {
    auto* h = (CapiHandle*)handle;
    if (!h || i < 0 || (size_t)i >= h->last_detections.size() || !out_xyxy) return -1;
    const auto& d = h->last_detections[i];
    out_xyxy[0] = d.x1;
    out_xyxy[1] = d.y1;
    out_xyxy[2] = d.x2;
    out_xyxy[3] = d.y2;
    return 0;
}

extern "C" float rfdetr_capi_get_detection_score(rfdetr_handle_t handle, int i) {
    auto* h = (CapiHandle*)handle;
    if (!h || i < 0 || (size_t)i >= h->last_detections.size()) return -1.0f;
    return h->last_detections[i].score;
}

extern "C" int rfdetr_capi_get_detection_class_name(rfdetr_handle_t handle, int i,
                                                    char* buf, int buf_size) {
    auto* h = (CapiHandle*)handle;
    if (!h || i < 0 || (size_t)i >= h->last_detections.size()) return -1;
    const std::string& s = h->last_detections[i].class_name;
    int needed = (int)s.size() + 1;  // include NUL
    if (buf == nullptr || buf_size < needed) return needed;
    std::memcpy(buf, s.data(), s.size());
    buf[s.size()] = '\0';
    return needed;
}

extern "C" int rfdetr_capi_get_detection_mask_png(rfdetr_handle_t handle, int i,
                                                  unsigned char* buf, int buf_size) {
    auto* h = (CapiHandle*)handle;
    if (!h || i < 0 || (size_t)i >= h->last_detections.size()) return -1;
    const auto& mp = h->last_detections[i].mask_png;
    if (mp.empty()) return 0;  // no mask (detection-only model)
    int needed = (int)mp.size();
    if (buf == nullptr || buf_size < needed) return needed;
    std::memcpy(buf, mp.data(), (size_t)needed);
    return needed;
}
