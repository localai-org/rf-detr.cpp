#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

extern "C" void rfdetr_bbox_cxcywh_to_xyxy(const float in[4], int img_w, int img_h, float out[4]) {
    const float cx = in[0], cy = in[1], w = in[2], h = in[3];
    float x1 = (cx - 0.5f * w) * (float)img_w;
    float y1 = (cy - 0.5f * h) * (float)img_h;
    float x2 = (cx + 0.5f * w) * (float)img_w;
    float y2 = (cy + 0.5f * h) * (float)img_h;
    out[0] = std::clamp(x1, 0.0f, (float)img_w);
    out[1] = std::clamp(y1, 0.0f, (float)img_h);
    out[2] = std::clamp(x2, 0.0f, (float)img_w);
    out[3] = std::clamp(y2, 0.0f, (float)img_h);
}

namespace {

inline float sigmoidf(float x) {
    /* Numerically stable: avoid overflow on large negative x. */
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = std::exp(x);
        return z / (1.0f + z);
    }
}

struct Candidate {
    uint32_t query;
    uint32_t class_id;
    float    score;
};

}  // namespace

extern "C" void rfdetr_select_detections(const float* class_logits,
                                         const float* bbox_cxcywh,
                                         size_t num_queries, size_t num_classes,
                                         float threshold, uint32_t top_k,
                                         const uint32_t* class_filter, size_t class_filter_len,
                                         int img_w, int img_h,
                                         rfdetr_detection** out_detections, size_t* out_n) {
    *out_detections = nullptr;
    *out_n = 0;

    if (!class_logits || !bbox_cxcywh || num_queries == 0 || num_classes == 0) return;

    try {
        /* Fast-membership lookup for the class filter. For small allowlists this
         * linear scan is fine; revisit if num_classes grows. */
        auto class_allowed = [&](uint32_t cid) -> bool {
            if (!class_filter || class_filter_len == 0) return true;
            for (size_t i = 0; i < class_filter_len; ++i) {
                if (class_filter[i] == cid) return true;
            }
            return false;
        };

        std::vector<Candidate> cands;
        cands.reserve(num_queries);

        for (size_t q = 0; q < num_queries; ++q) {
            /* Take argmax over classes after sigmoid (DETR convention: per-class
             * sigmoid, not softmax across classes). */
            const float* row = class_logits + q * num_classes;
            uint32_t best_c = 0;
            float    best_s = -1.0f;
            for (size_t c = 0; c < num_classes; ++c) {
                float s = sigmoidf(row[c]);
                if (s > best_s) { best_s = s; best_c = (uint32_t)c; }
            }
            if (best_s <= threshold) continue;
            if (!class_allowed(best_c)) continue;
            cands.push_back({(uint32_t)q, best_c, best_s});
        }

        /* Sort by score descending. */
        std::sort(cands.begin(), cands.end(),
                  [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

        if (top_k > 0 && cands.size() > top_k) cands.resize(top_k);
        if (cands.empty()) return;

        auto* out = (rfdetr_detection*)std::calloc(cands.size(), sizeof(rfdetr_detection));
        if (!out) return;

        for (size_t i = 0; i < cands.size(); ++i) {
            const auto& c = cands[i];
            out[i].class_id   = c.class_id;
            out[i].class_name = nullptr;
            out[i].score      = c.score;
            float box[4];
            rfdetr_bbox_cxcywh_to_xyxy(bbox_cxcywh + c.query * 4, img_w, img_h, box);
            out[i].x1 = box[0];
            out[i].y1 = box[1];
            out[i].x2 = box[2];
            out[i].y2 = box[3];
        }
        *out_detections = out;
        *out_n          = cands.size();
    } catch (const std::bad_alloc&) {
        *out_detections = nullptr;
        *out_n = 0;
        return;
    }
}

extern "C" void rfdetr_detections_free(rfdetr_detection* detections, size_t n) {
    if (!detections) return;
    for (size_t i = 0; i < n; ++i) {
        if (detections[i].mask) {
            std::free(const_cast<uint8_t*>(detections[i].mask));
        }
    }
    std::free(detections);
}

namespace {

/* Bilinear upsample a (src_w, src_h) F32 mask to (dst_w, dst_h), threshold,
 * and write a uint8 0/255 buffer.  Uses the same align_corners=False
 * convention as torch's F.interpolate (matches ggml_interpolate).
 *
 * Returns a heap-allocated (caller-owned) uint8 buffer of size dst_w * dst_h. */
uint8_t* upsample_and_threshold_mask(const float* src,
                                     int src_w, int src_h,
                                     int dst_w, int dst_h,
                                     float threshold) {
    auto* out = (uint8_t*)std::calloc((size_t)dst_w * dst_h, sizeof(uint8_t));
    if (!out) return nullptr;

    const float sf_x = (float)src_w / (float)dst_w;
    const float sf_y = (float)src_h / (float)dst_h;

    auto sigmoidf = [](float x) -> float {
        if (x >= 0.0f) {
            float z = std::exp(-x);
            return 1.0f / (1.0f + z);
        } else {
            float z = std::exp(x);
            return z / (1.0f + z);
        }
    };

    for (int dy = 0; dy < dst_h; ++dy) {
        const float y = ((float)dy + 0.5f) * sf_y - 0.5f;
        int y0 = (int)std::floor(y);
        int y1 = y0 + 1;
        y0 = std::clamp(y0, 0, src_h - 1);
        y1 = std::clamp(y1, 0, src_h - 1);
        const float dyf = std::clamp(y - (float)y0, 0.0f, 1.0f);

        for (int dx = 0; dx < dst_w; ++dx) {
            const float x = ((float)dx + 0.5f) * sf_x - 0.5f;
            int x0 = (int)std::floor(x);
            int x1 = x0 + 1;
            x0 = std::clamp(x0, 0, src_w - 1);
            x1 = std::clamp(x1, 0, src_w - 1);
            const float dxf = std::clamp(x - (float)x0, 0.0f, 1.0f);

            /* src is (W, H) row-major-on-W layout from ggml's per-query slice. */
            const float a = src[(size_t)y0 * src_w + x0];
            const float b = src[(size_t)y0 * src_w + x1];
            const float c = src[(size_t)y1 * src_w + x0];
            const float d = src[(size_t)y1 * src_w + x1];
            const float v = a * (1 - dxf) * (1 - dyf) +
                            b * dxf * (1 - dyf) +
                            c * (1 - dxf) * dyf +
                            d * dxf * dyf;
            out[(size_t)dy * dst_w + dx] = (sigmoidf(v) > threshold) ? 255 : 0;
        }
    }
    return out;
}

}  // namespace

extern "C" void rfdetr_select_detections_with_masks(
    const float* class_logits,
    const float* bbox_cxcywh,
    const float* masks_logits, int mask_w, int mask_h, float mask_threshold,
    size_t num_queries, size_t num_classes,
    float threshold, uint32_t top_k,
    const uint32_t* class_filter, size_t class_filter_len,
    int img_w, int img_h,
    rfdetr_detection** out_detections, size_t* out_n) {
    *out_detections = nullptr;
    *out_n = 0;

    /* First do the regular selection (without masks). */
    rfdetr_detection* base = nullptr;
    size_t base_n = 0;
    rfdetr_select_detections(class_logits, bbox_cxcywh,
                             num_queries, num_classes,
                             threshold, top_k,
                             class_filter, class_filter_len,
                             img_w, img_h,
                             &base, &base_n);
    if (!base || base_n == 0) {
        *out_detections = base;
        *out_n = base_n;
        return;
    }

    /* Without masks we're done. */
    if (!masks_logits || mask_w <= 0 || mask_h <= 0) {
        *out_detections = base;
        *out_n = base_n;
        return;
    }

    /* For each surviving detection, we need to know which query slot it
     * came from. Reconstruct the same {query, class, score} mapping the
     * inner selector built so we can pick the right mask row.
     *
     * The simpler path: re-run the inner selection logic and match by
     * (class_id, score) — but this is fragile. Instead, redo the same
     * candidate construction here (it's cheap: O(num_queries * num_classes))
     * and use the per-candidate query index directly. */
    auto sigmoidf = [](float x) -> float {
        if (x >= 0.0f) {
            float z = std::exp(-x);
            return 1.0f / (1.0f + z);
        } else {
            float z = std::exp(x);
            return z / (1.0f + z);
        }
    };
    auto class_allowed = [&](uint32_t cid) -> bool {
        if (!class_filter || class_filter_len == 0) return true;
        for (size_t i = 0; i < class_filter_len; ++i) {
            if (class_filter[i] == cid) return true;
        }
        return false;
    };
    struct Cand { uint32_t query; uint32_t cls; float score; };
    std::vector<Cand> cands;
    cands.reserve(num_queries);
    for (size_t q = 0; q < num_queries; ++q) {
        const float* row = class_logits + q * num_classes;
        uint32_t best_c = 0; float best_s = -1.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            float s = sigmoidf(row[c]);
            if (s > best_s) { best_s = s; best_c = (uint32_t)c; }
        }
        if (best_s <= threshold) continue;
        if (!class_allowed(best_c)) continue;
        cands.push_back({(uint32_t)q, best_c, best_s});
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.score > b.score; });
    if (top_k > 0 && cands.size() > top_k) cands.resize(top_k);

    /* cands and base must be in the same order — they were built by the
     * same logic. Attach masks. */
    const size_t plane = (size_t)mask_w * mask_h;
    for (size_t i = 0; i < base_n && i < cands.size(); ++i) {
        const uint32_t q = cands[i].query;
        const float* src = masks_logits + q * plane;
        uint8_t* upsampled = upsample_and_threshold_mask(
            src, mask_w, mask_h, img_w, img_h, mask_threshold);
        base[i].mask        = upsampled;
        base[i].mask_width  = upsampled ? img_w : 0;
        base[i].mask_height = upsampled ? img_h : 0;
    }

    *out_detections = base;
    *out_n = base_n;
}
