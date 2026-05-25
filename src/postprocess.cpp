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

extern "C" void rfdetr_detections_free(rfdetr_detection* detections, size_t /*n*/) {
    std::free(detections);
}
