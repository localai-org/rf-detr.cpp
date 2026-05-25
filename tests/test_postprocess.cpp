#include "test_assert.hpp"
#include "postprocess.hpp"
#include "rfdetr.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

int main() {
    // ---- bbox_cxcywh_to_xyxy ----
    // (cx=0.5, cy=0.5, w=0.4, h=0.6) on a 200x100 image
    // → x1 = (0.5 - 0.2) * 200 = 60
    //   y1 = (0.5 - 0.3) * 100 = 20
    //   x2 = (0.5 + 0.2) * 200 = 140
    //   y2 = (0.5 + 0.3) * 100 = 80
    {
        float in[4]  = {0.5f, 0.5f, 0.4f, 0.6f};
        float out[4];
        rfdetr_bbox_cxcywh_to_xyxy(in, 200, 100, out);
        RFDETR_ASSERT_NEAR(out[0],  60.0f, 1e-4);
        RFDETR_ASSERT_NEAR(out[1],  20.0f, 1e-4);
        RFDETR_ASSERT_NEAR(out[2], 140.0f, 1e-4);
        RFDETR_ASSERT_NEAR(out[3],  80.0f, 1e-4);
    }

    // Clamping to image bounds: out-of-range boxes clip to [0..W, 0..H]
    {
        float in[4]  = {0.5f, 0.5f, 2.0f, 2.0f};  /* far larger than image */
        float out[4];
        rfdetr_bbox_cxcywh_to_xyxy(in, 100, 50, out);
        RFDETR_ASSERT_NEAR(out[0],   0.0f, 1e-4);
        RFDETR_ASSERT_NEAR(out[1],   0.0f, 1e-4);
        RFDETR_ASSERT_NEAR(out[2], 100.0f, 1e-4);
        RFDETR_ASSERT_NEAR(out[3],  50.0f, 1e-4);
    }

    // ---- rfdetr_select_detections ----
    //
    // 3 queries × 4 classes. We construct logits so that, after sigmoid:
    //   query 0 → class 2 with score ~0.95
    //   query 1 → class 0 with score ~0.7
    //   query 2 → class 3 with score ~0.1  (below threshold 0.5)
    {
        const size_t Q = 3, C = 4;

        // logit such that sigmoid(logit) ≈ desired score:
        auto logit_of = [](float s) { return std::log(s / (1.0f - s)); };

        float logits[Q*C] = {
            /* query 0 */ -3.0f, -3.0f, logit_of(0.95f), -3.0f,
            /* query 1 */ logit_of(0.70f), -3.0f, -3.0f, -3.0f,
            /* query 2 */ -3.0f, -3.0f, -3.0f, logit_of(0.10f),
        };
        float boxes[Q*4] = {
            0.5f, 0.5f, 0.2f, 0.2f,   /* query 0 */
            0.25f, 0.25f, 0.1f, 0.1f, /* query 1 */
            0.5f, 0.5f, 0.5f, 0.5f,   /* query 2 */
        };

        rfdetr_detection* dets = nullptr;
        size_t n = 0;
        rfdetr_select_detections(logits, boxes,
                                 Q, C,
                                 /*threshold*/ 0.5f, /*top_k*/ 10,
                                 /*class_filter*/ nullptr, 0,
                                 /*img_w*/ 100, /*img_h*/ 100,
                                 &dets, &n);

        RFDETR_ASSERT_EQ_INT(n, 2);                  // query 2 dropped (score 0.10 < 0.5)
        // Sorted by score descending: query 0 (0.95) first, query 1 (0.70) second
        RFDETR_ASSERT_EQ_INT(dets[0].class_id, 2);
        RFDETR_ASSERT_NEAR(dets[0].score, 0.95f, 1e-3);
        RFDETR_ASSERT_EQ_INT(dets[1].class_id, 0);
        RFDETR_ASSERT_NEAR(dets[1].score, 0.70f, 1e-3);

        // Box of query 0: cx=cy=0.5, w=h=0.2 on a 100x100 image → (40,40,60,60)
        RFDETR_ASSERT_NEAR(dets[0].x1, 40.0f, 1e-4);
        RFDETR_ASSERT_NEAR(dets[0].y1, 40.0f, 1e-4);
        RFDETR_ASSERT_NEAR(dets[0].x2, 60.0f, 1e-4);
        RFDETR_ASSERT_NEAR(dets[0].y2, 60.0f, 1e-4);

        rfdetr_detections_free(dets, n);
    }

    // Top-K cap
    {
        const size_t Q = 5, C = 1;
        float logits[Q*C];
        for (size_t i = 0; i < Q; ++i) {
            // sigmoid(0) = 0.5 + small perturbation so scores differ
            logits[i] = (float)i * 0.01f;
        }
        float boxes[Q*4] = {0};
        for (size_t i = 0; i < Q; ++i) {
            boxes[i*4 + 0] = 0.5f;
            boxes[i*4 + 1] = 0.5f;
            boxes[i*4 + 2] = 0.1f;
            boxes[i*4 + 3] = 0.1f;
        }
        rfdetr_detection* dets = nullptr;
        size_t n = 0;
        rfdetr_select_detections(logits, boxes, Q, C,
                                 /*threshold*/ 0.0f, /*top_k*/ 3,
                                 nullptr, 0, 100, 100, &dets, &n);
        RFDETR_ASSERT_EQ_INT(n, 3);
        rfdetr_detections_free(dets, n);
    }

    // Class filter
    {
        const size_t Q = 2, C = 3;
        auto logit_of = [](float s) { return std::log(s / (1.0f - s)); };
        float logits[Q*C] = {
            logit_of(0.9f), -3.0f, -3.0f,   /* query 0 → class 0 */
            -3.0f, logit_of(0.9f), -3.0f,   /* query 1 → class 1 */
        };
        float boxes[Q*4] = {
            0.5f, 0.5f, 0.1f, 0.1f,
            0.5f, 0.5f, 0.1f, 0.1f,
        };
        uint32_t allow[1] = {1};   /* only class 1 */
        rfdetr_detection* dets = nullptr;
        size_t n = 0;
        rfdetr_select_detections(logits, boxes, Q, C, 0.5f, 10,
                                 allow, 1, 100, 100, &dets, &n);
        RFDETR_ASSERT_EQ_INT(n, 1);
        RFDETR_ASSERT_EQ_INT(dets[0].class_id, 1);
        rfdetr_detections_free(dets, n);
    }

    // top_k == 0 means "no cap": keep every candidate that passes the threshold
    {
        const size_t Q = 4, C = 1;
        auto logit_of = [](float s) { return std::log(s / (1.0f - s)); };
        float logits[Q*C] = { logit_of(0.9f), logit_of(0.8f), logit_of(0.7f), logit_of(0.6f) };
        float boxes[Q*4];
        for (size_t i = 0; i < Q; ++i) {
            boxes[i*4 + 0] = 0.5f; boxes[i*4 + 1] = 0.5f;
            boxes[i*4 + 2] = 0.1f; boxes[i*4 + 3] = 0.1f;
        }
        rfdetr_detection* dets = nullptr;
        size_t n = 0;
        rfdetr_select_detections(logits, boxes, Q, C,
                                 /*threshold*/ 0.5f, /*top_k*/ 0,
                                 nullptr, 0, 100, 100, &dets, &n);
        RFDETR_ASSERT_EQ_INT(n, 4);  // unlimited cap, all 4 pass
        rfdetr_detections_free(dets, n);
    }

    return 0;
}
