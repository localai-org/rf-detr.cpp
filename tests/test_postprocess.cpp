#include "test_assert.hpp"
#include "postprocess.hpp"
#include "rfdetr.h"
#include <cmath>
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

    return 0;
}
