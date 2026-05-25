#include "postprocess.hpp"
#include <algorithm>

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
