#include "visualize.hpp"

#include <algorithm>
#include <cmath>

namespace {

inline void set_px(rfdetr_image* img, int x, int y,
                   uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || y < 0 || x >= img->width || y >= img->height) return;
    size_t i = ((size_t)y * img->width + x) * 3;
    img->rgb[i + 0] = r;
    img->rgb[i + 1] = g;
    img->rgb[i + 2] = b;
}

}  // namespace

extern "C" void rfdetr_visualize_draw_box(rfdetr_image* img, rfdetr_detection det, int thickness) {
    if (!img || thickness <= 0) return;

    int x1 = (int)std::lround(det.x1);
    int y1 = (int)std::lround(det.y1);
    int x2 = (int)std::lround(det.x2);
    int y2 = (int)std::lround(det.y2);
    if (x2 < x1) std::swap(x1, x2);
    if (y2 < y1) std::swap(y1, y2);

    const uint8_t R = 255, G = 0, B = 0;
    for (int t = 0; t < thickness; ++t) {
        // top & bottom edges
        for (int x = x1; x <= x2; ++x) {
            set_px(img, x, y1 + t, R, G, B);
            set_px(img, x, y2 - t, R, G, B);
        }
        // left & right edges
        for (int y = y1; y <= y2; ++y) {
            set_px(img, x1 + t, y, R, G, B);
            set_px(img, x2 - t, y, R, G, B);
        }
    }
}
