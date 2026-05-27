#include "test_assert.hpp"
#include "rfdetr.h"
#include "image_io.hpp"
#include "visualize.hpp"
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdint>

int main() {
    // Synthesize a black image, draw one box, verify the renderer uses the
    // per-class palette color (class_id=0 -> red (230, 25, 75)), draws a
    // thicker stroke (>=3 px), and renders a label background above/inside
    // the box.

    rfdetr_image img;
    img.width = 64;
    img.height = 64;
    img.channels = 3;
    img.rgb.assign(64*64*3, 0);

    rfdetr_detection d{};
    d.class_id = 0;
    d.class_name = "person";
    d.score = 0.99f;
    d.x1 = 8.0f; d.y1 = 24.0f; d.x2 = 56.0f; d.y2 = 56.0f;

    rfdetr_visualize_draw_box(&img, d, 2 /* thickness; min stroke is 3 */);

    auto px = [&](int x, int y, int c) {
        return img.rgb[(y*64 + x)*3 + c];
    };

    // Class 0 = red palette entry (230, 25, 75). The top edge at y=24 should
    // carry that color along the box outline.
    RFDETR_ASSERT_EQ_INT(px(20, 24, 0), 230);
    RFDETR_ASSERT_EQ_INT(px(20, 24, 1), 25);
    RFDETR_ASSERT_EQ_INT(px(20, 24, 2), 75);

    // The bottom edge at y=56 must also be drawn in the same color.
    RFDETR_ASSERT_EQ_INT(px(20, 56, 0), 230);
    RFDETR_ASSERT_EQ_INT(px(20, 56, 1), 25);
    RFDETR_ASSERT_EQ_INT(px(20, 56, 2), 75);

    // Pixel well outside the box (top-left corner) → still black.
    RFDETR_ASSERT_EQ_INT(px(0, 0, 0), 0);
    RFDETR_ASSERT_EQ_INT(px(0, 0, 1), 0);
    RFDETR_ASSERT_EQ_INT(px(0, 0, 2), 0);

    // Interior of the box, far below the label region → still black (outline
    // only; mask overlay was not called).
    RFDETR_ASSERT_EQ_INT(px(30, 50, 0), 0);
    RFDETR_ASSERT_EQ_INT(px(30, 50, 1), 0);
    RFDETR_ASSERT_EQ_INT(px(30, 50, 2), 0);

    // Stroke must be at least 3 px thick: pixel just inside the top edge
    // (y=24+2) must still be the palette color.
    RFDETR_ASSERT_EQ_INT(px(20, 26, 0), 230);
    RFDETR_ASSERT_EQ_INT(px(20, 26, 1), 25);
    RFDETR_ASSERT_EQ_INT(px(20, 26, 2), 75);

    // Mask-overlay smoke test: build a tiny mask covering the top half of a
    // separate image and verify the alpha blend lays down the class color.
    rfdetr_image img2;
    img2.width = 32;
    img2.height = 32;
    img2.channels = 3;
    img2.rgb.assign(32*32*3, 0);

    std::vector<uint8_t> mask(32*32, 0);
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 32; ++x) mask[y*32 + x] = 255;
    }
    rfdetr_detection d2{};
    d2.class_id = 0;
    d2.class_name = "person";
    d2.score = 0.5f;
    d2.x1 = 0.0f; d2.y1 = 0.0f; d2.x2 = 31.0f; d2.y2 = 31.0f;
    d2.mask = mask.data();
    d2.mask_width = 32;
    d2.mask_height = 32;

    rfdetr_visualize_overlay_mask(&img2, d2, 0.4f);

    auto px2 = [&](int x, int y, int c) {
        return img2.rgb[(y*32 + x)*3 + c];
    };
    // Masked pixel (top half): black * 0.6 + (230,25,75) * 0.4 = (92,10,30).
    RFDETR_ASSERT_EQ_INT(px2(10, 5, 0), 92);
    RFDETR_ASSERT_EQ_INT(px2(10, 5, 1), 10);
    RFDETR_ASSERT_EQ_INT(px2(10, 5, 2), 30);
    // Unmasked pixel (bottom half) → still black.
    RFDETR_ASSERT_EQ_INT(px2(10, 20, 0), 0);
    RFDETR_ASSERT_EQ_INT(px2(10, 20, 1), 0);
    RFDETR_ASSERT_EQ_INT(px2(10, 20, 2), 0);

    return 0;
}
