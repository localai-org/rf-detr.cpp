#include "test_assert.hpp"
#include "rfdetr.h"
#include "image_io.hpp"
#include "visualize.hpp"
#include <string>
#include <cstdlib>

int main() {
    // Synthesize a 32x32 black image, draw one red box, verify pixels on the
    // box outline became red.
    rfdetr_image img;
    img.width = 32;
    img.height = 32;
    img.channels = 3;
    img.rgb.assign(32*32*3, 0);

    rfdetr_detection d{};
    d.class_id = 0;
    d.score = 0.99f;
    d.x1 = 4.0f; d.y1 = 4.0f; d.x2 = 28.0f; d.y2 = 28.0f;

    rfdetr_visualize_draw_box(&img, d, 2 /* thickness */);

    auto px = [&](int x, int y, int c) {
        return img.rgb[(y*32 + x)*3 + c];
    };

    // Top edge: y = 4, x = 4..27 → red
    RFDETR_ASSERT_EQ_INT(px(10, 4, 0), 255);
    RFDETR_ASSERT_EQ_INT(px(10, 4, 1), 0);
    RFDETR_ASSERT_EQ_INT(px(10, 4, 2), 0);

    // Pixel well outside the box → still black
    RFDETR_ASSERT_EQ_INT(px(0, 0, 0), 0);
    RFDETR_ASSERT_EQ_INT(px(0, 0, 1), 0);
    RFDETR_ASSERT_EQ_INT(px(0, 0, 2), 0);

    // Interior of the box (not on the edge) → still black (we draw outline only)
    RFDETR_ASSERT_EQ_INT(px(16, 16, 0), 0);

    return 0;
}
