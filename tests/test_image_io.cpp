#include "test_assert.hpp"
#include "rfdetr.h"
#include "image_io.hpp"
#include <string>

int main() {
    const std::string fixtures = RFDETR_TEST_FIXTURES;
    const std::string path = fixtures + "/cats.png";

    rfdetr_status st;
    rfdetr_image* img = rfdetr_image_load_file(path.c_str(), &st);
    RFDETR_ASSERT(img != nullptr);
    RFDETR_ASSERT(st == RFDETR_OK);

    RFDETR_ASSERT_EQ_INT(rfdetr_image_width(img),  16);
    RFDETR_ASSERT_EQ_INT(rfdetr_image_height(img), 16);

    // Check a known pixel via the internal accessor (declared in image_io.hpp)
    const uint8_t* rgb = rfdetr_image_rgb_data(img);
    RFDETR_ASSERT(rgb != nullptr);
    // pixel (1,2): R=16, G=32, B=2  (from the generator formula)
    int i = (2*16 + 1) * 3;
    RFDETR_ASSERT_EQ_INT(rgb[i+0], 16);
    RFDETR_ASSERT_EQ_INT(rgb[i+1], 32);
    RFDETR_ASSERT_EQ_INT(rgb[i+2], 2);

    rfdetr_image_free(img);

    // Non-existent file -> returns nullptr + status set
    rfdetr_image* img2 = rfdetr_image_load_file("/no/such/file.png", &st);
    RFDETR_ASSERT(img2 == nullptr);
    RFDETR_ASSERT(st == RFDETR_ERR_FILE_NOT_FOUND || st == RFDETR_ERR_IO || st == RFDETR_ERR_DECODE);

    return 0;
}
