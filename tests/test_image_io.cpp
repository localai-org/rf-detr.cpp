#include "test_assert.hpp"
#include "rfdetr.h"
#include "image_io.hpp"
#include <string>
#include <fstream>
#include <iterator>
#include <cstdlib>
#include <vector>

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

    // ---- Buffer load: round-trip a tiny in-memory PNG ----
    {
        std::string fixture_path = fixtures + "/cats.png";
        std::ifstream f(fixture_path, std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                    std::istreambuf_iterator<char>());
        RFDETR_ASSERT(!bytes.empty());

        rfdetr_status st2;
        rfdetr_image* img = rfdetr_image_load_buffer(bytes.data(), bytes.size(), &st2);
        RFDETR_ASSERT(st2 == RFDETR_OK);
        RFDETR_ASSERT(img != nullptr);
        RFDETR_ASSERT_EQ_INT(rfdetr_image_width(img),  16);
        RFDETR_ASSERT_EQ_INT(rfdetr_image_height(img), 16);
        rfdetr_image_free(img);
    }

    // ---- Render round-trip: load → render with zero detections → reload ----
    {
        rfdetr_status st3;
        rfdetr_image* img = rfdetr_image_load_file((fixtures + "/cats.png").c_str(), &st3);
        RFDETR_ASSERT(st3 == RFDETR_OK);

        const std::string out_path = std::string(fixtures) + "/generated/cats_out.png";
        std::string mkdir_cmd = "mkdir -p " + std::string(fixtures) + "/generated";
        std::system(mkdir_cmd.c_str());

        rfdetr_status r = rfdetr_render(img, nullptr, 0, out_path.c_str());
        RFDETR_ASSERT(r == RFDETR_OK);

        // Re-load the output and check dims match
        rfdetr_status st4;
        rfdetr_image* img2 = rfdetr_image_load_file(out_path.c_str(), &st4);
        RFDETR_ASSERT(st4 == RFDETR_OK);
        RFDETR_ASSERT_EQ_INT(rfdetr_image_width(img2),  rfdetr_image_width(img));
        RFDETR_ASSERT_EQ_INT(rfdetr_image_height(img2), rfdetr_image_height(img));
        rfdetr_image_free(img);
        rfdetr_image_free(img2);
    }

    return 0;
}
