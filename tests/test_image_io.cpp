#include "test_assert.hpp"
#include "rfdetr.h"
#include "image_io.hpp"
#include <string>
#include <filesystem>
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
        std::filesystem::create_directories(std::string(fixtures) + "/generated");

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

    /* Preprocess test: load cats.png (16x16), resize to 56x56, normalize, verify shape */
    {
        rfdetr_status st;
        rfdetr_image* img = rfdetr_image_load_file((fixtures + "/cats.png").c_str(), &st);
        RFDETR_ASSERT(img != nullptr);
        RFDETR_ASSERT_EQ_INT(st, RFDETR_OK);

        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std_[3] = {0.229f, 0.224f, 0.225f};

        float* data = nullptr;
        int w = 0, h = 0;
        rfdetr_status pp_st = rfdetr_preprocess(img, 56, 56, mean, std_, &data, &w, &h);
        RFDETR_ASSERT_EQ_INT(pp_st, RFDETR_OK);
        RFDETR_ASSERT_EQ_INT(w, 56);
        RFDETR_ASSERT_EQ_INT(h, 56);
        RFDETR_ASSERT(data != nullptr);

        /* Sanity: not all zeros (preprocessing shouldn't kill the signal) */
        bool all_zero = true;
        for (int i = 0; i < 56 * 56 * 3 && all_zero; ++i) {
            if (data[i] != 0.0f) all_zero = false;
        }
        RFDETR_ASSERT(!all_zero);

        std::free(data);
        rfdetr_image_free(img);
    }

    return 0;
}
