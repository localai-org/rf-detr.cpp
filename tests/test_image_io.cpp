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

    /* ---- rfdetr_image_from_rgb_buffer: wrap a raw RGB buffer ---- */
    {
        /* Synthesize a 4x3 RGB buffer with known per-pixel values. */
        const int W = 4, H = 3;
        std::vector<uint8_t> rgb;
        rgb.reserve((size_t)W * H * 3);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                rgb.push_back((uint8_t)(x * 10));        /* R */
                rgb.push_back((uint8_t)(y * 20));        /* G */
                rgb.push_back((uint8_t)(x + y * W));     /* B */
            }
        }

        rfdetr_status st5;
        rfdetr_image* img = rfdetr_image_from_rgb_buffer(rgb.data(), W, H, &st5);
        RFDETR_ASSERT(img != nullptr);
        RFDETR_ASSERT_EQ_INT(st5, RFDETR_OK);
        RFDETR_ASSERT_EQ_INT(rfdetr_image_width(img),  W);
        RFDETR_ASSERT_EQ_INT(rfdetr_image_height(img), H);

        /* The wrapper must COPY (mutating the source must not affect the image). */
        const uint8_t* px = rfdetr_image_rgb_data(img);
        RFDETR_ASSERT(px != nullptr);
        rgb[0] = 0xAB;  /* clobber the source */
        RFDETR_ASSERT_EQ_INT(px[0], 0);   /* still the original 0 */
        RFDETR_ASSERT_EQ_INT(px[1], 0);
        RFDETR_ASSERT_EQ_INT(px[2], 0);
        /* Spot check the (x=2, y=1) pixel: r=20, g=20, b=6 */
        const int i_offset = (1 * W + 2) * 3;
        RFDETR_ASSERT_EQ_INT(px[i_offset + 0], 20);
        RFDETR_ASSERT_EQ_INT(px[i_offset + 1], 20);
        RFDETR_ASSERT_EQ_INT(px[i_offset + 2], 6);

        rfdetr_image_free(img);

        /* Invalid args -> nullptr + RFDETR_ERR_INVALID_ARG. */
        rfdetr_status st_bad;
        RFDETR_ASSERT(rfdetr_image_from_rgb_buffer(nullptr, 4, 4, &st_bad) == nullptr);
        RFDETR_ASSERT_EQ_INT(st_bad, RFDETR_ERR_INVALID_ARG);
        std::vector<uint8_t> dummy(12, 0);
        RFDETR_ASSERT(rfdetr_image_from_rgb_buffer(dummy.data(), 0, 4, &st_bad) == nullptr);
        RFDETR_ASSERT_EQ_INT(st_bad, RFDETR_ERR_INVALID_ARG);
        RFDETR_ASSERT(rfdetr_image_from_rgb_buffer(dummy.data(), 4, -1, &st_bad) == nullptr);
        RFDETR_ASSERT_EQ_INT(st_bad, RFDETR_ERR_INVALID_ARG);
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
        rfdetr_status pp_st = rfdetr_preprocess(
            img, 56, 56, mean, std_, false, &data, &w, &h);
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

    /* RF-DETR 1.9 preprocessing: float bilinear, align_corners=false,
     * antialias=false, and no intermediate uint8 rounding. */
    {
        const uint8_t rgb[] = {
              0,  10,  20,   100, 110, 120,
            200, 210, 220,   255, 250, 245,
        };
        rfdetr_status st_;
        rfdetr_image* img_ = rfdetr_image_from_rgb_buffer(rgb, 2, 2, &st_);
        RFDETR_ASSERT(img_ != nullptr);
        RFDETR_ASSERT_EQ_INT(st_, RFDETR_OK);

        const float mean0[3] = {0.0f, 0.0f, 0.0f};
        const float std1[3]  = {1.0f, 1.0f, 1.0f};
        float* data = nullptr;
        int w = 0, h = 0;
        rfdetr_status pp_st = rfdetr_preprocess(
            img_, 3, 3, mean0, std1, true, &data, &w, &h);
        RFDETR_ASSERT_EQ_INT(pp_st, RFDETR_OK);
        RFDETR_ASSERT_EQ_INT(w, 3);
        RFDETR_ASSERT_EQ_INT(h, 3);

        /* Upscaling 2x2 -> 3x3 with half-pixel centers puts the output
         * center exactly between all four source pixels, so it is their
         * unweighted mean. Corners land outside the source centers and clamp
         * to the nearest source pixel. Channels are planar (3x3 = 9 floats
         * each), so the centers are at offsets 4, 13, 22. */
        RFDETR_ASSERT_NEAR(data[4],  138.75f / 255.0f, 1e-6);
        RFDETR_ASSERT_NEAR(data[13], 145.00f / 255.0f, 1e-6);
        RFDETR_ASSERT_NEAR(data[22], 151.25f / 255.0f, 1e-6);
        /* Top-middle: mean of the two top pixels, red channel. */
        RFDETR_ASSERT_NEAR(data[1],   50.00f / 255.0f, 1e-6);
        /* Bottom-right corner clamps to source (1,1), red channel = 255. */
        RFDETR_ASSERT_NEAR(data[8],  255.00f / 255.0f, 1e-6);

        std::free(data);
        rfdetr_image_free(img_);
    }

    return 0;
}
