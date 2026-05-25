#include "test_assert.hpp"
#include "rfdetr.h"
#include <string>

int main() {
    const std::string fixtures = RFDETR_TEST_FIXTURES;
    const std::string model_path = fixtures + "/model_base.gguf";

    // Successful init
    rfdetr_params p{};
    p.model_path = model_path.c_str();
    p.n_threads  = 1;

    rfdetr_status st;
    rfdetr_context* ctx = rfdetr_init(&p, &st);
    RFDETR_ASSERT(ctx != nullptr);
    RFDETR_ASSERT_EQ_INT(st, RFDETR_OK);

    rfdetr_free(ctx);

    // Bad path -> nullptr, status set
    rfdetr_params p2{};
    p2.model_path = "/no/such/model.gguf";
    p2.n_threads  = 1;
    rfdetr_status st2;
    rfdetr_context* ctx2 = rfdetr_init(&p2, &st2);
    RFDETR_ASSERT(ctx2 == nullptr);
    RFDETR_ASSERT(st2 != RFDETR_OK);

    // Null params -> nullptr, RFDETR_ERR_INVALID_ARG
    rfdetr_status st3;
    rfdetr_context* ctx3 = rfdetr_init(nullptr, &st3);
    RFDETR_ASSERT(ctx3 == nullptr);
    RFDETR_ASSERT_EQ_INT(st3, RFDETR_ERR_INVALID_ARG);

    // rfdetr_free(nullptr) must not crash
    rfdetr_free(nullptr);

    // rfdetr_detect still returns NOT_IMPLEMENTED (Plan 3 wires inference)
    rfdetr_context* ctx4 = rfdetr_init(&p, &st);
    rfdetr_image*   img  = rfdetr_image_load_file((fixtures + "/cats.png").c_str(), nullptr);
    RFDETR_ASSERT(ctx4 && img);
    rfdetr_detect_params dp{};
    dp.threshold = 0.5f; dp.top_k = 300;
    rfdetr_detection* dets = nullptr;
    size_t n = 0;
    rfdetr_status det_st = rfdetr_detect(ctx4, img, &dp, &dets, &n);
    RFDETR_ASSERT_EQ_INT(det_st, RFDETR_ERR_NOT_IMPLEMENTED);
    RFDETR_ASSERT_EQ_INT(n, 0);
    RFDETR_ASSERT(dets == nullptr);

    rfdetr_image_free(img);
    rfdetr_free(ctx4);
    return 0;
}
