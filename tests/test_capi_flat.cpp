#include "test_assert.hpp"
#include "rfdetr_capi.h"
#include <cstdlib>
#include <cstring>
#include <string>

int main() {
    const std::string fixtures = RFDETR_TEST_FIXTURES;
    const std::string model_path = fixtures + "/model_base.gguf";

    // Load
    rfdetr_handle_t h = 0;
    int rc = rfdetr_capi_load(model_path.c_str(), 1, &h);
    RFDETR_ASSERT_EQ_INT(rc, 0);
    RFDETR_ASSERT(h != 0);

    // Detect — returns JSON envelope. With the synthesized random-weight fixture,
    // scores rarely exceed the 0.5 threshold so detections is typically empty,
    // but the flat ABI's response shape (status + image + detections) is the
    // same either way.
    char* json = nullptr;
    int dr = rfdetr_capi_detect_path(h, (fixtures + "/cats.png").c_str(),
                                     0.5f, 300, &json);
    RFDETR_ASSERT_EQ_INT(dr, 0);   /* 0 = handled */
    RFDETR_ASSERT(json != nullptr);
    std::string body(json);
    rfdetr_capi_free_string(json);

    RFDETR_ASSERT(body.find("\"status\"")     != std::string::npos);
    RFDETR_ASSERT(body.find("\"detections\"") != std::string::npos);
    RFDETR_ASSERT(body.find("\"width\"")      != std::string::npos);
    RFDETR_ASSERT(body.find("\"height\"")     != std::string::npos);

    // Bad image path -> non-zero rc
    char* json2 = nullptr;
    int dr2 = rfdetr_capi_detect_path(h, "/no/such/image.png", 0.5f, 300, &json2);
    RFDETR_ASSERT(dr2 != 0);
    if (json2) rfdetr_capi_free_string(json2);

    // Unload
    int ur = rfdetr_capi_unload(h);
    RFDETR_ASSERT_EQ_INT(ur, 0);

    // Re-unload zero handle is a no-op (0)
    int ur2 = rfdetr_capi_unload(0);
    RFDETR_ASSERT_EQ_INT(ur2, 0);

    // Bad path on load
    rfdetr_handle_t h2 = 0;
    int br = rfdetr_capi_load("/no/such/file.gguf", 1, &h2);
    RFDETR_ASSERT(br != 0);
    RFDETR_ASSERT(h2 == 0);

    return 0;
}
