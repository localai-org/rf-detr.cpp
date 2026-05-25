#include "test_assert.hpp"
#include "rfdetr.h"
#include <string>
#include <vector>

static std::vector<std::pair<rfdetr_log_level, std::string>> captured;

static void capture_cb(rfdetr_log_level lvl, const char* msg, void* /*ud*/) {
    captured.emplace_back(lvl, std::string(msg));
}

int main() {
    // Status strings exist and are not empty for known codes
    RFDETR_ASSERT(rfdetr_status_str(RFDETR_OK) != nullptr);
    RFDETR_ASSERT_STR_EQ(rfdetr_status_str(RFDETR_OK), "ok");
    RFDETR_ASSERT(std::string(rfdetr_status_str(RFDETR_ERR_INVALID_ARG)).find("invalid") != std::string::npos);
    RFDETR_ASSERT(std::string(rfdetr_status_str(RFDETR_ERR_NOT_IMPLEMENTED)).find("not implemented") != std::string::npos);

    // Unknown code yields a stable "unknown" string (not a crash)
    RFDETR_ASSERT(rfdetr_status_str((rfdetr_status)12345) != nullptr);

    // Log callback receives messages it was sent
    rfdetr_set_log_callback(capture_cb, nullptr);

    // Internal use: emit a message via the C++ helper (declared in common.hpp)
    extern void rfdetr_internal_log(rfdetr_log_level, const char*);
    rfdetr_internal_log(RFDETR_LOG_INFO, "hello");
    rfdetr_internal_log(RFDETR_LOG_WARN, "world");

    RFDETR_ASSERT_EQ_INT(captured.size(), 2);
    RFDETR_ASSERT(captured[0].first == RFDETR_LOG_INFO);
    RFDETR_ASSERT_STR_EQ(captured[0].second.c_str(), "hello");
    RFDETR_ASSERT(captured[1].first == RFDETR_LOG_WARN);
    RFDETR_ASSERT_STR_EQ(captured[1].second.c_str(), "world");

    // Unsetting the callback (nullptr) must not crash
    rfdetr_set_log_callback(nullptr, nullptr);
    rfdetr_internal_log(RFDETR_LOG_INFO, "ignored");
    RFDETR_ASSERT_EQ_INT(captured.size(), 2);  // unchanged

    return 0;
}
