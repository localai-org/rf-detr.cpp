#include "common.hpp"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace {

std::mutex          g_log_mutex;
rfdetr_log_cb       g_log_cb       = nullptr;
void*               g_log_user     = nullptr;

}  // namespace

extern "C" const char* rfdetr_status_str(rfdetr_status s) {
    switch (s) {
        case RFDETR_OK:                  return "ok";
        case RFDETR_ERR_INVALID_ARG:     return "invalid argument";
        case RFDETR_ERR_FILE_NOT_FOUND:  return "file not found";
        case RFDETR_ERR_IO:              return "i/o error";
        case RFDETR_ERR_OUT_OF_MEMORY:   return "out of memory";
        case RFDETR_ERR_DECODE:          return "image decode error";
        case RFDETR_ERR_MODEL_FORMAT:    return "model format error";
        case RFDETR_ERR_MODEL_LOAD:      return "model load error";
        case RFDETR_ERR_INFERENCE:       return "inference error";
        case RFDETR_ERR_NOT_IMPLEMENTED: return "not implemented";
    }
    return "unknown error";
}

extern "C" void rfdetr_set_log_callback(rfdetr_log_cb cb, void* user_data) {
    std::lock_guard<std::mutex> lk(g_log_mutex);
    g_log_cb   = cb;
    g_log_user = user_data;
}

void rfdetr_internal_log(rfdetr_log_level lvl, const char* msg) {
    rfdetr_log_cb cb;
    void* ud;
    {
        std::lock_guard<std::mutex> lk(g_log_mutex);
        cb = g_log_cb;
        ud = g_log_user;
    }
    if (cb && msg) {
        cb(lvl, msg, ud);
    }
}

void rfdetr_logf(rfdetr_log_level lvl, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    rfdetr_internal_log(lvl, buf);
}
