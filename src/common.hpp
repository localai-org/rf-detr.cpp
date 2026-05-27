#ifndef RFDETR_COMMON_HPP
#define RFDETR_COMMON_HPP

#include "rfdetr.h"

/* Internal helper used by every source file to emit a log message
 * via the registered callback. No-op if no callback is set.
 * Uses C++ linkage so it can be re-declared with `extern void ...`
 * in tests without needing to include this header. */
void rfdetr_internal_log(rfdetr_log_level lvl, const char* msg);

/* printf-style wrapper. Builds the string then dispatches. */
void rfdetr_logf(rfdetr_log_level lvl, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));

#endif
