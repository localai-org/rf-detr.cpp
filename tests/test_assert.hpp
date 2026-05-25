#ifndef RFDETR_TEST_ASSERT_HPP
#define RFDETR_TEST_ASSERT_HPP

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define RFDETR_ASSERT(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "ASSERT FAILED: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

#define RFDETR_ASSERT_EQ_INT(a, b) do { \
    long long _a = (long long)(a), _b = (long long)(b); \
    if (_a != _b) { \
        std::fprintf(stderr, "ASSERT_EQ_INT FAILED: %s (%lld) != %s (%lld)\n  at %s:%d\n", \
                     #a, _a, #b, _b, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

#define RFDETR_ASSERT_STR_EQ(a, b) do { \
    const char* _a = (a); const char* _b = (b); \
    if (std::strcmp(_a, _b) != 0) { \
        std::fprintf(stderr, "ASSERT_STR_EQ FAILED: \"%s\" != \"%s\"\n  at %s:%d\n", \
                     _a, _b, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

#define RFDETR_ASSERT_NEAR(a, b, eps) do { \
    double _a = (double)(a), _b = (double)(b), _eps = (double)(eps); \
    if (std::fabs(_a - _b) > _eps) { \
        std::fprintf(stderr, "ASSERT_NEAR FAILED: |%s (%g) - %s (%g)| > %g\n  at %s:%d\n", \
                     #a, _a, #b, _b, _eps, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

#endif
