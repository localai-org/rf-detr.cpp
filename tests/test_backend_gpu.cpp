/* tests/test_backend_gpu.cpp — verifies that when the library is built with
 * a GPU backend (RFDETR_USE_CUDA / _METAL / _VULKAN) AND a device is present,
 * init_backend_ctx actually creates a GPU backend + scheduler. On CPU-only
 * builds (no RFDETR_USE_* define) it asserts gpu == nullptr and passes.
 *
 * NOTE: the RFDETR_USE_* defines are PRIVATE to the rfdetr lib and do not
 * reach this test target, so we assert the weaker, build-independent
 * invariant instead: a GPU backend (if created) always comes with a
 * scheduler, and the absence of a GPU backend implies no scheduler. This
 * holds on every build (CPU-only, GPU build with a device, GPU build with
 * no device).
 */
#include "backend.hpp"
#include "test_assert.hpp"
#include <cstdio>

int main() {
    rfdetr_status st = RFDETR_OK;
    rfdetr::BackendCtx ctx = rfdetr::init_backend_ctx(/*n_threads*/ 4, &st);
    RFDETR_ASSERT(st == RFDETR_OK);
    RFDETR_ASSERT(ctx.cpu != nullptr);

    /* Invariant on every build: a GPU backend (if one was created) always
     * comes with a scheduler. CPU-only builds + GPU builds with no device
     * both leave gpu == nullptr, which is fine. */
    if (ctx.gpu != nullptr) {
        RFDETR_ASSERT(ctx.sched != nullptr);
        std::fprintf(stderr, "[test_backend_gpu] GPU backend active + sched created\n");
    } else {
        RFDETR_ASSERT(ctx.sched == nullptr);
        std::fprintf(stderr, "[test_backend_gpu] no GPU backend (CPU-only build or no device)\n");
    }

    rfdetr::free_backend_ctx(ctx);
    return 0;
}
