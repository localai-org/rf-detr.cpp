/* tests/test_custom_classes.cpp — verify the C-API correctly reports a
 * non-default num_classes for a fine-tuned-style checkpoint.
 *
 * Skips gracefully if /tmp/custom5.gguf is not present. To generate it:
 *
 *   .venv/bin/python scripts/build_custom_checkpoint.py \
 *       --output /tmp/custom5.pth --num-classes 5
 *   .venv/bin/python scripts/convert_rfdetr_to_gguf.py \
 *       --checkpoint /tmp/custom5.pth --output /tmp/custom5.gguf --dtype f32
 *
 * The default expected head_size is 5 (matches build_custom_checkpoint.py's
 * --num-classes default). Override via the RFDETR_CUSTOM_GGUF and
 * RFDETR_CUSTOM_NUM_CLASSES environment variables when testing other
 * fine-tuned checkpoints.
 */
#include "rfdetr.h"
#include "test_assert.hpp"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <sys/stat.h>

namespace {

bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

}  // namespace

int main() {
    const char* env_path = std::getenv("RFDETR_CUSTOM_GGUF");
    const std::string path = env_path ? env_path : "/tmp/custom5.gguf";

    const char* env_nc = std::getenv("RFDETR_CUSTOM_NUM_CLASSES");
    const uint32_t expected_num_classes = env_nc ? (uint32_t)std::atoi(env_nc) : 5u;

    if (!file_exists(path)) {
        std::fprintf(stderr,
                     "[test_custom_classes] SKIP: %s not present.\n"
                     "  To generate it:\n"
                     "    .venv/bin/python scripts/build_custom_checkpoint.py "
                     "--output /tmp/custom5.pth --num-classes 5\n"
                     "    .venv/bin/python scripts/convert_rfdetr_to_gguf.py "
                     "--checkpoint /tmp/custom5.pth --output /tmp/custom5.gguf "
                     "--dtype f32\n",
                     path.c_str());
        return 0;
    }

    rfdetr_params p{};
    p.model_path = path.c_str();
    p.n_threads  = 4;

    rfdetr_status st = RFDETR_OK;
    rfdetr_context* ctx = rfdetr_init(&p, &st);
    RFDETR_ASSERT(ctx != nullptr);
    RFDETR_ASSERT(st == RFDETR_OK);

    const uint32_t n_classes = rfdetr_context_num_classes(ctx);
    RFDETR_ASSERT_EQ_INT((int)n_classes, (int)expected_num_classes);

    /* Shared invariants: variant tag is non-empty, num_queries / n_tensors > 0. */
    const char* variant = rfdetr_context_variant(ctx);
    RFDETR_ASSERT(variant != nullptr && variant[0] != '\0');
    RFDETR_ASSERT(rfdetr_context_num_queries(ctx) > 0);
    RFDETR_ASSERT(rfdetr_context_n_tensors(ctx) > 0);

    std::fprintf(stderr,
                 "[test_custom_classes] OK %s -- variant=%s num_classes=%u "
                 "(expected %u)\n",
                 path.c_str(), variant, n_classes, expected_num_classes);

    rfdetr_free(ctx);
    return 0;
}
