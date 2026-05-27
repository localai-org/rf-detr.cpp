/* tests/test_variants.cpp - verify each RF-DETR detection variant loads
 * through the C-API and reports the expected metadata.
 *
 * Per-variant we check (when the GGUF is present at the standard models/
 * path):
 *   - rfdetr_init succeeds
 *   - variant string round-trips through the loader
 *   - image_size matches the per-variant constant
 *   - num_classes / num_queries are 91 / 300 (shared across all 5)
 *   - n_tensors is non-zero
 *
 * Skips per-variant gracefully if its GGUF isn't present, so a fresh clone
 * passes ctest without needing to convert anything first. To generate the
 * variant GGUFs:
 *
 *   scripts/convert_all_variants.sh
 */
#include "rfdetr.h"
#include "test_assert.hpp"

#include <sys/stat.h>
#include <cstdio>
#include <cstdint>
#include <string>

namespace {

bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

struct VariantSpec {
    const char* name;
    uint32_t    image_size;
};

}  // namespace

int main() {
    /* Per-variant expected image_size. All five share num_classes=91,
     * num_queries=300, group_detr=13. */
    const VariantSpec variants[] = {
        {"nano",   384},
        {"small",  512},
        {"base",   560},
        {"medium", 576},
        {"large",  704},
    };

    const std::string root = std::string(RFDETR_SOURCE_DIR) + "/models/";
    int checked = 0;
    int skipped = 0;

    for (const auto& v : variants) {
        const std::string path = root + "rfdetr-" + v.name + "-f32.gguf";
        if (!file_exists(path)) {
            std::fprintf(stderr, "[test_variants] SKIP %s (not present)\n",
                         path.c_str());
            ++skipped;
            continue;
        }

        rfdetr_params p{};
        p.model_path = path.c_str();
        p.n_threads  = 4;

        rfdetr_status st = RFDETR_OK;
        rfdetr_context* ctx = rfdetr_init(&p, &st);
        RFDETR_ASSERT(ctx != nullptr);
        RFDETR_ASSERT(st == RFDETR_OK);

        const char* variant = rfdetr_context_variant(ctx);
        RFDETR_ASSERT(variant != nullptr);
        RFDETR_ASSERT_STR_EQ(variant, v.name);

        const uint32_t img = rfdetr_context_image_size(ctx);
        RFDETR_ASSERT_EQ_INT((int)img, (int)v.image_size);

        const uint32_t n_classes = rfdetr_context_num_classes(ctx);
        RFDETR_ASSERT_EQ_INT((int)n_classes, 91);

        const uint32_t n_queries = rfdetr_context_num_queries(ctx);
        RFDETR_ASSERT_EQ_INT((int)n_queries, 300);

        const size_t n_tensors = rfdetr_context_n_tensors(ctx);
        RFDETR_ASSERT(n_tensors > 0);

        std::fprintf(stderr,
                     "[test_variants] OK %s -- variant=%s image_size=%u "
                     "n_classes=%u n_queries=%u n_tensors=%zu\n",
                     path.c_str(), variant, img, n_classes, n_queries,
                     n_tensors);

        rfdetr_free(ctx);
        ++checked;
    }

    if (checked == 0) {
        std::fprintf(stderr,
                     "[test_variants] no variant GGUFs found -- generate "
                     "via scripts/convert_all_variants.sh\n");
    }
    std::fprintf(stderr,
                 "[test_variants] PASS (%d checked, %d skipped)\n",
                 checked, skipped);
    return 0;
}
