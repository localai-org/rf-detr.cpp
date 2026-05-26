/* tests/test_q8_quant.cpp - Q8_0 quantized model sanity check.
 *
 * Loads models/rfdetr-base-q8_0.gguf (if present) and verifies:
 *   - Format version, variant, num_classes / num_queries metadata
 *   - At least one tensor is actually stored as Q8_0
 *   - All expected tensors still resolve (mixed-precision is fine)
 *
 * Skips gracefully if the model isn't present so a fresh clone passes ctest.
 * Parity vs torch baseline is intentionally NOT enforced — Q8_0 drift exceeds
 * the F32 parity tolerance by design; the end-to-end detection comparison
 * lives in scripts/CLI usage, not in this test.
 */
#include "test_assert.hpp"
#include "ggml.h"
#include "gguf.h"

#include <sys/stat.h>
#include <cstdio>
#include <cstring>
#include <string>

namespace {
bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}
}  // namespace

int main() {
    const std::string path = std::string(RFDETR_SOURCE_DIR) +
                             "/models/rfdetr-base-q8_0.gguf";

    if (!file_exists(path)) {
        std::fprintf(stderr,
            "[test_q8_quant] SKIPPED: %s not present. Generate with\n"
            "  .venv/bin/python scripts/convert_rfdetr_to_gguf.py "
            "--dtype q8_0 --output models/rfdetr-base-q8_0.gguf\n",
            path.c_str());
        return 0;
    }

    ggml_context* ctx = nullptr;
    gguf_init_params init_p{};
    init_p.no_alloc = true;
    init_p.ctx      = &ctx;
    gguf_context* g = gguf_init_from_file(path.c_str(), init_p);
    RFDETR_ASSERT(g != nullptr);
    RFDETR_ASSERT(ctx != nullptr);

    /* Metadata sanity. */
    {
        int64_t kid = gguf_find_key(g, "rfdetr.format.version");
        RFDETR_ASSERT(kid >= 0);
        RFDETR_ASSERT_STR_EQ(gguf_get_val_str(g, kid), "2");
    }
    {
        int64_t kid = gguf_find_key(g, "rfdetr.variant");
        RFDETR_ASSERT(kid >= 0);
        RFDETR_ASSERT_STR_EQ(gguf_get_val_str(g, kid), "base");
    }
    {
        int64_t kid = gguf_find_key(g, "rfdetr.num_classes");
        RFDETR_ASSERT(kid >= 0);
        RFDETR_ASSERT_EQ_INT((int)gguf_get_val_u32(g, kid), 91);
    }
    {
        int64_t kid = gguf_find_key(g, "rfdetr.num_queries");
        RFDETR_ASSERT(kid >= 0);
        RFDETR_ASSERT_EQ_INT((int)gguf_get_val_u32(g, kid), 300);
    }

    /* Tensor type accounting: at least one Q8_0, several stay F32. */
    const int64_t nt = gguf_get_n_tensors(g);
    RFDETR_ASSERT(nt > 0);

    int n_q8_0 = 0;
    int n_f32  = 0;
    int n_other = 0;
    for (int64_t i = 0; i < nt; ++i) {
        const char* name = gguf_get_tensor_name(g, i);
        ggml_tensor* t = ggml_get_tensor(ctx, name);
        RFDETR_ASSERT(t != nullptr);
        if (t->type == GGML_TYPE_Q8_0)      ++n_q8_0;
        else if (t->type == GGML_TYPE_F32)  ++n_f32;
        else                                ++n_other;
    }

    std::fprintf(stderr,
        "[test_q8_quant] tensors: Q8_0=%d  F32=%d  other=%d  total=%lld\n",
        n_q8_0, n_f32, n_other, (long long)nt);

    /* We expect a substantial number quantized (linear weights dominate). */
    RFDETR_ASSERT(n_q8_0 > 100);
    RFDETR_ASSERT(n_f32  > 100);
    RFDETR_ASSERT_EQ_INT(n_other, 0);

    /* Spot-check that a known linear weight is Q8_0 (e.g. backbone attn.q). */
    {
        ggml_tensor* t = ggml_get_tensor(ctx, "backbone.blocks.0.attn.q.weight");
        RFDETR_ASSERT(t != nullptr);
        RFDETR_ASSERT(t->type == GGML_TYPE_Q8_0);
    }
    /* Spot-check that a 1D bias stays F32. */
    {
        ggml_tensor* t = ggml_get_tensor(ctx, "backbone.blocks.0.attn.q.bias");
        RFDETR_ASSERT(t != nullptr);
        RFDETR_ASSERT(t->type == GGML_TYPE_F32);
    }
    /* Spot-check pos_embed stays F32 (it's bicubic-interpolated at load). */
    {
        ggml_tensor* t = ggml_get_tensor(ctx, "backbone.pos_embed");
        RFDETR_ASSERT(t != nullptr);
        RFDETR_ASSERT(t->type == GGML_TYPE_F32);
    }

    gguf_free(g);
    ggml_free(ctx);

    std::fprintf(stderr, "[test_q8_quant] PASS\n");
    return 0;
}
