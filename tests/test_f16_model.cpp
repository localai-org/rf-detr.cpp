/* tests/test_f16_model.cpp - F16 model sanity check.
 *
 * Regression test for a read-overrun bug where the loader assumed
 * backbone.pos_embed was F32 and read dim*n_train_tokens*sizeof(float)
 * bytes from the tensor — which overran the buffer for F16 tensors and
 * tripped ggml's "tensor read out of bounds" assert. See
 * src/model_loader.cpp's pos_embed handling for the fix.
 *
 * Loads models/rfdetr-base-f16.gguf (if present) and verifies:
 *   - Format version + metadata round-trip
 *   - At least some tensors are F16 (matmul-eligible weights), and lookup
 *     tensors (cls_token, pos_embed, biases, norms) stayed F32 so mixed-
 *     dtype ops like ggml_concat(cls, activation) don't error at graph build
 *   - The model loads + realizes weights without aborting (this is the
 *     actual regression — pre-fix it crashed inside model_realize_weights)
 *
 * Skips gracefully if the model isn't present so a fresh clone passes ctest.
 * Generate with:
 *   .venv/bin/python scripts/convert_rfdetr_to_gguf.py --variant base \
 *       --dtype f16 --output models/rfdetr-base-f16.gguf
 */
#include "test_assert.hpp"
#include "rfdetr.h"
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
                             "/models/rfdetr-base-f16.gguf";

    if (!file_exists(path)) {
        std::fprintf(stderr,
            "[test_f16_model] SKIPPED: %s not present. Generate with\n"
            "  .venv/bin/python scripts/convert_rfdetr_to_gguf.py "
            "--variant base --dtype f16 "
            "--output models/rfdetr-base-f16.gguf\n",
            path.c_str());
        return 0;
    }

    /* --- Step 1: GGUF-level structural checks. --- */
    ggml_context* ctx = nullptr;
    gguf_init_params init_p{};
    init_p.no_alloc = true;
    init_p.ctx      = &ctx;
    gguf_context* g = gguf_init_from_file(path.c_str(), init_p);
    RFDETR_ASSERT(g != nullptr);
    RFDETR_ASSERT(ctx != nullptr);

    /* Metadata. */
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

    /* Tensor type accounting. */
    const int64_t nt = gguf_get_n_tensors(g);
    RFDETR_ASSERT(nt > 0);

    int n_f16 = 0;
    int n_f32 = 0;
    int n_other = 0;
    for (int64_t i = 0; i < nt; ++i) {
        const char* name = gguf_get_tensor_name(g, i);
        ggml_tensor* t = ggml_get_tensor(ctx, name);
        RFDETR_ASSERT(t != nullptr);
        if (t->type == GGML_TYPE_F16)       ++n_f16;
        else if (t->type == GGML_TYPE_F32)  ++n_f32;
        else                                ++n_other;
    }

    std::fprintf(stderr,
        "[test_f16_model] tensors: F16=%d  F32=%d  other=%d  total=%lld\n",
        n_f16, n_f32, n_other, (long long)nt);

    /* Same matmul-only heuristic as the quantized variants: at least 100
     * weights cast to F16, plenty staying F32 (biases, norms, cls_token,
     * pos_embed, conv kernels). */
    RFDETR_ASSERT(n_f16 > 100);
    RFDETR_ASSERT(n_f32 > 100);
    RFDETR_ASSERT_EQ_INT(n_other, 0);

    /* Spot-check a known linear weight is F16 (e.g. backbone attn.q). */
    {
        ggml_tensor* t = ggml_get_tensor(ctx, "backbone.blocks.0.attn.q.weight");
        RFDETR_ASSERT(t != nullptr);
        RFDETR_ASSERT(t->type == GGML_TYPE_F16);
    }
    /* Spot-checks: tensors used in mixed-dtype graph ops must stay F32. */
    {
        /* cls_token gets broadcast and concatted with F32 activations. */
        ggml_tensor* t = ggml_get_tensor(ctx, "backbone.cls_token");
        RFDETR_ASSERT(t != nullptr);
        RFDETR_ASSERT(t->type == GGML_TYPE_F32);
    }
    {
        /* pos_embed is read on CPU for bicubic resampling — this is the
         * actual regression site. Pre-fix the loader would crash on F16
         * even with this F32-stay invariant absent. The fix in
         * model_loader.cpp also tolerates an F16 pos_embed via
         * ggml_fp16_to_fp32_row; we additionally keep it F32 here so the
         * resample path stays on its fast (memcpy) branch. */
        ggml_tensor* t = ggml_get_tensor(ctx, "backbone.pos_embed");
        RFDETR_ASSERT(t != nullptr);
        RFDETR_ASSERT(t->type == GGML_TYPE_F32);
    }
    {
        /* Biases stay F32 (ggml_add with F32 activation needs matching type). */
        ggml_tensor* t = ggml_get_tensor(ctx, "backbone.blocks.0.attn.q.bias");
        RFDETR_ASSERT(t != nullptr);
        RFDETR_ASSERT(t->type == GGML_TYPE_F32);
    }

    gguf_free(g);
    ggml_free(ctx);

    /* --- Step 2: full C-API load. This exercises model_realize_weights —
     * the pre-fix crash site for F16 pos_embed. --- */
    rfdetr_params p{};
    p.model_path = path.c_str();
    p.n_threads  = 1;
    rfdetr_status st = RFDETR_OK;
    rfdetr_context* rc = rfdetr_init(&p, &st);
    RFDETR_ASSERT_EQ_INT((int)st, (int)RFDETR_OK);
    RFDETR_ASSERT(rc != nullptr);

    RFDETR_ASSERT_EQ_INT((int)rfdetr_context_num_classes(rc), 91);
    RFDETR_ASSERT_EQ_INT((int)rfdetr_context_num_queries(rc), 300);
    RFDETR_ASSERT_STR_EQ(rfdetr_context_variant(rc), "base");

    rfdetr_free(rc);

    std::fprintf(stderr, "[test_f16_model] PASS\n");
    return 0;
}
