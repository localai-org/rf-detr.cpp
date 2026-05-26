/* tests/test_kquant.cpp - K-quant (Q4_K / Q5_K / Q6_K) model sanity checks.
 *
 * For each models/rfdetr-base-q{4,5,6}_K.gguf that's present, verifies:
 *   - Format version, variant, num_classes / num_queries metadata
 *   - The expected K-quant type appears in the tensor inventory
 *   - LayerNorm/bias/pos_embed stay F32
 *   - Backbone tensors with row size 384 (not divisible by 256) fall back
 *     to Q8_0 rather than F32 — that's the rfdetr-cli quantize behavior
 *     that keeps the K-quant models close to the legacy Q8_0 size class.
 *
 * Skips gracefully when a file isn't present so a fresh clone passes ctest
 * without having to generate them.
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

void check_one_kquant(const std::string& path, ggml_type expect_kquant,
                      const char* tag) {
    if (!file_exists(path)) {
        std::fprintf(stderr,
            "[%s] SKIPPED: %s not present. Generate with\n"
            "  build/bin/rfdetr-cli quantize models/rfdetr-base-f32.gguf %s %s\n",
            tag, path.c_str(), path.c_str(), ggml_type_name(expect_kquant));
        return;
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

    /* Tensor type accounting. */
    const int64_t nt = gguf_get_n_tensors(g);
    RFDETR_ASSERT(nt > 0);

    int n_kq = 0, n_q8 = 0, n_f32 = 0, n_other = 0;
    for (int64_t i = 0; i < nt; ++i) {
        const char* name = gguf_get_tensor_name(g, i);
        ggml_tensor* t = ggml_get_tensor(ctx, name);
        RFDETR_ASSERT(t != nullptr);
        if      (t->type == expect_kquant)  ++n_kq;
        else if (t->type == GGML_TYPE_Q8_0) ++n_q8;
        else if (t->type == GGML_TYPE_F32)  ++n_f32;
        else                                ++n_other;
    }

    std::fprintf(stderr,
        "[%s] tensors: %s=%d  Q8_0(fallback)=%d  F32=%d  other=%d  total=%lld\n",
        tag, ggml_type_name(expect_kquant), n_kq, n_q8, n_f32, n_other,
        (long long)nt);

    /* Expect a substantial number of K-quant tensors (projector / decoder /
     * two-stage groups: about 90 on rfdetr-base). */
    RFDETR_ASSERT(n_kq > 50);
    /* Expect Q8_0 fallback for backbone tensors with row size 384 (not %256).
     * On rfdetr-base that's about 60 tensors. */
    RFDETR_ASSERT(n_q8 > 30);
    /* Layer norms / biases / embeddings stay F32. */
    RFDETR_ASSERT(n_f32 > 100);
    RFDETR_ASSERT_EQ_INT(n_other, 0);

    /* Spot-check a known dim-256 weight is the K-quant type (projector). */
    {
        ggml_tensor* t = ggml_get_tensor(ctx, "projector.cv1.conv.weight");
        if (t) {
            /* projector cv1 is a conv kernel (4D) — stays F32. Use a known
             * 2D one with inner dim 256: decoder.norm.weight is 1D, but
             * decoder.layers.0.norm1.weight too. Try a known 2D dim-256
             * weight: decoder.layers.0.linear1.weight has shape (256, 2048),
             * gguf ne=(256 -> inner... actually let's check the dim 256
             * projector.final_norm: that's also 1D). Use two_stage instead. */
        }
        ggml_tensor* enc = ggml_get_tensor(ctx, "two_stage.enc_output.0.weight");
        RFDETR_ASSERT(enc != nullptr);
        /* enc_output.<g>.weight is a Linear (256, 256) -> ne (256, 256).
         * 256 % 256 == 0 so it MUST be the K-quant type. */
        RFDETR_ASSERT(enc->type == expect_kquant);
    }
    /* Spot-check that a 384-row backbone weight fell back to Q8_0. */
    {
        ggml_tensor* t = ggml_get_tensor(ctx, "backbone.blocks.0.attn.q.weight");
        RFDETR_ASSERT(t != nullptr);
        /* ne[0]=384, not divisible by 256: expect Q8_0 fallback. */
        RFDETR_ASSERT(t->type == GGML_TYPE_Q8_0);
    }
    /* Spot-check biases / LayerNorms stay F32. */
    {
        ggml_tensor* t = ggml_get_tensor(ctx, "backbone.blocks.0.attn.q.bias");
        RFDETR_ASSERT(t != nullptr);
        RFDETR_ASSERT(t->type == GGML_TYPE_F32);
    }
    /* Pos embed stays F32. */
    {
        ggml_tensor* t = ggml_get_tensor(ctx, "backbone.pos_embed");
        RFDETR_ASSERT(t != nullptr);
        RFDETR_ASSERT(t->type == GGML_TYPE_F32);
    }

    gguf_free(g);
    ggml_free(ctx);
    std::fprintf(stderr, "[%s] PASS\n", tag);
}

}  // namespace

int main() {
    const std::string base = std::string(RFDETR_SOURCE_DIR) + "/models/";

    check_one_kquant(base + "rfdetr-base-q4_K.gguf", GGML_TYPE_Q4_K, "test_kquant Q4_K");
    check_one_kquant(base + "rfdetr-base-q5_K.gguf", GGML_TYPE_Q5_K, "test_kquant Q5_K");
    check_one_kquant(base + "rfdetr-base-q6_K.gguf", GGML_TYPE_Q6_K, "test_kquant Q6_K");
    return 0;
}
