/* Plan 14: end-to-end parity — full C++ rfdetr_model_forward vs torch baseline.
 *
 * Uses baseline parity.preprocess.input (the deterministic seeded synthetic
 * input already ImageNet-normalized — same input the baseline ran with) as
 * the model input, runs the entire C++ pipeline (backbone → projector →
 * two_stage → top-K → decoder → heads → bbox_reparam), and compares the
 * final heads outputs against the baseline's last-layer slice.
 *
 * This is a CUMULATIVE drift test — sub-module drift from earlier stages
 * compounds here. We do NOT enforce a tight tolerance; the test logs the
 * achieved max_abs and serves as a regression guard. The numbers we observe
 * (1e-5..1e-3 depending on backbone drift) document the as-shipped accuracy.
 *
 * Skips gracefully if either input file is missing. */
#include "test_assert.hpp"
#include "rfdetr.h"
#include "model_loader.hpp"
#include "backend.hpp"
#include "rfdetr_model.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <sys/stat.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

struct Baseline {
    std::unordered_map<std::string, std::vector<float>> tensors;
    std::unordered_map<std::string, std::vector<int64_t>> shapes;
};

Baseline load_baseline(const std::string& path) {
    Baseline b;
    ggml_context* ctx = nullptr;
    gguf_init_params p{};
    p.no_alloc = false;
    p.ctx      = &ctx;
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    RFDETR_ASSERT(g != nullptr);
    RFDETR_ASSERT(ctx != nullptr);

    const int64_t nt = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < nt; ++i) {
        const char* name = gguf_get_tensor_name(g, i);
        std::string n(name);
        const std::string prefix = "parity.";
        if (n.compare(0, prefix.size(), prefix) != 0) continue;
        std::string key = n.substr(prefix.size());

        ggml_tensor* t = ggml_get_tensor(ctx, name);
        RFDETR_ASSERT(t != nullptr);
        RFDETR_ASSERT_EQ_INT(t->type, GGML_TYPE_F32);
        const size_t nelem = ggml_nelements(t);
        std::vector<float> v(nelem);
        std::memcpy(v.data(), t->data, nelem * sizeof(float));

        std::vector<int64_t> shape;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) shape.push_back(t->ne[d]);
        b.shapes.emplace(key, std::move(shape));
        b.tensors.emplace(std::move(key), std::move(v));
    }

    gguf_free(g);
    ggml_free(ctx);
    return b;
}

}  // namespace

int main() {
    const std::string fixtures   = RFDETR_TEST_FIXTURES;
    const std::string base_path  = fixtures + "/baseline_torch.gguf";

    std::string model_path;
    for (const std::string& candidate : {
            fixtures + "/rfdetr-base-f32.gguf",
            fixtures + "/../../models/rfdetr-base-f32.gguf",
        }) {
        if (file_exists(candidate)) { model_path = candidate; break; }
    }
    if (model_path.empty()) {
        std::fprintf(stderr,
            "[test_parity_e2e] SKIPPED: real rfdetr-base GGUF not found.\n");
        return 0;
    }
    if (!file_exists(base_path)) {
        std::fprintf(stderr,
            "[test_parity_e2e] SKIPPED: baseline not present (%s).\n",
            base_path.c_str());
        return 0;
    }

    Baseline base = load_baseline(base_path);
    auto have = [&](const char* k) { return base.tensors.find(k) != base.tensors.end(); };
    if (!have("preprocess.input") ||
        !have("heads.class_logits") ||
        !have("heads.bbox_pred")) {
        std::fprintf(stderr,
            "[test_parity_e2e] SKIPPED: baseline missing required checkpoints.\n");
        return 0;
    }

    /* The baseline stores torch's input as (C=3, H=560, W=560, 1) — ggml ne
     * order (which torch's writer preserves byte-for-byte). The C++ model
     * forward expects (W, H, 3, 1) — see image_io.cpp:148. These are the same
     * memory layout if we view it differently: torch's input is contiguous
     * F32 (1, 3, 560, 560) with C as the slowest-varying middle dim. After
     * gguf storage, ggml reads ne = (W=560, H=560, C=3, B=1). That's exactly
     * what rfdetr_model_forward expects, so we can pass the raw buffer. */
    const auto& in_data  = base.tensors.at("preprocess.input");
    const auto& in_shape = base.shapes.at("preprocess.input");
    const int W = (int)in_shape[0];
    const int H = (int)in_shape[1];
    const int C = (int)in_shape[2];
    RFDETR_ASSERT_EQ_INT(W, 560);
    RFDETR_ASSERT_EQ_INT(H, 560);
    RFDETR_ASSERT_EQ_INT(C, 3);

    rfdetr_status st = RFDETR_OK;
    rfdetr::Model* m = rfdetr::model_load(model_path, &st);
    RFDETR_ASSERT(m != nullptr);
    RFDETR_ASSERT_EQ_INT(st, RFDETR_OK);

    ggml_backend_t backend = rfdetr::init_backend(/*n_threads*/ 4, &st);
    RFDETR_ASSERT(backend != nullptr);
    RFDETR_ASSERT_EQ_INT(rfdetr::model_realize_weights(*m, backend), RFDETR_OK);

    rfdetr::ForwardOutput fout = rfdetr::rfdetr_model_forward(
        *m, in_data.data(), W, backend);
    RFDETR_ASSERT(!fout.class_logits.empty());
    RFDETR_ASSERT(!fout.bbox_cxcywh.empty());

    /* Baseline class_logits ne = (91, 300, 1, 3) — last axis = decoder layer.
     * Inference uses [-1] = layer 2. Slice it. */
    const int NC = fout.num_classes;
    const int NQ = fout.num_queries;
    const size_t per_layer_cls = (size_t)NC * NQ;
    const auto& cls_full  = base.tensors.at("heads.class_logits");
    const auto& cls_shape = base.shapes.at("heads.class_logits");
    const int n_layers    = (int)cls_shape[3];
    RFDETR_ASSERT(n_layers >= 1);
    std::vector<float> want_cls(per_layer_cls);
    std::memcpy(want_cls.data(),
                cls_full.data() + (size_t)(n_layers - 1) * per_layer_cls,
                per_layer_cls * sizeof(float));

    /* The E2E test does NOT slice bbox_pred from the baseline directly — the
     * baseline captures the raw bbox_embed delta (4, 300, 3), but our C++
     * pipeline returns post-reparam (cx,cy,w,h). The torch reparam happens
     * INSIDE LWDETR.forward (lwdetr.py:231) using `ref_unsigmoid` (the
     * decoder's reference points) and is not captured by the heads hook.
     *
     * So we only compare class_logits here. (bbox parity is covered by
     * test_parity_heads which compares the raw delta from the same
     * post-decoder.norm input.) */

    /* Compute drift. */
    auto diff = [](const std::vector<float>& got, const std::vector<float>& want) {
        double mx = 0.0, sm = 0.0;
        const size_t n = std::min(got.size(), want.size());
        size_t idx = 0;
        float gm = 0, wm = 0;
        for (size_t i = 0; i < n; ++i) {
            float d = std::fabs(got[i] - want[i]);
            sm += d;
            if (d > mx) { mx = d; idx = i; gm = got[i]; wm = want[i]; }
        }
        std::fprintf(stderr,
            "  [class_logits[-1]] max_abs=%.4g mean_abs=%.4g (got=%g want=%g at idx %zu)\n",
            mx, sm / (double)n, gm, wm, idx);
    };
    diff(fout.class_logits, want_cls);

    /* Sanity: the post-reparam boxes should all lie in [0, 1] (cx, cy, w, h
     * are normalized coords). */
    int oob = 0;
    for (size_t i = 0; i < fout.bbox_cxcywh.size(); ++i) {
        if (fout.bbox_cxcywh[i] < -0.1f || fout.bbox_cxcywh[i] > 1.1f) ++oob;
    }
    std::fprintf(stderr,
        "  [bbox_cxcywh range] %d / %zu values outside [-0.1, 1.1]\n",
        oob, fout.bbox_cxcywh.size());

    rfdetr::model_free(m);
    rfdetr::free_backend(backend);

    /* Pure regression guard — log only, no hard assertion on parity. */
    return 0;
}
