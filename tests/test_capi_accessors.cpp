/* tests/test_capi_accessors.cpp — exercises every rfdetr_capi accessor on a
 * real detection model (and, if available, a seg model for mask coverage).
 *
 * Skips gracefully when no real GGUF model is present, so a fresh clone of
 * the repo (with only the synthesized fixtures) still passes ctest. */

#include "rfdetr_capi.h"
#include "test_assert.hpp"

#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool file_exists(const char* p) {
    struct stat st;
    return ::stat(p, &st) == 0;
}

static void exercise(const char* model_path, const char* image_path, bool expect_masks) {
    std::fprintf(stderr, "[test_capi_accessors] model=%s image=%s expect_masks=%d\n",
                 model_path, image_path, (int)expect_masks);

    rfdetr_handle_t h = 0;
    int rc = rfdetr_capi_load(model_path, 4, &h);
    RFDETR_ASSERT_EQ_INT(rc, 0);
    RFDETR_ASSERT(h != 0);

    /* Before any detect call, n_detections should be 0 (handle valid, batch empty). */
    RFDETR_ASSERT_EQ_INT(rfdetr_capi_get_n_detections(h), 0);

    char* json = nullptr;
    rc = rfdetr_capi_detect_path(h, image_path, 0.5f, 300, &json);
    RFDETR_ASSERT_EQ_INT(rc, 0);
    RFDETR_ASSERT(json != nullptr);
    rfdetr_capi_free_string(json);

    int n = rfdetr_capi_get_n_detections(h);
    RFDETR_ASSERT(n > 0);
    std::fprintf(stderr, "[test_capi_accessors]   -> %d detections\n", n);

    int n_masks_seen = 0;
    for (int i = 0; i < n; ++i) {
        int cid = rfdetr_capi_get_detection_class_id(h, i);
        RFDETR_ASSERT(cid >= 0 && cid < 91);

        float bbox[4];
        rc = rfdetr_capi_get_detection_box(h, i, bbox);
        RFDETR_ASSERT_EQ_INT(rc, 0);
        RFDETR_ASSERT(bbox[2] > bbox[0]);   /* x2 > x1 */
        RFDETR_ASSERT(bbox[3] > bbox[1]);   /* y2 > y1 */
        RFDETR_ASSERT(bbox[0] >= 0.0f);
        RFDETR_ASSERT(bbox[1] >= 0.0f);

        float score = rfdetr_capi_get_detection_score(h, i);
        RFDETR_ASSERT(score >= 0.5f && score <= 1.0f);

        /* Two-call sizing: first NULL+0, then a sized buffer. */
        int sz = rfdetr_capi_get_detection_class_name(h, i, nullptr, 0);
        RFDETR_ASSERT(sz > 1);                       /* at least 1 char + NUL */
        std::vector<char> name(sz);
        rc = rfdetr_capi_get_detection_class_name(h, i, name.data(), sz);
        RFDETR_ASSERT_EQ_INT(rc, sz);
        RFDETR_ASSERT(name[sz - 1] == '\0');         /* NUL-terminated */
        RFDETR_ASSERT(std::strlen(name.data()) == (size_t)sz - 1);

        /* Mask: detection-only models return 0; seg models return >0. */
        int msz = rfdetr_capi_get_detection_mask_png(h, i, nullptr, 0);
        RFDETR_ASSERT(msz >= 0);
        if (msz > 0) {
            ++n_masks_seen;
            std::vector<unsigned char> mask(msz);
            int wrote = rfdetr_capi_get_detection_mask_png(h, i, mask.data(), msz);
            RFDETR_ASSERT_EQ_INT(wrote, msz);
            /* PNG magic: 0x89 'P' 'N' 'G' 0x0D 0x0A 0x1A 0x0A */
            RFDETR_ASSERT(mask[0] == 0x89);
            RFDETR_ASSERT(mask[1] == 'P');
            RFDETR_ASSERT(mask[2] == 'N');
            RFDETR_ASSERT(mask[3] == 'G');
        }
    }

    if (expect_masks) {
        RFDETR_ASSERT(n_masks_seen > 0);
        std::fprintf(stderr, "[test_capi_accessors]   -> %d masks (PNG-encoded)\n", n_masks_seen);
    } else {
        RFDETR_ASSERT_EQ_INT(n_masks_seen, 0);
    }

    /* Sentinel checks: invalid index and NULL handle. */
    RFDETR_ASSERT_EQ_INT(rfdetr_capi_get_detection_class_id(h, -1), -1);
    RFDETR_ASSERT_EQ_INT(rfdetr_capi_get_detection_class_id(h, n + 99), -1);
    RFDETR_ASSERT(rfdetr_capi_get_detection_score(h, -1) < 0.0f);
    RFDETR_ASSERT_EQ_INT(rfdetr_capi_get_detection_mask_png(h, -1, nullptr, 0), -1);
    RFDETR_ASSERT_EQ_INT(rfdetr_capi_get_n_detections(0), -1);

    /* Two-call sizing edge case: buf_size too small returns needed size, not -1. */
    int needed = rfdetr_capi_get_detection_class_name(h, 0, nullptr, 0);
    char tiny[1];
    int got = rfdetr_capi_get_detection_class_name(h, 0, tiny, 1);
    RFDETR_ASSERT_EQ_INT(got, needed);

    /* Stale-result check: a detect call with the same image still yields >0;
     * a detect call on a non-existent path yields rc != 0 and clears the batch. */
    char* json2 = nullptr;
    int dr_bad = rfdetr_capi_detect_path(h, "/no/such/image.png", 0.5f, 300, &json2);
    RFDETR_ASSERT(dr_bad != 0);
    if (json2) rfdetr_capi_free_string(json2);
    RFDETR_ASSERT_EQ_INT(rfdetr_capi_get_n_detections(h), 0);  /* batch cleared */

    rfdetr_capi_unload(h);
}

int main() {
    /* Resolve paths relative to the source tree so the test works regardless
     * of ctest's working directory. */
    const std::string root = std::string(RFDETR_SOURCE_DIR) + "/";
    const std::string det_paths[] = {
        root + "models/rfdetr-nano-f16.gguf",
        root + "models/rfdetr-base-f16.gguf",
        root + "models/rfdetr-small-f16.gguf",
    };
    const std::string seg_paths[] = {
        root + "models/rfdetr-seg-nano-f16.gguf",
        root + "models/rfdetr-seg-small-f16.gguf",
        root + "models/rfdetr-seg-medium-f16.gguf",
    };
    const std::string image_path = root + "tests/fixtures/ci/test_image.jpg";

    if (!file_exists(image_path.c_str())) {
        std::fprintf(stderr, "[test_capi_accessors] SKIP: %s not found\n",
                     image_path.c_str());
        return 0;
    }

    const char* det_model = nullptr;
    for (const auto& p : det_paths) {
        if (file_exists(p.c_str())) { det_model = p.c_str(); break; }
    }
    const char* seg_model = nullptr;
    for (const auto& p : seg_paths) {
        if (file_exists(p.c_str())) { seg_model = p.c_str(); break; }
    }

    if (!det_model && !seg_model) {
        std::fprintf(stderr,
                     "[test_capi_accessors] SKIP: no rfdetr GGUF found in %smodels/\n",
                     root.c_str());
        return 0;
    }

    if (det_model) exercise(det_model, image_path.c_str(), /*expect_masks=*/false);
    if (seg_model) exercise(seg_model, image_path.c_str(), /*expect_masks=*/true);
    return 0;
}
