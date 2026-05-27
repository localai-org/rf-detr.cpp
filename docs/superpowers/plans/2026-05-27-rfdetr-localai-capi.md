# Phase A — rfdetr.cpp C-API for LocalAI Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend rfdetr.cpp's flat C-API (`include/rfdetr_capi.h`) with accessor functions and PNG-encoded masks so that LocalAI's native backend can consume rfdetr.cpp via purego's `RegisterLibFunc` — following the exact pattern used by [sam3.cpp](https://github.com/PABannier/sam3.cpp) in the LocalAI `backend/go/sam3-cpp/` directory.

**Architecture:** Add 5 new C functions that expose per-detection fields (class id, class name, bbox, score, mask PNG) using the sam3-cpp two-call-sizing pattern: first call with NULL buffer returns required size, second call writes the bytes. Re-use the existing `stb_image_write` library (already vendored at `third_party/stb/`) to PNG-encode masks C-side. Keep the existing `rfdetr_capi_detect_path`/`_buffer` JSON-envelope path for backward compatibility; the new accessor functions live alongside it. The `_detect_path`/`_buffer` now also store the last detection result in a per-handle slot that the accessors read from.

**Tech Stack:** C++ (rfdetr_capi shim), ggml (no changes), stb_image_write (already vendored), CMake (RFDETR_SHARED already exists).

---

## File Structure

- Modify: `include/rfdetr_capi.h` — add 6 new function declarations
- Modify: `src/rfdetr_capi.cpp` — add storage struct + 6 accessor implementations
- Create: `tests/test_capi_accessors.cpp` — exercises every accessor on nano-f16
- Modify: `tests/CMakeLists.txt` — register the new test
- Modify: `examples/cli/main.cpp` — confirmation that nothing breaks (existing CLI uses the higher-level C-API, not the flat one)
- Verify (don't modify): `CMakeLists.txt` already exposes `RFDETR_SHARED=ON` → builds `librfdetr.so`. Nothing to change there.

---

## Storage design

`rfdetr_capi_load` returns a handle (opaque pointer). The handle owns:
- The underlying `rfdetr_context*`
- A buffer holding the last detection batch (vector of `rfdetr_detection` copies + their PNG-encoded masks)
- The last raw JSON string (kept for backward compat with `_detect_path`/`_buffer`)

`rfdetr_capi_detect_path` and `_detect_buffer` populate the buffer + return JSON. Accessors read from the buffer. Each new `_detect_*` call clears the previous batch.

The buffer struct (private to `src/rfdetr_capi.cpp`):

```cpp
struct CapiHandle {
    rfdetr_context* ctx;
    // Last detection batch (owned copies — original returned by rfdetr_detect is freed).
    struct DetectionStore {
        int class_id;
        std::string class_name;
        float score;
        float x1, y1, x2, y2;
        // Mask PNG bytes (encoded via stb_image_write_to_func). Empty for non-seg models.
        std::vector<uint8_t> mask_png;
        int mask_w, mask_h;
    };
    std::vector<DetectionStore> last_detections;
    std::string last_json;
};
```

---

## Task 1: Add storage struct + clear-on-detect

**Files:**
- Modify: `src/rfdetr_capi.cpp`

- [ ] **Step 1: Add the `DetectionStore` + `CapiHandle` types**

In `src/rfdetr_capi.cpp`, find the existing `CapiHandle` struct (or wherever `rfdetr_capi_load` allocates its handle). Replace it with the version above. Note that `last_detections` and `last_json` are populated by the detect functions, accessed by the new accessors.

- [ ] **Step 2: Clear `last_detections` at the start of every detect call**

In `rfdetr_capi_detect_path` and `rfdetr_capi_detect_buffer`, before invoking `rfdetr_detect`:

```cpp
h->last_detections.clear();
h->last_json.clear();
```

This ensures stale results from a prior call can never leak.

- [ ] **Step 3: After `rfdetr_detect`, copy results into the store**

After the existing detect returns, walk the result array and push into `last_detections`:

```cpp
for (size_t i = 0; i < n; ++i) {
    const rfdetr_detection& d = dets[i];
    CapiHandle::DetectionStore s;
    s.class_id = d.class_id;
    s.class_name = d.class_name ? d.class_name : "";
    s.score = d.score;
    s.x1 = d.x1; s.y1 = d.y1; s.x2 = d.x2; s.y2 = d.y2;
    s.mask_w = d.mask_width;
    s.mask_h = d.mask_height;
    // PNG-encode mask if present (covered in Task 3)
    s.mask_png = {};  // placeholder for now; Task 3 fills it
    h->last_detections.push_back(std::move(s));
}
```

- [ ] **Step 4: Commit**

```bash
git add src/rfdetr_capi.cpp
git commit -m "feat(capi): persist detection batch on handle for accessor pattern"
```

---

## Task 2: Add bbox + score + class accessors

**Files:**
- Modify: `include/rfdetr_capi.h`
- Modify: `src/rfdetr_capi.cpp`

- [ ] **Step 1: Declare 4 accessors in the header**

Add to `include/rfdetr_capi.h`:

```c
/* Number of detections from the most recent detect call on `handle`.
 * Returns 0 if no detect call has been made or it returned no detections.
 * Returns -1 if handle is invalid. */
int rfdetr_capi_get_n_detections(void* handle);

/* Class id of detection `i` (0-indexed). Returns -1 if i is out of range
 * or handle is invalid. */
int rfdetr_capi_get_detection_class_id(void* handle, int i);

/* Bounding box of detection `i` in original image pixel coordinates.
 * Writes 4 floats to `out_xyxy` (x1, y1, x2, y2). Returns 0 on success,
 * -1 on invalid handle/index. */
int rfdetr_capi_get_detection_box(void* handle, int i, float out_xyxy[4]);

/* Confidence score of detection `i`. Returns -1.0 on invalid handle/index. */
float rfdetr_capi_get_detection_score(void* handle, int i);

/* Class name of detection `i` (NUL-terminated UTF-8). Two-call protocol:
 *   - pass NULL/0 to get required buffer size (including NUL byte)
 *   - pass a buffer >= required size to write the string
 * Returns required size (>= 1) on success, -1 on invalid handle/index. */
int rfdetr_capi_get_detection_class_name(void* handle, int i, char* buf, int buf_size);
```

- [ ] **Step 2: Implement all 5 accessors**

In `src/rfdetr_capi.cpp`:

```cpp
extern "C" int rfdetr_capi_get_n_detections(void* handle) {
    auto h = (CapiHandle*)handle;
    if (!h) return -1;
    return (int)h->last_detections.size();
}

extern "C" int rfdetr_capi_get_detection_class_id(void* handle, int i) {
    auto h = (CapiHandle*)handle;
    if (!h || i < 0 || i >= (int)h->last_detections.size()) return -1;
    return h->last_detections[i].class_id;
}

extern "C" int rfdetr_capi_get_detection_box(void* handle, int i, float out_xyxy[4]) {
    auto h = (CapiHandle*)handle;
    if (!h || i < 0 || i >= (int)h->last_detections.size() || !out_xyxy) return -1;
    const auto& d = h->last_detections[i];
    out_xyxy[0] = d.x1; out_xyxy[1] = d.y1;
    out_xyxy[2] = d.x2; out_xyxy[3] = d.y2;
    return 0;
}

extern "C" float rfdetr_capi_get_detection_score(void* handle, int i) {
    auto h = (CapiHandle*)handle;
    if (!h || i < 0 || i >= (int)h->last_detections.size()) return -1.0f;
    return h->last_detections[i].score;
}

extern "C" int rfdetr_capi_get_detection_class_name(void* handle, int i, char* buf, int buf_size) {
    auto h = (CapiHandle*)handle;
    if (!h || i < 0 || i >= (int)h->last_detections.size()) return -1;
    const std::string& s = h->last_detections[i].class_name;
    int needed = (int)s.size() + 1;
    if (buf == nullptr || buf_size < needed) return needed;
    std::memcpy(buf, s.data(), s.size());
    buf[s.size()] = '\0';
    return needed;
}
```

- [ ] **Step 3: Quick smoke test from a one-off C program**

Write a temp test (in `/tmp/`) that loads nano-f16 + an image, calls detect, then exercises every accessor. Make sure the output matches the JSON envelope's content.

```c
// /tmp/quick_capi.c
#include "rfdetr_capi.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    void* h;
    if (rfdetr_capi_load("models/rfdetr-base-f16.gguf", 4, &h) != 0) return 1;
    char* json = NULL;
    if (rfdetr_capi_detect_path(h, "tests/fixtures/ci/test_image.jpg", 0.5f, 300, &json) != 0) return 1;
    rfdetr_capi_free_string(json);
    int n = rfdetr_capi_get_n_detections(h);
    printf("n=%d\n", n);
    for (int i = 0; i < n; ++i) {
        int sz = rfdetr_capi_get_detection_class_name(h, i, NULL, 0);
        char* name = malloc(sz);
        rfdetr_capi_get_detection_class_name(h, i, name, sz);
        float bbox[4];
        rfdetr_capi_get_detection_box(h, i, bbox);
        printf("  [%d] class=%d (%s) score=%.3f bbox=[%.1f,%.1f,%.1f,%.1f]\n",
               i, rfdetr_capi_get_detection_class_id(h, i), name,
               rfdetr_capi_get_detection_score(h, i),
               bbox[0], bbox[1], bbox[2], bbox[3]);
        free(name);
    }
    rfdetr_capi_unload(h);
    return 0;
}
```

Build + run:

```bash
gcc -I include /tmp/quick_capi.c -L build/lib -lrfdetr -o /tmp/quick_capi
LD_LIBRARY_PATH=build/lib /tmp/quick_capi
```

Compare the printed output to the JSON `tests/fixtures/ci/expected_base-f16.json` — same class ids, same scores (within FP noise), same boxes.

- [ ] **Step 4: Commit**

```bash
git add include/rfdetr_capi.h src/rfdetr_capi.cpp
git commit -m "feat(capi): accessor functions for class id, name, bbox, score"
```

---

## Task 3: Add PNG-encoded mask accessor

**Files:**
- Modify: `include/rfdetr_capi.h`
- Modify: `src/rfdetr_capi.cpp`

- [ ] **Step 1: Verify stb_image_write is accessible from src/rfdetr_capi.cpp**

```bash
grep -n "stb_image_write" src/rfdetr_capi.cpp src/image_io.cpp 2>/dev/null
```

If stb is only used in `src/image_io.cpp`, you can either:
- Include `stb_image_write.h` directly in `src/rfdetr_capi.cpp` (it's header-only) AND add a `#define STB_IMAGE_WRITE_IMPLEMENTATION` exactly once (likely already done in image_io.cpp), then declare `stbi_write_png_to_func` and use it
- OR add a helper to `src/image_io.{hpp,cpp}` exporting `void rfdetr_encode_gray_png(const uint8_t* data, int w, int h, std::vector<uint8_t>& out)` and call that from the capi shim. **Recommended** — avoids `#define STB_IMAGE_WRITE_IMPLEMENTATION` duplication issues.

- [ ] **Step 2: Add the encode helper (if going with the recommended approach)**

In `src/image_io.hpp`:

```cpp
// Encode a grayscale 0/255 mask as PNG bytes into `out`. Returns true on success.
bool rfdetr_encode_gray_png(const uint8_t* data, int w, int h, std::vector<uint8_t>& out);
```

In `src/image_io.cpp` (alongside the existing `rfdetr_write_gray_png`):

```cpp
static void png_writer_callback(void* ctx, void* data, int size) {
    auto* v = (std::vector<uint8_t>*)ctx;
    v->insert(v->end(), (uint8_t*)data, (uint8_t*)data + size);
}

bool rfdetr_encode_gray_png(const uint8_t* data, int w, int h, std::vector<uint8_t>& out) {
    out.clear();
    int rc = stbi_write_png_to_func(png_writer_callback, &out, w, h, 1, data, w);
    return rc != 0;
}
```

- [ ] **Step 3: Wire mask PNG encoding in Task 1's detection-store loop**

Update the loop from Task 1 step 3 to also encode masks:

```cpp
if (d.mask && d.mask_width > 0 && d.mask_height > 0) {
    rfdetr_encode_gray_png(d.mask, d.mask_width, d.mask_height, s.mask_png);
}
```

- [ ] **Step 4: Declare the mask accessor in the header**

Add to `include/rfdetr_capi.h`:

```c
/* PNG-encoded binary segmentation mask of detection `i` (1 byte per pixel,
 * 0 = background, 255 = foreground; same shape as the source image).
 *
 * Two-call protocol:
 *   - pass NULL/0 to get the required buffer size (PNG byte length).
 *   - pass a buffer >= required size to write the encoded bytes.
 *
 * Returns required size on success (or 0 if this detection has no mask
 * — i.e., the model is a detection-only variant). Returns -1 on invalid
 * handle/index.
 */
int rfdetr_capi_get_detection_mask_png(void* handle, int i, unsigned char* buf, int buf_size);
```

- [ ] **Step 5: Implement the mask accessor**

```cpp
extern "C" int rfdetr_capi_get_detection_mask_png(void* handle, int i, unsigned char* buf, int buf_size) {
    auto h = (CapiHandle*)handle;
    if (!h || i < 0 || i >= (int)h->last_detections.size()) return -1;
    const auto& mp = h->last_detections[i].mask_png;
    if (mp.empty()) return 0;
    int needed = (int)mp.size();
    if (buf == nullptr || buf_size < needed) return needed;
    std::memcpy(buf, mp.data(), needed);
    return needed;
}
```

- [ ] **Step 6: Smoke test with seg-nano-f16**

```bash
# Extend /tmp/quick_capi.c with a mask round-trip: get_mask_png on each detection,
# write any non-empty PNGs to /tmp/quick_mask_${i}.png, visually inspect at least one.

# Reload with seg-nano-f16 instead of base-f16, threshold 0.5.
gcc -I include /tmp/quick_capi.c -L build/lib -lrfdetr -o /tmp/quick_capi
LD_LIBRARY_PATH=build/lib /tmp/quick_capi
file /tmp/quick_mask_0.png
# Expected: "PNG image data, 480 x 320, 8-bit grayscale, non-interlaced"
```

The PNG should open in any viewer and show a recognizable object silhouette.

- [ ] **Step 7: Commit**

```bash
git add include/rfdetr_capi.h src/rfdetr_capi.cpp src/image_io.hpp src/image_io.cpp
git commit -m "feat(capi): PNG-encoded mask accessor for segmentation models"
```

---

## Task 4: Add a ctest target exercising every accessor

**Files:**
- Create: `tests/test_capi_accessors.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the test**

Create `tests/test_capi_accessors.cpp`:

```cpp
/* tests/test_capi_accessors.cpp — exercises every rfdetr_capi accessor on
 * nano-f16 (detection) and skips seg-specific checks if no seg model is
 * available. Skips gracefully when no model is present.
 */
#include "rfdetr_capi.h"
#include "test_assert.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

static bool file_exists(const char* p) {
    struct stat st; return ::stat(p, &st) == 0;
}

int main() {
    // Try a few common locations for a detection model.
    const char* candidates[] = {
        "models/rfdetr-nano-f16.gguf",
        "models/rfdetr-base-f16.gguf",
        nullptr,
    };
    const char* model = nullptr;
    for (int i = 0; candidates[i]; ++i) {
        if (file_exists(candidates[i])) { model = candidates[i]; break; }
    }
    if (!model) {
        std::fprintf(stderr, "[test_capi_accessors] SKIP: no rfdetr GGUF found\n");
        return 0;
    }

    void* h = nullptr;
    int rc = rfdetr_capi_load(model, 4, &h);
    RFDETR_ASSERT(rc == 0 && h != nullptr);

    char* json = nullptr;
    rc = rfdetr_capi_detect_path(h, "tests/fixtures/ci/test_image.jpg", 0.5f, 300, &json);
    RFDETR_ASSERT(rc == 0 && json != nullptr);
    rfdetr_capi_free_string(json);

    int n = rfdetr_capi_get_n_detections(h);
    RFDETR_ASSERT(n > 0);
    std::fprintf(stderr, "[test_capi_accessors] %s -> %d detections\n", model, n);

    for (int i = 0; i < n; ++i) {
        int cid = rfdetr_capi_get_detection_class_id(h, i);
        RFDETR_ASSERT(cid >= 0 && cid < 91);

        float bbox[4];
        rc = rfdetr_capi_get_detection_box(h, i, bbox);
        RFDETR_ASSERT(rc == 0);
        RFDETR_ASSERT(bbox[2] > bbox[0]);  // x2 > x1
        RFDETR_ASSERT(bbox[3] > bbox[1]);  // y2 > y1

        float score = rfdetr_capi_get_detection_score(h, i);
        RFDETR_ASSERT(score >= 0.5f && score <= 1.0f);

        int sz = rfdetr_capi_get_detection_class_name(h, i, nullptr, 0);
        RFDETR_ASSERT(sz > 1);  // at least 1 char + NUL
        std::vector<char> name(sz);
        rc = rfdetr_capi_get_detection_class_name(h, i, name.data(), sz);
        RFDETR_ASSERT(rc == sz);

        // Mask: for detection-only models this returns 0; for seg models it returns >0.
        int msz = rfdetr_capi_get_detection_mask_png(h, i, nullptr, 0);
        RFDETR_ASSERT(msz >= 0);
        if (msz > 0) {
            std::vector<unsigned char> mask(msz);
            rc = rfdetr_capi_get_detection_mask_png(h, i, mask.data(), msz);
            RFDETR_ASSERT(rc == msz);
            // Verify PNG magic.
            RFDETR_ASSERT(mask[0] == 0x89 && mask[1] == 'P' && mask[2] == 'N' && mask[3] == 'G');
        }
    }

    // Invalid index returns sane sentinels.
    RFDETR_ASSERT(rfdetr_capi_get_detection_class_id(h, -1) == -1);
    RFDETR_ASSERT(rfdetr_capi_get_detection_class_id(h, n + 99) == -1);
    RFDETR_ASSERT(rfdetr_capi_get_detection_score(h, -1) < 0.0f);

    // NULL handle.
    RFDETR_ASSERT(rfdetr_capi_get_n_detections(nullptr) == -1);

    rfdetr_capi_unload(h);
    return 0;
}
```

- [ ] **Step 2: Register in CMake**

In `tests/CMakeLists.txt`:

```cmake
rfdetr_add_test(test_capi_accessors)
```

- [ ] **Step 3: Build + run**

```bash
cmake --build build -j 2>&1 | tail -3
ctest --test-dir build -R test_capi_accessors --output-on-failure
```

Expected: passes; prints "[test_capi_accessors] models/rfdetr-base-f16.gguf -> N detections".

- [ ] **Step 4: Commit**

```bash
git add tests/test_capi_accessors.cpp tests/CMakeLists.txt
git commit -m "test: capi accessors round-trip on real model"
```

---

## Task 5: Build with RFDETR_SHARED=ON and verify symbol exposure

**Files:**
- None modified (verification step)

- [ ] **Step 1: Reconfigure + build shared**

```bash
rm -rf build-shared
cmake -B build-shared -DRFDETR_SHARED=ON -DRFDETR_BUILD_TESTS=OFF -DRFDETR_BUILD_CLI=OFF \
    -DGGML_NATIVE=ON 2>&1 | tail -10
cmake --build build-shared -j 2>&1 | tail -3
ls -lh build-shared/lib/librfdetr.so
```

Expected: `librfdetr.so` is built (a few MB, plus a `libggml*.so` siblings).

- [ ] **Step 2: Confirm all 6 new symbols are exported**

```bash
nm -D --defined-only build-shared/lib/librfdetr.so | grep rfdetr_capi_get | sort
```

Expected output (one line per symbol):

```
rfdetr_capi_get_detection_box
rfdetr_capi_get_detection_class_id
rfdetr_capi_get_detection_class_name
rfdetr_capi_get_detection_mask_png
rfdetr_capi_get_detection_score
rfdetr_capi_get_n_detections
```

If any are missing, check for `__attribute__((visibility("default")))` requirements (look at how the existing `rfdetr_capi_load` is decorated) and match.

- [ ] **Step 3: Confirm the test still passes against the shared build**

```bash
rm -rf build
cmake -B build -DRFDETR_SHARED=ON -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=ON \
    -DGGML_NATIVE=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: 24/24 tests pass (23 prior + new `test_capi_accessors`).

- [ ] **Step 4: Commit (no-op probably, but record any visibility fixes)**

If you had to add `__attribute__((visibility("default")))` to any of the new functions, that's the commit content. Otherwise, this task is just a verification step — no commit needed.

---

## Task 6: Push

- [ ] **Step 1: Push all commits to origin**

```bash
git push origin main
```

After push, the CI workflow from commit `3206020` will fire and validate that the new accessor functions don't break the existing smoke test.

---

## Self-Review

- **Spec coverage**: 6 accessor functions added (n_detections, class_id, class_name, box, score, mask_png) ✓; PNG encoding via stb_image_write ✓; shared library exposes them ✓; ctest covers them ✓.
- **Placeholders**: none.
- **Type consistency**: `CapiHandle::DetectionStore` and its `mask_png` field are used consistently across all 6 accessors; `out_xyxy[4]` and PNG `unsigned char*` types match C conventions LocalAI's purego layer expects.

**Open questions to verify at execution time:**

1. Does `src/rfdetr_capi.cpp` actually have a `CapiHandle` struct already, or is the handle just a `rfdetr_context*` cast through `void*`? Adapt Task 1 to whichever it is.
2. Does `stb_image_write.h` already have `STB_IMAGE_WRITE_IMPLEMENTATION` defined exactly once in the project? Check `grep -rn "STB_IMAGE_WRITE_IMPLEMENTATION" src/`. If yes, follow the recommended approach (call from image_io.cpp). If no, you'll need to add the `#define` somewhere — but only once.
3. Will the symbols need `extern "C"` AND `__attribute__((visibility("default")))` on Linux? The existing capi functions are a guide; copy their decoration. On macOS, default visibility is "default" so no decoration needed.
4. The `rfdetr_detection.class_name` is a borrowed pointer per the header — verify the underlying class-name table is stable for the lifetime of the handle. If not, we need to copy strings into the store immediately (Task 1 step 3 already does this via `std::string`, which copies). Safe.
