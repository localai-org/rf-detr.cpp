# rt-detr.cpp Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the rt-detr.cpp repo with a working build system, public C API headers, image I/O, postprocessing math, and a CLI skeleton — all unit-tested. No model loading or inference yet; those live in Plan 2 and Plan 3. After this plan, `rfdetr-cli detect` runs end-to-end on an image and emits a JSON envelope (with synthetic empty detections) plus an annotated PNG, validating every piece *around* the model.

**Architecture:** Mirror vibevoice.cpp layout. `ggml` as a git submodule; `stb_image`/`stb_image_write` vendored as header-only. Root CMake builds `librfdetr` (static by default) and `rfdetr-cli`. C++17. ctest harness with one binary per test file, plain `assert()` + a tiny `RFDETR_ASSERT_*` macro pack — no gtest dep. Backend flags (`RFDETR_GGML_CUDA`, etc.) are wired now even though ggml is not yet used at runtime, so Plans 2-3 don't have to touch the build system.

**Tech Stack:** C++17, CMake ≥ 3.14, ggml (submodule, ggerganov/ggml), stb (vendored), ctest, Apache-2.0.

---

## File map (created in this plan)

```
rt-detr.cpp/
├── .gitignore
├── .gitmodules
├── LICENSE
├── README.md
├── CMakeLists.txt
├── third_party/
│   ├── ggml/                     # submodule (Task 2)
│   └── stb/
│       ├── stb_image.h
│       ├── stb_image_write.h
│       └── stb_truetype.h
├── include/
│   ├── rfdetr.h                  # opaque-pointer API (decls only in this plan)
│   └── rfdetr_capi.h             # flat ABI (decls only)
├── src/
│   ├── common.cpp / common.hpp
│   ├── image_io.cpp / image_io.hpp
│   ├── postprocess.cpp / postprocess.hpp
│   ├── visualize.cpp / visualize.hpp
│   └── cli.cpp / cli.hpp         # CLI argument parsing
├── examples/
│   └── cli/
│       ├── CMakeLists.txt
│       └── main.cpp              # rfdetr-cli entry point
├── tests/
│   ├── CMakeLists.txt
│   ├── test_assert.hpp           # tiny assertion helpers
│   ├── fixtures/
│   │   └── cats.png              # generated 16x16 test image (Task 8)
│   ├── test_common.cpp
│   ├── test_image_io.cpp
│   ├── test_postprocess.cpp
│   ├── test_visualize.cpp
│   └── test_cli_smoke.cpp
└── docs/
    └── (existing spec stays put)
```

---

### Task 1: Initialize repo skeleton

**Files:**
- Create: `.gitignore`, `LICENSE`, `README.md`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
# Build
build/
build-*/
cmake-build-*/
out/

# Binaries / artifacts
*.o
*.a
*.so
*.dylib
*.dll
*.exe

# Models
*.gguf
*.safetensors
*.pt
models/

# Python
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/

# Editors / IDEs
.vscode/
.idea/
*.swp
*~

# OS
.DS_Store
Thumbs.db

# Test artifacts
tests/fixtures/generated/
tests/parity/bundles/
```

- [ ] **Step 2: Create `LICENSE`** (Apache-2.0, full text)

Download the canonical Apache-2.0 text from `https://www.apache.org/licenses/LICENSE-2.0.txt` and save it as `LICENSE`. Replace the copyright placeholder at the top with:

```
Copyright 2026 Ettore Di Giacinto <mudler@localai.io>

Licensed under the Apache License, Version 2.0 (the "License");
...
```

- [ ] **Step 3: Create minimal `README.md`**

```markdown
# rt-detr.cpp

C++/ggml inference for [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).

Project layout mirrors [vibevoice.cpp](https://github.com/mudler/vibevoice.cpp).
See `docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md` for the design.

## Status

Foundation only. No model loading or inference yet — see plans under `docs/superpowers/plans/`.

## Build

\`\`\`
git clone --recursive <repo-url>
cd rt-detr.cpp
cmake -B build -DRFDETR_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
\`\`\`

## License

Apache-2.0. See `LICENSE`.
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore LICENSE README.md
git commit -m "chore: initialize repo skeleton (gitignore, license, readme)"
```

---

### Task 2: Add ggml submodule

**Files:**
- Create: `.gitmodules`
- Add submodule at: `third_party/ggml`

- [ ] **Step 1: Add the submodule**

```bash
git submodule add https://github.com/ggerganov/ggml.git third_party/ggml
```

- [ ] **Step 2: Pin to a known-good commit**

Use the latest stable tag at time of writing. Run:

```bash
cd third_party/ggml
git fetch --tags
git checkout master   # ggml does not use semver tags; master is the working ref
cd ../..
```

If the engineer is reading this later and `master` has drifted, they should pin to the commit hash they tested with. Update `.gitmodules` if a branch other than `master` is desired.

- [ ] **Step 3: Verify submodule is initialized**

```bash
git submodule status
```

Expected output (one line, starting with a commit hash):
```
 <hash> third_party/ggml (heads/master)
```

- [ ] **Step 4: Commit**

```bash
git add .gitmodules third_party/ggml
git commit -m "chore: add ggml as a submodule under third_party/ggml"
```

---

### Task 3: Vendor stb headers

**Files:**
- Create: `third_party/stb/stb_image.h`
- Create: `third_party/stb/stb_image_write.h`
- Create: `third_party/stb/stb_truetype.h`
- Create: `third_party/stb/LICENSE`

- [ ] **Step 1: Create directory**

```bash
mkdir -p third_party/stb
```

- [ ] **Step 2: Download the three headers**

```bash
curl -fsSL https://raw.githubusercontent.com/nothings/stb/master/stb_image.h       -o third_party/stb/stb_image.h
curl -fsSL https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h -o third_party/stb/stb_image_write.h
curl -fsSL https://raw.githubusercontent.com/nothings/stb/master/stb_truetype.h    -o third_party/stb/stb_truetype.h
```

- [ ] **Step 3: Save the stb license**

```bash
cat > third_party/stb/LICENSE <<'EOF'
The stb libraries (stb_image.h, stb_image_write.h, stb_truetype.h) are
dual-licensed under the MIT License and the Public Domain. See the
license text at the bottom of each header file. Source:
https://github.com/nothings/stb
EOF
```

- [ ] **Step 4: Verify the headers exist and look right**

```bash
ls -la third_party/stb/
head -1 third_party/stb/stb_image.h
head -1 third_party/stb/stb_image_write.h
head -1 third_party/stb/stb_truetype.h
```

Each should start with `/* stb_<name> - <version> ... */`.

- [ ] **Step 5: Commit**

```bash
git add third_party/stb/
git commit -m "chore: vendor stb_image, stb_image_write, stb_truetype headers"
```

---

### Task 4: Root CMakeLists.txt

**Files:**
- Create: `CMakeLists.txt`

- [ ] **Step 1: Write `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.14)
project(rfdetr LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options
option(RFDETR_BUILD_TESTS "Build unit tests"             OFF)
option(RFDETR_BUILD_CLI   "Build the rfdetr-cli binary"  ON)
option(RFDETR_SHARED      "Build librfdetr as a shared library" OFF)

# Backend forwarding to ggml (mirrors vibevoice.cpp pattern)
option(RFDETR_GGML_CUDA    "Enable CUDA backend in ggml"    OFF)
option(RFDETR_GGML_METAL   "Enable Metal backend in ggml"   OFF)
option(RFDETR_GGML_VULKAN  "Enable Vulkan backend in ggml"  OFF)
option(RFDETR_GGML_HIPBLAS "Enable HIPBLAS backend in ggml" OFF)

if(RFDETR_GGML_CUDA)    set(GGML_CUDA    ON CACHE BOOL "" FORCE) endif()
if(RFDETR_GGML_METAL)   set(GGML_METAL   ON CACHE BOOL "" FORCE) endif()
if(RFDETR_GGML_VULKAN)  set(GGML_VULKAN  ON CACHE BOOL "" FORCE) endif()
if(RFDETR_GGML_HIPBLAS) set(GGML_HIP     ON CACHE BOOL "" FORCE) endif()

# Common compile flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ggml (built but not yet used in this plan — wires the integration early)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# stb (header-only interface target)
add_library(rfdetr_stb INTERFACE)
target_include_directories(rfdetr_stb INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)

# librfdetr
set(RFDETR_SOURCES
    src/common.cpp
    src/image_io.cpp
    src/postprocess.cpp
    src/visualize.cpp
    src/cli.cpp
)

if(RFDETR_SHARED)
    add_library(rfdetr SHARED ${RFDETR_SOURCES})
else()
    add_library(rfdetr STATIC ${RFDETR_SOURCES})
endif()

target_include_directories(rfdetr
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(rfdetr PRIVATE rfdetr_stb ggml)

# CLI
if(RFDETR_BUILD_CLI)
    add_subdirectory(examples/cli)
endif()

# Tests
if(RFDETR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

- [ ] **Step 2: Verify the build configures**

```bash
cmake -B build -DRFDETR_BUILD_TESTS=OFF -DRFDETR_BUILD_CLI=OFF
```

Expected: configure succeeds, ends with `-- Configuring done` and `-- Generating done`. No targets are built yet because we haven't written the sources, but configuration must work (will fail at build time, which is fine here).

Note: build itself will fail because `src/common.cpp` etc. do not exist yet. That's intentional — Task 5+ fills them in.

- [ ] **Step 3: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: add root CMakeLists with ggml + stb wiring and backend forwarding"
```

---

### Task 5: Public C API headers (declarations only)

**Files:**
- Create: `include/rfdetr.h`
- Create: `include/rfdetr_capi.h`

These define the public surface. Implementations come later (model-dependent ones in Plan 2/3). In this plan we will implement only the model-independent pieces (logging, image, render).

- [ ] **Step 1: Write `include/rfdetr.h`**

```c
#ifndef RFDETR_H
#define RFDETR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes. 0 = success. */
typedef enum {
    RFDETR_OK                  =  0,
    RFDETR_ERR_INVALID_ARG     = -1,
    RFDETR_ERR_FILE_NOT_FOUND  = -2,
    RFDETR_ERR_IO              = -3,
    RFDETR_ERR_OUT_OF_MEMORY   = -4,
    RFDETR_ERR_DECODE          = -5,
    RFDETR_ERR_MODEL_FORMAT    = -6,   /* GGUF parse / variant detection */
    RFDETR_ERR_MODEL_LOAD      = -7,   /* tensor binding */
    RFDETR_ERR_INFERENCE       = -8,
    RFDETR_ERR_NOT_IMPLEMENTED = -99
} rfdetr_status;

const char* rfdetr_status_str(rfdetr_status s);

/* Logging */
typedef enum {
    RFDETR_LOG_DEBUG = 0,
    RFDETR_LOG_INFO  = 1,
    RFDETR_LOG_WARN  = 2,
    RFDETR_LOG_ERROR = 3
} rfdetr_log_level;

typedef void (*rfdetr_log_cb)(rfdetr_log_level lvl, const char* msg, void* user_data);
void rfdetr_set_log_callback(rfdetr_log_cb cb, void* user_data);

/* Opaque types */
typedef struct rfdetr_context rfdetr_context;
typedef struct rfdetr_image   rfdetr_image;

/* Detections */
typedef struct {
    uint32_t    class_id;
    const char* class_name;  /* borrowed; lifetime tied to rfdetr_context */
    float       score;
    float       x1, y1, x2, y2;  /* pixel coords on the original image */
} rfdetr_detection;

/* Init / detect parameters (forward-declared; full impl in Plan 2/3) */
typedef struct {
    const char*   model_path;
    int           n_threads;
    rfdetr_log_cb log_cb;
    void*         log_user_data;
} rfdetr_params;

typedef struct {
    float           threshold;          /* default 0.5 */
    uint32_t        top_k;              /* default 300 */
    const uint32_t* class_filter;       /* optional allowlist */
    size_t          class_filter_len;
} rfdetr_detect_params;

/* Lifecycle (not yet implemented; returns RFDETR_ERR_NOT_IMPLEMENTED in this plan) */
rfdetr_context* rfdetr_init(const rfdetr_params* params, rfdetr_status* out_status);
void            rfdetr_free(rfdetr_context* ctx);

/* Image I/O (IMPLEMENTED in this plan) */
rfdetr_image* rfdetr_image_load_file(const char* path, rfdetr_status* out_status);
rfdetr_image* rfdetr_image_load_buffer(const uint8_t* bytes, size_t len, rfdetr_status* out_status);
void          rfdetr_image_free(rfdetr_image* img);
int           rfdetr_image_width(const rfdetr_image* img);
int           rfdetr_image_height(const rfdetr_image* img);

/* Detection (not yet implemented in this plan) */
rfdetr_status rfdetr_detect(rfdetr_context* ctx,
                            const rfdetr_image* img,
                            const rfdetr_detect_params* params,
                            rfdetr_detection** out_detections,
                            size_t* out_n);
void rfdetr_detections_free(rfdetr_detection* detections, size_t n);

/* Render annotated image (IMPLEMENTED in this plan, boxes only — labels in Plan 2) */
rfdetr_status rfdetr_render(const rfdetr_image* img,
                            const rfdetr_detection* detections, size_t n,
                            const char* out_path);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RFDETR_H */
```

- [ ] **Step 2: Write `include/rfdetr_capi.h`** (flat ABI)

```c
#ifndef RFDETR_CAPI_H
#define RFDETR_CAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Flat handle. All functions return int status; 0 = ok, negative = error. */
typedef uintptr_t rfdetr_handle_t;

int rfdetr_capi_load(const char* model_path, int n_threads, rfdetr_handle_t* out_handle);
int rfdetr_capi_unload(rfdetr_handle_t handle);

/* Detect on an image file path. Output is a JSON string (caller frees with
 * rfdetr_capi_free_string). */
int rfdetr_capi_detect_path(rfdetr_handle_t handle,
                            const char* image_path,
                            float threshold, uint32_t top_k,
                            char** out_json);

/* Detect on an in-memory image buffer (encoded PNG/JPEG bytes). */
int rfdetr_capi_detect_buffer(rfdetr_handle_t handle,
                              const uint8_t* bytes, size_t len,
                              float threshold, uint32_t top_k,
                              char** out_json);

void rfdetr_capi_free_string(char* s);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RFDETR_CAPI_H */
```

- [ ] **Step 3: Commit**

```bash
git add include/rfdetr.h include/rfdetr_capi.h
git commit -m "feat(api): declare public C API and flat dlopen ABI"
```

---

### Task 6: common.cpp — logging and status strings

**Files:**
- Create: `src/common.hpp`, `src/common.cpp`
- Create: `tests/test_assert.hpp`
- Create: `tests/CMakeLists.txt`
- Create: `tests/test_common.cpp`

- [ ] **Step 1: Write `tests/test_assert.hpp`** (tiny helper macros)

```cpp
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
```

- [ ] **Step 2: Write `tests/CMakeLists.txt`**

```cmake
# Each test_*.cpp builds into its own binary and is registered as a ctest test.

set(RFDETR_TEST_FIXTURES ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

function(rfdetr_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE rfdetr rfdetr_stb)
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )
    target_compile_definitions(${name} PRIVATE
        RFDETR_TEST_FIXTURES="${RFDETR_TEST_FIXTURES}")
    add_test(NAME ${name} COMMAND ${name})
endfunction()

rfdetr_add_test(test_common)
rfdetr_add_test(test_image_io)
rfdetr_add_test(test_postprocess)
rfdetr_add_test(test_visualize)
rfdetr_add_test(test_cli_smoke)
```

- [ ] **Step 3: Write the failing test `tests/test_common.cpp`**

```cpp
#include "test_assert.hpp"
#include "rfdetr.h"
#include <string>
#include <vector>

static std::vector<std::pair<rfdetr_log_level, std::string>> captured;

static void capture_cb(rfdetr_log_level lvl, const char* msg, void* /*ud*/) {
    captured.emplace_back(lvl, std::string(msg));
}

int main() {
    // Status strings exist and are not empty for known codes
    RFDETR_ASSERT(rfdetr_status_str(RFDETR_OK) != nullptr);
    RFDETR_ASSERT_STR_EQ(rfdetr_status_str(RFDETR_OK), "ok");
    RFDETR_ASSERT(std::string(rfdetr_status_str(RFDETR_ERR_INVALID_ARG)).find("invalid") != std::string::npos);
    RFDETR_ASSERT(std::string(rfdetr_status_str(RFDETR_ERR_NOT_IMPLEMENTED)).find("not implemented") != std::string::npos);

    // Unknown code yields a stable "unknown" string (not a crash)
    RFDETR_ASSERT(rfdetr_status_str((rfdetr_status)12345) != nullptr);

    // Log callback receives messages it was sent
    rfdetr_set_log_callback(capture_cb, nullptr);

    // Internal use: emit a message via the C++ helper (declared in common.hpp)
    extern void rfdetr_internal_log(rfdetr_log_level, const char*);
    rfdetr_internal_log(RFDETR_LOG_INFO, "hello");
    rfdetr_internal_log(RFDETR_LOG_WARN, "world");

    RFDETR_ASSERT_EQ_INT(captured.size(), 2);
    RFDETR_ASSERT(captured[0].first == RFDETR_LOG_INFO);
    RFDETR_ASSERT_STR_EQ(captured[0].second.c_str(), "hello");
    RFDETR_ASSERT(captured[1].first == RFDETR_LOG_WARN);
    RFDETR_ASSERT_STR_EQ(captured[1].second.c_str(), "world");

    // Unsetting the callback (nullptr) must not crash
    rfdetr_set_log_callback(nullptr, nullptr);
    rfdetr_internal_log(RFDETR_LOG_INFO, "ignored");
    RFDETR_ASSERT_EQ_INT(captured.size(), 2);  // unchanged

    return 0;
}
```

- [ ] **Step 4: Run the test to confirm it fails**

```bash
cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=OFF
cmake --build build --target test_common -j 2>&1 | tail -20
```

Expected: build failure with linker errors like `undefined reference to rfdetr_status_str`, `rfdetr_set_log_callback`, `rfdetr_internal_log`. This is the "test fails" state — it cannot even link.

- [ ] **Step 5: Write `src/common.hpp`**

```cpp
#ifndef RFDETR_COMMON_HPP
#define RFDETR_COMMON_HPP

#include "rfdetr.h"

/* Internal helper used by every source file to emit a log message
 * via the registered callback. No-op if no callback is set. */
extern "C" void rfdetr_internal_log(rfdetr_log_level lvl, const char* msg);

/* printf-style wrapper. Builds the string then dispatches. */
void rfdetr_logf(rfdetr_log_level lvl, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));

#endif
```

- [ ] **Step 6: Write `src/common.cpp`**

```cpp
#include "common.hpp"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace {

std::mutex          g_log_mutex;
rfdetr_log_cb       g_log_cb       = nullptr;
void*               g_log_user     = nullptr;

}  // namespace

extern "C" const char* rfdetr_status_str(rfdetr_status s) {
    switch (s) {
        case RFDETR_OK:                  return "ok";
        case RFDETR_ERR_INVALID_ARG:     return "invalid argument";
        case RFDETR_ERR_FILE_NOT_FOUND:  return "file not found";
        case RFDETR_ERR_IO:              return "i/o error";
        case RFDETR_ERR_OUT_OF_MEMORY:   return "out of memory";
        case RFDETR_ERR_DECODE:          return "image decode error";
        case RFDETR_ERR_MODEL_FORMAT:    return "model format error";
        case RFDETR_ERR_MODEL_LOAD:      return "model load error";
        case RFDETR_ERR_INFERENCE:       return "inference error";
        case RFDETR_ERR_NOT_IMPLEMENTED: return "not implemented";
    }
    return "unknown error";
}

extern "C" void rfdetr_set_log_callback(rfdetr_log_cb cb, void* user_data) {
    std::lock_guard<std::mutex> lk(g_log_mutex);
    g_log_cb   = cb;
    g_log_user = user_data;
}

extern "C" void rfdetr_internal_log(rfdetr_log_level lvl, const char* msg) {
    rfdetr_log_cb cb;
    void* ud;
    {
        std::lock_guard<std::mutex> lk(g_log_mutex);
        cb = g_log_cb;
        ud = g_log_user;
    }
    if (cb && msg) {
        cb(lvl, msg, ud);
    }
}

void rfdetr_logf(rfdetr_log_level lvl, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    rfdetr_internal_log(lvl, buf);
}
```

- [ ] **Step 7: Build only `common.cpp` first (other sources still missing)**

We need stubs so the library links. Create *placeholder* (empty) bodies for the other src files so the lib builds.

```bash
for f in image_io postprocess visualize cli; do
    cat > src/${f}.cpp <<'EOF'
/* Implemented in a later task in this plan. */
EOF
done
```

- [ ] **Step 8: Build the test**

```bash
cmake --build build --target test_common -j
```

Expected: build succeeds.

- [ ] **Step 9: Run the test**

```bash
ctest --test-dir build -R test_common --output-on-failure
```

Expected: `Test #1: test_common ........ Passed`.

- [ ] **Step 10: Commit**

```bash
git add include/ src/common.hpp src/common.cpp src/image_io.cpp src/postprocess.cpp src/visualize.cpp src/cli.cpp tests/
git commit -m "feat(common): status strings, logging, internal log helper + tests"
```

---

### Task 7: image_io — load PNG/JPEG to RGB buffer

**Files:**
- Modify: `src/image_io.cpp` (replace placeholder)
- Create: `src/image_io.hpp`
- Modify: `tests/test_image_io.cpp` (new content)
- Create: `tests/fixtures/cats.png` (generated programmatically)

- [ ] **Step 1: Create the fixture image programmatically**

We avoid checking in binary blobs we cannot verify. Create a small generator helper that builds `cats.png` at configure time from raw bytes encoded in source — but stb_image_write requires linking. Simpler: write a one-shot generator binary.

Create `tests/fixtures/gen_fixture.cpp`:

```cpp
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gen_fixture <out_dir>\n");
        return 1;
    }
    const int W = 16, H = 16, C = 3;
    uint8_t px[W*H*C];
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int i = (y*W + x) * C;
            px[i+0] = (uint8_t)((x*16) & 0xff);  // R = horizontal ramp
            px[i+1] = (uint8_t)((y*16) & 0xff);  // G = vertical ramp
            px[i+2] = (uint8_t)((x*y) & 0xff);   // B = product
        }
    }
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/cats.png", argv[1]);
    if (!stbi_write_png(path, W, H, C, px, W*C)) {
        std::fprintf(stderr, "failed to write %s\n", path);
        return 1;
    }
    std::printf("wrote %s\n", path);
    return 0;
}
```

Update `tests/CMakeLists.txt` to build the generator and run it at configure time:

```cmake
# Generate fixtures
add_executable(gen_fixture fixtures/gen_fixture.cpp)
target_link_libraries(gen_fixture PRIVATE rfdetr_stb)
add_custom_command(
    OUTPUT  ${RFDETR_TEST_FIXTURES}/cats.png
    COMMAND gen_fixture ${RFDETR_TEST_FIXTURES}
    DEPENDS gen_fixture
    VERBATIM
)
add_custom_target(rfdetr_fixtures ALL DEPENDS ${RFDETR_TEST_FIXTURES}/cats.png)
```

Add `target_compile_definitions(gen_fixture PRIVATE STB_IMAGE_WRITE_IMPLEMENTATION)` — actually the `#define` is already in the source, so no extra CMake flag needed.

- [ ] **Step 2: Write the failing test `tests/test_image_io.cpp`**

```cpp
#include "test_assert.hpp"
#include "rfdetr.h"
#include "image_io.hpp"
#include <string>

int main() {
    const std::string fixtures = RFDETR_TEST_FIXTURES;
    const std::string path = fixtures + "/cats.png";

    rfdetr_status st;
    rfdetr_image* img = rfdetr_image_load_file(path.c_str(), &st);
    RFDETR_ASSERT(img != nullptr);
    RFDETR_ASSERT(st == RFDETR_OK);

    RFDETR_ASSERT_EQ_INT(rfdetr_image_width(img),  16);
    RFDETR_ASSERT_EQ_INT(rfdetr_image_height(img), 16);

    // Check a known pixel via the internal accessor (declared in image_io.hpp)
    const uint8_t* rgb = rfdetr_image_rgb_data(img);
    RFDETR_ASSERT(rgb != nullptr);
    // pixel (1,2): R=16, G=32, B=2  (from the generator formula)
    int i = (2*16 + 1) * 3;
    RFDETR_ASSERT_EQ_INT(rgb[i+0], 16);
    RFDETR_ASSERT_EQ_INT(rgb[i+1], 32);
    RFDETR_ASSERT_EQ_INT(rgb[i+2], 2);

    rfdetr_image_free(img);

    // Non-existent file → returns nullptr + status set
    rfdetr_image* img2 = rfdetr_image_load_file("/no/such/file.png", &st);
    RFDETR_ASSERT(img2 == nullptr);
    RFDETR_ASSERT(st == RFDETR_ERR_FILE_NOT_FOUND || st == RFDETR_ERR_IO || st == RFDETR_ERR_DECODE);

    // Buffer load: write a tiny PNG via stb on the fly and load it
    // (skip — covered by file path; buffer variant test is in Task 8)

    return 0;
}
```

- [ ] **Step 3: Run to confirm failure**

```bash
cmake -B build -DRFDETR_BUILD_TESTS=ON
cmake --build build --target test_image_io -j 2>&1 | tail -20
```

Expected: link errors (`rfdetr_image_load_file`, `rfdetr_image_width`, `rfdetr_image_rgb_data` undefined).

- [ ] **Step 4: Write `src/image_io.hpp`**

```cpp
#ifndef RFDETR_IMAGE_IO_HPP
#define RFDETR_IMAGE_IO_HPP

#include "rfdetr.h"

#include <cstdint>
#include <vector>

struct rfdetr_image {
    int width = 0;
    int height = 0;
    int channels = 3;            /* always RGB after load */
    std::vector<uint8_t> rgb;    /* row-major, HWC, 0..255 */
};

/* C++-only accessors (used by other modules in this codebase). */
const uint8_t* rfdetr_image_rgb_data(const rfdetr_image* img);

#endif
```

Also re-export the accessor in C linkage for the test:

```cpp
extern "C" const uint8_t* rfdetr_image_rgb_data(const rfdetr_image* img);
```

(Put the `extern "C"` form in `image_io.hpp` so the test can `extern "C"` declare it without C++ name mangling.)

Final `image_io.hpp`:

```cpp
#ifndef RFDETR_IMAGE_IO_HPP
#define RFDETR_IMAGE_IO_HPP

#include "rfdetr.h"

#include <cstdint>
#include <vector>

struct rfdetr_image {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<uint8_t> rgb;  /* HWC, row-major, 0..255 */
};

#ifdef __cplusplus
extern "C" {
#endif

const uint8_t* rfdetr_image_rgb_data(const rfdetr_image* img);

#ifdef __cplusplus
}
#endif

#endif
```

- [ ] **Step 5: Write `src/image_io.cpp`**

```cpp
#include "image_io.hpp"
#include "common.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <vector>

extern "C" rfdetr_image* rfdetr_image_load_buffer(const uint8_t* bytes, size_t len, rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) { if (out_status) *out_status = s; };
    if (!bytes || len == 0) { set(RFDETR_ERR_INVALID_ARG); return nullptr; }

    int w = 0, h = 0, c = 0;
    uint8_t* px = stbi_load_from_memory(bytes, (int)len, &w, &h, &c, 3 /* force RGB */);
    if (!px) {
        rfdetr_logf(RFDETR_LOG_ERROR, "stbi_load_from_memory failed: %s", stbi_failure_reason());
        set(RFDETR_ERR_DECODE);
        return nullptr;
    }

    auto* img = new (std::nothrow) rfdetr_image();
    if (!img) { stbi_image_free(px); set(RFDETR_ERR_OUT_OF_MEMORY); return nullptr; }
    img->width    = w;
    img->height   = h;
    img->channels = 3;
    img->rgb.assign(px, px + (size_t)w * h * 3);
    stbi_image_free(px);
    set(RFDETR_OK);
    return img;
}

extern "C" rfdetr_image* rfdetr_image_load_file(const char* path, rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) { if (out_status) *out_status = s; };
    if (!path) { set(RFDETR_ERR_INVALID_ARG); return nullptr; }

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "image_load_file: cannot open '%s' (%s)", path, std::strerror(errno));
        set(RFDETR_ERR_FILE_NOT_FOUND);
        return nullptr;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.empty()) { set(RFDETR_ERR_IO); return nullptr; }

    return rfdetr_image_load_buffer(buf.data(), buf.size(), out_status);
}

extern "C" void rfdetr_image_free(rfdetr_image* img) {
    delete img;
}

extern "C" int rfdetr_image_width(const rfdetr_image* img) {
    return img ? img->width : 0;
}

extern "C" int rfdetr_image_height(const rfdetr_image* img) {
    return img ? img->height : 0;
}

extern "C" const uint8_t* rfdetr_image_rgb_data(const rfdetr_image* img) {
    return img ? img->rgb.data() : nullptr;
}
```

- [ ] **Step 6: Build and run the test**

```bash
cmake --build build --target test_image_io -j
ctest --test-dir build -R test_image_io --output-on-failure
```

Expected: `test_image_io ... Passed`.

- [ ] **Step 7: Commit**

```bash
git add src/image_io.hpp src/image_io.cpp tests/test_image_io.cpp tests/fixtures/gen_fixture.cpp tests/CMakeLists.txt
git commit -m "feat(image_io): load PNG/JPEG to RGB buffer via stb_image + tests"
```

---

### Task 8: image_io — buffer load + render PNG

**Files:**
- Modify: `src/image_io.cpp` (add `rfdetr_render`)
- Modify: `tests/test_image_io.cpp` (add tests)

- [ ] **Step 1: Extend `tests/test_image_io.cpp`** — add to `main()` before `return 0;`:

```cpp
    // ---- Buffer load: round-trip a tiny in-memory PNG ----
    {
        // Re-encode our 16x16 fixture into PNG bytes via stb_image_write
        // (use the on-disk file we already have)
        std::string fixture_path = fixtures + "/cats.png";
        std::ifstream f(fixture_path, std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                    std::istreambuf_iterator<char>());
        RFDETR_ASSERT(!bytes.empty());

        rfdetr_status st2;
        rfdetr_image* img = rfdetr_image_load_buffer(bytes.data(), bytes.size(), &st2);
        RFDETR_ASSERT(st2 == RFDETR_OK);
        RFDETR_ASSERT(img != nullptr);
        RFDETR_ASSERT_EQ_INT(rfdetr_image_width(img),  16);
        RFDETR_ASSERT_EQ_INT(rfdetr_image_height(img), 16);
        rfdetr_image_free(img);
    }

    // ---- Render round-trip: load → render with zero detections → reload ----
    {
        rfdetr_status st3;
        rfdetr_image* img = rfdetr_image_load_file((fixtures + "/cats.png").c_str(), &st3);
        RFDETR_ASSERT(st3 == RFDETR_OK);

        const std::string out_path = std::string(fixtures) + "/generated/cats_out.png";
        // ensure dir exists (single mkdir is fine — test harness can pre-create it,
        // but we exercise via the function itself)
        std::string mkdir_cmd = "mkdir -p " + std::string(fixtures) + "/generated";
        std::system(mkdir_cmd.c_str());

        rfdetr_status r = rfdetr_render(img, nullptr, 0, out_path.c_str());
        RFDETR_ASSERT(r == RFDETR_OK);

        // Re-load the output and check dims match
        rfdetr_status st4;
        rfdetr_image* img2 = rfdetr_image_load_file(out_path.c_str(), &st4);
        RFDETR_ASSERT(st4 == RFDETR_OK);
        RFDETR_ASSERT_EQ_INT(rfdetr_image_width(img2),  rfdetr_image_width(img));
        RFDETR_ASSERT_EQ_INT(rfdetr_image_height(img2), rfdetr_image_height(img));
        rfdetr_image_free(img);
        rfdetr_image_free(img2);
    }
```

Add `#include <fstream>`, `#include <iterator>`, `#include <cstdlib>` to the test if not present.

- [ ] **Step 2: Run to confirm failure**

```bash
cmake --build build --target test_image_io -j 2>&1 | tail -10
```

Expected: link error `undefined reference to rfdetr_render`. (`rfdetr_image_load_buffer` is already implemented from Task 7.)

- [ ] **Step 3: Add `rfdetr_render` to `src/image_io.cpp`** (append at end)

```cpp
extern "C" rfdetr_status rfdetr_render(const rfdetr_image* img,
                                       const rfdetr_detection* /*detections*/, size_t /*n*/,
                                       const char* out_path) {
    if (!img || !out_path) return RFDETR_ERR_INVALID_ARG;

    /* For this plan we copy the image through unchanged (no detections drawn).
     * Task 11 adds bbox drawing; labels come in Plan 2. */
    int w = img->width, h = img->height;
    if (!stbi_write_png(out_path, w, h, 3, img->rgb.data(), w * 3)) {
        rfdetr_logf(RFDETR_LOG_ERROR, "stbi_write_png failed for '%s'", out_path);
        return RFDETR_ERR_IO;
    }
    return RFDETR_OK;
}
```

- [ ] **Step 4: Build and run**

```bash
cmake --build build --target test_image_io -j
ctest --test-dir build -R test_image_io --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/image_io.cpp tests/test_image_io.cpp
git commit -m "feat(image_io): rfdetr_render (no-detection passthrough) + buffer-load test"
```

---

### Task 9: postprocess — bbox cxcywh → xyxy

**Files:**
- Create: `src/postprocess.hpp`
- Modify: `src/postprocess.cpp` (replace placeholder)
- Modify: `tests/test_postprocess.cpp` (new content)

- [ ] **Step 1: Write the failing test `tests/test_postprocess.cpp`**

```cpp
#include "test_assert.hpp"
#include "postprocess.hpp"
#include "rfdetr.h"
#include <cmath>
#include <vector>

int main() {
    // ---- bbox_cxcywh_to_xyxy ----
    // (cx=0.5, cy=0.5, w=0.4, h=0.6) on a 200x100 image
    // → x1 = (0.5 - 0.2) * 200 = 60
    //   y1 = (0.5 - 0.3) * 100 = 20
    //   x2 = (0.5 + 0.2) * 200 = 140
    //   y2 = (0.5 + 0.3) * 100 = 80
    {
        float in[4]  = {0.5f, 0.5f, 0.4f, 0.6f};
        float out[4];
        rfdetr_bbox_cxcywh_to_xyxy(in, 200, 100, out);
        RFDETR_ASSERT_NEAR(out[0],  60.0f, 1e-4);
        RFDETR_ASSERT_NEAR(out[1],  20.0f, 1e-4);
        RFDETR_ASSERT_NEAR(out[2], 140.0f, 1e-4);
        RFDETR_ASSERT_NEAR(out[3],  80.0f, 1e-4);
    }

    // Clamping to image bounds: out-of-range boxes clip to [0..W, 0..H]
    {
        float in[4]  = {0.5f, 0.5f, 2.0f, 2.0f};  /* far larger than image */
        float out[4];
        rfdetr_bbox_cxcywh_to_xyxy(in, 100, 50, out);
        RFDETR_ASSERT_NEAR(out[0],   0.0f, 1e-4);
        RFDETR_ASSERT_NEAR(out[1],   0.0f, 1e-4);
        RFDETR_ASSERT_NEAR(out[2], 100.0f, 1e-4);
        RFDETR_ASSERT_NEAR(out[3],  50.0f, 1e-4);
    }

    return 0;
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
cmake --build build --target test_postprocess -j 2>&1 | tail -10
```

Expected: linker error, `rfdetr_bbox_cxcywh_to_xyxy` undefined.

- [ ] **Step 3: Write `src/postprocess.hpp`**

```cpp
#ifndef RFDETR_POSTPROCESS_HPP
#define RFDETR_POSTPROCESS_HPP

#include "rfdetr.h"
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

/* Convert a single normalized (cx, cy, w, h) box to pixel-space (x1, y1, x2, y2),
 * clamped to [0, img_w] x [0, img_h]. Input and output may not alias. */
void rfdetr_bbox_cxcywh_to_xyxy(const float in[4], int img_w, int img_h, float out[4]);

#ifdef __cplusplus
}
#endif

#endif
```

- [ ] **Step 4: Implement in `src/postprocess.cpp`** (replace placeholder)

```cpp
#include "postprocess.hpp"
#include <algorithm>

extern "C" void rfdetr_bbox_cxcywh_to_xyxy(const float in[4], int img_w, int img_h, float out[4]) {
    const float cx = in[0], cy = in[1], w = in[2], h = in[3];
    float x1 = (cx - 0.5f * w) * (float)img_w;
    float y1 = (cy - 0.5f * h) * (float)img_h;
    float x2 = (cx + 0.5f * w) * (float)img_w;
    float y2 = (cy + 0.5f * h) * (float)img_h;
    out[0] = std::clamp(x1, 0.0f, (float)img_w);
    out[1] = std::clamp(y1, 0.0f, (float)img_h);
    out[2] = std::clamp(x2, 0.0f, (float)img_w);
    out[3] = std::clamp(y2, 0.0f, (float)img_h);
}
```

- [ ] **Step 5: Build and run**

```bash
cmake --build build --target test_postprocess -j
ctest --test-dir build -R test_postprocess --output-on-failure
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/postprocess.hpp src/postprocess.cpp tests/test_postprocess.cpp
git commit -m "feat(postprocess): bbox cxcywh -> xyxy with clamping + tests"
```

---

### Task 10: postprocess — sigmoid, top-k, threshold, class filter

**Files:**
- Modify: `src/postprocess.hpp`, `src/postprocess.cpp`
- Modify: `tests/test_postprocess.cpp`

- [ ] **Step 1: Extend `src/postprocess.hpp`** (add inside `extern "C"`)

```cpp
/* Score selection.
 *
 * Inputs:
 *   class_logits     [num_queries * num_classes] — raw logits, row-major (query, class)
 *   bbox_cxcywh      [num_queries * 4]            — normalized predicted boxes
 *   num_queries, num_classes
 *   threshold        — drop predictions with sigmoid(logit) <= threshold
 *   top_k            — keep at most this many predictions (after threshold)
 *   class_filter     — optional allowlist of class ids; NULL = all classes
 *   class_filter_len — length of allowlist
 *   img_w, img_h     — original image dimensions, for xyxy projection
 *
 * Output:
 *   detections — vector cleared and populated by the function. class_name is
 *   set to nullptr; caller may attach names from the model's class list.
 */
void rfdetr_select_detections(const float* class_logits,
                              const float* bbox_cxcywh,
                              size_t num_queries, size_t num_classes,
                              float threshold, uint32_t top_k,
                              const uint32_t* class_filter, size_t class_filter_len,
                              int img_w, int img_h,
                              rfdetr_detection** out_detections, size_t* out_n);
```

- [ ] **Step 2: Extend `tests/test_postprocess.cpp`** — add before `return 0;`:

```cpp
    // ---- rfdetr_select_detections ----
    //
    // 3 queries × 4 classes. We construct logits so that, after sigmoid:
    //   query 0 → class 2 with score ~0.95
    //   query 1 → class 0 with score ~0.7
    //   query 2 → class 3 with score ~0.1  (below threshold 0.5)
    {
        const size_t Q = 3, C = 4;

        // logit such that sigmoid(logit) ≈ desired score:
        auto logit_of = [](float s) { return std::log(s / (1.0f - s)); };

        float logits[Q*C] = {
            /* query 0 */ -3.0f, -3.0f, logit_of(0.95f), -3.0f,
            /* query 1 */ logit_of(0.70f), -3.0f, -3.0f, -3.0f,
            /* query 2 */ -3.0f, -3.0f, -3.0f, logit_of(0.10f),
        };
        float boxes[Q*4] = {
            0.5f, 0.5f, 0.2f, 0.2f,   /* query 0 */
            0.25f, 0.25f, 0.1f, 0.1f, /* query 1 */
            0.5f, 0.5f, 0.5f, 0.5f,   /* query 2 */
        };

        rfdetr_detection* dets = nullptr;
        size_t n = 0;
        rfdetr_select_detections(logits, boxes,
                                 Q, C,
                                 /*threshold*/ 0.5f, /*top_k*/ 10,
                                 /*class_filter*/ nullptr, 0,
                                 /*img_w*/ 100, /*img_h*/ 100,
                                 &dets, &n);

        RFDETR_ASSERT_EQ_INT(n, 2);                  // query 2 dropped (score 0.10 < 0.5)
        // Sorted by score descending: query 0 (0.95) first, query 1 (0.70) second
        RFDETR_ASSERT_EQ_INT(dets[0].class_id, 2);
        RFDETR_ASSERT_NEAR(dets[0].score, 0.95f, 1e-3);
        RFDETR_ASSERT_EQ_INT(dets[1].class_id, 0);
        RFDETR_ASSERT_NEAR(dets[1].score, 0.70f, 1e-3);

        // Box of query 0: cx=cy=0.5, w=h=0.2 on a 100x100 image → (40,40,60,60)
        RFDETR_ASSERT_NEAR(dets[0].x1, 40.0f, 1e-4);
        RFDETR_ASSERT_NEAR(dets[0].y1, 40.0f, 1e-4);
        RFDETR_ASSERT_NEAR(dets[0].x2, 60.0f, 1e-4);
        RFDETR_ASSERT_NEAR(dets[0].y2, 60.0f, 1e-4);

        rfdetr_detections_free(dets, n);
    }

    // Top-K cap
    {
        const size_t Q = 5, C = 1;
        float logits[Q*C];
        for (size_t i = 0; i < Q; ++i) {
            // sigmoid(0) = 0.5 + small perturbation so scores differ
            logits[i] = (float)i * 0.01f;
        }
        float boxes[Q*4] = {0};
        for (size_t i = 0; i < Q; ++i) {
            boxes[i*4 + 0] = 0.5f;
            boxes[i*4 + 1] = 0.5f;
            boxes[i*4 + 2] = 0.1f;
            boxes[i*4 + 3] = 0.1f;
        }
        rfdetr_detection* dets = nullptr;
        size_t n = 0;
        rfdetr_select_detections(logits, boxes, Q, C,
                                 /*threshold*/ 0.0f, /*top_k*/ 3,
                                 nullptr, 0, 100, 100, &dets, &n);
        RFDETR_ASSERT_EQ_INT(n, 3);
        rfdetr_detections_free(dets, n);
    }

    // Class filter
    {
        const size_t Q = 2, C = 3;
        auto logit_of = [](float s) { return std::log(s / (1.0f - s)); };
        float logits[Q*C] = {
            logit_of(0.9f), -3.0f, -3.0f,   /* query 0 → class 0 */
            -3.0f, logit_of(0.9f), -3.0f,   /* query 1 → class 1 */
        };
        float boxes[Q*4] = {
            0.5f, 0.5f, 0.1f, 0.1f,
            0.5f, 0.5f, 0.1f, 0.1f,
        };
        uint32_t allow[1] = {1};   /* only class 1 */
        rfdetr_detection* dets = nullptr;
        size_t n = 0;
        rfdetr_select_detections(logits, boxes, Q, C, 0.5f, 10,
                                 allow, 1, 100, 100, &dets, &n);
        RFDETR_ASSERT_EQ_INT(n, 1);
        RFDETR_ASSERT_EQ_INT(dets[0].class_id, 1);
        rfdetr_detections_free(dets, n);
    }
```

- [ ] **Step 3: Run to confirm failure**

```bash
cmake --build build --target test_postprocess -j 2>&1 | tail -10
```

Expected: link error for `rfdetr_select_detections` and `rfdetr_detections_free`.

- [ ] **Step 4: Implement in `src/postprocess.cpp`** (append)

```cpp
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

inline float sigmoidf(float x) {
    /* Numerically stable: avoid overflow on large negative x. */
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = std::exp(x);
        return z / (1.0f + z);
    }
}

struct Candidate {
    uint32_t query;
    uint32_t class_id;
    float    score;
};

}  // namespace

extern "C" void rfdetr_select_detections(const float* class_logits,
                                         const float* bbox_cxcywh,
                                         size_t num_queries, size_t num_classes,
                                         float threshold, uint32_t top_k,
                                         const uint32_t* class_filter, size_t class_filter_len,
                                         int img_w, int img_h,
                                         rfdetr_detection** out_detections, size_t* out_n) {
    *out_detections = nullptr;
    *out_n = 0;

    if (!class_logits || !bbox_cxcywh || num_queries == 0 || num_classes == 0) return;

    /* Fast-membership lookup for the class filter. For small allowlists this
     * linear scan is fine; revisit if num_classes grows. */
    auto class_allowed = [&](uint32_t cid) -> bool {
        if (!class_filter || class_filter_len == 0) return true;
        for (size_t i = 0; i < class_filter_len; ++i) {
            if (class_filter[i] == cid) return true;
        }
        return false;
    };

    std::vector<Candidate> cands;
    cands.reserve(num_queries);

    for (size_t q = 0; q < num_queries; ++q) {
        /* Take argmax over classes after sigmoid (DETR convention: per-class
         * sigmoid, not softmax across classes). */
        const float* row = class_logits + q * num_classes;
        uint32_t best_c = 0;
        float    best_s = -1.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            float s = sigmoidf(row[c]);
            if (s > best_s) { best_s = s; best_c = (uint32_t)c; }
        }
        if (best_s <= threshold) continue;
        if (!class_allowed(best_c)) continue;
        cands.push_back({(uint32_t)q, best_c, best_s});
    }

    /* Sort by score descending. */
    std::sort(cands.begin(), cands.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    if (top_k > 0 && cands.size() > top_k) cands.resize(top_k);
    if (cands.empty()) return;

    auto* out = (rfdetr_detection*)std::calloc(cands.size(), sizeof(rfdetr_detection));
    if (!out) return;

    for (size_t i = 0; i < cands.size(); ++i) {
        const auto& c = cands[i];
        out[i].class_id   = c.class_id;
        out[i].class_name = nullptr;
        out[i].score      = c.score;
        float box[4];
        rfdetr_bbox_cxcywh_to_xyxy(bbox_cxcywh + c.query * 4, img_w, img_h, box);
        out[i].x1 = box[0];
        out[i].y1 = box[1];
        out[i].x2 = box[2];
        out[i].y2 = box[3];
    }
    *out_detections = out;
    *out_n          = cands.size();
}

extern "C" void rfdetr_detections_free(rfdetr_detection* detections, size_t /*n*/) {
    std::free(detections);
}
```

- [ ] **Step 5: Build and run**

```bash
cmake --build build --target test_postprocess -j
ctest --test-dir build -R test_postprocess --output-on-failure
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/postprocess.hpp src/postprocess.cpp tests/test_postprocess.cpp
git commit -m "feat(postprocess): sigmoid + top-k + threshold + class filter"
```

---

### Task 11: visualize — draw bounding boxes

**Files:**
- Create: `src/visualize.hpp`
- Modify: `src/visualize.cpp` (replace placeholder)
- Modify: `src/image_io.cpp` (call into visualize from rfdetr_render)
- Create: `tests/test_visualize.cpp`

- [ ] **Step 1: Write `tests/test_visualize.cpp`**

```cpp
#include "test_assert.hpp"
#include "rfdetr.h"
#include "image_io.hpp"
#include "visualize.hpp"
#include <string>
#include <cstdlib>

int main() {
    // Synthesize a 32x32 black image, draw one red box, verify pixels on the
    // box outline became red.
    rfdetr_image img;
    img.width = 32;
    img.height = 32;
    img.channels = 3;
    img.rgb.assign(32*32*3, 0);

    rfdetr_detection d{};
    d.class_id = 0;
    d.score = 0.99f;
    d.x1 = 4.0f; d.y1 = 4.0f; d.x2 = 28.0f; d.y2 = 28.0f;

    rfdetr_visualize_draw_box(&img, d, 2 /* thickness */);

    auto px = [&](int x, int y, int c) {
        return img.rgb[(y*32 + x)*3 + c];
    };

    // Top edge: y = 4, x = 4..27 → red
    RFDETR_ASSERT_EQ_INT(px(10, 4, 0), 255);
    RFDETR_ASSERT_EQ_INT(px(10, 4, 1), 0);
    RFDETR_ASSERT_EQ_INT(px(10, 4, 2), 0);

    // Pixel well outside the box → still black
    RFDETR_ASSERT_EQ_INT(px(0, 0, 0), 0);
    RFDETR_ASSERT_EQ_INT(px(0, 0, 1), 0);
    RFDETR_ASSERT_EQ_INT(px(0, 0, 2), 0);

    // Interior of the box (not on the edge) → still black (we draw outline only)
    RFDETR_ASSERT_EQ_INT(px(16, 16, 0), 0);

    return 0;
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
cmake --build build --target test_visualize -j 2>&1 | tail -10
```

Expected: link error for `rfdetr_visualize_draw_box`.

- [ ] **Step 3: Write `src/visualize.hpp`**

```cpp
#ifndef RFDETR_VISUALIZE_HPP
#define RFDETR_VISUALIZE_HPP

#include "rfdetr.h"
#include "image_io.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/* Draw a single detection's bounding box (red, hollow rectangle) on the
 * image. The image is mutated in place. Box coords are in pixel space;
 * out-of-bounds pixels are clipped. */
void rfdetr_visualize_draw_box(rfdetr_image* img, rfdetr_detection det, int thickness);

#ifdef __cplusplus
}
#endif

#endif
```

- [ ] **Step 4: Write `src/visualize.cpp`**

```cpp
#include "visualize.hpp"

#include <algorithm>
#include <cmath>

namespace {

inline void set_px(rfdetr_image* img, int x, int y,
                   uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || y < 0 || x >= img->width || y >= img->height) return;
    size_t i = ((size_t)y * img->width + x) * 3;
    img->rgb[i + 0] = r;
    img->rgb[i + 1] = g;
    img->rgb[i + 2] = b;
}

}  // namespace

extern "C" void rfdetr_visualize_draw_box(rfdetr_image* img, rfdetr_detection det, int thickness) {
    if (!img || thickness <= 0) return;

    int x1 = (int)std::lround(det.x1);
    int y1 = (int)std::lround(det.y1);
    int x2 = (int)std::lround(det.x2);
    int y2 = (int)std::lround(det.y2);
    if (x2 < x1) std::swap(x1, x2);
    if (y2 < y1) std::swap(y1, y2);

    const uint8_t R = 255, G = 0, B = 0;
    for (int t = 0; t < thickness; ++t) {
        // top & bottom edges
        for (int x = x1; x <= x2; ++x) {
            set_px(img, x, y1 + t, R, G, B);
            set_px(img, x, y2 - t, R, G, B);
        }
        // left & right edges
        for (int y = y1; y <= y2; ++y) {
            set_px(img, x1 + t, y, R, G, B);
            set_px(img, x2 - t, y, R, G, B);
        }
    }
}
```

- [ ] **Step 5: Wire `rfdetr_render` to draw boxes**

Replace the body of `rfdetr_render` in `src/image_io.cpp`:

```cpp
extern "C" rfdetr_status rfdetr_render(const rfdetr_image* img,
                                       const rfdetr_detection* detections, size_t n,
                                       const char* out_path) {
    if (!img || !out_path) return RFDETR_ERR_INVALID_ARG;

    /* Copy so we don't mutate the caller's image. */
    rfdetr_image copy = *img;
    for (size_t i = 0; i < n; ++i) {
        rfdetr_visualize_draw_box(&copy, detections[i], /*thickness*/ 2);
    }

    int w = copy.width, h = copy.height;
    if (!stbi_write_png(out_path, w, h, 3, copy.rgb.data(), w * 3)) {
        rfdetr_logf(RFDETR_LOG_ERROR, "stbi_write_png failed for '%s'", out_path);
        return RFDETR_ERR_IO;
    }
    return RFDETR_OK;
}
```

Add `#include "visualize.hpp"` at the top of `src/image_io.cpp`.

- [ ] **Step 6: Build and run**

```bash
cmake --build build --target test_visualize test_image_io -j
ctest --test-dir build -R "test_visualize|test_image_io" --output-on-failure
```

Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
git add src/visualize.hpp src/visualize.cpp src/image_io.cpp tests/test_visualize.cpp
git commit -m "feat(visualize): draw bounding boxes; wire into rfdetr_render"
```

---

### Task 12: CLI argument parser (hand-rolled, zero deps)

**Files:**
- Create: `src/cli.hpp`
- Modify: `src/cli.cpp` (replace placeholder)
- Modify: `tests/test_cli_smoke.cpp`

- [ ] **Step 1: Write `src/cli.hpp`**

```cpp
#ifndef RFDETR_CLI_HPP
#define RFDETR_CLI_HPP

#include <string>
#include <vector>

namespace rfdetr_cli {

struct DetectArgs {
    std::string model;
    std::string input;
    std::string output;
    std::string annotated;
    float       threshold = 0.5f;
    uint32_t    top_k     = 300;
    std::vector<uint32_t> classes;  /* allowlist */
};

struct InfoArgs    { std::string model; };
struct BenchArgs   { std::string model; std::string input; int iters = 50; int warmup = 5; };
struct CompareArgs { std::string baseline_dir; std::string image; std::string model; };

enum class Subcommand { None, Help, Detect, Info, Bench, Compare };

struct ParseResult {
    Subcommand   sub = Subcommand::None;
    std::string  error;        /* non-empty on parse failure */
    DetectArgs   detect;
    InfoArgs     info;
    BenchArgs    bench;
    CompareArgs  compare;
};

/* Parse argv. argv[0] is the program name. */
ParseResult parse(int argc, char** argv);

/* Print --help text to stdout. */
void print_help();

}  // namespace rfdetr_cli

#endif
```

- [ ] **Step 2: Write the failing test `tests/test_cli_smoke.cpp`**

```cpp
#include "test_assert.hpp"
#include "cli.hpp"
#include <vector>
#include <cstring>

static rfdetr_cli::ParseResult run(std::vector<const char*> argv) {
    return rfdetr_cli::parse((int)argv.size(), const_cast<char**>(argv.data()));
}

int main() {
    // No args → Help
    {
        auto r = run({"rfdetr-cli"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Help);
        RFDETR_ASSERT(r.error.empty());
    }

    // --help → Help
    {
        auto r = run({"rfdetr-cli", "--help"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Help);
    }

    // detect with required args
    {
        auto r = run({"rfdetr-cli", "detect",
                      "--model", "m.gguf",
                      "--input", "cats.png",
                      "--output", "out.json",
                      "--annotated", "out.png",
                      "--threshold", "0.4",
                      "--topk", "100",
                      "--classes", "0,16,17"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Detect);
        RFDETR_ASSERT(r.error.empty());
        RFDETR_ASSERT_STR_EQ(r.detect.model.c_str(),     "m.gguf");
        RFDETR_ASSERT_STR_EQ(r.detect.input.c_str(),     "cats.png");
        RFDETR_ASSERT_STR_EQ(r.detect.output.c_str(),    "out.json");
        RFDETR_ASSERT_STR_EQ(r.detect.annotated.c_str(), "out.png");
        RFDETR_ASSERT_NEAR(r.detect.threshold, 0.4f, 1e-5);
        RFDETR_ASSERT_EQ_INT(r.detect.top_k, 100);
        RFDETR_ASSERT_EQ_INT(r.detect.classes.size(), 3);
        RFDETR_ASSERT_EQ_INT(r.detect.classes[0], 0);
        RFDETR_ASSERT_EQ_INT(r.detect.classes[1], 16);
        RFDETR_ASSERT_EQ_INT(r.detect.classes[2], 17);
    }

    // detect with missing required arg → error
    {
        auto r = run({"rfdetr-cli", "detect", "--model", "m.gguf"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Detect);
        RFDETR_ASSERT(!r.error.empty());
    }

    // Unknown subcommand → error
    {
        auto r = run({"rfdetr-cli", "nope"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::None);
        RFDETR_ASSERT(!r.error.empty());
    }

    // info
    {
        auto r = run({"rfdetr-cli", "info", "--model", "m.gguf"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Info);
        RFDETR_ASSERT_STR_EQ(r.info.model.c_str(), "m.gguf");
    }

    // bench
    {
        auto r = run({"rfdetr-cli", "bench",
                      "--model", "m.gguf", "--input", "cats.png",
                      "--iters", "10", "--warmup", "2"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Bench);
        RFDETR_ASSERT_EQ_INT(r.bench.iters, 10);
        RFDETR_ASSERT_EQ_INT(r.bench.warmup, 2);
    }

    // compare
    {
        auto r = run({"rfdetr-cli", "compare",
                      "--baseline", "bundle/", "--image", "cats.png", "--model", "m.gguf"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Compare);
        RFDETR_ASSERT_STR_EQ(r.compare.baseline_dir.c_str(), "bundle/");
        RFDETR_ASSERT_STR_EQ(r.compare.image.c_str(),        "cats.png");
        RFDETR_ASSERT_STR_EQ(r.compare.model.c_str(),        "m.gguf");
    }

    return 0;
}
```

- [ ] **Step 3: Run to confirm failure**

```bash
cmake --build build --target test_cli_smoke -j 2>&1 | tail -10
```

Expected: link error — `rfdetr_cli::parse` undefined.

- [ ] **Step 4: Write `src/cli.cpp`** (replace placeholder)

```cpp
#include "cli.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace rfdetr_cli {

namespace {

bool eat_value(int argc, char** argv, int& i, const char* flag, std::string& out, std::string& err) {
    if (i + 1 >= argc) {
        err = std::string("missing value for ") + flag;
        return false;
    }
    out = argv[++i];
    return true;
}

bool parse_uint(const std::string& s, uint32_t& out, std::string& err, const char* flag) {
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || v < 0) {
        err = std::string("invalid integer for ") + flag + ": " + s;
        return false;
    }
    out = (uint32_t)v;
    return true;
}

bool parse_float(const std::string& s, float& out, std::string& err, const char* flag) {
    char* end = nullptr;
    float v = std::strtof(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') {
        err = std::string("invalid float for ") + flag + ": " + s;
        return false;
    }
    out = v;
    return true;
}

bool parse_classes(const std::string& s, std::vector<uint32_t>& out, std::string& err) {
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        uint32_t v;
        if (!parse_uint(tok, v, err, "--classes")) return false;
        out.push_back(v);
    }
    return true;
}

}  // namespace

void print_help() {
    std::puts(
        "rfdetr-cli — RF-DETR inference CLI\n"
        "\n"
        "Usage:\n"
        "  rfdetr-cli detect  --model <gguf> --input <image> --output <json>\n"
        "                     [--annotated <png>] [--threshold N] [--topk N]\n"
        "                     [--classes id,id,...]\n"
        "  rfdetr-cli info    --model <gguf>\n"
        "  rfdetr-cli bench   --model <gguf> --input <image> [--iters N] [--warmup N]\n"
        "  rfdetr-cli compare --baseline <dir> --image <png> --model <gguf>\n"
        "  rfdetr-cli --help\n");
}

ParseResult parse(int argc, char** argv) {
    ParseResult r;

    if (argc < 2) { r.sub = Subcommand::Help; return r; }

    std::string first = argv[1];
    if (first == "--help" || first == "-h") { r.sub = Subcommand::Help; return r; }

    if (first == "detect") {
        r.sub = Subcommand::Detect;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--model")     { if (!eat_value(argc, argv, i, "--model",    r.detect.model,     r.error)) return r; }
            else if (a == "--input")     { if (!eat_value(argc, argv, i, "--input",    r.detect.input,     r.error)) return r; }
            else if (a == "--output")    { if (!eat_value(argc, argv, i, "--output",   r.detect.output,    r.error)) return r; }
            else if (a == "--annotated") { if (!eat_value(argc, argv, i, "--annotated",r.detect.annotated, r.error)) return r; }
            else if (a == "--threshold") {
                std::string v; if (!eat_value(argc, argv, i, "--threshold", v, r.error)) return r;
                if (!parse_float(v, r.detect.threshold, r.error, "--threshold")) return r;
            }
            else if (a == "--topk") {
                std::string v; if (!eat_value(argc, argv, i, "--topk", v, r.error)) return r;
                if (!parse_uint(v, r.detect.top_k, r.error, "--topk")) return r;
            }
            else if (a == "--classes") {
                std::string v; if (!eat_value(argc, argv, i, "--classes", v, r.error)) return r;
                if (!parse_classes(v, r.detect.classes, r.error)) return r;
            }
            else {
                r.error = "unknown flag: " + a;
                return r;
            }
        }
        if (r.detect.model.empty())  { r.error = "detect: --model is required";  return r; }
        if (r.detect.input.empty())  { r.error = "detect: --input is required";  return r; }
        if (r.detect.output.empty()) { r.error = "detect: --output is required"; return r; }
        return r;
    }

    if (first == "info") {
        r.sub = Subcommand::Info;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--model") { if (!eat_value(argc, argv, i, "--model", r.info.model, r.error)) return r; }
            else { r.error = "unknown flag: " + a; return r; }
        }
        if (r.info.model.empty()) { r.error = "info: --model is required"; return r; }
        return r;
    }

    if (first == "bench") {
        r.sub = Subcommand::Bench;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--model")  { if (!eat_value(argc, argv, i, "--model",  r.bench.model, r.error)) return r; }
            else if (a == "--input")  { if (!eat_value(argc, argv, i, "--input",  r.bench.input, r.error)) return r; }
            else if (a == "--iters")  {
                std::string v; if (!eat_value(argc, argv, i, "--iters",  v, r.error)) return r;
                uint32_t u; if (!parse_uint(v, u, r.error, "--iters")) return r;  r.bench.iters = (int)u;
            }
            else if (a == "--warmup") {
                std::string v; if (!eat_value(argc, argv, i, "--warmup", v, r.error)) return r;
                uint32_t u; if (!parse_uint(v, u, r.error, "--warmup")) return r; r.bench.warmup = (int)u;
            }
            else { r.error = "unknown flag: " + a; return r; }
        }
        if (r.bench.model.empty()) { r.error = "bench: --model is required"; return r; }
        if (r.bench.input.empty()) { r.error = "bench: --input is required"; return r; }
        return r;
    }

    if (first == "compare") {
        r.sub = Subcommand::Compare;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--baseline") { if (!eat_value(argc, argv, i, "--baseline", r.compare.baseline_dir, r.error)) return r; }
            else if (a == "--image")    { if (!eat_value(argc, argv, i, "--image",    r.compare.image,        r.error)) return r; }
            else if (a == "--model")    { if (!eat_value(argc, argv, i, "--model",    r.compare.model,        r.error)) return r; }
            else { r.error = "unknown flag: " + a; return r; }
        }
        if (r.compare.baseline_dir.empty()) { r.error = "compare: --baseline is required"; return r; }
        if (r.compare.image.empty())        { r.error = "compare: --image is required";    return r; }
        if (r.compare.model.empty())        { r.error = "compare: --model is required";    return r; }
        return r;
    }

    r.sub = Subcommand::None;
    r.error = "unknown subcommand: " + first;
    return r;
}

}  // namespace rfdetr_cli
```

- [ ] **Step 5: Build and run**

```bash
cmake --build build --target test_cli_smoke -j
ctest --test-dir build -R test_cli_smoke --output-on-failure
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cli.hpp src/cli.cpp tests/test_cli_smoke.cpp
git commit -m "feat(cli): hand-rolled argument parser with subcommand validation + tests"
```

---

### Task 13: rfdetr-cli main entry point

**Files:**
- Create: `examples/cli/CMakeLists.txt`
- Create: `examples/cli/main.cpp`

- [ ] **Step 1: Write `examples/cli/CMakeLists.txt`**

```cmake
add_executable(rfdetr-cli main.cpp)
target_link_libraries(rfdetr-cli PRIVATE rfdetr)
target_include_directories(rfdetr-cli PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src)
```

- [ ] **Step 2: Write `examples/cli/main.cpp`**

This implements `detect` end-to-end (without a model — Plan 2/3 connects the model). For now `detect` loads the image, calls the postprocessor with an empty input → emits `{"detections": []}` JSON and an annotated PNG copy. `info`, `bench`, `compare` print `not implemented yet`.

```cpp
#include "rfdetr.h"
#include "cli.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

static void default_log_cb(rfdetr_log_level lvl, const char* msg, void* /*ud*/) {
    const char* tag = "?";
    switch (lvl) {
        case RFDETR_LOG_DEBUG: tag = "DEBUG"; break;
        case RFDETR_LOG_INFO:  tag = "INFO";  break;
        case RFDETR_LOG_WARN:  tag = "WARN";  break;
        case RFDETR_LOG_ERROR: tag = "ERROR"; break;
    }
    std::fprintf(stderr, "[%s] %s\n", tag, msg);
}

static int cmd_detect(const rfdetr_cli::DetectArgs& a) {
    rfdetr_status st;
    rfdetr_image* img = rfdetr_image_load_file(a.input.c_str(), &st);
    if (!img) {
        std::fprintf(stderr, "failed to load image '%s': %s\n", a.input.c_str(), rfdetr_status_str(st));
        return 2;
    }

    /* Model loading lives in Plan 2/3. For now we emit zero detections,
     * proving the I/O + output pipeline works end-to-end. */
    rfdetr_detection* dets = nullptr;
    size_t n = 0;

    /* Write JSON */
    std::ofstream out(a.output);
    if (!out.is_open()) {
        std::fprintf(stderr, "failed to open '%s' for writing\n", a.output.c_str());
        rfdetr_image_free(img);
        return 3;
    }
    out << "{\n  \"image\": {\"width\": " << rfdetr_image_width(img)
        << ", \"height\": " << rfdetr_image_height(img) << "},\n"
        << "  \"detections\": [";
    for (size_t i = 0; i < n; ++i) {
        if (i) out << ",";
        out << "\n    {"
            << "\"class_id\": " << dets[i].class_id
            << ", \"score\": " << dets[i].score
            << ", \"x1\": " << dets[i].x1
            << ", \"y1\": " << dets[i].y1
            << ", \"x2\": " << dets[i].x2
            << ", \"y2\": " << dets[i].y2
            << "}";
    }
    out << "\n  ]\n}\n";

    /* Optional annotated PNG */
    if (!a.annotated.empty()) {
        rfdetr_status r = rfdetr_render(img, dets, n, a.annotated.c_str());
        if (r != RFDETR_OK) {
            std::fprintf(stderr, "render failed: %s\n", rfdetr_status_str(r));
            rfdetr_detections_free(dets, n);
            rfdetr_image_free(img);
            return 4;
        }
    }

    rfdetr_detections_free(dets, n);
    rfdetr_image_free(img);
    return 0;
}

int main(int argc, char** argv) {
    rfdetr_set_log_callback(default_log_cb, nullptr);

    auto r = rfdetr_cli::parse(argc, argv);

    if (!r.error.empty()) {
        std::fprintf(stderr, "error: %s\n\n", r.error.c_str());
        rfdetr_cli::print_help();
        return 1;
    }

    switch (r.sub) {
        case rfdetr_cli::Subcommand::Help:
            rfdetr_cli::print_help();
            return 0;
        case rfdetr_cli::Subcommand::Detect:
            return cmd_detect(r.detect);
        case rfdetr_cli::Subcommand::Info:
        case rfdetr_cli::Subcommand::Bench:
        case rfdetr_cli::Subcommand::Compare:
            std::fprintf(stderr, "this subcommand is not yet implemented (see Plan 2/3)\n");
            return 99;
        case rfdetr_cli::Subcommand::None:
            rfdetr_cli::print_help();
            return 1;
    }
    return 1;
}
```

- [ ] **Step 3: Build it**

```bash
cmake --build build --target rfdetr-cli -j
ls -la build/bin/rfdetr-cli
```

Expected: binary exists.

- [ ] **Step 4: Smoke-run it**

```bash
build/bin/rfdetr-cli --help
build/bin/rfdetr-cli detect --model dummy.gguf \
    --input tests/fixtures/cats.png \
    --output /tmp/dets.json \
    --annotated /tmp/dets.png
cat /tmp/dets.json
```

Expected output for the second command: nothing on stdout (exit 0). For `cat /tmp/dets.json`:
```json
{
  "image": {"width": 16, "height": 16},
  "detections": [
  ]
}
```

And `/tmp/dets.png` is a copy of the input (no boxes since no detections).

- [ ] **Step 5: Commit**

```bash
git add examples/cli/CMakeLists.txt examples/cli/main.cpp
git commit -m "feat(cli): rfdetr-cli main entry point with working detect pipeline (no model yet)"
```

---

### Task 14: Integration test — CLI on fixture image

**Files:**
- Modify: `tests/CMakeLists.txt`
- Create: `tests/test_cli_integration.cpp`

A binary-level test that invokes the CLI via `std::system` and asserts its output. This catches end-to-end regressions in the CLI plumbing.

- [ ] **Step 1: Register the new test in `tests/CMakeLists.txt`**

Add at the bottom of `tests/CMakeLists.txt`:

```cmake
rfdetr_add_test(test_cli_integration)
target_compile_definitions(test_cli_integration PRIVATE
    RFDETR_CLI_BINARY="$<TARGET_FILE:rfdetr-cli>")
add_dependencies(test_cli_integration rfdetr-cli)
```

- [ ] **Step 2: Write `tests/test_cli_integration.cpp`**

```cpp
#include "test_assert.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static bool file_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

int main() {
    const std::string fixtures = RFDETR_TEST_FIXTURES;
    const std::string out_json = std::string(fixtures) + "/generated/cli_out.json";
    const std::string out_png  = std::string(fixtures) + "/generated/cli_out.png";

    std::system(("mkdir -p " + std::string(fixtures) + "/generated").c_str());
    std::remove(out_json.c_str());
    std::remove(out_png.c_str());

    std::string cmd = std::string(RFDETR_CLI_BINARY) +
                      " detect --model dummy.gguf"
                      " --input "      + fixtures + "/cats.png"
                      " --output "     + out_json +
                      " --annotated "  + out_png;
    int rc = std::system(cmd.c_str());
    RFDETR_ASSERT_EQ_INT(WEXITSTATUS(rc), 0);

    RFDETR_ASSERT(file_exists(out_json));
    RFDETR_ASSERT(file_exists(out_png));

    std::string body = read_file(out_json);
    RFDETR_ASSERT(body.find("\"detections\": [") != std::string::npos);
    RFDETR_ASSERT(body.find("\"width\": 16")    != std::string::npos);
    RFDETR_ASSERT(body.find("\"height\": 16")   != std::string::npos);

    return 0;
}
```

- [ ] **Step 3: Build and run**

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: all six tests pass (`test_common`, `test_image_io`, `test_postprocess`, `test_visualize`, `test_cli_smoke`, `test_cli_integration`).

- [ ] **Step 4: Commit**

```bash
git add tests/test_cli_integration.cpp tests/CMakeLists.txt
git commit -m "test: end-to-end CLI integration test on fixture image"
```

---

### Task 15: Top-level smoke run + README build instructions

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Verify a clean build from scratch**

```bash
rm -rf build
cmake -B build -DRFDETR_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: all tests pass. Capture the output:

```
Test project /home/.../rt-detr.cpp/build
    Start 1: test_common
1/6 Test #1: test_common ............... Passed
    Start 2: test_image_io
2/6 Test #2: test_image_io ............. Passed
    Start 3: test_postprocess
3/6 Test #3: test_postprocess .......... Passed
    Start 4: test_visualize
4/6 Test #4: test_visualize ............ Passed
    Start 5: test_cli_smoke
5/6 Test #5: test_cli_smoke ............ Passed
    Start 6: test_cli_integration
6/6 Test #6: test_cli_integration ...... Passed

100% tests passed, 0 tests failed out of 6
```

- [ ] **Step 2: Update `README.md` Status section**

Replace the existing Status paragraph with:

```markdown
## Status

**Foundation (Plan 1) complete.** The repo builds, the CLI binary runs
end-to-end on an image (without a model — emits empty detections), and
all six tests pass. No model loading or inference yet — see Plans 2-4
under `docs/superpowers/plans/`.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: mark foundation plan complete in README"
```

---

## Self-Review

Coverage check against the spec (`docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md`):

- **§4 layout** — covered (all foundational files exist). Plan 2/3 fills in model files.
- **§6 API** — headers fully declared. Model-dependent functions return `RFDETR_ERR_NOT_IMPLEMENTED` (out of CLI scope for this plan); image, render, postprocess, logging are real.
- **§7 CLI** — all four subcommands recognized by the parser. `detect` works end-to-end without a model; the other three are deliberately stubbed (Plan 2/3).
- **§8 tests** — unit tests for image I/O, postprocess, visualize; integration test for CLI. Model-dependent tests live in Plan 2/3.
- **§10 build** — backend flags wired (`RFDETR_GGML_*`); ggml linked but not yet exercised. Shared/static toggle in place.

Not covered (and not expected to be in Plan 1):
- GGUF loading, variant detection (Plan 2)
- Forward pass (Plan 3)
- Parity baseline (Plan 3)
- Quantization (Plan 4)
- Label rendering with stb_truetype + embedded font (Plan 2; the font header is not yet generated)
- Flat C ABI impl (Plan 2)

Type consistency: every function declared in a header is implemented or explicitly stubbed; the `rfdetr_image` struct has the same fields in `image_io.hpp` as used by `visualize.cpp` and `image_io.cpp`. Names match between tests and impl (`rfdetr_bbox_cxcywh_to_xyxy`, `rfdetr_select_detections`, etc.).

Placeholder scan: searched for "TODO", "TBD", "implement later" — none in the executable code. Stubbed subcommands are explicitly documented as deferred to Plans 2/3.

---

## Next plans

After this plan is green:

- **Plan 2** — `scripts/convert_rfdetr_to_gguf.py` for the `base` variant, `model_loader.{cpp,hpp}`, variant config parsing, font-embedding header generator and label rendering in `visualize.cpp`, flat C ABI implementation against an in-memory dummy model.
- **Plan 3** — DINOv2 backbone, projector, encoder, decoder, heads, `rfdetr_model.cpp` graph build, trace callback, `scripts/run_rfdetr_baseline.py`, `test_parity` against the `base` variant.
- **Plan 4** — `rfdetr-quantize` binary, `small`/`nano`/`medium`/`large` bring-up, per-variant parity bundles.
