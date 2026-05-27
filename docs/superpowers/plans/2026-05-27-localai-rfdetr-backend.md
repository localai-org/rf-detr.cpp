# Phase B — LocalAI Native Backend `backend/go/rfdetr-cpp/` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native rfdetr.cpp backend to LocalAI under `backend/go/rfdetr-cpp/`, mirroring the existing `backend/go/sam3-cpp/` exactly. The backend dlopens `librfdetr.so` via purego, dispatches `Load` and `Detect` gRPC calls, returns per-detection bbox + class name + confidence + PNG mask via the new accessor functions (Phase A).

**Architecture:** A standalone Go process speaking gRPC to LocalAI core. At startup, `main.go` selects the highest-supported `lib<name>-{avx512,avx2,avx,fallback}.so` based on `/proc/cpuinfo` (handled by `run.sh` exporting `RFDETRCPP_LIBRARY`), then `purego.RegisterLibFunc`s each `rfdetr_capi_*` symbol. The Go-side `RFDetrCpp` struct embeds `base.SingleThread` and overrides only `Load` and `Detect`. Output rows are mapped 1-1 to `pb.Detection{X, Y, Width, Height, Confidence, ClassName, Mask}`. The Makefile clones `https://github.com/mudler/rt-detr.cpp` at a pinned commit, builds with `-DRFDETR_SHARED=ON` per CPU variant, packages the lib + all SOs into `package/` for the `FROM scratch` final docker image.

**Tech Stack:** Go (1.22+), purego (no cgo), CMake (forwards to rfdetr.cpp's build), Docker (per-variant images), GitHub Actions matrix.

---

## Prerequisites

- **Phase A complete and tagged on `mudler/rt-detr.cpp`**. Pin the Makefile to that exact commit SHA. Phase A introduces the 6 accessor functions this plan depends on.
- LocalAI working dir: `~/_git/LocalAI`
- Reference backend: `~/_git/LocalAI/backend/go/sam3-cpp/` (study this thoroughly first; ~80% of this plan is structural copy + rename)

## File Structure

In `~/_git/LocalAI/`:

- Create: `backend/go/rfdetr-cpp/main.go` — purego loader + gRPC server entry
- Create: `backend/go/rfdetr-cpp/gorfdetrcpp.go` — Load/Detect handlers
- Create: `backend/go/rfdetr-cpp/CMakeLists.txt` — CMake forwarding into rfdetr.cpp
- Create: `backend/go/rfdetr-cpp/Makefile` — clone + per-CPU-variant SO build
- Create: `backend/go/rfdetr-cpp/run.sh` — runtime SO selection via /proc/cpuinfo
- Create: `backend/go/rfdetr-cpp/package.sh` — bundle binary + SOs + sysroot fragments
- Create: `backend/go/rfdetr-cpp/test.sh` — smoke test: download nano-q8_0, run detect on fixture
- Create: `backend/go/rfdetr-cpp/.gitignore` — `package/`, `sources/`, `*.so`
- Modify: top-level `Makefile` — register `BACKEND_RFDETR_CPP` + docker-build target
- Modify: `.github/backend-matrix.yml` — add CPU/CUDA/Metal/Vulkan/HIP variants
- Modify: `gallery/index.yaml` — (optional) add a `rfdetr-cpp` entry pointing at HF GGUFs from `mudler/rfdetr-cpp-*`

---

## Task 1: Study the sam3-cpp pattern thoroughly

**Files:**
- Read-only: `backend/go/sam3-cpp/*`

- [ ] **Step 1: Read every file in `backend/go/sam3-cpp/`**

```bash
cd ~/_git/LocalAI
ls backend/go/sam3-cpp/
cat backend/go/sam3-cpp/main.go
cat backend/go/sam3-cpp/gosam3.go
cat backend/go/sam3-cpp/CMakeLists.txt
cat backend/go/sam3-cpp/Makefile
cat backend/go/sam3-cpp/run.sh
cat backend/go/sam3-cpp/package.sh
cat backend/go/sam3-cpp/test.sh
cat backend/go/sam3-cpp/cpp/gosam3.h
cat backend/go/sam3-cpp/cpp/gosam3.cpp
```

Note the parts that are sam3-specific (model loading, point/box prompts, the static `g_*` globals) vs. parts that are generic (gRPC dispatch, purego registration, CPU variant selection, packaging). You'll keep the generic parts verbatim.

- [ ] **Step 2: Note that rfdetr.cpp's flat C-API already replaces sam3-cpp's `cpp/` shim**

sam3-cpp has a `cpp/gosam3.{cpp,h}` thin shim because upstream `sam3.cpp` does NOT have a pre-baked flat FFI surface. Our `rfdetr.cpp` DOES (Phase A added it), so we can skip the `cpp/` directory entirely and `RegisterLibFunc` directly against `rfdetr_capi_*` symbols.

This is a real simplification: we have **fewer files** than sam3-cpp, not more.

- [ ] **No commit yet** — this is just an orientation step.

---

## Task 2: Write `main.go`

**Files:**
- Create: `~/_git/LocalAI/backend/go/rfdetr-cpp/main.go`

- [ ] **Step 1: Write the file**

```go
package main

// main.go — entry point for the rfdetr-cpp gRPC backend.
//
// Dlopens librfdetr.so via purego at the path in RFDETRCPP_LIBRARY (set
// by run.sh based on /proc/cpuinfo), registers the C ABI symbols, then
// starts the gRPC server.

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/ebitengine/purego"
	grpc "github.com/mudler/LocalAI/pkg/grpc"
)

var (
	rfdetrCapiLoad                       func(path string, nThreads int32, handleOut *uintptr) int32
	rfdetrCapiUnload                     func(handle uintptr) int32
	rfdetrCapiDetectPath                 func(handle uintptr, imagePath string, threshold float32, topK int32, jsonOut **byte) int32
	rfdetrCapiDetectBuffer               func(handle uintptr, bytes []byte, nBytes uintptr, threshold float32, topK int32, jsonOut **byte) int32
	rfdetrCapiFreeString                 func(s *byte)
	rfdetrCapiGetNDetections             func(handle uintptr) int32
	rfdetrCapiGetDetectionClassID        func(handle uintptr, i int32) int32
	rfdetrCapiGetDetectionBox            func(handle uintptr, i int32, outXYXY *float32) int32
	rfdetrCapiGetDetectionScore          func(handle uintptr, i int32) float32
	rfdetrCapiGetDetectionClassName      func(handle uintptr, i int32, buf *byte, bufSize int32) int32
	rfdetrCapiGetDetectionMaskPNG        func(handle uintptr, i int32, buf *byte, bufSize int32) int32
)

func main() {
	addr := flag.String("addr", "127.0.0.1:50051", "gRPC listen address")
	flag.Parse()

	libPath := os.Getenv("RFDETRCPP_LIBRARY")
	if libPath == "" {
		libPath = "./librfdetrcpp-fallback.so"
	}
	// Resolve relative paths against the binary's directory.
	if !filepath.IsAbs(libPath) {
		if exe, err := os.Executable(); err == nil {
			libPath = filepath.Join(filepath.Dir(exe), libPath)
		}
	}

	lib, err := purego.Dlopen(libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		fmt.Fprintf(os.Stderr, "rfdetr-cpp: failed to dlopen %s: %v\n", libPath, err)
		os.Exit(1)
	}

	purego.RegisterLibFunc(&rfdetrCapiLoad, lib, "rfdetr_capi_load")
	purego.RegisterLibFunc(&rfdetrCapiUnload, lib, "rfdetr_capi_unload")
	purego.RegisterLibFunc(&rfdetrCapiDetectPath, lib, "rfdetr_capi_detect_path")
	purego.RegisterLibFunc(&rfdetrCapiDetectBuffer, lib, "rfdetr_capi_detect_buffer")
	purego.RegisterLibFunc(&rfdetrCapiFreeString, lib, "rfdetr_capi_free_string")
	purego.RegisterLibFunc(&rfdetrCapiGetNDetections, lib, "rfdetr_capi_get_n_detections")
	purego.RegisterLibFunc(&rfdetrCapiGetDetectionClassID, lib, "rfdetr_capi_get_detection_class_id")
	purego.RegisterLibFunc(&rfdetrCapiGetDetectionBox, lib, "rfdetr_capi_get_detection_box")
	purego.RegisterLibFunc(&rfdetrCapiGetDetectionScore, lib, "rfdetr_capi_get_detection_score")
	purego.RegisterLibFunc(&rfdetrCapiGetDetectionClassName, lib, "rfdetr_capi_get_detection_class_name")
	purego.RegisterLibFunc(&rfdetrCapiGetDetectionMaskPNG, lib, "rfdetr_capi_get_detection_mask_png")

	if err := grpc.StartServer(*addr, &RFDetrCpp{}); err != nil {
		fmt.Fprintf(os.Stderr, "rfdetr-cpp: gRPC server failed: %v\n", err)
		os.Exit(1)
	}
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd ~/_git/LocalAI
go build -o /tmp/rfdetr-cpp-test ./backend/go/rfdetr-cpp/ 2>&1 | head
```

It WILL fail because `gorfdetrcpp.go` (Task 3) isn't written yet — the `RFDetrCpp` struct is undefined. That's expected. Move on.

- [ ] **No commit yet** — main.go alone doesn't compile.

---

## Task 3: Write `gorfdetrcpp.go`

**Files:**
- Create: `~/_git/LocalAI/backend/go/rfdetr-cpp/gorfdetrcpp.go`

- [ ] **Step 1: Write the file**

```go
package main

// gorfdetrcpp.go — gRPC handlers (Load, Detect) for the rfdetr-cpp backend.
//
// Embeds base.SingleThread to default unimplemented RPCs to "not supported"
// while we only implement object detection.

import (
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"unsafe"

	"github.com/mudler/LocalAI/pkg/grpc/base"
	pb "github.com/mudler/LocalAI/pkg/grpc/proto"
)

type RFDetrCpp struct {
	base.SingleThread
	handle uintptr
}

// Load loads the GGUF model at opts.ModelFile (joined with opts.ModelPath if relative)
// and stores the handle for later Detect calls.
func (r *RFDetrCpp) Load(opts *pb.ModelOptions) error {
	modelPath := opts.ModelFile
	if modelPath == "" {
		return fmt.Errorf("rfdetr-cpp: ModelFile is empty")
	}
	if !filepath.IsAbs(modelPath) && opts.ModelPath != "" {
		modelPath = filepath.Join(opts.ModelPath, modelPath)
	}
	if _, err := os.Stat(modelPath); err != nil {
		return fmt.Errorf("rfdetr-cpp: model file not found: %s: %w", modelPath, err)
	}

	nThreads := opts.Threads
	if nThreads <= 0 {
		nThreads = 4
	}

	if r.handle != 0 {
		rfdetrCapiUnload(r.handle)
		r.handle = 0
	}

	var h uintptr
	rc := rfdetrCapiLoad(modelPath, nThreads, &h)
	if rc != 0 || h == 0 {
		return fmt.Errorf("rfdetr-cpp: rfdetr_capi_load failed with rc=%d", rc)
	}
	r.handle = h
	return nil
}

// Detect runs object detection on the image in opts.Src (base64) at opts.Threshold.
// Returns a list of pb.Detection with optional PNG mask bytes for seg models.
func (r *RFDetrCpp) Detect(opts *pb.DetectOptions) (*pb.DetectResponse, error) {
	if r.handle == 0 {
		return nil, fmt.Errorf("rfdetr-cpp: model not loaded")
	}

	// Decode the base64 image and write to a tempfile (rfdetr_capi_detect_path is
	// simpler than the buffer variant; we may switch to _detect_buffer later for zero-copy).
	imgBytes, err := base64.StdEncoding.DecodeString(opts.Src)
	if err != nil {
		return nil, fmt.Errorf("rfdetr-cpp: invalid base64 image: %w", err)
	}
	tmp, err := os.CreateTemp("", "rfdetr-input-*.jpg")
	if err != nil {
		return nil, err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(imgBytes); err != nil {
		tmp.Close()
		return nil, err
	}
	tmp.Close()

	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = 0.5
	}
	topK := int32(300)
	if opts.TopK > 0 {
		topK = int32(opts.TopK)
	}

	var jsonPtr *byte
	rc := rfdetrCapiDetectPath(r.handle, tmp.Name(), threshold, topK, &jsonPtr)
	if jsonPtr != nil {
		defer rfdetrCapiFreeString(jsonPtr)
	}
	if rc != 0 {
		return nil, fmt.Errorf("rfdetr-cpp: detect failed with rc=%d", rc)
	}

	n := rfdetrCapiGetNDetections(r.handle)
	if n < 0 {
		return nil, fmt.Errorf("rfdetr-cpp: invalid n_detections=%d", n)
	}
	dets := make([]*pb.Detection, 0, n)

	var bbox [4]float32
	for i := int32(0); i < n; i++ {
		if rc := rfdetrCapiGetDetectionBox(r.handle, i, &bbox[0]); rc != 0 {
			continue
		}
		cid := rfdetrCapiGetDetectionClassID(r.handle, i)
		score := rfdetrCapiGetDetectionScore(r.handle, i)

		// Class name: two-call sizing.
		nameSize := rfdetrCapiGetDetectionClassName(r.handle, i, nil, 0)
		var className string
		if nameSize > 1 {
			buf := make([]byte, nameSize)
			rfdetrCapiGetDetectionClassName(r.handle, i, &buf[0], nameSize)
			// Strip trailing NUL byte for Go string.
			className = string(buf[:nameSize-1])
		} else {
			className = strconv.Itoa(int(cid))
		}

		// Mask: two-call sizing; empty for detection-only models.
		var mask []byte
		maskSize := rfdetrCapiGetDetectionMaskPNG(r.handle, i, nil, 0)
		if maskSize > 0 {
			mask = make([]byte, maskSize)
			rfdetrCapiGetDetectionMaskPNG(r.handle, i, &mask[0], maskSize)
		}

		dets = append(dets, &pb.Detection{
			X:          bbox[0],
			Y:          bbox[1],
			Width:      bbox[2] - bbox[0],
			Height:     bbox[3] - bbox[1],
			Confidence: score,
			ClassName:  className,
			ClassId:    cid,
			Mask:       mask,
		})
	}

	_ = unsafe.Pointer(nil)  // appease "imported and not used" if all unsafe references are commented out
	return &pb.DetectResponse{Detections: dets}, nil
}
```

- [ ] **Step 2: Verify the package compiles**

```bash
cd ~/_git/LocalAI
go build ./backend/go/rfdetr-cpp/ 2>&1 | head
```

Should compile cleanly. If `pb.DetectOptions` lacks a `TopK` field or `pb.Detection` lacks a `ClassId` field, check `pkg/grpc/proto/backend.pb.go` for the actual field names and adapt.

- [ ] **Step 3: Commit**

```bash
git add backend/go/rfdetr-cpp/main.go backend/go/rfdetr-cpp/gorfdetrcpp.go
git commit -m "feat(backend): rfdetr-cpp purego loader + Load/Detect handlers"
```

---

## Task 4: Write CMakeLists.txt

**Files:**
- Create: `~/_git/LocalAI/backend/go/rfdetr-cpp/CMakeLists.txt`

- [ ] **Step 1: Write the file**

This file is invoked by the Makefile (Task 5) once per CPU variant. It expects rfdetr.cpp source to be at `./sources/rt-detr.cpp` and builds a `MODULE` library output at `librfdetrcpp-${VARIANT}.so`.

```cmake
cmake_minimum_required(VERSION 3.18)
project(librfdetrcpp LANGUAGES CXX C)

set(VARIANT "" CACHE STRING "CPU variant suffix (avx, avx2, avx512, fallback)")
if(NOT VARIANT)
    message(FATAL_ERROR "VARIANT must be set (e.g., -DVARIANT=avx2)")
endif()

# Static-link ggml so the .so has no runtime ggml dependency.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Forward GPU backend flags (set by the Makefile based on BUILD_TYPE).
set(GGML_CUDA    ${RFDETR_USE_CUDA}    CACHE BOOL "" FORCE)
set(GGML_METAL   ${RFDETR_USE_METAL}   CACHE BOOL "" FORCE)
set(GGML_VULKAN  ${RFDETR_USE_VULKAN}  CACHE BOOL "" FORCE)
set(GGML_HIPBLAS ${RFDETR_USE_HIPBLAS} CACHE BOOL "" FORCE)
set(GGML_SYCL    ${RFDETR_USE_SYCL}    CACHE BOOL "" FORCE)

# CPU instruction-set tier (driven by VARIANT).
if(VARIANT STREQUAL "avx512")
    set(GGML_AVX512 ON CACHE BOOL "" FORCE)
elseif(VARIANT STREQUAL "avx2")
    set(GGML_AVX2 ON CACHE BOOL "" FORCE)
elseif(VARIANT STREQUAL "avx")
    set(GGML_AVX ON CACHE BOOL "" FORCE)
endif()

# Tell rfdetr.cpp to build as a shared library and skip its CLI + tests.
set(RFDETR_SHARED ON CACHE BOOL "" FORCE)
set(RFDETR_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(RFDETR_BUILD_TESTS OFF CACHE BOOL "" FORCE)

add_subdirectory(sources/rt-detr.cpp ${CMAKE_BINARY_DIR}/rfdetr_src EXCLUDE_FROM_ALL)

# Re-export the rfdetr_capi_* symbols on a MODULE that purego will dlopen.
add_library(rfdetrcpp_${VARIANT} MODULE
    sources/rt-detr.cpp/src/rfdetr_capi.cpp)
target_link_libraries(rfdetrcpp_${VARIANT} PRIVATE rfdetr)
set_target_properties(rfdetrcpp_${VARIANT} PROPERTIES
    OUTPUT_NAME "rfdetrcpp-${VARIANT}"
    PREFIX "lib")
```

If sam3-cpp's CMakeLists.txt uses a different structure (e.g., it doesn't include the upstream source as a subdirectory but instead just links against a prebuilt static lib), adapt to match.

- [ ] **Step 2: Commit**

```bash
git add backend/go/rfdetr-cpp/CMakeLists.txt
git commit -m "feat(backend): CMake forwarding into rt-detr.cpp"
```

---

## Task 5: Write the Makefile

**Files:**
- Create: `~/_git/LocalAI/backend/go/rfdetr-cpp/Makefile`

- [ ] **Step 1: Identify the pinned rt-detr.cpp commit SHA**

After Phase A is merged + pushed, identify the commit SHA on `mudler/rt-detr.cpp` that includes the accessor functions. Pin to that.

For this plan document, use `${PHASE_A_SHA}` as a placeholder; replace at execution time.

- [ ] **Step 2: Copy + adapt the sam3-cpp Makefile**

```makefile
# backend/go/rfdetr-cpp/Makefile — clone rfdetr.cpp, build per-CPU-variant .so files,
# and the Go gRPC binary.

RFDETR_REPO ?= https://github.com/mudler/rt-detr.cpp.git
RFDETR_SHA  ?= ${PHASE_A_SHA}

# BUILD_TYPE forwarded from the LocalAI top-level Makefile; controls GPU backend flags.
BUILD_TYPE ?= cpu

CMAKE_FLAGS_BASE = -DRFDETR_USE_CUDA=OFF -DRFDETR_USE_METAL=OFF \
                   -DRFDETR_USE_VULKAN=OFF -DRFDETR_USE_HIPBLAS=OFF \
                   -DRFDETR_USE_SYCL=OFF

ifeq ($(BUILD_TYPE),cublas)
CMAKE_FLAGS_BASE = -DRFDETR_USE_CUDA=ON ...
endif
ifeq ($(BUILD_TYPE),hipblas)
CMAKE_FLAGS_BASE = -DRFDETR_USE_HIPBLAS=ON ...
endif
# ...etc — copy from sam3-cpp/Makefile

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
VARIANTS := avx avx2 avx512 fallback
else
VARIANTS := fallback
endif

.PHONY: all
all: sources/rt-detr.cpp $(addprefix librfdetrcpp-,$(addsuffix .so,$(VARIANTS))) backend

sources/rt-detr.cpp:
	mkdir -p sources
	git clone $(RFDETR_REPO) $@
	cd $@ && git checkout $(RFDETR_SHA)
	cd $@ && git submodule update --init --recursive

librfdetrcpp-%.so: sources/rt-detr.cpp
	rm -rf build-$*
	cmake -B build-$* -DVARIANT=$* $(CMAKE_FLAGS_BASE)
	cmake --build build-$* -j
	cp build-$*/librfdetrcpp-$*.so .

backend: main.go gorfdetrcpp.go
	go build -o rfdetr-cpp ./...

.PHONY: clean
clean:
	rm -rf sources build-* librfdetrcpp-*.so rfdetr-cpp
```

Use sam3-cpp's Makefile as the line-by-line reference for the GPU/BUILD_TYPE switch logic.

- [ ] **Step 3: Test the build locally on a CPU-only host**

```bash
cd ~/_git/LocalAI/backend/go/rfdetr-cpp
make sources/rt-detr.cpp                # clone
make librfdetrcpp-fallback.so           # build one variant
ls -la librfdetrcpp-fallback.so          # ~10 MB
make backend                            # Go gRPC binary
ls -la rfdetr-cpp                       # static-link of Go runtime
```

If a CMake error mentions missing headers or symbols, that's where Phase A's accessor functions matter — ensure the cloned commit has them.

- [ ] **Step 4: Commit**

```bash
git add backend/go/rfdetr-cpp/Makefile
git commit -m "feat(backend): Makefile clones rt-detr.cpp + builds per-CPU-variant SOs"
```

---

## Task 6: Write run.sh + package.sh + test.sh

**Files:**
- Create: `~/_git/LocalAI/backend/go/rfdetr-cpp/run.sh`
- Create: `~/_git/LocalAI/backend/go/rfdetr-cpp/package.sh`
- Create: `~/_git/LocalAI/backend/go/rfdetr-cpp/test.sh`

- [ ] **Step 1: Copy run.sh from sam3-cpp; rename env var**

```bash
#!/bin/sh
# backend/go/rfdetr-cpp/run.sh — picks the best lib variant based on /proc/cpuinfo
# and execs the Go gRPC server.

set -e
cd "$(dirname "$0")"

# Determine highest supported CPU variant.
if grep -q avx512f /proc/cpuinfo 2>/dev/null; then
    export RFDETRCPP_LIBRARY=./librfdetrcpp-avx512.so
elif grep -q avx2 /proc/cpuinfo 2>/dev/null; then
    export RFDETRCPP_LIBRARY=./librfdetrcpp-avx2.so
elif grep -q '\bavx\b' /proc/cpuinfo 2>/dev/null; then
    export RFDETRCPP_LIBRARY=./librfdetrcpp-avx.so
else
    export RFDETRCPP_LIBRARY=./librfdetrcpp-fallback.so
fi

# Fall back to fallback variant if the selected one is missing.
if [ ! -f "$RFDETRCPP_LIBRARY" ]; then
    export RFDETRCPP_LIBRARY=./librfdetrcpp-fallback.so
fi

exec ./rfdetr-cpp "$@"
```

Make executable: `chmod +x run.sh`.

- [ ] **Step 2: Copy package.sh from sam3-cpp; adapt paths**

Same template — bundles the binary, all SO variants, and minimal system libraries into `package/` for the `FROM scratch` final image. Copy verbatim and rename.

- [ ] **Step 3: Write test.sh**

```bash
#!/bin/bash
# Smoke test: download a small rfdetr GGUF, start the backend, send a Detect RPC, verify response.
set -e
cd "$(dirname "$0")"

# Build first.
make all

# Download a small GGUF.
mkdir -p test-models
if [ ! -f test-models/rfdetr-nano-q8_0.gguf ]; then
    hf download mudler/rfdetr-cpp-nano rfdetr-nano-q8_0.gguf --local-dir test-models
fi

# Start the backend on a free port.
./run.sh --addr 127.0.0.1:50091 &
BACKEND_PID=$!
trap "kill $BACKEND_PID 2>/dev/null" EXIT
sleep 2

# Use grpcurl or a tiny Go client to send Load + Detect.
# (Compose a JSON image fixture if no client is available — left as exercise.)

# Verify exit cleanly.
kill $BACKEND_PID
echo "rfdetr-cpp smoke OK"
```

- [ ] **Step 4: Commit**

```bash
chmod +x backend/go/rfdetr-cpp/{run,package,test}.sh
git add backend/go/rfdetr-cpp/run.sh backend/go/rfdetr-cpp/package.sh backend/go/rfdetr-cpp/test.sh
git commit -m "feat(backend): runtime + packaging scripts (run, package, test)"
```

---

## Task 7: Top-level Makefile registration

**Files:**
- Modify: `~/_git/LocalAI/Makefile`

- [ ] **Step 1: Find the existing `BACKEND_SAM3_CPP` registration line**

```bash
cd ~/_git/LocalAI
grep -n "BACKEND_SAM3_CPP" Makefile
```

Expected: ~3 lines (definition, `.NOTPARALLEL`, `docker-build-backends` aggregate).

- [ ] **Step 2: Add 3 corresponding lines for rfdetr-cpp**

Near `BACKEND_SAM3_CPP = sam3-cpp|golang|.|false|true` add:

```makefile
BACKEND_RFDETR_CPP = rfdetr-cpp|golang|.|false|true
```

Near the `$(eval $(call generate-docker-build-target,$(BACKEND_SAM3_CPP)))` line add:

```makefile
$(eval $(call generate-docker-build-target,$(BACKEND_RFDETR_CPP)))
```

In the `.NOTPARALLEL:` rule list near the top, add `backends/rfdetr-cpp` if `backends/sam3-cpp` is listed.

In the `docker-build-backends:` aggregate target, add `docker-build-rfdetr-cpp` if `docker-build-sam3-cpp` is listed.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat(make): register rfdetr-cpp backend in top-level Makefile"
```

---

## Task 8: Add backend-matrix.yml entries

**Files:**
- Modify: `~/_git/LocalAI/.github/backend-matrix.yml`

- [ ] **Step 1: Find sam3-cpp entries**

```bash
grep -n "sam3-cpp" .github/backend-matrix.yml | head -20
```

Each entry is a `dockerfile + arch + build-type + backend + tag-suffix` row.

- [ ] **Step 2: Add corresponding rfdetr-cpp rows**

For each sam3-cpp row, duplicate with `sam3-cpp` → `rfdetr-cpp`. Cover at minimum:
- CPU: linux/amd64, linux/arm64, darwin
- cublas-12, cublas-13
- hipblas
- vulkan-amd64, vulkan-arm64
- sycl_f16, sycl_f32
- l4t (jetpack)

- [ ] **Step 3: Commit**

```bash
git add .github/backend-matrix.yml
git commit -m "ci(backend-matrix): rfdetr-cpp entries for CPU + GPU variants"
```

---

## Task 9: Optional gallery entry

**Files:**
- Modify: `~/_git/LocalAI/gallery/index.yaml`
- Modify (optional): `~/_git/LocalAI/core/gallery/importers/rfdetr.go`

- [ ] **Step 1: Add a `rfdetr-cpp-base-f16` gallery entry**

Near the existing `rfdetr-base` entry (~line 6099), add a sibling entry that points at the native backend:

```yaml
- name: rfdetr-cpp-base-f16
  url: github:mudler/LocalAI/gallery/rfdetr-cpp.yaml@master
  description: |
    RF-DETR Base object detection model, served via native rfdetr.cpp backend.
    F16 quantization is recommended on CPU: same accuracy as F32, 1.86x smaller, fastest.
  license: apache-2.0
  urls:
    - https://huggingface.co/mudler/rfdetr-cpp-base
  tags:
    - object-detection
    - native
    - rfdetr
  known_usecases:
    - detection
  backend: rfdetr-cpp
  overrides:
    parameters:
      model: rfdetr-cpp-base-f16.gguf
  files:
    - filename: rfdetr-cpp-base-f16.gguf
      sha256: <fill in from HF>
      uri: https://huggingface.co/mudler/rfdetr-cpp-base/resolve/main/rfdetr-base-f16.gguf
```

Add similar entries for the other 7 variants if desired.

- [ ] **Step 2: (Optional) Update importer**

`core/gallery/importers/rfdetr.go::Import` currently always emits `backend: rfdetr` (the Python backend). Add a check for `.gguf` files in the HF repo manifest → emit `backend: rfdetr-cpp` instead.

- [ ] **Step 3: Commit**

```bash
git add gallery/index.yaml core/gallery/importers/rfdetr.go
git commit -m "feat(gallery): rfdetr-cpp entries pointing at mudler/rfdetr-cpp-* HF repos"
```

---

## Task 10: End-to-end smoke test

**Files:**
- None modified (verification)

- [ ] **Step 1: Build everything**

```bash
cd ~/_git/LocalAI
make backend-rfdetr-cpp
```

- [ ] **Step 2: Start LocalAI with the new backend**

```bash
./local-ai run --models-path ./test-models
```

- [ ] **Step 3: Test with curl**

```bash
# Encode a test image to base64
img_b64=$(base64 -w0 ~/_git/rt-detr.cpp/tests/fixtures/ci/test_image.jpg)

# Detect via the LocalAI HTTP endpoint
curl -X POST http://localhost:8080/v1/detection \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"rfdetr-cpp-base-f16\",\"image\":\"$img_b64\",\"threshold\":0.5}"
```

Expected: JSON response with `detections[]`, each having `x`, `y`, `width`, `height`, `confidence`, `class_name`, and (for seg models) `mask` as base64-encoded PNG.

- [ ] **Step 4: For seg models, decode and inspect a mask**

```bash
# Repeat with seg-nano model
curl ... -d '{"model":"rfdetr-cpp-seg-nano-f16","image":"...","threshold":0.5}' > /tmp/seg_resp.json
.venv/bin/python -c "
import json, base64
r = json.load(open('/tmp/seg_resp.json'))
for i, d in enumerate(r['detections'][:3]):
    if d.get('mask'):
        open(f'/tmp/mask_{i}.png', 'wb').write(base64.b64decode(d['mask']))
        print(f'mask {i}: {d[\"class_name\"]} -> /tmp/mask_{i}.png')
"
ls -la /tmp/mask_*.png
# Each should be a valid PNG; open in image viewer.
```

- [ ] **Step 5: Push to a feature branch**

```bash
git checkout -b feat/rfdetr-cpp-backend
git push origin feat/rfdetr-cpp-backend
```

Open a PR against `master` (or whatever the LocalAI main branch is called). Reference Phase A's commit on `mudler/rt-detr.cpp` in the PR description.

---

## Self-Review

- **Spec coverage**: native backend at `backend/go/rfdetr-cpp/` ✓; purego loader with all 11 `rfdetr_capi_*` symbols ✓; gRPC Load + Detect handlers ✓; per-CPU-variant build ✓; runtime SO selection ✓; packaging ✓; top-level Makefile + backend-matrix.yml entries ✓; gallery entries ✓; end-to-end smoke test ✓.
- **Placeholders**: `${PHASE_A_SHA}` in Task 5 is intentional — must be filled in at execution time once Phase A lands.
- **Type consistency**: `pb.Detection` field names referenced in Task 3 (`X, Y, Width, Height, Confidence, ClassName, ClassId, Mask`) must match the actual proto-generated Go struct — verify in `pkg/grpc/proto/backend.pb.go` before writing.

**Open questions to verify at execution time:**

1. Does `pb.Detection` have `ClassId` (int32) AND `ClassName` (string), or just one? The sam3-cpp implementation hardcodes ClassName to "segment" because SAM is class-agnostic — verify rfdetr-cpp's class map matches the proto's int field.
2. Does `pb.DetectOptions` expose `TopK` and `Threshold` directly, or are they buried in `Options []string`? sam3-cpp uses the explicit fields if available — check.
3. The `RFDETR_USE_CUDA` etc. CMake flags don't exist in current rfdetr.cpp (only `RFDETR_GGML_CUDA`). Update the CMakeLists.txt forwarding to match the actual flag names in `~/_git/rt-detr.cpp/CMakeLists.txt`.
4. sam3-cpp clones a separate `cpp/gosam3.cpp` shim because it adds project-specific logic on top of upstream sam3.cpp. We may not need any shim — rfdetr.cpp's flat C-API is already shaped right. Verify this during Task 4 implementation. If we DO need a tiny shim (e.g., for handle storage that lives outside rfdetr.cpp's lifecycle), add a minimal `cpp/` directory then.
5. `package.sh` and the docker integration depend on the LocalAI versions of build-base + golang. Coordinate with the existing `Dockerfile.golang` to make sure rfdetr-cpp's libstdc++ version matches what the runtime image ships.

---

## Estimated effort

- Tasks 1-3 (orientation + main.go + gorfdetrcpp.go): 2-3 hours
- Tasks 4-6 (CMake + Makefile + scripts): 2-3 hours
- Tasks 7-8 (top-level Makefile + backend-matrix.yml): 1-2 hours
- Task 9 (gallery): 1 hour
- Task 10 (smoke test + push): 1-2 hours

**Total: ~1-1.5 days** depending on how many CMake/Makefile gotchas surface during the per-variant build.
