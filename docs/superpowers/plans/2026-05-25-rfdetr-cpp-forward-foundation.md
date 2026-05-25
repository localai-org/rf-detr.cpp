# rt-detr.cpp Forward-Pass Foundation Plan (Plan 3 of 8)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring up the foundation that the rest of the forward pass will stand on: weight realization into a real ggml backend buffer, a `trace_callback` mechanism that publishes named intermediate tensors, the DINOv2 `patch_embed` operator, and a single DINOv2 transformer block (norm → multi-head self-attention → norm → MLP). Build a numerical-parity test harness against a deterministic numpy reference — no `rfdetr` PyPI install required for CI. End state: clean rebuild produces a C++ forward pass through backbone block 0 that matches a numpy reference within 1e-3 absolute tolerance, with the parity harness ready to grow as Plans 4–6 add more layers.

**Architecture:** Three new C++ TUs (`backend.{cpp,hpp}`, `trace.{cpp,hpp}`, `dinov2.{cpp,hpp}`) plus a `model_realize_weights` extension to `model_loader.cpp` that takes the metadata-only `Model` and loads tensor data into a CPU backend buffer (other backends are wired but only CPU is exercised here). `rfdetr_init` grows a backend-selection step and triggers weight realization. The parity harness has three pieces: a deterministic synthetic-weight GGUF generator (extends Plan 2's `gen_model_gguf` with `--seed`), a Python numpy reference (no torch/rfdetr — pure numpy) that consumes the same GGUF and dumps expected intermediate tensors to a *baseline bundle GGUF*, and a `test_parity_block0` C++ test that loads both, runs the C++ forward up through backbone block 0 with the trace callback active, and `allclose`s each published checkpoint.

**Tech Stack:** C++17, ggml (already linked from Plan 1), Python 3.9+ with numpy 1.26+ (no torch dependency in this plan), CMake ≥ 3.14, ctest. The Plan 2 conversion script and Plan 6 will eventually swap the numpy reference for a torch+rfdetr reference using the same baseline-bundle format.

---

## Scope decisions

- **One transformer block only.** Bringing up the second block is mechanically identical (same parametrized graph code, different config-driven loop bound) — Plan 4 turns the loop on.
- **CPU backend only.** ggml's backend abstraction is wired (so `model_realize_weights` accepts a `ggml_backend_t` parameter), but only the CPU backend is exercised. CUDA/Metal/Vulkan land later (their build flags already work from Plan 1).
- **No window attention yet.** Plan 4 adds it. Plan 3's transformer block is "global" (full self-attention over all tokens). The test only exercises `patch_embed` + block 0 in global mode, and the config option `backbone.window_size` is read but unused in this plan.
- **No CLS token handling, no positional embeddings.** Patch embed produces a `(B, N_patches, dim)` token tensor; we add the positional embedding tensor and skip CLS token concatenation/inclusion (DINOv2 includes it but block 0 produces a sensible per-token output either way for parity testing). Plan 4 introduces CLS + finalizes the production-fidelity DINOv2 path.
- **Synthetic numpy reference, not PyTorch.** Plan 6 swaps in a torch baseline. The numpy reference is ~50 lines of pure-numpy ops (LayerNorm, MatMul, GELU, softmax) — verifiable by inspection.
- **Deterministic weights from a fixed seed.** Plan 2's fixture wrote zero-filled tensors; that makes the forward pass trivially zero. Plan 3 adds a `--seed` flag to the generator that fills each tensor with `rng.normal(0, 0.02)` style values — same convention PyTorch's default init uses.
- **`detect` still returns NOT_IMPLEMENTED** in this plan. The forward pass runs through block 0 only when the parity test calls it via a debug-only entry point (`rfdetr::forward_through_backbone_block(Model&, input, block_idx)`). Public `rfdetr_detect` stays stubbed until Plan 5.

---

## File map (created or modified in this plan)

```
rt-detr.cpp/
├── docs/
│   └── parity.md                       # NEW — parity workflow doc
├── scripts/
│   └── gen_numpy_baseline.py           # NEW — numpy reference + baseline bundle writer
├── src/
│   ├── backend.{cpp,hpp}               # NEW — ggml_backend selection (CPU only for now)
│   ├── trace.{cpp,hpp}                 # NEW — named-checkpoint callback plumbing
│   ├── dinov2.{cpp,hpp}                # NEW — patch_embed + single transformer block
│   ├── model_loader.{cpp,hpp}          # MODIFY — add model_realize_weights() phase
│   ├── rfdetr.cpp                      # MODIFY — wire backend select + weight realize into init
│   └── (other src files unchanged)
├── tests/
│   ├── CMakeLists.txt                  # MODIFY — register test_parity_block0
│   ├── fixtures/
│   │   ├── gen_model_gguf.cpp          # MODIFY — add --seed flag (deterministic random weights)
│   │   └── baseline_block0.gguf        # GENERATED at build via gen_numpy_baseline.py
│   ├── test_model_loader.cpp           # MODIFY — assert weight realize works
│   └── test_parity_block0.cpp          # NEW — C++ forward vs numpy reference parity
└── include/
    └── rfdetr.h                        # unchanged (debug entry point is C++-only)
```

---

### Task 1: docs/parity.md — document the parity workflow

Lock the contract before code. Both the C++ harness and the Python baseline script reference this doc.

**Files:**
- Create: `docs/parity.md`

- [ ] **Step 1: Write `docs/parity.md`**

```markdown
# rt-detr.cpp Parity Workflow

## Goal

Verify that the C++ forward pass produces the same intermediate tensors as a
reference implementation, layer by layer, with declared per-checkpoint
tolerances. Catch divergences at the layer where they first appear.

## Reference implementations

Plan 3 ships a **numpy reference** (`scripts/gen_numpy_baseline.py`). It uses
the same Plan 2 GGUF format and produces a baseline bundle GGUF containing
expected intermediate tensors at named checkpoints. No torch / no rfdetr; CI
runs it directly.

Plan 6 will add a **torch + rfdetr reference** that consumes the same input
and produces a baseline bundle in the same format. The C++ parity harness
is reference-agnostic — it only consumes baseline bundles.

## Baseline bundle format

A baseline bundle is a GGUF file with:

- All tensors named `parity.<checkpoint_name>` — e.g.
  `parity.preprocess.input`, `parity.backbone.patch_embed.output`,
  `parity.backbone.block.0.norm1.output`, `parity.backbone.block.0.output`.
- Metadata `parity.format.version = "1"`.
- Metadata `parity.reference = "numpy" | "torch"`.
- Metadata `parity.input_shape = uint32[4]` describing the input the
  reference consumed (NHWC: 1 × H × W × 3).

## Named checkpoints captured by Plan 3

- `preprocess.input` — `(1, H, W, 3)` float32, post normalization (mean/std)
- `backbone.patch_embed.output` — `(1, N_patches, dim)` float32
- `backbone.block.0.norm1.output` — `(1, N_patches, dim)` float32
- `backbone.block.0.attn.output` — `(1, N_patches, dim)` float32
- `backbone.block.0.mlp.output` — `(1, N_patches, dim)` float32
- `backbone.block.0.output` — `(1, N_patches, dim)` float32 (full block output)

Plans 4–6 add more checkpoints (block 1..11, projector levels, encoder/decoder
layers, heads).

## C++ trace callback

`src/trace.{cpp,hpp}` exposes:

```cpp
namespace rfdetr {
using trace_cb = std::function<void(const std::string& name, ggml_tensor* t)>;
void set_trace_callback(trace_cb cb);
void publish(const std::string& name, ggml_tensor* t);  // no-op if no cb
}
```

The forward-pass code calls `rfdetr::publish("backbone.patch_embed.output", t)`
at each defined checkpoint. Production inference doesn't register a callback;
the publish call is a hash-map lookup + early return.

## Per-checkpoint tolerances

Configured in `tests/test_parity_block0.cpp` via a small table. Defaults:

| Checkpoint                            | atol  | rtol  |
|---------------------------------------|-------|-------|
| `preprocess.input`                    | 1e-6  | 0     |
| `backbone.patch_embed.output`         | 1e-4  | 1e-3  |
| `backbone.block.0.norm1.output`       | 1e-4  | 1e-3  |
| `backbone.block.0.attn.output`        | 1e-3  | 1e-2  |
| `backbone.block.0.mlp.output`         | 1e-3  | 1e-2  |
| `backbone.block.0.output`             | 1e-3  | 1e-2  |

The attention checkpoint has looser tolerance because softmax-then-matmul
accumulates float-precision differences between numpy (float64 by default)
and ggml's F32. Tolerances will tighten as we accumulate confidence.

## Regeneration

```bash
python3 scripts/gen_numpy_baseline.py \
    --model tests/fixtures/model_base.gguf \
    --output tests/fixtures/baseline_block0.gguf
```

CMake runs this as a custom_command at build time (declared in
`tests/CMakeLists.txt`). Bundle is regenerated whenever the script changes
or the source GGUF fixture changes.

## Diagnosing a parity failure

`test_parity_block0` prints, for each failing checkpoint:

- Checkpoint name
- Tensor shape
- Max absolute error and its location (flat index)
- Mean absolute error
- Sample values: `cpp[i] = X, ref[i] = Y` at the worst location

A failing checkpoint earlier in the graph causes all later checkpoints to
fail. Always fix from the earliest divergence forward.
```

- [ ] **Step 2: Commit**

```bash
git add docs/parity.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
docs: document parity workflow (baseline bundle format, checkpoints, tolerances)

Locks the contract between the numpy reference (Plan 3), future torch
reference (Plan 6), and the C++ test harness. Both sides consume the same
Plan 2 GGUF format and produce baseline bundle GGUFs with parity.*
tensors at named checkpoints.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: trace callback infrastructure

**Files:**
- Create: `src/trace.hpp`
- Create: `src/trace.cpp`
- Modify: `CMakeLists.txt` (add `src/trace.cpp` to RFDETR_SOURCES)

- [ ] **Step 1: Add `src/trace.cpp` to `RFDETR_SOURCES`**

```cmake
set(RFDETR_SOURCES
    src/common.cpp
    src/image_io.cpp
    src/postprocess.cpp
    src/visualize.cpp
    src/cli.cpp
    src/model_loader.cpp
    src/rfdetr.cpp
    src/rfdetr_capi.cpp
    src/trace.cpp
)
```

- [ ] **Step 2: Write `src/trace.hpp`**

```cpp
#ifndef RFDETR_TRACE_HPP
#define RFDETR_TRACE_HPP

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct ggml_tensor;

namespace rfdetr {

/* A trace callback receives named intermediate tensors. The tensor pointer
 * is borrowed and only valid for the duration of the call (graph teardown
 * may invalidate it). The callback is expected to copy out any data it needs.
 *
 * Tensors passed to the callback have been ggml_backend_synchronize-d so
 * their data is readable via ggml_backend_tensor_get (or directly via
 * tensor->data on CPU). */
using trace_cb = std::function<void(const std::string& name, const ggml_tensor* t)>;

/* Install a callback for the current thread. Pass nullptr to clear. */
void set_trace_callback(trace_cb cb);

/* Forward-pass code calls this at each named checkpoint. No-op if no
 * callback is installed. */
void publish(const std::string& name, const ggml_tensor* t);

/* Helper: copy a published tensor's data into a std::vector<float>. The
 * source is assumed to be F32 on the CPU backend (Plan 3 is CPU-only). */
std::vector<float> copy_tensor_to_f32(const ggml_tensor* t);

/* Helper: shape vector. Returns up to GGML_MAX_DIMS entries; trailing 1's
 * are NOT trimmed. */
std::vector<int64_t> tensor_shape(const ggml_tensor* t);

}  // namespace rfdetr

#endif
```

- [ ] **Step 3: Write `src/trace.cpp`**

```cpp
#include "trace.hpp"

#include "ggml.h"

#include <cstring>

namespace rfdetr {

namespace {
thread_local trace_cb tls_cb;  // thread-local so concurrent contexts don't clash
}

void set_trace_callback(trace_cb cb) {
    tls_cb = std::move(cb);
}

void publish(const std::string& name, const ggml_tensor* t) {
    if (tls_cb) {
        tls_cb(name, t);
    }
}

std::vector<float> copy_tensor_to_f32(const ggml_tensor* t) {
    const size_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)t->data, out.data(), n);
    } else {
        // Generic path via ggml's quantize/dequantize would land here.
        // Plan 3 only ever publishes F32/F16; throw to surface unsupported.
        throw std::runtime_error("copy_tensor_to_f32: unsupported tensor type");
    }
    return out;
}

std::vector<int64_t> tensor_shape(const ggml_tensor* t) {
    std::vector<int64_t> s(GGML_MAX_DIMS);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) s[i] = t->ne[i];
    return s;
}

}  // namespace rfdetr
```

- [ ] **Step 4: Build to verify**

```bash
rm -rf build
cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=OFF
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build --output-on-failure
```

Expected: still 9/9 passing (no test exercises trace yet). The new `trace.cpp`
should compile cleanly. If `ggml_fp16_to_fp32_row` isn't available or has a
different name in v0.13.0, check `third_party/ggml/include/ggml.h` and adapt.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt src/trace.hpp src/trace.cpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(trace): thread-local trace callback for named intermediate tensors

Forward-pass code in subsequent tasks calls rfdetr::publish("name", t) at
each parity checkpoint. The callback is thread-local so concurrent contexts
don't clash; production inference installs no callback and publish becomes
a single function-pointer test + early return.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: backend selection (CPU only for now)

**Files:**
- Create: `src/backend.hpp`
- Create: `src/backend.cpp`
- Modify: `CMakeLists.txt` (add `src/backend.cpp` to RFDETR_SOURCES)

- [ ] **Step 1: Add `src/backend.cpp` to `RFDETR_SOURCES`**

```cmake
set(RFDETR_SOURCES
    src/common.cpp
    src/image_io.cpp
    src/postprocess.cpp
    src/visualize.cpp
    src/cli.cpp
    src/model_loader.cpp
    src/rfdetr.cpp
    src/rfdetr_capi.cpp
    src/trace.cpp
    src/backend.cpp
)
```

- [ ] **Step 2: Write `src/backend.hpp`**

```cpp
#ifndef RFDETR_BACKEND_HPP
#define RFDETR_BACKEND_HPP

#include "rfdetr.h"

#include <cstdint>

struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;

namespace rfdetr {

/* Initialize a backend. Plan 3: CPU only. Later plans honor build flags
 * RFDETR_GGML_CUDA / METAL / VULKAN / HIPBLAS. */
ggml_backend_t init_backend(int n_threads, rfdetr_status* out_status);

/* Release a backend created by init_backend. */
void free_backend(ggml_backend_t b);

/* True if the backend is CPU. Used to choose direct-memcpy vs
 * ggml_backend_tensor_get in tests. */
bool is_cpu(ggml_backend_t b);

}  // namespace rfdetr

#endif
```

- [ ] **Step 3: Write `src/backend.cpp`**

```cpp
#include "backend.hpp"
#include "common.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

namespace rfdetr {

ggml_backend_t init_backend(int n_threads, rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) { if (out_status) *out_status = s; };

    ggml_backend_t b = ggml_backend_cpu_init();
    if (!b) {
        rfdetr_logf(RFDETR_LOG_ERROR, "ggml_backend_cpu_init returned null");
        set(RFDETR_ERR_INFERENCE);
        return nullptr;
    }
    if (n_threads > 0) {
        ggml_backend_cpu_set_n_threads(b, n_threads);
    }
    set(RFDETR_OK);
    return b;
}

void free_backend(ggml_backend_t b) {
    if (b) ggml_backend_free(b);
}

bool is_cpu(ggml_backend_t b) {
    if (!b) return false;
    return ggml_backend_is_cpu(b);
}

}  // namespace rfdetr
```

- [ ] **Step 4: Build to verify**

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build --output-on-failure
```

Expected: 9/9 still pass.

If `ggml_backend_cpu_set_n_threads` or `ggml_backend_is_cpu` are spelled
differently in v0.13.0, check `third_party/ggml/include/ggml-backend.h` and
`third_party/ggml/include/ggml-cpu.h`. The structure is right; only names
may need adjustment.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt src/backend.hpp src/backend.cpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(backend): ggml backend selection (CPU only for now)

Plan 3 is CPU-only. The CMake backend flags from Plan 1 (RFDETR_GGML_CUDA
etc.) compile in the relevant ggml backends, but init_backend currently
always picks CPU. Plan 7+ extends this to prefer GPU when available.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: model_realize_weights — load tensor data into a backend buffer

This extends Plan 2's metadata-only loader to actually copy tensor data into a backend buffer.

**Files:**
- Modify: `src/model_loader.hpp` (add `model_realize_weights` declaration)
- Modify: `src/model_loader.cpp` (implement it)
- Modify: `tests/test_model_loader.cpp` (add a small block exercising it)

- [ ] **Step 1: Extend `src/model_loader.hpp`**

Add to the existing `Model` struct (inside namespace rfdetr):

```cpp
struct Model {
    Config config;
    ::gguf_context* gguf  = nullptr;
    ::ggml_context* meta  = nullptr;
    std::unordered_map<std::string, ::ggml_tensor*> tensors;

    /* Populated by model_realize_weights. nullptr until then. */
    ggml_backend_buffer_t weights = nullptr;
};
```

And forward-declare the new symbol at file scope (after the existing forward
decls):

```cpp
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
```

Add a new function declaration in the rfdetr namespace, after
`model_validate_tensors`:

```cpp
/* Allocate a backend buffer for the model's tensors and copy data from the
 * GGUF file into it. After this call, every tensor descriptor in
 * `m.tensors` is backed by real data on the supplied backend.
 *
 * Idempotent: if `m.weights` is already non-null, returns RFDETR_OK without
 * doing anything. Returns RFDETR_ERR_MODEL_LOAD on failure (logged). */
rfdetr_status model_realize_weights(Model& m, ggml_backend_t backend);
```

Also update `model_free` to release the buffer if present (visible in
the declaration's contract — actual change is in the .cpp).

- [ ] **Step 2: Extend `src/model_loader.cpp`**

At the top, add the new include:

```cpp
#include "ggml-backend.h"
```

In the `model_free` body, before `delete m;`, add:

```cpp
if (m->weights) ggml_backend_buffer_free(m->weights);
```

At the bottom of the file (after `model_validate_tensors`), add the new function:

```cpp
namespace rfdetr {

rfdetr_status model_realize_weights(Model& m, ggml_backend_t backend) {
    if (m.weights) return RFDETR_OK;

    if (!backend) {
        rfdetr_logf(RFDETR_LOG_ERROR, "model_realize_weights: null backend");
        return RFDETR_ERR_INVALID_ARG;
    }

    /* Allocate a buffer big enough for every tensor in m.meta, on the
     * supplied backend. */
    m.weights = ggml_backend_alloc_ctx_tensors(m.meta, backend);
    if (!m.weights) {
        rfdetr_logf(RFDETR_LOG_ERROR, "model_realize_weights: backend alloc failed");
        return RFDETR_ERR_MODEL_LOAD;
    }

    /* Re-open the GGUF file so we can stream tensor data. The original
     * gguf_context has the offsets; we re-read from disk to copy bytes. */
    const char* path = gguf_get_path(m.gguf);  // see note below if API differs
    if (!path) {
        /* gguf_init_from_file doesn't always retain the path. Fall back to
         * iterating tensors: each ggml_tensor* in m.tensors should have data
         * pointed at the mmapped GGUF body, depending on no_alloc mode.
         *
         * Plan 2 used no_alloc=true, which means the descriptors have no
         * data attached. We need the path here. If gguf_get_path doesn't
         * exist (it may not in v0.13.0), the caller must supply it via a
         * thread-local stash set by model_load. */
        rfdetr_logf(RFDETR_LOG_ERROR, "model_realize_weights: gguf path unavailable");
        return RFDETR_ERR_MODEL_LOAD;
    }

    FILE* fp = std::fopen(path, "rb");
    if (!fp) {
        rfdetr_logf(RFDETR_LOG_ERROR, "model_realize_weights: open failed: %s", path);
        return RFDETR_ERR_MODEL_LOAD;
    }

    const int64_t n_tensors = gguf_get_n_tensors(m.gguf);
    const size_t data_offset = gguf_get_data_offset(m.gguf);
    std::vector<uint8_t> buf;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(m.gguf, i);
        const size_t offset = data_offset + gguf_get_tensor_offset(m.gguf, i);
        ggml_tensor* t = ggml_get_tensor(m.meta, name);
        if (!t) {
            rfdetr_logf(RFDETR_LOG_ERROR, "model_realize_weights: tensor '%s' missing in ctx", name);
            std::fclose(fp);
            return RFDETR_ERR_MODEL_LOAD;
        }
        const size_t nbytes = ggml_nbytes(t);
        buf.resize(nbytes);

        if (std::fseek(fp, (long)offset, SEEK_SET) != 0 ||
            std::fread(buf.data(), 1, nbytes, fp) != nbytes) {
            rfdetr_logf(RFDETR_LOG_ERROR, "model_realize_weights: read failed for '%s'", name);
            std::fclose(fp);
            return RFDETR_ERR_MODEL_LOAD;
        }

        ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
    }

    std::fclose(fp);
    return RFDETR_OK;
}

}  // namespace rfdetr
```

**Critical caveats** the implementer must verify:

1. **`gguf_get_path` may not exist** in v0.13.0. If it doesn't, the loader
   needs to stash the path itself. The easiest fix: add `std::string path`
   to the `Model` struct, populate it during `model_load`, and read from
   `m.path` here. Use whichever approach actually compiles.
2. **`ggml_backend_alloc_ctx_tensors`** — confirm this function name in
   `third_party/ggml/include/ggml-backend.h`. It's the canonical pattern
   for "give me a buffer sized for all tensors in this context"; if the
   name differs, the equivalent function is the one to use.
3. **`gguf_get_data_offset`** vs **`gguf_get_tensor_offset`** — the former
   is the absolute file offset where the data section begins; the latter
   is the per-tensor offset RELATIVE TO that data section. Sum to get the
   absolute file offset. If the names differ, check the header.

Add `#include <cstdio>` and `#include <vector>` if not already present.

- [ ] **Step 3: Update `tests/test_model_loader.cpp`**

Add before `return 0;`:

```cpp
    // ---- Realize weights into a backend buffer ----
    {
        #include "backend.hpp"  // top-of-file include actually; add it there
        rfdetr_status st_;
        rfdetr::Model* mm = rfdetr::model_load(path, &st_);
        RFDETR_ASSERT(mm != nullptr);

        ggml_backend_t backend = rfdetr::init_backend(1, &st_);
        RFDETR_ASSERT(backend != nullptr);
        RFDETR_ASSERT_EQ_INT(st_, RFDETR_OK);

        rfdetr_status rs = rfdetr::model_realize_weights(*mm, backend);
        RFDETR_ASSERT_EQ_INT(rs, RFDETR_OK);
        RFDETR_ASSERT(mm->weights != nullptr);

        /* Idempotent: a second call is OK */
        rfdetr_status rs2 = rfdetr::model_realize_weights(*mm, backend);
        RFDETR_ASSERT_EQ_INT(rs2, RFDETR_OK);

        rfdetr::model_free(mm);
        rfdetr::free_backend(backend);
    }
```

Move the `#include "backend.hpp"` to the top of the file alongside the
existing `#include "model_loader.hpp"`. The inline `#include` shown above
is just a marker.

- [ ] **Step 4: Build and run**

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build -R test_model_loader --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model_loader.hpp src/model_loader.cpp tests/test_model_loader.cpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(loader): model_realize_weights — copy tensor data into backend buffer

Plan 2 kept the loader metadata-only (no_alloc=true) so rfdetr-cli info
could introspect without paying weight load cost. Plan 3 adds a separate
phase that allocates a ggml backend buffer and streams tensor data from
the GGUF data section into it. Idempotent; free path releases the buffer.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: rfdetr_init wires backend + realize_weights

**Files:**
- Modify: `src/rfdetr.cpp`
- Modify: `tests/test_capi.cpp` (assert successful init now means weights are loaded)

- [ ] **Step 1: Extend `src/rfdetr.cpp`**

Update the `rfdetr_context` struct:

```cpp
struct rfdetr_context {
    rfdetr::Model* model     = nullptr;
    ggml_backend_t backend   = nullptr;
    int            n_threads = 1;
};
```

Add the include at top:

```cpp
#include "backend.hpp"
```

In `rfdetr_init`, after `model_validate_tensors` succeeds and before allocating
the context, add:

```cpp
    rfdetr_status bk_st;
    ggml_backend_t backend = rfdetr::init_backend(
        params->n_threads > 0 ? params->n_threads : 1, &bk_st);
    if (!backend) {
        rfdetr::model_free(m);
        set(bk_st);
        return nullptr;
    }

    rfdetr_status rw_st = rfdetr::model_realize_weights(*m, backend);
    if (rw_st != RFDETR_OK) {
        rfdetr::free_backend(backend);
        rfdetr::model_free(m);
        set(rw_st);
        return nullptr;
    }
```

After the `ctx = new (std::nothrow) rfdetr_context()`, assign:

```cpp
    ctx->backend = backend;
```

In `rfdetr_free`, before `delete ctx;`, add:

```cpp
    rfdetr::free_backend(ctx->backend);
```

(model_free must run first so it sees `ctx->model->weights` and frees the
buffer before the backend is freed — order matters. Verify the order in
the existing code; if needed, restructure.)

- [ ] **Step 2: Update `tests/test_capi.cpp`**

The existing assertions still hold (init succeeds, detect returns NOT_IMPLEMENTED).
Add a check that init now uses more memory than before — actually no, that's
hard to assert portably. Skip.

Just verify init still passes and is faster than ~1 second (weight realize
for the 1.9MB fixture should be sub-second):

```cpp
    // After existing successful-init block:
    // (No new assertions; if init succeeds, weights are loaded.)
```

Actually, no new test assertions needed — the existing test exercises init+free
and the new behavior is "init now also loads weights." A failure in
`model_realize_weights` would make rfdetr_init return nullptr, and the
existing test would catch that.

- [ ] **Step 3: Build and run**

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: 9/9 pass. `rfdetr-cli info` should still work.

```bash
build/bin/rfdetr-cli info --model tests/fixtures/model_base.gguf
```

Should print the same info as before, just now with weights actually loaded.

- [ ] **Step 4: Commit**

```bash
git add src/rfdetr.cpp tests/test_capi.cpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(api): rfdetr_init now selects backend and realizes weights

Adds the two missing init phases — backend selection and
model_realize_weights. The context now owns a ggml_backend in addition
to the Model; rfdetr_free releases both in the correct order (model
first so its backend buffer is freed before the backend it lives on).
detect remains NOT_IMPLEMENTED.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: gen_model_gguf — add `--seed` flag for deterministic random weights

**Files:**
- Modify: `tests/fixtures/gen_model_gguf.cpp`

Plan 2's fixture writes zero-filled tensors, which makes the forward pass
trivially zero. Plan 3 needs non-trivial weights for parity testing. The
`--seed N` flag fills every tensor with `normal(0, 0.02)` (standard
PyTorch init scale for transformer weights).

- [ ] **Step 1: Extend `gen_model_gguf.cpp`**

At top of file, add includes (alongside existing ones):

```cpp
#include <random>
```

In `main()`, after parsing `argv[1]` and `--missing`, parse `--seed`:

```cpp
unsigned seed = 0;
bool seeded = false;
for (int i = 2; i + 1 < argc; ++i) {
    // ... existing --missing parse ...
    if (std::strcmp(argv[i], "--seed") == 0) {
        seed = (unsigned)std::strtoul(argv[i + 1], nullptr, 10);
        seeded = true;
    }
}
```

Replace the existing `std::memset(t->data, 0, ggml_nbytes(t));` loop with:

```cpp
if (seeded) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (auto* t : tensors) {
        size_t nelem = ggml_nelements(t);
        if (t->type == GGML_TYPE_F32) {
            float* p = (float*)t->data;
            for (size_t i = 0; i < nelem; ++i) p[i] = dist(rng);
        } else if (t->type == GGML_TYPE_F16) {
            ggml_fp16_t* p = (ggml_fp16_t*)t->data;
            for (size_t i = 0; i < nelem; ++i) {
                p[i] = ggml_fp32_to_fp16(dist(rng));
            }
        } else {
            std::fprintf(stderr, "gen_model_gguf: unsupported type for tensor %s\n",
                         ggml_get_name(t));
            return 4;
        }
    }
} else {
    for (auto* t : tensors) {
        std::memset(t->data, 0, ggml_nbytes(t));
    }
}
```

(If `ggml_fp32_to_fp16` is spelled differently in v0.13.0, check the header
and adapt.)

- [ ] **Step 2: Add a seeded fixture target in `tests/CMakeLists.txt`**

After the existing `rfdetr_model_fixture_missing` target, add:

```cmake
add_custom_command(
    OUTPUT  ${RFDETR_TEST_FIXTURES}/model_base_seeded.gguf
    COMMAND gen_model_gguf ${RFDETR_TEST_FIXTURES}/model_base_seeded.gguf --seed 42
    DEPENDS gen_model_gguf
    VERBATIM
)
add_custom_target(rfdetr_model_fixture_seeded ALL
    DEPENDS ${RFDETR_TEST_FIXTURES}/model_base_seeded.gguf)
```

- [ ] **Step 3: Build to verify it runs**

```bash
cmake --build build -j 2>&1 | tail -5
ls -lh tests/fixtures/model_base_seeded.gguf
```

Expected: file exists, ~1.9MB. Sanity-check that two regenerations with the
same seed produce byte-identical output:

```bash
sha256sum tests/fixtures/model_base_seeded.gguf
build/bin/gen_model_gguf /tmp/again.gguf --seed 42
sha256sum /tmp/again.gguf
```

Both hashes should match.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/gen_model_gguf.cpp tests/CMakeLists.txt
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
test(fixture): --seed flag for deterministic random weights

Without a seed the generator still zero-fills (Plan 2 behavior unchanged).
With --seed N, every tensor is filled with N(0, 0.02) via std::mt19937 —
matches PyTorch's default transformer init scale. Plan 3 parity tests use
model_base_seeded.gguf which is generated with --seed 42 at build time;
two regens with the same seed produce byte-identical output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: dinov2.hpp — declare the operations

**Files:**
- Create: `src/dinov2.hpp`

- [ ] **Step 1: Write `src/dinov2.hpp`**

```cpp
#ifndef RFDETR_DINOV2_HPP
#define RFDETR_DINOV2_HPP

#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace rfdetr {

/* Build the patch_embed forward graph node.
 *
 * Input:  `input` — (1, H, W, 3) F32 image already mean/std normalized
 * Output: a token tensor (1, N_patches, dim) F32, where
 *           N_patches = (H / 14) * (W / 14)
 *
 * Publishes "backbone.patch_embed.output" via the trace callback. */
ggml_tensor* dinov2_patch_embed(ggml_context* ctx, const Model& m,
                                ggml_tensor* input);

/* Build one DINOv2 transformer block (pre-LN style):
 *
 *   x = x + attn(norm1(x))
 *   x = x + mlp(norm2(x))
 *
 * Input/output shape: (1, N_patches, dim) F32.
 *
 * Publishes:
 *   "backbone.block.{idx}.norm1.output"
 *   "backbone.block.{idx}.attn.output"
 *   "backbone.block.{idx}.mlp.output"
 *   "backbone.block.{idx}.output"
 *
 * Plan 3 uses global self-attention (window_size ignored). Plan 4 adds
 * window-attention switching. */
ggml_tensor* dinov2_block(ggml_context* ctx, const Model& m,
                          ggml_tensor* x, int block_idx);

}  // namespace rfdetr

#endif
```

- [ ] **Step 2: Commit**

```bash
git add src/dinov2.hpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(dinov2): declare patch_embed and transformer-block builders

Header only. Implementation lands in Task 8.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: dinov2.cpp — patch_embed implementation

**Files:**
- Create: `src/dinov2.cpp`
- Modify: `CMakeLists.txt` (add to RFDETR_SOURCES)

We split implementation across Tasks 8 (patch_embed) and Task 9 (block). Keeps task size manageable.

- [ ] **Step 1: Add `src/dinov2.cpp` to `RFDETR_SOURCES`**

```cmake
set(RFDETR_SOURCES
    ...
    src/backend.cpp
    src/dinov2.cpp
)
```

- [ ] **Step 2: Write `src/dinov2.cpp` (patch_embed only)**

```cpp
#include "dinov2.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <cmath>
#include <string>

namespace rfdetr {

ggml_tensor* dinov2_patch_embed(ggml_context* ctx, const Model& m,
                                ggml_tensor* input) {
    /* DINOv2 patch embedding is a Conv2d with kernel=stride=14.
     *
     * Input shape: (1, H, W, 3)  [NHWC layout in ggml convention is (3, W, H, 1)]
     * Weight:      `backbone.patch_embed.weight` shape (14, 14, 3, dim) in ggml
     *              column-major convention  ⇒ in PyTorch convention it's
     *              (dim, 3, 14, 14).
     * Bias:        `backbone.patch_embed.bias` shape (dim).
     *
     * Output shape: (1, N_patches, dim)  where N_patches = (H/14) * (W/14).
     *
     * Implementation: ggml_conv_2d with stride 14, padding 0. Result is
     * (1, H/14, W/14, dim); we reshape to (1, N_patches, dim). */

    auto it_w = m.tensors.find("backbone.patch_embed.weight");
    auto it_b = m.tensors.find("backbone.patch_embed.bias");
    if (it_w == m.tensors.end() || it_b == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "dinov2_patch_embed: missing patch_embed weight or bias");
        return nullptr;
    }
    ggml_tensor* W = it_w->second;
    ggml_tensor* b = it_b->second;

    /* ggml_conv_2d(ctx, kernel, input, s0, s1, p0, p1, d0, d1).
     * Stride 14, no padding, no dilation. */
    ggml_tensor* conv = ggml_conv_2d(ctx, W, input,
                                     /*s0*/ 14, /*s1*/ 14,
                                     /*p0*/ 0,  /*p1*/ 0,
                                     /*d0*/ 1,  /*d1*/ 1);

    /* Add bias broadcast across spatial dims. ggml_add with a (dim) tensor
     * broadcasts across the leading dims by default if shapes are compatible.
     * Verify with the actual ggml API; may need ggml_repeat first. */
    ggml_tensor* with_bias = ggml_add(ctx, conv, b);

    /* Reshape (H/14, W/14, dim, 1) → (dim, N_patches) → permute → (N_patches, dim, 1, 1)
     *
     * In ggml's column-major convention, "dim is the first axis" means
     * `ne[0] = dim`. We need the final tensor with `ne[0] = dim` and
     * `ne[1] = N_patches` so the rest of the graph sees (N_patches, dim)
     * in row-major terms.
     *
     * Steps:
     *   conv output ne = (H/14, W/14, dim, 1)  // ggml convention
     *   reshape to (H/14 * W/14, dim, 1, 1)
     *   permute to (dim, N_patches, 1, 1)
     *
     * The order of operations depends on ggml's reshape/permute semantics.
     * Verify by inspecting conv->ne after creation and adjusting. */
    int64_t N = with_bias->ne[0] * with_bias->ne[1];  // H/14 * W/14
    ggml_tensor* tokens = ggml_reshape_3d(ctx, with_bias,
                                          N,
                                          with_bias->ne[2],  // dim
                                          1);
    /* Now ne = (N, dim, 1). Permute to (dim, N, 1, 1) so downstream code
     * can index as (dim, token_index). */
    ggml_tensor* out = ggml_permute(ctx, tokens, 1, 0, 2, 3);
    out = ggml_cont(ctx, out);

    publish("backbone.patch_embed.output", out);
    return out;
}

/* dinov2_block lands in Task 9. */

}  // namespace rfdetr
```

**Critical caveats** the implementer must verify:

1. **ggml_conv_2d input layout.** ggml's conv_2d expects input in (W, H, C, N)
   column-major convention. The input image will arrive as (3, W, H, 1) if
   we set it up that way. Be deliberate about layout when constructing the
   input tensor in Task 11.
2. **`ggml_add` broadcasting with bias.** ggml may require explicit
   `ggml_repeat` or `ggml_add1` for broadcast. If `ggml_add` doesn't broadcast
   automatically, use `ggml_add` after `ggml_repeat(b, conv_shape_template)`.
3. **Reshape arithmetic** must respect column-major (ggml convention). When in
   doubt, print `ne[]` of the tensor after each operation in a small standalone
   test program and verify against expectations.

- [ ] **Step 3: Smoke build (no test exercises it yet)**

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build --output-on-failure
```

Expected: 9/9 pass (no new test exercises patch_embed in isolation; Task 11 adds the test).

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt src/dinov2.cpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(dinov2): patch_embed forward graph builder

Conv2d (kernel=stride=14) + bias add + reshape/permute to (dim, N_patches)
token layout. Publishes "backbone.patch_embed.output" via trace callback.
No test yet — Task 11 wires the parity harness.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: dinov2.cpp — one transformer block

**Files:**
- Modify: `src/dinov2.cpp` (append `dinov2_block`)

- [ ] **Step 1: Append `dinov2_block` to `src/dinov2.cpp`**

```cpp
namespace rfdetr {

namespace {

/* LayerNorm: x_normalized = (x - mean) / sqrt(var + eps) * weight + bias
 * Operates on the last dim. eps = 1e-5 matches PyTorch default. */
ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* weight, ggml_tensor* bias) {
    constexpr float eps = 1e-5f;
    ggml_tensor* y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, weight);
    y = ggml_add(ctx, y, bias);
    return y;
}

/* Multi-head self-attention with packed QKV projection.
 *
 *   qkv = x @ Wqkv.T + bqkv         shape (N, 3*dim)
 *   q,k,v = split qkv into thirds   each shape (N, dim)
 *   reshape each to (N, n_heads, head_dim) and transpose to (n_heads, N, head_dim)
 *   attn_logits = q @ k.T / sqrt(head_dim)    shape (n_heads, N, N)
 *   attn = softmax(attn_logits)
 *   out = attn @ v                  shape (n_heads, N, head_dim)
 *   merge heads back to (N, dim)
 *   out = out @ Wproj.T + bproj     shape (N, dim)
 */
ggml_tensor* mha(ggml_context* ctx, ggml_tensor* x,
                 ggml_tensor* Wqkv, ggml_tensor* bqkv,
                 ggml_tensor* Wproj, ggml_tensor* bproj,
                 int n_heads) {
    /* In ggml: x has ne = (dim, N, 1, 1).
     * Wqkv has ne = (dim, 3*dim, 1, 1) so ggml_mul_mat(Wqkv, x) returns (3*dim, N). */
    ggml_tensor* qkv = ggml_mul_mat(ctx, Wqkv, x);
    qkv = ggml_add(ctx, qkv, bqkv);  // bias broadcast over N
    /* qkv ne = (3*dim, N, 1, 1). Split into thirds along dim 0. */
    const int dim = (int)x->ne[0];
    const int head_dim = dim / n_heads;
    ggml_tensor* q = ggml_view_2d(ctx, qkv, dim, qkv->ne[1],
                                  qkv->nb[1], 0 * dim * sizeof(float));
    ggml_tensor* k = ggml_view_2d(ctx, qkv, dim, qkv->ne[1],
                                  qkv->nb[1], 1 * dim * sizeof(float));
    ggml_tensor* v = ggml_view_2d(ctx, qkv, dim, qkv->ne[1],
                                  qkv->nb[1], 2 * dim * sizeof(float));

    /* Reshape each to (head_dim, n_heads, N, 1) then permute to (head_dim, N, n_heads, 1)
     * so per-head ggml_mul_mat works. */
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, q->ne[1]);
    q = ggml_permute(ctx, q, 0, 2, 1, 3); q = ggml_cont(ctx, q);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, k->ne[1]);
    k = ggml_permute(ctx, k, 0, 2, 1, 3); k = ggml_cont(ctx, k);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, v->ne[1]);
    v = ggml_permute(ctx, v, 0, 2, 1, 3); v = ggml_cont(ctx, v);

    /* attn = softmax(q @ k.T / sqrt(head_dim)) @ v
     *
     * ggml_flash_attn_ext (if available) does the whole thing fused. If not,
     * compose manually. */
    ggml_tensor* out = ggml_flash_attn_ext(ctx, q, k, v, /*mask*/ nullptr,
                                           /*scale*/ 1.0f / std::sqrt((float)head_dim),
                                           /*max_bias*/ 0.0f,
                                           /*logit_softcap*/ 0.0f);

    /* out ne = (head_dim, N, n_heads, 1). Permute back to (head_dim, n_heads, N, 1)
     * and reshape to (dim, N, 1, 1). */
    out = ggml_permute(ctx, out, 0, 2, 1, 3);
    out = ggml_cont(ctx, out);
    out = ggml_reshape_2d(ctx, out, dim, out->ne[2]);

    /* Project: (dim, dim) @ (dim, N) = (dim, N) */
    out = ggml_mul_mat(ctx, Wproj, out);
    out = ggml_add(ctx, out, bproj);
    return out;
}

/* MLP: x → fc1 → GELU → fc2. fc1: (dim, 4*dim), fc2: (4*dim, dim). */
ggml_tensor* mlp(ggml_context* ctx, ggml_tensor* x,
                 ggml_tensor* W1, ggml_tensor* b1,
                 ggml_tensor* W2, ggml_tensor* b2) {
    ggml_tensor* h = ggml_mul_mat(ctx, W1, x);
    h = ggml_add(ctx, h, b1);
    h = ggml_gelu(ctx, h);   // use ggml_gelu, not ggml_gelu_quick — match PyTorch exactly
    h = ggml_mul_mat(ctx, W2, h);
    h = ggml_add(ctx, h, b2);
    return h;
}

}  // namespace

ggml_tensor* dinov2_block(ggml_context* ctx, const Model& m,
                          ggml_tensor* x, int block_idx) {
    const std::string p = "backbone.blocks." + std::to_string(block_idx) + ".";
    auto get = [&](const char* suffix) -> ggml_tensor* {
        auto it = m.tensors.find(p + suffix);
        if (it == m.tensors.end()) {
            rfdetr_logf(RFDETR_LOG_ERROR, "dinov2_block: missing tensor '%s'",
                        (p + suffix).c_str());
            return nullptr;
        }
        return it->second;
    };

    ggml_tensor* n1w = get("norm1.weight");
    ggml_tensor* n1b = get("norm1.bias");
    ggml_tensor* qkvW = get("attn.qkv.weight");
    ggml_tensor* qkvB = get("attn.qkv.bias");
    ggml_tensor* prW  = get("attn.proj.weight");
    ggml_tensor* prB  = get("attn.proj.bias");
    ggml_tensor* n2w = get("norm2.weight");
    ggml_tensor* n2b = get("norm2.bias");
    ggml_tensor* f1W = get("mlp.fc1.weight");
    ggml_tensor* f1B = get("mlp.fc1.bias");
    ggml_tensor* f2W = get("mlp.fc2.weight");
    ggml_tensor* f2B = get("mlp.fc2.bias");
    if (!n1w || !n1b || !qkvW || !qkvB || !prW || !prB ||
        !n2w || !n2b || !f1W || !f1B || !f2W || !f2B) {
        return nullptr;
    }

    /* x = x + attn(norm1(x)) */
    ggml_tensor* y = layer_norm(ctx, x, n1w, n1b);
    publish(p + "norm1.output", y);
    y = mha(ctx, y, qkvW, qkvB, prW, prB, (int)m.config.backbone.heads);
    publish(p + "attn.output", y);
    x = ggml_add(ctx, x, y);

    /* x = x + mlp(norm2(x)) */
    y = layer_norm(ctx, x, n2w, n2b);
    y = mlp(ctx, y, f1W, f1B, f2W, f2B);
    publish(p + "mlp.output", y);
    x = ggml_add(ctx, x, y);

    publish(p + "output", x);
    return x;
}

}  // namespace rfdetr
```

**Critical caveats**:

1. **`ggml_flash_attn_ext`** may not exist in v0.13.0, or may have a different
   signature. If it's absent, compose manually:
   ```cpp
   ggml_tensor* kt = ggml_permute(ctx, k, 1, 0, 2, 3);  // transpose last two dims
   ggml_tensor* logits = ggml_mul_mat(ctx, kt, q);      // q @ k.T
   logits = ggml_scale(ctx, logits, 1.0f / std::sqrt((float)head_dim));
   logits = ggml_soft_max(ctx, logits);
   ggml_tensor* out = ggml_mul_mat(ctx, v, logits);     // attn @ v
   ```
   Adapt to the actual API.
2. **GELU vs gelu_approx**: DINOv2 uses exact GELU (tanh approximation in some
   PyTorch versions). Use whichever matches the upstream rfdetr 1.x. The numpy
   reference must use the same.
3. **LayerNorm epsilon**: 1e-5 matches PyTorch's default `torch.nn.LayerNorm`.
   If rfdetr uses a custom epsilon, change here AND in the numpy reference.

- [ ] **Step 2: Build to verify compilation**

```bash
cmake --build build -j 2>&1 | tail -15
```

Expected: clean compile. No test exercises this code yet (Task 11 does). If
any of the cited ggml functions don't exist by that exact name, adapt.

- [ ] **Step 3: Commit**

```bash
git add src/dinov2.cpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(dinov2): one transformer block (LN + MHA + LN + MLP, pre-LN style)

Single block implementation with packed-QKV attention, residual connections,
and trace publishes at norm1/attn/mlp/output. No test yet — Task 11 adds
parity coverage. Window attention deferred to Plan 4; this is the global path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Python numpy reference and baseline-bundle writer

This Python script reads the seeded GGUF fixture, computes patch_embed + block 0 in numpy, and writes the expected intermediate tensors to a baseline-bundle GGUF.

**Files:**
- Create: `scripts/gen_numpy_baseline.py`
- Modify: `scripts/requirements.txt` — add numpy & gguf only (no torch)

- [ ] **Step 1: Add deps to `scripts/requirements.txt`**

Note: gguf is already pinned from Plan 2. Just ensure numpy is too. Plan 3
does not add torch.

- [ ] **Step 2: Write `scripts/gen_numpy_baseline.py`**

```python
#!/usr/bin/env python3
"""Generate a baseline-bundle GGUF from a Plan-2 model GGUF by computing
forward through patch_embed + backbone block 0 in numpy.

The intermediate tensors are written to a new GGUF as parity.<checkpoint>
tensors. The C++ test_parity_block0 consumes this bundle.

Usage:
    python3 scripts/gen_numpy_baseline.py \\
        --model tests/fixtures/model_base_seeded.gguf \\
        --output tests/fixtures/baseline_block0.gguf \\
        [--input-seed 7]

Format version: "1" (see docs/parity.md).
"""

import argparse
import sys
import numpy as np

try:
    import gguf
except ImportError:
    print("error: 'gguf' package not installed. pip install -r scripts/requirements.txt",
          file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--input-seed", type=int, default=7,
                   help="Seed for the deterministic synthetic input image.")
    return p.parse_args()


def read_model(path):
    """Open a Plan-2 GGUF and return (config_dict, tensors_dict)."""
    reader = gguf.GGUFReader(path)

    def get_field(name):
        f = reader.get_field(name)
        if f is None:
            raise KeyError(f"missing metadata: {name}")
        # gguf-py returns fields as (data, type, …) — extract appropriately
        return f

    cfg = {
        "variant":        reader.get_field("rfdetr.variant").parts[-1].tobytes().decode(),
        "image_size":     int(reader.get_field("rfdetr.image_size").parts[-1][0]),
        "num_queries":    int(reader.get_field("rfdetr.num_queries").parts[-1][0]),
        "num_classes":    int(reader.get_field("rfdetr.num_classes").parts[-1][0]),
        "bb_dim":         int(reader.get_field("rfdetr.backbone.dim").parts[-1][0]),
        "bb_depth":       int(reader.get_field("rfdetr.backbone.depth").parts[-1][0]),
        "bb_heads":       int(reader.get_field("rfdetr.backbone.heads").parts[-1][0]),
    }

    tensors = {}
    for t in reader.tensors:
        # convert F16/F32 raw to numpy
        if t.tensor_type == gguf.GGMLQuantizationType.F16:
            arr = np.frombuffer(t.data, dtype=np.float16).astype(np.float32)
        elif t.tensor_type == gguf.GGMLQuantizationType.F32:
            arr = np.frombuffer(t.data, dtype=np.float32)
        else:
            raise ValueError(f"unsupported tensor type for {t.name}: {t.tensor_type}")
        arr = arr.reshape(tuple(reversed(t.shape)))  # ggml ne[] -> numpy shape
        tensors[t.name] = arr
    return cfg, tensors


def layer_norm(x, w, b, eps=1e-5):
    """Last-axis LayerNorm. Matches PyTorch torch.nn.LayerNorm(elementwise_affine=True)."""
    mean = x.mean(-1, keepdims=True)
    var  = x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def gelu(x):
    """Exact GELU (PyTorch default torch.nn.GELU with no 'tanh' approximation)."""
    return 0.5 * x * (1.0 + np.vectorize(np.math.erf)(x / np.sqrt(2.0)))


def mha(x, Wqkv, bqkv, Wproj, bproj, n_heads):
    # x: (N, dim)
    N, dim = x.shape
    head_dim = dim // n_heads
    qkv = x @ Wqkv.T + bqkv  # (N, 3*dim)
    q, k, v = np.split(qkv, 3, axis=-1)  # each (N, dim)
    q = q.reshape(N, n_heads, head_dim).transpose(1, 0, 2)  # (n_heads, N, head_dim)
    k = k.reshape(N, n_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(N, n_heads, head_dim).transpose(1, 0, 2)
    scale = 1.0 / np.sqrt(head_dim)
    logits = q @ k.transpose(0, 2, 1) * scale  # (n_heads, N, N)
    logits -= logits.max(-1, keepdims=True)
    a = np.exp(logits)
    a /= a.sum(-1, keepdims=True)
    out = a @ v  # (n_heads, N, head_dim)
    out = out.transpose(1, 0, 2).reshape(N, dim)
    return out @ Wproj.T + bproj


def mlp(x, W1, b1, W2, b2):
    h = x @ W1.T + b1
    h = gelu(h)
    return h @ W2.T + b2


def forward(cfg, tensors, input_img):
    """Run patch_embed + block 0. Returns dict of named intermediate tensors."""
    out = {"preprocess.input": input_img}

    dim = cfg["bb_dim"]
    # patch_embed weight in PyTorch convention: (dim, 3, 14, 14)
    pew = tensors["backbone.patch_embed.weight"]
    peb = tensors["backbone.patch_embed.bias"]
    # input_img: (1, H, W, 3). Reshape to (H/14, 14, W/14, 14, 3), then
    # sum-reduce to (H/14, W/14, dim) by einsum with kernel.
    H, W = input_img.shape[1], input_img.shape[2]
    Hp = H // 14
    Wp = W // 14
    img = input_img[0]  # (H, W, 3)
    img = img.reshape(Hp, 14, Wp, 14, 3)
    # einsum: 'hpwqc, dchp wq -> hpwpd' style. Easier: build patches and matmul.
    patches = img.transpose(0, 2, 1, 3, 4).reshape(Hp * Wp, 14 * 14 * 3)  # (N, 588)
    kernel = pew.reshape(dim, 14 * 14 * 3)  # (dim, 588)
    tokens = patches @ kernel.T + peb  # (N, dim)
    out["backbone.patch_embed.output"] = tokens.reshape(1, Hp * Wp, dim)

    x = tokens  # (N, dim)

    p = "backbone.blocks.0."
    n1 = layer_norm(x, tensors[p + "norm1.weight"], tensors[p + "norm1.bias"])
    out["backbone.block.0.norm1.output"] = n1.reshape(1, *n1.shape)
    y = mha(n1,
            tensors[p + "attn.qkv.weight"], tensors[p + "attn.qkv.bias"],
            tensors[p + "attn.proj.weight"], tensors[p + "attn.proj.bias"],
            cfg["bb_heads"])
    out["backbone.block.0.attn.output"] = y.reshape(1, *y.shape)
    x = x + y

    n2 = layer_norm(x, tensors[p + "norm2.weight"], tensors[p + "norm2.bias"])
    z = mlp(n2,
            tensors[p + "mlp.fc1.weight"], tensors[p + "mlp.fc1.bias"],
            tensors[p + "mlp.fc2.weight"], tensors[p + "mlp.fc2.bias"])
    out["backbone.block.0.mlp.output"] = z.reshape(1, *z.shape)
    x = x + z

    out["backbone.block.0.output"] = x.reshape(1, *x.shape)
    return out


def write_baseline(path, intermediates, input_shape):
    w = gguf.GGUFWriter(path, arch="rfdetr-parity")
    w.add_string("parity.format.version", "1")
    w.add_string("parity.reference", "numpy")
    w.add_array("parity.input_shape", [int(x) for x in input_shape])
    for name, arr in intermediates.items():
        w.add_tensor(f"parity.{name}", arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main():
    args = parse_args()
    cfg, tensors = read_model(args.model)

    rng = np.random.default_rng(args.input_seed)
    H = W = cfg["image_size"]
    # ImageNet-normalized: input image is roughly N(0, ~1) after normalization.
    # Use that to keep arithmetic in a sensible range.
    input_img = rng.normal(0.0, 1.0, size=(1, H, W, 3)).astype(np.float32)

    intermediates = forward(cfg, tensors, input_img)
    write_baseline(args.output, intermediates, input_img.shape)

    print(f"Wrote {args.output} with {len(intermediates)} parity tensors", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`chmod +x scripts/gen_numpy_baseline.py`.

- [ ] **Step 3: Test it manually**

```bash
# Need numpy + gguf installed. If not, install just those two (NOT torch/rfdetr):
pip install numpy gguf
# Or use a venv: python -m venv .venv && source .venv/bin/activate && pip install numpy gguf

# The seeded fixture must exist; build it if needed:
cmake --build build --target rfdetr_model_fixture_seeded

python3 scripts/gen_numpy_baseline.py \
    --model tests/fixtures/model_base_seeded.gguf \
    --output tests/fixtures/baseline_block0.gguf
ls -lh tests/fixtures/baseline_block0.gguf
```

Expected: a baseline bundle file is written (~few MB).

If the gguf-py field access (e.g. `reader.get_field(...).parts[-1]`) doesn't
work in the installed gguf version, adjust. The gguf-py reader API is
slightly different across versions; whichever pattern is current is fine
as long as it correctly extracts the metadata.

- [ ] **Step 4: Wire as a CMake custom command**

In `tests/CMakeLists.txt`, after the seeded fixture target:

```cmake
find_program(PYTHON3 python3 REQUIRED)
add_custom_command(
    OUTPUT  ${RFDETR_TEST_FIXTURES}/baseline_block0.gguf
    COMMAND ${PYTHON3} ${CMAKE_SOURCE_DIR}/scripts/gen_numpy_baseline.py
            --model  ${RFDETR_TEST_FIXTURES}/model_base_seeded.gguf
            --output ${RFDETR_TEST_FIXTURES}/baseline_block0.gguf
    DEPENDS ${RFDETR_TEST_FIXTURES}/model_base_seeded.gguf
            ${CMAKE_SOURCE_DIR}/scripts/gen_numpy_baseline.py
    VERBATIM
)
add_custom_target(rfdetr_baseline_block0 ALL
    DEPENDS ${RFDETR_TEST_FIXTURES}/baseline_block0.gguf)
```

This makes the baseline regenerate whenever either the seeded fixture or
the script changes. Note: this requires `numpy` and `gguf` installed
system-wide (or via the user's venv). If the build host doesn't have
them, the custom command fails and `test_parity_block0` won't run — that's
a tolerable trade-off and the failure is clear.

If the user/CI doesn't want a hard python+numpy dep at build time, an
alternative is to check in the baseline_block0.gguf as a binary fixture
(after a manual regen). Plan 3 picks the "regen at build" path because the
gguf is data-driven from the seeded model and any drift in either side
should be caught immediately.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_numpy_baseline.py tests/CMakeLists.txt
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(parity): numpy reference for patch_embed + backbone block 0

Pure-numpy implementation of DINOv2 patch embedding and a single transformer
block (LN + MHA + LN + MLP). Reads a Plan-2 GGUF (seeded with --seed 42),
computes the forward through block 0, writes intermediate tensors to a
parity baseline-bundle GGUF (format version "1") for the C++ test to
consume. No torch dependency — Plan 6 swaps in a torch reference using the
same baseline-bundle format.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: test_parity_block0 — C++ vs numpy parity

**Files:**
- Create: `tests/test_parity_block0.cpp`
- Modify: `tests/CMakeLists.txt` (register the test)

- [ ] **Step 1: Register the test**

In `tests/CMakeLists.txt`, append:

```cmake
rfdetr_add_test(test_parity_block0)
add_dependencies(test_parity_block0
    rfdetr_model_fixture_seeded
    rfdetr_baseline_block0)
```

- [ ] **Step 2: Write `tests/test_parity_block0.cpp`**

```cpp
/* C++ forward pass through patch_embed + backbone block 0 vs numpy baseline. */
#include "test_assert.hpp"
#include "rfdetr.h"
#include "model_loader.hpp"
#include "backend.hpp"
#include "trace.hpp"
#include "dinov2.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Tol { float atol; float rtol; };

const std::map<std::string, Tol> kTolerances = {
    {"backbone.patch_embed.output",    {1e-4f, 1e-3f}},
    {"backbone.block.0.norm1.output",  {1e-4f, 1e-3f}},
    {"backbone.block.0.attn.output",   {1e-3f, 1e-2f}},
    {"backbone.block.0.mlp.output",    {1e-3f, 1e-2f}},
    {"backbone.block.0.output",        {1e-3f, 1e-2f}},
};

/* Load a baseline-bundle GGUF: returns a map from checkpoint name (without
 * the "parity." prefix) to its expected float32 data. Also reads
 * parity.input_shape so the test can reconstruct the input. */
struct Baseline {
    std::unordered_map<std::string, std::vector<float>> tensors;
    std::vector<int64_t> input_shape;
};

Baseline load_baseline(const std::string& path) {
    Baseline b;

    ggml_context* ctx = nullptr;
    gguf_init_params p{ /* no_alloc */ false, /* ctx */ &ctx };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    RFDETR_ASSERT(g != nullptr);

    int64_t kid = gguf_find_key(g, "parity.input_shape");
    RFDETR_ASSERT(kid >= 0);
    size_t n_shape = gguf_get_arr_n(g, kid);
    const int32_t* sd = (const int32_t*)gguf_get_arr_data(g, kid);
    for (size_t i = 0; i < n_shape; ++i) b.input_shape.push_back(sd[i]);

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
        size_t nelem = ggml_nelements(t);
        std::vector<float> v(nelem);
        std::memcpy(v.data(), t->data, nelem * sizeof(float));
        b.tensors.emplace(std::move(key), std::move(v));
    }

    gguf_free(g);
    ggml_free(ctx);
    return b;
}

/* Compare two flat arrays. Returns true on pass. On failure prints details. */
bool allclose(const std::string& name,
              const std::vector<float>& got,
              const std::vector<float>& want,
              float atol, float rtol) {
    if (got.size() != want.size()) {
        std::fprintf(stderr, "[%s] size mismatch: got=%zu want=%zu\n",
                     name.c_str(), got.size(), want.size());
        return false;
    }
    float max_abs = 0.0f;
    size_t max_idx = 0;
    double sum_abs = 0.0;
    size_t fail_count = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        float diff = std::fabs(got[i] - want[i]);
        float tol  = atol + rtol * std::fabs(want[i]);
        sum_abs += diff;
        if (diff > max_abs) { max_abs = diff; max_idx = i; }
        if (diff > tol) ++fail_count;
    }
    if (fail_count == 0) {
        std::fprintf(stderr, "[%s] OK (max_abs=%g, mean_abs=%g)\n",
                     name.c_str(), max_abs, sum_abs / got.size());
        return true;
    }
    std::fprintf(stderr,
                 "[%s] FAIL (max_abs=%g at idx %zu — got=%g want=%g) "
                 "atol=%g rtol=%g, %zu/%zu over tol\n",
                 name.c_str(), max_abs, max_idx,
                 got[max_idx], want[max_idx],
                 atol, rtol, fail_count, got.size());
    return false;
}

}  // namespace

int main() {
    const std::string fixtures = RFDETR_TEST_FIXTURES;
    const std::string model_path    = fixtures + "/model_base_seeded.gguf";
    const std::string baseline_path = fixtures + "/baseline_block0.gguf";

    /* Load baseline first so we know the input shape. */
    Baseline base = load_baseline(baseline_path);
    RFDETR_ASSERT(base.input_shape.size() == 4);  // (1, H, W, 3)

    auto it_input = base.tensors.find("preprocess.input");
    RFDETR_ASSERT(it_input != base.tensors.end());
    const std::vector<float>& input_data = it_input->second;

    /* Open model + backend + realize weights. */
    rfdetr_status st;
    rfdetr::Model* m = rfdetr::model_load(model_path, &st);
    RFDETR_ASSERT(m != nullptr);
    ggml_backend_t backend = rfdetr::init_backend(1, &st);
    RFDETR_ASSERT(backend != nullptr);
    RFDETR_ASSERT_EQ_INT(rfdetr::model_realize_weights(*m, backend), RFDETR_OK);

    /* Build a compute graph for patch_embed + block 0. */
    ggml_init_params ip{};
    ip.mem_size   = 64 * 1024 * 1024;  // 64MB scratch for the graph
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context* gctx = ggml_init(ip);
    RFDETR_ASSERT(gctx != nullptr);

    /* Input tensor in ggml convention (W, H, C, N) = (W, H, 3, 1) */
    const int64_t H = base.input_shape[1], W = base.input_shape[2];
    ggml_tensor* input = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, W, H, 3, 1);
    ggml_set_name(input, "input");

    ggml_tensor* t = rfdetr::dinov2_patch_embed(gctx, *m, input);
    RFDETR_ASSERT(t != nullptr);
    t = rfdetr::dinov2_block(gctx, *m, t, /*block_idx*/ 0);
    RFDETR_ASSERT(t != nullptr);

    /* Allocate compute buffers + set the input data + run. */
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
    RFDETR_ASSERT(buf != nullptr);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    ggml_cgraph* graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, t);

    /* Install trace callback that captures published tensors. */
    std::map<std::string, std::vector<float>> got;
    rfdetr::set_trace_callback(
        [&](const std::string& name, const ggml_tensor* tt) {
            got[name] = rfdetr::copy_tensor_to_f32(tt);
        });

    /* Run. */
    auto status = ggml_backend_graph_compute(backend, graph);
    RFDETR_ASSERT(status == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    /* For every checkpoint with a tolerance, diff and fail if not within tol. */
    bool ok = true;
    for (const auto& [name, tol] : kTolerances) {
        auto g = got.find(name);
        auto w = base.tensors.find(name);
        if (g == got.end()) {
            std::fprintf(stderr, "[%s] FAIL: not published by C++ forward pass\n",
                         name.c_str());
            ok = false;
            continue;
        }
        if (w == base.tensors.end()) {
            std::fprintf(stderr, "[%s] FAIL: not present in baseline\n", name.c_str());
            ok = false;
            continue;
        }
        if (!allclose(name, g->second, w->second, tol.atol, tol.rtol)) {
            ok = false;
        }
    }

    rfdetr::set_trace_callback(nullptr);
    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    rfdetr::model_free(m);
    rfdetr::free_backend(backend);

    return ok ? 0 : 1;
}
```

- [ ] **Step 3: Build and run**

```bash
rm -rf build
cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=OFF
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build --output-on-failure
```

Expected: 10/10 pass — or, more realistically, `test_parity_block0` fails
the first time because of a layout mismatch between the C++ graph and the
numpy reference. This is the moment we diagnose and fix. Common first
divergences:

1. **patch_embed**: numpy's einsum order vs ggml's conv_2d layout. Print the
   `ne[]` of the conv output and check which axis is `dim` vs which is
   `N_patches`.
2. **MHA**: head split direction (whether heads are leading or interleaved
   in the qkv concatenation).
3. **LayerNorm epsilon mismatch**: numpy uses 1e-5; verify ggml_norm uses
   the same.

When a checkpoint fails, fix the C++ side (the numpy is the canonical
reference for this plan) and re-run.

- [ ] **Step 4: Commit (after parity is green)**

```bash
git add tests/test_parity_block0.cpp tests/CMakeLists.txt
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
test(parity): C++ patch_embed + block 0 vs numpy reference

Loads the seeded model GGUF and the numpy-generated baseline bundle, builds
a ggml compute graph for patch_embed + dinov2_block(0), runs it on CPU
backend, captures published intermediate tensors via the trace callback,
and allcloses each against the baseline within per-checkpoint tolerances.

The test is the first place the C++ forward pass touches real numbers; any
layout/scale/init drift surfaces here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(If parity took multiple iterations to land, also commit any fixes to
`src/dinov2.cpp` that you made along the way, with descriptive messages
for each fix.)

---

### Task 12: Final smoke + README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Clean rebuild + full test run**

```bash
rm -rf build
cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=ON
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build --output-on-failure
```

Expected: 10 tests pass (`test_common`, `test_image_io`, `test_postprocess`,
`test_visualize`, `test_cli_smoke`, `test_cli_integration`, `test_model_loader`,
`test_capi`, `test_capi_flat`, `test_parity_block0`).

- [ ] **Step 2: Verify three build configurations**

```bash
# Library-only build
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=OFF -DRFDETR_BUILD_CLI=OFF
cmake --build build -j 2>&1 | tail -5

# Tests on, CLI off
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=OFF
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -3

# Both on
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=ON
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -3
```

All three must configure and build. The two tests=on builds must pass.

- [ ] **Step 3: Update `README.md` Status section**

Replace the "Loader (Plan 2) complete..." paragraph with:

```markdown
## Status

**Forward-pass foundation (Plan 3) complete.** The C++ runtime now loads
real tensor data into a ggml backend buffer, exposes a thread-local
`trace_callback` for named intermediate tensors, implements DINOv2's
`patch_embed` and one transformer block, and verifies parity against a
pure-numpy reference within 1e-3 absolute tolerance. Ten tests pass on
a clean build. `detect` still returns `not_implemented` until Plans 4–5
land the remaining blocks, projector, encoder/decoder, and heads.

The parity workflow doc is at `docs/parity.md`. Plan 6 will swap the numpy
reference for a torch+rfdetr reference using the same baseline-bundle format.
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
docs: mark forward-pass foundation plan (Plan 3) complete in README

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage** against `docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md`:

- §3 Architecture (forward graph) — partially landed: patch_embed + 1 block. Remaining blocks, projector, encoder, decoder, heads are Plans 4–5.
- §6 API — `rfdetr_detect` still NOT_IMPLEMENTED; Plan 5 wires it.
- §8 Tests — `test_parity_block0` is the new entry; `test_inference` (E2E COCO) is Plan 5.
- §9 Parity workflow — fully covered by `docs/parity.md`, `scripts/gen_numpy_baseline.py`, and the trace-callback infrastructure. The torch reference (§9 bullet) is deferred to Plan 6.
- §10 Build — `backend.cpp`, `trace.cpp`, `dinov2.cpp` added to RFDETR_SOURCES.

**Placeholder scan:** No "TBD" or "fill in later" in the plan body. All
caveats are explicit ("if ggml_X is spelled differently in v0.13.0, adapt")
and apply to specific function names the implementer must verify against the
real headers — this is honest, not vague.

**Type consistency:**
- `rfdetr::Model::weights` is `ggml_backend_buffer_t` — declared in Task 4,
  used in Tasks 4 and 5.
- `rfdetr::trace_cb` is `std::function<void(const std::string&, const ggml_tensor*)>` —
  consistent across `trace.hpp`, `trace.cpp`, `dinov2.cpp`, and `test_parity_block0`.
- Tensor names in C++ publish calls match the names the numpy reference writes
  with the `parity.` prefix: `backbone.patch_embed.output`,
  `backbone.block.0.norm1.output`, etc. The test strips the prefix before
  diffing.

**Risks** the executor should be prepared for:

1. **ggml API drift**: `ggml_flash_attn_ext`, `ggml_norm`, `ggml_gelu`,
   `ggml_conv_2d` — confirm signatures in the real headers before assuming the
   plan's names. The plan does say "adapt" at each site.
2. **Numpy/ggml layout misalignment**: The most likely first parity failure.
   Diagnose with the trace callback's `tensor_shape` helper and the failing
   checkpoint's max_abs idx output.
3. **GELU variant**: PyTorch's default `nn.GELU()` is exact GELU. If rfdetr
   actually uses tanh-approx GELU, the parity will be off in the MLP block.
   Both numpy and C++ must agree.
4. **The numpy baseline regen requires `numpy` + `gguf` installed at build time**.
   Plan 3 chooses regen-at-build over checked-in binary fixture; users without
   numpy/gguf can install them (no torch needed, fast install) or skip the
   parity test by disabling `rfdetr_baseline_block0`.

---

## Next plan

After this plan lands:

- **Plan 4** — Remaining DINOv2 blocks (loop over `backbone.depth`), window
  attention switching, CLS token + positional embedding handling, multi-scale
  layer taps. Parity checkpoints for blocks 1–11 and multi-scale outputs.
- **Plan 5** — Projector, encoder, decoder, heads, end-to-end `rfdetr_detect`
  on COCO image, the `compare` CLI subcommand.
- **Plan 6** — `scripts/run_rfdetr_baseline.py` (torch + rfdetr reference)
  replacing or augmenting the numpy reference; tightens tolerances against
  real upstream.
- **Plan 7** — `rfdetr-quantize` binary, Q8_0 support.
- **Plan 8** — small/nano/medium/large variants.
