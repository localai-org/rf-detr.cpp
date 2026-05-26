#ifndef RFDETR_BACKEND_HPP
#define RFDETR_BACKEND_HPP

#include "rfdetr.h"

#include <cstdint>

struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
struct ggml_backend_sched;
typedef struct ggml_backend_sched* ggml_backend_sched_t;
struct ggml_cgraph;

namespace rfdetr {

/* Initialize a CPU backend. Used by tests and as the "primary" / weight-buffer
 * backend everywhere. Plan 3: CPU only. Later plans honor build flags
 * RFDETR_GGML_CUDA / METAL / VULKAN / HIPBLAS. */
ggml_backend_t init_backend(int n_threads, rfdetr_status* out_status);

/* Release a backend created by init_backend. */
void free_backend(ggml_backend_t b);

/* True if the backend is CPU. Used to choose direct-memcpy vs
 * ggml_backend_tensor_get in tests. */
bool is_cpu(ggml_backend_t b);

/* Compute-side backend bundle, owned by rfdetr_context. Carries the CPU
 * backend (used for tensor I/O and weight realization — host buffers) plus
 * an optional BLAS backend wired through ggml_backend_sched so large F32
 * mul_mat ops dispatch to OpenBLAS / MKL / Accelerate.
 *
 * When BLAS is not compiled in (or sched_new failed), `blas` and `sched` are
 * both nullptr and graph compute falls back to the single CPU backend. */
struct BackendCtx {
    ggml_backend_t       cpu   = nullptr;
    ggml_backend_t       blas  = nullptr;
    ggml_backend_sched_t sched = nullptr;
    int                  n_threads = 1;
};

/* Initialize the compute backend bundle. Always creates a CPU backend; if
 * RFDETR_HAVE_BLAS is defined, also creates a BLAS backend and wraps both
 * in a ggml_backend_sched_t (BLAS first → CPU fallback).
 *
 * On failure returns an empty BackendCtx (all members nullptr) and writes
 * the error to *out_status. */
BackendCtx init_backend_ctx(int n_threads, rfdetr_status* out_status);

/* Release a BackendCtx. Safe to call on a zero-initialized struct. */
void free_backend_ctx(BackendCtx& ctx);

/* Run a graph on the bundle. Uses sched (BLAS + CPU) when available; falls
 * back to single-backend graph_compute on the CPU backend otherwise. */
int /* ggml_status */ backend_ctx_graph_compute(BackendCtx& ctx, ::ggml_cgraph* graph);

}  // namespace rfdetr

#endif
