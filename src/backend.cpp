#include "backend.hpp"
#include "common.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
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

BackendCtx init_backend_ctx(int n_threads, rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) { if (out_status) *out_status = s; };

    BackendCtx ctx{};
    ctx.n_threads = (n_threads > 0) ? n_threads : 1;

    ctx.cpu = init_backend(n_threads, out_status);
    if (!ctx.cpu) {
        return ctx;  // status already set
    }

    /* Attach a persistent threadpool to the CPU backend. ggml's
     * ggml_graph_compute() otherwise builds (and tears down) a disposable
     * threadpool on every call — allocating workers state, computing
     * cpumasks, etc. Amortizing this setup across the run is worth a few
     * hundred bytes of long-lived state.
     *
     * With GGML_USE_OPENMP=ON the OpenMP team itself is still created per
     * call (ggml uses `#pragma omp parallel num_threads(N)` inside
     * ggml_graph_compute regardless), but everything *outside* the OMP
     * region — and notably the disposable-threadpool malloc — is now paid
     * only once. */
    {
        ggml_threadpool_params tpp = ggml_threadpool_params_default(ctx.n_threads);
        ctx.threadpool = ggml_threadpool_new(&tpp);
        if (ctx.threadpool) {
            ggml_backend_cpu_set_threadpool(ctx.cpu, ctx.threadpool);
        } else {
            rfdetr_logf(RFDETR_LOG_WARN,
                        "init_backend_ctx: ggml_threadpool_new failed; "
                        "falling back to per-call disposable threadpool");
        }
    }

    set(RFDETR_OK);
    return ctx;
}

void free_backend_ctx(BackendCtx& ctx) {
    /* Free the gallocrs BEFORE the backends. The gallocr owns the compute
     * scratch buffers (allocated via the backend's buffer_type); freeing
     * it after the backend would still be safe (the buffer keeps a ref to
     * its buffer_type), but doing it first matches the construction order
     * (gallocrs created lazily during forward, on top of the backend). */
    if (ctx.galloc_a) {
        ggml_gallocr_free(ctx.galloc_a);
        ctx.galloc_a = nullptr;
    }
    if (ctx.galloc_b) {
        ggml_gallocr_free(ctx.galloc_b);
        ctx.galloc_b = nullptr;
    }
    if (ctx.cpu) {
        ggml_backend_free(ctx.cpu);
        ctx.cpu = nullptr;
    }
    /* Free the threadpool AFTER the CPU backend that referenced it. The CPU
     * backend's destructor doesn't touch the threadpool (it's borrowed, not
     * owned), but pausing/freeing in the right order keeps the worker
     * threads from being torn down underneath the backend. */
    if (ctx.threadpool) {
        ggml_threadpool_free(ctx.threadpool);
        ctx.threadpool = nullptr;
    }
}

int backend_ctx_graph_compute(BackendCtx& ctx, ::ggml_cgraph* graph) {
    ggml_status st = ggml_backend_graph_compute(ctx.cpu, graph);
    ggml_backend_synchronize(ctx.cpu);
    return (int)st;
}

}  // namespace rfdetr
