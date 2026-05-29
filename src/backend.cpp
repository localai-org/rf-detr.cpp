#include "backend.hpp"
#include "common.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#if defined(RFDETR_USE_CUDA)
#include "ggml-cuda.h"
#endif
#if defined(RFDETR_USE_METAL)
#include "ggml-metal.h"
#endif
#if defined(RFDETR_USE_VULKAN)
#include "ggml-vulkan.h"
#endif

#include <vector>

namespace rfdetr {

/* Try to create a GPU backend if one was compiled in and a device exists.
 * Returns nullptr (not an error) when no GPU backend is built or no device
 * is present — the caller falls back to CPU-only. */
static ggml_backend_t try_init_gpu_backend() {
#if defined(RFDETR_USE_CUDA)
    int n = ggml_backend_cuda_get_device_count();
    if (n > 0) {
        ggml_backend_t b = ggml_backend_cuda_init(0);  // device 0
        if (b) {
            rfdetr_logf(RFDETR_LOG_INFO, "GPU backend: CUDA device 0 (%d available)", n);
            return b;
        }
        rfdetr_logf(RFDETR_LOG_WARN, "ggml_backend_cuda_init(0) failed; using CPU");
    }
    return nullptr;
#elif defined(RFDETR_USE_METAL)
    ggml_backend_t b = ggml_backend_metal_init();
    if (b) {
        rfdetr_logf(RFDETR_LOG_INFO, "GPU backend: Metal");
        return b;
    }
    rfdetr_logf(RFDETR_LOG_WARN, "ggml_backend_metal_init failed; using CPU");
    return nullptr;
#elif defined(RFDETR_USE_VULKAN)
    int n = ggml_backend_vk_get_device_count();
    if (n > 0) {
        ggml_backend_t b = ggml_backend_vk_init(0);
        if (b) {
            rfdetr_logf(RFDETR_LOG_INFO, "GPU backend: Vulkan device 0 (%d available)", n);
            return b;
        }
        rfdetr_logf(RFDETR_LOG_WARN, "ggml_backend_vk_init(0) failed; using CPU");
    }
    return nullptr;
#else
    return nullptr;  // CPU-only build
#endif
}

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

    /* Try a GPU backend. If present, build a scheduler spanning [gpu, cpu]
     * so ops the GPU can't run (the deformable ggml_custom_4d) fall back to
     * CPU automatically. */
    ctx.gpu = try_init_gpu_backend();
    if (ctx.gpu) {
        std::vector<ggml_backend_t> backends = { ctx.gpu, ctx.cpu };
        std::vector<ggml_backend_buffer_type_t> bufts = {
            ggml_backend_get_default_buffer_type(ctx.gpu),
            ggml_backend_get_default_buffer_type(ctx.cpu),
        };
        ctx.sched = ggml_backend_sched_new(
            backends.data(), bufts.data(), (int)backends.size(),
            /*graph_size*/ 16384, /*parallel*/ false, /*op_offload*/ true);
        if (!ctx.sched) {
            rfdetr_logf(RFDETR_LOG_WARN,
                        "ggml_backend_sched_new failed; falling back to CPU-only");
            ggml_backend_free(ctx.gpu);
            ctx.gpu = nullptr;
        }
    }

    set(RFDETR_OK);
    return ctx;
}

ggml_backend_buffer_type_t backend_ctx_weight_buft(const BackendCtx& ctx) {
    if (ctx.gpu) {
        return ggml_backend_get_default_buffer_type(ctx.gpu);
    }
    return ggml_backend_get_default_buffer_type(ctx.cpu);
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
    if (ctx.sched) {
        ggml_backend_sched_free(ctx.sched);
        ctx.sched = nullptr;
    }
    if (ctx.gpu) {
        ggml_backend_free(ctx.gpu);
        ctx.gpu = nullptr;
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

bool backend_ctx_graph_alloc(BackendCtx& ctx, ::ggml_cgraph* graph, int which_graph) {
    if (ctx.sched) {
        ggml_backend_sched_reset(ctx.sched);
        if (!ggml_backend_sched_alloc_graph(ctx.sched, graph)) {
            rfdetr_logf(RFDETR_LOG_ERROR, "backend_ctx_graph_alloc: sched alloc failed");
            return false;
        }
        return true;
    }
    /* CPU path: persistent gallocr per graph. */
    ggml_gallocr_t* slot = (which_graph == 0) ? &ctx.galloc_a : &ctx.galloc_b;
    if (!*slot) {
        *slot = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx.cpu));
        if (!*slot) {
            rfdetr_logf(RFDETR_LOG_ERROR, "backend_ctx_graph_alloc: gallocr_new failed");
            return false;
        }
    }
    if (!ggml_gallocr_alloc_graph(*slot, graph)) {
        rfdetr_logf(RFDETR_LOG_ERROR, "backend_ctx_graph_alloc: gallocr_alloc_graph failed");
        return false;
    }
    return true;
}

int backend_ctx_graph_compute(BackendCtx& ctx, ::ggml_cgraph* graph, int which_graph) {
    (void)which_graph;
    if (ctx.sched) {
        ggml_status st = ggml_backend_sched_graph_compute(ctx.sched, graph);
        ggml_backend_sched_synchronize(ctx.sched);
        return (int)st;
    }
    ggml_status st = ggml_backend_graph_compute(ctx.cpu, graph);
    ggml_backend_synchronize(ctx.cpu);
    return (int)st;
}

}  // namespace rfdetr
