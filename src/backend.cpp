#include "backend.hpp"
#include "common.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef RFDETR_HAVE_BLAS
#include "ggml-blas.h"
/* OpenBLAS exposes openblas_get_parallel() so we can detect at runtime
 * whether libopenblas was built with pthread or with OpenMP. The two BLAS
 * backends behave very differently when combined with ggml's OpenMP CPU
 * backend (see init_backend_ctx for the rationale). On hosts where the
 * symbol isn't actually OpenBLAS (e.g. Accelerate on macOS, MKL, generic
 * libblas), the weak reference resolves to nullptr and we skip the gating. */
extern "C" {
int openblas_get_parallel(void) __attribute__((weak));
}
#ifndef OPENBLAS_OPENMP
#define OPENBLAS_OPENMP 2
#endif
#endif

#include <cstdlib>
#include <cstring>

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

namespace {

/* Decide whether to wire the BLAS backend through the sched. Returns true
 * when we expect BLAS to help (or when the user explicitly forces it).
 *
 * The heuristic: enable BLAS unless we're on OpenBLAS-pthread mixed with
 * ggml's OpenMP CPU backend. In that combo the two thread pools (OpenBLAS's
 * pthread workers and ggml's OpenMP team) over-subscribe and contend on
 * every mul_mat, making BLAS a net loss for our workload.
 *
 *   RFDETR_BLAS = 1 -> force ON  (skip the auto-disable; useful for testing)
 *   RFDETR_BLAS = 0 -> force OFF (skip BLAS even if the library is OK)
 *   unset           -> auto      (above heuristic) */
bool blas_enabled_for_runtime() {
    if (const char* env = std::getenv("RFDETR_BLAS")) {
        return std::atoi(env) != 0;
    }
#ifdef RFDETR_HAVE_BLAS
    /* If openblas_get_parallel is linked AND reports a non-OpenMP value,
     * we're on the pthread/sequential variant: skip BLAS. */
    if (openblas_get_parallel) {
        const int mode = openblas_get_parallel();
        if (mode != OPENBLAS_OPENMP) {
            rfdetr_logf(RFDETR_LOG_INFO,
                        "BLAS: detected OpenBLAS without OpenMP support (mode=%d); "
                        "skipping BLAS backend to avoid pthread/OpenMP contention. "
                        "Install libopenblas0-openmp (or set RFDETR_BLAS=1 to force).",
                        mode);
            return false;
        }
    }
    return true;
#else
    return false;
#endif
}

#ifdef RFDETR_HAVE_BLAS
/* Decide whether a single mul_mat op is worth dispatching to BLAS for this
 * workload (RF-DETR / DINOv2 / 24-layer ViT-Base).
 *
 * The ggml BLAS backend's default supports_op accepts anything with ne0,
 * ne1, ne10 >= 32. That pulls in many medium-sized attention mul_mats
 * (e.g. Q×Kᵀ with ne10 = head_dim = 64) where the per-call cblas_sgemm
 * thread-pool overhead exceeds the SIMD win.
 *
 * Even worse, every BLAS-routed mul_mat forces a split between the BLAS
 * backend and ggml's CPU backend. Each split's CPU-side compute spawns a
 * fresh `#pragma omp parallel` region (see ggml_graph_compute in
 * ggml-cpu.c), and the cumulative team-setup cost for hundreds of splits
 * dwarfs whatever BLAS saves on each individual GEMM. On a typical ViT-B
 * forward pass we'd get ~140 BLAS-routed mul_mats → ~280 splits → ~3x
 * slowdown vs. a single CPU graph compute.
 *
 * Heuristic: require a very high minimum FLOP count so the cost amortizes.
 * Tunable via RFDETR_BLAS_MIN_FLOPS. The default (2 GFLOPs) is calibrated
 * so that on RF-DETR ViT-B *no* mul_mat passes — the sched is bypassed
 * entirely (see backend_ctx_graph_compute's n_blas_nodes==0 fast-path)
 * and we get the same throughput as RFDETR_BLAS=0. Users on hardware
 * where BLAS dispatch genuinely wins (e.g. Apple Accelerate with much
 * cheaper per-call overhead, or much larger models) can lower the
 * threshold to opt in. */
bool blas_worth_it(const ggml_tensor* op) {
    if (op->op != GGML_OP_MUL_MAT) return false;
    const ggml_tensor* src0 = op->src[0];
    const ggml_tensor* src1 = op->src[1];
    if (!src0 || !src1) return false;
    const int64_t ne0  = op->ne[0];
    const int64_t ne1  = op->ne[1];
    const int64_t ne10 = src1->ne[0];

    static const int64_t min_flops = [](){
        const char* env = std::getenv("RFDETR_BLAS_MIN_FLOPS");
        if (env) {
            long v = std::atol(env);
            if (v > 0) return (int64_t)v;
        }
        return (int64_t)2 * 1000 * 1000 * 1000;
    }();

    const int64_t flops = 2 * ne0 * ne1 * ne10;
    return flops >= min_flops;
}
#endif

}  // namespace

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
     * cpumasks, etc. The scheduler can invoke graph_compute many times per
     * forward pass (one per split), so amortizing this setup across the run
     * is worth a few hundred bytes of long-lived state.
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

#ifdef RFDETR_HAVE_BLAS
    if (blas_enabled_for_runtime()) {
        ctx.blas = ggml_backend_blas_init();
        if (ctx.blas) {
            /* Default BLAS thread count: match the ggml CPU backend so the
             * two pools share the same parallelism budget. OpenBLAS-OpenMP
             * actually shares ggml's pool here — no oversubscription —
             * because both go through libgomp. For OpenBLAS-pthread this
             * code path is skipped (see blas_enabled_for_runtime).
             *
             * Override with RFDETR_BLAS_THREADS=N for tuning. */
            int blas_threads = ctx.n_threads;
            if (const char* env = std::getenv("RFDETR_BLAS_THREADS")) {
                int v = std::atoi(env);
                if (v > 0) blas_threads = v;
            }
            ggml_backend_blas_set_n_threads(ctx.blas, blas_threads);

            /* BLAS first, CPU fallback. The sched routes each op to the
             * first backend whose supports_op returns true. We additionally
             * pin small mul_mats to CPU in backend_ctx_graph_compute via
             * blas_worth_it(). */
            ggml_backend_t backends[2] = { ctx.blas, ctx.cpu };
            ctx.sched = ggml_backend_sched_new(backends, /*bufts*/ nullptr,
                                               /*n_backends*/ 2,
                                               /*graph_size*/ 8192,
                                               /*parallel*/ false,
                                               /*op_offload*/ false);
            if (!ctx.sched) {
                rfdetr_logf(RFDETR_LOG_WARN,
                            "init_backend_ctx: ggml_backend_sched_new failed; "
                            "falling back to CPU only");
                ggml_backend_free(ctx.blas);
                ctx.blas = nullptr;
            } else {
                rfdetr_logf(RFDETR_LOG_INFO,
                            "init_backend_ctx: BLAS backend active "
                            "(cpu_threads=%d, blas_threads=%d)",
                            ctx.n_threads, blas_threads);
            }
        } else {
            rfdetr_logf(RFDETR_LOG_WARN,
                        "init_backend_ctx: ggml_backend_blas_init returned null; "
                        "using CPU only");
        }
    }
#endif

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
    if (ctx.sched) {
        ggml_backend_sched_free(ctx.sched);
        ctx.sched = nullptr;
    }
    if (ctx.blas) {
        ggml_backend_free(ctx.blas);
        ctx.blas = nullptr;
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
    if (ctx.sched) {
        /* The sched re-plans op→backend assignment on every graph; reset
         * clears prior tensor->backend overrides so we can re-apply our
         * blas_worth_it() pinning on the new graph. */
        ggml_backend_sched_reset(ctx.sched);

#ifdef RFDETR_HAVE_BLAS
        /* Pre-pass: pin every node to the CPU backend EXCEPT the big mul_mats
         * we explicitly want BLAS to handle.
         *
         * Why: the BLAS backend's supports_op() returns true for RESHAPE,
         * VIEW, PERMUTE, TRANSPOSE in addition to MUL_MAT/OUT_PROD. When the
         * sched assigns those view ops to BLAS (because BLAS is priority 0),
         * its expansion passes pull adjacent ops along, and the resulting
         * graph alternates BLAS↔CPU dozens of times. Every CPU split spawns
         * a *disposable* threadpool (ggml_graph_compute does an
         * `omp parallel num_threads(n)` per call). For a 1000+ node graph
         * that produces ~280 splits, the OpenMP team setup/teardown
         * dominates (≈3.7× slowdown vs. a single direct CPU call).
         *
         * Fix: explicitly route everything to CPU first, then opt-in only
         * the mul_mats that pass blas_worth_it(). After this, the sched
         * sees at most one BLAS split per chunky GEMM with CPU
         * handling everything else — typically O(num_big_gemms) splits
         * instead of O(num_view_ops). If no mul_mat passes the threshold
         * (BLAS contributes nothing), we still pay only ~1 split (all on
         * CPU) instead of 280. */
        const int n_nodes = ggml_graph_n_nodes(graph);
        int n_blas_nodes = 0;
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor* node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT && blas_worth_it(node)) {
                /* Leave for sched to assign — it will pick BLAS since BLAS
                 * has higher priority and supports the op. */
                ++n_blas_nodes;
            } else {
                ggml_backend_sched_set_tensor_backend(ctx.sched, node, ctx.cpu);
            }
        }

        /* If no node will go to BLAS, skip the sched entirely and run on
         * the CPU backend directly. This avoids the sched's split-planning
         * overhead and the per-split synchronization cost for the common
         * case where BLAS won't help (small mul_mats, OpenBLAS-pthread on
         * mid-sized models, etc.). */
        if (n_blas_nodes == 0) {
            ggml_status st = ggml_backend_graph_compute(ctx.cpu, graph);
            ggml_backend_synchronize(ctx.cpu);
            return (int)st;
        }
#endif

        ggml_status st = ggml_backend_sched_graph_compute(ctx.sched, graph);
        ggml_backend_sched_synchronize(ctx.sched);
        return (int)st;
    }
    /* Fallback: single CPU backend path. Matches the pre-BLAS behavior. */
    ggml_status st = ggml_backend_graph_compute(ctx.cpu, graph);
    ggml_backend_synchronize(ctx.cpu);
    return (int)st;
}

}  // namespace rfdetr
