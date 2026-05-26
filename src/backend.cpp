#include "backend.hpp"
#include "common.hpp"

#include "ggml.h"
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
 * Heuristic: require a minimum FLOP count (FLOPs ≈ 2 * ne0 * ne1 * ne10).
 * Tunable via RFDETR_BLAS_MIN_FLOPS (default 32M FLOPs ≈ a 128×128×128
 * GEMM). That admits the big backbone Q/K/V (250M), FFN (1B), and the
 * projector mul_mats while excluding small attention mul_mats and the
 * decoder's per-query projections. */
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
        return (int64_t)32 * 1000 * 1000;
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
}

int backend_ctx_graph_compute(BackendCtx& ctx, ::ggml_cgraph* graph) {
    if (ctx.sched) {
        /* The sched re-plans op→backend assignment on every graph; reset
         * clears prior tensor->backend overrides so we can re-apply our
         * blas_worth_it() pinning on the new graph. */
        ggml_backend_sched_reset(ctx.sched);

#ifdef RFDETR_HAVE_BLAS
        /* Pre-pass: for every mul_mat that doesn't pass our FLOP-based
         * profitability check, pin it to the CPU backend. The sched will
         * still route the big mul_mats (Q/K/V, FFN, projector) to BLAS but
         * keep small attention ops on CPU where ggml's in-place dispatch
         * is cheaper. */
        const int n_nodes = ggml_graph_n_nodes(graph);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor* node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT && !blas_worth_it(node)) {
                ggml_backend_sched_set_tensor_backend(ctx.sched, node, ctx.cpu);
            }
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
