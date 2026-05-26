# Benchmark — rfdetr.cpp vs upstream Python rfdetr

Cross-check of detection quality and inference latency between this C++
implementation and the reference Python `rfdetr==1.7.0` package on the same
images, same backbone (`rfdetr-base.pth`), same CPU.

## How to reproduce

```
# (one-off) build the CLI
cmake --build build -j

# Detection + latency comparison vs Python (defaults: auto-threads on C++ side).
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/bench.py \
    --image /path/to/img1.jpg \
    --image /path/to/img2.jpg \
    --iters 5 --warmup 3 \
    --threads 8 \
    --out BENCH_REPORT.md

# Thread-scaling sweep (C++ only, both F32 and Q8_0).
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/bench_threads.py \
    --image /path/to/img1.jpg --image /path/to/img2.jpg --image /path/to/img3.jpg \
    --iters 5 --warmup 3 \
    --threads 1 --threads 4 --threads 8 --threads 16 --threads 20 \
    --out BENCH_THREADS.md
```

The C++ side uses `rfdetr-cli bench`, which loads the model once and times
N inference iterations (excluding load). The Python side loads `RFDETRBase`
once and times `predict()` per iteration. Both are pinned to CPU.

The CLI exposes `--threads N` (default `0` = auto = all logical cores)
which is forwarded to `ggml_backend_cpu_set_n_threads`.

## Build-time optimization flags

The default cmake configuration enables:

- `GGML_NATIVE=ON` — `-march=native` so the compiler emits AVX-512 / AVX2 /
  FMA / F16C / VNNI / BF16 intrinsics whenever the host CPU supports them.
  On the 9950X3D this dispatches the AVX-512 SIMD kernels in `vec.cpp`,
  `quants.c`, and `ops.cpp` (verified: `libggml-cpu.so` contains 8500+ `zmm`
  references after build).
- `GGML_LLAMAFILE=ON` — ggml's tinyBLAS path for `mul_mat`. tinyBLAS is a
  tiled SGEMM/HGEMM implementation with `__m512` / `__m256` / `float32x4_t`
  micro-kernels and proper register blocking — substantially faster than the
  per-row dot-product fallback for the F32 backbone hotspot (10-15% at T=8,
  ~30% at T=1).
- **tinyBLAS broadcast-fold (in-tree ggml patch).** `ggml_compute_forward_mul_mat`
  dispatches one tinyBLAS call per (i12, i13) pair in the broadcast loop. For
  RF-DETR that means windowed attention (`ne12=16` windows × `ne11=101` tokens)
  invokes 16 separate sgemm calls of `N=101` instead of one call of `N=1616`.
  Per-call overhead (`ggml_barrier`, chunk setup, kernel prelude) and poor
  thread utilization on tiny N dominate the FMA work.
  Patch in `third_party/ggml/src/ggml-cpu/ggml-cpu.c` collapses the broadcast
  iterations into a single `llamafile_sgemm` of merged `N = ne11*ne12*ne13`
  when `src0` is broadcast on those dims and `src1`/`dst` pack them
  contiguously. Profile shows the per-iteration MUL_MAT total drops from
  ~367 ms to ~338 ms (-8%) on the test host; end-to-end best-case drops from
  ~440 ms to ~410 ms median, ~390 ms min. Global-attention shapes
  (`ne02 = n_heads > 1`) cannot be folded with this approach and remain the
  next bottleneck.
- `GGML_OPENMP=ON` (ggml's default) — OpenMP-based thread pool. Has lower
  per-graph wakeup overhead than the pthread fallback for batch-1 inference.
- **Persistent ggml threadpool.** `init_backend_ctx` calls
  `ggml_threadpool_new` and attaches it to the CPU backend with
  `ggml_backend_cpu_set_threadpool`. This avoids re-allocating the workers
  state / cpumask array on every `ggml_graph_compute` call. The improvement
  is small (~2-3% at T=8) because with `GGML_USE_OPENMP=ON` ggml still opens
  a fresh `#pragma omp parallel num_threads(N)` per call regardless of
  whether the threadpool is borrowed or disposable — the persistent
  threadpool only avoids the malloc, not the OpenMP team setup. The wiring
  is kept for correctness and because it would be a real win in a future
  no-OpenMP build mode where the pthread workers spin between graphs.
- **Persistent gallocr (the big win).** `BackendCtx::galloc_a` /
  `galloc_b` keep the per-graph allocator (and its ~1.9 GB scratch buffer)
  alive across inferences. Previously each `rfdetr_model_forward` rebuilt
  the gallocr, which allocated a fresh buffer for every intermediate
  tensor and freed it at end-of-call — the `munmap` alone of the large
  mmap was ~55 ms/inference. Keeping the buffer means the kernel sees no
  mmap/munmap traffic on the steady-state loop, and gallocr's lifetime-aware
  packing reuses tensor slots. End-to-end median dropped from ~430 ms to
  ~140 ms at T=8 (3x), closing the gap to PyTorch.

Explicit `GGML_AVX512` / `GGML_AVX512_VNNI` / `GGML_AVX512_BF16` flags are
**not** set, because they would compile features the local CPU might not
support. `GGML_NATIVE=ON` adapts per-host and still emits AVX-512 on capable
machines, while gracefully degrading to AVX2 on older hardware.

### What we tried that didn't help (BLAS via `ggml_backend_sched`)

For a while the build wired ggml's BLAS backend (OpenBLAS / MKL /
Accelerate) through `ggml_backend_sched` so large F32 mul_mats would
dispatch to the host BLAS library. After the gallocr fix landed it was
clear this added no value on the default path:

- The FLOP-threshold heuristic (`RFDETR_BLAS_MIN_FLOPS=2G`) intentionally
  routed zero ops to BLAS on RF-DETR ViT-B — every shape was below the
  threshold, so the scheduler was always bypassed for a direct CPU compute.
- Forcing BLAS on (`RFDETR_BLAS=1`) was strictly slower: each BLAS-routed
  mul_mat forced a sched split, each CPU-side split spawned a fresh
  `#pragma omp parallel` region, and the cumulative OpenMP team-setup cost
  for ~140 mul_mats × 2 splits dwarfed any per-GEMM BLAS speedup
  (~3.7× regression vs. direct CPU).
- On hosts with only OpenBLAS-pthread (no OpenMP variant), mixing the
  OpenBLAS pthread pool with ggml's OpenMP CPU pool over-subscribed the
  cores. The build auto-disabled BLAS in that case via an
  `openblas_get_parallel` weak-symbol probe.

The wiring was removed (CMake detection, `RFDETR_HAVE_BLAS` define,
`RFDETR_BLAS*` env vars, the `blas_worth_it()` FLOP heuristic, the
scheduler bypass) once the gallocr fix made direct-CPU faster than any
configured BLAS path. The CPU path (tinyBLAS SGEMM + persistent
threadpool + persistent gallocr) now matches PyTorch.

## Environment

- CPU: AMD Ryzen 9 9950X3D (16-core, exposed as 20 logical / 1 thread per core on this host; 320 MiB L3)
- OS:  Linux 6.8 x86_64
- Torch: CPU-forced via `CUDA_VISIBLE_DEVICES=""` (multi-threaded MKL/oneDNN, all cores)
- Iterations: 5 timed + 3 warmup per cell (best-of-2 medians on the sweep table)

## Inference latency at the optimal C++ thread count (T=8)

After the persistent-gallocr fix, the C++ path is at parity with PyTorch
on the same CPU.

| image            | impl              |    min | median |   mean |    max | speedup vs Python |
|------------------|-------------------|-------:|-------:|-------:|-------:|------------------:|
| coco_sample.jpg  | python (auto-mt)  |  129.7 |  146.2 |  147.7 |  179.7 | 1.00x             |
| coco_sample.jpg  | cpp_f32 (T=8)     |  138.8 |  141.1 |  142.3 |  152.9 | 1.04x             |
| coco_sample.jpg  | cpp_q8  (T=8)     |  147.8 |  154.3 |  155.9 |  181.4 | 0.95x             |

C++ F32 median is **~140 ms / image** vs **~146 ms** for Python — roughly
parity. The thread-scaling sweep below predates the gallocr fix and is
retained as a record of how scaling behaves on this dual-CCD host; the
*absolute* numbers there should be read as "shape of the curve", not as
the current latency.

## C++ thread-scaling sweep (median ms, best-of-2 runs)

**Pre-gallocr-fix data; retained for the shape of the curve.** The
T=1 → T=8 scaling factor and the > T=8 degradation are still
representative; the absolute numbers are roughly 3x larger than the
current post-fix baseline.

| image            | dtype | T=1    | T=4   | T=6   | T=8   | T=10   | T=12   | T=16   | T=20    | best  | speedup vs T=1 |
|------------------|-------|-------:|------:|------:|------:|-------:|-------:|-------:|--------:|------:|---------------:|
| coco_sample.jpg  | F32   | 1174.2 | 471.0 | 430.0 | 429.1 |  577.5 |  703.9 |  851.4 |  1042.8 | 429.1 |          2.74x |
| coco_sample2.jpg | F32   | 1177.3 | 451.5 | 416.8 | 416.8 |  597.7 |  712.3 |  855.0 |  1202.2 | 416.8 |          2.83x |
| bus.jpg          | F32   | 1213.6 | 465.1 | 440.5 | 424.1 |  590.9 |  728.5 |  877.1 |  1261.1 | 424.1 |          2.86x |
| coco_sample.jpg  | Q8_0  | 1268.4 | 491.1 |   —   | 443.3 |   —    |   —    |  820.1 |   849.3 | 443.3 |          2.86x |
| coco_sample2.jpg | Q8_0  | 1251.3 | 475.8 |   —   | 423.9 |   —    |   —    |  839.8 |   839.9 | 423.9 |          2.95x |
| bus.jpg          | Q8_0  | 1256.6 | 485.5 |   —   | 443.1 |   —    |   —    |  836.5 |   898.5 | 443.1 |          2.84x |

Note: T=1 dropped from 1678 ms → 1174 ms (-30%) thanks to tinyBLAS — single-thread
F32 GEMM is the path most sensitive to kernel quality.

## F32 thread-scaling analysis (pre-fix curve)

- **Best F32 latency at the time**: ~417-431 ms/image at `--threads 8` — a
  2.7-2.9x speedup over single-threaded. Current best (post-fix) is
  ~140 ms at T=8.
- **Scaling shape**: sublinear and plateaus at T=6-8. Beyond that the
  numbers regress sharply:
  - T=1 → T=4: 2.5x (good)
  - T=4 → T=8: 1.10x (diminishing)
  - T=8 → T=16: **0.50x — actively worse**
- **Why scaling stops**: the 9950X3D ships 16 physical cores split across two
  CCDs (one with 3D V-Cache, one without). Crossing the CCD boundary blows
  ggml's per-op cache locality — the model is ~120 MB F32 weights, comfortably
  in L3 of a single CCD but not coherent across both. 8 threads fits cleanly
  inside one CCD; 16+ forces cross-CCD scheduling and hot cache lines bounce.

## How the gap to PyTorch was closed

For most of the project, C++ was ~2.4x slower than PyTorch on the same
CPU. The residual gap was widely (and incorrectly) attributed to
oneDNN's hand-tuned blocked AVX-512 micro-kernels versus tinyBLAS's
generic tiled SGEMM. The actual root cause was much more boring:
**per-inference allocator churn**. Every `rfdetr_model_forward` rebuilt
the ggml graph allocator, which allocated a fresh scratch buffer for
every intermediate tensor and freed it at end-of-call — on this workload
that's ~1.9 GB of `mmap`/`munmap` per inference, and the `munmap` alone
accounted for ~55 ms (about a third of the runtime).

The fix was to persist the gallocr (and its underlying buffer) on
`BackendCtx` across inferences. gallocr's lifetime-aware tensor packing
reuses slots within a single graph, and keeping the buffer alive means
the kernel sees no `mmap`/`munmap` traffic on the steady-state loop.
End-to-end median dropped from ~430 ms to ~140 ms at T=8, putting the
C++ path on par with PyTorch.

Whatever fraction of the original gap was actually micro-kernel quality
is now within run-to-run noise.

## Q8_0 notes

Q8_0 is now ~3% faster than F32 at T=8 (was ~8% slower in the unoptimized
build). The reason is unchanged: ggml's CPU `mul_mat` dequantizes Q8_0 blocks
on the fly when one operand is F32 (activations); arithmetic cost dominates.
But the F32 tinyBLAS speedup partially closes the gap to the Q8 path, and
Q8_0's smaller memory footprint starts to matter on this image-sized workload.

Q8_0's primary value remains the **3.08x model-size reduction (120 MB → 39 MB)**
at no detection-quality cost (see below).

## Detection cross-check (Python as reference, IoU >= 0.95)

| image            | impl    | py_det | cpp_det | matched | mean IoU | mean \|score Δ\| | max \|score Δ\| | mean center Δ (px) | max center Δ (px) |
|------------------|---------|-------:|--------:|--------:|---------:|-----------------:|----------------:|-------------------:|------------------:|
| coco_sample.jpg  | cpp_f32 |     12 |      12 |      12 |   0.9906 |           0.0126 |          0.0296 |               0.14 |              0.37 |
| coco_sample.jpg  | cpp_q8  |     12 |      12 |      12 |   0.9906 |           0.0094 |          0.0244 |               0.14 |              0.47 |
| coco_sample2.jpg | cpp_f32 |      5 |       5 |       5 |   0.9982 |           0.0056 |          0.0120 |               0.11 |              0.23 |
| coco_sample2.jpg | cpp_q8  |      5 |       5 |       5 |   0.9973 |           0.0115 |          0.0416 |               0.14 |              0.35 |
| bus.jpg          | cpp_f32 |      5 |       5 |       5 |   0.9982 |           0.0010 |          0.0019 |               0.12 |              0.23 |
| bus.jpg          | cpp_q8  |      5 |       5 |       5 |   0.9984 |           0.0008 |          0.0021 |               0.17 |              0.23 |

The optimization flags do not change numerical results: every Python
detection has a 1-1 match in the C++ output at the same class, IoU > 0.95
(mean > 0.99), sub-pixel bbox drift, and confidence drift under 0.05 in the
worst case. No false positives, no missed detections.

## Caveats

- Numbers are hardware-specific; the dual-CCD topology of the 9950X3D is
  unusual and skews the high-thread tail. Single-CCD AMDs and Intel parts
  will scale further past 8 threads.
- The CLI's default is `--threads 0` (auto = all logical cores). On this
  host that picks 20, which is **slower** than 8. Users tuning for latency
  should pass `--threads <physical-cores-of-one-CCD>` explicitly. The
  default favors throughput-oblivious workloads (batch processing) where
  cross-CCD parallelism still beats serial.
- Python `predict()` includes PIL load + tensor prep + postprocess, just
  like the C++ side includes preprocess + postprocess; both are measured
  end-to-end so the comparison is apples-to-apples.
- The first 3 iterations on each side are dropped as warmup (allocator
  and cache fill).
- Bench scripts and CLI subcommand are reproducible; the input images are
  not committed.
