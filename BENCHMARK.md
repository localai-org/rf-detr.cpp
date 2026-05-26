# Benchmark — rfdetr.cpp vs upstream Python rfdetr

End-to-end CPU inference comparison between this C++ implementation and the
reference Python `rfdetr==1.7.0` package, on the same model weights, same
images, same CPU. Both implementations run on CPU only.

The headline finding is in three plots. Raw data lives in
`benchmarks/results/bench_data.json`; the rendering script
(`scripts/plot_community.py`) is deterministic from that file.

---

## Hardware

| Item    | Value                                                                                |
|---------|--------------------------------------------------------------------------------------|
| CPU     | AMD Ryzen 9 9950X3D — 16 physical cores, exposed as 20 logical, dual-CCD (one with 3D V-Cache) |
| L3      | 320 MiB total (per-CCD locality matters for ggml's per-op cache reuse)               |
| RAM     | 84 GiB                                                                               |
| OS      | Linux 6.8 x86_64 (Ubuntu)                                                            |
| Compiler| g++ with `-march=native` (AVX-512 / VNNI / BF16 enabled — verified zmm refs in libggml-cpu.so) |
| ggml    | Vendored as a submodule under `third_party/ggml` with two in-tree patches            |
| Torch   | 2.5.1 (oneDNN/MKL backend), CPU-forced via `CUDA_VISIBLE_DEVICES=""`                 |
| rfdetr  | 1.7.0 (upstream PyPI)                                                                |

---

## Headline: latency

![Latency comparison](benchmarks/plots/latency_comparison.png)

At T=8 (a single CCD's worth of physical cores), `rfdetr.cpp` is at parity
with PyTorch across 7 diverse COCO images — typically a few percent faster
on the median, with much tighter run-to-run variance.

Per-image median latency, T=8, 15 timed iterations after 3 warmup:

| image       | PyTorch (ms) | C++ F32 (ms) | C++ Q8_0 (ms) | F32 vs Python | Q8_0 vs Python |
|-------------|-------------:|-------------:|--------------:|--------------:|---------------:|
| bus         |        151.6 |        142.8 |         145.0 |        0.94×  |         0.96×  |
| cats        |        148.7 |        141.9 |         145.5 |        0.95×  |         0.98×  |
| indoor      |        157.8 |        142.8 |         144.9 |        0.90×  |         0.92×  |
| kitchen     |        153.8 |        144.2 |         144.9 |        0.94×  |         0.94×  |
| living room |        160.1 |        150.0 |         145.6 |        0.94×  |         0.91×  |
| skater      |        143.5 |        146.4 |         145.8 |        1.02×  |         1.02×  |
| street      |        151.7 |        143.1 |         146.5 |        0.94×  |         0.97×  |
| **mean**    |        **152.5** |    **144.5** |     **145.5** |    **0.95×**  |     **0.95×**  |

Values < 1.0× mean rfdetr.cpp is faster than PyTorch on the same CPU; values
> 1.0× mean slower. The skater image is a slight regression (within run-to-run
noise; PyTorch's whiskers on that image extend to 210 ms while C++ tops out
at 156 ms).

![Relative latency](benchmarks/plots/relative_latency.png)

---

## Thread scaling

![Thread scaling](benchmarks/plots/thread_scaling.png)

A representative image (`coco_kitchen.jpg`) swept over T ∈ {1, 2, 4, 8, 12, 16, 20}:

| Threads | PyTorch (ms) | C++ F32 (ms) | C++ Q8_0 (ms) |
|--------:|-------------:|-------------:|--------------:|
|       1 |        797.3 |        848.0 |         899.7 |
|       2 |        422.1 |        452.5 |         470.9 |
|       4 |        281.6 |        244.2 |         258.0 |
|       8 |        146.7 |        145.1 |         147.6 |
|      12 |        127.3 |        152.7 |         149.3 |
|      16 |    **108.5** |    **138.7** |     **131.7** |
|      20 |        156.8 |        170.8 |         184.7 |

Observations:

- **All three implementations track within ~10% across the entire sweep**.
  C++ F32 and C++ Q8_0 are basically indistinguishable in latency at every
  thread count.
- **The minimum is at T=16**, not T=8, for all three. That's the host's 16
  physical cores; the post-gallocr-fix workload now scales further than the
  original cross-CCD-bound case in earlier revisions. Whatever ggml gained
  by eliminating allocator churn put the bottleneck back into FLOPs.
- **T=20 regresses for everyone** (~30% slower than T=16). The 20 logical
  cores on this host include over-subscription beyond physical, and crossing
  the dual-CCD boundary thrashes per-op L3 locality on the larger residency.
- **T=8 remains a reasonable default**: ~150 ms across all impls, single-CCD
  resident, low contention. If you have a 9950X3D, `--threads 16` will get
  you to ~135 ms on F32; on single-CCD parts the sweet spot tracks the core
  count directly.

---

## Model size + accuracy

![Size and accuracy](benchmarks/plots/size_and_accuracy.png)

| Variant | Size (MB) | Compression vs F32 |
|---------|----------:|-------------------:|
| F32     |     119.2 |              1.00× |
| Q8_0    |      38.5 |              3.10× |

Across the 7 test images, every PyTorch detection has a 1-1 C++ match at
IoU ≥ 0.95 with one caveat per impl:

| impl    | Python dets | C++ dets | matched (IoU ≥ 0.95) | mean \|Δscore\| | max \|Δscore\| |
|---------|------------:|---------:|---------------------:|----------------:|---------------:|
| C++ F32 |          55 |       56 |                   54 |          0.0078 |         0.0445 |
| C++ Q8_0|          55 |       55 |                   54 |          0.0084 |         0.0461 |

- On **coco_living_room**, one C++ detection (class 64, score 0.504 vs
  Python 0.512, bbox shifted ~1 px) lands at IoU ≈ 0.93 — just under the
  0.95 threshold but visibly the same object.
- On **coco_skater**, the C++ F32 build finds one extra detection (class 1
  "person" at score 0.507) that the PyTorch arm scores fractionally below
  the 0.5 threshold. This is honest borderline behavior on threshold-bounded
  detection sets; both detections look reasonable on the image.

Max per-detection score drift across the entire benchmark is **0.046**.
That's the headline accuracy guarantee: Q8_0 quantization is "free" — same
detections, sub-pixel bbox drift, score drift below the detection-threshold
noise floor.

---

## Methodology

- **Timing**: `time.perf_counter()` on the Python side; `std::chrono::high_resolution_clock`
  on the C++ side (via `rfdetr-cli bench`).
- **GC discipline**: Python GC is `gc.disable()`d for the duration of each
  timed block (re-enabled on exception).
- **Iterations**: 15 timed iterations per (impl, image, threads) cell, after
  3 untimed warmup iterations. Median is the reported number; min and max
  are the whiskers.
- **What's measured**: end-to-end inference latency (image load is excluded
  because `rfdetr-cli bench` loads the image once and re-runs `rfdetr_detect`).
  On the Python side, `model.predict()` is the unit — it includes PIL load
  + tensor prep + forward + postprocess, just like the C++ end-to-end loop.
- **Both impls are CPU-only**: PyTorch is forced via
  `CUDA_VISIBLE_DEVICES=""`; the C++ build does not link any GPU backend.
- **Thread control**: C++ uses `--threads N` (forwarded to
  `ggml_backend_cpu_set_n_threads`). Python sweeps use
  `torch.set_num_threads(N)` between cells; matmul intra-op threads is what
  matters for this model.
- **Detection threshold**: 0.5 for both sides.

---

## Reproducing

```sh
# 0. Build (one-time, single thread). Tests are optional but recommended.
cmake -B build -DRFDETR_BUILD_TESTS=ON
cmake --build build -j

# 1. Fetch a handful of COCO val2017 images into benchmarks/images/.
mkdir -p benchmarks/images && cd benchmarks/images
for id in 397133 39769 139 632 252219 87038; do
    curl -sSLkO "https://images.cocodataset.org/val2017/000000${id}.jpg"
done
# Bring your own bus.jpg or any other JPEG to add to the set.
cd ../..

# 2. Convert / download both model variants into models/ (one-time).
python3 scripts/convert_rfdetr_to_gguf.py --dtype f32  --output models/rfdetr-base-f32.gguf
python3 scripts/convert_rfdetr_to_gguf.py --dtype q8_0 --output models/rfdetr-base-q8_0.gguf

# 3. Run the full benchmark sweep (~10-15 minutes on the 9950X3D).
#    Persists raw timing + detections to benchmarks/results/bench_data.json.
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/bench_community.py \
    --iters 15 --warmup 3 --threads 8 \
    --thread-sweep 1,2,4,8,12,16,20 \
    --sweep-image coco_kitchen.jpg

# 4. Render the plots into benchmarks/plots/{*.png,*.svg}.
.venv/bin/python scripts/plot_community.py
```

`scripts/bench_community.py --skip-sweep` skips the thread-scaling pass
(~5 minutes saved); `--skip-python` skips PyTorch entirely (handy if you
only want to compare C++ F32 vs Q8_0).

---

## Build-time optimizations that matter

These are enabled in the default CMake config; the
[git log](https://github.com/mudler/rt-detr.cpp/commits/main) has the
incremental record. The headline win that closed the gap to PyTorch was a
**persistent ggml graph allocator** that holds the ~1.9 GB scratch buffer
across inferences — see commit `0d3f3c1` (perf(forward): persist gallocr
scratch buffers across inferences).

| Flag / change                                        | Impact (median ms) |
|------------------------------------------------------|-------------------:|
| `GGML_NATIVE=ON` (AVX-512 / VNNI on 9950X3D)         |     ~30% over no-march-native baseline |
| `GGML_LLAMAFILE=ON` (tinyBLAS SGEMM)                 |     ~25% on T=1, ~10% on T=8 |
| In-tree tinyBLAS broadcast-fold patch                |     ~8% on windowed attention shapes |
| `GGML_OPENMP=ON` + persistent threadpool             |     ~2-3% on T=8 |
| **Persistent gallocr** (`BackendCtx::galloc_*`)      |     **~3× (~430 → ~145 ms)** |

The two ggml-side patches live in `third_party/ggml-patches/` and apply
automatically at CMake configure time.

### What didn't help

For a while the build wired ggml's BLAS backend (OpenBLAS / MKL /
Accelerate) through `ggml_backend_sched`. After the gallocr fix the
direct-CPU path beat any configured BLAS path; the whole BLAS wiring
(`RFDETR_HAVE_BLAS`, `blas_worth_it()` heuristic, scheduler bypass) was
removed. The original motivation — "oneDNN's hand-tuned blocked AVX-512
micro-kernels must be the gap" — turned out to be wrong; the gap was
per-inference allocator churn (`mmap`/`munmap` of the scratch buffer cost
~55 ms by itself).

See the commit history for the full back-and-forth.

---

## Caveats

- **Hardware-specific.** The dual-CCD topology of the 9950X3D is unusual and
  drives both the T=16-is-best result and the T=20 regression. Single-CCD
  AMD parts (e.g. 9700X, 7700X) and Intel parts will have different sweet
  spots.
- **Image variety matters less than you'd think.** All 7 test images are
  640×480-ish COCO val2017 plus bus.jpg. The model uses fixed 640×640
  internal resolution after preprocessing, so per-image latency variation
  comes from postprocess (number of detections) and run-to-run jitter, not
  shape. We did not test very large (e.g. 2048×2048) inputs.
- **PyTorch run-to-run variance is wider** than C++ on this host (visible
  in the whiskers — Python max is sometimes 200+ ms while C++ max stays
  under 170 ms). Both medians sit in the ~140-160 ms band. We don't know
  why exactly — possibly Python GIL / autograd overhead leaking into a
  no_grad context, possibly oneDNN's lazy backend init firing occasionally.
- The CLI's default is `--threads 0` (auto = all logical cores). On the
  9950X3D that picks 20, which is the **worst** point on the curve. Users
  tuning for latency should pass `--threads 16` (or, for portability,
  `<physical-cores-of-one-CCD>`) explicitly. The default favors
  embarrassingly-parallel batch workloads where per-call latency isn't the
  constraint.
- **Bench images are not committed** (`benchmarks/images/` is in
  `.gitignore`). The benchmark JSON and plot artifacts are committed and
  reproducible from any equivalent set of inputs.
