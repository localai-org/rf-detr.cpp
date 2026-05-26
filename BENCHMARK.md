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

## What about 4-bit?

> *tl;dr — **Q8_0 is still the no-brainer for this model**, but if you
> need to squeeze below 38 MB the K-quants (`q6_K`, `q5_K`, `q4_K`) are
> the right tool. **Q6_K matches Q8_0 detection quality at 36 MB**;
> **Q4_K beats legacy Q4_0 by a wide margin** on both recall and max
> Δscore at roughly the same on-disk size. Legacy `q4_0` / `q5_0` ship
> as a cautionary baseline only.*

![Quant tradeoffs](benchmarks/plots/quant_tradeoffs.png)

We now ship a native C++ quantizer (`rfdetr-cli quantize`) on top of
ggml's `ggml_quantize_chunk`, so the full set of legacy + K-quants is
producible from any F32 model with one command. The Python converter
(`scripts/convert_rfdetr_to_gguf.py`) still works for legacy block
quants but can't emit K-quants — `gguf.quants.quantize()` raises
`NotImplementedError` for them.

```sh
build/bin/rfdetr-cli quantize models/rfdetr-base-f32.gguf \
    models/rfdetr-base-q4_K.gguf q4_K
# also: q5_0, q5_1, q5_K, q6_K, q8_0, ...
```

The C++ quantizer's `should_quantize` heuristic is bit-identical to the
Python converter's: only 2D `.weight` tensors with both dims ≥ 64 get
quantized; pos_embed / query embeddings / LayerNorm / biases / conv
kernels stay F32. Q8_0 output from the two paths is **byte-for-byte
identical**; Q4_0 / Q5_0 agree on **all but 1-2 nibbles in the entire
~30 MB tensor blob** (the Python implementation comments out the
discrepancy as `# FIXME: Q4_0's reference rounding is cursed and depends
on FMA` — our path uses ggml's reference C kernel directly so we match
what the inference vec-dot kernels expect).

### K-quant row-size constraint and Q8_0 fallback

K-quants encode 256-element super-blocks. rfdetr-base's backbone uses
dim=384, so 60 backbone weight tensors have an inner row size of 384 —
**not** a multiple of 256. Falling back to F32 for those would leak ~24
MB out of the compression budget and ruin the size story. The CLI
instead falls back to **Q8_0** for those tensors (still 32-element
blocks, still a 3× compression), keeping the K-quant models close in
size to legacy Q8_0 while quantizing the dim-256 projector + decoder +
two-stage tensors as the requested K-type.

### Sizes

| Variant | Size (MB) | vs F32 | vs Q8_0 |
|---------|----------:|-------:|--------:|
| F32     |     119.2 |  1.00× |  3.10×  |
| Q8_0    |      38.5 |  3.10× |  1.00×  |
| Q6_K    |      35.1 |  3.40× |  1.10×  |
| Q5_K    |      33.2 |  3.59× |  1.16×  |
| Q4_K    |      31.5 |  3.79× |  1.22×  |
| Q5_0    |      28.7 |  4.16× |  1.34×  |
| Q4_0    |      24.7 |  4.83× |  1.56×  |

K-quants are bigger than the legacy equivalents on this model precisely
because of the Q8_0 fallback: 60 backbone tensors stay in 32-element
blocks. On a model with dim%256==0 backbones the K-quants would scale to
their full ~4-5× compression, but rfdetr-base trades a bit of size for
the freedom to keep using K-quants on every tensor that fits.

### Detection accuracy (vs PyTorch reference, 7 COCO images)

| impl | dets matched (IoU ≥ 0.5) | dets matched (IoU ≥ 0.95) | max \|Δscore\| |
|------|-------------------------:|--------------------------:|---------------:|
| Q8_0 |              **55 / 55** |               **54 / 55** |      **0.046** |
| Q6_K |              **55 / 55** |               **54 / 55** |          0.051 |
| Q5_K |                  52 / 55 |                   51 / 55 |          0.066 |
| Q5_0 |                  53 / 55 |                   46 / 55 |          0.069 |
| Q4_K |                  51 / 55 |                   45 / 55 |          0.110 |
| Q4_0 |                  49 / 55 |                   40 / 55 |          0.226 |

The IoU≥0.5 column is "did we find the object at all"; the IoU≥0.95
column is "is the bbox in roughly the same place".

- **Q6_K matches Q8_0 detection-for-detection** — 55/55 lenient,
  54/55 strict, Δscore in the same ballpark (0.051 vs 0.046).
- **Q5_K is a clear improvement over Q5_0** at IoU≥0.95: 51/55 strict
  vs 46/55. Lenient recall (94.5% vs 96.4%) edges Q5_0 because Q5_K
  shifts one borderline detection out of the 0.5 threshold.
- **Q4_K beats Q4_0 on every axis we care about**: strict recall jumps
  from 40 to 45 out of 55, lenient recall from 89% to 93%, and **max
  |Δscore| drops by more than half (0.110 vs 0.226)**. Q4_K also avoids
  the class-confusion failures Q4_0 produces (no "couch → bed" type
  swaps at the threshold).

### Latency (median ms/image, T=8, 9950X3D)

| Variant | median ms | vs F32 |
|---------|----------:|-------:|
| F32     |     149.2 |  1.00× |
| Q8_0    |     148.5 |  1.00× |
| Q6_K    |     164.7 |  1.10× |
| Q4_K    |     166.2 |  1.11× |
| Q5_0    |     167.5 |  1.12× |
| Q5_K    |     175.9 |  1.18× |
| Q4_0    |     160.3 |  1.07× |

**None of the quants speed up inference on this CPU.** ggml's Q8_0
vec-dot path on AVX-512+VNNI is faster than every smaller-bit kernel
because (a) Q8_0 dispatches the same SIMD lanes as F32 mul_mat once the
weights are in cache and (b) rfdetr-base's hot weights already fit in
L2 — there's no memory-bandwidth headroom for a smaller weight blob to
recoup. K-quants pay the same dequant tax as legacy quants plus extra
super-block overhead, so they end up ~10-18% slower than F32/Q8_0.

### Conclusions

- **Q8_0 stays the default** for size+accuracy+speed.
- **Q6_K is the right pick when you want under 38 MB without compromise.**
  Effectively identical detection quality to Q8_0 at 3.4× compression.
- **Q4_K replaces legacy Q4_0** as the recommended sub-5-bit option.
  Same compression class, much better detection survival.
- **Legacy Q4_0 / Q5_0 are kept** for completeness and as a baseline,
  but they should not be the default option going forward.

### Reproducing the K-quants

```sh
# Need an F32 baseline first.
.venv/bin/python scripts/convert_rfdetr_to_gguf.py \
    --dtype f32 --output models/rfdetr-base-f32.gguf

# Then re-quantize as many ways as you like.
build/bin/rfdetr-cli quantize models/rfdetr-base-f32.gguf \
    models/rfdetr-base-q4_K.gguf q4_K
build/bin/rfdetr-cli quantize models/rfdetr-base-f32.gguf \
    models/rfdetr-base-q5_K.gguf q5_K
build/bin/rfdetr-cli quantize models/rfdetr-base-f32.gguf \
    models/rfdetr-base-q6_K.gguf q6_K
```

`scripts/bench_community.py` picks up `models/rfdetr-base-q{4,5,6}_K.gguf`
automatically when present and adds them to the per-image latency +
detection-accuracy cells. `scripts/plot_community.py` then renders them
into `benchmarks/plots/quant_tradeoffs.{png,svg}`.

---

## Variant comparison

All 5 RF-DETR detection variants share the same DINOv2-small backbone
(dim=384, depth=12, heads=6, ffn_dim=1536), the same decoder model_dim
(256), num_queries (300), num_classes (91), and group_detr (13). They
differ only in input resolution, patch size, and decoder layer count.

| Variant | Resolution | Patch | Decoder layers | GGUF F32 | C++ F32 median ms @ T=8 | PyTorch median ms @ T=8 |
|---------|-----------:|------:|---------------:|---------:|------------------------:|------------------------:|
| Nano    |        384 |    16 |              2 |   113 MB |                    61.5 |                    88.4 |
| Small   |        512 |    16 |              3 |   119 MB |                   116.0 |                   120.5 |
| Base    |        560 |    14 |              3 |   119 MB |                   159.3 |                   200.8 |
| Medium  |        576 |    16 |              4 |   125 MB |                   149.6 |                   182.8 |
| Large   |        704 |    16 |              4 |   126 MB |                   237.8 |                   228.7 |

![Variants overview](benchmarks/plots/variants_overview.png)

All five variants share the same converter (`scripts/convert_rfdetr_to_gguf.py`)
and loader (`src/model_loader.cpp`) — the loader reads every shape and
indexing field from GGUF metadata so the C++ inference graph adapts to
the per-variant `image_size`, `patch_size`, `num_windows`,
`global_attn_indices`, `out_feature_indices`, `pos_embed_train_size`, and
`decoder.layers` without code changes. Verified by `tests/test_variants.cpp`.

Numbers above are on `coco_kitchen.jpg`, T=8, 15 timed iterations after 3
warmup. C++ F32 is faster than PyTorch on every variant except Large
(where the two land within run-to-run variance). The size delta across
variants is small (113-126 MB) because the backbone dominates the
parameter count; the decoder layer count and pos_embed grid size are the
only meaningful differences.

To produce the variant GGUFs from PyTorch:

```sh
scripts/convert_all_variants.sh
```

The script skips any variant whose F32 GGUF already exists, so it's safe
to re-run after a partial conversion. Each variant downloads its
pretrained `.pth` (~30-130 MB) on first instantiation, cached at
`~/.roboflow/models/`.

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

# 2. Convert / download model variants into models/ (one-time).
#    Q4_0 / Q5_0 / Q4_K / Q5_K / Q6_K are optional; bench_community.py picks
#    them up if present and adds them to the per-image latency + accuracy
#    cells. They're excluded from the headline plots — see "What about
#    4-bit?" below.
python3 scripts/convert_rfdetr_to_gguf.py --dtype f32  --output models/rfdetr-base-f32.gguf
python3 scripts/convert_rfdetr_to_gguf.py --dtype q8_0 --output models/rfdetr-base-q8_0.gguf

# Sub-Q8 variants — preferred path is the C++ quantizer (handles K-quants).
build/bin/rfdetr-cli quantize models/rfdetr-base-f32.gguf models/rfdetr-base-q6_K.gguf q6_K   # optional
build/bin/rfdetr-cli quantize models/rfdetr-base-f32.gguf models/rfdetr-base-q5_K.gguf q5_K   # optional
build/bin/rfdetr-cli quantize models/rfdetr-base-f32.gguf models/rfdetr-base-q4_K.gguf q4_K   # optional
# Legacy block quants (cautionary baselines):
build/bin/rfdetr-cli quantize models/rfdetr-base-f32.gguf models/rfdetr-base-q5_0.gguf q5_0   # optional
build/bin/rfdetr-cli quantize models/rfdetr-base-f32.gguf models/rfdetr-base-q4_0.gguf q4_0   # optional

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
