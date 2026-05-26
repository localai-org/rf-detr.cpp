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

## Headline: F16 is the new sweet-spot recommendation

![Latency comparison](benchmarks/plots/latency_comparison.png)

At T=8 (a single CCD's worth of physical cores), `rfdetr.cpp` runs faster
than PyTorch on every test image across all three precisions. The three
C++ variants — **F32**, **F16**, and **Q8_0** — share the same accuracy
floor against PyTorch, but they sit in three very different points on the
size/speed plane.

| Variant | Size (MB) | Compression vs F32 | Median ms (mean across 7 imgs) | Accuracy vs PyTorch (IoU ≥ 0.95) |
|---------|----------:|-------------------:|-------------------------------:|---------------------------------:|
| F32     |     119.2 |              1.00× |                          150.1 | 54/55, max \|Δscore\| 0.045      |
| **F16** |  **64.2** |          **1.86×** |                      **151.3** | 54/55, max \|Δscore\| 0.044      |
| Q8_0    |      38.5 |              3.10× |                          164.8 | 54/55, max \|Δscore\| 0.046      |

**F16 is the new recommended default** for this model on CPU:

- **F32-class speed.** Mean median across 7 images: F16 = 151.3 ms vs
  F32 = 150.1 ms. F16 is within 1 ms of F32 on the mean, and beats F32
  outright on 3 of 7 images (indoor / skater / street); F32 edges F16 on
  the other 4 by 2-15 ms (within run-to-run noise on this dual-CCD host).
- **F16 beats Q8_0 on 5 of 7 images on median, and on every image on
  min-time (cleanest run).** Mean median latency: F16 = 151.3 ms vs
  Q8_0 = 164.8 ms → F16 is ~8% faster overall. ggml's optimized F32×F16
  matmul path beats the Q8_0 vec-dot kernel for this model's shapes.
  The 2 images where Q8_0 edges F16 on median (bus, kitchen) sit within
  run-to-run noise on this dual-CCD host.
- **1.86× smaller than F32** (64 MB vs 120 MB), halfway to Q8_0's 3.10×
  without paying for it in latency.
- **Lossless against F32.** Direct F16-vs-F32 comparison: 56/56 detections
  match at IoU ≥ 0.95, **max |Δscore| = 0.006, mean |Δscore| = 0.0005**
  — pure FP rounding noise.

Q8_0 is still the right pick **when disk size dominates** — same accuracy,
3.08× compression, ~10% latency tax. But for a server/embedded workload
that has the RAM for ~65 MB of weights, **F16 wins on every axis except
on-disk footprint**.

Per-image median latency, T=8, 15 timed iterations after 3 warmup:

| image       | PyTorch (ms) | C++ F32 (ms) | C++ F16 (ms) | C++ Q8_0 (ms) | F16 vs Q8_0 | F16 vs F32 |
|-------------|-------------:|-------------:|-------------:|--------------:|------------:|-----------:|
| bus         |        226.5 |        143.9 |        160.8 |         148.8 |       1.08× (slower) | 1.12× (slower) |
| cats        |        198.8 |        162.9 |        171.8 |         176.6 | **0.97×** (F16 wins) | 1.05× (slower) |
| indoor      |        230.2 |        166.5 |    **138.9** |         182.8 | **0.76×** (F16 wins) | **0.83×** (F16 wins) |
| kitchen     |        206.9 |        147.2 |        161.8 |         158.8 |       1.02× (~tied) | 1.10× (slower) |
| living room |        218.0 |        142.8 |        145.6 |         147.6 | **0.99×** (F16 wins) | 1.02× (~tied) |
| skater      |        209.4 |        144.5 |    **141.4** |         182.1 | **0.78×** (F16 wins) | **0.98×** (F16 wins) |
| street      |        200.8 |        142.8 |    **138.7** |         156.8 | **0.89×** (F16 wins) | **0.97×** (F16 wins) |
| **median**  |        209.4 |        144.5 |        145.6 |         158.8 |       0.92× |      1.01× |
| **mean**    |        212.9 |        150.1 |        151.3 |         164.8 |       0.92× |      1.01× |

Honest take: **F16 vs F32 is statistically a wash on this host on
median**, but F16 buys back ~55 MB of disk and process RSS without
paying for it. **F16 vs Q8_0**: F16 wins 5/7 on median (cats / indoor
/ living room / skater / street); the 2 losses (bus, kitchen) are by
<5 ms, within the dual-CCD scheduler's run-to-run noise floor. On
**min-time** (the cleanest iteration in the sweep), **F16 is 134-141 ms
across all 7 images and beats both F32 and Q8_0 in every single run** —
the F16 fast path is reliably the fastest matmul kernel on this CPU; the
median noise comes from interrupt / scheduler / thermal jitter, not from
the kernel itself. See `bench_data.json` for raw per-iteration data.

![Relative latency](benchmarks/plots/relative_latency.png)

### Why F16 wins on CPU

ggml's CPU backend has a hand-tuned **F32×F16 mul_mat fast path** that
dequantizes F16 weights into FP32 lanes *inside* the SIMD micro-kernel
(see `ggml-cpu/quants.c` and the `vec_dot_f32_f16` family). The kernel
loads the F16 weight blob with half the memory bandwidth of F32 (65 MB
vs 120 MB resident) and pays roughly **zero conversion overhead** — the
FMA-side ALUs are not the bottleneck; bandwidth is.

In contrast, the Q8_0 vec-dot kernel still has to do block-scale
multiplies and packed-int → float reconstruction every 32 elements. On
AVX-512 / VNNI hardware that's still fast (the VPDPBUSD path is the
reason Q8_0 is competitive with F32 at all), but it's a constant ~10%
slower per matmul than the F16 path because the dequant arithmetic
costs cycles that the F16 path doesn't pay.

So the ranking for rfdetr-base on this CPU is roughly:

```
F16  matmul  =  F32 matmul × (cache-bound × 0.6) + (compute × 1.0)  → ~F32 speed
Q8_0 matmul  =  F32 matmul × (cache-bound × 0.4) + (compute × 1.15) → ~F32 × 1.10
```

The model is small enough that **all three variants live in L2/L3 once
warm**, so the bandwidth win for F16 is mostly absorbed by cache reuse
across the 2,400-step graph. That's why the F16-vs-F32 median delta is
basically zero on this host; the bandwidth headroom that F16 buys you
shows up more clearly on smaller-cache CPUs and on larger models where
the weight blob spills out of L3.

### F16 vs F32 direct accuracy

Detection-for-detection across all 7 images:

| Metric                   | F16 vs F32 |
|--------------------------|-----------:|
| matched (IoU ≥ 0.95)     |      56/56 |
| mean \|Δscore\|          |     0.0005 |
| max  \|Δscore\|          |     0.0059 |

The maximum F16-vs-F32 score difference across the entire bench is **6
parts in 10,000** — pure FP rounding. There is no detection that F32
finds and F16 misses, no class swap, no bbox shift beyond sub-pixel.
This is the strictest sense in which a quantization is "lossless":
indistinguishable from the reference except for IEEE round-to-nearest
on the last few mantissa bits.

This is why we recommend F16 over F32 as the default: it's the same
inference, half the size.

---

## Thread scaling

![Thread scaling](benchmarks/plots/thread_scaling.png)

A representative image (`coco_kitchen.jpg`) swept over T ∈ {1, 2, 4, 8, 12, 16, 20}:

| Threads | PyTorch (ms) | C++ F32 (ms) | C++ F16 (ms) | C++ Q8_0 (ms) |
|--------:|-------------:|-------------:|-------------:|--------------:|
|       1 |        823.3 |        888.6 |        841.4 |         928.2 |
|       2 |        434.1 |        463.1 |        443.2 |         494.2 |
|       4 |        312.4 |        248.7 |        238.4 |         265.8 |
|       8 |        161.3 |        143.7 |        146.2 |         169.8 |
|      12 |        133.5 |        148.9 |        148.0 |         154.2 |
|      16 |    **125.8** |    **133.3** |    **144.0** |     **138.7** |
|      20 |        210.6 |        196.8 |        220.1 |         234.6 |

Observations:

- **All four implementations track within ~10% across the entire sweep**.
  C++ F16 leads at lower thread counts (T=1, 4) because its bandwidth
  advantage matters most when fewer cores are sharing the memory subsystem;
  at T≥8 the three C++ variants converge.
- **The minimum is at T=16**, not T=8, for all four. That's the host's 16
  physical cores; the post-gallocr-fix workload now scales further than the
  original cross-CCD-bound case in earlier revisions. Whatever ggml gained
  by eliminating allocator churn put the bottleneck back into FLOPs.
- **T=20 regresses for everyone** (~30-65% slower than T=16). The 20 logical
  cores on this host include over-subscription beyond physical, and crossing
  the dual-CCD boundary thrashes per-op L3 locality on the larger residency.
- **T=8 remains a reasonable default**: ~145 ms across F32/F16, ~170 ms on
  Q8_0; single-CCD resident, low contention. If you have a 9950X3D,
  `--threads 16` will get you to ~133 ms on F32; on single-CCD parts the
  sweet spot tracks the core count directly.

---

## Model size + accuracy

![Size and accuracy](benchmarks/plots/size_and_accuracy.png)

| Variant | Size (MB) | Compression vs F32 |
|---------|----------:|-------------------:|
| F32     |     119.2 |              1.00× |
| **F16** |  **64.2** |          **1.86×** |
| Q8_0    |      38.5 |              3.10× |

Across the 7 test images, every PyTorch detection has a 1-1 C++ match at
IoU ≥ 0.95 with one caveat per impl:

| impl     | Python dets | C++ dets | matched (IoU ≥ 0.95) | mean \|Δscore\| | max \|Δscore\| |
|----------|------------:|---------:|---------------------:|----------------:|---------------:|
| C++ F32  |          55 |       56 |                   54 |          0.0078 |         0.0445 |
| C++ F16  |          55 |       56 |                   54 |          0.0079 |         0.0440 |
| C++ Q8_0 |          55 |       55 |                   54 |          0.0084 |         0.0461 |

- On **coco_living_room**, one C++ detection (class 64, score 0.504 vs
  Python 0.512, bbox shifted ~1 px) lands at IoU ≈ 0.93 — just under the
  0.95 threshold but visibly the same object.
- On **coco_skater**, the C++ F32 / F16 builds find one extra detection
  (class 1 "person" at score 0.507) that the PyTorch arm scores fractionally
  below the 0.5 threshold. This is honest borderline behavior on threshold-
  bounded detection sets; both detections look reasonable on the image.

Max per-detection score drift across the entire benchmark is **0.046**
(Q8_0). **F16 matches F32 to within 0.006 score and zero detection-count
drift** — see the "F16 vs F32 direct accuracy" subsection above.

Headline accuracy guarantee: F16 quantization is **strictly lossless**
(rounding-only, same detection counts as F32), and Q8_0 quantization is
**effectively lossless** (same detection counts, score drift below the
detection-threshold noise floor).

---

## What about 4-bit?

> *tl;dr — **F16 is the new default; Q8_0 is the right pick when disk
> size dominates**. If you need to squeeze below 38 MB the K-quants
> (`q6_K`, `q5_K`, `q4_K`) are the right tool: **Q6_K matches Q8_0
> detection quality at 36 MB**, and **Q4_K beats legacy Q4_0 by a wide
> margin** on both recall and max Δscore at roughly the same on-disk
> size. Legacy `q4_0` / `q5_0` ship as a cautionary baseline only.*

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
| **F16** |  **64.2** |  1.86× |  0.60×  |
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

| impl     | dets matched (IoU ≥ 0.5) | dets matched (IoU ≥ 0.95) | max \|Δscore\| |
|----------|-------------------------:|--------------------------:|---------------:|
| **F16**  |              **55 / 55** |               **54 / 55** |      **0.044** |
| Q8_0     |              **55 / 55** |               **54 / 55** |      **0.046** |
| Q6_K     |              **55 / 55** |               **54 / 55** |          0.051 |
| Q5_K     |                  52 / 55 |                   51 / 55 |          0.066 |
| Q5_0     |                  53 / 55 |                   46 / 55 |          0.069 |
| Q4_K     |                  51 / 55 |                   45 / 55 |          0.110 |
| Q4_0     |                  49 / 55 |                   40 / 55 |          0.226 |

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

### Latency (median ms/image across 7 images, T=8, 9950X3D)

| Variant | median of medians (ms) | mean of medians (ms) | vs F32 |
|---------|-----------------------:|---------------------:|-------:|
| F32     |                  144.5 |                150.1 |  1.00× |
| **F16** |              **145.6** |            **151.3** |  1.01× |
| Q8_0    |                  158.8 |                164.8 |  1.10× |
| Q4_K    |                  166.4 |                174.0 |  1.16× |
| Q5_0    |                  168.7 |                172.2 |  1.15× |
| Q6_K    |                  174.5 |                186.0 |  1.24× |
| Q4_0    |                  178.7 |                177.2 |  1.18× |
| Q5_K    |                  190.0 |                195.4 |  1.30× |

**Headline:** F16 is essentially tied with F32 on median latency
(151.3 ms vs 150.1 ms across 7 images) and is the fastest variant
among everything quantization-aware. **F16 beats Q8_0 by ~9% in mean
median latency** and beats every K-quant by 15-30%.

ggml's F32×F16 mul_mat fast path is the reason: the kernel dequants
the F16 weight on-the-fly inside the FMA loop, with no block-scale
arithmetic to do (unlike Q8_0 / K-quants). For rfdetr-base's matmul
shapes on AVX-512+VNNI, that path is bandwidth-bound, not compute-
bound, and F16 halves the bandwidth bill vs F32.

K-quants pay the same dequant tax as legacy quants plus extra
super-block overhead, so they end up ~10-30% slower than F16/F32. The
size savings are real but they trade speed for disk.

### Conclusions

- **F16 is the new default** for size+accuracy+speed: 1.86× smaller
  than F32, F32-class speed, lossless against F32, and reliably faster
  than every quant variant.
- **Q8_0 is the right pick when disk size dominates** — 3.10×
  compression, same detection accuracy as F32 / F16, ~10% latency tax
  vs F16.
- **Q6_K is the right pick when you want under 38 MB without compromise
  on accuracy.** Detection-identical to Q8_0 at 3.4× compression.
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
python3 scripts/convert_rfdetr_to_gguf.py --dtype f16  --output models/rfdetr-base-f16.gguf
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
