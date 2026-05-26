# Benchmark — rfdetr.cpp vs upstream Python rfdetr

Cross-check of detection quality and inference latency between this C++
implementation and the reference Python `rfdetr==1.7.0` package on the same
images, same backbone (`rfdetr-base.pth`), same CPU.

## How to reproduce

```
# (one-off) build the CLI
cmake --build build -j

# run the bench script — pass each image with --image
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/bench.py \
    --image /path/to/img1.jpg \
    --image /path/to/img2.jpg \
    --iters 5 --warmup 2 \
    --out BENCH_REPORT.md
```

The C++ side uses `rfdetr-cli bench`, which loads the model once and times
N inference iterations (excluding load). The Python side loads `RFDETRBase`
once and times `predict()` per iteration. Both are pinned to CPU.

## Environment

- CPU: AMD Ryzen 9 9950X3D (16 cores / 20 threads exposed)
- OS:  Linux 6.8 x86_64
- C++ threads: 1 (CLI default)
- Torch: CPU-forced via `CUDA_VISIBLE_DEVICES=""`
- Iterations: 5 timed + 2 warmup per (model, image)

## Inference latency (ms per image, lower is better)

| image            | impl    |    min | median |   mean |    max | speedup vs Python |
|------------------|---------|-------:|-------:|-------:|-------:|------------------:|
| coco_sample.jpg  | python  |  143.7 |  167.9 |  166.3 |  198.7 | 1.00x             |
| coco_sample.jpg  | cpp_f32 | 1591.5 | 1607.2 | 1604.9 | 1615.0 | 0.10x             |
| coco_sample.jpg  | cpp_q8  | 1813.2 | 1815.2 | 1816.9 | 1824.7 | 0.09x             |
| coco_sample2.jpg | python  |  144.0 |  145.0 |  147.4 |  155.2 | 1.00x             |
| coco_sample2.jpg | cpp_f32 | 1591.2 | 1601.9 | 1603.4 | 1621.5 | 0.09x             |
| coco_sample2.jpg | cpp_q8  | 1822.1 | 1824.6 | 1825.8 | 1830.0 | 0.08x             |
| bus.jpg          | python  |  146.5 |  155.5 |  157.2 |  171.8 | 1.00x             |
| bus.jpg          | cpp_f32 | 1621.7 | 1626.1 | 1632.8 | 1647.7 | 0.10x             |
| bus.jpg          | cpp_q8  | 1834.1 | 1836.0 | 1838.4 | 1844.1 | 0.08x             |

The Python implementation runs on PyTorch's multi-threaded CPU backend with
heavily tuned MKL/oneDNN kernels and reaches ~150 ms/image. The C++ build
currently runs on `ggml`'s CPU backend with `n_threads = 1` (CLI default)
and lands at ~1.6 s for F32 and ~1.8 s for Q8_0. Two notes on the gap:

- We're running single-threaded; raising `n_threads` should narrow this a
  lot. The CLI doesn't expose a `--threads` flag yet — a one-line addition.
- Q8_0 is slower than F32 here because ggml's CPU `mul_mat` dequantizes
  Q8_0 blocks on the fly when one operand is F32 (activations); the
  arithmetic cost dominates the smaller memory footprint at this batch=1
  shape. Q8_0's main payoff today is the 3.08x model-size reduction
  (120 MB → 39 MB) with no measurable detection-quality loss (see below).

## Detection cross-check (Python as reference, IoU >= 0.95)

| image            | impl    | py_det | cpp_det | matched | mean IoU | mean \|score Δ\| | max \|score Δ\| | mean center Δ (px) | max center Δ (px) |
|------------------|---------|-------:|--------:|--------:|---------:|-----------------:|----------------:|-------------------:|------------------:|
| coco_sample.jpg  | cpp_f32 |     12 |      12 |      12 |   0.9906 |           0.0126 |          0.0296 |               0.14 |              0.37 |
| coco_sample.jpg  | cpp_q8  |     12 |      12 |      12 |   0.9898 |           0.0133 |          0.0279 |               0.13 |              0.30 |
| coco_sample2.jpg | cpp_f32 |      5 |       5 |       5 |   0.9982 |           0.0056 |          0.0120 |               0.11 |              0.23 |
| coco_sample2.jpg | cpp_q8  |      5 |       5 |       5 |   0.9972 |           0.0110 |          0.0305 |               0.16 |              0.33 |
| bus.jpg          | cpp_f32 |      5 |       5 |       5 |   0.9982 |           0.0010 |          0.0019 |               0.12 |              0.23 |
| bus.jpg          | cpp_q8  |      5 |       5 |       5 |   0.9984 |           0.0013 |          0.0032 |               0.17 |              0.26 |

Across all three images and both quantizations every Python detection has a
1-1 match in the C++ output at the same class, IoU > 0.95 (mean > 0.99),
sub-pixel bbox drift, and confidence drift under 0.04 in the worst case.
No false positives, no missed detections.

## Caveats

- Numbers are hardware-specific; speedups will look different on Intel or
  Arm CPUs, and very different once threading is enabled.
- Python `predict()` includes PIL load + tensor prep + postprocess, just
  like the C++ side includes preprocess + postprocess; both are measured
  end-to-end so the comparison is apples-to-apples.
- The first 2 iterations on each side are dropped as warmup (allocator and
  cache fill).
- Bench script and CLI subcommand are reproducible; the input images are
  not committed.
