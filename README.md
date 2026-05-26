# rt-detr.cpp

C++/ggml inference for [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).

Project layout mirrors [vibevoice.cpp](https://github.com/mudler/vibevoice.cpp).
See `docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md` for the design.

## Benchmark results

End-to-end CPU inference on AMD Ryzen 9 9950X3D (F32, single batch,
`--threads 8`) — matches PyTorch (oneDNN-backed `aten::matmul`) and ships
in 1/3 the disk footprint via Q8_0 quantization:

![Latency comparison: PyTorch vs rfdetr.cpp F32 vs rfdetr.cpp Q8_0 across 7 COCO images](benchmarks/plots/latency_comparison.png)

| impl                          | median ms/image | model size | relative speed | detection match vs PyTorch |
|-------------------------------|----------------:|-----------:|---------------:|---------------------------:|
| Python rfdetr (PyTorch+oneDNN) |          152.5 |     120 MB | 1.00× (ref)    | reference                  |
| C++ rfdetr.cpp F32 (T=8)       |      **144.5** |     120 MB | **0.95×**      | 54/55 IoU ≥ 0.95, max \|Δscore\| 0.045 |
| C++ rfdetr.cpp Q8_0 (T=8)      |      **145.5** |  **39 MB** | **0.95×**      | 54/55 IoU ≥ 0.95, max \|Δscore\| 0.046 |

Numbers are means across 7 diverse COCO val2017 images, 15 iterations each,
3 warmup. Build uses `-march=native` + ggml's tinyBLAS SGEMM
(`GGML_LLAMAFILE=ON`) + OpenMP + a persistent ggml graph allocator. See
[BENCHMARK.md](BENCHMARK.md) for the per-image breakdown, thread-scaling
sweep, methodology, and reproduction recipe.

## Status

**End-to-end inference works on rfdetr-base.** Detections on real COCO
images match the upstream Python rfdetr to within sub-pixel box drift and
≤0.05 confidence drift on every detection above the 0.5 threshold.

Module parity vs torch baseline (max_abs):

| Module     | max_abs   | Notes |
|------------|-----------|-------|
| Backbone   | ~1.14     | Structural; compounding accumulator drift TBD |
| Projector  | ~1e-5     | C2f single-scale (P4) |
| Two-stage  | ~3e-6     | Group-0 enc_output + reparam |
| Decoder    | ~5e-5     | 3 layers, deformable cross-attn |
| Heads      | ~2e-6     | Shared class_embed + 3-layer bbox MLP |
| E2E logits | ~3.4      | Cumulative — driven by backbone drift |

The class-rank ordering is preserved end-to-end despite the backbone gap,
which is why detections survive intact. Fixing the backbone drift will
tighten the cumulative number.

### Verified detection example (COCO val 397133, kitchen scene)

| Class       | C++ score | rfdetr score | Δ box (px) |
|-------------|-----------|--------------|------------|
| person      | 0.956     | 0.958        | < 0.2      |
| bowl        | 0.916     | 0.916        | < 0.1      |
| bowl        | 0.885     | 0.874        | < 0.2      |
| bowl        | 0.791     | 0.791        | < 0.2      |
| potted plant| 0.661     | 0.657        | < 0.1      |
| person      | 0.659     | 0.684        | < 0.5      |
| bowl        | 0.651     | 0.657        | < 0.4      |
| sink        | 0.637     | 0.624        | < 0.2      |
| bowl        | 0.595     | 0.608        | < 0.1      |
| oven        | 0.574     | 0.570        | < 0.7      |
| cup         | 0.571     | 0.587        | < 0.4      |
| spoon       | 0.509     | 0.520        | < 0.1      |

12/12 detections match in class, with no false positives or negatives.

### Quantization (Q8_0 + K-quants)

Pass `--dtype q8_0` to the Python converter to write a Q8_0 GGUF. Only
2D linear weights with both dims ≥ 64 are quantized; LayerNorm params,
biases, conv kernels, layer-scale gammas, and embeddings stay F32.

| Model               | Size  | Compression | Detections | Class mismatches |
|---------------------|-------|-------------|------------|------------------|
| rfdetr-base-f32     | 120 MB | 1.0x       | 12         | 0                |
| rfdetr-base-q8_0    |  39 MB | 3.08x      | 12         | 0                |

Same kitchen image; max score Δ < 0.02, max box Δ < 1 px. ggml's CPU
backend handles F32 × Q8_0 `mul_mat` natively, so the loader and module
code are unchanged — only the converter writes Q8_0 blocks.

For sub-Q8 footprints, **use the C++ quantizer** — it supports the full
ggml set including K-quants (which the Python `gguf` package can't write):

```sh
# K-quants — the recommended sub-Q8 path.
build/bin/rfdetr-cli quantize models/rfdetr-base-f32.gguf \
    models/rfdetr-base-q6_K.gguf q6_K    # ~36 MB, detection-identical to Q8_0
build/bin/rfdetr-cli quantize models/rfdetr-base-f32.gguf \
    models/rfdetr-base-q4_K.gguf q4_K    # ~32 MB, beats legacy Q4_0 by 2× on Δscore
```

Legacy block quants (`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`) are still
available through both the Python converter and the C++ quantizer; the
two paths produce **byte-for-byte identical Q8_0 tensor data** (and
differ in only 1-2 nibbles total across ~30 MB for Q4_0 / Q5_0, an
FMA-rounding artifact that's documented as a known quirk in the upstream
gguf package). Sub-Q8 variants don't speed up CPU inference on this
model — see `BENCHMARK.md → "What about 4-bit?"` for the full
size/speed/accuracy tradeoff. **Q8_0 remains the default; Q6_K is the
recommended sub-Q8 option for size-constrained deployment.**

### Benchmark vs upstream Python

A cross-implementation benchmark (latency + detection cross-check) is
documented in [BENCHMARK.md](BENCHMARK.md). On the same backbone and CPU,
C++ detections match Python 1-1 (IoU > 0.99 mean, < 0.05 confidence drift,
sub-pixel boxes); inference latency matches PyTorch at the optimal thread
count (~140 ms median vs ~142 ms).

Reproduce with:

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/bench.py \
    --image path/to/img.jpg --iters 5 --warmup 3 --threads 8
```

### Detection variants

All 5 RF-DETR detection variants (Nano / Small / Base / Medium / Large)
load through the same C++ pipeline. They share the DINOv2-small backbone
and differ only in input resolution and decoder layer count:

| Variant | Resolution | Decoder layers | GGUF F32 | C++ F32 median ms @ T=8 |
|---------|-----------:|---------------:|---------:|------------------------:|
| Nano    |        384 |              2 |   113 MB |                    61.5 |
| Small   |        512 |              3 |   119 MB |                   116.0 |
| Base    |        560 |              3 |   119 MB |                   159.3 |
| Medium  |        576 |              4 |   125 MB |                   149.6 |
| Large   |        704 |              4 |   126 MB |                   237.8 |

![Variants overview](benchmarks/plots/variants_overview.png)

Generate all five F32 GGUFs in one shot:

```sh
scripts/convert_all_variants.sh
```

See [BENCHMARK.md → "Variant comparison"](BENCHMARK.md#variant-comparison)
for the full per-variant breakdown vs PyTorch.

### Roadmap

- Backbone drift root-cause + fix
- GPU backends (CUDA / Metal / Vulkan)
- Segmentation variants (`RFDETRSeg*`)
- Custom-checkpoint conversion (fine-tuned models)

## Build

```
git clone --recursive <repo-url>
cd rt-detr.cpp
cmake -B build -DRFDETR_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The build automatically applies two patches to `third_party/ggml` at configure
time (stored in `third_party/ggml-patches/`). These are local performance and
debug-instrumentation improvements not yet upstreamed. Re-running CMake is a
no-op once they're in place. Run `scripts/apply_ggml_patches.sh` manually if
you want to inspect the patch flow.

## Convert + run

```
# One-time: convert upstream rfdetr-base.pth → GGUF (requires .venv with rfdetr)
python3 scripts/convert_rfdetr_to_gguf.py \
    --dtype f32 --output models/rfdetr-base-f32.gguf

# Or Q8_0 (~39 MB, ~3.1x smaller, near-identical detections):
python3 scripts/convert_rfdetr_to_gguf.py \
    --dtype q8_0 --output models/rfdetr-base-q8_0.gguf

# Or re-quantize an existing F32 GGUF to any ggml type (incl. K-quants):
./build/bin/rfdetr-cli quantize \
    models/rfdetr-base-f32.gguf models/rfdetr-base-q6_K.gguf q6_K
# Supported types: f32 | f16 | q4_0 | q4_1 | q5_0 | q5_1 | q8_0 | q4_K | q5_K | q6_K

# Detect
./build/bin/rfdetr-cli detect \
    --model models/rfdetr-base-f32.gguf \
    --input  my_image.jpg \
    --output detections.json \
    --threshold 0.5 \
    --annotated out.png
```

## License

Apache-2.0. See `LICENSE`.
