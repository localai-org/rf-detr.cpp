# rt-detr.cpp

C++/ggml inference for [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).

Project layout mirrors [vibevoice.cpp](https://github.com/mudler/vibevoice.cpp).
See `docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md` for the design.

## Performance

End-to-end CPU inference on AMD Ryzen 9 9950X3D, F32, single batch,
`--threads 8` — matches PyTorch (oneDNN-backed `aten::matmul`) at the
optimal thread count:

| impl                              | median ms/image | speedup vs Python |
|-----------------------------------|----------------:|------------------:|
| Python rfdetr (PyTorch + oneDNN)  |             142 | 1.00x (reference) |
| C++ rfdetr.cpp F32 (`--threads 8`)|             140 | ~1.0x             |

Build is configured with `-march=native` + ggml's tinyBLAS SGEMM
(`GGML_LLAMAFILE=ON`) + OpenMP + a persistent gallocr that holds the graph
scratch buffer across inferences — see [BENCHMARK.md](BENCHMARK.md) for
flag-by-flag analysis.

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

### Quantization (Q8_0)

Pass `--dtype q8_0` to the converter to write a Q8_0 GGUF. Only 2D linear
weights with both dims ≥ 64 are quantized; LayerNorm params, biases, conv
kernels, layer-scale gammas, and embeddings stay F32.

| Model               | Size  | Compression | Detections | Class mismatches |
|---------------------|-------|-------------|------------|------------------|
| rfdetr-base-f32     | 120 MB | 1.0x       | 12         | 0                |
| rfdetr-base-q8_0    |  39 MB | 3.08x      | 12         | 0                |

Same kitchen image; max score Δ < 0.02, max box Δ < 1 px. ggml's CPU backend
handles F32 × Q8_0 `mul_mat` natively, so the loader and module code are
unchanged — only the converter writes Q8_0 blocks.

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

### Roadmap

- Backbone drift root-cause + fix
- Variants beyond `base` (`nano`, `small`, `medium`, `large`)
- GPU backends (CUDA / Metal / Vulkan)

## Build

```
git clone --recursive <repo-url>
cd rt-detr.cpp
cmake -B build -DRFDETR_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Convert + run

```
# One-time: convert upstream rfdetr-base.pth → GGUF (requires .venv with rfdetr)
python3 scripts/convert_rfdetr_to_gguf.py \
    --dtype f32 --output models/rfdetr-base-f32.gguf

# Or Q8_0 (~39 MB, ~3.1x smaller, near-identical detections):
python3 scripts/convert_rfdetr_to_gguf.py \
    --dtype q8_0 --output models/rfdetr-base-q8_0.gguf

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
