# rf-detr.cpp Model Manifest

Generated 2026-07-30. All models converted from `rfdetr==1.9.0` PyTorch
checkpoints via `scripts/convert_rfdetr_to_gguf.py` (F32) and re-quantized
in-place by the C++ quantizer (`build/bin/rfdetr-cli quantize`), at commit
[`52b1650`](https://github.com/localai-org/rf-detr.cpp/commit/52b1650864016b272d377b99fdbbda3339364013)
(pre-merge; see the PR for the final merged SHA).

The `.gguf` files themselves are gitignored; this manifest is the canonical
record of what should exist and at what size. To rebuild from scratch:

```bash
# 1. Convert PyTorch -> F32 GGUF (one per variant)
for v in nano small base medium large \
         seg-nano seg-small seg-medium seg-large seg-xlarge seg-2xlarge; do
    scripts/convert_rfdetr_to_gguf.py --variant "$v" --dtype f32 \
        --output "models/rfdetr-${v}-f32.gguf"
done

# 2. Materialize F16, Q8_0, Q4_K for every F32 source
scripts/build_all_quants.sh
```

All 44 models have been verified end-to-end:

- `rfdetr-cli info` reports the correct variant, image size, class count,
  query count, and tensor count for every file.
- `rfdetr.preprocess.resize_mode` is present and equals
  `bilinear_no_antialias` on every file (new RF-DETR 1.9 preprocessing path).
- Detection variants complete a real detection on
  `tests/fixtures/ci/test_image.jpg`. Segmentation variants additionally
  produce per-detection mask PNGs.
- PyTorch-vs-C++ parity was swept across all 44 (variant × quant) cells
  (`scripts/sweep_accuracy.py`, results in
  `benchmarks/results/accuracy_sweep.json`): F32/F16/Q8_0 all reach
  Recall@IoU-0.5 = 1.0 and mean segmentation mask IoU ≥ 0.95 everywhere; Q4_K
  shows the expected accuracy/size tradeoff (Recall@IoU-0.95 as low as 0.125
  on seg-small — see the sweep JSON for the full per-cell breakdown, and do
  not treat Q4_K as parity-equivalent to F16/Q8_0).
- Backward compatibility was reconfirmed against a GGUF built from the
  pre-1.9 converter (no `resize_mode` key): it loads and detects correctly
  via the legacy stb resize path.

Full per-file SHA-256 checksums are published as `SHA256SUMS` in each
variant's Hugging Face repository (see README for the repo list) and are not
duplicated here to keep this file readable.

## Detection variants

| Variant | F32 | F16 | Q8_0 | Q4_K |
|---|---:|---:|---:|---:|
| Nano   | 118.2 MB | 63.4 MB | 37.8 MB | 31.2 MB |
| Small  | 124.7 MB | 67.1 MB | 40.1 MB | 32.8 MB |
| Base   | 125.0 MB | 67.4 MB | 40.3 MB | 33.0 MB |
| Medium | 131.0 MB | 70.5 MB | 42.1 MB | 34.0 MB |
| Large  | 132.0 MB | 71.5 MB | 43.1 MB | 35.0 MB |

## Segmentation variants

| Variant      | F32 | F16 | Q8_0 | Q4_K |
|---|---:|---:|---:|---:|
| Seg-Nano    | 133.3 MB | 71.1 MB | 41.9 MB | 33.4 MB |
| Seg-Small   | 133.8 MB | 71.6 MB | 42.4 MB | 33.9 MB |
| Seg-Medium  | 140.4 MB | 75.1 MB | 44.5 MB | 35.3 MB |
| Seg-Large   | 141.1 MB | 75.8 MB | 45.2 MB | 36.0 MB |
| Seg-XLarge  | 148.8 MB | 80.4 MB | 48.4 MB | 38.4 MB |
| Seg-2XLarge | 150.9 MB | 82.5 MB | 50.5 MB | 40.5 MB |

Note: segmentation variants medium/large/xlarge/2xlarge are each ~1 block
larger than in the pre-1.9 manifest (5 or 6 `DepthwiseConvBlock`s instead of
a hardcoded 4) after fixing a real segmentation-head coverage bug — see
commit `52b1650` — so these sizes are not directly comparable to any
previously published rf-detr.cpp seg-medium/large/xlarge/2xlarge GGUF.

## Quant choice notes

- **F32**: full precision reference, about 120-150 MB per variant.
- **F16**: only matmul-multiplicand weights converted; non-matmul tensors
  (norms, conv kernels, embeddings) stay F32. Loader handles F16 `pos_embed`
  via bicubic resample in F32 (see commit 2145c7d).
- **Q8_0**: best accuracy/size trade for production; about 3x smaller than F32.
- **Q4_K**: smallest practical quant; rows with `ne[0] % 256 != 0` (the
  decoder's 128-dim MLP halves) silently fall back to Q8_0 per the C++
  quantizer's logic. Net result is still about 3.7 to 3.9x compression, but
  expect a real, sometimes large, Recall@IoU-0.95 drop — see
  `benchmarks/results/accuracy_sweep.json` per variant before choosing Q4_K
  for an accuracy-sensitive deployment.

Heavier K-quants (Q5_K, Q6_K) and the older legacy quants (Q4_0/Q4_1/Q5_0/Q5_1)
are supported by `rfdetr-cli quantize` but are not part of the standard
matrix; generate on demand if needed.
