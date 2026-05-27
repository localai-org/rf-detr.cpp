# rf-detr.cpp Model Manifest

Generated 2026-05-27. All models converted from rfdetr 1.7.0 PyTorch
checkpoints via `scripts/convert_rfdetr_to_gguf.py` (F32) and re-quantized
in-place by the C++ quantizer (`build/bin/rfdetr-cli quantize`).

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

All 44 models have been verified to load and run `rfdetr-cli detect` on
`/tmp/coco_sample.jpg` without error.

## Detection variants

| Variant | F32 | F16 | Q8_0 | Q4_K |
|---|---:|---:|---:|---:|
| Nano   | 112.7 MB | 60.5 MB | 36.0 MB | 29.7 MB |
| Small  | 119.0 MB | 64.0 MB | 38.2 MB | 31.2 MB |
| Base   | 119.2 MB | 64.2 MB | 38.5 MB | 31.5 MB |
| Medium | 125.0 MB | 67.2 MB | 40.2 MB | 32.5 MB |
| Large  | 125.9 MB | 68.2 MB | 41.1 MB | 33.4 MB |

## Segmentation variants

| Variant      | F32 | F16 | Q8_0 | Q4_K |
|---|---:|---:|---:|---:|
| Seg-Nano    | 127.1 MB | 67.8 MB | 39.9 MB | 31.8 MB |
| Seg-Small   | 127.6 MB | 68.3 MB | 40.4 MB | 32.3 MB |
| Seg-Medium  | 133.7 MB | 71.5 MB | 42.4 MB | 33.6 MB |
| Seg-Large   | 134.3 MB | 72.2 MB | 43.1 MB | 34.3 MB |
| Seg-XLarge  | 141.3 MB | 76.4 MB | 46.0 MB | 36.5 MB |
| Seg-2XLarge | 143.4 MB | 78.4 MB | 48.0 MB | 38.5 MB |

## Quant choice notes

- **F32**: full precision reference, about 120 MB per variant.
- **F16**: only matmul-multiplicand weights converted; non-matmul tensors
  (norms, conv kernels, embeddings) stay F32. Loader handles F16 `pos_embed`
  via bicubic resample in F32 (see commit 2145c7d).
- **Q8_0**: best accuracy/size trade for production; about 3x smaller than F32.
- **Q4_K**: smallest practical quant; rows with `ne[0] % 256 != 0` (the
  decoder's 128-dim MLP halves, 60 tensors) silently fall back to Q8_0 per
  the C++ quantizer's logic. Net result is still about 3.8 to 4.0x
  compression.

Heavier K-quants (Q5_K, Q6_K) and the older legacy quants (Q4_0/Q4_1/Q5_0/Q5_1)
are supported by `rfdetr-cli quantize` but are not part of the standard
matrix; generate on demand if needed.
