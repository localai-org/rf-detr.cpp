# rf-detr.cpp Model Manifest

The five detection variants were generated 2026-05-27 from rfdetr 1.7.0
checkpoints. All six segmentation variants were re-generated 2026-07-30 from
rfdetr 1.9.0 checkpoints, after the segmentation-head block-count fix. Every
model is converted with `scripts/convert_rfdetr_to_gguf.py` (F32) and
re-quantized in-place by the C++ quantizer (`build/bin/rfdetr-cli quantize`).
Re-running the steps below today converts from rfdetr 1.9.0 throughout, since
that is what `scripts/requirements.txt` pins.

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

All 44 models were verified to load and run `rfdetr-cli detect` without error at
the time they were generated, and all 44 are current: the 16 stale segmentation
files described in earlier revisions of this manifest have been replaced.

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
| Seg-Small   | 127.6 MB | 68.3 MB | 40.4 MB | 32.4 MB |
| Seg-Medium  | 133.9 MB | 71.6 MB | 42.5 MB | 33.6 MB |
| Seg-Large   | 134.6 MB | 72.3 MB | 43.1 MB | 34.3 MB |
| Seg-XLarge  | 141.9 MB | 76.7 MB | 46.1 MB | 36.6 MB |
| Seg-2XLarge | 143.9 MB | 78.7 MB | 48.2 MB | 38.6 MB |

Sizes are as measured on the files that are published, in MiB, and they are the
post-fix sizes: seg-medium, seg-large, seg-xlarge and seg-2xlarge each carry one
or two more `DepthwiseConvBlock`s than the GGUFs originally published for them
(one per decoder layer, so 5 for medium/large and 6 for xlarge/2xlarge, instead
of a hardcoded 4). All six segmentation variants have now been re-converted from
rfdetr 1.9.0 and re-published, and the loader rejects any surviving copy of an
old short-block file with an explanatory error.

A second, unrelated defect lived in the C++ two-stage module rather than in the
weights: it omitted upstream's `gen_encoder_output_proposals` validity mask
until commit `d256d3e`. That mask can only fire on a patch grid of side 50 or
more, which means seg-xlarge (grid 52) and seg-2xlarge (grid 64) and no other
variant. It changed no GGUF, so nothing here needed re-converting on its
account; see `docs/conversion.md`.

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
