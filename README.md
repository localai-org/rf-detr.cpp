# rt-detr.cpp

C++/ggml inference for [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).

Project layout mirrors [vibevoice.cpp](https://github.com/mudler/vibevoice.cpp).
See `docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md` for the design.

## Status

**Plan 7 schema rewrite landed; Plans 8-12 in progress (transitional).**

This project is currently in the middle of an architectural pivot to
exactly match rfdetr 1.7.0 (real upstream). What works today:

- GGUF schema (format version 2) matches real rfdetr-base
- `scripts/convert_rfdetr_to_gguf.py` produces a real
  `models/rfdetr-base-f32.gguf` from the upstream PyTorch checkpoint
- `rfdetr-cli info models/rfdetr-base-f32.gguf` loads and introspects
  the real model (486 tensors, 91 classes, 300 queries, image_size 560)
- Fixture generator, loader, and C-API init/free path are all on the
  new schema

What's red:

- Numerical modules (`dinov2`, `encoder`, `decoder`, `projector`, `heads`)
  still implement the v1 (aspirational) architecture and don't run against
  v2 weights. `rfdetr_detect` returns `RFDETR_ERR_INFERENCE` until Plans
  8-12 rewrite them.
- `test_parity_full_forward` and its numpy reference are disabled until
  Plan 13 swaps in a torch baseline matching real upstream.
- `rfdetr-cli detect` runs the loader successfully but fails at inference.

Roadmap:

- **Plan 8**: backbone redesign (separate Q/K/V, layer-scale, DINOv2-small)
- **Plan 9**: delete standalone encoder; two-stage init
- **Plan 10**: conv-based projector (C2f)
- **Plan 11**: deformable cross-attention (the biggest single change)
- **Plan 12**: heads redesign + 91 classes
- **Plan 13**: torch parity baseline (replaces numpy reference)
- **Plan 14**: Q8_0 quantization
- **Plan 15**: real E2E demo on a COCO image

## Build

```
git clone --recursive <repo-url>
cd rt-detr.cpp
cmake -B build -DRFDETR_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## License

Apache-2.0. See `LICENSE`.
