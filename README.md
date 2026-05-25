# rt-detr.cpp

C++/ggml inference for [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).

Project layout mirrors [vibevoice.cpp](https://github.com/mudler/vibevoice.cpp).
See `docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md` for the design.

## Status

**End-to-end detection (Plan 6c) complete.** `rfdetr-cli detect` now runs
the full RF-DETR forward pipeline — image preprocessing (resize + ImageNet
normalize), DINOv2 backbone (windowed + global attention), multi-scale
projector, 3-layer encoder, 3-layer decoder (300 queries), class + bbox
heads — and emits real JSON detections via Plan 1's postprocess. With the
synthesized random-weight fixture, detection scores rarely exceed the 0.5
threshold (output is typically empty), but the C++ pipeline is fully
exercised end-to-end. 97 parity checkpoints all green at 1e-5 absolute
tolerance against the numpy reference. Ten tests pass on a clean build.

The Python conversion script body (`scripts/convert_rfdetr_to_gguf.py`) is
still deferred — see Plan 2 Task 3. Once a real upstream `rfdetr-base`
checkpoint is converted to GGUF, `rfdetr-cli detect` will produce
meaningful detections.

Plan 7 swaps the numpy reference for a torch+rfdetr baseline (verifying
against the real model). Plan 8 adds Q8_0 quantization. Plan 9 adds the
nano/small/medium/large variants.

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
