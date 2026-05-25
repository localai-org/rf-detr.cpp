# rt-detr.cpp

C++/ggml inference for [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).

Project layout mirrors [vibevoice.cpp](https://github.com/mudler/vibevoice.cpp).
See `docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md` for the design.

## Status

**Decoder (Plan 6b) complete.** The forward pipeline now runs:
DINOv2 backbone → multi-scale projector → 3-layer encoder → 3-layer decoder
(300 learnable queries with self-attention + cross-attention against the
encoder output + FFN). 91 parity checkpoints all green at 1e-5 absolute
tolerance against the numpy reference. Ten tests pass on a clean build.

Plan 6c attaches the class+bbox heads and wires `rfdetr_detect` end-to-end
(CLI `detect` produces real JSON detections, even if from random-weight
nonsense scores).

The Python conversion script body is still deferred (see Plan 2 Task 3).
The C++ side uses a synthesized F32 GGUF fixture for tests.

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
