# rt-detr.cpp

C++/ggml inference for [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).

Project layout mirrors [vibevoice.cpp](https://github.com/mudler/vibevoice.cpp).
See `docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md` for the design.

## Status

**Window attention (Plan 5) complete.** The DINOv2 backbone now dispatches
between global and windowed self-attention per block — the four
`multi_scale_layers` indices `[2, 5, 8, 11]` use global attention; the other
eight blocks window the patch tokens into 2×2 partitions and attend within
each window. All 55 parity checkpoints still pass at 1e-5 absolute
tolerance against the numpy reference; windowed-block max_abs values
(~7e-9) are indistinguishable from global-block values (~4e-9). Ten tests
pass on a clean build.

Plan 6 wires the projector, encoder, decoder, heads, and end-to-end
detect.

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
