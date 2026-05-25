# rt-detr.cpp

C++/ggml inference for [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).

Project layout mirrors [vibevoice.cpp](https://github.com/mudler/vibevoice.cpp).
See `docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md` for the design.

## Status

**Foundation (Plan 1) complete.** The repo builds, the CLI binary runs
end-to-end on an image (without a model — emits empty detections), and
all six tests pass. No model loading or inference yet — see Plans 2-4
under `docs/superpowers/plans/`.

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
