# rfdetr-video

Annotate a video file with rfdetr.cpp detections or segmentation masks, frame
by frame, and write the result as an x264-encoded MP4.

This example is intentionally a thin tool — its purpose is to show how to feed
a frame stream through the rfdetr C API and how to wire the result back into
an external pipeline without dragging libav into librfdetr.

## How it works (and why)

The main `librfdetr.{a,so}` and `rfdetr-cli` binaries deliberately have zero
ffmpeg dependency, neither at build time nor at link time. To keep that
property, this example invokes ffmpeg as a **child process via `popen()`**:

```
  ffmpeg -i input.mp4 -f rawvideo -pix_fmt rgb24 -      (decoder, stdout)
            │
            ▼
  rfdetr_image_from_rgb_buffer(buf, W, H)               (zero-copy wrap)
            │
            ▼
  rfdetr_detect(ctx, img, ...)                          (model forward)
            │
            ▼
  rfdetr_visualize_overlay_mask + draw_box (in place)   (annotation)
            │
            ▼
  ffmpeg -f rawvideo ... -c:v libx264 output.mp4        (encoder, stdin)
```

If you don't want this example, just don't pass `-DRFDETR_BUILD_EXAMPLES=ON`
to CMake — the default build does not include it and does not need ffmpeg.

## Requirements

- `ffmpeg` and `ffprobe` on `$PATH` at **runtime** (`apt install ffmpeg`,
  `brew install ffmpeg`, etc.). They are *not* required at build time.
- An rfdetr GGUF model — any detection or segmentation variant.

## Build

```bash
cmake -B build -DRFDETR_BUILD_EXAMPLES=ON
cmake --build build -j
# binary: build/bin/rfdetr-video
```

The `RFDETR_BUILD_EXAMPLES` option is off by default; a plain
`cmake -B build` does not build `rfdetr-video` and does not add any new
dependencies.

## Usage

```
rfdetr-video --model <gguf> --input <video> --output <out.mp4>
             [--threshold N] [--threads N] [--max-frames N]
             [--fps N] [--ffmpeg PATH] [--ffprobe PATH]
```

### Examples

Detection model on a short clip:

```bash
build/bin/rfdetr-video \
    --model models/rfdetr-base-f16.gguf \
    --input  /tmp/people-demo.mp4 \
    --output /tmp/people_det.mp4 \
    --threshold 0.5 --threads 8
```

Segmentation model (boxes + tinted mask overlay):

```bash
build/bin/rfdetr-video \
    --model models/rfdetr-seg-nano-f16.gguf \
    --input  /tmp/people-demo.mp4 \
    --output /tmp/people_seg.mp4 \
    --threshold 0.5 --threads 8
```

Quick smoke test (first 30 frames only):

```bash
build/bin/rfdetr-video \
    --model models/rfdetr-base-f16.gguf \
    --input  /tmp/people-demo.mp4 \
    --output /tmp/people_first30.mp4 \
    --max-frames 30
```

## Notes & limitations

- **Input paths cannot contain double-quote characters**. The popen() command
  lines wrap them in `"..."` for shell expansion; embedded quotes will break
  the shell parse. Other characters (spaces, parens, unicode) are fine.
- **x264 is hard-coded on the encode side** (`-c:v libx264 -pix_fmt yuv420p
  -crf 23 -movflags +faststart`). Most ffmpeg packages ship libx264; if yours
  doesn't, edit the encoder command in `main.cpp` to use a different codec.
- **Per-frame inference, no batching**. This is the most readable design and
  matches the existing C API; on a CPU build, expect throughput of one frame
  every ~80–500 ms depending on model size, quantization, and thread count.
  See the project root benchmark notes for numbers.
- **No audio** — the input's audio tracks are dropped. Re-mux from the
  original if you need them:

  ```bash
  ffmpeg -i /tmp/people_det.mp4 -i /tmp/people-demo.mp4 \
         -c copy -map 0:v:0 -map 1:a:0? out_with_audio.mp4
  ```

- **Progress** is printed to stderr every 25 frames. stdout is reserved for
  potential future structured output (currently unused).

## Programmatic equivalent

Everything this example does is reachable from the public C header. The key
calls are:

```c
rfdetr_init(...)                                /* once at startup */
rfdetr_image_from_rgb_buffer(rgb, w, h, &st)    /* per frame */
rfdetr_detect(ctx, img, &dp, &dets, &n)         /* per frame */
rfdetr_visualize_overlay_mask(img, dets[i], 0.4)/* per detection (seg only) */
rfdetr_visualize_draw_box(img, dets[i], 3)      /* per detection */
rfdetr_image_rgb_data(img)                      /* annotated pixels back out */
rfdetr_detections_free(dets, n)                 /* per frame */
rfdetr_image_free(img)                          /* per frame */
rfdetr_free(ctx)                                /* once at shutdown */
```

Drop-in for any video pipeline (GStreamer, libav, OpenCV, custom DMA — pick
your poison) by replacing the popen pipes with your decode/encode source.
