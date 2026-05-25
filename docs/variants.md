# RF-DETR Variant Parameters

This document tabulates the per-variant configuration that gets stamped into
GGUF metadata. Plan 2 only ships `base`; Plan 4 adds the others.

## base

| Parameter                | Value |
|--------------------------|-------|
| `image_size`             | 560   |
| `num_queries`            | 300   |
| `num_classes`            | 80 (COCO) |
| `backbone.dim`           | 768   |
| `backbone.depth`         | 12    |
| `backbone.heads`         | 12    |
| `backbone.window_size`   | 14    |
| `backbone.multi_scale_layers` | `[2, 5, 8, 11]` |
| `encoder.layers`         | 3     |
| `encoder.model_dim`      | 256   |
| `encoder.ffn_dim`        | 2048  |
| `encoder.heads`          | 8     |
| `decoder.layers`         | 3     |
| `decoder.model_dim`      | 256   |
| `decoder.ffn_dim`        | 2048  |
| `decoder.heads`          | 8     |

Values reflect the upstream `rfdetr-base` checkpoint at the pinned version of
the `rfdetr` PyPI package. The conversion script reads them dynamically from
the loaded model; the loader treats them as opaque metadata.

## nano / small / medium / large

Filled in by Plan 4 when those variants land.
