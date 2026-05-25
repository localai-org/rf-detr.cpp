# rt-detr.cpp GGUF Conversion

This document is the *contract* between `scripts/convert_rfdetr_to_gguf.py` and
`src/model_loader.cpp`. Both sides reference the same keys and tensor names.
Changes here require updating both sides and bumping `rfdetr.format.version`.

## Format version

Key: `rfdetr.format.version` (string)
Current: `"1"`

Bumped whenever the metadata schema or tensor-name convention changes
incompatibly. The loader refuses any value other than `"1"`.

## Metadata keys

All under the `rfdetr.` namespace.

| Key                          | Type      | Description                                  |
|------------------------------|-----------|----------------------------------------------|
| `rfdetr.format.version`      | string    | `"1"` (see above).                           |
| `rfdetr.variant`             | string    | One of `nano|small|base|medium|large`.       |
| `rfdetr.image_size`          | uint32    | Square input side, e.g. 560.                 |
| `rfdetr.num_queries`         | uint32    | Detection queries (e.g. 300).                |
| `rfdetr.num_classes`         | uint32    | Output classes (e.g. 80 for COCO).           |
| `rfdetr.class_names`         | string[]  | One per class, length `num_classes`.         |
| `rfdetr.preprocess.mean`     | float32[3]| Per-channel mean (ImageNet: 0.485, 0.456, 0.406). |
| `rfdetr.preprocess.std`      | float32[3]| Per-channel std  (ImageNet: 0.229, 0.224, 0.225). |
| `rfdetr.backbone.dim`        | uint32    | Backbone model dim.                          |
| `rfdetr.backbone.depth`      | uint32    | Number of backbone blocks.                   |
| `rfdetr.backbone.heads`      | uint32    | Backbone attention heads.                    |
| `rfdetr.backbone.window_size`| uint32    | Windowed-attention window (0 = global only). |
| `rfdetr.backbone.multi_scale_layers` | uint32[] | Backbone layer indices tapped for multi-scale features. |
| `rfdetr.encoder.layers`      | uint32    | Encoder layer count.                         |
| `rfdetr.encoder.model_dim`   | uint32    | Encoder/decoder hidden dim.                  |
| `rfdetr.encoder.ffn_dim`     | uint32    | Encoder FFN hidden dim.                      |
| `rfdetr.encoder.heads`       | uint32    | Encoder attention heads.                     |
| `rfdetr.decoder.layers`      | uint32    | Decoder layer count.                         |
| `rfdetr.decoder.model_dim`   | uint32    | (Usually equals encoder.model_dim.)          |
| `rfdetr.decoder.ffn_dim`     | uint32    | Decoder FFN hidden dim.                      |
| `rfdetr.decoder.heads`       | uint32    | Decoder attention heads.                     |

## Tensor naming

Tensor names follow the C++ runtime's expectations. The conversion script maps
PyTorch state_dict keys → these names. Names use `.` separators and zero-based
indices.

**Backbone (DINOv2 ViT):**

| Name pattern                              | Shape (example)         | Source PyTorch key (rfdetr-base) |
|-------------------------------------------|-------------------------|-----------------------------------|
| `backbone.patch_embed.weight`             | `[dim, 3, 14, 14]`      | `backbone.patch_embed.proj.weight` |
| `backbone.patch_embed.bias`               | `[dim]`                 | `backbone.patch_embed.proj.bias`   |
| `backbone.pos_embed`                      | `[1, n_tokens, dim]`    | `backbone.pos_embed`               |
| `backbone.cls_token`                      | `[1, 1, dim]`           | `backbone.cls_token`               |
| `backbone.blocks.{i}.norm1.{weight,bias}` | `[dim]` each            | `backbone.blocks.{i}.norm1.{weight,bias}` |
| `backbone.blocks.{i}.attn.qkv.{weight,bias}` | `[3*dim, dim]`, `[3*dim]` | `backbone.blocks.{i}.attn.qkv.{weight,bias}` |
| `backbone.blocks.{i}.attn.proj.{weight,bias}` | `[dim, dim]`, `[dim]` | `backbone.blocks.{i}.attn.proj.{weight,bias}` |
| `backbone.blocks.{i}.norm2.{weight,bias}` | `[dim]` each            | `backbone.blocks.{i}.norm2.{weight,bias}` |
| `backbone.blocks.{i}.mlp.fc1.{weight,bias}` | `[ffn, dim]`, `[ffn]` | `backbone.blocks.{i}.mlp.fc1.{weight,bias}` |
| `backbone.blocks.{i}.mlp.fc2.{weight,bias}` | `[dim, ffn]`, `[dim]` | `backbone.blocks.{i}.mlp.fc2.{weight,bias}` |
| `backbone.norm.{weight,bias}`             | `[dim]` each            | `backbone.norm.{weight,bias}`     |

**Projector (multi-scale):**

| Name pattern                            | Shape (example)        |
|-----------------------------------------|------------------------|
| `projector.level{j}.weight`             | `[model_dim, dim]`     |
| `projector.level{j}.bias`               | `[model_dim]`          |
| `projector.level_embed`                 | `[n_levels, model_dim]`|

**Transformer encoder:**

| Name pattern                                  | Shape (example)                |
|-----------------------------------------------|--------------------------------|
| `encoder.layers.{i}.self_attn.qkv.{w,b}`      | `[3*model_dim, model_dim]`, `[3*model_dim]` |
| `encoder.layers.{i}.self_attn.out.{w,b}`      | `[model_dim, model_dim]`, `[model_dim]`     |
| `encoder.layers.{i}.norm1.{weight,bias}`      | `[model_dim]` each             |
| `encoder.layers.{i}.ffn.fc1.{weight,bias}`    | `[ffn_dim, model_dim]`, `[ffn_dim]`         |
| `encoder.layers.{i}.ffn.fc2.{weight,bias}`    | `[model_dim, ffn_dim]`, `[model_dim]`       |
| `encoder.layers.{i}.norm2.{weight,bias}`      | `[model_dim]` each             |

**Transformer decoder:**

| Name pattern                                  | Shape (example)                |
|-----------------------------------------------|--------------------------------|
| `decoder.queries`                             | `[num_queries, model_dim]`     |
| `decoder.layers.{i}.self_attn.qkv.{w,b}`      | `[3*model_dim, model_dim]`, `[3*model_dim]` |
| `decoder.layers.{i}.self_attn.out.{w,b}`      | `[model_dim, model_dim]`, `[model_dim]`     |
| `decoder.layers.{i}.norm1.{weight,bias}`      | `[model_dim]` each             |
| `decoder.layers.{i}.cross_attn.q.{w,b}`       | `[model_dim, model_dim]`, `[model_dim]`     |
| `decoder.layers.{i}.cross_attn.kv.{w,b}`      | `[2*model_dim, model_dim]`, `[2*model_dim]` |
| `decoder.layers.{i}.cross_attn.out.{w,b}`     | `[model_dim, model_dim]`, `[model_dim]`     |
| `decoder.layers.{i}.norm2.{weight,bias}`      | `[model_dim]` each             |
| `decoder.layers.{i}.ffn.fc1.{weight,bias}`    | `[ffn_dim, model_dim]`, `[ffn_dim]`         |
| `decoder.layers.{i}.ffn.fc2.{weight,bias}`    | `[model_dim, ffn_dim]`, `[model_dim]`       |
| `decoder.layers.{i}.norm3.{weight,bias}`      | `[model_dim]` each             |

**Heads:**

| Name pattern                  | Shape (example)                  |
|-------------------------------|----------------------------------|
| `heads.class.fc.{w,b}`        | `[num_classes, model_dim]`, `[num_classes]` |
| `heads.bbox.fc1.{w,b}`        | `[model_dim, model_dim]`, `[model_dim]`     |
| `heads.bbox.fc2.{w,b}`        | `[model_dim, model_dim]`, `[model_dim]`     |
| `heads.bbox.fc3.{w,b}`        | `[4, model_dim]`, `[4]`                     |

## Discovery workflow

The PyTorch tensor names above are *expected* for the rfdetr-base release at
the version pinned in `scripts/requirements.txt`. Upstream renames are possible.
The conversion script's first task is to enumerate `state_dict().keys()`,
diff against the expected set, and refuse to convert if there are missing or
unmapped keys. Bringing up a new variant or upstream version starts by running
`python scripts/convert_rfdetr_to_gguf.py --dry-run` and reading the diff.
