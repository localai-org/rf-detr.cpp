# rf-detr.cpp GGUF Conversion (rfdetr-base, matching upstream 1.7.0)

This doc is the contract between `scripts/convert_rfdetr_to_gguf.py` and
`src/model_loader.cpp`. Both sides reference the same keys and tensor names.
Bumping the schema requires updating both sides and `rfdetr.format.version`.

## Format version

Key: `rfdetr.format.version` (string)
Current: `"2"`

Plan 7 bumped the schema from `"1"` to `"2"` because the prior schema was
aspirational (an LW-DETR-shaped encoder/decoder with packed QKV) and is
incompatible with the real rfdetr 1.7.0 release weights:

- DINOv2-small backbone with separate Q/K/V projections (not packed).
- Per-block layer-scale gammas (`layer_scale1`, `layer_scale2`).
- No standalone transformer encoder; features go straight from the conv-based
  projector into the two-stage decoder init.
- Two-stage init with 13 parallel head groups (`group_detr=13`).
- Deformable cross-attention in decoder (MSDeformAttn: sampling offsets +
  attention weights, not standard QKV).
- Asymmetric attention heads: self-attn 8 heads, cross-attn 16 heads.
- Shared single class/bbox heads at top level (one Linear + one MLP), not
  per-decoder-layer copies.

The loader rejects any value other than `"2"`.

## Metadata keys

All keys live under the `rfdetr.` namespace.

| Key                                          | Type        | Description / example |
|----------------------------------------------|-------------|-----------------------|
| `rfdetr.format.version`                      | string      | `"2"` (see above)     |
| `rfdetr.variant`                             | string      | `"base"` (only one supported for now) |
| `rfdetr.image_size`                          | uint32      | Square input side; `560` for rfdetr-base |
| `rfdetr.patch_size`                          | uint32      | DINOv2 patch side; `14` |
| `rfdetr.num_queries`                         | uint32      | `300` (group 0 of 13 active at inference) |
| `rfdetr.group_detr`                          | uint32      | `13` (training-time groups; only group 0 read at inference) |
| `rfdetr.num_classes`                         | uint32      | `91` (raw logit width; only 80 valid COCO IDs) |
| `rfdetr.class_names`                         | string[91]  | One entry per logit index. Unused IDs are `""`; the 80 COCO names sit at their COCO-spec positions. |
| `rfdetr.preprocess.mean`                     | float32[3]  | `[0.485, 0.456, 0.406]` (ImageNet) |
| `rfdetr.preprocess.std`                      | float32[3]  | `[0.229, 0.224, 0.225]` (ImageNet) |
| `rfdetr.backbone.dim`                        | uint32      | `384` |
| `rfdetr.backbone.depth`                      | uint32      | `12` |
| `rfdetr.backbone.heads`                      | uint32      | `6` |
| `rfdetr.backbone.ffn_dim`                    | uint32      | `1536` |
| `rfdetr.backbone.num_windows`                | uint32      | `4` (windows per side in windowed blocks) |
| `rfdetr.backbone.global_attn_indices`        | int32[4]    | `[2, 5, 8, 11]` (blocks that do global attention; remaining do windowed) |
| `rfdetr.backbone.out_feature_indices`        | int32[4]    | `[2, 5, 8, 11]` (block indices tapped for projector input; concatenated channelwise) |
| `rfdetr.backbone.pos_embed_train_size`       | uint32      | `37` (side length of the stored positional grid; runtime image of 560/14=40 patches is bilinearly interpolated from 37x37+1) |
| `rfdetr.projector.in_dim`                    | uint32      | `1536` (= 4 x backbone.dim) |

| `rfdetr.projector.out_dim`                   | uint32      | `256` |
| `rfdetr.projector.bottleneck_dim`            | uint32      | `128` |
| `rfdetr.projector.n_bottlenecks`             | uint32      | `3` |
| `rfdetr.decoder.layers`                      | uint32      | `3` |
| `rfdetr.decoder.model_dim`                   | uint32      | `256` |
| `rfdetr.decoder.ffn_dim`                     | uint32      | `2048` |
| `rfdetr.decoder.self_attn_heads`             | uint32      | `8` |
| `rfdetr.decoder.cross_attn_heads`            | uint32      | `16` (asymmetric: deformable attn uses more heads) |
| `rfdetr.decoder.cross_attn_n_levels`         | uint32      | `1` (rfdetr-base is single-scale, P4 only) |
| `rfdetr.decoder.cross_attn_n_points`         | uint32      | `2` (sampling points per head per level) |
| `rfdetr.two_stage.n_groups`                  | uint32      | `13` (= `group_detr`; one enc_output set per group) |

## Conventions

### Shape order (ggml `ne` vs PyTorch)

ggml stores tensors **column-major** with `ne[0]` as the fastest-varying axis.
PyTorch stores **row-major** with the last axis fastest-varying. For the same
linear layer:

- PyTorch `Linear(in, out).weight` has shape `(out, in)`
- ggml `ne` for the same tensor is `(in, out)` (axes reversed)

For 4D conv weights:

- PyTorch `Conv2d` weight has shape `(out, in, kh, kw)`
- ggml `ne` is `(kw, kh, in, out)`

The tables below show `ne` (the ggml view, what the converter writes and the
loader reads).

### Slicing convention for queries

Upstream stores 3900-row tensors for `query_feat` and `refpoint_embed`
(`= num_queries x group_detr = 300 x 13`). Only group 0 (the first 300 rows)
is used at inference. The converter slices to group 0 only; storing all
3900 rows would just bloat the GGUF.

If you later want to do training-style decoding with all 13 groups, you'd
re-emit the unsliced tensors (and bump format version).

### Layer-scale gammas

DINOv2's per-block layer scales are stored as flat 1D `(dim,)` tensors. The
forward pass multiplies elementwise *before* the residual add:

    h = h + layer_scale * attn(norm1(h))
    h = h + layer_scale * mlp (norm2(h))

### Projector "BatchNorm" naming is a misnomer

Upstream state_dict keys say `.bn.weight` / `.bn.bias`, but the actual tensors
are 1-D `(channels,)` **LayerNorm** parameters (no `running_mean` /
`running_var`). The converter renames them to `.norm.weight` / `.norm.bias`
to keep the C++ side honest. ConvX = `Conv2d -> LayerNorm -> SiLU`.

## Tensor naming

GGUF names flatten the upstream hierarchy. Indices are zero-based.

### Backbone (DINOv2-small windowed): 222 tensors for rfdetr-base

| GGUF name | ggml ne | PyTorch source key |
|-----------|---------|--------------------|
| `backbone.patch_embed.weight`               | `(14, 14, 3, 384)` | `backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight` |
| `backbone.patch_embed.bias`                 | `(384,)`           | `backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.bias` |
| `backbone.cls_token`                        | `(384,)`           | `backbone.0.encoder.encoder.embeddings.cls_token` (squeezed) |
| `backbone.pos_embed`                        | `(384, 1370)`      | `backbone.0.encoder.encoder.embeddings.position_embeddings` (squeezed to 2D) |
| `backbone.blocks.{i}.norm1.weight`          | `(384,)`           | `…encoder.layer.{i}.norm1.weight` |
| `backbone.blocks.{i}.norm1.bias`            | `(384,)`           | `…encoder.layer.{i}.norm1.bias` |
| `backbone.blocks.{i}.attn.q.weight`         | `(384, 384)`       | `…encoder.layer.{i}.attention.attention.query.weight` |
| `backbone.blocks.{i}.attn.q.bias`           | `(384,)`           | `…encoder.layer.{i}.attention.attention.query.bias` |
| `backbone.blocks.{i}.attn.k.weight`         | `(384, 384)`       | `…encoder.layer.{i}.attention.attention.key.weight` |
| `backbone.blocks.{i}.attn.k.bias`           | `(384,)`           | `…encoder.layer.{i}.attention.attention.key.bias` |
| `backbone.blocks.{i}.attn.v.weight`         | `(384, 384)`       | `…encoder.layer.{i}.attention.attention.value.weight` |
| `backbone.blocks.{i}.attn.v.bias`           | `(384,)`           | `…encoder.layer.{i}.attention.attention.value.bias` |
| `backbone.blocks.{i}.attn.proj.weight`      | `(384, 384)`       | `…encoder.layer.{i}.attention.output.dense.weight` |
| `backbone.blocks.{i}.attn.proj.bias`        | `(384,)`           | `…encoder.layer.{i}.attention.output.dense.bias` |
| `backbone.blocks.{i}.layer_scale1`          | `(384,)`           | `…encoder.layer.{i}.layer_scale1.lambda1` |
| `backbone.blocks.{i}.norm2.weight`          | `(384,)`           | `…encoder.layer.{i}.norm2.weight` |
| `backbone.blocks.{i}.norm2.bias`            | `(384,)`           | `…encoder.layer.{i}.norm2.bias` |
| `backbone.blocks.{i}.mlp.fc1.weight`        | `(384, 1536)`      | `…encoder.layer.{i}.mlp.fc1.weight` |
| `backbone.blocks.{i}.mlp.fc1.bias`          | `(1536,)`          | `…encoder.layer.{i}.mlp.fc1.bias` |
| `backbone.blocks.{i}.mlp.fc2.weight`        | `(1536, 384)`      | `…encoder.layer.{i}.mlp.fc2.weight` |
| `backbone.blocks.{i}.mlp.fc2.bias`          | `(384,)`           | `…encoder.layer.{i}.mlp.fc2.bias` |
| `backbone.blocks.{i}.layer_scale2`          | `(384,)`           | `…encoder.layer.{i}.layer_scale2.lambda1` |
| `backbone.norm.weight`                      | `(384,)`           | `backbone.0.encoder.encoder.layernorm.weight` |
| `backbone.norm.bias`                        | `(384,)`           | `backbone.0.encoder.encoder.layernorm.bias` |

Per-block: 18 tensors. Total backbone: `4 + 18 x 12 + 2 = 222`.

The upstream `mask_token` (used at training time for masked-image-modeling)
is dropped by the converter; inference doesn't need it.

### Projector (single-scale C2f, P4 only): 26 tensors

The projector is the **conv-based MultiScaleProjector** (`n_levels=1`). It
takes the concatenation of the 4 backbone out_features channel-wise
(`4 x 384 = 1536` channels) and emits a single 256-channel feature map.

| GGUF name | ggml ne | PyTorch source key |
|-----------|---------|--------------------|
| `projector.cv1.conv.weight`                 | `(1, 1, 1536, 256)` | `backbone.0.projector.stages.0.0.cv1.conv.weight` |
| `projector.cv1.norm.weight`                 | `(256,)`            | `…stages.0.0.cv1.bn.weight` (renamed; actually LN) |
| `projector.cv1.norm.bias`                   | `(256,)`            | `…stages.0.0.cv1.bn.bias` |
| `projector.cv2.conv.weight`                 | `(1, 1, 640, 256)`  | `…stages.0.0.cv2.conv.weight` (640 = 256 + 128x3 after C2f split+bottleneck concat) |
| `projector.cv2.norm.weight`                 | `(256,)`            | `…stages.0.0.cv2.bn.weight` |
| `projector.cv2.norm.bias`                   | `(256,)`            | `…stages.0.0.cv2.bn.bias` |
| `projector.bottleneck.{j}.cv1.conv.weight`  | `(3, 3, 128, 128)`  | `…stages.0.0.m.{j}.cv1.conv.weight` (j ∈ 0..2) |
| `projector.bottleneck.{j}.cv1.norm.weight`  | `(128,)`            | `…stages.0.0.m.{j}.cv1.bn.weight` |
| `projector.bottleneck.{j}.cv1.norm.bias`    | `(128,)`            | `…stages.0.0.m.{j}.cv1.bn.bias` |
| `projector.bottleneck.{j}.cv2.conv.weight`  | `(3, 3, 128, 128)`  | `…stages.0.0.m.{j}.cv2.conv.weight` |
| `projector.bottleneck.{j}.cv2.norm.weight`  | `(128,)`            | `…stages.0.0.m.{j}.cv2.bn.weight` |
| `projector.bottleneck.{j}.cv2.norm.bias`    | `(128,)`            | `…stages.0.0.m.{j}.cv2.bn.bias` |
| `projector.final_norm.weight`               | `(256,)`            | `backbone.0.projector.stages.0.1.weight` (post-C2f LayerNorm) |
| `projector.final_norm.bias`                 | `(256,)`            | `backbone.0.projector.stages.0.1.bias` |

Total: `6 + 3 x 6 + 2 = 26`.

### Two-stage initialization (replaces the standalone encoder): 156 tensors

`group_detr = 13` parallel groups. Each group has its own `enc_output` Linear,
LayerNorm, class head, and 3-layer bbox-MLP. Only group 0 is exercised at
inference, but all are stored to support fine-tuning workflows that re-use
upstream init.

| GGUF name | ggml ne | PyTorch source key |
|-----------|---------|--------------------|
| `two_stage.enc_output.{g}.weight`                          | `(256, 256)` | `transformer.enc_output.{g}.weight` (g ∈ 0..12) |
| `two_stage.enc_output.{g}.bias`                            | `(256,)`     | `transformer.enc_output.{g}.bias` |
| `two_stage.enc_output_norm.{g}.weight`                     | `(256,)`     | `transformer.enc_output_norm.{g}.weight` |
| `two_stage.enc_output_norm.{g}.bias`                       | `(256,)`     | `transformer.enc_output_norm.{g}.bias` |
| `two_stage.enc_out_class_embed.{g}.weight`                 | `(256, 91)`  | `transformer.enc_out_class_embed.{g}.weight` |
| `two_stage.enc_out_class_embed.{g}.bias`                   | `(91,)`      | `transformer.enc_out_class_embed.{g}.bias` |
| `two_stage.enc_out_bbox_embed.{g}.layers.0.weight`         | `(256, 256)` | `transformer.enc_out_bbox_embed.{g}.layers.0.weight` |
| `two_stage.enc_out_bbox_embed.{g}.layers.0.bias`           | `(256,)`     | `transformer.enc_out_bbox_embed.{g}.layers.0.bias` |
| `two_stage.enc_out_bbox_embed.{g}.layers.1.weight`         | `(256, 256)` | `transformer.enc_out_bbox_embed.{g}.layers.1.weight` |
| `two_stage.enc_out_bbox_embed.{g}.layers.1.bias`           | `(256,)`     | `transformer.enc_out_bbox_embed.{g}.layers.1.bias` |
| `two_stage.enc_out_bbox_embed.{g}.layers.2.weight`         | `(256, 4)`   | `transformer.enc_out_bbox_embed.{g}.layers.2.weight` |
| `two_stage.enc_out_bbox_embed.{g}.layers.2.bias`           | `(4,)`       | `transformer.enc_out_bbox_embed.{g}.layers.2.bias` |

Per group: 12 tensors. Total: `13 x 12 = 156`.

### Decoder: 74 tensors

3 layers of `TransformerDecoderLayer`. Each layer has:

- Self-attention: standard `nn.MultiheadAttention` with packed QKV
  (`in_proj_weight (768, 256)`, `in_proj_bias (768,)`); 8 heads.
- Cross-attention: `MSDeformAttn` (deformable, single-scale): 16 heads, 1
  level, 2 sampling points per head:
  - `sampling_offsets: Linear(256 -> 64)` (`64 = 2 x heads x n_levels x n_points = 2 x 16 x 1 x 2`)
  - `attention_weights: Linear(256 -> 32)` (`32 = heads x n_levels x n_points = 16 x 1 x 2`)
  - `value_proj: Linear(256 -> 256)`
  - `output_proj: Linear(256 -> 256)`
- FFN: `linear1: 256 -> 2048`, `linear2: 2048 -> 256`.
- 3 LayerNorms: `norm1` (post self-attn), `norm2` (post cross-attn), `norm3`
  (post FFN).

Plus shared decoder-level state:

- `decoder.norm`: final LayerNorm(256).
- `decoder.ref_point_head`: 2-layer MLP `(512 -> 256 -> 256)`, projects
  sinusoidally embedded 4D reference points (cx, cy, w, h x 128 freq -> 512)
  down to 256.
- `decoder.queries.feat`: group-0 slice of `query_feat` (300, 256).
- `decoder.queries.refpoints`: group-0 slice of `refpoint_embed` (300, 4).

| GGUF name | ggml ne | PyTorch source key |
|-----------|---------|--------------------|
| `decoder.queries.feat`                                     | `(256, 300)`  | `query_feat.weight[:300]` (top-level Embedding, sliced) |
| `decoder.queries.refpoints`                                | `(4, 300)`    | `refpoint_embed.weight[:300]` (sliced) |
| `decoder.ref_point_head.layers.0.weight`                   | `(512, 256)`  | `transformer.decoder.ref_point_head.layers.0.weight` |
| `decoder.ref_point_head.layers.0.bias`                     | `(256,)`      | `transformer.decoder.ref_point_head.layers.0.bias` |
| `decoder.ref_point_head.layers.1.weight`                   | `(256, 256)`  | `transformer.decoder.ref_point_head.layers.1.weight` |
| `decoder.ref_point_head.layers.1.bias`                     | `(256,)`      | `transformer.decoder.ref_point_head.layers.1.bias` |
| `decoder.layers.{i}.self_attn.in_proj.weight`              | `(256, 768)`  | `transformer.decoder.layers.{i}.self_attn.in_proj_weight` |
| `decoder.layers.{i}.self_attn.in_proj.bias`                | `(768,)`      | `…layers.{i}.self_attn.in_proj_bias` |
| `decoder.layers.{i}.self_attn.out_proj.weight`             | `(256, 256)`  | `…layers.{i}.self_attn.out_proj.weight` |
| `decoder.layers.{i}.self_attn.out_proj.bias`               | `(256,)`      | `…layers.{i}.self_attn.out_proj.bias` |
| `decoder.layers.{i}.norm1.weight`                          | `(256,)`      | `…layers.{i}.norm1.weight` |
| `decoder.layers.{i}.norm1.bias`                            | `(256,)`      | `…layers.{i}.norm1.bias` |
| `decoder.layers.{i}.cross_attn.sampling_offsets.weight`    | `(256, 64)`   | `…layers.{i}.cross_attn.sampling_offsets.weight` |
| `decoder.layers.{i}.cross_attn.sampling_offsets.bias`      | `(64,)`       | `…layers.{i}.cross_attn.sampling_offsets.bias` |
| `decoder.layers.{i}.cross_attn.attention_weights.weight`   | `(256, 32)`   | `…layers.{i}.cross_attn.attention_weights.weight` |
| `decoder.layers.{i}.cross_attn.attention_weights.bias`     | `(32,)`       | `…layers.{i}.cross_attn.attention_weights.bias` |
| `decoder.layers.{i}.cross_attn.value_proj.weight`          | `(256, 256)`  | `…layers.{i}.cross_attn.value_proj.weight` |
| `decoder.layers.{i}.cross_attn.value_proj.bias`            | `(256,)`      | `…layers.{i}.cross_attn.value_proj.bias` |
| `decoder.layers.{i}.cross_attn.output_proj.weight`         | `(256, 256)`  | `…layers.{i}.cross_attn.output_proj.weight` |
| `decoder.layers.{i}.cross_attn.output_proj.bias`           | `(256,)`      | `…layers.{i}.cross_attn.output_proj.bias` |
| `decoder.layers.{i}.norm2.weight`                          | `(256,)`      | `…layers.{i}.norm2.weight` |
| `decoder.layers.{i}.norm2.bias`                            | `(256,)`      | `…layers.{i}.norm2.bias` |
| `decoder.layers.{i}.linear1.weight`                        | `(256, 2048)` | `…layers.{i}.linear1.weight` |
| `decoder.layers.{i}.linear1.bias`                          | `(2048,)`     | `…layers.{i}.linear1.bias` |
| `decoder.layers.{i}.linear2.weight`                        | `(2048, 256)` | `…layers.{i}.linear2.weight` |
| `decoder.layers.{i}.linear2.bias`                          | `(256,)`      | `…layers.{i}.linear2.bias` |
| `decoder.layers.{i}.norm3.weight`                          | `(256,)`      | `…layers.{i}.norm3.weight` |
| `decoder.layers.{i}.norm3.bias`                            | `(256,)`      | `…layers.{i}.norm3.bias` |
| `decoder.norm.weight`                                      | `(256,)`      | `transformer.decoder.norm.weight` |
| `decoder.norm.bias`                                        | `(256,)`      | `transformer.decoder.norm.bias` |

Per decoder layer: 22 tensors. Total decoder: `2 + 4 + 22 x 3 + 2 = 74`.

### Heads: 8 tensors

Both heads are shared single instances at the top of the model (upstream
calls these `inner.class_embed` and `inner.bbox_embed`). There is no
per-decoder-layer head.

| GGUF name | ggml ne | PyTorch source key |
|-----------|---------|--------------------|
| `heads.class_embed.weight`               | `(256, 91)`  | `class_embed.weight` |
| `heads.class_embed.bias`                 | `(91,)`      | `class_embed.bias` |
| `heads.bbox_embed.layers.0.weight`       | `(256, 256)` | `bbox_embed.layers.0.weight` |
| `heads.bbox_embed.layers.0.bias`         | `(256,)`     | `bbox_embed.layers.0.bias` |
| `heads.bbox_embed.layers.1.weight`       | `(256, 256)` | `bbox_embed.layers.1.weight` |
| `heads.bbox_embed.layers.1.bias`         | `(256,)`     | `bbox_embed.layers.1.bias` |
| `heads.bbox_embed.layers.2.weight`       | `(256, 4)`   | `bbox_embed.layers.2.weight` |
| `heads.bbox_embed.layers.2.bias`         | `(4,)`       | `bbox_embed.layers.2.bias` |

Total heads: 8.

### Tensor count summary (rfdetr-base)

| Section          | Count |
|------------------|-------|
| Backbone         | 222   |
| Projector        | 26    |
| Two-stage init   | 156   |
| Decoder          | 74    |
| Heads            | 8     |
| **Total**        | **486** |

Upstream `state_dict` has 487 tensors; the +1 is `mask_token` (training only,
dropped by converter).

## Per-variant notes

Only `base` is supported for now. `nano`, `small`, `medium`, `large` are
deferred. They reuse the same schema but with different `backbone.dim`,
`backbone.depth`, `backbone.heads`, `projector.in_dim`, `projector.out_dim`,
and (potentially) `decoder.layers`. Each variant must be introspected to
confirm whether single-scale (P4 only) holds.

## Discovery workflow

The PyTorch keys above are valid for the rfdetr-base release at the version
pinned in `scripts/requirements.txt` (rfdetr 1.7.0). Upstream renames are
possible. The conversion script's first task is to enumerate
`state_dict().keys()`, diff against the expected set, and refuse to convert
on any missing or unmapped key. Bringing up a new variant or upstream
version starts with `python scripts/convert_rfdetr_to_gguf.py --dry-run` and
reading the diff.
