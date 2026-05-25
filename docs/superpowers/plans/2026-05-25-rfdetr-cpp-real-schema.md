# rt-detr.cpp Real Schema + Conversion Implementation Plan (Plan 7 of 15)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align this project's GGUF schema and conversion path with the actual rfdetr-base architecture (rfdetr 1.7.0). Plan 2 Task 3's probe (commit `b4dce05`) revealed 74 structurally-unbridgeable tensors between our aspirational schema and real upstream. This plan rips the bandaid: rewrites `docs/conversion.md`, `Config`, `expected_tensor_names`, the fixture generator's variant config, and implements the conversion script body so that `rfdetr-cli info models/rfdetr-base-f32.gguf` succeeds.

End state: a real converted GGUF that the loader can introspect. Plans 8-12 will rewrite the numerical modules (backbone, projector, encoder→two-stage, decoder w/ deformable attention, heads) to actually run inference on it. Plan 13 adds the torch parity baseline. Plan 15 is the real E2E demo.

**Architectural reality** (from Plan 2 Task 3 probe — see commit `b4dce05` docstring):

- DINOv2-**small** backbone: `dim=384`, `ffn=1536`, 12 blocks, layer-scale gammas (`lambda1` per block, two scales per block)
- Backbone uses **separate Q/K/V** (`attention.attention.{query,key,value}.{weight,bias}`), NOT packed `qkv`
- Backbone is `dinov2_windowed_small` — windowed attention specifics
- Backbone norm: `backbone.0.encoder.encoder.layernorm.{w,b}`
- Single **conv-based `MultiScaleProjector`** (not 4 linear levels)
- **No standalone encoder** — replaced by two-stage initialization: `enc_output` (ModuleList of 13 Linears) + `enc_output_norm` (13 LayerNorms) + `enc_out_class_embed` (13) + `enc_out_bbox_embed` (13)
- Decoder self-attn uses PyTorch `nn.MultiheadAttention`-style `in_proj_weight/bias` + `out_proj`
- Decoder cross-attn is **deformable**: `sampling_offsets`, `attention_weights`, `value_proj`, `output_proj` (different algorithm than what we have)
- Heads: `class_embed.{w,b}` (single Linear, 91 classes including background) + `bbox_embed.layers.{0,1,2}.{w,b}` (MLP) + 13× `enc_out_*` aux heads
- Queries: `refpoint_embed.weight = (3900, 4)` — 300 active queries × `group_detr=13`
- 91 classes (not 80)
- `image_size=560`

**Architecture:** This plan does NOT rewrite the numerical math. It only changes the schema, the GGUF tensor set, the loader's expected names, and the conversion script. After Plan 7 lands, **the parity test and end-to-end tests will FAIL** because the C++ numerical modules (`dinov2.cpp`, `encoder.cpp`, `decoder.cpp`, `projector.cpp`, `heads.cpp`) still expect the old shapes. This is a deliberate transitional state — Plans 8-12 fix each module.

**Tech Stack:** Same. `.venv/` already exists from Plan 2 Task 3's probe with `rfdetr==1.7.0` + `torch==2.5.1` installed.

---

## Scope decisions

- **Rip-the-bandaid migration.** Schema and conversion change in Plan 7; numerical modules fixed in Plans 8-12. During Plans 8-12 some tests will be RED — documented as expected. The CI bar relaxes to "loader works" for Plan 7's commit, "loader + backbone parity" after Plan 8, etc.
- **Drop tests that test the obsolete architecture.** `test_parity_full_forward.cpp` exercises a forward pass that no longer matches anything. Plans 8-13 incrementally rebuild it. For Plan 7, the test is **disabled** (registered with a skip via `if(FALSE)` in CMakeLists.txt, with a comment pointing to Plan 8+ for re-enablement).
- **Keep tests that test infrastructure**: `test_common`, `test_image_io`, `test_postprocess`, `test_visualize`, `test_cli_smoke`, `test_model_loader`, `test_capi` (init/free path), `test_capi_flat` (init/unload). These don't depend on the numerical modules and stay green.
- **Synthesized fixture continues to work** but with the new tensor set and new variant config. `gen_model_gguf` produces a fixture that the new loader accepts.
- **`rfdetr-cli detect` will fail at runtime** during Plans 7-11 because the numerical modules expect the old shapes. The CLI's `cmd_detect` becomes effectively a smoke test of init/free. End-to-end detection comes back online in Plan 12.
- **Variant focus stays `base`.** Plan 16+ adds nano/small/medium/large after the redesign is verified.

---

## File map (created or modified in this plan)

```
rt-detr.cpp/
├── docs/
│   └── conversion.md                   # REWRITE — real rfdetr schema
├── scripts/
│   ├── convert_rfdetr_to_gguf.py       # MODIFY — implement real conversion body
│   └── inspect_rfdetr.py               # NEW — detailed introspection helper (not committed; for dev)
├── src/
│   ├── model_loader.hpp                # MODIFY — Config struct (add layer_scale, group_detr,
│   │                                   #          enc_output count, etc.; remove standalone encoder)
│   └── model_loader.cpp                # MODIFY — read new metadata keys; expected_tensor_names rewritten
├── tests/
│   ├── CMakeLists.txt                  # MODIFY — disable test_parity_full_forward
│   ├── fixtures/gen_model_gguf.cpp     # REWRITE — emit the real tensor set + new variant config
│   └── test_model_loader.cpp           # MODIFY — assertions match new config values
└── README.md                           # MODIFY — Plan 7 status (transitional)
```

---

### Task 1: Deep architectural introspection

Land a probe script that produces an exhaustive, structured dump of rfdetr-base's architecture. Output drives all subsequent tasks.

**Files:**
- Create (NOT committed — under `/tmp/` for dev only): a Python script that introspects rfdetr-base
- Capture output into a temporary file the implementer reads to write `docs/conversion.md`

### Step 1: Use the existing venv from Task 67

```bash
cd /home/mudler/_git/rt-detr.cpp
source .venv/bin/activate
python3 -c "import rfdetr; print(rfdetr.__version__)"
```

Should print `1.7.0` (or whatever was installed). If `.venv/` is missing, recreate per Plan 2 Task 3.

### Step 2: Write `/tmp/inspect_rfdetr.py` (not committed)

```python
"""Detailed introspection of rfdetr-base. Used as input to Plan 7's
docs/conversion.md rewrite."""

import torch
from rfdetr import RFDETRBase

m = RFDETRBase()
inner = m.model.model  # discovered access path; type = LWDETR

# 1. Top-level structure
print("=" * 80)
print("TOP-LEVEL nn.Modules under inner:")
for name, mod in inner.named_children():
    print(f"  {name}: {type(mod).__name__}")

# 2. Full state_dict listing — grouped by top-level prefix
print("\n" + "=" * 80)
print("STATE_DICT BY PREFIX:")
sd = inner.state_dict()
from collections import defaultdict
by_pfx = defaultdict(list)
for k, v in sd.items():
    by_pfx[k.split(".")[0]].append((k, tuple(v.shape), str(v.dtype)))

for pfx in sorted(by_pfx.keys()):
    entries = by_pfx[pfx]
    print(f"\n--- prefix: {pfx} ({len(entries)} tensors) ---")
    for k, shape, dt in entries[:50]:  # cap at 50 per prefix
        print(f"  {k} {shape} {dt}")
    if len(entries) > 50:
        print(f"  ... and {len(entries) - 50} more")

# 3. Backbone deep-dive
print("\n" + "=" * 80)
print("BACKBONE DEEP-DIVE (rfdetr's DINOv2):")
bb = inner.backbone[0]  # backbone is a Joiner; [0] is the actual backbone
print(f"  type: {type(bb).__name__}")
print(f"  attributes: {[a for a in dir(bb) if not a.startswith('_')][:30]}")
if hasattr(bb, "encoder"):
    print(f"  bb.encoder type: {type(bb.encoder).__name__}")
    if hasattr(bb.encoder, "encoder"):
        print(f"  bb.encoder.encoder type: {type(bb.encoder.encoder).__name__}")

# Configuration values (dim, depth, etc.)
for attr in ["config", "embed_dim", "patch_embeddings", "layer", "layers"]:
    if hasattr(bb, attr):
        v = getattr(bb, attr)
        print(f"  bb.{attr}: {type(v).__name__}")
        if hasattr(v, "embed_dim"): print(f"    .embed_dim = {v.embed_dim}")
        if hasattr(v, "num_attention_heads"): print(f"    .num_attention_heads = {v.num_attention_heads}")
        if hasattr(v, "num_hidden_layers"): print(f"    .num_hidden_layers = {v.num_hidden_layers}")

# 4. Projector deep-dive
print("\n" + "=" * 80)
print("PROJECTOR DEEP-DIVE:")
proj = getattr(inner, "input_proj", None) or getattr(inner, "projector", None)
print(f"  type: {type(proj).__name__}")
print(f"  module structure:")
def dump_module(mod, indent=2):
    for name, child in mod.named_children():
        print(f"{' '*indent}{name}: {type(child).__name__}")
        if list(child.named_children()):
            dump_module(child, indent + 2)
if proj is not None:
    dump_module(proj)

# 5. Transformer (encoder/decoder/two-stage) deep-dive
print("\n" + "=" * 80)
print("TRANSFORMER DEEP-DIVE:")
tr = inner.transformer
print(f"  type: {type(tr).__name__}")
print(f"  attributes: {[a for a in dir(tr) if not a.startswith('_')][:30]}")
print(f"  named_children:")
dump_module(tr)
for a in ["enc_output", "enc_output_norm", "enc_out_class_embed", "enc_out_bbox_embed",
          "encoder", "decoder", "num_queries", "group_detr", "two_stage"]:
    if hasattr(tr, a):
        v = getattr(tr, a)
        if isinstance(v, torch.nn.Module):
            print(f"  tr.{a}: nn.Module {type(v).__name__}")
        elif isinstance(v, (int, float, bool, str)):
            print(f"  tr.{a} = {v}")
        else:
            print(f"  tr.{a}: {type(v).__name__}")

# 6. Decoder layer deep-dive (first layer only)
if hasattr(tr, "decoder") and hasattr(tr.decoder, "layers"):
    print("\n" + "=" * 80)
    print("DECODER LAYER 0 DEEP-DIVE:")
    l0 = tr.decoder.layers[0]
    print(f"  type: {type(l0).__name__}")
    dump_module(l0)
    # Check for deformable-attention-specific attrs
    for attr in ["cross_attn", "self_attn", "n_levels", "n_points", "n_heads",
                 "sampling_offsets", "attention_weights"]:
        if hasattr(l0, attr) or (hasattr(l0, "cross_attn") and hasattr(l0.cross_attn, attr)):
            v = getattr(l0, attr, None) or getattr(l0.cross_attn, attr, None)
            print(f"  l0.{attr if hasattr(l0, attr) else 'cross_attn.' + attr}: {v}")

# 7. Heads deep-dive
print("\n" + "=" * 80)
print("HEADS DEEP-DIVE:")
for attr in ["class_embed", "bbox_embed", "refpoint_embed", "tgt_embed"]:
    if hasattr(inner, attr):
        v = getattr(inner, attr)
        print(f"  inner.{attr}: {type(v).__name__}")
        if isinstance(v, torch.nn.Module):
            dump_module(v)
```

Run:

```bash
python3 /tmp/inspect_rfdetr.py > /tmp/rfdetr_arch.txt 2>&1
wc -l /tmp/rfdetr_arch.txt
head -100 /tmp/rfdetr_arch.txt
```

Read the full output. This is the authoritative source for everything that follows.

### Step 3: Commit-prep — sanity-only

No commit yet. Read /tmp/rfdetr_arch.txt thoroughly. Identify:
- Every metadata constant needed (dim, ffn, n_heads, depth, n_queries, n_classes, group_detr, n_levels, n_points, image_size, patch_size)
- Every tensor name's exact path in the state_dict
- Shapes (which need to match in the GGUF metadata for the loader to validate)

---

### Task 2: Rewrite `docs/conversion.md`

Based on Task 1's introspection, write the new schema. This becomes the contract for everything downstream.

**Files:**
- Modify: `docs/conversion.md`

### Step 1: New top-of-file structure

The doc must cover:

```markdown
# rt-detr.cpp GGUF Conversion (rfdetr-base, matching upstream 1.7.0)

## Format version

`rfdetr.format.version = "2"` — Plan 7 bumped from "1" to "2" because the
tensor set is incompatible with the prior schema. The loader refuses other
values.

## Metadata keys

| Key | Type | Value (base) |
|-----|------|--------------|
| `rfdetr.format.version` | string | "2" |
| `rfdetr.variant` | string | "base" |
| `rfdetr.image_size` | uint32 | 560 |
| `rfdetr.patch_size` | uint32 | 14 |
| `rfdetr.num_queries` | uint32 | 300 |
| `rfdetr.group_detr` | uint32 | 13 |
| `rfdetr.num_classes` | uint32 | 91 |
| `rfdetr.class_names` | string[91] | COCO + "N/A" placeholder for 0/background |
| `rfdetr.preprocess.mean` | float32[3] | from rfdetr defaults |
| `rfdetr.preprocess.std` | float32[3] | from rfdetr defaults |
| `rfdetr.backbone.dim` | uint32 | 384 |
| `rfdetr.backbone.depth` | uint32 | 12 |
| `rfdetr.backbone.heads` | uint32 | 6 (verify from probe) |
| `rfdetr.backbone.ffn_dim` | uint32 | 1536 |
| `rfdetr.backbone.window_size` | uint32 | (from probe) |
| `rfdetr.backbone.global_attn_indices` | uint32[] | (from probe — which blocks are global) |
| `rfdetr.decoder.layers` | uint32 | (from probe; rfdetr-base typically 6) |
| `rfdetr.decoder.model_dim` | uint32 | 256 |
| `rfdetr.decoder.heads` | uint32 | 8 |
| `rfdetr.decoder.ffn_dim` | uint32 | 1024 (verify) |
| `rfdetr.decoder.n_levels` | uint32 | (from probe — deformable attention scales) |
| `rfdetr.decoder.n_points` | uint32 | (from probe — deformable sampling points per head) |
| `rfdetr.two_stage.n_enc` | uint32 | 13 |

(No `rfdetr.encoder.*` keys — there is no standalone encoder.)

## Tensor naming

### Backbone (DINOv2-small windowed)

| GGUF name | PyTorch source | ggml ne |
|-----------|----------------|---------|
| `backbone.patch_embed.weight` | `backbone.0.encoder.embeddings.patch_embeddings.projection.weight` | `(14, 14, 3, 384)` |
| `backbone.patch_embed.bias` | `backbone.0.encoder.embeddings.patch_embeddings.projection.bias` | `(384,)` |
| `backbone.cls_token` | `backbone.0.encoder.embeddings.cls_token` | `(384,)` |
| `backbone.pos_embed` | `backbone.0.encoder.embeddings.position_embeddings` | `(384, 1370)` |
| `backbone.blocks.{i}.norm1.weight` | `backbone.0.encoder.encoder.layer.{i}.norm1.weight` | `(384,)` |
| `backbone.blocks.{i}.norm1.bias` | `backbone.0.encoder.encoder.layer.{i}.norm1.bias` | `(384,)` |
| `backbone.blocks.{i}.attn.q.weight` | `backbone.0.encoder.encoder.layer.{i}.attention.attention.query.weight` | `(384, 384)` |
| `backbone.blocks.{i}.attn.q.bias` | (same).query.bias | `(384,)` |
| `backbone.blocks.{i}.attn.k.weight` | (same).key.weight | `(384, 384)` |
| `backbone.blocks.{i}.attn.k.bias` | (same).key.bias | `(384,)` |
| `backbone.blocks.{i}.attn.v.weight` | (same).value.weight | `(384, 384)` |
| `backbone.blocks.{i}.attn.v.bias` | (same).value.bias | `(384,)` |
| `backbone.blocks.{i}.attn.proj.weight` | `backbone.0.encoder.encoder.layer.{i}.attention.output.dense.weight` | `(384, 384)` |
| `backbone.blocks.{i}.attn.proj.bias` | (same).bias | `(384,)` |
| `backbone.blocks.{i}.layer_scale1` | `backbone.0.encoder.encoder.layer.{i}.layer_scale1.lambda1` | `(384,)` |
| `backbone.blocks.{i}.norm2.{w,b}` | `backbone.0.encoder.encoder.layer.{i}.norm2.{weight,bias}` | `(384,)` |
| `backbone.blocks.{i}.mlp.fc1.{w,b}` | `backbone.0.encoder.encoder.layer.{i}.mlp.fc1.{weight,bias}` | `(384, 1536)`, `(1536,)` |
| `backbone.blocks.{i}.mlp.fc2.{w,b}` | (same).fc2.{w,b} | `(1536, 384)`, `(384,)` |
| `backbone.blocks.{i}.layer_scale2` | (same).layer_scale2.lambda1 | `(384,)` |
| `backbone.norm.{w,b}` | `backbone.0.encoder.encoder.layernorm.{w,b}` | `(384,)` |

### Multi-scale Projector (conv-based)

(Document what the introspection reveals — `MultiScaleProjector` with conv + BN layers.
List each conv weight/bias and BN weight/bias/running_mean/running_var.)

### Two-stage initialization (replaces standalone encoder)

| GGUF name | PyTorch source |
|-----------|----------------|
| `two_stage.enc_output.{i}.{w,b}` | `transformer.enc_output.{i}.{weight,bias}` (i in 0..12) |
| `two_stage.enc_output_norm.{i}.{w,b}` | `transformer.enc_output_norm.{i}.{weight,bias}` |
| `two_stage.enc_out_class_embed.{i}.{w,b}` | `transformer.enc_out_class_embed.{i}.{weight,bias}` |
| `two_stage.enc_out_bbox_embed.{i}.layers.{j}.{w,b}` | `transformer.enc_out_bbox_embed.{i}.layers.{j}.{weight,bias}` |

### Decoder

| GGUF name | PyTorch source |
|-----------|----------------|
| `decoder.queries` | `transformer.tgt_embed.weight` (300, 256) |
| `decoder.refpoints` | `transformer.refpoint_embed.weight` (3900, 4) — see group_detr |
| `decoder.layers.{i}.self_attn.in_proj_weight` | `transformer.decoder.layers.{i}.self_attn.in_proj_weight` (768, 256) |
| `decoder.layers.{i}.self_attn.in_proj_bias` | (same).in_proj_bias (768,) |
| `decoder.layers.{i}.self_attn.out_proj.{w,b}` | (same).out_proj.{w,b} (256,256), (256,) |
| `decoder.layers.{i}.norm1.{w,b}` | (...).norm1.{w,b} |
| `decoder.layers.{i}.cross_attn.sampling_offsets.{w,b}` | (deformable attn) |
| `decoder.layers.{i}.cross_attn.attention_weights.{w,b}` | (deformable attn) |
| `decoder.layers.{i}.cross_attn.value_proj.{w,b}` | (deformable attn) |
| `decoder.layers.{i}.cross_attn.output_proj.{w,b}` | (deformable attn) |
| `decoder.layers.{i}.norm2.{w,b}` | (...).norm2.{w,b} |
| `decoder.layers.{i}.ffn.linear1.{w,b}` | (...).linear1.{w,b} |
| `decoder.layers.{i}.ffn.linear2.{w,b}` | (...).linear2.{w,b} |
| `decoder.layers.{i}.norm3.{w,b}` | (...).norm3.{w,b} |
| `decoder.norm.{w,b}` | (...) — verify path |

### Heads

| GGUF name | PyTorch source |
|-----------|----------------|
| `heads.class_embed.{w,b}` | `class_embed.{w,b}` (91, 256), (91,) |
| `heads.bbox_embed.layers.{i}.{w,b}` | `bbox_embed.layers.{i}.{w,b}` (3-layer MLP: 256→256→256→4) |
| `heads.aux.{i}.class_embed.{w,b}` | per-decoder-layer aux class head (13 of them) |
| `heads.aux.{i}.bbox_embed.layers.{j}.{w,b}` | per-decoder-layer aux bbox MLP |
```

Fill in the actual numbers/paths from Task 1's introspection output. The above is a template.

### Step 2: Commit

```bash
git add docs/conversion.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
docs: rewrite conversion schema to match real rfdetr-base (format version 2)

Plan 7 schema redesign. The prior format-version-1 schema described an
aspirational architecture that does not match rfdetr 1.7.0. New schema:

- DINOv2-small backbone (dim=384, ffn=1536) with separate Q/K/V + layer scale
- No standalone encoder — two-stage init via enc_output / enc_output_norm /
  enc_out_class_embed / enc_out_bbox_embed (13 each)
- Conv-based MultiScaleProjector (not 4 linear levels)
- Deformable cross-attention in decoder
- 91 classes (incl. background); 300 queries × group_detr=13 = 3900 refpoints
- Heads renamed: class_embed (Linear) + bbox_embed (3-layer MLP)

Plans 8-12 rewrite the corresponding C++ numerical modules.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Update `Config` struct + metadata keys

Bring `src/model_loader.{hpp,cpp}` in line with the new schema.

**Files:**
- Modify: `src/model_loader.hpp` — restructure `Config` (drop encoder sub-struct, add two-stage + deformable fields)
- Modify: `src/model_loader.cpp` — `model_load` reads new keys; bump format-version check to "2"

### Steps

1. **`Config` struct changes:**
   - Remove `encoder` sub-struct entirely
   - Add `backbone.ffn_dim`, `backbone.global_attn_indices`, drop `backbone.multi_scale_layers`
   - Decoder gains `n_levels`, `n_points` (for deformable)
   - Add top-level: `patch_size`, `group_detr`, `two_stage_n_enc`
2. **`model_load`** reads the new keys; bumps `kFormatVersion` to `"2"`.
3. **`expected_tensor_names`** rewritten against the new schema (will be a long function — generate programmatically).

Test_model_loader assertions will need updates. Update them.

Commit.

---

### Task 4: Rewrite `gen_model_gguf.cpp`

The synthesized fixture has to produce the new tensor set (so `test_model_loader` and any other still-runnable tests pass against the new schema).

**Files:**
- Modify: `tests/fixtures/gen_model_gguf.cpp` — new variant config (dim=384 or shrunk equivalent), new tensor set
- Modify: `tests/test_model_loader.cpp` — assertion values

Pick shrunk dims for fast tests (e.g., `bb_dim=64`, `bb_ffn=128`, `image_size=56` like the old fixture had). Set `num_classes=91` to match real, `group_detr=13`, `n_levels=N`, `n_points=4`. Keep `bb_heads=8` for divisibility.

Commit.

---

### Task 5: Implement the conversion script body

Using `docs/conversion.md` as the contract, implement `build_tensor_name_map`, `validate`, `write_gguf` in `scripts/convert_rfdetr_to_gguf.py`.

Handles:
- Direct renames (most tensors)
- Backbone q/k/v separate → can stay separate (we changed the schema to match) — no concat needed
- group_detr expansion: `refpoint_embed.weight = (3900, 4)` either pass as-is or document slicing logic
- Two-stage modules: enc_output, enc_output_norm, etc. — straight renames
- Aux heads: 13 per-layer class + bbox heads, indexed

Verify by:
1. `python3 scripts/convert_rfdetr_to_gguf.py --variant base --dry-run` → 0 missing
2. `python3 scripts/convert_rfdetr_to_gguf.py --variant base --output models/rfdetr-base-f32.gguf --dtype f32` → real file
3. `build/bin/rfdetr-cli info --model models/rfdetr-base-f32.gguf` → loads, prints variant=base, num_classes=91, etc.

Commit (with the produced GGUF NOT committed — `models/` is gitignored).

---

### Task 6: Disable broken tests; final smoke

`test_parity_full_forward` exercises math that no longer matches anything. Disable it in `tests/CMakeLists.txt` with a comment pointing to Plan 13 for re-enablement.

`test_capi`'s `rfdetr_detect` assertion needs to allow either RFDETR_OK or RFDETR_ERR_INFERENCE (the numerical modules are now broken). Adjust.

`test_cli_integration` similarly tolerates either exit 0 or exit 4 (render failed) from the CLI. Adjust.

Final smoke: verify what's left passes; document what's deferred.

Update README.md to reflect transitional state. Commit.

---

## Self-Review

**Risk:** The most likely failure is incomplete tensor-name mapping in Task 5 — rfdetr's nested module paths are deep and some tensors may be missed. Iterate on the dry-run output until `missing_count = 0`.

**Honest assessment:** This plan is itself substantial (5-6 subagent tasks, each non-trivial). The conversion script alone may need debug iterations as upstream names get pinned down.

---

## Next plans

After this lands:

- **Plan 8** — Backbone redesign (separate Q/K/V, layer-scale, DINOv2-small dims)
- **Plan 9** — Delete standalone encoder; implement two-stage init
- **Plan 10** — Conv-based multi-scale projector
- **Plan 11** — Deformable cross-attention
- **Plan 12** — Heads redesign + 91 classes + aux heads
- **Plan 13** — PyTorch parity baseline (replaces numpy)
- **Plan 14** — Q8_0 quantization
- **Plan 15** — Real E2E demo
