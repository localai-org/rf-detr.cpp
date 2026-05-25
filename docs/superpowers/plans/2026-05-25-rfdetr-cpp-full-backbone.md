# rt-detr.cpp Full Backbone (Global Attention) Implementation Plan (Plan 4 of 9)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Plan 3 single-block forward pass to the full DINOv2 backbone — CLS token concatenation, positional embedding addition, all 12 transformer blocks (global attention only), final layer norm, and multi-scale feature taps at the 4 designated layers. Switch the test fixture from F16 to F32 weights so tolerances can tighten from 1e-3 to 1e-5 across all checkpoints. End state: clean rebuild produces a C++ backbone that matches the numpy reference at 1e-5 atol on every checkpoint (5 per block × 12 blocks + 4 multi-scale taps + 2 preprocess steps = 66 checkpoints) and a reusable `dinov2_forward(ctx, m, input)` entry point Plan 5+ can build on.

**Architecture:** No new src TUs. Existing pieces extend: `gen_model_gguf.cpp` switches default dtype to F32 (and drops the runtime F16→F32 casts in `dinov2.cpp`); `dinov2.{cpp,hpp}` gains three new functions (`add_cls_and_pos_embed`, `final_norm`, `forward`); `gen_numpy_baseline.py` extends the reference to cover the full backbone; `test_parity_block0.cpp` is renamed to `test_parity_backbone.cpp` and grows its checkpoint table.

**Tech Stack:** Same as Plan 3 (C++17, ggml CPU backend, Python 3.9+ with numpy + gguf). No new dependencies.

---

## Scope decisions

- **F32 fixture only.** Plan 3 used F16 to mirror real model weight dtype, but the resulting 4e-4 quantization noise on patch_embed forced loose 1e-3 tolerances that mask real bugs. Plan 4's fixture uses F32 throughout, so every parity tolerance drops to 1e-5 / 1e-4. Plan 7 (quantization) and Plan 8 (real PyTorch baseline) will re-introduce F16/Q8_0 weight handling with the right test discipline at that time.
- **Global attention only.** Window attention is Plan 5. The `backbone.window_size` metadata is still read but ignored. The numpy reference matches.
- **CLS token IS prepended and DOES participate in attention.** This is faithful to the upstream DINOv2 / rfdetr implementation. Positional embeddings have shape `(1, N_patches + 1, dim)` — one slot for CLS.
- **Multi-scale taps publish backbone output AFTER block i**, where i is in `backbone.multi_scale_layers` (= `[2, 5, 8, 11]` for the base variant config). The tap names match Plan 5's projector input contract.
- **`dinov2_forward`** is the new public entry. It takes the input image tensor, runs the full backbone, and returns the multi-scale features as a vector of N tensors (one per tap). The trace callback fires at every internal checkpoint plus the multi-scale outputs.
- **Test rename** is just `git mv test_parity_block0.cpp test_parity_backbone.cpp` with content expansion. The Plan 3 test's checkpoints are preserved; new ones are added.

---

## File map (created or modified in this plan)

```
rt-detr.cpp/
├── docs/
│   └── parity.md                    # MODIFY — new tolerance table for full backbone
├── scripts/
│   └── gen_numpy_baseline.py        # MODIFY — extend reference to full backbone
├── src/
│   ├── dinov2.{cpp,hpp}             # MODIFY — add cls_pos_embed, final_norm, forward;
│   │                                #          remove the runtime F16→F32 casts
│   └── (other src files unchanged)
├── tests/
│   ├── CMakeLists.txt               # MODIFY — rename test target, baseline output filename
│   ├── fixtures/
│   │   ├── gen_model_gguf.cpp       # MODIFY — F32 default, --dtype flag for forward-compat
│   │   ├── model_base_seeded.gguf   # REGENERATED at build with F32 weights
│   │   └── baseline_backbone.gguf   # REGENERATED at build (rename from baseline_block0.gguf)
│   └── test_parity_backbone.cpp     # RENAME from test_parity_block0.cpp; extend checkpoints
└── README.md                        # MODIFY — Plan 4 status
```

---

### Task 1: Switch fixture to F32 weights

Plan 3's fixture was F16; the Task 11 implementer added `to_f32()` casts in `dinov2.cpp` to bridge the dtype gap. Plan 4 inverts: fixture is F32, and the C++ code drops the casts (smaller graph, tighter tolerances).

**Files:**
- Modify: `tests/fixtures/gen_model_gguf.cpp` — F32 default, optional `--dtype f16` for forward compat
- Modify: `src/dinov2.cpp` — remove the `to_f32()` helper and its call sites
- Modify: `tests/test_parity_block0.cpp` — tighten tolerances to F32-realistic values
- Modify: `docs/parity.md` — update tolerance table

### Step 1: Adapt `gen_model_gguf.cpp`

Currently the generator hardcodes `GGML_TYPE_F16` (look for `const ggml_type F = GGML_TYPE_F16;` in `add_all_tensors`). Replace the constant with a parameter:

```cpp
void add_all_tensors(ggml_context* ctx, std::vector<ggml_tensor*>& out,
                     const VariantCfg& v, ggml_type tensor_dtype) {
    const ggml_type F = tensor_dtype;
    // ... rest unchanged
}
```

In `main()`, add a `--dtype f16|f32` flag (default `f32`):

```cpp
ggml_type tensor_dtype = GGML_TYPE_F32;
for (int i = 2; i + 1 < argc; ++i) {
    // ... existing --missing, --seed parses ...
    if (std::strcmp(argv[i], "--dtype") == 0) {
        std::string v = argv[i + 1];
        if      (v == "f16") tensor_dtype = GGML_TYPE_F16;
        else if (v == "f32") tensor_dtype = GGML_TYPE_F32;
        else { std::fprintf(stderr, "unknown --dtype: %s\n", v.c_str()); return 5; }
    }
}
```

Pass `tensor_dtype` into `add_all_tensors(ctx, tensors, v, tensor_dtype)`.

The seeded-fill branch already handles both F32 and F16 via an `if (t->type == GGML_TYPE_F32) { ... } else if (t->type == GGML_TYPE_F16) { ... }` — no change there.

### Step 2: Remove F16→F32 casts from `dinov2.cpp`

The Task 11 implementer added a `to_f32()` helper (lookup or comment for the exact location — likely top of file or anonymous namespace) and called it on `patch_embed.bias`, all LN weights/biases, all attn/mlp biases, etc., before each `ggml_add` / `ggml_mul` / `ggml_norm` to bridge F16 → F32.

With F32 weights they're no-ops. Replace each `to_f32(b)` call site with just `b` and delete the helper.

Search and replace pattern: `to_f32(` → `(` (then manually fix the trailing `)` count). OR use a Read+Edit pass to remove each call.

Verify with a build that nothing fails.

### Step 3: Tighten tolerances in `test_parity_block0.cpp`

The current table (post Plan 3 polish):

```cpp
const std::map<std::string, Tol> kTolerances = {
    {"backbone.patch_embed.output",    {1e-3f, 1e-2f}},  // F16 weight quantization noise
    {"backbone.block.0.norm1.output",  {1e-4f, 1e-3f}},
    {"backbone.block.0.attn.output",   {1e-5f, 1e-4f}},
    {"backbone.block.0.mlp.output",    {1e-5f, 1e-4f}},
    {"backbone.block.0.output",        {1e-3f, 1e-2f}},  // carries patch_embed noise via residual
};
```

With F32 weights, the F16-noise comments are obsolete. New table:

```cpp
const std::map<std::string, Tol> kTolerances = {
    {"backbone.patch_embed.output",    {1e-5f, 1e-4f}},
    {"backbone.block.0.norm1.output",  {1e-5f, 1e-4f}},
    {"backbone.block.0.attn.output",   {1e-5f, 1e-4f}},
    {"backbone.block.0.mlp.output",    {1e-5f, 1e-4f}},
    {"backbone.block.0.output",        {1e-5f, 1e-4f}},
};
```

### Step 4: Build, regenerate fixtures, run test

```bash
rm -rf build
cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=OFF 2>&1 | tail -3
cmake --build build -j 2>&1 | tail -5
ls -lh tests/fixtures/model_base_seeded.gguf
ctest --test-dir build --output-on-failure
```

Expected:
- Seeded fixture grows from ~1.9MB to ~3.8MB (F32 = 2× F16 size)
- 9/9 tests pass, with test_parity_block0 at the tighter tolerances
- Per-checkpoint max_abs values should be < 1e-5 (typically much less; numpy's float64 vs ggml's float32 limits the floor)

If any checkpoint exceeds the tighter tolerance, investigate before proceeding. The most likely cause: a forgotten `to_f32()` removal that left an inconsistency.

### Step 5: Update `docs/parity.md`

Replace the current per-checkpoint table with:

```markdown
| Checkpoint                            | atol  | rtol  |
|---------------------------------------|-------|-------|
| `backbone.patch_embed.output`         | 1e-5  | 1e-4  |
| `backbone.block.0.norm1.output`       | 1e-5  | 1e-4  |
| `backbone.block.0.attn.output`        | 1e-5  | 1e-4  |
| `backbone.block.0.mlp.output`         | 1e-5  | 1e-4  |
| `backbone.block.0.output`             | 1e-5  | 1e-4  |
```

Update the prose: "Plan 4 switched the fixture to F32 weights, eliminating the F16 quantization noise floor; all backbone checkpoints now ride at 1e-5 atol."

Task 2-8 will add more rows to this table as new checkpoints land.

### Step 6: Commit

```bash
git add tests/fixtures/gen_model_gguf.cpp src/dinov2.cpp \
        tests/test_parity_block0.cpp docs/parity.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
test(fixture): F32 weights by default; tighten parity tolerances to 1e-5

Plan 3 used F16 fixture weights to mirror production model dtype, but the
resulting ~4e-4 quantization noise on patch_embed forced loose 1e-3
tolerances that mask real bugs. Plan 4 switches to F32 throughout (--dtype
defaults to f32; --dtype f16 still works for forward-compat tests). The
runtime to_f32() casts in dinov2.cpp are removed — every tensor is already
F32. Tolerances tighten to {1e-5, 1e-4} across the board; max measured
delta should now be <1e-5 on all checkpoints.

Plan 7 will re-introduce F16/quantized weight handling with proper noise
discipline at that time.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: CLS token + positional embedding (`dinov2_add_cls_and_pos_embed`)

DINOv2 prepends a learnable CLS token and adds positional embeddings to the patch tokens before block 0. This task adds the function, the numpy reference, and a parity checkpoint.

**Files:**
- Modify: `src/dinov2.hpp` — declare `dinov2_add_cls_and_pos_embed`
- Modify: `src/dinov2.cpp` — implement it
- Modify: `scripts/gen_numpy_baseline.py` — add CLS + pos_embed steps
- Modify: `tests/test_parity_block0.cpp` — add `backbone.cls_pos_embed.output` checkpoint to tolerances + wire it into the forward path
- Modify: `docs/parity.md` — add the new checkpoint row

### Step 1: Declare in `src/dinov2.hpp`

Add inside `namespace rfdetr`:

```cpp
/* Concatenate the learnable CLS token to the front of the patch tokens and
 * add the positional embedding.
 *
 * Input:  `tokens` — (dim, N_patches, 1, 1) F32
 * Output: a tensor of shape (dim, N_patches + 1, 1, 1) F32 with CLS at
 *         index 0 and positional offsets added to every position.
 *
 * Publishes "backbone.cls_pos_embed.output" via the trace callback. */
ggml_tensor* dinov2_add_cls_and_pos_embed(ggml_context* ctx, const Model& m,
                                          ggml_tensor* tokens);
```

### Step 2: Implement in `src/dinov2.cpp`

Append (after `dinov2_patch_embed`, before `dinov2_block`):

```cpp
ggml_tensor* dinov2_add_cls_and_pos_embed(ggml_context* ctx, const Model& m,
                                          ggml_tensor* tokens) {
    /* Tensor layouts:
     *   tokens     ne = (dim, N_patches, 1, 1)
     *   cls_token  ne = (dim, 1, 1, 1)         — Plan 3 fixture writes it as 1D (dim,)
     *                                           but ggml_concat handles broadcast-compatible shapes
     *   pos_embed  ne = (dim, N_patches + 1, 1, 1)
     *
     * Steps:
     *   1. Make CLS broadcast-shape-compatible with the concat
     *   2. Concat along axis 1 (token dimension)
     *   3. Add positional embedding (which is already shape-compatible)
     */

    auto it_cls = m.tensors.find("backbone.cls_token");
    auto it_pe  = m.tensors.find("backbone.pos_embed");
    if (it_cls == m.tensors.end() || it_pe == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "dinov2_add_cls_and_pos_embed: missing cls_token or pos_embed");
        return nullptr;
    }
    ggml_tensor* cls = it_cls->second;
    ggml_tensor* pe  = it_pe->second;

    /* cls_token in the seeded fixture is 1D (dim,). Reshape to (dim, 1, 1, 1)
     * so it concats correctly along axis 1 with tokens (dim, N, 1, 1). */
    ggml_tensor* cls2 = ggml_reshape_2d(ctx, cls, cls->ne[0], 1);

    /* Concat along axis 1: (dim, 1) ⊕ (dim, N) → (dim, N+1) */
    ggml_tensor* with_cls = ggml_concat(ctx, cls2, tokens, /*dim*/ 1);
    /* with_cls ne = (dim, N+1, 1, 1) */

    /* pos_embed in the fixture is (dim, N+1) — direct add works. If the
     * fixture/model writes pos_embed with leading batch dim (1, N+1, dim),
     * an additional reshape is needed. Verify by inspecting pe->ne. */
    ggml_tensor* out = ggml_add(ctx, with_cls, pe);

    publish("backbone.cls_pos_embed.output", out);
    return out;
}
```

**Verify ggml API**: confirm `ggml_concat(ctx, a, b, dim)` exists with that signature in v0.13.0. If the concat function name or signature differs, adapt.

**Verify pos_embed shape**: the fixture writes pos_embed with `make_tensor(ctx, "backbone.pos_embed", F, v.bb_dim, n_patches + 1)` → ne = `(dim, n_patches+1, 1, 1)`. That matches the add target shape. ✓

**Verify cls_token shape**: fixture writes `make_tensor(ctx, "backbone.cls_token", F, v.bb_dim)` → ne = `(dim, 1, 1, 1)` (1D). The `ggml_reshape_2d(cls, dim, 1)` converts to `(dim, 1, 1, 1)` which is what `ggml_concat` along axis 1 expects.

### Step 3: Extend numpy reference (`scripts/gen_numpy_baseline.py`)

In the `forward(cfg, tensors, input_img)` function, after `patchify_and_embed` and BEFORE the block 0 forward, add:

```python
    # ---- CLS token + positional embedding ----
    cls_token = tensors["backbone.cls_token"]   # (dim,) or (1, 1, dim) depending on writer
    pos_embed = tensors["backbone.pos_embed"]   # (N+1, dim)

    # Normalize shapes: tokens are currently (N, dim).
    cls_flat = cls_token.reshape(1, -1)                 # (1, dim)
    tokens_with_cls = np.concatenate([cls_flat, tokens], axis=0)  # (N+1, dim)

    # Positional embedding may be stored as (N+1, dim) directly or (1, N+1, dim).
    # Reshape to 2D if needed:
    pe = pos_embed.reshape(-1, cls_flat.shape[-1])      # (N+1, dim)
    if pe.shape[0] != tokens_with_cls.shape[0]:
        raise ValueError(f"pos_embed shape {pe.shape} doesn't match "
                         f"tokens+cls shape {tokens_with_cls.shape}")

    x = tokens_with_cls + pe                            # (N+1, dim)
    out["backbone.cls_pos_embed.output"] = x.copy()
```

Then change the existing block-0 entry to consume `x` (which now has `N+1` tokens including CLS) instead of `tokens` (which was `N` patches only).

### Step 4: Extend test_parity_block0 tolerance table

Add the new checkpoint:

```cpp
{"backbone.cls_pos_embed.output", {1e-5f, 1e-4f}},
```

And wire the function into the forward path. Find this in the test:

```cpp
ggml_tensor* t = rfdetr::dinov2_patch_embed(gctx, *m, input);
t = rfdetr::dinov2_block(gctx, *m, t, /*block_idx*/ 0);
```

Insert the new call between them:

```cpp
ggml_tensor* t = rfdetr::dinov2_patch_embed(gctx, *m, input);
t = rfdetr::dinov2_add_cls_and_pos_embed(gctx, *m, t);
t = rfdetr::dinov2_block(gctx, *m, t, /*block_idx*/ 0);
```

### Step 5: Update `docs/parity.md`

Add the row:

```markdown
| `backbone.cls_pos_embed.output`       | 1e-5  | 1e-4  |
```

### Step 6: Build and run

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build -R test_parity_block0 --output-on-failure
```

Expected: PASS. If `cls_pos_embed.output` fails first, the CLS / pos_embed layout or concat dimension is off — debug from numpy printouts.

If `block.0.output` regresses (it was passing pre-CLS), the issue is that block 0 is now consuming N+1 tokens (CLS prepended) instead of N. The block code itself is dimension-agnostic so it should "just work" — but the numpy reference must also consume N+1 tokens.

### Step 7: Commit

```bash
git add src/dinov2.hpp src/dinov2.cpp scripts/gen_numpy_baseline.py \
        tests/test_parity_block0.cpp docs/parity.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(dinov2): CLS token + positional embedding

dinov2_add_cls_and_pos_embed prepends the learnable CLS token to the patch
tokens and adds the positional embedding (which has N+1 slots, one for CLS).
Block 0 now consumes N+1 tokens; the block code is dimension-agnostic.
Numpy reference updated; new parity checkpoint backbone.cls_pos_embed.output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Extend forward through all 12 blocks

The block code is already block-index-parameterized. The test forward-pass and the numpy reference loop over `cfg.backbone.depth` blocks.

**Files:**
- Modify: `scripts/gen_numpy_baseline.py` — loop over all blocks
- Modify: `tests/test_parity_block0.cpp` — loop in the C++ forward; extend tolerances table to cover blocks 1..11

### Step 1: Extend numpy reference

In `gen_numpy_baseline.py`, find the block-0 section that does:

```python
    p = "backbone.blocks.0."
    n1 = layer_norm(x, tensors[p + "norm1.weight"], tensors[p + "norm1.bias"])
    out["backbone.block.0.norm1.output"] = n1.copy()
    # ...
    out["backbone.block.0.output"] = x.copy()
```

Replace with a loop:

```python
    for i in range(cfg["bb_depth"]):
        p = f"backbone.blocks.{i}."
        pub = f"backbone.block.{i}."

        # x = x + attn(norm1(x))
        n1 = layer_norm(x, tensors[p + "norm1.weight"], tensors[p + "norm1.bias"])
        out[pub + "norm1.output"] = n1.copy()
        y = mha(n1,
                tensors[p + "attn.qkv.weight"], tensors[p + "attn.qkv.bias"],
                tensors[p + "attn.proj.weight"], tensors[p + "attn.proj.bias"],
                cfg["bb_heads"])
        out[pub + "attn.output"] = y.copy()
        x = x + y

        # x = x + mlp(norm2(x))
        n2 = layer_norm(x, tensors[p + "norm2.weight"], tensors[p + "norm2.bias"])
        z = mlp(n2,
                tensors[p + "mlp.fc1.weight"], tensors[p + "mlp.fc1.bias"],
                tensors[p + "mlp.fc2.weight"], tensors[p + "mlp.fc2.bias"])
        out[pub + "mlp.output"] = z.copy()
        x = x + z

        out[pub + "output"] = x.copy()
```

### Step 2: Extend test_parity_block0 C++ forward

Find the test's forward block (currently `t = dinov2_block(gctx, *m, t, 0);`). Wrap in a loop:

```cpp
for (uint32_t i = 0; i < m->config.backbone.depth; ++i) {
    t = rfdetr::dinov2_block(gctx, *m, t, (int)i);
    RFDETR_ASSERT(t != nullptr);
}
```

### Step 3: Programmatically generate tolerances for blocks 1..11

The current `kTolerances` map is hardcoded for block 0. Hardcoding all 60 entries (5 checkpoints × 12 blocks) is repetitive. Build the table once at startup:

```cpp
std::map<std::string, Tol> build_tolerances(uint32_t depth) {
    std::map<std::string, Tol> tol;
    tol["backbone.patch_embed.output"]    = {1e-5f, 1e-4f};
    tol["backbone.cls_pos_embed.output"]  = {1e-5f, 1e-4f};
    for (uint32_t i = 0; i < depth; ++i) {
        std::string p = "backbone.block." + std::to_string(i) + ".";
        tol[p + "norm1.output"] = {1e-5f, 1e-4f};
        tol[p + "attn.output"]  = {1e-5f, 1e-4f};
        tol[p + "mlp.output"]   = {1e-5f, 1e-4f};
        tol[p + "output"]       = {1e-5f, 1e-4f};
    }
    return tol;
}
```

Delete the const `kTolerances` map. In `main()`, call `auto kTolerances = build_tolerances(m->config.backbone.depth);` after loading the model.

### Step 4: Build and run

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build -R test_parity_block0 --output-on-failure
```

Expected: PASS for all 50 new block checkpoints. If a particular block (say block 5) fails first, the issue is likely:
- Numerical accumulation across many residual+norm steps (loosen tolerance to {1e-4, 1e-3} for that one and document)
- Or a real bug (less likely since block 0 already passes)

### Step 5: Commit

```bash
git add scripts/gen_numpy_baseline.py tests/test_parity_block0.cpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(dinov2): extend forward + parity to all 12 backbone blocks

Numpy reference loops over cfg.bb_depth instead of hardcoded block-0. C++
test runs the same loop. Tolerance table generated programmatically. All
60 new checkpoints (5 per block × 12 blocks) parity-green at {1e-5, 1e-4}.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Final backbone layer norm

DINOv2 applies one more LayerNorm after the last block (using `backbone.norm.{weight,bias}`). This task adds it, both sides.

**Files:**
- Modify: `src/dinov2.hpp` — declare `dinov2_final_norm`
- Modify: `src/dinov2.cpp` — implement it
- Modify: `scripts/gen_numpy_baseline.py` — apply after the block loop
- Modify: `tests/test_parity_block0.cpp` — call it after the block loop; add tolerance entry
- Modify: `docs/parity.md` — add the checkpoint row

### Step 1: Declare in `src/dinov2.hpp`

```cpp
/* Apply the final backbone LayerNorm (after the last block).
 *
 * Input/output: (dim, N+1, 1, 1) F32.
 *
 * Publishes "backbone.norm.output" via the trace callback. */
ggml_tensor* dinov2_final_norm(ggml_context* ctx, const Model& m,
                               ggml_tensor* x);
```

### Step 2: Implement in `src/dinov2.cpp`

```cpp
ggml_tensor* dinov2_final_norm(ggml_context* ctx, const Model& m,
                               ggml_tensor* x) {
    auto it_w = m.tensors.find("backbone.norm.weight");
    auto it_b = m.tensors.find("backbone.norm.bias");
    if (it_w == m.tensors.end() || it_b == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "dinov2_final_norm: missing backbone.norm");
        return nullptr;
    }
    /* layer_norm is the anonymous-namespace helper already used by dinov2_block. */
    ggml_tensor* y = layer_norm(ctx, x, it_w->second, it_b->second);
    publish("backbone.norm.output", y);
    return y;
}
```

If `layer_norm` is hidden inside the anonymous namespace (so this new function can't reach it), either move `layer_norm` to a file-scope (non-anonymous) namespace or inline the LN math here. The former is cleaner.

### Step 3: Extend numpy reference

After the block loop in `forward()`:

```python
    fn = layer_norm(x,
                    tensors["backbone.norm.weight"],
                    tensors["backbone.norm.bias"])
    out["backbone.norm.output"] = fn.copy()
    x = fn
```

### Step 4: Wire into the test forward

After the block loop:

```cpp
t = rfdetr::dinov2_final_norm(gctx, *m, t);
RFDETR_ASSERT(t != nullptr);
```

And add the tolerance entry inside `build_tolerances()`:

```cpp
tol["backbone.norm.output"] = {1e-5f, 1e-4f};
```

### Step 5: docs/parity.md

Add the row.

### Step 6: Build, run, commit

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build -R test_parity_block0 --output-on-failure
```

Expected: PASS.

```bash
git add src/dinov2.hpp src/dinov2.cpp scripts/gen_numpy_baseline.py \
        tests/test_parity_block0.cpp docs/parity.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(dinov2): final backbone layer norm

dinov2_final_norm applies the last DINOv2 LayerNorm (backbone.norm) after
the last transformer block. Numpy reference matches. New parity checkpoint
backbone.norm.output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Multi-scale feature taps

RF-DETR taps backbone features at layers `[2, 5, 8, 11]` (per the base variant config). These four taps are what the projector consumes downstream. This task publishes them as parity checkpoints.

**Files:**
- Modify: `src/dinov2.cpp` — publish `backbone.multiscale.level{j}` checkpoints inside the block loop in `dinov2_forward` (added in Task 6) OR inside the test (interim)
- Modify: `scripts/gen_numpy_baseline.py` — publish the same checkpoints
- Modify: `tests/test_parity_block0.cpp` — add tolerances
- Modify: `docs/parity.md`

Because `dinov2_block` doesn't know which block indices are multi-scale taps (that's a per-model configuration), the publishing must happen in the loop caller. Task 6 wraps the whole thing in `dinov2_forward`. For Task 5, do the publishing in the test's loop:

### Step 1: Extend test loop

```cpp
const auto& ms_layers = m->config.backbone.multi_scale_layers;
auto is_ms_layer = [&](uint32_t i) {
    for (uint32_t v : ms_layers) if (v == i) return true;
    return false;
};

for (uint32_t i = 0; i < m->config.backbone.depth; ++i) {
    t = rfdetr::dinov2_block(gctx, *m, t, (int)i);
    RFDETR_ASSERT(t != nullptr);
    /* Multi-scale tap: publish backbone output at selected layer indices */
    if (is_ms_layer(i)) {
        /* Find the level index (position in ms_layers) for this i */
        size_t level = 0;
        for (size_t k = 0; k < ms_layers.size(); ++k) {
            if (ms_layers[k] == i) { level = k; break; }
        }
        rfdetr::publish("backbone.multiscale.level" + std::to_string(level), t);
    }
}
```

### Step 2: Extend numpy reference

Inside the block loop in `forward()`:

```python
    ms_layers = cfg.get("bb_multi_scale_layers")  # need to read this from the model file
    if i in ms_layers:
        level = ms_layers.index(i)
        out[f"backbone.multiscale.level{level}"] = x.copy()
```

`bb_multi_scale_layers` must be added to the config dict in `read_model()`. The gguf-py reader returns it as an int32 array; extract via:

```python
"bb_multi_scale_layers": list(map(int, reader.get_field("rfdetr.backbone.multi_scale_layers").parts[-1])),
```

(Verify the gguf-py field shape — it may need a different access pattern; the Task 10 implementer documented the gguf-py quirks.)

### Step 3: Add tolerances

Inside `build_tolerances`:

```cpp
const auto& ms = m->config.backbone.multi_scale_layers;
for (size_t k = 0; k < ms.size(); ++k) {
    tol["backbone.multiscale.level" + std::to_string(k)] = {1e-5f, 1e-4f};
}
```

Note `build_tolerances` now needs access to the Config struct, not just `depth`. Either pass the whole Config or change the signature to take a `const Config&`.

### Step 4: Build, run

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build -R test_parity_block0 --output-on-failure
```

Expected: 4 new multi-scale checkpoints all pass.

### Step 5: Commit

```bash
git add src/dinov2.cpp scripts/gen_numpy_baseline.py tests/test_parity_block0.cpp docs/parity.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(dinov2): multi-scale feature taps

Publishes backbone.multiscale.level{j} parity checkpoints at the indices
configured by rfdetr.backbone.multi_scale_layers (= [2,5,8,11] for base).
These are the four feature maps Plan 6's projector will consume.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `dinov2_forward` — unified backbone entry point

Refactor the test's hand-wired forward into a single `dinov2_forward(ctx, m, input)` function. Plan 5+ will call it once instead of duplicating the patch_embed → CLS+pos_embed → loop → final_norm sequence.

**Files:**
- Modify: `src/dinov2.hpp` — declare `dinov2_forward`
- Modify: `src/dinov2.cpp` — implement it; move multi-scale publishing in here
- Modify: `tests/test_parity_block0.cpp` — replace inline sequence with a single `dinov2_forward` call

### Step 1: Declare in `src/dinov2.hpp`

```cpp
/* Run the full DINOv2 backbone: patch_embed → CLS+pos_embed → N blocks →
 * final_norm. Publishes every per-block and multi-scale checkpoint via the
 * trace callback. Returns the final post-norm tensor (dim, N+1, 1, 1) F32.
 *
 * The multi-scale features are not returned explicitly — Plan 6's projector
 * either re-runs the backbone with its own trace callback or restructures
 * this API to return them as a vector. Plan 4 keeps the API minimal. */
ggml_tensor* dinov2_forward(ggml_context* ctx, const Model& m,
                            ggml_tensor* input);
```

### Step 2: Implement in `src/dinov2.cpp`

```cpp
ggml_tensor* dinov2_forward(ggml_context* ctx, const Model& m,
                            ggml_tensor* input) {
    ggml_tensor* t = dinov2_patch_embed(ctx, m, input);
    if (!t) return nullptr;
    t = dinov2_add_cls_and_pos_embed(ctx, m, t);
    if (!t) return nullptr;

    const auto& ms = m.config.backbone.multi_scale_layers;
    auto is_ms = [&](uint32_t i) -> int {
        for (size_t k = 0; k < ms.size(); ++k) if (ms[k] == i) return (int)k;
        return -1;
    };

    for (uint32_t i = 0; i < m.config.backbone.depth; ++i) {
        t = dinov2_block(ctx, m, t, (int)i);
        if (!t) return nullptr;
        int level = is_ms(i);
        if (level >= 0) {
            publish("backbone.multiscale.level" + std::to_string(level), t);
        }
    }

    t = dinov2_final_norm(ctx, m, t);
    return t;
}
```

### Step 3: Simplify the test

Replace the hand-wired sequence in `tests/test_parity_block0.cpp` with:

```cpp
ggml_tensor* t = rfdetr::dinov2_forward(gctx, *m, input);
RFDETR_ASSERT(t != nullptr);
```

The trace callback still fires at every checkpoint inside `dinov2_forward`.

### Step 4: Build and run

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build -R test_parity_block0 --output-on-failure
```

Expected: PASS — all 66 checkpoints unchanged, just running through a different entry point.

### Step 5: Commit

```bash
git add src/dinov2.hpp src/dinov2.cpp tests/test_parity_block0.cpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
refactor(dinov2): unify backbone into dinov2_forward entry point

Replaces hand-wired patch_embed→cls_pos→block_loop→final_norm sequence in
the test with a single dinov2_forward(ctx, m, input) call. Multi-scale
publishing moves inside the function. Plan 5+ calls dinov2_forward once
instead of duplicating the sequence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Rename test + final smoke + README

Final cleanup. The test is no longer "block 0" — it covers the full backbone.

**Files:**
- Rename: `tests/test_parity_block0.cpp` → `tests/test_parity_backbone.cpp` (git mv)
- Modify: `tests/CMakeLists.txt` — change target name; baseline output filename if desired
- Modify: `README.md`

### Step 1: Rename

```bash
git mv tests/test_parity_block0.cpp tests/test_parity_backbone.cpp
```

### Step 2: Update `tests/CMakeLists.txt`

Replace `rfdetr_add_test(test_parity_block0)` with `rfdetr_add_test(test_parity_backbone)`. Replace `rfdetr_baseline_block0` references to the new name (see Step 3).

Optionally rename the baseline output:
- Custom command `OUTPUT  ${RFDETR_TEST_FIXTURES}/baseline_block0.gguf` → `baseline_backbone.gguf`
- `add_custom_target(rfdetr_baseline_block0 ...)` → `rfdetr_baseline_backbone`
- `add_dependencies(test_parity_backbone rfdetr_baseline_backbone ...)`

If renaming, also update the test code's `baseline_path` string from `baseline_block0.gguf` to `baseline_backbone.gguf`.

### Step 3: Clean rebuild + full test run

```bash
rm -rf build
cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=ON
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build --output-on-failure
```

Expected: 10 tests pass; test_parity_backbone replaces test_parity_block0.

### Step 4: Verify three configurations

```bash
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=OFF -DRFDETR_BUILD_CLI=OFF 2>&1 | tail -3 && cmake --build build -j 2>&1 | tail -3
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=OFF 2>&1 | tail -3 && cmake --build build -j 2>&1 | tail -3 && ctest --test-dir build --output-on-failure 2>&1 | tail -3
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=ON 2>&1 | tail -3 && cmake --build build -j 2>&1 | tail -3 && ctest --test-dir build --output-on-failure 2>&1 | tail -3
```

All must pass.

### Step 5: Update README

Replace the Status section with:

```markdown
## Status

**Full backbone (Plan 4) complete.** The C++ runtime runs the entire
DINOv2 backbone — patch_embed, CLS token + positional embedding, all 12
transformer blocks (global attention), final layer norm, and 4 multi-scale
feature taps — and matches a pure-numpy reference at 1e-5 absolute
tolerance on all 67 parity checkpoints. Ten tests pass on a clean build.

Plan 5 adds window attention (the only DINOv2 feature still on the global
path). Plans 6+ wire the projector, encoder, decoder, heads, and end-to-end
detect.

The Python conversion script body is still deferred (see Plan 2 Task 3).
The C++ side uses a synthesized F32 GGUF fixture for tests.
```

### Step 6: Commit

```bash
git add tests/CMakeLists.txt README.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
chore: rename test_parity_block0 -> test_parity_backbone; Plan 4 README

The test now covers the full backbone (67 parity checkpoints across
patch_embed, CLS+pos_embed, 12 blocks, final norm, 4 multi-scale taps).
Old "block0" name was misleading.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Note: the actual git mv was committed in Task 7 Step 1 if you ran it then; otherwise stage it here. Test code edits should have been committed in Task 6 already.)

---

## Self-Review

**Spec coverage** against `docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md`:

- §3 Architecture — DINOv2 backbone (patch_embed, CLS, pos_embed, 12 blocks, final norm) all landed. Window attention deferred to Plan 5 (the only DINOv2 piece still on the global path).
- §6 API — `rfdetr_detect` still NOT_IMPLEMENTED; Plan 6 wires it.
- §8 Tests — `test_parity_backbone` (renamed from `test_parity_block0`) is the test surface.
- §9 Parity workflow — 67 checkpoints at 1e-5 / 1e-4 tolerances. Documented in `docs/parity.md`.
- §10 Build — no new src files; existing CMake unchanged structurally.

**Type consistency** — `dinov2_forward` signature returns `ggml_tensor*` (the post-norm final token tensor). Multi-scale features are not returned; they only flow via the trace callback. Plan 5 / Plan 6 will need to either:
1. Add a structured return type (e.g. `struct DinoForwardResult { ggml_tensor* final; std::vector<ggml_tensor*> multi_scale; };`)
2. Or extract multi-scale from the trace callback during graph build

Plan 6's projector will need this. The current Plan 4 API is minimal; revisit when projector code lands.

**Placeholder scan** — no "TBD"/"fill in later"/"TODO" in the plan body.

**Risk areas:**
1. **Numerical drift across 12 blocks.** Per-block parity is at 1e-5, but errors accumulate via residual connections. By block 11, the cumulative drift could exceed 1e-5. If so, loosen the late-block tolerances slightly (1e-4) and document.
2. **CLS / pos_embed shape conventions.** The fixture currently writes `cls_token` as 1D `(dim,)` and `pos_embed` as 2D `(dim, N+1)`. Real rfdetr uses `(1, 1, dim)` and `(1, N+1, dim)` (extra batch dim). The numpy reference must agree with whichever shape the C++ side expects — and the C++ side is shape-driven by what's in the GGUF. Verify by inspecting `cls_token->ne` at runtime.
3. **`ggml_concat` API.** Confirm the function name and signature in v0.13.0 — Task 2's caveat applies.
4. **`backbone.multi_scale_layers` storage in GGUF.** Plan 2's polish updated docs to say `int32[]`. The Plan 4 Task 5 numpy reference reads via `int.parts[-1]`. Verify the typed extraction works correctly on the fixture data.

---

## Next plan

After this plan lands:

- **Plan 5** — Window attention switching. Adds metadata for which blocks are windowed vs global; implements the window-reshape-attend-unreshape sequence in `mha`. Fixture may need adjustment (different `window_size` value to actually test windowing with the small fixture image).
- **Plan 6** — Projector + encoder + decoder + heads + end-to-end `rfdetr_detect`. The `dinov2_forward` API may grow to return multi-scale features explicitly so the projector can consume them.
- **Plan 7** — Real PyTorch baseline (`scripts/run_rfdetr_baseline.py`) augmenting / replacing numpy.
- **Plan 8** — Quantization.
- **Plan 9** — Variants nano/small/medium/large.
