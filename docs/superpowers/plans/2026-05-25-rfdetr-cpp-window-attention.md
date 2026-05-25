# rt-detr.cpp Window Attention Implementation Plan (Plan 5 of 9)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add windowed self-attention to the DINOv2 backbone. Per the Plan 4 final-review recommendation, dispatch by reusing the existing `backbone.multi_scale_layers` metadata: those four indices `[2, 5, 8, 11]` are exactly the *global*-attention blocks; the other eight blocks become *windowed*. Shrink the synthesized fixture's `window_size` from 14 to 2 so 8 windowed blocks across a 4×4 patch grid genuinely exercise the windowing math (4 windows × 4 patches each per windowed block). Both numpy reference and C++ implementation grow a matching dispatch. End state: 55 parity checkpoints still green at 1e-5 atol, with windowed blocks producing materially different math than global blocks (verifiable by inspection of the regenerated baseline).

**Architecture:** No new src TUs. `src/dinov2.{cpp,hpp}` gains a public `is_global_block(cfg, i)` predicate plus an internal `mha_window()` helper; `dinov2_block()` dispatches on the predicate. The CLS token is set aside before windowing and re-prepended after — windowed blocks operate on patch tokens only, mirroring the most common ViT-with-window-attention convention. The numpy reference gains a parallel `mha_window()` and the same dispatch.

**Tech Stack:** Same as Plan 4 (C++17, ggml CPU backend, numpy + gguf). No new dependencies.

---

## Scope decisions

- **Dispatch via `multi_scale_layers`** — no new GGUF metadata. Global blocks are the indices in `multi_scale_layers`; everything else is windowed. The Plan 4 final reviewer pointed out this isn't a coincidence: rfdetr/DINOv2's globally-attending blocks are the tap points by design (they produce the multi-scale features the projector consumes). Reusing the field keeps the schema lean.
- **Fixture `window_size = 2`** — current fixture has 16 patches (4×4) + CLS = 17 tokens. window_size=14 (14×14=196 tokens per window) degenerates to global. window_size=2 (2×2=4 patches per window) gives 4 non-overlapping windows on the 4×4 patch grid, genuinely exercising the windowing math. Hp and Wp divide cleanly by 2.
- **CLS token bypasses windowed blocks** — i.e. windowed blocks attend on patch tokens only, CLS passes through unchanged. This matches the most common ViT-window implementation. Plan 7 (PyTorch baseline) will reconcile against the actual rfdetr convention; until then, numpy and C++ agree by construction.
- **No padding handling** — patch grid must divide cleanly by window_size. Fixture is configured to satisfy this; real models with non-divisible grids land in Plan 7+.
- **No shifted-window attention** (Swin style) — DINOv2 doesn't use it. Plan 5 is non-shifted only.
- **Existing parity checkpoint names unchanged.** Windowed blocks publish to the same `backbone.block.{i}.attn.output` checkpoint names — the test will verify the actual values, not the names. Names stay stable so Plan 6+ doesn't have to re-thread.

---

## File map (created or modified in this plan)

```
rt-detr.cpp/
├── docs/
│   └── parity.md                       # MODIFY — note window-attention dispatch
├── scripts/
│   └── gen_numpy_baseline.py           # MODIFY — add mha_window + dispatch
├── src/
│   └── dinov2.{cpp,hpp}                # MODIFY — add is_global_block predicate,
│                                       #          mha_window helper, dispatch in dinov2_block
├── tests/
│   ├── fixtures/
│   │   ├── gen_model_gguf.cpp          # MODIFY — bb_window = 2 (was 14)
│   │   ├── model_base_seeded.gguf      # REGENERATED at build
│   │   └── baseline_backbone.gguf      # REGENERATED at build
│   └── test_model_loader.cpp           # MODIFY — assertion bb_window 14 → 2
└── README.md                           # MODIFY — Plan 5 status
```

---

### Task 1: Shrink fixture `window_size` to 2

Single-line config change in the fixture generator + matching test assertion update. Numpy reads `window_size` from the GGUF (well, actually, it doesn't yet — but it will after Task 4). C++ reads `window_size` from the loaded config in subsequent tasks. This task just makes the test fixture's value sensible for the fixture's image dimensions.

**Files:**
- Modify: `tests/fixtures/gen_model_gguf.cpp` — change `bb_window = 14` to `bb_window = 2`
- Modify: `tests/test_model_loader.cpp` — change the assertion `backbone.window_size == 14` to `== 2`

### Step 1: Find and update the fixture default

In `tests/fixtures/gen_model_gguf.cpp`, find the `VariantCfg` struct (top of the file). Find the line:

```cpp
uint32_t bb_window = 14;
```

Change to:

```cpp
uint32_t bb_window = 2;
```

### Step 2: Update matching test assertion

In `tests/test_model_loader.cpp`, find the assertion:

```cpp
RFDETR_ASSERT_EQ_INT(m->config.backbone.window_size, 14);
```

Change `14` to `2`.

### Step 3: Build + run

```bash
rm -rf build
cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=OFF 2>&1 | tail -3
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure
```

Expected: 9/9 pass. The seeded fixture and the parity baseline both regenerate; the parity test still passes because nothing actually dispatches on window_size yet — every block is still running global attention on both sides.

### Step 4: Commit

```bash
git add tests/fixtures/gen_model_gguf.cpp tests/test_model_loader.cpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
test(fixture): shrink window_size to 2 (was 14)

The fixture has 16 patches (4×4 grid) + CLS = 17 tokens. window_size=14
means 14×14=196 tokens per window, which is way bigger than the grid —
windowed attention would degenerate to global. window_size=2 gives 4
non-overlapping 2×2 windows on the 4×4 patch grid, genuinely exercising
the windowing math that Plan 5 is about to add.

Parity remains green because no code currently dispatches on window_size
yet; subsequent Plan 5 tasks add the dispatch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: numpy `mha_window` helper (defined, not yet called)

Add the windowed-attention helper to the numpy reference. Don't dispatch yet — the helper is defined but `forward()` still calls plain `mha()` for every block. This keeps parity green.

**Files:**
- Modify: `scripts/gen_numpy_baseline.py` — add `mha_window` function

### Step 1: Add `mha_window` to the numpy script

In `scripts/gen_numpy_baseline.py`, find the existing `mha(...)` function. Add a new function below it:

```python
def mha_window(x, Wqkv, bqkv, Wproj, bproj, n_heads, window_size, hp, wp):
    """Windowed multi-head self-attention on patch tokens only.

    Args:
        x:           (N+1, dim) — token 0 is CLS, tokens 1..N are patches in row-major (h, w) order
        window_size: int — window side in patches (W). Must divide both hp and wp.
        hp, wp:      int — patch grid height and width. N = hp * wp.

    Returns: (N+1, dim) tensor. CLS at index 0 passes through unchanged (no attention applied);
             patches are window-partitioned, attended per-window, and unpartitioned.

    Layout:
        Input patches shape (hp, wp, dim) in row-major order.
        Partition: (hp/W, W, wp/W, W, dim) -> (hp/W, wp/W, W, W, dim) ->
                   reshape to (n_windows, W*W, dim) where n_windows = (hp/W) * (wp/W).
        Run vanilla MHA on each window (batched along axis 0).
        Reverse partition: (n_windows, W*W, dim) -> (hp/W, wp/W, W, W, dim) ->
                           transpose to (hp/W, W, wp/W, W, dim) -> reshape to (hp, wp, dim) ->
                           flatten to (N, dim).
    """
    N1 = x.shape[0]
    dim = x.shape[1]
    N = hp * wp
    assert N1 == N + 1, f"x has {N1} tokens; expected N+1 = {N+1}"
    assert hp % window_size == 0 and wp % window_size == 0, \
        f"patch grid {hp}x{wp} not divisible by window_size {window_size}"

    cls = x[0:1, :]                                    # (1, dim) — pass through
    patches = x[1:, :]                                  # (N, dim)
    grid = patches.reshape(hp, wp, dim)                 # (hp, wp, dim)

    # Window-partition
    W = window_size
    n_hw = hp // W
    n_ww = wp // W
    windows = (grid
               .reshape(n_hw, W, n_ww, W, dim)
               .transpose(0, 2, 1, 3, 4)
               .reshape(n_hw * n_ww, W * W, dim))      # (n_windows, W*W, dim)

    # Batched MHA — reuse the per-window math from mha() but vectorized over the window axis
    n_windows, T, _ = windows.shape
    head_dim = dim // n_heads
    qkv = windows @ Wqkv.T + bqkv                      # (n_windows, T, 3*dim)
    q, k, v = np.split(qkv, 3, axis=-1)                # each (n_windows, T, dim)
    q = q.reshape(n_windows, T, n_heads, head_dim).transpose(0, 2, 1, 3)  # (n_w, h, T, hd)
    k = k.reshape(n_windows, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(n_windows, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    scale = 1.0 / math.sqrt(head_dim)
    logits = q @ k.transpose(0, 1, 3, 2) * scale       # (n_w, h, T, T)
    logits -= logits.max(-1, keepdims=True)
    a = np.exp(logits)
    a = a / a.sum(-1, keepdims=True)
    attn = a @ v                                       # (n_w, h, T, hd)
    attn = attn.transpose(0, 2, 1, 3).reshape(n_windows, T, dim)  # (n_w, T, dim)
    out = attn @ Wproj.T + bproj                       # (n_w, T, dim)

    # Reverse window-partition
    grid_back = (out
                 .reshape(n_hw, n_ww, W, W, dim)
                 .transpose(0, 2, 1, 3, 4)
                 .reshape(hp, wp, dim))                # (hp, wp, dim)
    patches_out = grid_back.reshape(N, dim)             # (N, dim)

    # Re-prepend CLS (unchanged)
    return np.concatenate([cls, patches_out], axis=0)   # (N+1, dim)
```

Note: this function is defined but not called yet. Task 4 wires the dispatch.

### Step 2: Build + run

```bash
cmake --build build -j 2>&1 | tail -3
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: 9/9 pass. The new function is unused; parity unchanged.

Sanity-check the Python syntactically:

```bash
python3 -c "import ast; ast.parse(open('scripts/gen_numpy_baseline.py').read()); print('OK')"
```

### Step 3: Commit

```bash
git add scripts/gen_numpy_baseline.py
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(parity): numpy mha_window helper (defined, not yet called)

Adds a windowed multi-head self-attention function to the numpy reference.
CLS token is set aside and passes through unchanged; patch tokens are
window-partitioned, attended per-window with vanilla MHA, then
unpartitioned and re-concatenated with CLS.

The function isn't called yet — Task 4 wires the dispatch into forward().
Parity remains green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: C++ `mha_window` helper (defined, not yet called)

Mirror Task 2 on the C++ side. Add the windowed-attention helper to `src/dinov2.cpp` as a file-local function (anonymous namespace). Don't dispatch yet.

**Files:**
- Modify: `src/dinov2.cpp` — add `mha_window` function

### Step 1: Read the current `mha()` in src/dinov2.cpp

Familiarize yourself with the existing `mha()` function (file-local, in the anonymous namespace, defined alongside `layer_norm`, `mlp`). Note its parameter list: `(ctx, x, Wqkv, bqkv, Wproj, bproj, n_heads)`. The windowed version adds three more: `window_size, hp, wp`.

### Step 2: Add `mha_window` after `mha()`

In the anonymous namespace inside `src/dinov2.cpp`, append after `mha()`:

```cpp
/* Windowed multi-head self-attention on patch tokens only.
 *
 * Layout: x ne = (dim, N+1, 1, 1) — token 0 is CLS, tokens 1..N are patches
 * in row-major (h, w) order. hp * wp must equal N. window_size must divide
 * both hp and wp. CLS passes through unchanged; patches are
 * window-partitioned, attended per window via vanilla MHA, then
 * unpartitioned and re-concatenated with CLS.
 *
 * Output: same shape as input. */
ggml_tensor* mha_window(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* Wqkv, ggml_tensor* bqkv,
                        ggml_tensor* Wproj, ggml_tensor* bproj,
                        int n_heads, int window_size, int hp, int wp) {
    const int dim = (int)x->ne[0];
    const int N   = hp * wp;
    const int W   = window_size;
    const int n_hw = hp / W;
    const int n_ww = wp / W;
    const int n_windows = n_hw * n_ww;
    const int T   = W * W;          /* tokens per window */
    const int head_dim = dim / n_heads;

    /* Split off CLS (axis 1, position 0..1) and patches (axis 1, positions 1..N+1). */
    ggml_tensor* cls = ggml_view_2d(ctx, x, dim, 1,
                                    x->nb[1],
                                    /* offset */ 0);
    ggml_tensor* patches = ggml_view_2d(ctx, x, dim, N,
                                        x->nb[1],
                                        /* offset */ x->nb[1]);

    /* Make patches contiguous so subsequent reshape/permute can work. */
    patches = ggml_cont(ctx, patches);
    /* patches ne = (dim, N, 1, 1) */

    /* Window-partition: (dim, N) → grid (dim, wp, hp) → (dim, W, n_ww, W, n_hw) → permute */
    /* In ggml column-major ne, this is (dim, wp, hp, 1). The desired final
     * layout for batched attention is (dim, T, n_windows) where tokens are
     * grouped by window. */
    ggml_tensor* grid = ggml_reshape_3d(ctx, patches, dim, wp, hp);
    /* grid ne = (dim, wp, hp, 1) */

    /* Partition: think of wp = n_ww * W and hp = n_hw * W. We need to rearrange
     * so that the W-sized chunks along wp and hp end up grouped. Step by step:
     *   ne = (dim, n_ww, W, hp, 1)            via reshape splitting wp
     *   ne = (dim, n_ww, W, n_hw, W, 1)        via reshape splitting hp (but ggml
     *                                          reshape doesn't support 5d directly —
     *                                          we'll use sequential 3d/4d reshapes)
     *
     * Strategy: collapse into a 4d view with ne = (dim, W, n_ww, hp), permute, then
     * do the second split. Iterate until layout is correct.
     *
     * The cleanest approach (and one ggml supports well):
     *   grid ne = (dim, wp, hp)
     *   reshape_4d: ne = (dim, W, n_ww, hp)              — split wp axis into (W, n_ww)
     *   permute (0, 2, 1, 3): ne = (dim, n_ww, W, hp)    — interleave so W is at axis 2
     *   cont
     *   reshape_4d: ne = (dim, n_ww, W*hp, 1)            — flatten (W, hp)
     *
     * Actually it's easier to compose two transpose-and-reshapes via a temporary
     * larger tensor. The cleanest Python-friendly equivalent in ggml:
     *
     *   reshape_4d: ne = (dim, W, n_ww, hp)              — wp = n_ww * W
     *   reshape with another split would need 5d. Instead:
     *   permute (0, 1, 3, 2): ne = (dim, W, hp, n_ww)    — move n_ww out of the way
     *   cont
     *   reshape_4d: ne = (dim, W, W, n_hw*n_ww)          — split hp into (W, n_hw)
     *                                                      WAIT this isn't right either.
     *
     * OK pragmatic approach: do the inner reshape as (dim, W*W, n_windows) directly
     * by permuting things one step at a time. The numpy reference is the spec; we just
     * need the C++ tensor ops to land at (dim, T, n_windows) with tokens correctly
     * grouped by window.
     */

    /* Reshape grid (dim, wp, hp) -> (dim, W, n_ww, hp) */
    ggml_tensor* w1 = ggml_reshape_4d(ctx, grid, dim, W, n_ww, hp);

    /* Permute so n_ww moves to axis 2 (rightmost batch-like), preparing for the hp split:
     * (dim, W, n_ww, hp) -> permute(0, 1, 3, 2) -> (dim, W, hp, n_ww) */
    ggml_tensor* w2 = ggml_cont(ctx, ggml_permute(ctx, w1, 0, 1, 3, 2));

    /* Now reshape the hp axis into (W, n_hw): (dim, W, hp, n_ww) -> (dim, W, W, n_hw * n_ww) */
    ggml_tensor* w3 = ggml_reshape_4d(ctx, w2, dim, W, W, n_hw * n_ww);

    /* w3 ne = (dim, W, W, n_windows). Collapse the two W axes into T = W*W. */
    ggml_tensor* windows = ggml_reshape_3d(ctx, w3, dim, T, n_windows);
    /* windows ne = (dim, T, n_windows, 1) — ready for batched attention. */

    /* Batched MHA. ggml_mul_mat broadcasts over the rightmost batch dim:
     *   Wqkv ne = (dim, 3*dim, 1, 1); windows ne = (dim, T, n_windows, 1)
     *   mul_mat result ne = (3*dim, T, n_windows, 1)
     */
    ggml_tensor* qkv = ggml_mul_mat(ctx, Wqkv, windows);
    qkv = ggml_add(ctx, qkv, bqkv);

    /* Split qkv into q, k, v along axis 0 (each takes `dim` slots). */
    const size_t row_size = qkv->nb[1];           /* T stride */
    const size_t col_size = qkv->nb[2];           /* n_windows stride */
    /* Each of q/k/v: ne = (dim, T, n_windows). */
    ggml_tensor* q = ggml_view_3d(ctx, qkv, dim, T, n_windows, row_size, col_size, 0 * dim * sizeof(float));
    ggml_tensor* k = ggml_view_3d(ctx, qkv, dim, T, n_windows, row_size, col_size, 1 * dim * sizeof(float));
    ggml_tensor* v = ggml_view_3d(ctx, qkv, dim, T, n_windows, row_size, col_size, 2 * dim * sizeof(float));

    /* Reshape each (dim, T, n_windows) -> (head_dim, n_heads, T, n_windows) */
    q = ggml_cont(ctx, q);
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);
    q = ggml_reshape_4d(ctx, q, head_dim, n_heads, T, n_windows);
    k = ggml_reshape_4d(ctx, k, head_dim, n_heads, T, n_windows);
    v = ggml_reshape_4d(ctx, v, head_dim, n_heads, T, n_windows);

    /* Permute to (head_dim, T, n_heads, n_windows) so per-head attention groups along axis 2,
     * and the outer batch axis 3 carries n_windows. */
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    /* Attention: kt @ q (manual path; ggml_mul_mat broadcasts over axis 2 and 3) */
    ggml_tensor* kt = ggml_cont(ctx, ggml_permute(ctx, k, 1, 0, 2, 3));
    /* kt ne = (T, head_dim, n_heads, n_windows); q ne = (head_dim, T, n_heads, n_windows) */
    ggml_tensor* logits = ggml_mul_mat(ctx, kt, q);
    /* logits ne = (T, T, n_heads, n_windows) */
    logits = ggml_scale(ctx, logits, 1.0f / std::sqrt((float)head_dim));
    logits = ggml_soft_max(ctx, logits);
    /* attn_out = v_t @ logits */
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    /* vt ne = (T, head_dim, n_heads, n_windows). mul_mat(vt, logits):
     *   ne[0..1] of vt = (T, head_dim); ne[0..1] of logits = (T, T)
     *   result ne = (head_dim, T, n_heads, n_windows) */
    ggml_tensor* attn_out = ggml_mul_mat(ctx, vt, logits);

    /* Merge heads: (head_dim, T, n_heads, n_windows) -> permute to (head_dim, n_heads, T, n_windows)
     * -> reshape to (dim, T, n_windows). */
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    attn_out = ggml_reshape_3d(ctx, attn_out, dim, T, n_windows);

    /* Output projection: Wproj ne = (dim, dim); attn_out ne = (dim, T, n_windows) */
    attn_out = ggml_mul_mat(ctx, Wproj, attn_out);
    attn_out = ggml_add(ctx, attn_out, bproj);

    /* Reverse window-partition: (dim, T, n_windows) -> (dim, W, W, n_windows)
     * -> reshape to (dim, W, W, n_hw * n_ww) — already that shape
     * -> reshape to (dim, W, hp, n_ww) — split n_windows into (n_hw, n_ww)? No, the
     *    partition packed (n_hw, n_ww) into n_windows via row-major. To reverse:
     *    ne[3] = n_windows = n_hw * n_ww. Reshape to (dim, W, W, n_hw, n_ww) would be 5d.
     * Use the same intermediate strategy as Step 1, in reverse:
     */
    ggml_tensor* u1 = ggml_reshape_4d(ctx, attn_out, dim, W, W, n_hw * n_ww);
    /* u1 ne = (dim, W, W, n_windows) where n_windows lays out as (n_hw, n_ww) row-major.
     * Reinterpret n_windows as (W * n_ww, ?) — wait, we want to unpack n_windows into
     * (n_hw, n_ww). With n_windows = n_hw * n_ww and row-major flattening, n_windows is
     * arranged so that fast-varying = n_ww, slow-varying = n_hw. In ggml's column-major
     * ne, the way we flatten (dim, W, W, n_hw, n_ww) → (dim, W, W, n_hw * n_ww) means
     * n_ww is fast-varying (= ne[3] mod n_ww), n_hw is slow-varying. Reshape ne[3] back
     * to (n_ww, n_hw):
     */
    ggml_tensor* u2 = ggml_reshape_4d(ctx, u1, dim, W, W * n_hw, n_ww);
    /* That doesn't decompose cleanly with reshape alone. We need permute-then-reshape:
     * collapse the inner (W, W) into one axis, separate n_windows, permute, then expand.
     *
     * Pragmatic shape choreography below, following the inverse of Step 1's ordering.
     * The key invariant: the final patches tensor must have ne = (dim, wp, hp) with
     * tokens in row-major (h fast-varying inside w? or w fast-varying inside h?)
     * matching numpy's reshape(hp, wp, dim).
     *
     * Numpy partition: grid(hp, wp, dim) -> reshape(n_hw, W, n_ww, W, dim) -> transpose(0,2,1,3,4)
     *                  -> reshape(n_hw*n_ww, W*W, dim)
     * Numpy unpartition: reshape(n_hw, n_ww, W, W, dim) -> transpose(0,2,1,3,4) -> reshape(hp, wp, dim)
     *
     * In ggml column-major, the numpy "last axis is fastest" maps to ggml "axis 0 is fastest".
     * So numpy shape (hp, wp, dim) reads in numpy memory as (dim fastest, then wp, then hp).
     * That's ggml ne = (dim, wp, hp). Confirms our grid shape above.
     *
     * Numpy windows shape (n_windows, T, dim) reads as (dim fastest, then T, then n_windows).
     * In ggml: ne = (dim, T, n_windows). Confirms our windows shape above. */

    /* Reverse path: undo the (dim, W, W, n_windows) → reshape that grouped (n_hw, n_ww) into n_windows
     * row-major numpy-style, which in ggml column-major means n_hw is slow-varying (high index) and
     * n_ww is fast-varying (low index) inside the flat n_windows.
     *
     * Steps:
     *   ne = (dim, W, W, n_windows)
     *   reshape to ne = (dim, W, W, n_ww, n_hw)        — would be 5d, can't directly.
     *   Instead: split n_windows by reshaping to ne = (dim, W*W*n_ww, n_hw)  — collapse interior, split outer
     *   permute to bring n_hw inwards:  (dim, n_hw, W*W*n_ww)? doesn't help.
     *
     * Honest approach: do it as two reshapes + one permute using ggml_reshape_4d twice:
     *   ne = (dim, W, W*n_ww, n_hw)                   — split n_windows into (n_ww, n_hw), interleave with the inner W
     *   permute (0, 1, 3, 2): (dim, W, n_hw, W*n_ww)  — move n_hw inward
     *   cont
     *   reshape to (dim, W, hp, n_ww) since n_hw * W = hp:
     *     ne = (dim, W * n_hw, ???) ...
     *
     * This is getting fiddly. Either:
     *  (A) Use ggml_view_3d/4d offsets to extract each window separately, build hp-sized rows
     *      manually. Slow at graph build but correct.
     *  (B) Compose multiple reshape+permute steps until ne lands at (dim, wp, hp) with the
     *      right element-mapping.
     *
     * Recommend (B) but TEST each intermediate by inspecting tensor data after a forward run.
     * Start with the equivalent inverse of the forward partition: */

    /* INVERSE of: w1=(dim,W,n_ww,hp) -> permute(0,1,3,2)=(dim,W,hp,n_ww) -> reshape=(dim,W,W,n_hw*n_ww) */

    /* From u1=(dim,W,W,n_windows) where n_windows = n_hw*n_ww with n_ww fast and n_hw slow:
     *   reshape to (dim, W, W*n_ww, n_hw) — collapse W (inner spatial) with n_ww (inner window) */
    ggml_tensor* r1 = ggml_reshape_4d(ctx, u1, dim, W, W * n_ww, n_hw);
    /*   permute to (dim, W, n_hw, W*n_ww) — move n_hw inward (was axis 3, now axis 2) */
    ggml_tensor* r2 = ggml_cont(ctx, ggml_permute(ctx, r1, 0, 1, 3, 2));
    /*   reshape to (dim, W*n_hw, W*n_ww) — collapse W with n_hw on axis 1, leaving W*n_ww on axis 2 */
    ggml_tensor* r3 = ggml_reshape_3d(ctx, r2, dim, W * n_hw, W * n_ww);
    /* r3 ne = (dim, hp, wp).  But our target patches ne = (dim, wp, hp), so transpose axes 1 and 2: */
    ggml_tensor* r4 = ggml_cont(ctx, ggml_permute(ctx, r3, 0, 2, 1, 3));
    /* r4 ne = (dim, wp, hp). Reshape to flat (dim, N): */
    ggml_tensor* patches_out = ggml_reshape_2d(ctx, r4, dim, N);

    /* Re-prepend CLS along axis 1. */
    ggml_tensor* full = ggml_concat(ctx, cls, patches_out, /*dim*/ 1);
    /* full ne = (dim, N+1, 1, 1) */
    return full;
}
```

(The shape choreography is intricate and likely to need 1-2 debugging iterations against the numpy reference once dispatch is enabled in Task 4. The above is a best-effort first draft; expect adjustments. The comments explain the intended semantics.)

### Step 3: Build to verify compilation

```bash
cmake --build build -j 2>&1 | tail -15
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: clean build (mha_window is not yet called); 9/9 tests pass.

**If the build fails** because of ggml API issues (e.g. `ggml_view_3d` signature is wrong), check `third_party/ggml/include/ggml.h` and adapt. The implementer is expected to verify each ggml call against the actual header.

### Step 4: Commit

```bash
git add src/dinov2.cpp
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(dinov2): mha_window helper (defined, not yet called)

Anonymous-namespace function mirroring numpy's mha_window. Splits off CLS,
window-partitions patch tokens into (n_windows, T, dim), runs batched MHA
with ggml_mul_mat broadcasting over the window axis, then unpartitions and
re-prepends CLS.

The shape choreography is intricate. Not yet called by dinov2_block;
Task 4 wires the dispatch and is the moment numerical correctness gets
verified against the numpy reference.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Dispatch in `dinov2_block` and numpy `forward()` (parity-bearing commit)

This is the moment both sides actually run the new code path. Critical: BOTH numpy AND C++ must switch dispatch in the same commit so parity stays meaningful.

**Files:**
- Modify: `src/dinov2.hpp` — declare `is_global_block(cfg, i)` predicate
- Modify: `src/dinov2.cpp` — implement `is_global_block`; change `dinov2_block` to dispatch
- Modify: `scripts/gen_numpy_baseline.py` — add the same dispatch in `forward()`

### Step 1: Add the predicate declaration in `src/dinov2.hpp`

Inside `namespace rfdetr`, near the other dinov2_* declarations, add:

```cpp
/* True iff block `i` uses global self-attention (vs windowed).
 *
 * RF-DETR / DINOv2 reuse the multi_scale_layers indices as global-attention
 * blocks: every block whose output gets tapped for the projector also gets
 * the full receptive field of global attention. The intermediate (windowed)
 * blocks attend within local W×W windows only.
 *
 * For the default base variant: globals = {2, 5, 8, 11}, windowed = the rest. */
bool is_global_block(const Config& cfg, uint32_t i);
```

### Step 2: Implement `is_global_block` and update `dinov2_block`

In `src/dinov2.cpp`:

Add the predicate implementation (file-scope, in namespace rfdetr):

```cpp
bool is_global_block(const Config& cfg, uint32_t i) {
    for (uint32_t v : cfg.backbone.multi_scale_layers) {
        if (v == i) return true;
    }
    return false;
}
```

Find `dinov2_block` and locate the line that calls `mha(...)`. Wrap in a dispatch:

```cpp
    ggml_tensor* attn_in = layer_norm(ctx, x, n1w, n1b);
    publish(pub + "norm1.output", attn_in);

    ggml_tensor* attn_out;
    if (is_global_block(m.config, (uint32_t)block_idx)) {
        attn_out = mha(ctx, attn_in, qkvW, qkvB, prW, prB, (int)m.config.backbone.heads);
    } else {
        /* Windowed: derive hp, wp from image_size and patch size 14. */
        const int hp = (int)m.config.image_size / 14;
        const int wp = (int)m.config.image_size / 14;
        attn_out = mha_window(ctx, attn_in, qkvW, qkvB, prW, prB,
                              (int)m.config.backbone.heads,
                              (int)m.config.backbone.window_size, hp, wp);
    }
    publish(pub + "attn.output", attn_out);
    x = ggml_add(ctx, x, attn_out);
```

(The exact variable names depend on the existing code; adapt to match.)

### Step 3: Add the matching dispatch in numpy's `forward()`

In `scripts/gen_numpy_baseline.py`, find the block loop. Find the line that calls `mha(n1, ...)`. Wrap in a dispatch:

```python
        # Dispatch global vs windowed
        if i in cfg["bb_multi_scale_layers"]:
            y = mha(n1,
                    tensors[p + "attn.qkv.weight"], tensors[p + "attn.qkv.bias"],
                    tensors[p + "attn.proj.weight"], tensors[p + "attn.proj.bias"],
                    cfg["bb_heads"])
        else:
            hp = wp = cfg["image_size"] // 14
            y = mha_window(n1,
                           tensors[p + "attn.qkv.weight"], tensors[p + "attn.qkv.bias"],
                           tensors[p + "attn.proj.weight"], tensors[p + "attn.proj.bias"],
                           cfg["bb_heads"],
                           cfg["bb_window_size"], hp, wp)
        out[pub + "attn.output"] = y.copy()
```

The numpy script needs `bb_window_size` in cfg. Update `read_model()` to extract it:

```python
"bb_window_size": _u32(reader, "rfdetr.backbone.window_size"),
```

(Verify the existing `_u32` helper handles the scalar correctly.)

### Step 4: Build and run — expect debugging

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build -R test_parity_backbone --output-on-failure 2>&1 | tail -30
```

**Expect this to fail on the first run** on the windowed blocks (block 0, 1, 3, 4, 6, 7, 9, 10). Global blocks (2, 5, 8, 11) should still pass since their math hasn't changed.

When it fails on a windowed block (likely block 0 first):
1. The numpy reference is the source of truth (we wrote it carefully and it matches the shape choreography we want).
2. The C++ `mha_window` is the more likely source of error — shape choreography is intricate.
3. Debug by comparing C++ output to numpy at the window-partition step. Add a temporary `publish` inside `mha_window` for the partitioned tensor (`backbone.block.{i}.attn.window_partitioned`) and compare numerically.
4. Iterate on the reshape/permute sequence until the unpartitioned output matches.

Common failure modes:
- Window partition order: numpy uses `transpose(0, 2, 1, 3, 4)` — the C++ side may have the W and n_ww axes swapped
- CLS handling: numpy slices `[0:1]` then `[1:]`; C++ uses `ggml_view_2d` with offsets — verify the offset/stride math
- Final concat axis: must be axis 1, same convention as `dinov2_add_cls_and_pos_embed`

Iteration is expected. Budget time for it.

### Step 5: When parity is green

```bash
ctest --test-dir build --output-on-failure
```

Expected: 9/9 pass. Windowed blocks now produce different math than global blocks but both sides produce the same numbers.

### Step 6: Commit (all the iteration commits AND the final dispatch commit)

The dispatch wiring is one commit. Any debug-iteration fixes to `mha_window` (in src/dinov2.cpp) land as additional commits BEFORE the dispatch commit if you split, OR squash into the dispatch commit if everything lands together cleanly.

Recommended approach: commit the dispatch + numpy change first (failing); then commit each `mha_window` fix as you find them. This makes the iteration history visible. Final commit:

```bash
git add src/dinov2.hpp src/dinov2.cpp scripts/gen_numpy_baseline.py
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(dinov2): dispatch global vs windowed attention per block

DINOv2's multi_scale_layers [2,5,8,11] double as the global-attention
block indices (rfdetr/DINOv2's convention: blocks whose output gets
tapped for the projector also receive the full global receptive field;
other blocks attend within local W×W windows). New is_global_block(cfg, i)
predicate reuses the field; no new GGUF metadata.

Both numpy reference and C++ dinov2_block dispatch on the predicate.
Windowed blocks call mha_window (added in Task 3); global blocks call
mha (unchanged). 55 parity checkpoints all green at 1e-5 atol; windowed
blocks' attn.output values now genuinely differ from global blocks'.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(With descriptive messages for any debug-iteration fix commits.)

---

### Task 5: Final smoke + README

**Files:**
- Modify: `docs/parity.md` — small note about window-attention dispatch
- Modify: `README.md` — Plan 5 status

### Step 1: Clean rebuild across configurations

```bash
rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=OFF -DRFDETR_BUILD_CLI=OFF 2>&1 | tail -3
cmake --build build -j 2>&1 | tail -3

rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=OFF 2>&1 | tail -3
cmake --build build -j 2>&1 | tail -3
ctest --test-dir build --output-on-failure 2>&1 | tail -3

rm -rf build && cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=ON 2>&1 | tail -3
cmake --build build -j 2>&1 | tail -3
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: all three configurations green; 9 or 10 tests pass.

### Step 2: Update docs/parity.md

Add a short paragraph (after the tolerance table) explaining the dispatch:

```markdown
## Window vs global attention

Backbone blocks dispatch between two attention paths based on `is_global_block(cfg, i)`:
- Global (= block index ∈ `multi_scale_layers`): standard MHA over all N+1 tokens
- Windowed: CLS bypasses; patches are W×W-window-partitioned, attended per
  window, unpartitioned, then re-concatenated with CLS

Both paths share the same `backbone.block.{i}.attn.output` parity checkpoint;
the test verifies windowed blocks' values match the numpy reference at the
same 1e-5 tolerance as global blocks.
```

### Step 3: Update README.md

Replace the Plan 4 status with:

```markdown
## Status

**Window attention (Plan 5) complete.** The DINOv2 backbone now dispatches
between global and windowed self-attention per block — the four
`multi_scale_layers` indices `[2,5,8,11]` use global attention; the other
eight blocks window the patch tokens into 2×2 partitions and attend within
each window. All 55 parity checkpoints still pass at 1e-5 absolute
tolerance against the numpy reference. Ten tests pass on a clean build.

Plan 6 wires the projector, encoder, decoder, heads, and end-to-end
detect.

The Python conversion script body is still deferred (see Plan 2 Task 3).
The C++ side uses a synthesized F32 GGUF fixture for tests.
```

### Step 4: Commit

```bash
git add docs/parity.md README.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
docs: mark window-attention plan (Plan 5) complete in README

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage** against the design spec:

- §3 Architecture (DINOv2 windowed + global blocks) — fully covered.
- §6 API — `rfdetr_detect` still NOT_IMPLEMENTED; Plan 6 wires it.
- §8 Tests — `test_parity_backbone` covers windowed and global both.
- §9 Parity workflow — windowed dispatch documented in `docs/parity.md`.

**Risk areas:**

1. **`mha_window` shape choreography.** The C++ tensor reshape/permute sequence is intricate and almost certainly will require 1-2 debug iterations against the numpy reference. The Task 4 instructions explicitly budget for this and recommend publishing intermediate tensors for diff.
2. **CLS-bypass semantics.** Numpy slices `x[0:1]` and `x[1:]` cleanly; C++ uses `ggml_view_2d` with offsets. The offset/stride math must match. Likely-first-failure-mode: CLS slot value drifts because C++ either includes it in the windowing or fails to re-prepend cleanly.
3. **`hp * wp != N`** if the fixture grid isn't divisible by `window_size`. Mitigation: Plan 5 hard-codes Hp=Wp=4 and window_size=2 (4/2 = 2), so divisibility is guaranteed. If anyone changes the fixture image_size, they need to ensure divisibility.
4. **`mha_window` performance** — the eight windowed blocks each emit ~20 ggml graph nodes. Plan 6+ will care; Plan 5 prioritizes correctness over efficiency.

---

## Next plan

After this plan lands:

- **Plan 6** — Projector + encoder + decoder + heads + end-to-end `rfdetr_detect`. The `dinov2_forward` API may grow a structured return type (per Plan 4 reviewer recommendation) so the projector can consume multi-scale features explicitly.
- **Plan 7** — Real PyTorch baseline (`scripts/run_rfdetr_baseline.py`) augmenting / replacing numpy.
- **Plan 8** — Quantization (Q8_0).
- **Plan 9** — Variants nano/small/medium/large.
