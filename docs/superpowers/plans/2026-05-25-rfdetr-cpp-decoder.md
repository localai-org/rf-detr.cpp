# rt-detr.cpp Decoder Implementation Plan (Plan 6b of 11)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the RF-DETR transformer decoder: 300 learnable queries flow through 3 decoder layers, each doing self-attention over queries, cross-attention against the encoder output, and a feed-forward MLP. End state: parity-green C++ pipeline from input image through decoder, ready for Plan 6c to attach class+bbox heads and wire `rfdetr_detect` end-to-end. 18 new parity checkpoints (1 queries-init + 5 per layer × 3 layers + 2 trailing = 18, total 91).

**Architecture:** One new TU — `src/decoder.{cpp,hpp}`. The decoder layer is structurally a DETR-style decoder layer: pre-LN → self-attn → residual → pre-LN → cross-attn → residual → pre-LN → MLP → residual. Reuses `ops::layer_norm`, `ops::mha`, `ops::mlp` from Plan 6a; adds a new `ops::cross_attn` to `transformer_ops` (cross-attention has separate Q vs KV projections, distinct from self-attn's packed QKV). Numpy reference grows correspondingly. Decoder consumes `decoder.queries` (a learnable `(model_dim, num_queries)` constant from the GGUF) and the encoder output from Plan 6a.

**Tech Stack:** Same as Plans 1-6a (C++17, ggml CPU backend, numpy + gguf). No new dependencies.

---

## Scope decisions

- **Cross-attention as a new shared op** in `transformer_ops`. Q projection is separate from KV; both Q and KV inputs are passed explicitly. Plan 6c may also need it if any heads use attention (currently they don't — class+bbox heads are MLPs). Even so, decoder is the only consumer for Plan 6b.
- **Decoder queries are a constant tensor**, not an embedding lookup or positional encoding addition. The fixture writes `decoder.queries` ne = `(model_dim, num_queries)` = `(64, 300)`. We feed it as the layer-0 input directly.
- **Three layer norms per decoder layer** (norm1 pre-self-attn, norm2 pre-cross-attn, norm3 pre-FFN) — this matches the fixture's tensor naming (Plan 2 Task 5 wrote `decoder.layers.{i}.norm{1,2,3}.{weight,bias}`).
- **No decoder positional encoding** — RF-DETR uses learned queries that effectively act as positional anchors. Plan 7 will verify against the real model.
- **Test name unchanged for now** — `test_parity_backbone.cpp` continues to grow checkpoints. Plan 6c may rename to `test_parity_full_forward.cpp` when heads land.

---

## File map (created or modified in this plan)

```
rt-detr.cpp/
├── docs/
│   └── parity.md                       # MODIFY — add decoder checkpoint rows
├── scripts/
│   └── gen_numpy_baseline.py           # MODIFY — add cross_attn, decoder_layer, decoder loop
├── src/
│   ├── transformer_ops.{cpp,hpp}       # MODIFY — add ops::cross_attn
│   ├── decoder.{cpp,hpp}               # NEW — decoder_layer + decoder_forward
│   └── (other src files unchanged)
├── tests/
│   ├── CMakeLists.txt                  # MODIFY — link src/decoder.cpp
│   └── test_parity_backbone.cpp        # MODIFY — call decoder_forward; new checkpoints
└── README.md                           # MODIFY — Plan 6b status
```

---

### Task 1: `ops::cross_attn` helper

Add a cross-attention function to `transformer_ops`. Defined but not called by any current code path; Task 2's decoder layer will use it.

**Files:**
- Modify: `src/transformer_ops.hpp` — declare `cross_attn`
- Modify: `src/transformer_ops.cpp` — implement
- Modify: `scripts/gen_numpy_baseline.py` — add `cross_attn` function (defined, not called)

### Step 1: Add declaration to `src/transformer_ops.hpp`

Inside `namespace rfdetr::ops`, after the existing declarations:

```cpp
/* Multi-head cross-attention.
 *
 *   q_in ne   = (dim, N_q, 1, 1)         queries
 *   kv_in ne  = (dim, N_kv, 1, 1)        keys/values source
 *   Wq ne     = (dim, dim)               query projection
 *   bq ne     = (dim,)
 *   Wkv ne    = (dim, 2*dim)             packed K+V projection from kv_in
 *   bkv ne    = (2*dim,)
 *   Wo ne     = (dim, dim)               output projection
 *   bo ne     = (dim,)
 *
 * Output ne = (dim, N_q, 1, 1) — same shape as queries.
 *
 * Math (per head):
 *   q = Wq @ q_in + bq                  // (dim, N_q) projected
 *   kv = Wkv @ kv_in + bkv              // (2*dim, N_kv) → split into k, v
 *   logits = q^T @ k / sqrt(head_dim)   // (N_q, N_kv)
 *   attn = softmax(logits)
 *   out = v @ attn^T                    // (dim, N_q)
 *   out = Wo @ out + bo */
ggml_tensor* cross_attn(ggml_context* ctx,
                        ggml_tensor* q_in, ggml_tensor* kv_in,
                        ggml_tensor* Wq, ggml_tensor* bq,
                        ggml_tensor* Wkv, ggml_tensor* bkv,
                        ggml_tensor* Wo, ggml_tensor* bo,
                        int n_heads);
```

### Step 2: Implement in `src/transformer_ops.cpp`

The implementation mirrors `mha` but with separate Q and KV projections. Append to the file:

```cpp
ggml_tensor* cross_attn(ggml_context* ctx,
                        ggml_tensor* q_in, ggml_tensor* kv_in,
                        ggml_tensor* Wq, ggml_tensor* bq,
                        ggml_tensor* Wkv, ggml_tensor* bkv,
                        ggml_tensor* Wo, ggml_tensor* bo,
                        int n_heads) {
    const int dim  = (int)q_in->ne[0];
    const int N_q  = (int)q_in->ne[1];
    const int N_kv = (int)kv_in->ne[1];
    const int head_dim = dim / n_heads;

    /* Q projection: Wq @ q_in + bq → (dim, N_q) */
    ggml_tensor* q = ggml_mul_mat(ctx, Wq, q_in);
    q = ggml_add(ctx, q, bq);

    /* KV projection: Wkv @ kv_in + bkv → (2*dim, N_kv). Split along axis 0. */
    ggml_tensor* kv = ggml_mul_mat(ctx, Wkv, kv_in);
    kv = ggml_add(ctx, kv, bkv);
    /* kv ne = (2*dim, N_kv) */
    const size_t kv_row_stride = kv->nb[1];
    ggml_tensor* k = ggml_view_2d(ctx, kv, dim, N_kv,
                                  kv_row_stride,
                                  /* offset bytes */ 0 * dim * sizeof(float));
    ggml_tensor* v = ggml_view_2d(ctx, kv, dim, N_kv,
                                  kv_row_stride,
                                  1 * dim * sizeof(float));
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);

    /* Reshape to per-head: (dim, N) -> (head_dim, n_heads, N) */
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, N_q);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, N_kv);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, N_kv);

    /* Permute to (head_dim, N, n_heads): so head dim is the batch axis */
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  /* (head_dim, N_q, n_heads) */
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  /* (head_dim, N_kv, n_heads) */
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));  /* (head_dim, N_kv, n_heads) */

    /* Attention: mul_mat(k, q) computes k^T @ q per head
     *   k ne = (head_dim, N_kv, n_heads); q ne = (head_dim, N_q, n_heads)
     *   result ne = (N_kv, N_q, n_heads)
     * Then scale + softmax along the N_kv axis (axis 0). */
    ggml_tensor* logits = ggml_mul_mat(ctx, k, q);
    logits = ggml_scale(ctx, logits, 1.0f / std::sqrt((float)head_dim));
    logits = ggml_soft_max(ctx, logits);
    /* logits ne = (N_kv, N_q, n_heads) — each q row sums to 1 across N_kv */

    /* out = v @ logits per head. We need (head_dim, N_q, n_heads).
     *   v ne = (head_dim, N_kv, n_heads); logits ne = (N_kv, N_q, n_heads)
     *   mul_mat(v, logits) = v^T @ logits — would give (N_kv, N_q) per head
     *   wrong dim. Instead permute v to put N_kv first:
     *     vt ne = (N_kv, head_dim, n_heads)
     *     mul_mat(vt, logits) = (head_dim, N_q, n_heads)  ✓
     */
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    ggml_tensor* out = ggml_mul_mat(ctx, vt, logits);
    /* out ne = (head_dim, N_q, n_heads) */

    /* Merge heads: (head_dim, N_q, n_heads) → permute → (head_dim, n_heads, N_q) → reshape (dim, N_q) */
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    out = ggml_reshape_2d(ctx, out, dim, N_q);

    /* Output projection */
    out = ggml_mul_mat(ctx, Wo, out);
    out = ggml_add(ctx, out, bo);
    return out;
}
```

### Step 3: Add numpy `cross_attn`

In `scripts/gen_numpy_baseline.py`, after the existing `mha` function:

```python
def cross_attn(q_in, kv_in, Wq, bq, Wkv, bkv, Wo, bo, n_heads):
    """Multi-head cross-attention.

    Args:
        q_in:   (N_q, dim)  — queries
        kv_in:  (N_kv, dim) — keys/values source
        Wq:     (dim, dim)
        bq:     (dim,)
        Wkv:    (2*dim, dim)
        bkv:    (2*dim,)
        Wo:     (dim, dim)
        bo:     (dim,)

    Returns: (N_q, dim).
    """
    N_q, dim = q_in.shape
    N_kv, _ = kv_in.shape
    head_dim = dim // n_heads

    q = q_in @ Wq.T + bq                          # (N_q, dim)
    kv = kv_in @ Wkv.T + bkv                      # (N_kv, 2*dim)
    k, v = np.split(kv, 2, axis=-1)               # each (N_kv, dim)

    q = q.reshape(N_q, n_heads, head_dim).transpose(1, 0, 2)   # (h, N_q, hd)
    k = k.reshape(N_kv, n_heads, head_dim).transpose(1, 0, 2)  # (h, N_kv, hd)
    v = v.reshape(N_kv, n_heads, head_dim).transpose(1, 0, 2)  # (h, N_kv, hd)

    scale = 1.0 / math.sqrt(head_dim)
    logits = q @ k.transpose(0, 2, 1) * scale     # (h, N_q, N_kv)
    logits -= logits.max(-1, keepdims=True)
    a = np.exp(logits)
    a = a / a.sum(-1, keepdims=True)

    attn = a @ v                                  # (h, N_q, hd)
    attn = attn.transpose(1, 0, 2).reshape(N_q, dim)  # (N_q, dim)
    out = attn @ Wo.T + bo                        # (N_q, dim)
    return out
```

### Step 4: Build to verify

```bash
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -3
```

Expected: 9/9 pass (CLI=OFF). The new function is unused; parity unchanged.

`ast.parse` check on the Python script:

```bash
python3 -c "import ast; ast.parse(open('scripts/gen_numpy_baseline.py').read()); print('OK')"
```

### Step 5: Commit

```bash
git add src/transformer_ops.hpp src/transformer_ops.cpp scripts/gen_numpy_baseline.py
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(ops): cross_attn helper (defined, not yet called)

Multi-head cross-attention with separate Q vs packed-KV projections. Q is
projected from the query input; K and V are projected from a different
input (encoder output for the decoder). Both numpy and C++ versions added;
not yet called — Task 2's decoder_layer wires it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Single decoder layer + parity

Implement one decoder layer (norm1 → self-attn → residual → norm2 → cross-attn → residual → norm3 → MLP → residual). Numpy reference + parity. Just layer 0 for now.

**Files:**
- Create: `src/decoder.hpp`, `src/decoder.cpp`
- Modify: `CMakeLists.txt` — add `src/decoder.cpp` to `RFDETR_SOURCES`
- Modify: `scripts/gen_numpy_baseline.py` — add `decoder_layer`; add `dec_heads`/`dec_layers` to cfg
- Modify: `tests/test_parity_backbone.cpp` — load decoder queries, call layer 0, add 4 checkpoints
- Modify: `docs/parity.md` — new rows

### Step 1: Write `src/decoder.hpp`

```cpp
#ifndef RFDETR_DECODER_HPP
#define RFDETR_DECODER_HPP

#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace rfdetr {

/* One DETR-style transformer decoder layer (pre-LN):
 *   q = q + self_attn(norm1(q))
 *   q = q + cross_attn(norm2(q), encoder_out)
 *   q = q + mlp(norm3(q))
 *
 * Input:
 *   q ne           = (model_dim, num_queries, 1, 1)
 *   encoder_out ne = (model_dim, n_enc_tokens, 1, 1)
 *
 * Output: same shape as q.
 *
 * Publishes:
 *   decoder.layer{idx}.self_attn.output
 *   decoder.layer{idx}.cross_attn.output
 *   decoder.layer{idx}.mlp.output
 *   decoder.layer{idx}.output */
ggml_tensor* decoder_layer(ggml_context* ctx, const Model& m,
                           ggml_tensor* q, ggml_tensor* encoder_out,
                           int layer_idx);

/* decoder_forward declared here; implementation lands in Task 3. */
ggml_tensor* decoder_forward(ggml_context* ctx, const Model& m,
                             ggml_tensor* encoder_out);

}  // namespace rfdetr

#endif
```

### Step 2: Write `src/decoder.cpp` (just `decoder_layer`)

```cpp
#include "decoder.hpp"
#include "transformer_ops.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"

#include <string>

namespace rfdetr {

ggml_tensor* decoder_layer(ggml_context* ctx, const Model& m,
                           ggml_tensor* q, ggml_tensor* encoder_out,
                           int layer_idx) {
    const std::string p = "decoder.layers." + std::to_string(layer_idx) + ".";
    auto get = [&](const char* suffix) -> ggml_tensor* {
        auto it = m.tensors.find(p + suffix);
        if (it == m.tensors.end()) {
            rfdetr_logf(RFDETR_LOG_ERROR, "decoder_layer: missing tensor '%s'",
                        (p + suffix).c_str());
            return nullptr;
        }
        return it->second;
    };

    /* Self-attention weights (packed QKV like encoder) */
    ggml_tensor* n1w   = get("norm1.weight");
    ggml_tensor* n1b   = get("norm1.bias");
    ggml_tensor* sqkvW = get("self_attn.qkv.weight");
    ggml_tensor* sqkvB = get("self_attn.qkv.bias");
    ggml_tensor* soW   = get("self_attn.out.weight");
    ggml_tensor* soB   = get("self_attn.out.bias");

    /* Cross-attention weights (separate Q vs packed KV) */
    ggml_tensor* n2w   = get("norm2.weight");
    ggml_tensor* n2b   = get("norm2.bias");
    ggml_tensor* cqW   = get("cross_attn.q.weight");
    ggml_tensor* cqB   = get("cross_attn.q.bias");
    ggml_tensor* ckvW  = get("cross_attn.kv.weight");
    ggml_tensor* ckvB  = get("cross_attn.kv.bias");
    ggml_tensor* coW   = get("cross_attn.out.weight");
    ggml_tensor* coB   = get("cross_attn.out.bias");

    /* MLP weights */
    ggml_tensor* n3w   = get("norm3.weight");
    ggml_tensor* n3b   = get("norm3.bias");
    ggml_tensor* f1W   = get("ffn.fc1.weight");
    ggml_tensor* f1B   = get("ffn.fc1.bias");
    ggml_tensor* f2W   = get("ffn.fc2.weight");
    ggml_tensor* f2B   = get("ffn.fc2.bias");

    if (!n1w || !n1b || !sqkvW || !sqkvB || !soW || !soB ||
        !n2w || !n2b || !cqW || !cqB || !ckvW || !ckvB || !coW || !coB ||
        !n3w || !n3b || !f1W || !f1B || !f2W || !f2B) {
        return nullptr;
    }

    const std::string pub = "decoder.layer" + std::to_string(layer_idx) + ".";

    /* q = q + self_attn(norm1(q)) */
    ggml_tensor* y = ops::layer_norm(ctx, q, n1w, n1b);
    y = ops::mha(ctx, y, sqkvW, sqkvB, soW, soB, (int)m.config.decoder.heads);
    publish(pub + "self_attn.output", y);
    q = ggml_add(ctx, q, y);

    /* q = q + cross_attn(norm2(q), encoder_out) */
    y = ops::layer_norm(ctx, q, n2w, n2b);
    y = ops::cross_attn(ctx, y, encoder_out,
                        cqW, cqB, ckvW, ckvB, coW, coB,
                        (int)m.config.decoder.heads);
    publish(pub + "cross_attn.output", y);
    q = ggml_add(ctx, q, y);

    /* q = q + mlp(norm3(q)) */
    y = ops::layer_norm(ctx, q, n3w, n3b);
    ggml_tensor* mlp_out = ops::mlp(ctx, y, f1W, f1B, f2W, f2B);
    publish(pub + "mlp.output", mlp_out);
    q = ggml_add(ctx, q, mlp_out);

    publish(pub + "output", q);
    return q;
}

/* Stub — Task 3 implements. */
ggml_tensor* decoder_forward(ggml_context* /*ctx*/, const Model& /*m*/,
                             ggml_tensor* /*encoder_out*/) {
    return nullptr;
}

}  // namespace rfdetr
```

### Step 3: Update CMakeLists.txt

Add `src/decoder.cpp` to `RFDETR_SOURCES`.

### Step 4: Extend numpy reference

In `scripts/gen_numpy_baseline.py`, first add to `read_model()`:

```python
"dec_heads": _u32(reader, "rfdetr.decoder.heads"),
"dec_layers": _u32(reader, "rfdetr.decoder.layers"),
```

Then add the `decoder_layer` function (after `encoder_layer`):

```python
def decoder_layer(cfg, tensors, q, encoder_out, i):
    """One decoder layer.

    Args:
        q:           (num_queries, model_dim)
        encoder_out: (n_enc_tokens, model_dim)

    Returns: (out_dict, q_next).
    """
    p = f"decoder.layers.{i}."
    pub = f"decoder.layer{i}."
    out = {}

    # Self-attention
    y = layer_norm(q, tensors[p + "norm1.weight"], tensors[p + "norm1.bias"])
    y = mha(y,
            tensors[p + "self_attn.qkv.weight"], tensors[p + "self_attn.qkv.bias"],
            tensors[p + "self_attn.out.weight"], tensors[p + "self_attn.out.bias"],
            cfg["dec_heads"])
    out[pub + "self_attn.output"] = y.copy()
    q = q + y

    # Cross-attention
    y = layer_norm(q, tensors[p + "norm2.weight"], tensors[p + "norm2.bias"])
    y = cross_attn(y, encoder_out,
                   tensors[p + "cross_attn.q.weight"],  tensors[p + "cross_attn.q.bias"],
                   tensors[p + "cross_attn.kv.weight"], tensors[p + "cross_attn.kv.bias"],
                   tensors[p + "cross_attn.out.weight"],tensors[p + "cross_attn.out.bias"],
                   cfg["dec_heads"])
    out[pub + "cross_attn.output"] = y.copy()
    q = q + y

    # MLP
    y = layer_norm(q, tensors[p + "norm3.weight"], tensors[p + "norm3.bias"])
    z = mlp(y,
            tensors[p + "ffn.fc1.weight"], tensors[p + "ffn.fc1.bias"],
            tensors[p + "ffn.fc2.weight"], tensors[p + "ffn.fc2.bias"])
    out[pub + "mlp.output"] = z.copy()
    q = q + z

    out[pub + "output"] = q.copy()
    return out, q
```

Wire one decoder layer call into `forward()` after the encoder loop:

```python
    # Decoder queries: stored as (num_queries, model_dim) in numpy (gguf-py
    # reverses ggml's (model_dim, num_queries) layout)
    queries = tensors["decoder.queries"]
    out["decoder.queries"] = queries.copy()

    dec_out, q = decoder_layer(cfg, tensors, queries, x_enc, 0)
    out.update(dec_out)
```

### Step 5: Wire into the C++ test

In `tests/test_parity_backbone.cpp`, add `#include "decoder.hpp"` at the top.

After the encoder call:

```cpp
/* Decoder queries: load the learnable embedding tensor and use as layer 0 input */
auto it_q = m->tensors.find("decoder.queries");
RFDETR_ASSERT(it_q != m->tensors.end());
ggml_tensor* queries = it_q->second;
rfdetr::publish("decoder.queries", queries);  /* publish for parity */

ggml_tensor* dec = rfdetr::decoder_layer(gctx, *m, queries, enc, /*layer_idx*/ 0);
RFDETR_ASSERT(dec != nullptr);

/* Keep upstream tensors alive in the graph */
ggml_build_forward_expand(graph, dec);
/* (Earlier expands for bb.final, projected, enc already exist in the file) */
```

Add 5 new tolerances in `build_tolerances`:

```cpp
tol["decoder.queries"] = {1e-5f, 1e-4f};
{
    int i = 0;
    std::string p = "decoder.layer" + std::to_string(i) + ".";
    tol[p + "self_attn.output"]  = {1e-5f, 1e-4f};
    tol[p + "cross_attn.output"] = {1e-5f, 1e-4f};
    tol[p + "mlp.output"]        = {1e-5f, 1e-4f};
    tol[p + "output"]            = {1e-5f, 1e-4f};
}
```

### Step 6: Update docs/parity.md

Add 5 rows for `decoder.queries` and `decoder.layer0.{self_attn,cross_attn,mlp,output}`.

### Step 7: Build + run

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build -R test_parity_backbone --output-on-failure 2>&1 | tail -20
```

Expected outcomes ranked by likelihood:

1. **PASS first try** — self-attn and MLP have identical math to encoder (parity-proven); cross-attn shape choreography is new but symmetric to mha.
2. **Cross-attn parity fails** — most likely cause: kv split direction in C++ (`ggml_view_2d` offset for K vs V) doesn't match numpy's `np.split(kv, 2, axis=-1)`. Debug by adding a temporary publish for the per-head Q tensor and comparing.
3. **decoder.queries values differ** — the queries tensor is constant-loaded; mismatch implies a gguf-py reading issue. Unlikely since the loader is well-tested.

If parity fails, debug the earliest failing checkpoint. The first divergence will tell you which sublayer is wrong.

### Step 8: Commit

```bash
git add src/decoder.hpp src/decoder.cpp CMakeLists.txt \
        scripts/gen_numpy_baseline.py tests/test_parity_backbone.cpp docs/parity.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(decoder): single decoder layer

Pre-LN decoder layer: self-attn(queries) → cross-attn(queries → encoder) →
MLP. Uses ops::mha, ops::cross_attn (new), ops::mlp from transformer_ops.
Numpy reference matches; 5 new parity checkpoints (decoder.queries +
decoder.layer0.{self_attn,cross_attn,mlp,output}) all green at
{1e-5, 1e-4} tolerance.

Task 3 loops all 3 decoder layers.

[Note any debug iterations needed.]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Loop all 3 decoder layers + `decoder_forward`

Mirror Plan 6a Task 4's structure for the encoder. Loop over `cfg.decoder.layers` (= 3) and wrap in `decoder_forward`.

**Files:**
- Modify: `src/decoder.cpp` — implement `decoder_forward` (replace Task 2 stub)
- Modify: `scripts/gen_numpy_baseline.py` — loop decoder layers
- Modify: `tests/test_parity_backbone.cpp` — call `decoder_forward`; programmatic tolerance loop
- Modify: `docs/parity.md` — add rows for layer1, layer2, decoder.output

### Step 1: Implement `decoder_forward`

Replace the stub:

```cpp
ggml_tensor* decoder_forward(ggml_context* ctx, const Model& m,
                             ggml_tensor* encoder_out) {
    auto it_q = m.tensors.find("decoder.queries");
    if (it_q == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "decoder_forward: missing decoder.queries");
        return nullptr;
    }
    ggml_tensor* q = it_q->second;
    publish("decoder.queries", q);

    for (uint32_t i = 0; i < m.config.decoder.layers; ++i) {
        q = decoder_layer(ctx, m, q, encoder_out, (int)i);
        if (!q) return nullptr;
    }
    publish("decoder.output", q);
    return q;
}
```

### Step 2: numpy loop

In `gen_numpy_baseline.py::forward()`, replace the single decoder_layer call with:

```python
    queries = tensors["decoder.queries"]
    out["decoder.queries"] = queries.copy()
    q = queries
    for i in range(cfg["dec_layers"]):
        dec_out, q = decoder_layer(cfg, tensors, q, x_enc, i)
        out.update(dec_out)
    out["decoder.output"] = q.copy()
```

### Step 3: C++ test calls `decoder_forward`

Replace the single-layer call in `tests/test_parity_backbone.cpp`:

```cpp
/* Remove the manual queries lookup + publish + single layer call from Task 2.
 * Replace with: */
ggml_tensor* dec = rfdetr::decoder_forward(gctx, *m, enc);
RFDETR_ASSERT(dec != nullptr);

ggml_build_forward_expand(graph, dec);
```

In `build_tolerances`, replace the single-layer tolerance block with a loop:

```cpp
tol["decoder.queries"] = {1e-5f, 1e-4f};
for (uint32_t i = 0; i < cfg.decoder.layers; ++i) {
    std::string p = "decoder.layer" + std::to_string(i) + ".";
    tol[p + "self_attn.output"]  = {1e-5f, 1e-4f};
    tol[p + "cross_attn.output"] = {1e-5f, 1e-4f};
    tol[p + "mlp.output"]        = {1e-5f, 1e-4f};
    tol[p + "output"]            = {1e-5f, 1e-4f};
}
tol["decoder.output"] = {1e-5f, 1e-4f};
```

### Step 4: docs/parity.md

Add rows for layer1 (4), layer2 (4), and `decoder.output` (1) = 9 new rows.

### Step 5: Build + run

```bash
cmake --build build -j 2>&1 | tail -10
ctest --test-dir build -R test_parity_backbone --output-on-failure 2>&1 | tail -10
```

Expected: PASS. Should mirror Plan 6a Task 4's behavior (flat numerical drift across layers).

### Step 6: Commit

```bash
git add src/decoder.cpp scripts/gen_numpy_baseline.py tests/test_parity_backbone.cpp docs/parity.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
feat(decoder): loop all 3 decoder layers + decoder_forward wrapper

Extends the single-layer impl from Task 2 to all cfg.decoder.layers (= 3
for base). decoder_forward loads decoder.queries, runs the loop, publishes
decoder.output after the last layer.

14 total decoder checkpoints (queries + 4×3 + output) all green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Final smoke + README

**Files:**
- Modify: `README.md` — Plan 6b status

### Step 1: Triple-config clean rebuild

Same pattern as prior plans. Verify all three CMake configurations and tests pass.

### Step 2: Update README.md Status section

Replace the existing Plan 6a status with:

```markdown
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
```

### Step 3: Commit

```bash
git add README.md
git -c user.name="mudler" -c user.email="mudler@localai.io" commit -m "$(cat <<'EOF'
docs: mark decoder plan (Plan 6b) complete in README

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- §3 Architecture (decoder) — covered. Heads + end-to-end detect in Plan 6c.
- §6 API — `rfdetr_detect` still NOT_IMPLEMENTED; Plan 6c wires it.
- §8 Tests — `test_parity_backbone` now covers everything through the decoder.
- §9 Parity — 91 checkpoints total; documented.

**Risk areas:**
1. **Cross-attention shape choreography** — Q has N_q=300 tokens while KV has N_kv=64 (4×16 from encoder). The asymmetric shapes are the new variable. Most likely failure mode at parity time.
2. **KV split direction** — `ggml_view_2d` offset for K vs V must match numpy's `np.split(kv, 2, axis=-1)`. K is the first half (offset 0); V is the second half (offset `dim*sizeof(float)`).
3. **decoder.queries shape** — fixture writes ne=`(model_dim, num_queries)` = `(64, 300)`. Numpy reads as `(300, 64)`. Use directly without transposition.

---

## Next plan

After this plan lands:

- **Plan 6c** — Class head (MLP → num_classes), bbox head (3-layer MLP → 4), `rfdetr_model_forward`, wire `rfdetr_detect` end-to-end. CLI `detect` returns real JSON detections.
- **Plan 7** — Real PyTorch baseline replacing numpy.
- **Plan 8** — Quantization.
- **Plan 9** — Variants nano/small/medium/large.
