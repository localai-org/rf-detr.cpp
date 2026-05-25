#!/usr/bin/env python3
"""Generate a baseline-bundle GGUF from a Plan-2 model GGUF by computing
forward through patch_embed + CLS/pos_embed + all backbone blocks in numpy.

The intermediate tensors are written to a new GGUF as parity.<checkpoint>
tensors. The C++ test_parity_backbone consumes this bundle.

Usage:
    python3 scripts/gen_numpy_baseline.py \
        --model tests/fixtures/model_base_seeded.gguf \
        --output tests/fixtures/baseline_backbone.gguf \
        [--input-seed 7]

Format version: "1" (see docs/parity.md).
"""

import argparse
import math
import sys

import numpy as np

try:
    import gguf
except ImportError:
    print("error: 'gguf' package not installed. pip install gguf",
          file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Plan-2 GGUF (seeded fixture).")
    p.add_argument("--output", required=True)
    p.add_argument("--input-seed", type=int, default=7,
                   help="Seed for the deterministic synthetic input image.")
    return p.parse_args()


def _u32(reader, key):
    """Extract a uint32 scalar from a gguf-py field."""
    f = reader.get_field(key)
    if f is None:
        raise KeyError(f"missing metadata: {key}")
    # gguf-py field's last `parts` entry holds the typed numpy array value
    return int(f.parts[-1][0])


def _str(reader, key):
    f = reader.get_field(key)
    if f is None:
        raise KeyError(f"missing metadata: {key}")
    return bytes(f.parts[-1]).decode("utf-8")


def read_model(path):
    """Open a Plan-2 GGUF and return (config_dict, tensors_dict).

    Tensors come back in PyTorch convention (out, in, ...) — the gguf-py
    reader reverses ggml's column-major ne[] into numpy shape automatically.
    """
    reader = gguf.GGUFReader(path)

    # Multi-scale taps: INT32 array in the GGUF metadata. For typed arrays,
    # gguf-py stores each element as its own entry in `field.parts`, and
    # `field.data` holds the list of indices into `parts` for those elements
    # (so parts[-1] would only be the LAST element). We walk `field.data` to
    # collect every element.
    ms_field = reader.get_field("rfdetr.backbone.multi_scale_layers")
    if ms_field is None:
        raise KeyError("missing rfdetr.backbone.multi_scale_layers")
    ms_layers = [int(ms_field.parts[idx][0]) for idx in ms_field.data]

    cfg = {
        "variant":     _str(reader, "rfdetr.variant"),
        "image_size":  _u32(reader, "rfdetr.image_size"),
        "num_queries": _u32(reader, "rfdetr.num_queries"),
        "num_classes": _u32(reader, "rfdetr.num_classes"),
        "bb_dim":      _u32(reader, "rfdetr.backbone.dim"),
        "bb_depth":    _u32(reader, "rfdetr.backbone.depth"),
        "bb_heads":    _u32(reader, "rfdetr.backbone.heads"),
        "bb_window_size": _u32(reader, "rfdetr.backbone.window_size"),
        "bb_multi_scale_layers": ms_layers,
        "enc_layers":  _u32(reader, "rfdetr.encoder.layers"),
        "enc_heads":   _u32(reader, "rfdetr.encoder.heads"),
        "dec_layers":  _u32(reader, "rfdetr.decoder.layers"),
        "dec_heads":   _u32(reader, "rfdetr.decoder.heads"),
    }

    tensors = {}
    for t in reader.tensors:
        # gguf-py: t.data is the typed numpy array already reshaped for us
        # (it reverses ggml's ne[] order, so shape is PyTorch-style).
        arr = np.array(t.data, copy=True)
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        tensors[t.name] = arr
    return cfg, tensors


def layer_norm(x, w, b, eps=1e-5):
    """Last-axis LayerNorm with affine. Matches torch.nn.LayerNorm."""
    mean = x.mean(-1, keepdims=True)
    var  = x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def gelu_erf(x):
    """Exact erf-based GELU (PyTorch torch.nn.GELU() default, matches
    ggml_gelu_erf on the C++ side). """
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def mha(x, Wqkv, bqkv, Wproj, bproj, n_heads):
    """Multi-head self-attention with packed QKV.

    Shapes (PyTorch convention):
      x       (N, dim)
      Wqkv    (3*dim, dim)
      bqkv    (3*dim,)
      Wproj   (dim, dim)
      bproj   (dim,)
    Returns (N, dim) post-projection — matches the C++ side's published
    `attn.output` (which is post-Wproj).
    """
    N, dim = x.shape
    head_dim = dim // n_heads
    qkv = x @ Wqkv.T + bqkv                            # (N, 3*dim)
    q, k, v = np.split(qkv, 3, axis=-1)                # each (N, dim)
    q = q.reshape(N, n_heads, head_dim).transpose(1, 0, 2)  # (h, N, hd)
    k = k.reshape(N, n_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(N, n_heads, head_dim).transpose(1, 0, 2)
    scale = 1.0 / math.sqrt(head_dim)
    logits = q @ k.transpose(0, 2, 1) * scale          # (h, N, N)
    logits -= logits.max(-1, keepdims=True)            # softmax stability
    a = np.exp(logits)
    a = a / a.sum(-1, keepdims=True)
    attn = a @ v                                       # (h, N, hd)
    attn = attn.transpose(1, 0, 2).reshape(N, dim)     # (N, dim)
    out = attn @ Wproj.T + bproj                       # (N, dim)
    return out


def cross_attn(q_in, kv_in, Wq, bq, Wkv, bkv, Wo, bo, n_heads):
    """Multi-head cross-attention.

    Args:
        q_in:   (N_q, dim)   — queries
        kv_in:  (N_kv, dim)  — keys/values source
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


def mlp(x, W1, b1, W2, b2):
    """fc1 -> erf-GELU -> fc2. PyTorch shapes:
       W1 (ffn_dim, dim), W2 (dim, ffn_dim)."""
    h = x @ W1.T + b1
    h = gelu_erf(h)
    return h @ W2.T + b2


def projector_forward(cfg, tensors, multi_scale):
    """Multi-scale projector.

    Args:
        multi_scale: list of 4 numpy arrays, each (N+1, backbone.dim)
                     where token 0 is CLS, tokens 1..N are patches.

    Returns: (out_dict, concat) where:
        out_dict: dict with intermediate tensors:
            - projector.level{0..3}.output: per-level (N, model_dim)
            - projector.concat.output:       (4*N, model_dim)
        concat: the final concatenated tensor
    """
    out = {}
    n_levels = len(cfg["bb_multi_scale_layers"])
    level_embed = tensors["projector.level_embed"]  # numpy shape: (n_levels, model_dim)
                                                    # (gguf-py reverses ggml's (model_dim, n_levels))

    projected = []
    for j in range(n_levels):
        feat = multi_scale[j]                       # (N+1, backbone.dim)
        patches = feat[1:, :]                       # strip CLS → (N, backbone.dim)

        Wj = tensors[f"projector.level{j}.weight"]  # numpy shape: (model_dim, backbone.dim)
        bj = tensors[f"projector.level{j}.bias"]    # (model_dim,)
        y = patches @ Wj.T + bj                     # (N, model_dim)

        y = y + level_embed[j]                      # add level-j embedding (broadcast over N)

        out[f"projector.level{j}.output"] = y.copy()
        projected.append(y)

    concat = np.concatenate(projected, axis=0)      # (4*N, model_dim)
    out["projector.concat.output"] = concat.copy()
    return out, concat


def encoder_layer(cfg, tensors, x, i):
    """One encoder layer. x: (N_tokens, model_dim). Returns: (out_dict, x_next)."""
    p = f"encoder.layers.{i}."
    pub = f"encoder.layer{i}."
    out = {}

    n1 = layer_norm(x, tensors[p + "norm1.weight"], tensors[p + "norm1.bias"])
    out[pub + "norm1.output"] = n1.copy()
    y = mha(n1,
            tensors[p + "self_attn.qkv.weight"], tensors[p + "self_attn.qkv.bias"],
            tensors[p + "self_attn.out.weight"], tensors[p + "self_attn.out.bias"],
            cfg["enc_heads"])
    out[pub + "attn.output"] = y.copy()
    x = x + y

    n2 = layer_norm(x, tensors[p + "norm2.weight"], tensors[p + "norm2.bias"])
    z = mlp(n2,
            tensors[p + "ffn.fc1.weight"], tensors[p + "ffn.fc1.bias"],
            tensors[p + "ffn.fc2.weight"], tensors[p + "ffn.fc2.bias"])
    out[pub + "mlp.output"] = z.copy()
    x = x + z

    out[pub + "output"] = x.copy()
    return out, x


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


def patchify_and_embed(input_img, kernel, bias):
    """Replicate Conv2d(kernel=stride=14, padding=0).

    Inputs (PyTorch convention):
      input_img  (1, 3, H, W)   — note: NCHW
      kernel     (dim, 3, 14, 14)
      bias       (dim,)
    Returns tokens (N, dim) where N = (H/14) * (W/14).

    Uses an explicit unfold + matmul (clearer than scipy.signal):
      patches  (N, 3*14*14)
      kernel'  (dim, 3*14*14)
      tokens = patches @ kernel'.T + bias
    """
    _, C, H, W = input_img.shape
    assert C == 3, "expected NCHW input with 3 channels"
    Hp = H // 14
    Wp = W // 14
    # Stay in NCHW, unfold into (N_patches, C, 14, 14) so that the inner
    # 588 axis goes (channel, row, col) — same order as the PyTorch-shape
    # kernel (dim, 3, 14, 14) when flattened to (dim, 588). This is what
    # PyTorch's nn.Conv2d effectively contracts over.
    img = input_img[0]                              # (C, H, W)
    img = img.reshape(C, Hp, 14, Wp, 14)             # (C, Hp, 14, Wp, 14)
    # Move (Hp, Wp) to the front: (Hp, Wp, C, 14, 14)
    img = img.transpose(1, 3, 0, 2, 4)
    patches = img.reshape(Hp * Wp, C * 14 * 14)     # (N, C*14*14)
    K = kernel.reshape(kernel.shape[0], -1)         # (dim, C*14*14)
    return patches @ K.T + bias                     # (N, dim)


def forward(cfg, tensors, input_img):
    """Run patch_embed + CLS/pos_embed + all backbone blocks. Returns dict of
    named intermediate tensors. All values stored as float32, shape (N, dim) or
    compatible — the C++ side will reshape them as needed via
    copy_tensor_to_f32 + tensor_shape.
    """
    out = {"preprocess.input": input_img}  # NCHW (1, 3, H, W)

    pew = tensors["backbone.patch_embed.weight"]  # (dim, 3, 14, 14)
    peb = tensors["backbone.patch_embed.bias"]    # (dim,)

    tokens = patchify_and_embed(input_img, pew, peb)   # (N, dim)
    out["backbone.patch_embed.output"] = tokens.copy()

    # ---- CLS token + positional embedding ----
    # cls_token: ggml shape (dim,) -> numpy (dim,) (1D stays 1D after gguf-py reversal)
    # pos_embed: ggml shape (dim, N+1) -> numpy (N+1, dim) (gguf-py reverses axes)
    cls_token = tensors["backbone.cls_token"]
    pos_embed = tensors["backbone.pos_embed"]

    # tokens has shape (N, dim) from patchify_and_embed.
    cls_flat = cls_token.reshape(1, -1)                          # (1, dim)
    tokens_with_cls = np.concatenate([cls_flat, tokens], axis=0) # (N+1, dim)

    if pos_embed.shape != tokens_with_cls.shape:
        raise ValueError(
            f"pos_embed shape {pos_embed.shape} != tokens+cls shape "
            f"{tokens_with_cls.shape}; check fixture writer convention")

    x = tokens_with_cls + pos_embed                              # (N+1, dim)
    out["backbone.cls_pos_embed.output"] = x.copy()

    for i in range(cfg["bb_depth"]):
        p = f"backbone.blocks.{i}."
        pub = f"backbone.block.{i}."

        # x = x + attn(norm1(x))
        n1 = layer_norm(x, tensors[p + "norm1.weight"], tensors[p + "norm1.bias"])
        out[pub + "norm1.output"] = n1.copy()
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
        x = x + y

        # x = x + mlp(norm2(x))
        n2 = layer_norm(x, tensors[p + "norm2.weight"], tensors[p + "norm2.bias"])
        z = mlp(n2,
                tensors[p + "mlp.fc1.weight"], tensors[p + "mlp.fc1.bias"],
                tensors[p + "mlp.fc2.weight"], tensors[p + "mlp.fc2.bias"])
        out[pub + "mlp.output"] = z.copy()
        x = x + z

        out[pub + "output"] = x.copy()

        # Multi-scale tap: backbone output after this block, if this i is a tap layer
        ms = cfg["bb_multi_scale_layers"]
        if i in ms:
            level = ms.index(i)
            out[f"backbone.multiscale.level{level}"] = x.copy()

    # ---- Final backbone LayerNorm ----
    fn = layer_norm(x,
                    tensors["backbone.norm.weight"],
                    tensors["backbone.norm.bias"])
    out["backbone.norm.output"] = fn.copy()
    x = fn

    # ---- Multi-scale projector ----
    multi_scale = [out[f"backbone.multiscale.level{j}"]
                   for j in range(len(cfg["bb_multi_scale_layers"]))]
    proj_out, proj_concat = projector_forward(cfg, tensors, multi_scale)
    out.update(proj_out)

    # ---- Encoder (all layers) ----
    x_enc = proj_concat
    for i in range(cfg["enc_layers"]):
        enc_out, x_enc = encoder_layer(cfg, tensors, x_enc, i)
        out.update(enc_out)
    out["encoder.output"] = x_enc.copy()

    # ---- Decoder (layer 0 — Task 3 will loop all layers) ----
    # Decoder queries: stored as (num_queries, model_dim) in numpy
    # (gguf-py reverses ggml's (model_dim, num_queries) layout)
    queries = tensors["decoder.queries"]
    out["decoder.queries"] = queries.copy()

    dec_out, q = decoder_layer(cfg, tensors, queries, x_enc, 0)
    out.update(dec_out)

    return out


def write_baseline(path, intermediates, input_shape):
    w = gguf.GGUFWriter(path, arch="rfdetr-parity")
    w.add_string("parity.format.version", "1")
    w.add_string("parity.reference", "numpy")
    w.add_array("parity.input_shape", [int(x) for x in input_shape])
    for name, arr in intermediates.items():
        w.add_tensor(f"parity.{name}", arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main():
    args = parse_args()
    cfg, tensors = read_model(args.model)

    rng = np.random.default_rng(args.input_seed)
    H = W = cfg["image_size"]
    # ImageNet-normalized input: roughly N(0, ~1) after standard normalization.
    # NCHW convention to match PyTorch.
    input_img = rng.standard_normal(size=(1, 3, H, W)).astype(np.float32)

    intermediates = forward(cfg, tensors, input_img)
    write_baseline(args.output, intermediates, input_img.shape)

    print(f"Wrote {args.output} with {len(intermediates)} parity tensors",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
