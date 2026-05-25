#!/usr/bin/env python3
"""Generate a baseline-bundle GGUF from a Plan-2 model GGUF by computing
forward through patch_embed + CLS/pos_embed + all backbone blocks in numpy.

The intermediate tensors are written to a new GGUF as parity.<checkpoint>
tensors. The C++ test_parity_block0 consumes this bundle.

Usage:
    python3 scripts/gen_numpy_baseline.py \
        --model tests/fixtures/model_base_seeded.gguf \
        --output tests/fixtures/baseline_block0.gguf \
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

    cfg = {
        "variant":     _str(reader, "rfdetr.variant"),
        "image_size":  _u32(reader, "rfdetr.image_size"),
        "num_queries": _u32(reader, "rfdetr.num_queries"),
        "num_classes": _u32(reader, "rfdetr.num_classes"),
        "bb_dim":      _u32(reader, "rfdetr.backbone.dim"),
        "bb_depth":    _u32(reader, "rfdetr.backbone.depth"),
        "bb_heads":    _u32(reader, "rfdetr.backbone.heads"),
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


def mlp(x, W1, b1, W2, b2):
    """fc1 -> erf-GELU -> fc2. PyTorch shapes:
       W1 (ffn_dim, dim), W2 (dim, ffn_dim)."""
    h = x @ W1.T + b1
    h = gelu_erf(h)
    return h @ W2.T + b2


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
