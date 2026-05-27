#ifndef RFDETR_DECODER_HPP
#define RFDETR_DECODER_HPP

#include "model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace rfdetr {

/* LWDETR decoder (rfdetr-base): 3 layers of self-attn + deformable cross-attn
 * + FFN, followed by a final LayerNorm.
 *
 * Per-layer (post-LN):
 *   q = k = tgt + query_pos
 *   sa  = self_attn(q, k, v=tgt)              # nn.MultiheadAttention, 8 heads
 *   tgt = norm1(tgt + sa)
 *   ca  = deformable_cross_attn(tgt+query_pos, memory, refpoints)  # MSDeformAttn, 16 heads, 2 points
 *   tgt = norm2(tgt + ca)
 *   ffn = linear2(relu(linear1(tgt)))
 *   tgt = norm3(tgt + ffn)
 *
 * With lite_refpoint_refine=True and bbox_reparam=True, refpoints are NOT
 * refined inside the decoder — every layer receives the same initial
 * refpoints (from the two-stage init). The deformable cross-attention uses
 * the 4D (cx, cy, w, h) form: sampling_locations = ref_xy + sampling_offsets
 * / n_points * ref_wh * 0.5.
 *
 * Query positional embedding comes from gen_sineembed_for_position(refpoints)
 * → 512-dim sine-cosine encoding → ref_point_head (2-layer MLP) → 256-dim
 * query_pos. Computed once before the layer loop (lite_refpoint_refine).
 *
 * Inputs:
 *   tgt        ne = (model_dim=256, num_queries=300, 1)   content queries
 *   memory     ne = (model_dim=256, N_in=H*W=1600, 1)     post-projector memory
 *   refpoints  ne = (4, num_queries=300, 1)               (cx, cy, w, h)
 *
 *   H, W       spatial extent of memory feature map (40x40 for rfdetr-base);
 *              passed via model.config.image_size / patch_size.
 *
 * Outputs (published to trace callbacks; final returned):
 *   decoder.layer.{i}.self_attn.output     (256, 300, 1)
 *   decoder.layer.{i}.norm1.output         (256, 300, 1)
 *   decoder.layer.{i}.cross_attn.output    (256, 300, 1)
 *   decoder.layer.{i}.norm2.output         (256, 300, 1)
 *   decoder.layer.{i}.linear1.output       (2048, 300, 1)
 *   decoder.layer.{i}.linear2.output       (256, 300, 1)
 *   decoder.layer.{i}.output               (256, 300, 1)   full layer
 *   decoder.norm.output                    (256, 300, 1)   final
 *
 * Returns the final-norm output ne = (256, 300, 1). nullptr on error. */
ggml_tensor* decoder_forward(ggml_context* ctx, const Model& m,
                             ggml_tensor* tgt,
                             ggml_tensor* memory,
                             ggml_tensor* refpoints,
                             ggml_tensor* query_pos,
                             int memory_H, int memory_W);

/* Variant of decoder_forward that also returns per-layer post-norm outputs.
 *
 * For non-seg paths, the decoder only consumes the final layer (the bbox /
 * class heads run on the stacked `hs[-1]`). The SegmentationHead, however,
 * iterates each (block, decoder_layer_output) pair — so the seg path needs
 * each layer's output AFTER the decoder.norm has been applied.
 *
 * Behavior matches `TransformerDecoder.forward(return_intermediate=True)` in
 * rfdetr's transformer.py:475: append `self.norm(output)` after every layer.
 *
 * On success returns the same final-norm output as decoder_forward, and
 * populates `*out_per_layer` with `m.config.decoder.layers` tensors (each
 * ne=(256, NQ, 1)) — the i-th entry is the i-th layer's post-norm output.
 *
 * The caller must reserve space in `out_per_layer` for `decoder.layers`
 * pointers BEFORE calling. */
ggml_tensor* decoder_forward_with_intermediates(
    ggml_context* ctx, const Model& m,
    ggml_tensor* tgt,
    ggml_tensor* memory,
    ggml_tensor* refpoints,
    ggml_tensor* query_pos,
    int memory_H, int memory_W,
    ggml_tensor** out_per_layer);

/* Compute query_sine_embed for a (4, num_queries) refpoints buffer.
 *
 * Output layout: (4 * d_half, num_queries) F32 where d_half = 128. Token-major
 * (num_queries varies slowest), channel layout: [pos_y(128) | pos_x(128) |
 * pos_w(128) | pos_h(128)] — matching torch's gen_sineembed_for_position
 * concat order.
 *
 * Each per-component 128-dim block alternates sin/cos pairs:
 *   v = comp * 2*pi
 *   for k in [0, 64): out[2k]   = sin(v / 10000^(2k/128))
 *                    out[2k+1] = cos(v / 10000^(2k/128))
 *
 * Exposed for tests + the decoder pre-pass. CPU-only.  */
void compute_query_sine_embed(const float* refpoints, int num_queries,
                              int d_half, float* out);

}  // namespace rfdetr

#endif
