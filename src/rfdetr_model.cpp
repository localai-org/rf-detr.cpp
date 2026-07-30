#include "rfdetr_model.hpp"
#include "dinov2.hpp"
#include "projector.hpp"
#include "two_stage.hpp"
#include "decoder.hpp"
#include "heads.hpp"
#include "segmentation.hpp"
#include "trace.hpp"
#include "common.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace rfdetr {

namespace {

ggml_tensor* fetch(const Model& m, const std::string& name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "rfdetr_model_forward: missing tensor '%s'",
                    name.c_str());
        return nullptr;
    }
    return it->second;
}

/* Pick the top-K token indices by `cls_all.max(-1)` (per-token max class score),
 * matching `enc_outputs_class_unselected.max(-1)[0]` in
 * rfdetr/models/transformer.py:282. Returns K indices (descending by score). */
std::vector<int> top_k_by_max_class(const std::vector<float>& cls_all,
                                    int N_tokens, int num_classes, int K) {
    std::vector<float> max_per_token((size_t)N_tokens);
    for (int t = 0; t < N_tokens; ++t) {
        const float* row = cls_all.data() + (size_t)t * num_classes;
        float mx = row[0];
        for (int c = 1; c < num_classes; ++c) {
            if (row[c] > mx) mx = row[c];
        }
        max_per_token[(size_t)t] = mx;
    }

    std::vector<int> idx((size_t)N_tokens);
    std::iota(idx.begin(), idx.end(), 0);
    /* Partial sort: top K with largest scores first; torch.topk's ordering is
     * also descending. */
    const int k = std::min(K, N_tokens);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) {
                          return max_per_token[(size_t)a] > max_per_token[(size_t)b];
                      });
    idx.resize((size_t)k);
    return idx;
}

}  // namespace

ForwardOutput rfdetr_model_forward(const Model& m,
                                   const float* input_data, int input_size,
                                   ggml_backend_t backend) {
    /* Thin shim for legacy / test paths: wrap the single backend in a
     * BackendCtx with no BLAS, no sched. Goes through the same forward()
     * code below but skips the sched dispatch. */
    BackendCtx bctx{};
    bctx.cpu = backend;
    return rfdetr_model_forward(m, input_data, input_size, bctx);
}

ForwardOutput rfdetr_model_forward(const Model& m,
                                   const float* input_data, int input_size,
                                   BackendCtx& bctx) {
    ForwardOutput out;
    out.num_queries = (int)m.config.num_queries;
    out.num_classes = (int)m.config.num_classes;

    ggml_backend_t backend = bctx.cpu;  // buffer alloc + tensor I/O backend

    if (!input_data || input_size <= 0 || !backend) {
        rfdetr_logf(RFDETR_LOG_ERROR, "rfdetr_model_forward: invalid args");
        return out;
    }
    if (input_size != (int)m.config.image_size) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "rfdetr_model_forward: input_size %d != config.image_size %u",
                    input_size, m.config.image_size);
        return out;
    }

    const int model_dim = (int)m.config.decoder.model_dim;       // 256
    const int NQ        = (int)m.config.num_queries;             // 300
    const int NC        = (int)m.config.num_classes;             // 91
    const int feat_side = input_size / (int)m.config.patch_size; // 40 for 560/14
    const int N_tokens  = feat_side * feat_side;                 // 1600

    /* =====================================================================
     * Graph A: backbone + projector + two_stage init
     * Produces cls_all, bbox_all, enc_output_norm_out (memory tokens).
     * ===================================================================== */
    ggml_init_params ipA{};
    ipA.mem_size   = 256 * 1024 * 1024;
    ipA.mem_buffer = nullptr;
    ipA.no_alloc   = true;
    ggml_context* gctxA = ggml_init(ipA);
    if (!gctxA) {
        rfdetr_logf(RFDETR_LOG_ERROR, "rfdetr_model_forward: ggml_init (A) failed");
        return out;
    }

    ggml_tensor* input_t = ggml_new_tensor_4d(gctxA, GGML_TYPE_F32,
                                              input_size, input_size, 3, 1);
    ggml_set_name(input_t, "input");
    /* Mark as a graph input — the gallocr places input tensors at the
     * beginning of the buffer in non-overlapping addresses, so it's safe
     * to ggml_backend_tensor_set() into them before compute. */
    ggml_set_input(input_t);

    BackboneOutput bb = dinov2_forward(gctxA, m, input_t);
    if (!bb.final) {
        rfdetr_logf(RFDETR_LOG_ERROR, "rfdetr_model_forward: backbone failed");
        ggml_free(gctxA);
        return out;
    }
    ggml_tensor* proj = projector_forward(gctxA, m, bb);
    if (!proj) {
        ggml_free(gctxA);
        return out;
    }
    TwoStageOutput ts = two_stage_forward(gctxA, m, proj);
    if (!ts.cls_all || !ts.bbox_all || !ts.enc_output_norm_out) {
        ggml_free(gctxA);
        return out;
    }

    /* Flatten projector for downstream decoder memory: ne (W, H, 256, 1) →
     * (256, W*H, 1) using the same permute(1, 2, 0, 3) + reshape as
     * two_stage. This is the same source two_stage uses internally for the
     * Linear projection, but the decoder consumes the RAW (unprojected)
     * projector output for cross-attention's value_proj. */
    ggml_tensor* memory_tokens = ggml_cont(gctxA, ggml_permute(gctxA, proj, 1, 2, 0, 3));
    memory_tokens = ggml_reshape_3d(gctxA, memory_tokens, model_dim, N_tokens, 1);
    ggml_set_name(memory_tokens, "decoder.memory");

    /* Mark outputs so the gallocr doesn't recycle their storage before we
     * read them back. */
    ggml_set_output(ts.cls_all);
    ggml_set_output(ts.bbox_all);
    ggml_set_output(memory_tokens);
    /* For seg models we also need the projector output (W, H, C, 1) on
     * the host so we can re-stage it into graph B alongside the per-layer
     * decoder outputs. */
    const bool has_seg = m.config.has_segmentation_head;
    if (has_seg) {
        ggml_set_output(proj);
    }

    ggml_cgraph* graphA = ggml_new_graph_custom(gctxA, /*size*/ 16384, /*grads*/ false);
    ggml_build_forward_expand(graphA, ts.cls_all);
    ggml_build_forward_expand(graphA, ts.bbox_all);
    ggml_build_forward_expand(graphA, memory_tokens);
    if (has_seg) {
        ggml_build_forward_expand(graphA, proj);
    }

    /* Allocate buffers for the graph (sched on GPU, persistent gallocr on
     * CPU — the gallocr packs intermediate tensors with lifetime-aware reuse
     * and keeps the underlying compute buffer alive across calls, avoiding
     * the ~55 ms/iter `free(1.9 GB)` munmap that otherwise dominates
     * non-compute overhead). Inputs are set AFTER alloc, before compute. */
    if (!backend_ctx_graph_alloc(bctx, graphA, /*which_graph*/ 0)) {
        rfdetr_logf(RFDETR_LOG_ERROR, "rfdetr_model_forward: graph A alloc failed");
        ggml_free(gctxA);
        return out;
    }

    ggml_backend_tensor_set(input_t, input_data, 0,
                            (size_t)input_size * input_size * 3 * sizeof(float));

    ggml_status stA = (ggml_status)backend_ctx_graph_compute(bctx, graphA, /*which_graph*/ 0);
    if (stA != GGML_STATUS_SUCCESS) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "rfdetr_model_forward: graphA compute returned %d", (int)stA);
        ggml_free(gctxA);
        return out;
    }

    /* Read host-side. cls_all ne = (91, 1600, 1) — class fastest-varying:
     * memory layout (class, token) so cls[t * 91 + c] = score for token t,
     * class c. bbox_all ne = (4, 1600, 1) — (cx,cy,w,h) per token. */
    std::vector<float> cls_all((size_t)NC * N_tokens);
    std::vector<float> bbox_all((size_t)4 * N_tokens);
    std::vector<float> memory_flat((size_t)model_dim * N_tokens);
    ggml_backend_tensor_get(ts.cls_all,     cls_all.data(),     0, cls_all.size()     * sizeof(float));
    ggml_backend_tensor_get(ts.bbox_all,    bbox_all.data(),    0, bbox_all.size()    * sizeof(float));
    ggml_backend_tensor_get(memory_tokens,  memory_flat.data(), 0, memory_flat.size() * sizeof(float));

    /* For seg models also fetch the projector output in its native
     * (W, H, C, 1) ggml layout — needed as input to the seg head. */
    std::vector<float> proj_data;
    if (has_seg) {
        proj_data.resize((size_t)feat_side * feat_side * model_dim);
        ggml_backend_tensor_get(proj, proj_data.data(), 0,
                                proj_data.size() * sizeof(float));
    }

    /* gctxA owns the graph + tensor metadata. Tensor data lives in the
     * gallocr's buffer (kept alive in BackendCtx). */
    ggml_free(gctxA);

    /* =====================================================================
     * CPU step: top-K, gather refpoints, combine with learned
     * `decoder.queries.refpoints` via bbox_reparam to produce the decoder's
     * initial reference points.
     * ===================================================================== */
    std::vector<int> topk = top_k_by_max_class(cls_all, N_tokens, NC, NQ);
    if ((int)topk.size() != NQ) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "rfdetr_model_forward: top-K returned %zu (expected %d)",
                    topk.size(), NQ);
        return out;
    }

    /* Read the learned `decoder.queries.refpoints` weight (4, 300). The
     * Model::tensors map has it as a graph descriptor (data lives in the
     * weights backend buffer). Use ggml_backend_tensor_get to copy. */
    ggml_tensor* lrn_rp = fetch(m, "decoder.queries.refpoints");
    if (!lrn_rp) return out;
    std::vector<float> learned_refpoints((size_t)4 * NQ);
    ggml_backend_tensor_get(lrn_rp, learned_refpoints.data(), 0,
                            learned_refpoints.size() * sizeof(float));

    /* Per rfdetr/models/transformer.py:316-324 (bbox_reparam=True,
     * lite_refpoint_refine=True path):
     *   refpoint_embed_cxcy = refpoint_embed_ts_subset[:2] * top_K_box[2:] +
     *                          top_K_box[:2]
     *   refpoint_embed_wh   = exp(refpoint_embed_ts_subset[2:]) * top_K_box[2:]
     *
     * where `top_K_box` = bbox_all gathered at top-K indices (already in
     * (cx,cy,w,h) form post-reparam from two_stage), and
     * `refpoint_embed_ts_subset` = learned `decoder.queries.refpoints[:NQ]`. */
    std::vector<float> decoder_refpoints((size_t)4 * NQ);
    for (int q = 0; q < NQ; ++q) {
        const int t = topk[(size_t)q];
        const float* box  = bbox_all.data() + (size_t)t * 4;          // (cx, cy, w, h)
        const float* lrn  = learned_refpoints.data() + (size_t)q * 4; // (lcx, lcy, lw, lh)
        float* dst = decoder_refpoints.data() + (size_t)q * 4;
        dst[0] = lrn[0] * box[2] + box[0];                 // new_cx
        dst[1] = lrn[1] * box[3] + box[1];                 // new_cy
        dst[2] = std::exp(lrn[2]) * box[2];                // new_w
        dst[3] = std::exp(lrn[3]) * box[3];                // new_h
    }

    /* Sine-cosine embedding of the refpoints → ref_point_head input. */
    const int d_half = model_dim / 2;  // 128
    std::vector<float> sine_data((size_t)4 * d_half * NQ);
    compute_query_sine_embed(decoder_refpoints.data(), NQ, d_half, sine_data.data());

    /* Content queries = learned `decoder.queries.feat` (256, 300). Read it
     * once here so we can stage it into graph B. */
    ggml_tensor* lrn_feat = fetch(m, "decoder.queries.feat");
    if (!lrn_feat) return out;
    std::vector<float> tgt_data((size_t)model_dim * NQ);
    ggml_backend_tensor_get(lrn_feat, tgt_data.data(), 0,
                            tgt_data.size() * sizeof(float));

    /* =====================================================================
     * Graph B: decoder (with ref_point_head MLP in-graph) + heads.
     * Outputs class_logits + bbox_delta; bbox_reparam is applied on CPU
     * after read-back (cheap, 300×4 floats).
     * ===================================================================== */
    ggml_init_params ipB{};
    ipB.mem_size   = 256 * 1024 * 1024;
    ipB.mem_buffer = nullptr;
    ipB.no_alloc   = true;
    ggml_context* gctxB = ggml_init(ipB);
    if (!gctxB) {
        rfdetr_logf(RFDETR_LOG_ERROR, "rfdetr_model_forward: ggml_init (B) failed");
        return out;
    }

    ggml_tensor* tgt_in = ggml_new_tensor_3d(gctxB, GGML_TYPE_F32, model_dim, NQ, 1);
    ggml_set_name(tgt_in, "decoder.tgt");
    ggml_set_input(tgt_in);
    ggml_tensor* mem_in = ggml_new_tensor_3d(gctxB, GGML_TYPE_F32, model_dim, N_tokens, 1);
    ggml_set_name(mem_in, "decoder.memory");
    ggml_set_input(mem_in);
    ggml_tensor* rp_in  = ggml_new_tensor_3d(gctxB, GGML_TYPE_F32, 4, NQ, 1);
    ggml_set_name(rp_in, "decoder.refpoints");
    ggml_set_input(rp_in);
    ggml_tensor* sine_in = ggml_new_tensor_3d(gctxB, GGML_TYPE_F32, 4 * d_half, NQ, 1);
    ggml_set_name(sine_in, "decoder.sine_embed");
    ggml_set_input(sine_in);

    /* For seg models: also stage the projector output in (W, H, C, 1) layout
     * so the seg head can consume it without re-running the backbone. */
    ggml_tensor* proj_in = nullptr;
    if (has_seg) {
        proj_in = ggml_new_tensor_4d(gctxB, GGML_TYPE_F32,
                                     feat_side, feat_side, model_dim, 1);
        ggml_set_name(proj_in, "seg.projector_in");
        ggml_set_input(proj_in);
    }

    /* ref_point_head: 2-layer MLP `MLP(2*d_model=512, d_model=256, d_model, 2)`
     *   Linear(512, 256) → ReLU → Linear(256, 256). */
    ggml_tensor* rph_w0 = fetch(m, "decoder.ref_point_head.layers.0.weight");
    ggml_tensor* rph_b0 = fetch(m, "decoder.ref_point_head.layers.0.bias");
    ggml_tensor* rph_w1 = fetch(m, "decoder.ref_point_head.layers.1.weight");
    ggml_tensor* rph_b1 = fetch(m, "decoder.ref_point_head.layers.1.bias");
    if (!rph_w0 || !rph_b0 || !rph_w1 || !rph_b1) {
        ggml_free(gctxB);
        return out;
    }
    ggml_tensor* qpos = ggml_mul_mat(gctxB, rph_w0, sine_in);
    qpos = ggml_add(gctxB, qpos, rph_b0);
    qpos = ggml_relu(gctxB, qpos);
    qpos = ggml_mul_mat(gctxB, rph_w1, qpos);
    qpos = ggml_add(gctxB, qpos, rph_b1);
    ggml_set_name(qpos, "decoder.query_pos");

    ggml_tensor* dec_out = nullptr;
    /* Per-layer post-norm outputs (only populated for seg models). The
     * SegmentationHead iterates over zip(self.blocks, query_features) — each
     * decoder layer's post-norm output is one stream. */
    const int n_dec_layers = (int)m.config.decoder.layers;
    std::vector<ggml_tensor*> dec_per_layer((size_t)n_dec_layers, nullptr);
    if (has_seg) {
        dec_out = decoder_forward_with_intermediates(
            gctxB, m, tgt_in, mem_in, rp_in, qpos,
            feat_side, feat_side, dec_per_layer.data());
    } else {
        dec_out = decoder_forward(gctxB, m, tgt_in, mem_in, rp_in,
                                  qpos, feat_side, feat_side);
    }
    if (!dec_out) {
        ggml_free(gctxB);
        return out;
    }
    ggml_tensor* cls_logits_t = class_head_forward(gctxB, m, dec_out);
    ggml_tensor* bbox_delta_t = bbox_head_forward(gctxB, m, dec_out);
    if (!cls_logits_t || !bbox_delta_t) {
        ggml_free(gctxB);
        return out;
    }

    /* SegmentationHead. SegmentationHead.__init__ constructs exactly one
     * DepthwiseConvBlock per decoder layer (num_blocks == n_dec_layers), so
     * PyTorch's zip(self.blocks, query_features) always pairs 1:1 with no
     * truncation. nano/small have 4 decoder layers, medium/large have 5,
     * xlarge/2xlarge have 6 — always iterate all of them. */
    ggml_tensor* seg_masks_t = nullptr;
    if (has_seg) {
        seg_masks_t = segmentation_forward(
            gctxB, m,
            proj_in,
            dec_per_layer.data(),
            n_dec_layers,
            /*image_h*/ input_size, /*image_w*/ input_size,
            (int)m.config.mask_downsample_ratio);
        if (!seg_masks_t) {
            ggml_free(gctxB);
            return out;
        }
        ggml_set_output(seg_masks_t);
    }

    ggml_set_output(cls_logits_t);
    ggml_set_output(bbox_delta_t);

    ggml_cgraph* graphB = ggml_new_graph_custom(gctxB, /*size*/ 16384, /*grads*/ false);
    ggml_build_forward_expand(graphB, cls_logits_t);
    ggml_build_forward_expand(graphB, bbox_delta_t);
    if (seg_masks_t) {
        ggml_build_forward_expand(graphB, seg_masks_t);
    }

    /* Same alloc-then-set-then-compute pattern as graphA. See the comment at
     * graph A's alloc for the rationale. Inputs are set AFTER alloc, before
     * compute. */
    if (!backend_ctx_graph_alloc(bctx, graphB, /*which_graph*/ 1)) {
        rfdetr_logf(RFDETR_LOG_ERROR, "rfdetr_model_forward: graph B alloc failed");
        ggml_free(gctxB);
        return out;
    }

    ggml_backend_tensor_set(tgt_in,  tgt_data.data(),  0, tgt_data.size()  * sizeof(float));
    ggml_backend_tensor_set(mem_in,  memory_flat.data(), 0, memory_flat.size() * sizeof(float));
    ggml_backend_tensor_set(rp_in,   decoder_refpoints.data(), 0, decoder_refpoints.size() * sizeof(float));
    ggml_backend_tensor_set(sine_in, sine_data.data(), 0, sine_data.size() * sizeof(float));
    if (has_seg && proj_in) {
        ggml_backend_tensor_set(proj_in, proj_data.data(), 0,
                                proj_data.size() * sizeof(float));
    }

    ggml_status stB = (ggml_status)backend_ctx_graph_compute(bctx, graphB, /*which_graph*/ 1);
    if (stB != GGML_STATUS_SUCCESS) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "rfdetr_model_forward: graphB compute returned %d", (int)stB);
        ggml_free(gctxB);
        return out;
    }

    /* Read host-side outputs. */
    out.class_logits.resize((size_t)NC * NQ);
    std::vector<float> bbox_delta((size_t)4 * NQ);
    ggml_backend_tensor_get(cls_logits_t, out.class_logits.data(), 0,
                            out.class_logits.size() * sizeof(float));
    ggml_backend_tensor_get(bbox_delta_t, bbox_delta.data(), 0,
                            bbox_delta.size() * sizeof(float));

    if (seg_masks_t) {
        /* seg_masks_t ne = (W_mask, H_mask, NQ, 1). Copy contiguous. */
        const int W_mask = (int)seg_masks_t->ne[0];
        const int H_mask = (int)seg_masks_t->ne[1];
        out.mask_w = W_mask;
        out.mask_h = H_mask;
        out.masks.resize((size_t)W_mask * H_mask * NQ);
        ggml_backend_tensor_get(seg_masks_t, out.masks.data(), 0,
                                out.masks.size() * sizeof(float));
    }

    /* gctxB owns the graph + tensor metadata. Tensor data lives in the
     * gallocr's buffer (kept alive in BackendCtx). */
    ggml_free(gctxB);

    /* =====================================================================
     * CPU bbox_reparam: combine raw bbox_embed delta with the decoder input
     * refpoints to produce final (cx, cy, w, h) in [0, 1]. Matches
     * rfdetr/models/lwdetr.py:229-233 (bbox_reparam=True branch).
     * ===================================================================== */
    out.bbox_cxcywh.resize((size_t)4 * NQ);
    for (int q = 0; q < NQ; ++q) {
        const float* d   = bbox_delta.data()       + (size_t)q * 4;
        const float* ref = decoder_refpoints.data() + (size_t)q * 4;
        float* dst = out.bbox_cxcywh.data() + (size_t)q * 4;
        dst[0] = d[0] * ref[2] + ref[0];      // cx
        dst[1] = d[1] * ref[3] + ref[1];      // cy
        dst[2] = std::exp(d[2]) * ref[2];     // w
        dst[3] = std::exp(d[3]) * ref[3];     // h
    }

    return out;
}

}  // namespace rfdetr
