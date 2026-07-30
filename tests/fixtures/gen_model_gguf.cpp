/* Synthesizes a minimal rfdetr-base GGUF (format v2) for unit tests. No
 * PyTorch needed. Writes all metadata keys the loader will read, plus the
 * full expected tensor set (zero-initialized tensors of the right shapes).
 *
 * Per-tensor dimensions are shrunk (image_size=56, bb_dim=64, ffn=128,
 * etc.) so the output stays small while preserving the FULL shape
 * STRUCTURE of rfdetr-base (12 backbone blocks with separate Q/K/V +
 * layer-scale gammas, C2f single-scale projector, 13 two-stage groups,
 * 3 decoder layers with deformable cross-attn, shared heads).
 *
 * 486 tensors total. See docs/conversion.md for the schema contract.
 *
 * Tensor dtype defaults to F32; pass --dtype f16 to emit F16 (used to test
 * forward-compat with quantized weights). */

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct VariantCfg {
    const char*           name                   = "base";
    uint32_t              image_size             = 56;
    uint32_t              patch_size             = 14;
    uint32_t              num_queries            = 300;
    uint32_t              group_detr             = 13;
    uint32_t              num_classes            = 91;

    /* Backbone (DINOv2-small, shrunk) */
    uint32_t              bb_dim                 = 64;
    uint32_t              bb_depth               = 12;
    uint32_t              bb_heads               = 8;     // 8 divides 64 cleanly
    uint32_t              bb_ffn                 = 128;
    uint32_t              bb_num_windows         = 4;
    std::vector<int32_t>  bb_global_attn_indices = {2, 5, 8, 11};
    std::vector<int32_t>  bb_out_feature_indices = {2, 5, 8, 11};
    uint32_t              bb_pos_embed_train_size = 4;   // 4*4 + 1 = 17 tokens

    /* Projector (single P4 C2f, shrunk) */
    uint32_t              proj_in_dim            = 256;  // 4 * bb_dim
    uint32_t              proj_out_dim           = 64;
    uint32_t              proj_bottleneck_dim    = 32;
    uint32_t              proj_n_bottlenecks     = 3;

    /* Decoder (shrunk) */
    uint32_t              dec_layers             = 3;
    uint32_t              dec_model_dim          = 64;
    uint32_t              dec_ffn_dim            = 128;
    uint32_t              dec_self_attn_heads    = 8;
    uint32_t              dec_cross_attn_heads   = 8;    // 16 in real (with d=256, head=16); fixture uses 8 → head=8
    uint32_t              dec_cross_attn_n_levels = 1;
    uint32_t              dec_cross_attn_n_points = 2;

    /* Two-stage */
    uint32_t              two_stage_n_groups     = 13;
};

/* The 80 COCO class names placed at their canonical 1-indexed logit
 * positions in a 91-slot table. Index 0 is unused (background) in
 * standard COCO, but rfdetr's class_embed outputs are 91-wide with the
 * 80 COCO names interleaved with reserved/unused IDs.
 *
 * For test simplicity our fixture places "person" at slot 0 and
 * "toothbrush" at slot 90, with the rest filled in canonical positions.
 * The exact mapping is fixture-internal — real converted weights will use
 * the upstream COCO index convention.
 *
 * The historical "thing" gaps (reserved IDs) in standard COCO are at:
 *   12, 26, 29, 30, 45, 66, 68, 69, 71, 83.
 * Those slots get the placeholder "" in our fixture. */
struct CocoEntry { uint32_t idx; const char* name; };
const CocoEntry COCO[] = {
    { 0,"person"},{ 1,"bicycle"},{ 2,"car"},{ 3,"motorcycle"},{ 4,"airplane"},
    { 5,"bus"},{ 6,"train"},{ 7,"truck"},{ 8,"boat"},{ 9,"traffic light"},
    {10,"fire hydrant"},{11,"stop sign"},{13,"parking meter"},{14,"bench"},
    {15,"bird"},{16,"cat"},{17,"dog"},{18,"horse"},{19,"sheep"},{20,"cow"},
    {21,"elephant"},{22,"bear"},{23,"zebra"},{24,"giraffe"},{25,"backpack"},
    {27,"umbrella"},{28,"handbag"},{31,"tie"},{32,"suitcase"},{33,"frisbee"},
    {34,"skis"},{35,"snowboard"},{36,"sports ball"},{37,"kite"},
    {38,"baseball bat"},{39,"baseball glove"},{40,"skateboard"},
    {41,"surfboard"},{42,"tennis racket"},{43,"bottle"},{44,"wine glass"},
    {46,"cup"},{47,"fork"},{48,"knife"},{49,"spoon"},{50,"bowl"},
    {51,"banana"},{52,"apple"},{53,"sandwich"},{54,"orange"},{55,"broccoli"},
    {56,"carrot"},{57,"hot dog"},{58,"pizza"},{59,"donut"},{60,"cake"},
    {61,"chair"},{62,"couch"},{63,"potted plant"},{64,"bed"},{65,"dining table"},
    {67,"toilet"},{70,"tv"},{72,"laptop"},{73,"mouse"},{74,"remote"},
    {75,"keyboard"},{76,"cell phone"},{77,"microwave"},{78,"oven"},
    {79,"toaster"},{80,"sink"},{81,"refrigerator"},{82,"book"},{84,"clock"},
    {85,"vase"},{86,"scissors"},{87,"teddy bear"},{88,"hair drier"},
    /* Place toothbrush at slot 90 (last) so tests can assert on tail slot. */
    {90,"toothbrush"},
};

std::vector<std::string> build_class_names(uint32_t n_classes) {
    /* gguf_set_arr_str rejects empty strings on some implementations; use
     * "" placeholder elements for reserved/unused slots. */
    std::vector<std::string> v(n_classes, "");
    for (const auto& e : COCO) {
        if (e.idx < n_classes) v[e.idx] = e.name;
    }
    return v;
}

ggml_tensor* make_tensor(ggml_context* ctx, const char* name,
                         ggml_type type, int64_t d0, int64_t d1 = 1,
                         int64_t d2 = 1, int64_t d3 = 1) {
    ggml_tensor* t;
    if (d3 > 1)      t = ggml_new_tensor_4d(ctx, type, d0, d1, d2, d3);
    else if (d2 > 1) t = ggml_new_tensor_3d(ctx, type, d0, d1, d2);
    else if (d1 > 1) t = ggml_new_tensor_2d(ctx, type, d0, d1);
    else             t = ggml_new_tensor_1d(ctx, type, d0);
    ggml_set_name(t, name);
    return t;
}

void add_all_tensors(ggml_context* ctx, std::vector<ggml_tensor*>& out,
                     const VariantCfg& v, ggml_type tensor_dtype) {
    const ggml_type F = tensor_dtype;

    /* Convenient shorthands */
    const int64_t  bb_dim   = v.bb_dim;
    const int64_t  bb_ffn   = v.bb_ffn;
    const int64_t  ps       = v.patch_size;
    const int64_t  pos_grid = v.bb_pos_embed_train_size;
    const int64_t  n_tokens_train = pos_grid * pos_grid + 1; /* +1 for CLS */

    const int64_t  proj_in   = v.proj_in_dim;
    const int64_t  proj_out  = v.proj_out_dim;
    const int64_t  proj_bnk  = v.proj_bottleneck_dim;
    const int64_t  proj_cv2_in = proj_out + (int64_t)v.proj_n_bottlenecks * proj_bnk; // C2f cv2 input

    const int64_t  dm       = v.dec_model_dim;
    const int64_t  dffn     = v.dec_ffn_dim;
    const int64_t  n_offset = 2 * (int64_t)v.dec_cross_attn_heads
                              * v.dec_cross_attn_n_levels
                              * v.dec_cross_attn_n_points;        // sampling_offsets out
    const int64_t  n_wt     = (int64_t)v.dec_cross_attn_heads
                              * v.dec_cross_attn_n_levels
                              * v.dec_cross_attn_n_points;        // attention_weights out
    const int64_t  nq       = v.num_queries;
    const int64_t  nc       = v.num_classes;

    /* ---- Backbone embeddings (4) ---- */
    /* Conv2d weight: PyTorch (out, in, kh, kw) → ggml ne (kw, kh, in, out) */
    out.push_back(make_tensor(ctx, "backbone.patch_embed.weight", F, ps, ps, 3, bb_dim));
    out.push_back(make_tensor(ctx, "backbone.patch_embed.bias",   F, bb_dim));
    out.push_back(make_tensor(ctx, "backbone.cls_token",          F, bb_dim));
    out.push_back(make_tensor(ctx, "backbone.pos_embed",          F, bb_dim, n_tokens_train));

    /* ---- Backbone blocks (18 per block) ---- */
    for (uint32_t i = 0; i < v.bb_depth; ++i) {
        std::string p = "backbone.blocks." + std::to_string(i) + ".";
        out.push_back(make_tensor(ctx, (p + "norm1.weight").c_str(), F, bb_dim));
        out.push_back(make_tensor(ctx, (p + "norm1.bias").c_str(),   F, bb_dim));
        /* Linear weights: PyTorch (out, in) → ggml ne (in, out) */
        out.push_back(make_tensor(ctx, (p + "attn.q.weight").c_str(),    F, bb_dim, bb_dim));
        out.push_back(make_tensor(ctx, (p + "attn.q.bias").c_str(),      F, bb_dim));
        out.push_back(make_tensor(ctx, (p + "attn.k.weight").c_str(),    F, bb_dim, bb_dim));
        out.push_back(make_tensor(ctx, (p + "attn.k.bias").c_str(),      F, bb_dim));
        out.push_back(make_tensor(ctx, (p + "attn.v.weight").c_str(),    F, bb_dim, bb_dim));
        out.push_back(make_tensor(ctx, (p + "attn.v.bias").c_str(),      F, bb_dim));
        out.push_back(make_tensor(ctx, (p + "attn.proj.weight").c_str(), F, bb_dim, bb_dim));
        out.push_back(make_tensor(ctx, (p + "attn.proj.bias").c_str(),   F, bb_dim));
        out.push_back(make_tensor(ctx, (p + "layer_scale1").c_str(),     F, bb_dim));
        out.push_back(make_tensor(ctx, (p + "norm2.weight").c_str(),     F, bb_dim));
        out.push_back(make_tensor(ctx, (p + "norm2.bias").c_str(),       F, bb_dim));
        out.push_back(make_tensor(ctx, (p + "mlp.fc1.weight").c_str(),   F, bb_dim, bb_ffn));
        out.push_back(make_tensor(ctx, (p + "mlp.fc1.bias").c_str(),     F, bb_ffn));
        out.push_back(make_tensor(ctx, (p + "mlp.fc2.weight").c_str(),   F, bb_ffn, bb_dim));
        out.push_back(make_tensor(ctx, (p + "mlp.fc2.bias").c_str(),     F, bb_dim));
        out.push_back(make_tensor(ctx, (p + "layer_scale2").c_str(),     F, bb_dim));
    }
    /* ---- Backbone final norm (2) ---- */
    out.push_back(make_tensor(ctx, "backbone.norm.weight", F, bb_dim));
    out.push_back(make_tensor(ctx, "backbone.norm.bias",   F, bb_dim));

    /* ---- Projector (single P4 C2f, 26 tensors) ----
     * Conv2d weight ne: (kw, kh, in, out). cv1/cv2 are 1×1; bottlenecks 3×3. */
    out.push_back(make_tensor(ctx, "projector.cv1.conv.weight", F, 1, 1, proj_in,  proj_out));
    out.push_back(make_tensor(ctx, "projector.cv1.norm.weight", F, proj_out));
    out.push_back(make_tensor(ctx, "projector.cv1.norm.bias",   F, proj_out));
    out.push_back(make_tensor(ctx, "projector.cv2.conv.weight", F, 1, 1, proj_cv2_in, proj_out));
    out.push_back(make_tensor(ctx, "projector.cv2.norm.weight", F, proj_out));
    out.push_back(make_tensor(ctx, "projector.cv2.norm.bias",   F, proj_out));
    for (uint32_t j = 0; j < v.proj_n_bottlenecks; ++j) {
        std::string p = "projector.bottleneck." + std::to_string(j) + ".";
        out.push_back(make_tensor(ctx, (p + "cv1.conv.weight").c_str(), F, 3, 3, proj_bnk, proj_bnk));
        out.push_back(make_tensor(ctx, (p + "cv1.norm.weight").c_str(), F, proj_bnk));
        out.push_back(make_tensor(ctx, (p + "cv1.norm.bias").c_str(),   F, proj_bnk));
        out.push_back(make_tensor(ctx, (p + "cv2.conv.weight").c_str(), F, 3, 3, proj_bnk, proj_bnk));
        out.push_back(make_tensor(ctx, (p + "cv2.norm.weight").c_str(), F, proj_bnk));
        out.push_back(make_tensor(ctx, (p + "cv2.norm.bias").c_str(),   F, proj_bnk));
    }
    out.push_back(make_tensor(ctx, "projector.final_norm.weight", F, proj_out));
    out.push_back(make_tensor(ctx, "projector.final_norm.bias",   F, proj_out));

    /* ---- Two-stage groups (12 per group × 13 = 156) ---- */
    for (uint32_t g = 0; g < v.two_stage_n_groups; ++g) {
        const std::string gi = std::to_string(g);
        out.push_back(make_tensor(ctx, ("two_stage.enc_output." + gi + ".weight").c_str(), F, dm, dm));
        out.push_back(make_tensor(ctx, ("two_stage.enc_output." + gi + ".bias").c_str(),   F, dm));
        out.push_back(make_tensor(ctx, ("two_stage.enc_output_norm." + gi + ".weight").c_str(), F, dm));
        out.push_back(make_tensor(ctx, ("two_stage.enc_output_norm." + gi + ".bias").c_str(),   F, dm));
        out.push_back(make_tensor(ctx, ("two_stage.enc_out_class_embed." + gi + ".weight").c_str(), F, dm, nc));
        out.push_back(make_tensor(ctx, ("two_stage.enc_out_class_embed." + gi + ".bias").c_str(),   F, nc));
        /* MLP: dm → dm → dm → 4 */
        out.push_back(make_tensor(ctx, ("two_stage.enc_out_bbox_embed." + gi + ".layers.0.weight").c_str(), F, dm, dm));
        out.push_back(make_tensor(ctx, ("two_stage.enc_out_bbox_embed." + gi + ".layers.0.bias").c_str(),   F, dm));
        out.push_back(make_tensor(ctx, ("two_stage.enc_out_bbox_embed." + gi + ".layers.1.weight").c_str(), F, dm, dm));
        out.push_back(make_tensor(ctx, ("two_stage.enc_out_bbox_embed." + gi + ".layers.1.bias").c_str(),   F, dm));
        out.push_back(make_tensor(ctx, ("two_stage.enc_out_bbox_embed." + gi + ".layers.2.weight").c_str(), F, dm, 4));
        out.push_back(make_tensor(ctx, ("two_stage.enc_out_bbox_embed." + gi + ".layers.2.bias").c_str(),   F, 4));
    }

    /* ---- Decoder queries (group-0 slice, 2 tensors) ---- */
    out.push_back(make_tensor(ctx, "decoder.queries.feat",      F, dm, nq));
    out.push_back(make_tensor(ctx, "decoder.queries.refpoints", F, 4,  nq));

    /* ---- Decoder ref_point_head (4 tensors): MLP (2*dm) → dm → dm ----
     * Layer 0: PyTorch shape (dm, 2*dm) — refpoints sinusoidally embedded
     *   with num_pos_feats = dm/2 per axis × 4 axes (cx,cy,w,h) = 2*dm
     *   features, projected down to dm.
     *   ggml ne (in, out) = (2*dm, dm)
     * Layer 1: (dm, dm) */
    out.push_back(make_tensor(ctx, "decoder.ref_point_head.layers.0.weight", F, 2 * dm, dm));
    out.push_back(make_tensor(ctx, "decoder.ref_point_head.layers.0.bias",   F, dm));
    out.push_back(make_tensor(ctx, "decoder.ref_point_head.layers.1.weight", F, dm, dm));
    out.push_back(make_tensor(ctx, "decoder.ref_point_head.layers.1.bias",   F, dm));

    /* ---- Decoder layers (22 per layer) ---- */
    for (uint32_t i = 0; i < v.dec_layers; ++i) {
        std::string p = "decoder.layers." + std::to_string(i) + ".";
        /* Self-attention (packed QKV à la nn.MultiheadAttention) */
        out.push_back(make_tensor(ctx, (p + "self_attn.in_proj.weight").c_str(),  F, dm, 3 * dm));
        out.push_back(make_tensor(ctx, (p + "self_attn.in_proj.bias").c_str(),    F, 3 * dm));
        out.push_back(make_tensor(ctx, (p + "self_attn.out_proj.weight").c_str(), F, dm, dm));
        out.push_back(make_tensor(ctx, (p + "self_attn.out_proj.bias").c_str(),   F, dm));
        out.push_back(make_tensor(ctx, (p + "norm1.weight").c_str(),              F, dm));
        out.push_back(make_tensor(ctx, (p + "norm1.bias").c_str(),                F, dm));

        /* Deformable cross-attention */
        out.push_back(make_tensor(ctx, (p + "cross_attn.sampling_offsets.weight").c_str(),  F, dm, n_offset));
        out.push_back(make_tensor(ctx, (p + "cross_attn.sampling_offsets.bias").c_str(),    F, n_offset));
        out.push_back(make_tensor(ctx, (p + "cross_attn.attention_weights.weight").c_str(), F, dm, n_wt));
        out.push_back(make_tensor(ctx, (p + "cross_attn.attention_weights.bias").c_str(),   F, n_wt));
        out.push_back(make_tensor(ctx, (p + "cross_attn.value_proj.weight").c_str(),        F, dm, dm));
        out.push_back(make_tensor(ctx, (p + "cross_attn.value_proj.bias").c_str(),          F, dm));
        out.push_back(make_tensor(ctx, (p + "cross_attn.output_proj.weight").c_str(),       F, dm, dm));
        out.push_back(make_tensor(ctx, (p + "cross_attn.output_proj.bias").c_str(),         F, dm));
        out.push_back(make_tensor(ctx, (p + "norm2.weight").c_str(),                        F, dm));
        out.push_back(make_tensor(ctx, (p + "norm2.bias").c_str(),                          F, dm));

        /* FFN */
        out.push_back(make_tensor(ctx, (p + "linear1.weight").c_str(), F, dm, dffn));
        out.push_back(make_tensor(ctx, (p + "linear1.bias").c_str(),   F, dffn));
        out.push_back(make_tensor(ctx, (p + "linear2.weight").c_str(), F, dffn, dm));
        out.push_back(make_tensor(ctx, (p + "linear2.bias").c_str(),   F, dm));

        out.push_back(make_tensor(ctx, (p + "norm3.weight").c_str(), F, dm));
        out.push_back(make_tensor(ctx, (p + "norm3.bias").c_str(),   F, dm));
    }

    /* ---- Decoder final norm (2) ---- */
    out.push_back(make_tensor(ctx, "decoder.norm.weight", F, dm));
    out.push_back(make_tensor(ctx, "decoder.norm.bias",   F, dm));

    /* ---- Heads (shared single instances, 8 tensors) ---- */
    out.push_back(make_tensor(ctx, "heads.class_embed.weight", F, dm, nc));
    out.push_back(make_tensor(ctx, "heads.class_embed.bias",   F, nc));
    /* bbox MLP: dm → dm → dm → 4 */
    out.push_back(make_tensor(ctx, "heads.bbox_embed.layers.0.weight", F, dm, dm));
    out.push_back(make_tensor(ctx, "heads.bbox_embed.layers.0.bias",   F, dm));
    out.push_back(make_tensor(ctx, "heads.bbox_embed.layers.1.weight", F, dm, dm));
    out.push_back(make_tensor(ctx, "heads.bbox_embed.layers.1.bias",   F, dm));
    out.push_back(make_tensor(ctx, "heads.bbox_embed.layers.2.weight", F, dm, 4));
    out.push_back(make_tensor(ctx, "heads.bbox_embed.layers.2.bias",   F, 4));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gen_model_gguf <out_path>\n");
        return 1;
    }
    const char* out_path = argv[1];
    const char* skip_name = nullptr;
    /* Left null by default so the fixture carries no
     * rfdetr.preprocess.resize_mode key at all, which is what a GGUF converted
     * before that key existed looks like. */
    const char* resize_mode = nullptr;
    unsigned seed = 0;
    bool seeded = false;
    ggml_type tensor_dtype = GGML_TYPE_F32;
    for (int i = 2; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--missing") == 0) {
            skip_name = argv[i + 1];
        } else if (std::strcmp(argv[i], "--resize-mode") == 0) {
            resize_mode = argv[i + 1];
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            seed = (unsigned)std::strtoul(argv[i + 1], nullptr, 10);
            seeded = true;
        } else if (std::strcmp(argv[i], "--dtype") == 0) {
            std::string val = argv[i + 1];
            if      (val == "f16") tensor_dtype = GGML_TYPE_F16;
            else if (val == "f32") tensor_dtype = GGML_TYPE_F32;
            else {
                std::fprintf(stderr, "unknown --dtype: %s\n", val.c_str());
                return 5;
            }
        }
    }
    VariantCfg v;

    ggml_init_params iparams{};
    iparams.mem_size   = 512 * 1024 * 1024;
    iparams.mem_buffer = nullptr;
    iparams.no_alloc   = false;
    ggml_context* ctx = ggml_init(iparams);
    if (!ctx) { std::fprintf(stderr, "ggml_init failed\n"); return 2; }

    std::vector<ggml_tensor*> tensors;
    tensors.reserve(600);
    add_all_tensors(ctx, tensors, v, tensor_dtype);

    if (skip_name) {
        auto it = std::remove_if(tensors.begin(), tensors.end(),
            [&](ggml_tensor* t) { return std::string(ggml_get_name(t)) == skip_name; });
        tensors.erase(it, tensors.end());
    }

    /* Initialize tensor data. With --seed N, fill with N(0, 0.02) (PyTorch
     * default init scale); otherwise zero-fill so the GGUF writer reads
     * valid bytes. */
    if (seeded) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (auto* t : tensors) {
            size_t nelem = ggml_nelements(t);
            if (t->type == GGML_TYPE_F32) {
                float* p = (float*)t->data;
                for (size_t i = 0; i < nelem; ++i) p[i] = dist(rng);
            } else if (t->type == GGML_TYPE_F16) {
                ggml_fp16_t* p = (ggml_fp16_t*)t->data;
                for (size_t i = 0; i < nelem; ++i) {
                    p[i] = ggml_fp32_to_fp16(dist(rng));
                }
            } else {
                std::fprintf(stderr, "gen_model_gguf: unsupported type for tensor %s\n",
                             ggml_get_name(t));
                return 4;
            }
        }
    } else {
        for (auto* t : tensors) {
            std::memset(t->data, 0, ggml_nbytes(t));
        }
    }

    gguf_context* gguf = gguf_init_empty();

    /* --- Top-level / format --- */
    gguf_set_val_str(gguf, "general.architecture",    "rfdetr");
    gguf_set_val_str(gguf, "rfdetr.format.version",   "2");
    gguf_set_val_str(gguf, "rfdetr.variant",          v.name);
    gguf_set_val_u32(gguf, "rfdetr.image_size",       v.image_size);
    gguf_set_val_u32(gguf, "rfdetr.patch_size",       v.patch_size);
    gguf_set_val_u32(gguf, "rfdetr.num_queries",      v.num_queries);
    gguf_set_val_u32(gguf, "rfdetr.group_detr",       v.group_detr);
    gguf_set_val_u32(gguf, "rfdetr.num_classes",      v.num_classes);

    /* class_names: 91-slot vector, COCO names at canonical positions */
    std::vector<std::string> class_names = build_class_names(v.num_classes);
    std::vector<const char*> class_names_cstr;
    class_names_cstr.reserve(class_names.size());
    for (const auto& s : class_names) class_names_cstr.push_back(s.c_str());
    gguf_set_arr_str(gguf, "rfdetr.class_names", class_names_cstr.data(),
                     class_names_cstr.size());

    /* --- Preprocess --- */
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float stdv[3] = {0.229f, 0.224f, 0.225f};
    gguf_set_arr_data(gguf, "rfdetr.preprocess.mean", GGUF_TYPE_FLOAT32, mean, 3);
    gguf_set_arr_data(gguf, "rfdetr.preprocess.std",  GGUF_TYPE_FLOAT32, stdv, 3);
    /* Only written when --resize-mode is given, so the default fixture keeps
     * exercising the absent-key (legacy) path. */
    if (resize_mode) {
        gguf_set_val_str(gguf, "rfdetr.preprocess.resize_mode", resize_mode);
    }

    /* --- Backbone --- */
    gguf_set_val_u32(gguf, "rfdetr.backbone.dim",                  v.bb_dim);
    gguf_set_val_u32(gguf, "rfdetr.backbone.depth",                v.bb_depth);
    gguf_set_val_u32(gguf, "rfdetr.backbone.heads",                v.bb_heads);
    gguf_set_val_u32(gguf, "rfdetr.backbone.ffn_dim",              v.bb_ffn);
    gguf_set_val_u32(gguf, "rfdetr.backbone.num_windows",          v.bb_num_windows);
    gguf_set_val_u32(gguf, "rfdetr.backbone.pos_embed_train_size", v.bb_pos_embed_train_size);
    gguf_set_arr_data(gguf, "rfdetr.backbone.global_attn_indices",
                      GGUF_TYPE_INT32, v.bb_global_attn_indices.data(),
                      v.bb_global_attn_indices.size());
    gguf_set_arr_data(gguf, "rfdetr.backbone.out_feature_indices",
                      GGUF_TYPE_INT32, v.bb_out_feature_indices.data(),
                      v.bb_out_feature_indices.size());

    /* --- Projector --- */
    gguf_set_val_u32(gguf, "rfdetr.projector.in_dim",         v.proj_in_dim);
    gguf_set_val_u32(gguf, "rfdetr.projector.out_dim",        v.proj_out_dim);
    gguf_set_val_u32(gguf, "rfdetr.projector.bottleneck_dim", v.proj_bottleneck_dim);
    gguf_set_val_u32(gguf, "rfdetr.projector.n_bottlenecks",  v.proj_n_bottlenecks);

    /* --- Decoder --- */
    gguf_set_val_u32(gguf, "rfdetr.decoder.layers",              v.dec_layers);
    gguf_set_val_u32(gguf, "rfdetr.decoder.model_dim",           v.dec_model_dim);
    gguf_set_val_u32(gguf, "rfdetr.decoder.ffn_dim",             v.dec_ffn_dim);
    gguf_set_val_u32(gguf, "rfdetr.decoder.self_attn_heads",     v.dec_self_attn_heads);
    gguf_set_val_u32(gguf, "rfdetr.decoder.cross_attn_heads",    v.dec_cross_attn_heads);
    gguf_set_val_u32(gguf, "rfdetr.decoder.cross_attn_n_levels", v.dec_cross_attn_n_levels);
    gguf_set_val_u32(gguf, "rfdetr.decoder.cross_attn_n_points", v.dec_cross_attn_n_points);

    /* --- Two-stage --- */
    gguf_set_val_u32(gguf, "rfdetr.two_stage.n_groups", v.two_stage_n_groups);

    for (auto* t : tensors) gguf_add_tensor(gguf, t);

    if (!gguf_write_to_file(gguf, out_path, /*only_meta*/ false)) {
        std::fprintf(stderr, "gguf_write_to_file failed for %s\n", out_path);
        gguf_free(gguf);
        ggml_free(ctx);
        return 3;
    }

    gguf_free(gguf);
    ggml_free(ctx);
    std::printf("wrote %s (%zu tensors)\n", out_path, tensors.size());
    return 0;
}
