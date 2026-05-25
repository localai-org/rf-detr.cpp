/* Synthesizes a minimal rfdetr-base GGUF for unit tests. No PyTorch needed.
 * Writes all metadata keys the loader will read, plus the full expected
 * tensor set (zero-initialized F16 tensors of the right shapes). Result is
 * a few MB. */

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct VariantCfg {
    const char*   name             = "base";
    uint32_t      image_size       = 560;
    uint32_t      num_queries      = 300;
    uint32_t      num_classes      = 80;
    uint32_t      bb_dim           = 768;
    uint32_t      bb_depth         = 12;
    uint32_t      bb_heads         = 12;
    uint32_t      bb_window        = 14;
    std::vector<int32_t> bb_ms_layers = {2, 5, 8, 11};
    uint32_t      enc_layers       = 3;
    uint32_t      enc_model_dim    = 256;
    uint32_t      enc_ffn_dim      = 2048;
    uint32_t      enc_heads        = 8;
    uint32_t      dec_layers       = 3;
    uint32_t      dec_model_dim    = 256;
    uint32_t      dec_ffn_dim      = 2048;
    uint32_t      dec_heads        = 8;
};

const char* COCO_CLASSES[] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
};
const size_t COCO_N = sizeof(COCO_CLASSES) / sizeof(COCO_CLASSES[0]);

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
                     const VariantCfg& v) {
    const ggml_type F = GGML_TYPE_F16;

    // Backbone
    out.push_back(make_tensor(ctx, "backbone.patch_embed.weight", F, 14, 14, 3, v.bb_dim));
    out.push_back(make_tensor(ctx, "backbone.patch_embed.bias",   F, v.bb_dim));
    int n_patches = (v.image_size / 14) * (v.image_size / 14);
    out.push_back(make_tensor(ctx, "backbone.pos_embed", F, v.bb_dim, n_patches + 1));
    out.push_back(make_tensor(ctx, "backbone.cls_token", F, v.bb_dim));

    for (uint32_t i = 0; i < v.bb_depth; ++i) {
        std::string p = "backbone.blocks." + std::to_string(i) + ".";
        out.push_back(make_tensor(ctx, (p + "norm1.weight").c_str(), F, v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "norm1.bias").c_str(),   F, v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "attn.qkv.weight").c_str(),  F, v.bb_dim, 3 * v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "attn.qkv.bias").c_str(),    F, 3 * v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "attn.proj.weight").c_str(), F, v.bb_dim, v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "attn.proj.bias").c_str(),   F, v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "norm2.weight").c_str(), F, v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "norm2.bias").c_str(),   F, v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "mlp.fc1.weight").c_str(), F, v.bb_dim, 4 * v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "mlp.fc1.bias").c_str(),   F, 4 * v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "mlp.fc2.weight").c_str(), F, 4 * v.bb_dim, v.bb_dim));
        out.push_back(make_tensor(ctx, (p + "mlp.fc2.bias").c_str(),   F, v.bb_dim));
    }
    out.push_back(make_tensor(ctx, "backbone.norm.weight", F, v.bb_dim));
    out.push_back(make_tensor(ctx, "backbone.norm.bias",   F, v.bb_dim));

    // Projector
    const size_t n_levels = v.bb_ms_layers.size();
    for (size_t j = 0; j < n_levels; ++j) {
        std::string p = "projector.level" + std::to_string(j) + ".";
        out.push_back(make_tensor(ctx, (p + "weight").c_str(), F, v.bb_dim, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "bias").c_str(),   F, v.enc_model_dim));
    }
    out.push_back(make_tensor(ctx, "projector.level_embed", F, v.enc_model_dim, (int64_t)n_levels));

    // Encoder
    for (uint32_t i = 0; i < v.enc_layers; ++i) {
        std::string p = "encoder.layers." + std::to_string(i) + ".";
        out.push_back(make_tensor(ctx, (p + "self_attn.qkv.weight").c_str(), F, v.enc_model_dim, 3 * v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "self_attn.qkv.bias").c_str(),   F, 3 * v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "self_attn.out.weight").c_str(), F, v.enc_model_dim, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "self_attn.out.bias").c_str(),   F, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm1.weight").c_str(), F, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm1.bias").c_str(),   F, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc1.weight").c_str(), F, v.enc_model_dim, v.enc_ffn_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc1.bias").c_str(),   F, v.enc_ffn_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc2.weight").c_str(), F, v.enc_ffn_dim, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc2.bias").c_str(),   F, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm2.weight").c_str(), F, v.enc_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm2.bias").c_str(),   F, v.enc_model_dim));
    }

    // Decoder
    out.push_back(make_tensor(ctx, "decoder.queries", F, v.dec_model_dim, v.num_queries));
    for (uint32_t i = 0; i < v.dec_layers; ++i) {
        std::string p = "decoder.layers." + std::to_string(i) + ".";
        out.push_back(make_tensor(ctx, (p + "self_attn.qkv.weight").c_str(),  F, v.dec_model_dim, 3 * v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "self_attn.qkv.bias").c_str(),    F, 3 * v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "self_attn.out.weight").c_str(),  F, v.dec_model_dim, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "self_attn.out.bias").c_str(),    F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm1.weight").c_str(), F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm1.bias").c_str(),   F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "cross_attn.q.weight").c_str(),  F, v.dec_model_dim, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "cross_attn.q.bias").c_str(),    F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "cross_attn.kv.weight").c_str(), F, v.dec_model_dim, 2 * v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "cross_attn.kv.bias").c_str(),   F, 2 * v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "cross_attn.out.weight").c_str(),F, v.dec_model_dim, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "cross_attn.out.bias").c_str(),  F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm2.weight").c_str(), F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm2.bias").c_str(),   F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc1.weight").c_str(), F, v.dec_model_dim, v.dec_ffn_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc1.bias").c_str(),   F, v.dec_ffn_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc2.weight").c_str(), F, v.dec_ffn_dim, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "ffn.fc2.bias").c_str(),   F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm3.weight").c_str(), F, v.dec_model_dim));
        out.push_back(make_tensor(ctx, (p + "norm3.bias").c_str(),   F, v.dec_model_dim));
    }

    // Heads
    out.push_back(make_tensor(ctx, "heads.class.fc.weight", F, v.dec_model_dim, v.num_classes));
    out.push_back(make_tensor(ctx, "heads.class.fc.bias",   F, v.num_classes));
    out.push_back(make_tensor(ctx, "heads.bbox.fc1.weight", F, v.dec_model_dim, v.dec_model_dim));
    out.push_back(make_tensor(ctx, "heads.bbox.fc1.bias",   F, v.dec_model_dim));
    out.push_back(make_tensor(ctx, "heads.bbox.fc2.weight", F, v.dec_model_dim, v.dec_model_dim));
    out.push_back(make_tensor(ctx, "heads.bbox.fc2.bias",   F, v.dec_model_dim));
    out.push_back(make_tensor(ctx, "heads.bbox.fc3.weight", F, v.dec_model_dim, 4));
    out.push_back(make_tensor(ctx, "heads.bbox.fc3.bias",   F, 4));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gen_model_gguf <out_path>\n");
        return 1;
    }
    const char* out_path = argv[1];
    VariantCfg v;

    ggml_init_params iparams{};
    iparams.mem_size   = 256 * 1024 * 1024;
    iparams.mem_buffer = nullptr;
    iparams.no_alloc   = false;
    ggml_context* ctx = ggml_init(iparams);
    if (!ctx) { std::fprintf(stderr, "ggml_init failed\n"); return 2; }

    std::vector<ggml_tensor*> tensors;
    tensors.reserve(600);
    add_all_tensors(ctx, tensors, v);

    // Zero-init every tensor's data so the GGUF writer reads valid bytes
    for (auto* t : tensors) {
        std::memset(t->data, 0, ggml_nbytes(t));
    }

    gguf_context* gguf = gguf_init_empty();

    gguf_set_val_str(gguf, "general.architecture",    "rfdetr");
    gguf_set_val_str(gguf, "rfdetr.format.version",   "1");
    gguf_set_val_str(gguf, "rfdetr.variant",          v.name);
    gguf_set_val_u32(gguf, "rfdetr.image_size",       v.image_size);
    gguf_set_val_u32(gguf, "rfdetr.num_queries",      v.num_queries);
    gguf_set_val_u32(gguf, "rfdetr.num_classes",      v.num_classes);
    gguf_set_arr_str(gguf, "rfdetr.class_names",      COCO_CLASSES, COCO_N);

    float mean[3] = {0.485f, 0.456f, 0.406f};
    float stdv[3] = {0.229f, 0.224f, 0.225f};
    gguf_set_arr_data(gguf, "rfdetr.preprocess.mean", GGUF_TYPE_FLOAT32, mean, 3);
    gguf_set_arr_data(gguf, "rfdetr.preprocess.std",  GGUF_TYPE_FLOAT32, stdv, 3);

    gguf_set_val_u32(gguf, "rfdetr.backbone.dim",         v.bb_dim);
    gguf_set_val_u32(gguf, "rfdetr.backbone.depth",       v.bb_depth);
    gguf_set_val_u32(gguf, "rfdetr.backbone.heads",       v.bb_heads);
    gguf_set_val_u32(gguf, "rfdetr.backbone.window_size", v.bb_window);
    gguf_set_arr_data(gguf, "rfdetr.backbone.multi_scale_layers",
                      GGUF_TYPE_INT32, v.bb_ms_layers.data(), v.bb_ms_layers.size());

    gguf_set_val_u32(gguf, "rfdetr.encoder.layers",    v.enc_layers);
    gguf_set_val_u32(gguf, "rfdetr.encoder.model_dim", v.enc_model_dim);
    gguf_set_val_u32(gguf, "rfdetr.encoder.ffn_dim",   v.enc_ffn_dim);
    gguf_set_val_u32(gguf, "rfdetr.encoder.heads",     v.enc_heads);
    gguf_set_val_u32(gguf, "rfdetr.decoder.layers",    v.dec_layers);
    gguf_set_val_u32(gguf, "rfdetr.decoder.model_dim", v.dec_model_dim);
    gguf_set_val_u32(gguf, "rfdetr.decoder.ffn_dim",   v.dec_ffn_dim);
    gguf_set_val_u32(gguf, "rfdetr.decoder.heads",     v.dec_heads);

    for (auto* t : tensors) gguf_add_tensor(gguf, t);

    if (!gguf_write_to_file(gguf, out_path, /*only_meta*/ false)) {
        std::fprintf(stderr, "gguf_write_to_file failed for %s\n", out_path);
        gguf_free(gguf);
        ggml_free(ctx);
        return 3;
    }

    gguf_free(gguf);
    ggml_free(ctx);
    std::printf("wrote %s\n", out_path);
    return 0;
}
