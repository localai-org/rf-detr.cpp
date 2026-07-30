#ifndef RFDETR_MODEL_LOADER_HPP
#define RFDETR_MODEL_LOADER_HPP

#include "rfdetr.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;

namespace rfdetr {

/* Configuration loaded from GGUF metadata. Plan 3's graph builder consumes
 * this; Plan 2's loader only populates it.
 *
 * Schema: format version "2" — see docs/conversion.md. */
struct Config {
    std::string variant;          // "base" (only supported variant for now)
    uint32_t image_size  = 0;
    uint32_t patch_size  = 0;
    uint32_t num_queries = 0;
    uint32_t group_detr  = 0;     // training-time groups; only group 0 used at inference
    uint32_t num_classes = 0;
    std::vector<std::string> class_names;
    float preprocess_mean[3] = {0, 0, 0};
    float preprocess_std[3]  = {1, 1, 1};
    /* Optional in format v2 for backward compatibility. Newly converted
     * RF-DETR >=1.9 models set this to true; older GGUFs keep legacy stb
     * resize behavior so their established outputs do not change. */
    bool preprocess_bilinear_no_antialias = false;

    struct {
        uint32_t dim                  = 0;
        uint32_t depth                = 0;
        uint32_t heads                = 0;
        uint32_t ffn_dim              = 0;
        uint32_t num_windows          = 0;
        std::vector<uint32_t> global_attn_indices;
        std::vector<uint32_t> out_feature_indices;
        uint32_t pos_embed_train_size = 0;
    } backbone;

    struct {
        uint32_t in_dim         = 0;
        uint32_t out_dim        = 0;
        uint32_t bottleneck_dim = 0;
        uint32_t n_bottlenecks  = 0;
    } projector;

    struct {
        uint32_t layers              = 0;
        uint32_t model_dim           = 0;
        uint32_t ffn_dim             = 0;
        uint32_t self_attn_heads     = 0;
        uint32_t cross_attn_heads    = 0;
        uint32_t cross_attn_n_levels = 0;
        uint32_t cross_attn_n_points = 0;
    } decoder;

    struct {
        uint32_t n_groups = 0;
    } two_stage;

    /* Segmentation head: present only on RFDETRSeg* variants. When
     * `has_segmentation_head` is true, the model has 35 additional tensors
     * under `segmentation_head.*` and the forward pass emits per-query masks
     * at (H / mask_downsample_ratio, W / mask_downsample_ratio). */
    bool has_segmentation_head    = false;
    uint32_t mask_downsample_ratio = 4;
};

/* Loaded model. Plan 2 populates `config` and `tensors` (ggml_tensor
 * descriptors from gguf_init_from_file — data is NOT loaded into a backend
 * buffer yet). Plan 3 will add a backend buffer + actual data load. */
struct Model {
    Config config;
    ::gguf_context* gguf  = nullptr;
    ::ggml_context* meta  = nullptr;  // ggml_context produced by gguf_init_from_file
    std::unordered_map<std::string, ::ggml_tensor*> tensors;

    /* Stashed during model_load so model_realize_weights can re-open the file
     * for streaming tensor data. */
    std::string path;

    /* Populated by model_realize_weights. nullptr until then. */
    ::ggml_backend_buffer_t weights = nullptr;

    /* Bicubic-interpolated backbone pos_embed for the inference resolution.
     *
     * The stored `backbone.pos_embed` tensor is the training-time embedding
     * for a 37x37 patch grid (+ CLS) → 1370 tokens. At rfdetr-base inference
     * the image is 560x560 → 40x40 = 1600 patches (+ CLS) → 1601 tokens.
     * The HF DinoV2 embedding module bicubic-interpolates the stored grid
     * at every forward call; we cache the result once at weight-realize time.
     *
     * Stored as a `(dim, N_inference_tokens)` F32 ggml tensor in `extras_ctx`
     * (allocated separately from `meta` which has a fixed-size pool). */
    ::ggml_context* extras_ctx = nullptr;
    ::ggml_backend_buffer_t extras_buf = nullptr;
    ::ggml_tensor* backbone_pos_embed_interp = nullptr;

    /* Two-stage proposal grid (4, W_inf*H_inf) F32. Constant — depends only
     * on the inference spatial extent. Populated by model_realize_weights and
     * lives in extras_ctx alongside backbone_pos_embed_interp. */
    ::ggml_tensor* proposals_grid = nullptr;
};

/* Load a model from a GGUF file at `path`. Returns nullptr on error and sets
 * `*out_status`. Caller owns the returned pointer; free with
 * `rfdetr::model_free`. */
Model* model_load(const std::string& path, rfdetr_status* out_status);

/* Free a model returned by `model_load`. Releases the gguf_context and
 * underlying ggml_context. */
void model_free(Model* m);

/* Validate that the expected tensor set for the config's variant is present
 * in `m->tensors`. Returns RFDETR_OK if all expected tensors are present,
 * RFDETR_ERR_MODEL_LOAD with a logged error otherwise. */
rfdetr_status model_validate_tensors(const Model& m);

/* Allocate a backend buffer for the model's tensors and copy data from the
 * GGUF file into it. After this call, every tensor descriptor in
 * `m.tensors` is backed by real data on the supplied backend.
 *
 * Idempotent: if `m.weights` is already non-null, returns RFDETR_OK without
 * doing anything. Returns RFDETR_ERR_MODEL_LOAD on failure (logged). */
rfdetr_status model_realize_weights(Model& m, ::ggml_backend_t backend);

/* Build the list of *expected* tensor names for a given variant config. Used
 * by both `model_validate_tensors` and the test-fixture generator. */
std::vector<std::string> expected_tensor_names(const Config& cfg);

}  // namespace rfdetr

#endif
