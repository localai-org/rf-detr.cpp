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

namespace rfdetr {

/* Configuration loaded from GGUF metadata. Plan 3's graph builder consumes
 * this; Plan 2's loader only populates it. */
struct Config {
    std::string variant;          // "base" / "nano" / ...
    uint32_t image_size  = 0;
    uint32_t num_queries = 0;
    uint32_t num_classes = 0;
    std::vector<std::string> class_names;
    float preprocess_mean[3] = {0, 0, 0};
    float preprocess_std[3]  = {1, 1, 1};

    struct {
        uint32_t dim                 = 0;
        uint32_t depth               = 0;
        uint32_t heads               = 0;
        uint32_t window_size         = 0;
        std::vector<uint32_t> multi_scale_layers;
    } backbone;

    struct {
        uint32_t layers    = 0;
        uint32_t model_dim = 0;
        uint32_t ffn_dim   = 0;
        uint32_t heads     = 0;
    } encoder, decoder;
};

/* Loaded model. Plan 2 populates `config` and `tensors` (ggml_tensor
 * descriptors from gguf_init_from_file — data is NOT loaded into a backend
 * buffer yet). Plan 3 will add a backend buffer + actual data load. */
struct Model {
    Config config;
    ::gguf_context* gguf  = nullptr;
    ::ggml_context* meta  = nullptr;  // ggml_context produced by gguf_init_from_file
    std::unordered_map<std::string, ::ggml_tensor*> tensors;
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

/* Build the list of *expected* tensor names for a given variant config. Used
 * by both `model_validate_tensors` and the test-fixture generator. */
std::vector<std::string> expected_tensor_names(const Config& cfg);

}  // namespace rfdetr

#endif
