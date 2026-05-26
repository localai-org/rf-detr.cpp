#ifndef RFDETR_MODEL_HPP
#define RFDETR_MODEL_HPP

#include "model_loader.hpp"

#include <cstdint>
#include <vector>

struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;

namespace rfdetr {

/* Host-side result of the full forward pass at inference.
 *
 *   class_logits  shape (num_classes, num_queries) row-major in this vector
 *                 (i.e. class is fastest-varying — matches ggml ne[0]=num_classes
 *                 from heads.class_logits, which is what postprocess expects).
 *                 RAW logits (apply sigmoid in postproc).
 *   bbox_cxcywh   shape (4, num_queries) row-major — (cx, cy, w, h) in [0, 1]
 *                 after bbox_reparam (delta combined with refpoints).
 *
 * On failure both vectors are empty (and status is logged via rfdetr_logf). */
struct ForwardOutput {
    std::vector<float> class_logits;   // size = num_classes * num_queries
    std::vector<float> bbox_cxcywh;    // size = 4 * num_queries
    int num_queries = 0;
    int num_classes = 0;
};

/* Run the full forward pipeline end-to-end:
 *
 *   preprocessed input → backbone → projector → two_stage init → CPU top-K
 *   + refpoint construction → decoder → heads → bbox_reparam → output.
 *
 *   input_data: preprocessed F32 buffer, shape (image_size, image_size, 3, 1)
 *               (matches `rfdetr_preprocess` output). Already ImageNet-normalized.
 *   input_size: image_size (square, must equal model's `config.image_size`).
 *   backend:    backend the model weights were realized on.
 *
 * Returns empty vectors in the ForwardOutput on failure. */
ForwardOutput rfdetr_model_forward(const Model& m,
                                   const float* input_data, int input_size,
                                   ggml_backend_t backend);

}  // namespace rfdetr

#endif
