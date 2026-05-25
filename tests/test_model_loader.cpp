#include "test_assert.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "rfdetr.h"
#include <string>
#include <vector>

int main() {
    const std::string fixtures = RFDETR_TEST_FIXTURES;
    const std::string path = fixtures + "/model_base.gguf";

    rfdetr_status st;
    rfdetr::Model* m = rfdetr::model_load(path, &st);
    RFDETR_ASSERT(m != nullptr);
    RFDETR_ASSERT_EQ_INT(st, RFDETR_OK);

    // Config
    RFDETR_ASSERT_STR_EQ(m->config.variant.c_str(), "base");
    RFDETR_ASSERT_EQ_INT(m->config.image_size,  56);
    RFDETR_ASSERT_EQ_INT(m->config.num_queries, 300);
    RFDETR_ASSERT_EQ_INT(m->config.num_classes, 80);
    RFDETR_ASSERT_EQ_INT(m->config.class_names.size(), 80);
    RFDETR_ASSERT_STR_EQ(m->config.class_names[0].c_str(),  "person");
    RFDETR_ASSERT_STR_EQ(m->config.class_names[79].c_str(), "toothbrush");

    // Preprocess
    RFDETR_ASSERT_NEAR(m->config.preprocess_mean[0], 0.485f, 1e-4);
    RFDETR_ASSERT_NEAR(m->config.preprocess_std[2],  0.225f, 1e-4);

    // Backbone / encoder / decoder
    RFDETR_ASSERT_EQ_INT(m->config.backbone.dim,         64);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.depth,       12);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.heads,       12);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.window_size, 14);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.multi_scale_layers.size(), 4);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.multi_scale_layers[0], 2);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.multi_scale_layers[3], 11);
    RFDETR_ASSERT_EQ_INT(m->config.encoder.layers,    3);
    RFDETR_ASSERT_EQ_INT(m->config.encoder.model_dim, 64);
    RFDETR_ASSERT_EQ_INT(m->config.decoder.heads,     8);

    // Tensors present (sample a few)
    RFDETR_ASSERT(m->tensors.count("backbone.patch_embed.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("backbone.blocks.0.attn.qkv.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("backbone.blocks.11.mlp.fc2.bias") == 1);
    RFDETR_ASSERT(m->tensors.count("decoder.queries") == 1);
    RFDETR_ASSERT(m->tensors.count("heads.class.fc.weight") == 1);

    // Validation passes (stub returns OK in Task 5; Task 6 makes it real)
    rfdetr_status v = rfdetr::model_validate_tensors(*m);
    RFDETR_ASSERT_EQ_INT(v, RFDETR_OK);

    // Bad path -> error
    rfdetr_status st2;
    rfdetr::Model* bad = rfdetr::model_load("/no/such/path.gguf", &st2);
    RFDETR_ASSERT(bad == nullptr);
    RFDETR_ASSERT(st2 == RFDETR_ERR_FILE_NOT_FOUND || st2 == RFDETR_ERR_MODEL_FORMAT);

    rfdetr::model_free(m);

    // ---- Missing tensor -> validation fails ----
    {
        std::string bad_path = fixtures + "/model_base_missing.gguf";
        rfdetr_status st_;
        rfdetr::Model* bm = rfdetr::model_load(bad_path, &st_);
        RFDETR_ASSERT(bm != nullptr);
        RFDETR_ASSERT_EQ_INT(st_, RFDETR_OK);

        rfdetr_status v = rfdetr::model_validate_tensors(*bm);
        RFDETR_ASSERT_EQ_INT(v, RFDETR_ERR_MODEL_LOAD);

        rfdetr::model_free(bm);
    }

    // ---- expected_tensor_names produces the right count for base ----
    {
        rfdetr_status st_;
        rfdetr::Model* good = rfdetr::model_load(path, &st_);
        std::vector<std::string> expected = rfdetr::expected_tensor_names(good->config);
        // Loader counted all tensor names; expected list should match.
        RFDETR_ASSERT_EQ_INT(expected.size(), good->tensors.size());
        for (const auto& n : expected) {
            RFDETR_ASSERT(good->tensors.count(n) == 1);
        }
        rfdetr::model_free(good);
    }

    // ---- Realize weights into a backend buffer ----
    {
        rfdetr_status st_;
        rfdetr::Model* mm = rfdetr::model_load(path, &st_);
        RFDETR_ASSERT(mm != nullptr);

        ggml_backend_t backend = rfdetr::init_backend(1, &st_);
        RFDETR_ASSERT(backend != nullptr);
        RFDETR_ASSERT_EQ_INT(st_, RFDETR_OK);

        rfdetr_status rs = rfdetr::model_realize_weights(*mm, backend);
        RFDETR_ASSERT_EQ_INT(rs, RFDETR_OK);
        RFDETR_ASSERT(mm->weights != nullptr);

        /* Idempotent: a second call is OK */
        rfdetr_status rs2 = rfdetr::model_realize_weights(*mm, backend);
        RFDETR_ASSERT_EQ_INT(rs2, RFDETR_OK);

        rfdetr::model_free(mm);
        rfdetr::free_backend(backend);
    }

    return 0;
}
