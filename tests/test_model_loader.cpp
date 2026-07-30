#include "test_assert.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "rfdetr.h"
#include <algorithm>
#include <string>
#include <vector>

/* Collects RFDETR_LOG_ERROR messages into the vector passed as user_data, so a
 * test can assert on what an error path actually told the user. */
static void capture_errors_cb(rfdetr_log_level lvl, const char* msg, void* ud) {
    if (lvl == RFDETR_LOG_ERROR) {
        static_cast<std::vector<std::string>*>(ud)->emplace_back(msg);
    }
}

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
    RFDETR_ASSERT_EQ_INT(m->config.patch_size,  14);
    RFDETR_ASSERT_EQ_INT(m->config.num_queries, 300);
    RFDETR_ASSERT_EQ_INT(m->config.group_detr,  13);
    RFDETR_ASSERT_EQ_INT(m->config.num_classes, 91);
    RFDETR_ASSERT_EQ_INT(m->config.class_names.size(), 91);
    // Position 0 is COCO id 0 ("person"); position 90 is "toothbrush" (id 90).
    RFDETR_ASSERT_STR_EQ(m->config.class_names[0].c_str(),  "person");
    RFDETR_ASSERT_STR_EQ(m->config.class_names[90].c_str(), "toothbrush");

    // Preprocess
    RFDETR_ASSERT_NEAR(m->config.preprocess_mean[0], 0.485f, 1e-4);
    RFDETR_ASSERT_NEAR(m->config.preprocess_std[2],  0.225f, 1e-4);

    // Backbone (shrunk fixture: dim=64, depth=12, heads=8, ffn=128, 4 windows)
    RFDETR_ASSERT_EQ_INT(m->config.backbone.dim,          64);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.depth,        12);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.heads,        8);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.ffn_dim,      128);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.num_windows,  4);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.global_attn_indices.size(), 4);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.global_attn_indices[0], 2);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.global_attn_indices[3], 11);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.out_feature_indices.size(), 4);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.out_feature_indices[0], 2);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.out_feature_indices[3], 11);
    RFDETR_ASSERT_EQ_INT(m->config.backbone.pos_embed_train_size, 4);

    // Projector
    RFDETR_ASSERT_EQ_INT(m->config.projector.in_dim,         256); // 4 * 64
    RFDETR_ASSERT_EQ_INT(m->config.projector.out_dim,        64);
    RFDETR_ASSERT_EQ_INT(m->config.projector.bottleneck_dim, 32);
    RFDETR_ASSERT_EQ_INT(m->config.projector.n_bottlenecks,  3);

    // Decoder
    RFDETR_ASSERT_EQ_INT(m->config.decoder.layers,              3);
    RFDETR_ASSERT_EQ_INT(m->config.decoder.model_dim,           64);
    RFDETR_ASSERT_EQ_INT(m->config.decoder.ffn_dim,             128);
    RFDETR_ASSERT_EQ_INT(m->config.decoder.self_attn_heads,     8);
    RFDETR_ASSERT_EQ_INT(m->config.decoder.cross_attn_heads,    8);
    RFDETR_ASSERT_EQ_INT(m->config.decoder.cross_attn_n_levels, 1);
    RFDETR_ASSERT_EQ_INT(m->config.decoder.cross_attn_n_points, 2);

    // Two-stage
    RFDETR_ASSERT_EQ_INT(m->config.two_stage.n_groups, 13);

    // Tensors present (sample a few from each section)
    RFDETR_ASSERT(m->tensors.count("backbone.patch_embed.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("backbone.cls_token") == 1);
    RFDETR_ASSERT(m->tensors.count("backbone.pos_embed") == 1);
    RFDETR_ASSERT(m->tensors.count("backbone.blocks.0.attn.q.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("backbone.blocks.0.attn.k.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("backbone.blocks.0.attn.v.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("backbone.blocks.0.layer_scale1") == 1);
    RFDETR_ASSERT(m->tensors.count("backbone.blocks.11.layer_scale2") == 1);
    RFDETR_ASSERT(m->tensors.count("backbone.blocks.11.mlp.fc2.bias") == 1);
    RFDETR_ASSERT(m->tensors.count("projector.cv1.conv.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("projector.bottleneck.0.cv1.conv.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("projector.bottleneck.2.cv2.norm.bias") == 1);
    RFDETR_ASSERT(m->tensors.count("projector.final_norm.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("two_stage.enc_output.0.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("two_stage.enc_output.12.bias") == 1);
    RFDETR_ASSERT(m->tensors.count("two_stage.enc_out_bbox_embed.0.layers.2.bias") == 1);
    RFDETR_ASSERT(m->tensors.count("decoder.queries.feat") == 1);
    RFDETR_ASSERT(m->tensors.count("decoder.queries.refpoints") == 1);
    RFDETR_ASSERT(m->tensors.count("decoder.ref_point_head.layers.0.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("decoder.layers.0.self_attn.in_proj.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("decoder.layers.2.cross_attn.sampling_offsets.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("decoder.layers.2.cross_attn.value_proj.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("decoder.norm.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("heads.class_embed.weight") == 1);
    RFDETR_ASSERT(m->tensors.count("heads.bbox_embed.layers.2.weight") == 1);

    // Validation passes
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
        // Sanity: 486 tensors for the v2 base schema.
        RFDETR_ASSERT_EQ_INT(expected.size(), 486);
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

    // ---- realize_weights error paths ----
    {
        rfdetr_status st_;
        rfdetr::Model* mm = rfdetr::model_load(path, &st_);
        RFDETR_ASSERT(mm != nullptr);

        // Null backend → INVALID_ARG
        RFDETR_ASSERT_EQ_INT(rfdetr::model_realize_weights(*mm, nullptr),
                             RFDETR_ERR_INVALID_ARG);

        rfdetr::model_free(mm);
    }

    // ---- Seg head has one block per decoder layer, not a hardcoded 4 ----
    {
        // A 5-decoder-layer seg variant (seg-medium) must expect blocks 0..4.
        rfdetr::Config cfg;
        cfg.has_segmentation_head = true;
        cfg.decoder.layers = 5;
        std::vector<std::string> names = rfdetr::expected_tensor_names(cfg);

        auto has = [&names](const std::string& n) {
            return std::find(names.begin(), names.end(), n) != names.end();
        };
        RFDETR_ASSERT(has("segmentation_head.blocks.0.dwconv.weight"));
        RFDETR_ASSERT(has("segmentation_head.blocks.4.dwconv.weight"));
        RFDETR_ASSERT(has("segmentation_head.blocks.4.pwconv1.bias"));
        RFDETR_ASSERT(!has("segmentation_head.blocks.5.dwconv.weight"));

        // A 6-layer variant (seg-xlarge) expects blocks 0..5.
        rfdetr::Config cfg6;
        cfg6.has_segmentation_head = true;
        cfg6.decoder.layers = 6;
        std::vector<std::string> names6 = rfdetr::expected_tensor_names(cfg6);
        auto has6 = [&names6](const std::string& n) {
            return std::find(names6.begin(), names6.end(), n) != names6.end();
        };
        RFDETR_ASSERT(has6("segmentation_head.blocks.5.dwconv.weight"));

        // A 4-layer variant (seg-nano) is unchanged: blocks 0..3 only.
        rfdetr::Config cfg4;
        cfg4.has_segmentation_head = true;
        cfg4.decoder.layers = 4;
        std::vector<std::string> names4 = rfdetr::expected_tensor_names(cfg4);
        auto has4 = [&names4](const std::string& n) {
            return std::find(names4.begin(), names4.end(), n) != names4.end();
        };
        RFDETR_ASSERT(has4("segmentation_head.blocks.3.dwconv.weight"));
        RFDETR_ASSERT(!has4("segmentation_head.blocks.4.dwconv.weight"));
    }

    // ---- Pre-fix seg GGUFs are rejected with a specific error ----
    {
        // A Model is default-constructible; count_segmentation_blocks only
        // checks key presence, so nullptr tensor values are fine here.
        rfdetr::Model fake;
        fake.config.has_segmentation_head = true;
        fake.config.decoder.layers = 5;

        // Simulate a GGUF converted before the fix: only 4 blocks present.
        for (int b = 0; b < 4; ++b) {
            const std::string p =
                "segmentation_head.blocks." + std::to_string(b) + ".";
            fake.tensors[p + "dwconv.weight"] = nullptr;
        }
        RFDETR_ASSERT_EQ_INT((int)rfdetr::count_segmentation_blocks(fake), 4);

        /* The status alone cannot distinguish the guard from the generic
         * missing-tensor path: a short block count always implies missing
         * expected names, so both return RFDETR_ERR_MODEL_LOAD. The message is
         * the deliverable, so assert on the message. */
        std::vector<std::string> errors;
        rfdetr_set_log_callback(capture_errors_cb, &errors);

        rfdetr_status v = rfdetr::model_validate_tensors(fake);
        RFDETR_ASSERT_EQ_INT(v, RFDETR_ERR_MODEL_LOAD);

        RFDETR_ASSERT_EQ_INT(errors.size(), 1);
        const std::string& stale = errors[0];
        // Names both counts: what the file has, and what the model needs.
        RFDETR_ASSERT(stale.find("4 head block") != std::string::npos);
        RFDETR_ASSERT(stale.find("5 decoder layers") != std::string::npos);
        // Names the cause and the remedy.
        RFDETR_ASSERT(stale.find("converted before") != std::string::npos);
        RFDETR_ASSERT(stale.find("convert_rfdetr_to_gguf.py") != std::string::npos);
        // And is not the generic missing-tensor message the guard replaces.
        RFDETR_ASSERT(stale.find("missing tensor") == std::string::npos);

        // Seg metadata but no seg tensors at all is a different fault, and the
        // stale-file cause must not be asserted for it.
        errors.clear();
        rfdetr::Model empty_seg;
        empty_seg.config.has_segmentation_head = true;
        empty_seg.config.decoder.layers = 5;
        RFDETR_ASSERT_EQ_INT((int)rfdetr::count_segmentation_blocks(empty_seg), 0);
        RFDETR_ASSERT_EQ_INT(rfdetr::model_validate_tensors(empty_seg),
                             RFDETR_ERR_MODEL_LOAD);
        RFDETR_ASSERT_EQ_INT(errors.size(), 1);
        const std::string& none = errors[0];
        RFDETR_ASSERT(none.find("5 head block") != std::string::npos);
        RFDETR_ASSERT(none.find("segmentation_head.blocks.* tensors") != std::string::npos);
        RFDETR_ASSERT(none.find("converted before") == std::string::npos);

        // Restore the default (no callback); nothing else in this file sets one.
        rfdetr_set_log_callback(nullptr, nullptr);

        // A correctly-converted 5-block model gets past the block-count guard.
        // (It still fails on the other missing seg tensors, which is fine —
        // we only assert the guard itself counts correctly.)
        fake.tensors["segmentation_head.blocks.4.dwconv.weight"] = nullptr;
        RFDETR_ASSERT_EQ_INT((int)rfdetr::count_segmentation_blocks(fake), 5);

        // Detection models have no seg head and must count zero.
        rfdetr::Model det;
        det.config.has_segmentation_head = false;
        det.config.decoder.layers = 3;
        RFDETR_ASSERT_EQ_INT((int)rfdetr::count_segmentation_blocks(det), 0);
    }

    return 0;
}
