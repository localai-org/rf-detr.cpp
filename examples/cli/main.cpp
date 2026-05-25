#include "rfdetr.h"
#include "cli.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

static void default_log_cb(rfdetr_log_level lvl, const char* msg, void* /*ud*/) {
    const char* tag = "?";
    switch (lvl) {
        case RFDETR_LOG_DEBUG: tag = "DEBUG"; break;
        case RFDETR_LOG_INFO:  tag = "INFO";  break;
        case RFDETR_LOG_WARN:  tag = "WARN";  break;
        case RFDETR_LOG_ERROR: tag = "ERROR"; break;
    }
    std::fprintf(stderr, "[%s] %s\n", tag, msg);
}

static int cmd_detect(const rfdetr_cli::DetectArgs& a) {
    rfdetr_status st;
    rfdetr_image* img = rfdetr_image_load_file(a.input.c_str(), &st);
    if (!img) {
        std::fprintf(stderr, "failed to load image '%s': %s\n", a.input.c_str(), rfdetr_status_str(st));
        return 2;
    }

    /* Model loading lives in Plan 2/3. For now we emit zero detections,
     * proving the I/O + output pipeline works end-to-end. */
    rfdetr_detection* dets = nullptr;
    size_t n = 0;

    /* Write JSON */
    std::ofstream out(a.output);
    if (!out.is_open()) {
        std::fprintf(stderr, "failed to open '%s' for writing\n", a.output.c_str());
        rfdetr_image_free(img);
        return 3;
    }
    out << "{\n  \"image\": {\"width\": " << rfdetr_image_width(img)
        << ", \"height\": " << rfdetr_image_height(img) << "},\n"
        << "  \"detections\": [";
    for (size_t i = 0; i < n; ++i) {
        if (i) out << ",";
        out << "\n    {"
            << "\"class_id\": " << dets[i].class_id
            << ", \"score\": " << dets[i].score
            << ", \"x1\": " << dets[i].x1
            << ", \"y1\": " << dets[i].y1
            << ", \"x2\": " << dets[i].x2
            << ", \"y2\": " << dets[i].y2
            << "}";
    }
    out << "\n  ]\n}\n";

    /* Optional annotated PNG */
    if (!a.annotated.empty()) {
        rfdetr_status r = rfdetr_render(img, dets, n, a.annotated.c_str());
        if (r != RFDETR_OK) {
            std::fprintf(stderr, "render failed: %s\n", rfdetr_status_str(r));
            rfdetr_detections_free(dets, n);
            rfdetr_image_free(img);
            return 4;
        }
    }

    rfdetr_detections_free(dets, n);
    rfdetr_image_free(img);
    return 0;
}

int main(int argc, char** argv) {
    rfdetr_set_log_callback(default_log_cb, nullptr);

    auto r = rfdetr_cli::parse(argc, argv);

    if (!r.error.empty()) {
        std::fprintf(stderr, "error: %s\n\n", r.error.c_str());
        rfdetr_cli::print_help();
        return 1;
    }

    switch (r.sub) {
        case rfdetr_cli::Subcommand::Help:
            rfdetr_cli::print_help();
            return 0;
        case rfdetr_cli::Subcommand::Detect:
            return cmd_detect(r.detect);
        case rfdetr_cli::Subcommand::Info:
        case rfdetr_cli::Subcommand::Bench:
        case rfdetr_cli::Subcommand::Compare:
            std::fprintf(stderr, "this subcommand is not yet implemented (see Plan 2/3)\n");
            return 99;
        case rfdetr_cli::Subcommand::None:
            rfdetr_cli::print_help();
            return 1;
    }
    return 1;
}
