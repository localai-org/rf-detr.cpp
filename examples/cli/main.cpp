#include "rfdetr.h"
#include "cli.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

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

/* Resolve --threads N:
 *   - N > 0          → use N
 *   - N == 0 (auto)  → use std::thread::hardware_concurrency() (>=1)
 *   - N < 0          → clamped to 1
 */
static int resolve_n_threads(int requested) {
    if (requested > 0) return requested;
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 1;
    return (int)hc;
}

static int cmd_detect(const rfdetr_cli::DetectArgs& a) {
    /* 1. Initialize model context */
    rfdetr_params params{};
    params.model_path = a.model.c_str();
    params.n_threads  = resolve_n_threads(a.n_threads);

    rfdetr_status init_st;
    rfdetr_context* ctx = rfdetr_init(&params, &init_st);
    if (!ctx) {
        std::fprintf(stderr, "rfdetr_init failed: %s\n",
                     rfdetr_status_str(init_st));
        return 2;
    }

    /* 2. Load input image */
    rfdetr_status load_st;
    rfdetr_image* img = rfdetr_image_load_file(a.input.c_str(), &load_st);
    if (!img) {
        std::fprintf(stderr, "failed to load image '%s': %s\n",
                     a.input.c_str(), rfdetr_status_str(load_st));
        rfdetr_free(ctx);
        return 3;
    }

    /* 3. Build detect params from CLI args */
    rfdetr_detect_params dp{};
    dp.threshold        = a.threshold;
    dp.top_k            = a.top_k;
    dp.class_filter     = a.classes.empty() ? nullptr : a.classes.data();
    dp.class_filter_len = a.classes.size();

    /* 4. Run detection */
    rfdetr_detection* dets = nullptr;
    size_t n = 0;
    rfdetr_status det_st = rfdetr_detect(ctx, img, &dp, &dets, &n);
    if (det_st != RFDETR_OK) {
        std::fprintf(stderr, "rfdetr_detect failed: %s\n",
                     rfdetr_status_str(det_st));
        rfdetr_image_free(img);
        rfdetr_free(ctx);
        return 4;
    }

    /* 5. Write JSON output */
    std::ofstream out(a.output);
    if (!out.is_open()) {
        std::fprintf(stderr, "failed to open '%s' for writing\n", a.output.c_str());
        rfdetr_detections_free(dets, n);
        rfdetr_image_free(img);
        rfdetr_free(ctx);
        return 5;
    }
    out << "{\n";
    out << "  \"image\": {\"width\": " << rfdetr_image_width(img)
        << ", \"height\": " << rfdetr_image_height(img) << "},\n";
    out << "  \"detections\": [";
    for (size_t i = 0; i < n; ++i) {
        out << (i ? ",\n    " : "\n    ");
        out << "{"
            << "\"class_id\": " << dets[i].class_id
            << ", \"class_name\": \""
            << (dets[i].class_name ? dets[i].class_name : "")
            << "\""
            << ", \"score\": " << dets[i].score
            << ", \"bbox\": ["
            << dets[i].x1 << ", " << dets[i].y1 << ", "
            << dets[i].x2 << ", " << dets[i].y2
            << "]}";
    }
    if (n > 0) out << "\n  ";
    out << "]\n}\n";
    out.close();

    /* 6. Optional annotated PNG */
    if (!a.annotated.empty()) {
        rfdetr_status render_st = rfdetr_render(img, dets, n, a.annotated.c_str());
        if (render_st != RFDETR_OK) {
            std::fprintf(stderr, "rfdetr_render failed: %s\n",
                         rfdetr_status_str(render_st));
        }
    }

    /* 7. Cleanup */
    rfdetr_detections_free(dets, n);
    rfdetr_image_free(img);
    rfdetr_free(ctx);
    return 0;
}

static int cmd_bench(const rfdetr_cli::BenchArgs& a) {
    /* 1. Initialize model context (load happens once). */
    rfdetr_params params{};
    params.model_path = a.model.c_str();
    params.n_threads  = resolve_n_threads(a.n_threads);

    using clock = std::chrono::steady_clock;
    auto t_load_start = clock::now();
    rfdetr_status init_st;
    rfdetr_context* ctx = rfdetr_init(&params, &init_st);
    auto t_load_end = clock::now();
    if (!ctx) {
        std::fprintf(stderr, "rfdetr_init failed: %s\n",
                     rfdetr_status_str(init_st));
        return 2;
    }
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    /* 2. Load input image once. */
    rfdetr_status load_st;
    rfdetr_image* img = rfdetr_image_load_file(a.input.c_str(), &load_st);
    if (!img) {
        std::fprintf(stderr, "failed to load image '%s': %s\n",
                     a.input.c_str(), rfdetr_status_str(load_st));
        rfdetr_free(ctx);
        return 3;
    }

    rfdetr_detect_params dp{};
    dp.threshold = 0.5f;
    dp.top_k     = 300;

    const int warmup = std::max(0, a.warmup);
    const int iters  = std::max(1, a.iters);

    std::printf("model:     %s\n", a.model.c_str());
    std::printf("image:     %s (%dx%d)\n", a.input.c_str(),
                rfdetr_image_width(img), rfdetr_image_height(img));
    std::printf("threads:   %d\n", params.n_threads);
    std::printf("load_ms:   %.2f\n", load_ms);
    std::printf("warmup:    %d\n", warmup);
    std::printf("iters:     %d\n", iters);
    std::fflush(stdout);

    /* 3. Warmup. */
    for (int i = 0; i < warmup; ++i) {
        rfdetr_detection* dets = nullptr;
        size_t n = 0;
        rfdetr_status st = rfdetr_detect(ctx, img, &dp, &dets, &n);
        if (st != RFDETR_OK) {
            std::fprintf(stderr, "warmup %d: rfdetr_detect failed: %s\n", i,
                         rfdetr_status_str(st));
            rfdetr_image_free(img);
            rfdetr_free(ctx);
            return 4;
        }
        rfdetr_detections_free(dets, n);
    }

    /* 4. Timed iterations. */
    std::vector<double> ms_per_iter;
    ms_per_iter.reserve((size_t)iters);
    size_t last_n = 0;
    for (int i = 0; i < iters; ++i) {
        rfdetr_detection* dets = nullptr;
        size_t n = 0;
        auto t0 = clock::now();
        rfdetr_status st = rfdetr_detect(ctx, img, &dp, &dets, &n);
        auto t1 = clock::now();
        if (st != RFDETR_OK) {
            std::fprintf(stderr, "iter %d: rfdetr_detect failed: %s\n", i,
                         rfdetr_status_str(st));
            rfdetr_image_free(img);
            rfdetr_free(ctx);
            return 4;
        }
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ms_per_iter.push_back(ms);
        last_n = n;
        rfdetr_detections_free(dets, n);
    }

    /* 5. Aggregate. */
    std::vector<double> sorted = ms_per_iter;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0.0;
    for (double v : ms_per_iter) sum += v;
    double mean   = sum / (double)ms_per_iter.size();
    double minv   = sorted.front();
    double maxv   = sorted.back();
    double median = sorted[sorted.size() / 2];

    std::printf("detections: %zu\n", last_n);
    std::printf("min_ms:    %.2f\n", minv);
    std::printf("median_ms: %.2f\n", median);
    std::printf("mean_ms:   %.2f\n", mean);
    std::printf("max_ms:    %.2f\n", maxv);

    rfdetr_image_free(img);
    rfdetr_free(ctx);
    return 0;
}

static int cmd_info(const rfdetr_cli::InfoArgs& a) {
    rfdetr_params p{};
    p.model_path = a.model.c_str();
    p.n_threads  = resolve_n_threads(a.n_threads);

    rfdetr_status st;
    rfdetr_context* ctx = rfdetr_init(&p, &st);
    if (!ctx) {
        std::fprintf(stderr, "rfdetr_init failed: %s\n", rfdetr_status_str(st));
        return 2;
    }

    std::printf("variant:      %s\n", rfdetr_context_variant(ctx));
    std::printf("image_size:   %u\n", rfdetr_context_image_size(ctx));
    std::printf("num_classes:  %u\n", rfdetr_context_num_classes(ctx));
    std::printf("num_queries:  %u\n", rfdetr_context_num_queries(ctx));
    std::printf("n_tensors:    %zu\n", rfdetr_context_n_tensors(ctx));

    rfdetr_free(ctx);
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
            return cmd_info(r.info);
        case rfdetr_cli::Subcommand::Bench:
            return cmd_bench(r.bench);
        case rfdetr_cli::Subcommand::Compare:
            std::fprintf(stderr, "this subcommand is not yet implemented (see Plan 3)\n");
            return 99;
        case rfdetr_cli::Subcommand::None:
            rfdetr_cli::print_help();
            return 1;
    }
    return 1;
}
