/* rfdetr-video — detect / segment objects in a video file and write an
 * annotated MP4.
 *
 * Architecture:
 *   This example deliberately does NOT link against libavformat/libavcodec.
 *   ffmpeg is invoked as a child process via popen():
 *
 *     [ ffmpeg -i input -f rawvideo -pix_fmt rgb24 - ]  -> stdout pipe
 *           |
 *           v
 *     rfdetr_image_from_rgb_buffer + rfdetr_detect + visualize in place
 *           |
 *           v
 *     [ ffmpeg -f rawvideo ... -c:v libx264 output.mp4 ]  <- stdin pipe
 *
 * The main librfdetr.so and rfdetr-cli stay free of any ffmpeg dependency
 * (build-time or link-time). ffmpeg is a runtime requirement of THIS example
 * only — users who don't want it can simply not build examples/video/.
 *
 * Usage:
 *   rfdetr-video --model <gguf> --input <video> --output <annotated.mp4>
 *                [--threshold N] [--threads N] [--max-frames N]
 *                [--fps N] [--ffmpeg <path>] [--ffprobe <path>]
 *
 * Limitations:
 *   - Paths must not contain double-quote characters (the popen() command
 *     line wraps them in double quotes for shell expansion).
 *   - x264 is the only codec supported on the encode side (matches what most
 *     ffmpeg packages ship by default; encoder flag can be lifted later).
 */

#include "rfdetr.h"

#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

struct VideoArgs {
    std::string model;
    std::string input;
    std::string output;
    float       threshold  = 0.5f;
    int         threads    = 0;     /* 0 = auto */
    int         max_frames = -1;    /* -1 = no limit */
    double      fps        = 0.0;   /* 0 = use source fps */
    std::string ffmpeg     = "ffmpeg";
    std::string ffprobe    = "ffprobe";
};

void print_usage(FILE* f) {
    std::fprintf(f,
        "rfdetr-video — annotate a video file with rfdetr detections.\n"
        "\n"
        "Usage:\n"
        "  rfdetr-video --model <gguf> --input <video> --output <out.mp4>\n"
        "               [--threshold N] [--threads N] [--max-frames N]\n"
        "               [--fps N] [--ffmpeg PATH] [--ffprobe PATH]\n"
        "\n"
        "Required:\n"
        "  --model PATH         rfdetr GGUF model\n"
        "  --input PATH         input video file (anything ffmpeg can decode)\n"
        "  --output PATH        annotated output (.mp4, x264-encoded)\n"
        "\n"
        "Optional:\n"
        "  --threshold FLOAT    detection confidence threshold (default 0.5)\n"
        "  --threads N          inference threads (0 = auto, default)\n"
        "  --max-frames N       stop after N frames (debug; default: full video)\n"
        "  --fps N              override output framerate (default: source fps)\n"
        "  --ffmpeg PATH        ffmpeg binary (default: ffmpeg in $PATH)\n"
        "  --ffprobe PATH       ffprobe binary (default: ffprobe in $PATH)\n"
        "  -h, --help           print this help and exit\n"
        "\n"
        "ffmpeg + ffprobe must be installed at runtime; the rfdetr library\n"
        "itself does NOT link against libav (subprocess-only).\n");
}

bool parse_args(int argc, char** argv, VideoArgs* out, std::string* err) {
    auto need_val = [&](int i) -> bool {
        if (i + 1 >= argc) { *err = std::string("missing value for ") + argv[i]; return false; }
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(stdout);
            std::exit(0);
        }
        if (a == "--model") {
            if (!need_val(i)) return false;
            out->model = argv[++i];
        } else if (a == "--input") {
            if (!need_val(i)) return false;
            out->input = argv[++i];
        } else if (a == "--output") {
            if (!need_val(i)) return false;
            out->output = argv[++i];
        } else if (a == "--threshold") {
            if (!need_val(i)) return false;
            out->threshold = (float)std::atof(argv[++i]);
        } else if (a == "--threads") {
            if (!need_val(i)) return false;
            out->threads = std::atoi(argv[++i]);
        } else if (a == "--max-frames") {
            if (!need_val(i)) return false;
            out->max_frames = std::atoi(argv[++i]);
        } else if (a == "--fps") {
            if (!need_val(i)) return false;
            out->fps = std::atof(argv[++i]);
        } else if (a == "--ffmpeg") {
            if (!need_val(i)) return false;
            out->ffmpeg = argv[++i];
        } else if (a == "--ffprobe") {
            if (!need_val(i)) return false;
            out->ffprobe = argv[++i];
        } else {
            *err = std::string("unknown argument: ") + a;
            return false;
        }
    }
    if (out->model.empty())  { *err = "--model is required"; return false; }
    if (out->input.empty())  { *err = "--input is required"; return false; }
    if (out->output.empty()) { *err = "--output is required"; return false; }
    return true;
}

int resolve_n_threads(int requested) {
    if (requested > 0) return requested;
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 1;
    return (int)hc;
}

void default_log_cb(rfdetr_log_level lvl, const char* msg, void* /*ud*/) {
    const char* tag = "?";
    switch (lvl) {
        case RFDETR_LOG_DEBUG: tag = "DEBUG"; break;
        case RFDETR_LOG_INFO:  tag = "INFO";  break;
        case RFDETR_LOG_WARN:  tag = "WARN";  break;
        case RFDETR_LOG_ERROR: tag = "ERROR"; break;
    }
    std::fprintf(stderr, "[%s] %s\n", tag, msg);
}

/* -------------------------------------------------------------------------
 * Video probing via ffprobe.
 *
 * Run:
 *   ffprobe -v error -select_streams v:0 \
 *           -show_entries stream=width,height,r_frame_rate,nb_frames \
 *           -of default=noprint_wrappers=1:nokey=0 <input>
 *
 * Parse line-by-line (key=value). r_frame_rate comes as "num/den". nb_frames
 * may be "N/A" for some containers — we treat that as 0 and stream until EOF.
 * ------------------------------------------------------------------------- */
struct VideoInfo {
    int    width    = 0;
    int    height   = 0;
    double fps      = 0.0;
    long   nb_frames = 0;  /* may be 0 if unknown */
};

bool probe_video(const std::string& ffprobe, const std::string& input, VideoInfo* out) {
    char cmd[4096];
    std::snprintf(cmd, sizeof(cmd),
        "%s -v error -select_streams v:0 "
        "-show_entries stream=width,height,r_frame_rate,nb_frames "
        "-of default=noprint_wrappers=1:nokey=0 \"%s\" 2>&1",
        ffprobe.c_str(), input.c_str());
    FILE* p = popen(cmd, "r");
    if (!p) {
        std::fprintf(stderr, "rfdetr-video: failed to spawn ffprobe: %s\n", std::strerror(errno));
        return false;
    }
    char line[512];
    while (std::fgets(line, sizeof(line), p)) {
        /* Strip trailing newline */
        size_t n = std::strlen(line);
        while (n > 0 && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = '\0';
        const char* eq = std::strchr(line, '=');
        if (!eq) continue;
        std::string key(line, eq - line);
        const char* val = eq + 1;
        if (key == "width")  out->width  = std::atoi(val);
        else if (key == "height") out->height = std::atoi(val);
        else if (key == "r_frame_rate") {
            /* "num/den" */
            const char* slash = std::strchr(val, '/');
            if (slash) {
                double num = std::atof(val);
                double den = std::atof(slash + 1);
                if (den > 0.0) out->fps = num / den;
            } else {
                out->fps = std::atof(val);
            }
        } else if (key == "nb_frames") {
            if (std::strcmp(val, "N/A") != 0) out->nb_frames = std::atol(val);
        }
    }
    int rc = pclose(p);
    if (rc != 0) {
        std::fprintf(stderr, "rfdetr-video: ffprobe exited with status %d (cmd: %s)\n", rc, cmd);
        return false;
    }
    if (out->width <= 0 || out->height <= 0) {
        std::fprintf(stderr, "rfdetr-video: ffprobe did not return a valid width/height. "
                             "Is '%s' a video file?\n", input.c_str());
        return false;
    }
    if (out->fps <= 0.0) {
        std::fprintf(stderr, "rfdetr-video: warning: source fps is unknown — defaulting to 25\n");
        out->fps = 25.0;
    }
    return true;
}

/* Read exactly `n` bytes; returns false on EOF or short read. */
bool read_exact(FILE* f, uint8_t* buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        size_t r = std::fread(buf + got, 1, n - got, f);
        if (r == 0) return false;  /* EOF or error */
        got += r;
    }
    return true;
}

/* Write exactly `n` bytes; returns false on short write. */
bool write_exact(FILE* f, const uint8_t* buf, size_t n) {
    size_t put = 0;
    while (put < n) {
        size_t w = std::fwrite(buf + put, 1, n - put, f);
        if (w == 0) return false;
        put += w;
    }
    return true;
}

}  /* anon namespace */

int main(int argc, char** argv) {
    rfdetr_set_log_callback(default_log_cb, nullptr);

    VideoArgs args;
    std::string err;
    if (!parse_args(argc, argv, &args, &err)) {
        std::fprintf(stderr, "error: %s\n\n", err.c_str());
        print_usage(stderr);
        return 1;
    }

    /* 1. Probe the input. */
    VideoInfo info;
    if (!probe_video(args.ffprobe, args.input, &info)) {
        return 2;
    }
    const double out_fps = (args.fps > 0.0) ? args.fps : info.fps;
    std::fprintf(stderr,
        "input: %s  (%dx%d @ %.3f fps%s%ld frames)\n",
        args.input.c_str(), info.width, info.height, info.fps,
        info.nb_frames > 0 ? ", " : ", unknown ",
        info.nb_frames);

    /* 2. Load the model. */
    rfdetr_params init_params{};
    init_params.model_path = args.model.c_str();
    init_params.n_threads  = resolve_n_threads(args.threads);

    rfdetr_status init_st = RFDETR_OK;
    rfdetr_context* ctx = rfdetr_init(&init_params, &init_st);
    if (!ctx) {
        std::fprintf(stderr, "rfdetr_init failed: %s\n", rfdetr_status_str(init_st));
        return 3;
    }

    /* 3. Build the decode and encode commands. paths are wrapped in double
     *    quotes — keep this limitation documented (no quote chars in paths). */
    char dec_cmd[4096];
    std::snprintf(dec_cmd, sizeof(dec_cmd),
        "%s -loglevel error -i \"%s\" -f rawvideo -pix_fmt rgb24 -",
        args.ffmpeg.c_str(), args.input.c_str());

    char enc_cmd[4096];
    std::snprintf(enc_cmd, sizeof(enc_cmd),
        "%s -y -loglevel error -f rawvideo -pix_fmt rgb24 "
        "-video_size %dx%d -framerate %.6f -i - "
        "-c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart \"%s\"",
        args.ffmpeg.c_str(), info.width, info.height, out_fps, args.output.c_str());

    /* 4. Open both pipes. */
    FILE* dec = popen(dec_cmd, "r");
    if (!dec) {
        std::fprintf(stderr, "rfdetr-video: popen decode failed: %s\n  cmd: %s\n",
                     std::strerror(errno), dec_cmd);
        rfdetr_free(ctx);
        return 4;
    }
    FILE* enc = popen(enc_cmd, "w");
    if (!enc) {
        std::fprintf(stderr, "rfdetr-video: popen encode failed: %s\n  cmd: %s\n",
                     std::strerror(errno), enc_cmd);
        pclose(dec);
        rfdetr_free(ctx);
        return 5;
    }

    /* 5. Decode-detect-annotate-encode loop. */
    const size_t frame_bytes = (size_t)info.width * (size_t)info.height * 3;
    std::vector<uint8_t> frame_buf;
    try {
        frame_buf.resize(frame_bytes);
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "rfdetr-video: out of memory allocating frame buffer (%zu bytes)\n",
                     frame_bytes);
        pclose(enc);
        pclose(dec);
        rfdetr_free(ctx);
        return 6;
    }

    int frame_count = 0;
    int last_n_dets = 0;
    int rc = 0;
    while (read_exact(dec, frame_buf.data(), frame_bytes)) {
        if (args.max_frames > 0 && frame_count >= args.max_frames) break;

        rfdetr_status wrap_st = RFDETR_OK;
        rfdetr_image* img = rfdetr_image_from_rgb_buffer(
            frame_buf.data(), info.width, info.height, &wrap_st);
        if (!img) {
            std::fprintf(stderr, "rfdetr-video: rfdetr_image_from_rgb_buffer failed at frame %d: %s\n",
                         frame_count, rfdetr_status_str(wrap_st));
            rc = 7;
            break;
        }

        rfdetr_detect_params dp{};
        dp.threshold = args.threshold;
        dp.top_k     = 300;

        rfdetr_detection* dets = nullptr;
        size_t n = 0;
        rfdetr_status det_st = rfdetr_detect(ctx, img, &dp, &dets, &n);
        if (det_st != RFDETR_OK) {
            std::fprintf(stderr, "rfdetr-video: rfdetr_detect failed at frame %d: %s\n",
                         frame_count, rfdetr_status_str(det_st));
            rfdetr_image_free(img);
            rc = 8;
            break;
        }

        /* Mask overlay first (so bbox strokes sit on top), then boxes + labels. */
        for (size_t i = 0; i < n; ++i) {
            rfdetr_visualize_overlay_mask(img, dets[i], /*alpha*/ 0.4f);
        }
        for (size_t i = 0; i < n; ++i) {
            rfdetr_visualize_draw_box(img, dets[i], /*thickness*/ 3);
        }

        const uint8_t* annotated = rfdetr_image_rgb_data(img);
        if (!annotated || !write_exact(enc, annotated, frame_bytes)) {
            std::fprintf(stderr, "rfdetr-video: short write to encoder at frame %d\n", frame_count);
            rfdetr_detections_free(dets, n);
            rfdetr_image_free(img);
            rc = 9;
            break;
        }

        last_n_dets = (int)n;
        rfdetr_detections_free(dets, n);
        rfdetr_image_free(img);
        ++frame_count;

        if (frame_count % 25 == 0) {
            if (info.nb_frames > 0) {
                std::fprintf(stderr, "  processed %d / %ld frames (last frame: %d detections)\n",
                             frame_count, info.nb_frames, last_n_dets);
            } else {
                std::fprintf(stderr, "  processed %d frames (last frame: %d detections)\n",
                             frame_count, last_n_dets);
            }
        }
    }

    /* 6. Tear down both pipes. Close the encoder stdin first so ffmpeg can
     *    flush the final GOP and exit cleanly. */
    int enc_rc = pclose(enc);
    int dec_rc = pclose(dec);
    rfdetr_free(ctx);

    if (rc != 0) return rc;
    if (frame_count == 0) {
        std::fprintf(stderr, "rfdetr-video: no frames were produced. "
                             "Check that the input is a readable video file.\n");
        return 10;
    }
    if (enc_rc != 0) {
        std::fprintf(stderr, "rfdetr-video: ffmpeg encoder exited with status %d\n", enc_rc);
        return 11;
    }
    /* The decoder process usually exits 0 once we've consumed the whole
     * stream; some containers leave it pending with a non-zero rc when EOF
     * is reached mid-packet, so we log but don't fail on a non-zero dec rc. */
    if (dec_rc != 0) {
        std::fprintf(stderr, "rfdetr-video: note: ffmpeg decoder exited with status %d "
                             "(probably benign — the stream was fully consumed)\n", dec_rc);
    }

    std::fprintf(stderr, "wrote %d frames to %s\n", frame_count, args.output.c_str());
    return 0;
}
