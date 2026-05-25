#include "cli.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace rfdetr_cli {

namespace {

bool eat_value(int argc, char** argv, int& i, const char* flag, std::string& out, std::string& err) {
    if (i + 1 >= argc) {
        err = std::string("missing value for ") + flag;
        return false;
    }
    out = argv[++i];
    return true;
}

bool parse_uint(const std::string& s, uint32_t& out, std::string& err, const char* flag) {
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || v < 0) {
        err = std::string("invalid integer for ") + flag + ": " + s;
        return false;
    }
    out = (uint32_t)v;
    return true;
}

bool parse_float(const std::string& s, float& out, std::string& err, const char* flag) {
    char* end = nullptr;
    float v = std::strtof(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') {
        err = std::string("invalid float for ") + flag + ": " + s;
        return false;
    }
    out = v;
    return true;
}

bool parse_classes(const std::string& s, std::vector<uint32_t>& out, std::string& err) {
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        uint32_t v;
        if (!parse_uint(tok, v, err, "--classes")) return false;
        out.push_back(v);
    }
    return true;
}

}  // namespace

void print_help() {
    std::puts(
        "rfdetr-cli — RF-DETR inference CLI\n"
        "\n"
        "Usage:\n"
        "  rfdetr-cli detect  --model <gguf> --input <image> --output <json>\n"
        "                     [--annotated <png>] [--threshold N] [--topk N]\n"
        "                     [--classes id,id,...]\n"
        "  rfdetr-cli info    --model <gguf>\n"
        "  rfdetr-cli bench   --model <gguf> --input <image> [--iters N] [--warmup N]\n"
        "  rfdetr-cli compare --baseline <dir> --image <png> --model <gguf>\n"
        "  rfdetr-cli --help\n");
}

ParseResult parse(int argc, char** argv) {
    ParseResult r;

    if (argc < 2) { r.sub = Subcommand::Help; return r; }

    std::string first = argv[1];
    if (first == "--help" || first == "-h") { r.sub = Subcommand::Help; return r; }

    if (first == "detect") {
        r.sub = Subcommand::Detect;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--model")     { if (!eat_value(argc, argv, i, "--model",    r.detect.model,     r.error)) return r; }
            else if (a == "--input")     { if (!eat_value(argc, argv, i, "--input",    r.detect.input,     r.error)) return r; }
            else if (a == "--output")    { if (!eat_value(argc, argv, i, "--output",   r.detect.output,    r.error)) return r; }
            else if (a == "--annotated") { if (!eat_value(argc, argv, i, "--annotated",r.detect.annotated, r.error)) return r; }
            else if (a == "--threshold") {
                std::string v; if (!eat_value(argc, argv, i, "--threshold", v, r.error)) return r;
                if (!parse_float(v, r.detect.threshold, r.error, "--threshold")) return r;
            }
            else if (a == "--topk") {
                std::string v; if (!eat_value(argc, argv, i, "--topk", v, r.error)) return r;
                if (!parse_uint(v, r.detect.top_k, r.error, "--topk")) return r;
            }
            else if (a == "--classes") {
                std::string v; if (!eat_value(argc, argv, i, "--classes", v, r.error)) return r;
                if (!parse_classes(v, r.detect.classes, r.error)) return r;
            }
            else {
                r.error = "unknown flag: " + a;
                return r;
            }
        }
        if (r.detect.model.empty())  { r.error = "detect: --model is required";  return r; }
        if (r.detect.input.empty())  { r.error = "detect: --input is required";  return r; }
        if (r.detect.output.empty()) { r.error = "detect: --output is required"; return r; }
        return r;
    }

    if (first == "info") {
        r.sub = Subcommand::Info;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--model") { if (!eat_value(argc, argv, i, "--model", r.info.model, r.error)) return r; }
            else { r.error = "unknown flag: " + a; return r; }
        }
        if (r.info.model.empty()) { r.error = "info: --model is required"; return r; }
        return r;
    }

    if (first == "bench") {
        r.sub = Subcommand::Bench;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--model")  { if (!eat_value(argc, argv, i, "--model",  r.bench.model, r.error)) return r; }
            else if (a == "--input")  { if (!eat_value(argc, argv, i, "--input",  r.bench.input, r.error)) return r; }
            else if (a == "--iters")  {
                std::string v; if (!eat_value(argc, argv, i, "--iters",  v, r.error)) return r;
                uint32_t u; if (!parse_uint(v, u, r.error, "--iters")) return r;  r.bench.iters = (int)u;
            }
            else if (a == "--warmup") {
                std::string v; if (!eat_value(argc, argv, i, "--warmup", v, r.error)) return r;
                uint32_t u; if (!parse_uint(v, u, r.error, "--warmup")) return r; r.bench.warmup = (int)u;
            }
            else { r.error = "unknown flag: " + a; return r; }
        }
        if (r.bench.model.empty()) { r.error = "bench: --model is required"; return r; }
        if (r.bench.input.empty()) { r.error = "bench: --input is required"; return r; }
        return r;
    }

    if (first == "compare") {
        r.sub = Subcommand::Compare;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--baseline") { if (!eat_value(argc, argv, i, "--baseline", r.compare.baseline_dir, r.error)) return r; }
            else if (a == "--image")    { if (!eat_value(argc, argv, i, "--image",    r.compare.image,        r.error)) return r; }
            else if (a == "--model")    { if (!eat_value(argc, argv, i, "--model",    r.compare.model,        r.error)) return r; }
            else { r.error = "unknown flag: " + a; return r; }
        }
        if (r.compare.baseline_dir.empty()) { r.error = "compare: --baseline is required"; return r; }
        if (r.compare.image.empty())        { r.error = "compare: --image is required";    return r; }
        if (r.compare.model.empty())        { r.error = "compare: --model is required";    return r; }
        return r;
    }

    r.sub = Subcommand::None;
    r.error = "unknown subcommand: " + first;
    return r;
}

}  // namespace rfdetr_cli
