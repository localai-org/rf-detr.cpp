#ifndef RFDETR_CLI_HPP
#define RFDETR_CLI_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace rfdetr_cli {

struct DetectArgs {
    std::string model;
    std::string input;
    std::string output;
    std::string annotated;
    float       threshold = 0.5f;
    uint32_t    top_k     = 300;
    std::vector<uint32_t> classes;  /* allowlist */
    int         n_threads = 0;       /* 0 = auto (hardware_concurrency) */
};

struct InfoArgs    { std::string model; int n_threads = 0; };
struct BenchArgs   { std::string model; std::string input; int iters = 50; int warmup = 5; int n_threads = 0; };
struct CompareArgs { std::string baseline_dir; std::string image; std::string model; int n_threads = 0; };

/* Quantize an existing rfdetr GGUF to a different ggml type.
 *
 * Positional usage:
 *   rfdetr-cli quantize <input.gguf> <output.gguf> <type>
 *
 * Supported types: f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q4_K, q5_K, q6_K.
 * Only 2D weight tensors with both dims >= 64 are quantized, matching the
 * Python converter's `should_quantize` heuristic. Tensors whose row size is
 * not a multiple of the target type's block size are left F32 (logged).
 */
struct QuantizeArgs {
    std::string input;
    std::string output;
    std::string type;  /* lower-case dtype name */
};

enum class Subcommand { None, Help, Detect, Info, Bench, Compare, Quantize };

struct ParseResult {
    Subcommand   sub = Subcommand::None;
    std::string  error;        /* non-empty on parse failure */
    DetectArgs   detect;
    InfoArgs     info;
    BenchArgs    bench;
    CompareArgs  compare;
    QuantizeArgs quantize;
};

/* Parse argv. argv[0] is the program name. */
ParseResult parse(int argc, char** argv);

/* Print --help text to stdout. */
void print_help();

}  // namespace rfdetr_cli

#endif
