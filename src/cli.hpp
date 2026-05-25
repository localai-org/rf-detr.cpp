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
};

struct InfoArgs    { std::string model; };
struct BenchArgs   { std::string model; std::string input; int iters = 50; int warmup = 5; };
struct CompareArgs { std::string baseline_dir; std::string image; std::string model; };

enum class Subcommand { None, Help, Detect, Info, Bench, Compare };

struct ParseResult {
    Subcommand   sub = Subcommand::None;
    std::string  error;        /* non-empty on parse failure */
    DetectArgs   detect;
    InfoArgs     info;
    BenchArgs    bench;
    CompareArgs  compare;
};

/* Parse argv. argv[0] is the program name. */
ParseResult parse(int argc, char** argv);

/* Print --help text to stdout. */
void print_help();

}  // namespace rfdetr_cli

#endif
