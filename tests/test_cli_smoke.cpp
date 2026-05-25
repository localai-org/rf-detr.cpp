#include "test_assert.hpp"
#include "cli.hpp"
#include <vector>
#include <cstring>

static rfdetr_cli::ParseResult run(std::vector<const char*> argv) {
    return rfdetr_cli::parse((int)argv.size(), const_cast<char**>(argv.data()));
}

int main() {
    // No args → Help
    {
        auto r = run({"rfdetr-cli"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Help);
        RFDETR_ASSERT(r.error.empty());
    }

    // --help → Help
    {
        auto r = run({"rfdetr-cli", "--help"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Help);
    }

    // detect with required args
    {
        auto r = run({"rfdetr-cli", "detect",
                      "--model", "m.gguf",
                      "--input", "cats.png",
                      "--output", "out.json",
                      "--annotated", "out.png",
                      "--threshold", "0.4",
                      "--topk", "100",
                      "--classes", "0,16,17"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Detect);
        RFDETR_ASSERT(r.error.empty());
        RFDETR_ASSERT_STR_EQ(r.detect.model.c_str(),     "m.gguf");
        RFDETR_ASSERT_STR_EQ(r.detect.input.c_str(),     "cats.png");
        RFDETR_ASSERT_STR_EQ(r.detect.output.c_str(),    "out.json");
        RFDETR_ASSERT_STR_EQ(r.detect.annotated.c_str(), "out.png");
        RFDETR_ASSERT_NEAR(r.detect.threshold, 0.4f, 1e-5);
        RFDETR_ASSERT_EQ_INT(r.detect.top_k, 100);
        RFDETR_ASSERT_EQ_INT(r.detect.classes.size(), 3);
        RFDETR_ASSERT_EQ_INT(r.detect.classes[0], 0);
        RFDETR_ASSERT_EQ_INT(r.detect.classes[1], 16);
        RFDETR_ASSERT_EQ_INT(r.detect.classes[2], 17);
    }

    // detect with missing required arg → error
    {
        auto r = run({"rfdetr-cli", "detect", "--model", "m.gguf"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Detect);
        RFDETR_ASSERT(!r.error.empty());
    }

    // Unknown subcommand → error
    {
        auto r = run({"rfdetr-cli", "nope"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::None);
        RFDETR_ASSERT(!r.error.empty());
    }

    // info
    {
        auto r = run({"rfdetr-cli", "info", "--model", "m.gguf"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Info);
        RFDETR_ASSERT_STR_EQ(r.info.model.c_str(), "m.gguf");
    }

    // bench
    {
        auto r = run({"rfdetr-cli", "bench",
                      "--model", "m.gguf", "--input", "cats.png",
                      "--iters", "10", "--warmup", "2"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Bench);
        RFDETR_ASSERT_EQ_INT(r.bench.iters, 10);
        RFDETR_ASSERT_EQ_INT(r.bench.warmup, 2);
    }

    // compare
    {
        auto r = run({"rfdetr-cli", "compare",
                      "--baseline", "bundle/", "--image", "cats.png", "--model", "m.gguf"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Compare);
        RFDETR_ASSERT_STR_EQ(r.compare.baseline_dir.c_str(), "bundle/");
        RFDETR_ASSERT_STR_EQ(r.compare.image.c_str(),        "cats.png");
        RFDETR_ASSERT_STR_EQ(r.compare.model.c_str(),        "m.gguf");
    }

    // Out-of-range topk → error
    {
        auto r = run({"rfdetr-cli", "detect",
                      "--model", "m.gguf", "--input", "x.png", "--output", "y.json",
                      "--topk", "99999999999999"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Detect);
        RFDETR_ASSERT(!r.error.empty());
    }
    // Negative topk → error
    {
        auto r = run({"rfdetr-cli", "detect",
                      "--model", "m.gguf", "--input", "x.png", "--output", "y.json",
                      "--topk", "-5"});
        RFDETR_ASSERT(r.sub == rfdetr_cli::Subcommand::Detect);
        RFDETR_ASSERT(!r.error.empty());
    }

    return 0;
}
