#include "test_assert.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static bool file_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

int main() {
    const std::string fixtures = RFDETR_TEST_FIXTURES;
    const std::string out_json = std::string(fixtures) + "/generated/cli_out.json";
    const std::string out_png  = std::string(fixtures) + "/generated/cli_out.png";

    std::filesystem::create_directories(std::string(fixtures) + "/generated");
    std::remove(out_json.c_str());
    std::remove(out_png.c_str());

    std::string cmd = std::string(RFDETR_CLI_BINARY) +
                      " detect --model " + fixtures + "/model_base.gguf"
                      " --input "      + fixtures + "/cats.png"
                      " --output "     + out_json +
                      " --annotated "  + out_png;
    int rc = std::system(cmd.c_str());
    RFDETR_ASSERT_EQ_INT(WEXITSTATUS(rc), 0);

    RFDETR_ASSERT(file_exists(out_json));
    RFDETR_ASSERT(file_exists(out_png));

    std::string body = read_file(out_json);
    RFDETR_ASSERT(body.find("\"detections\": [") != std::string::npos);
    RFDETR_ASSERT(body.find("\"width\": 16")    != std::string::npos);
    RFDETR_ASSERT(body.find("\"height\": 16")   != std::string::npos);

    // ---- info on synthesized model ----
    {
        std::string cmd2 = std::string(RFDETR_CLI_BINARY) +
                           " info --model " + fixtures + "/model_base.gguf > " +
                           fixtures + "/generated/info_out.txt 2>&1";
        int rc2 = std::system(cmd2.c_str());
        RFDETR_ASSERT_EQ_INT(WEXITSTATUS(rc2), 0);

        std::string info_body = read_file(fixtures + "/generated/info_out.txt");
        RFDETR_ASSERT(info_body.find("variant:      base")   != std::string::npos);
        RFDETR_ASSERT(info_body.find("image_size:   56")     != std::string::npos);  // shrunken fixture
        RFDETR_ASSERT(info_body.find("num_classes:  80")     != std::string::npos);
        RFDETR_ASSERT(info_body.find("num_queries:  300")    != std::string::npos);
    }

    return 0;
}
