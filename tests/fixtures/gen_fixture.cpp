#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gen_fixture <out_dir>\n");
        return 1;
    }
    const int W = 16, H = 16, C = 3;
    uint8_t px[W*H*C];
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int i = (y*W + x) * C;
            px[i+0] = (uint8_t)((x*16) & 0xff);  // R = horizontal ramp
            px[i+1] = (uint8_t)((y*16) & 0xff);  // G = vertical ramp
            px[i+2] = (uint8_t)((x*y) & 0xff);   // B = product
        }
    }
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/cats.png", argv[1]);
    if (!stbi_write_png(path, W, H, C, px, W*C)) {
        std::fprintf(stderr, "failed to write %s\n", path);
        return 1;
    }
    std::printf("wrote %s\n", path);
    return 0;
}
