#ifndef RFDETR_BACKEND_HPP
#define RFDETR_BACKEND_HPP

#include "rfdetr.h"

#include <cstdint>

struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;

namespace rfdetr {

/* Initialize a backend. Plan 3: CPU only. Later plans honor build flags
 * RFDETR_GGML_CUDA / METAL / VULKAN / HIPBLAS. */
ggml_backend_t init_backend(int n_threads, rfdetr_status* out_status);

/* Release a backend created by init_backend. */
void free_backend(ggml_backend_t b);

/* True if the backend is CPU. Used to choose direct-memcpy vs
 * ggml_backend_tensor_get in tests. */
bool is_cpu(ggml_backend_t b);

}  // namespace rfdetr

#endif
