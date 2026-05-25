#ifndef RFDETR_TRACE_HPP
#define RFDETR_TRACE_HPP

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct ggml_tensor;

namespace rfdetr {

/* A trace callback receives named intermediate tensors. The tensor pointer
 * is borrowed and only valid for the duration of the call (graph teardown
 * may invalidate it). The callback is expected to copy out any data it needs.
 *
 * Tensors passed to the callback have been ggml_backend_synchronize-d so
 * their data is readable via ggml_backend_tensor_get (or directly via
 * tensor->data on CPU). */
using trace_cb = std::function<void(const std::string& name, const ggml_tensor* t)>;

/* Install a callback for the current thread. Pass nullptr to clear. */
void set_trace_callback(trace_cb cb);

/* Forward-pass code calls this at each named checkpoint. No-op if no
 * callback is installed. */
void publish(const std::string& name, const ggml_tensor* t);

/* Helper: copy a published tensor's data into a std::vector<float>. The
 * source is assumed to be F32 on the CPU backend (Plan 3 is CPU-only). */
std::vector<float> copy_tensor_to_f32(const ggml_tensor* t);

/* Helper: shape vector. Returns up to GGML_MAX_DIMS entries; trailing 1's
 * are NOT trimmed. */
std::vector<int64_t> tensor_shape(const ggml_tensor* t);

}  // namespace rfdetr

#endif
