#ifndef RFDETR_TRACE_HPP
#define RFDETR_TRACE_HPP

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct ggml_tensor;

namespace rfdetr {

/* A trace callback receives named intermediate tensors at graph-BUILD time.
 *
 * IMPORTANT LIFETIME NOTES:
 *   - The callback fires inline during graph construction (before compute).
 *   - The tensor pointer remains valid until the owning ggml_context is
 *     freed; it is safe to stash for later reading.
 *   - The tensor's data is NOT meaningful at callback time. To read values,
 *     run ggml_backend_graph_compute on the graph, then read the tensor's
 *     contents (e.g. via copy_tensor_to_f32) AFTER compute completes.
 *
 * Typical use: stash {name -> ggml_tensor*} during graph build, read data
 * after compute. */
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
