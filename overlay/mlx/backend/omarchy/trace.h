// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstdint>

#ifdef MLX_OMARCHY_GPU_PROFILING
#include <string_view>
#include <unordered_map>
#endif

// Backend dispatch trace counters (plan R8). mlx-omarchy-info and the runtime
// tests read these to prove which backend executed work.
namespace mlx::core::omarchy::trace {

struct Counters {
  // Number of tensor primitives dispatched to the Omarchy GPU evaluator.
  std::atomic<uint64_t> gpu_primitive_dispatches{0};
  // Number of vkQueueSubmit calls made by the backend.
  std::atomic<uint64_t> vk_submissions{0};
  // Number of recorded vkCmdCopyBuffer commands.
  std::atomic<uint64_t> vk_buffer_copies{0};
  // Number of recorded vkCmdFillBuffer commands.
  std::atomic<uint64_t> vk_buffer_fills{0};
  // Number of recorded Vulkan compute dispatches.
  std::atomic<uint64_t> vk_compute_dispatches{0};
  // Descriptor write counts from dispatch ComputeBinding updates (one
  // write per binding). Structural: moves with vk_compute_dispatches.
  std::atomic<uint64_t> vk_descriptor_update_writes{0};
  // Dependency-barrier decisions (MLX_OMARCHY_GATED_BARRIERS accounting;
  // the default unconditional path counts every pre+post dispatch
  // barrier as emitted, never skipped). Excludes the TAPE_FULL_BARRIERS
  // diagnostic barriers.
  std::atomic<uint64_t> barriers_emitted{0};
  std::atomic<uint64_t> barriers_skipped{0};
  // Number of gpu::finalize calls (throttle points and graph ends).
  std::atomic<uint64_t> omarchy_finalize_calls{0};
  // Commits that submitted a real batch (work, semaphores, or handlers).
  std::atomic<uint64_t> commit_calls_with_work{0};
  // Commits that found nothing pending (finalize on an idle encoder).
  std::atomic<uint64_t> commit_calls_noop{0};
  // Number of Vulkan compute dispatches recorded from inside
  // eval_compiled_tape, i.e. issued on behalf of a tape node rather
  // than an eager primitive. Attributes per-token dispatch counts
  // between the tape and eager paths.
  std::atomic<uint64_t> compiled_tape_dispatches{0};
  // Number of Compiled-tape nodes the interpreter evaluated, whether
  // they recorded a dispatch or not.
  std::atomic<uint64_t> compiled_tape_node_evaluations{0};
  std::atomic<uint64_t> ane_models_loaded{0};
  std::atomic<uint64_t> ane_packages_compiled{0};
  std::atomic<uint64_t> ane_package_cache_hits{0};
  std::atomic<uint64_t> ane_worker_starts{0};
  std::atomic<uint64_t> ane_submissions{0};
  std::atomic<uint64_t> ane_timeouts{0};
  std::atomic<uint64_t> ane_input_bytes{0};
  std::atomic<uint64_t> ane_output_bytes{0};
  std::atomic<uint64_t> ane_exec_ns{0};
};
inline Counters& counters() {
  static Counters counters_;
  return counters_;
}
struct AneTraceSnapshot {
  uint64_t ane_models_loaded;
  uint64_t ane_packages_compiled;
  uint64_t ane_package_cache_hits;
  uint64_t ane_worker_starts;
  uint64_t ane_submissions;
  uint64_t ane_timeouts;
  uint64_t ane_input_bytes;
  uint64_t ane_output_bytes;
  uint64_t ane_exec_ns;
};

static inline void ane_counter_add(
    std::atomic<uint64_t>& counter, uint64_t value = 1) noexcept {
#ifdef MLX_OMARCHY_ANE_TRACING
  counter.fetch_add(value, std::memory_order_relaxed);
#else
  (void)counter;
  (void)value;
#endif
}

inline void record_ane_model_loaded() noexcept {
  ane_counter_add(counters().ane_models_loaded);
}

inline void record_ane_package_compiled() noexcept {
  ane_counter_add(counters().ane_packages_compiled);
}

inline void record_ane_package_cache_hit() noexcept {
  ane_counter_add(counters().ane_package_cache_hits);
}

inline void record_ane_worker_start() noexcept {
  ane_counter_add(counters().ane_worker_starts);
}

inline void record_ane_submission(uint64_t input_bytes) noexcept {
  ane_counter_add(counters().ane_submissions);
  ane_counter_add(counters().ane_input_bytes, input_bytes);
}

inline void record_ane_completion(
    uint64_t output_bytes, uint64_t exec_ns) noexcept {
  ane_counter_add(counters().ane_output_bytes, output_bytes);
  ane_counter_add(counters().ane_exec_ns, exec_ns);
}

inline void record_ane_timeout(uint64_t exec_ns) noexcept {
  ane_counter_add(counters().ane_timeouts);
  ane_counter_add(counters().ane_exec_ns, exec_ns);
}

inline AneTraceSnapshot ane_trace_snapshot() noexcept {
  auto& values = counters();
  return {
      values.ane_models_loaded.load(std::memory_order_relaxed),
      values.ane_packages_compiled.load(std::memory_order_relaxed),
      values.ane_package_cache_hits.load(std::memory_order_relaxed),
      values.ane_worker_starts.load(std::memory_order_relaxed),
      values.ane_submissions.load(std::memory_order_relaxed),
      values.ane_timeouts.load(std::memory_order_relaxed),
      values.ane_input_bytes.load(std::memory_order_relaxed),
      values.ane_output_bytes.load(std::memory_order_relaxed),
      values.ane_exec_ns.load(std::memory_order_relaxed),
  };
}

// Process-wide snapshot for in-process readers (ctypes from Python, test
// harnesses). Plain C ABI because the C++ symbol for counters() is inlined
// away and never reaches the dynamic symbol table. libmlx.so is already
// loaded by the Python extension, so ctypes.CDLL on the same path resolves
// to the same static instance.
struct MlxOmarchyTraceSnapshot {
  uint64_t gpu_primitive_dispatches;
  uint64_t vk_submissions;
  uint64_t vk_buffer_copies;
  uint64_t vk_buffer_fills;
  uint64_t vk_compute_dispatches;
  uint64_t omarchy_finalize_calls;
  uint64_t commit_calls_with_work;
  uint64_t commit_calls_noop;
};

#ifdef MLX_OMARCHY_GPU_PROFILING
// Per-primitive-name gpu::eval counts for fragmentation attribution.
// Written on the evaluator thread only; the names are static string
// literals owned by the primitive classes, so the string_view keys
// outlive the map. Read back through mlx_omarchy_prim_dump() from the
// instrumentation driver (ctypes). Compiled out of release builds.
inline std::unordered_map<std::string_view, std::uint64_t>& prim_counts() {
  static std::unordered_map<std::string_view, std::uint64_t> counts;
  return counts;
}
#endif


// The MLX primitive currently being evaluated on this thread; set in
// eval.cpp around eval_gpu so transport-level traces (copy.cpp) can
// attribute their dispatches to the consuming primitive.
inline std::string_view& current_prim() {
  static thread_local std::string_view prim = "";
  return prim;
}

} // namespace mlx::core::omarchy::trace

// Global-scope C ABI (defined in eval.cpp): the C symbol must not live in
// a namespace, and a qualified namespace definition nested in another
// namespace does not resolve to this scope. ctypes resolves the plain
// symbol on the already-loaded libmlx.so.
extern "C" __attribute__((visibility("default"))) void
mlx_omarchy_trace_snapshot(
    mlx::core::omarchy::trace::MlxOmarchyTraceSnapshot* out);

#ifdef MLX_OMARCHY_GPU_PROFILING
extern "C" __attribute__((visibility("default"))) void
mlx_omarchy_prim_dump(const char* path);
extern "C" __attribute__((visibility("default"))) void
mlx_omarchy_prim_reset(void);
#endif
