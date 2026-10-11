// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <vulkan/vulkan.h>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <thread>
#include <unistd.h>
#include <signal.h>

namespace mlx::core::omarchy::diagnostic {
struct State {
  std::mutex mutex;
  std::map<std::tuple<std::string, size_t, uint64_t, std::string, std::string, std::string>, uint64_t> oversize;
  uint64_t allocation_high_water = 0;
  int signal_pipe[2]{-1, -1};
};
inline State& state() { static State* s = new State; return *s; }
inline bool enabled(const char* name) {
  const char* v = std::getenv(name);
  return v && *v && std::string(v) != "0";
}
inline void append_file(const std::string& line) {
  const char* path = std::getenv("MLX_OMARCHY_OVERSIZE_REPORT_FILE");
  if (path && *path) { if (FILE* f = std::fopen(path, "a")) { std::fprintf(f, "%s\n", line.c_str()); std::fclose(f); } }
}
inline void summary() {
  auto& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!enabled("MLX_OMARCHY_OVERSIZE_REPORT") && !enabled("MLX_OMARCHY_ALLOC_REPORT")) return;
  std::string line = "[omarchy-report] oversize summary: primitive binding bytes shape dtype reason count";
  std::fprintf(stderr, "%s\n", line.c_str()); append_file(line);
  for (const auto& [key, count] : s.oversize) {
    const auto& [prim, binding, bytes, shape, dtype, reason] = key;
    line = "[omarchy-report] " + prim + " " + std::to_string(binding) + " " + std::to_string(bytes) + " " + shape + " " + dtype + " " + reason + " " + std::to_string(count);
    std::fprintf(stderr, "%s\n", line.c_str()); append_file(line);
  }
  line = "[omarchy-alloc-report] high_water_bytes=" + std::to_string(s.allocation_high_water);
  std::fprintf(stderr, "%s\n", line.c_str()); append_file(line);
}
inline void signal_handler(int) {
  auto& s = state();
  if (s.signal_pipe[1] >= 0) { const char c = 1; (void)::write(s.signal_pipe[1], &c, 1); }
}
inline void install() {
  static std::once_flag once;
  std::call_once(once, [] {
    std::atexit(summary);
    auto& s = state();
    if (::pipe(s.signal_pipe) == 0) {
      std::thread([] { char c; while (::read(state().signal_pipe[0], &c, 1) == 1) summary(); }).detach();
      struct sigaction sa{}; sa.sa_handler = signal_handler; sigemptyset(&sa.sa_mask); sa.sa_flags = SA_RESTART;
      sigaction(SIGUSR1, &sa, nullptr);
    }
  });
}
inline void oversize(const std::string& prim, size_t binding, uint64_t bytes,
                     uint64_t limit, const char* shape = "?", const char* dtype = "?",
                     const char* reason = "range") {
  install();
  auto& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto key = std::make_tuple(prim, binding, bytes, std::string(shape),
                             std::string(dtype), std::string(reason));
  uint64_t count = ++s.oversize[key];
  std::string line = "[omarchy-oversize] primitive=" + prim + " binding=" + std::to_string(binding) +
      " bytes=" + std::to_string(bytes) + " limit=" + std::to_string(limit) +
      " shape=" + shape + " dtype=" + dtype + " reason=" + reason +
      " count=" + std::to_string(count);
  std::fprintf(stderr, "%s\n", line.c_str()); append_file(line);
}
inline void allocation(uint64_t bytes, uint64_t current, uint64_t high_water, VkResult result, const char* caller) {
  if (!enabled("MLX_OMARCHY_ALLOC_REPORT")) return;
  install();
  auto& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.allocation_high_water = std::max(s.allocation_high_water, high_water);
  if (bytes < (256ull << 20)) return;
  std::string line = "[omarchy-allocation] caller=" + std::string(caller && *caller ? caller : "unknown") +
      " bytes=" + std::to_string(bytes) + " current_device_bytes=" + std::to_string(current) +
      " high_water_bytes=" + std::to_string(s.allocation_high_water) + " vkAllocateMemory=" + std::to_string(result);
  std::fprintf(stderr, "%s\n", line.c_str()); append_file(line);
}
} // namespace mlx::core::omarchy::diagnostic
