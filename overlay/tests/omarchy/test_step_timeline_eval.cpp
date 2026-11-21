// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Verifies one-op batch_open timestamp ordering through the real GPU eval
// path (eval.cpp -> encoder.cpp -> compute.cpp -> step_timeline.h).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace {

bool extract_u64_field(
    const std::string& line,
    const char* key,
    uint64_t& out) {
  const auto pos = line.find(key);
  if (pos == std::string::npos) {
    return false;
  }
  char* end = nullptr;
  const char* start = line.c_str() + pos + std::strlen(key);
  out = std::strtoull(start, &end, 10);
  return end != start;
}

std::string extract_str_field(const std::string& line, const char* key) {
  const auto pos = line.find(key);
  if (pos == std::string::npos) {
    return {};
  }
  const size_t start = pos + std::strlen(key);
  const auto end = line.find('"', start);
  if (end == std::string::npos) {
    return {};
  }
  return line.substr(start, end - start);
}

int run_child_one_op() {
  if (!mx::gpu::is_available()) {
    std::fprintf(stderr, "Skipping: GPU backend unavailable\n");
    return 0;
  }
  auto stream = mx::default_stream(mx::Device::gpu);
  auto a = mx::array({1.0f, 2.0f, 3.0f, 4.0f}, mx::float32);
  auto b = mx::array({5.0f, 6.0f, 7.0f, 8.0f}, mx::float32);
  // Single-op GPU batch: eval() must stamp step_begin/batch_open BEFORE the
  // first primitive's pipeline_lookup, desc_update, cmd_begin, and cmd_record.
  auto c = mx::add(a, b, stream);
  mx::eval(c);
  mx::synchronize(stream);
  const float* data = c.data<float>();
  if (data[0] != 6.0f || data[3] != 12.0f) {
    std::fprintf(stderr, "one-op add mismatch: %f %f\n", data[0], data[3]);
    return 2;
  }
  return 0;
}

} // namespace

int main() {
  char tmpl[] = "/tmp/omarchy-tl-oneop-XXXXXX";
  int fd = ::mkstemp(tmpl);
  if (fd < 0) {
    std::fprintf(stderr, "mkstemp failed\n");
    return 1;
  }
  ::close(fd);
  const std::string path(tmpl);
  ::unlink(path.c_str());

  pid_t pid = ::fork();
  if (pid < 0) {
    std::fprintf(stderr, "fork failed\n");
    return 1;
  }
  if (pid == 0) {
    ::setenv("MLX_OMARCHY_STEP_TIMELINE", path.c_str(), 1);
    ::setenv("MLX_OMARCHY_STEP_TIMELINE_CAP", "4096", 1);
    std::exit(run_child_one_op());
  }

  int status = 0;
  ::waitpid(pid, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    ::unlink(path.c_str());
    std::fprintf(stderr, "child failed with status %d\n", status);
    return 1;
  }

  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string out = ss.str();
  ::unlink(path.c_str());

  if (out.empty()) {
    std::fprintf(stderr, "empty timeline output from one-op GPU child\n");
    return 1;
  }

  bool have_batch_open = false;
  uint64_t min_batch_open_t0 = UINT64_MAX;
  bool have_first_op = false;
  uint64_t min_first_op_t0 = UINT64_MAX;
  bool have_step_end = false;
  uint64_t max_step_end_t0 = 0;

  std::istringstream iss(out);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.rfind("{\"k\":\"e\",", 0) != 0) {
      continue;
    }
    const std::string kind = extract_str_field(line, "\"kind\":\"");
    const std::string sub = extract_str_field(line, "\"sub\":\"");
    uint64_t t0 = 0;
    if (!extract_u64_field(line, "\"t0\":", t0)) {
      std::fprintf(stderr, "event missing t0: %s\n", line.c_str());
      return 1;
    }
    if (kind == "step_begin" && sub == "batch_open") {
      have_batch_open = true;
      if (t0 < min_batch_open_t0) {
        min_batch_open_t0 = t0;
      }
    } else if (
        kind == "pipeline_lookup" || kind == "desc_update" ||
        kind == "cmd_begin" || kind == "cmd_record") {
      have_first_op = true;
      if (t0 < min_first_op_t0) {
        min_first_op_t0 = t0;
      }
    } else if (kind == "step_end") {
      have_step_end = true;
      if (t0 > max_step_end_t0) {
        max_step_end_t0 = t0;
      }
    }
  }

  if (!have_batch_open || !have_first_op || !have_step_end) {
    std::fprintf(
        stderr,
        "missing events: batch_open=%d first_op=%d step_end=%d\n",
        have_batch_open ? 1 : 0,
        have_first_op ? 1 : 0,
        have_step_end ? 1 : 0);
    return 1;
  }
  if (min_batch_open_t0 > min_first_op_t0) {
    std::fprintf(
        stderr,
        "ordering violation: batch_open t0=%llu > first op t0=%llu\n",
        static_cast<unsigned long long>(min_batch_open_t0),
        static_cast<unsigned long long>(min_first_op_t0));
    return 1;
  }
  if (min_batch_open_t0 > max_step_end_t0) {
    std::fprintf(
        stderr,
        "ordering violation: batch_open t0=%llu > step_end t0=%llu\n",
        static_cast<unsigned long long>(min_batch_open_t0),
        static_cast<unsigned long long>(max_step_end_t0));
    return 1;
  }

  std::printf(
      "test_step_timeline_eval: PASS (batch_open_t0=%llu <= first_op_t0=%llu <= step_end_t0=%llu)\n",
      static_cast<unsigned long long>(min_batch_open_t0),
      static_cast<unsigned long long>(min_first_op_t0),
      static_cast<unsigned long long>(max_step_end_t0));
  return 0;
}
