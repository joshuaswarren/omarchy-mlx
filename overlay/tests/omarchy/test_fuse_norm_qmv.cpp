// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// MLX_OMARCHY_FUSE_NORM_QMV: the decode RmsNorm folds into the tail of the
// Qmm group that produces its input (the group member's Add-epilogue sum):
// the group's last workgroup runs the norm. The fused result must be BYTE
// IDENTICAL to the separate norm dispatch and must save exactly one dispatch
// per fused norm. The flag is read once per process, so this binary turns it
// on before main and builds today's reference by evaluating the sum first (an
// evaluated leaf has no producer group to fold into).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/compile.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;
using mlx::core::omarchy::trace::counters;

namespace {

struct EnableFlag {
  EnableFlag() {
    setenv("MLX_OMARCHY_FUSE_NORM_QMV", "1", 1);
    setenv("MLX_OMARCHY_FUSED_CHAIN", "1", 1);
  }
} enable_flag;

void sync_stream(const Stream& stream) {
  omarchy::get_command_encoder(stream).synchronize();
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (!gpu::is_available()) {
    std::cout << "Skipping: no qualifying Vulkan device.\n";
    return false;
  }
  return true;
}

bool fold_available(const Stream& stream) {
  const auto& caps = omarchy::get_command_encoder(stream).device().capabilities();
  return caps.subgroup_size == 32u &&
      (caps.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0u;
}

struct Linear {
  array w = zeros({1});
  array scales = zeros({1});
  array biases = zeros({1});
};

Linear make_linear(int n, int k, Dtype dtype, const Stream& stream) {
  Linear linear;
  array w = multiply(
      random::normal(Shape{n, k}, float32, std::nullopt, stream),
      array(0.05f), stream);
  auto parts = quantize(w, 64, 4, "affine", std::nullopt, stream);
  linear.w = parts[0];
  linear.scales = astype(parts[1], dtype, stream);
  linear.biases = astype(parts[2], dtype, stream);
  for (array* a : {&linear.w, &linear.scales, &linear.biases}) {
    a->eval();
  }
  sync_stream(stream);
  return linear;
}

array project(const array& x, const Linear& l, const Stream& s) {
  return quantized_matmul(x, l.w, l.scales, l.biases, true, 64, 4, "affine", s);
}

enum class Data { normal, outliers, uniform };

// Row data that stresses the exact-sum contract: heavy outliers change the
// summation order's rounding, and the uniform row lands the mean square on
// exact bf16 ties.
array make_row(int rows, int k, Dtype dtype, Data kind, const Stream& stream) {
  array x = astype(
      random::normal(Shape{rows, 1, k}, float32, std::nullopt, stream), dtype,
      stream);
  if (kind == Data::outliers) {
    std::vector<float> host(static_cast<size_t>(rows) * k);
    array f = astype(x, float32, stream);
    f.eval();
    sync_stream(stream);
    for (size_t i = 0; i < host.size(); ++i) {
      host[i] = f.data<float>()[i] * ((i % 61 == 0) ? 900.0f : 0.37f);
    }
    host[17] = 3000.0f;
    host[k - 1] = -1e-30f;
    x = astype(array(host.data(), Shape{rows, 1, k}, float32), dtype, stream);
  } else if (kind == Data::uniform) {
    x = astype(
        where(
            greater(x, array(0.0f), stream), array(1.0f), array(-1.0f), stream),
        dtype, stream);
  }
  x.eval();
  sync_stream(stream);
  return x;
}

array make_weight(int k, Dtype dtype, const Stream& stream) {
  array w = astype(
      add(multiply(
              random::normal(Shape{k}, float32, std::nullopt, stream),
              array(0.3f), stream),
          array(1.0f), stream),
      dtype, stream);
  w.eval();
  sync_stream(stream);
  return w;
}

struct Outcome {
  std::vector<array> outs;
  uint64_t dispatches;
};

struct Case {
  // Producer group: Qmm over a 2048-wide row (members: the one whose sum
  // feeds the norm, plus an optional second member), N = norm width.
  int norm_width = 2048;
  Dtype dtype = bfloat16;
  Data kind = Data::normal;
  bool with_add = true;
  bool second_member = false;
  std::vector<int> consumers{2048};
  bool want_norm = false;
  bool want_out = false;
  int expected_saved = 1;
};

void expect_same_bytes(const array& a, const array& b, const Stream& stream) {
  array a32 = astype(a, float32, stream);
  array b32 = astype(b, float32, stream);
  a32.eval();
  b32.eval();
  sync_stream(stream);
  REQUIRE_EQ(a32.shape(), b32.shape());
  size_t bad = 0;
  for (size_t i = 0; i < a32.size(); ++i) {
    uint32_t ab, bb;
    std::memcpy(&ab, a32.data<float>() + i, 4);
    std::memcpy(&bb, b32.data<float>() + i, 4);
    if (ab != bb) {
      ++bad;
    }
  }
  CHECK_EQ(bad, 0);
}

struct Graph {
  array x_prev;
  array residual;
  array norm_weight;
  Linear producer;
  Linear other;
  std::vector<Linear> consumers;
};

// The residual-stream sum: Qmm(x_prev) + residual, one group member whose
// Add epilogue is the norm's input (without the Add the norm reads a raw Qmm
// output and nothing folds).
array make_sum(const Graph& g, const Case& c, const Stream& s) {
  array q = project(g.x_prev, g.producer, s);
  return c.with_add ? add(q, g.residual, s) : q;
}

// Today's path: the sum is evaluated first, so the norm node reads an
// evaluated leaf. Dispatches are summed over every eval.
Outcome run_reference(const Graph& g, const Case& c, const Stream& s) {
  uint64_t before = counters().vk_compute_dispatches.load();
  array sum = make_sum(g, c, s);
  std::vector<array> pre{sum};
  std::optional<array> other;
  if (c.second_member) {
    other = project(g.x_prev, g.other, s);
    pre.push_back(*other);
  }
  eval(pre);
  sync_stream(s);
  array y = fast::rms_norm(sum, g.norm_weight, 1e-6f, s);
  y.eval();
  sync_stream(s);
  std::vector<array> outs{sum, y};
  std::vector<array> consumers;
  for (const Linear& l : g.consumers) {
    consumers.push_back(project(y, l, s));
  }
  eval(consumers);
  sync_stream(s);
  outs.insert(outs.end(), consumers.begin(), consumers.end());
  if (other) {
    outs.push_back(*other);
  }
  return {outs, counters().vk_compute_dispatches.load() - before};
}

// Candidate: one eval over everything; the norm node is a tape descendant of
// the producer group's epilogue.
Outcome run_candidate(const Graph& g, const Case& c, const Stream& s) {
  array sum = make_sum(g, c, s);
  std::optional<array> other;
  if (c.second_member) {
    other = project(g.x_prev, g.other, s);
  }
  array y = fast::rms_norm(sum, g.norm_weight, 1e-6f, s);
  std::vector<array> consumers;
  for (const Linear& l : g.consumers) {
    consumers.push_back(project(y, l, s));
  }
  std::vector<array> roots = consumers;
  if (c.want_norm) {
    roots.push_back(y);
  }
  if (c.want_out) {
    roots.push_back(sum);
  }
  if (other) {
    roots.push_back(*other);
  }
  uint64_t before = counters().vk_compute_dispatches.load();
  eval(roots);
  sync_stream(s);
  uint64_t dispatches = counters().vk_compute_dispatches.load() - before;
  // Outputs not requested by the eval are computed here, after the count.
  std::vector<array> outs{sum, y};
  outs.insert(outs.end(), consumers.begin(), consumers.end());
  if (other) {
    outs.push_back(*other);
  }
  return {outs, dispatches};
}

void check_case(Case c) {
  Stream stream = gpu_stream();
  set_compile_mode(CompileMode::disabled);
  // The fold needs the 32-wide arithmetic subgroup column. Elsewhere (the
  // x86 software rasterizer reports 8 lanes) the norm stays its own
  // dispatch and only byte identity is asserted.
  if (!fold_available(stream)) {
    c.expected_saved = 0;
  }
  Graph g{
      make_row(1, 2048, c.dtype, c.kind, stream),
      make_row(1, c.norm_width, c.dtype, c.kind, stream),
      make_weight(c.norm_width, c.dtype, stream),
      make_linear(c.norm_width, 2048, c.dtype, stream),
      make_linear(512, 2048, c.dtype, stream),
      {}};
  for (int n : c.consumers) {
    g.consumers.push_back(make_linear(n, c.norm_width, c.dtype, stream));
  }
  Outcome ref = run_reference(g, c, stream);
  Outcome got = run_candidate(g, c, stream);
  CHECK_EQ(
      static_cast<int64_t>(ref.dispatches) -
          static_cast<int64_t>(got.dispatches),
      c.expected_saved);
  REQUIRE_EQ(ref.outs.size(), got.outs.size());
  for (size_t i = 0; i < ref.outs.size(); ++i) {
    INFO("output ", i);
    expect_same_bytes(ref.outs[i], got.outs[i], stream);
  }
}

} // namespace

TEST_CASE("fused norm tail: the producer group writes the norm, one dispatch saved") {
  if (!compute_available()) {
    return;
  }
  for (Data kind : {Data::normal, Data::outliers, Data::uniform}) {
    for (int n : {2048, 6144}) {
      INFO("n ", n, " data ", static_cast<int>(kind));
      Case c;
      c.kind = kind;
      c.consumers = {n};
      check_case(c);
    }
  }
}

TEST_CASE("fused norm tail: several consumers, a second group member, requested outputs") {
  if (!compute_available()) {
    return;
  }
  Case c;
  c.consumers = {4096, 512, 512};
  c.kind = Data::outliers;
  check_case(c);
  c = Case{};
  c.second_member = true;
  check_case(c);
  c = Case{};
  c.want_norm = true;
  c.want_out = true;
  c.kind = Data::outliers;
  check_case(c);
}

TEST_CASE("fused norm tail: unsupported shapes fall back with identical bytes") {
  if (!compute_available()) {
    return;
  }
  Case c;
  // Wider row: the non-Apple tree norm kernel runs, not covered by the fold.
  c.norm_width = 4096;
  c.expected_saved = 0;
  check_case(c);
  // Float16 stays on the per-node path.
  c = Case{};
  c.dtype = float16;
  c.expected_saved = 0;
  check_case(c);
  // The norm reads a raw Qmm output (no Add epilogue): nothing to fold into.
  c = Case{};
  c.with_add = false;
  c.expected_saved = 0;
  check_case(c);
}
