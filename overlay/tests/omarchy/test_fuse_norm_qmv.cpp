// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// MLX_OMARCHY_FUSE_NORM_QMV: the decode RmsNorm folds into the prologue of
// the next single-row Qmm group. The fused result must be BYTE IDENTICAL to
// fast::rms_norm followed by quantized_matmul, and must save exactly one
// dispatch per fused norm. The flag is read once per process, so this binary
// turns it on before main and builds today's two-kernel reference by
// evaluating the norm output first (an evaluated leaf has no norm node to
// fold).

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

// Today's path: the norm output is evaluated first, so the Qmm group sees an
// evaluated leaf. Dispatches are summed over both evals.
Outcome run_reference(
    const array& x, const array& weight, float eps,
    const std::vector<Linear>& linears, const Stream& stream) {
  uint64_t before = counters().vk_compute_dispatches.load();
  array y = fast::rms_norm(x, weight, eps, stream);
  y.eval();
  sync_stream(stream);
  std::vector<array> outs{y};
  std::vector<array> consumers;
  for (const Linear& l : linears) {
    consumers.push_back(project(y, l, stream));
  }
  eval(consumers);
  sync_stream(stream);
  for (array& c : consumers) {
    outs.push_back(c);
  }
  return {outs, counters().vk_compute_dispatches.load() - before};
}

// Candidate: one eval over the consumers, the norm node is a tape ancestor.
// `want_norm` also requests the norm output, which the fused dispatch must
// materialize.
Outcome run_candidate(
    const array& x, const array& weight, float eps,
    const std::vector<Linear>& linears, bool want_norm, const Stream& stream) {
  array y = fast::rms_norm(x, weight, eps, stream);
  std::vector<array> consumers;
  for (const Linear& l : linears) {
    consumers.push_back(project(y, l, stream));
  }
  std::vector<array> roots = consumers;
  if (want_norm) {
    roots.push_back(y);
  }
  uint64_t before = counters().vk_compute_dispatches.load();
  eval(roots);
  sync_stream(stream);
  uint64_t dispatches = counters().vk_compute_dispatches.load() - before;
  std::vector<array> outs{y};
  for (array& c : consumers) {
    outs.push_back(c);
  }
  return {outs, dispatches};
}

void check_case(
    int rows, int k, const std::vector<int>& ns, Dtype dtype, Data kind,
    bool want_norm, int expected_saved) {
  Stream stream = gpu_stream();
  // The fold needs the 32-wide arithmetic subgroup column. Elsewhere (the
  // x86 software rasterizer reports 8 lanes) the norm stays its own
  // dispatch and only byte identity is asserted.
  if (!fold_available(stream)) {
    expected_saved = 0;
  }
  set_compile_mode(CompileMode::disabled);
  array x = make_row(rows, k, dtype, kind, stream);
  array weight = make_weight(k, dtype, stream);
  std::vector<Linear> linears;
  for (int n : ns) {
    linears.push_back(make_linear(n, k, dtype, stream));
  }
  const float eps = 1e-6f;
  Outcome ref = run_reference(x, weight, eps, linears, stream);
  Outcome got = run_candidate(x, weight, eps, linears, want_norm, stream);
  CHECK_EQ(
      static_cast<int64_t>(ref.dispatches) -
          static_cast<int64_t>(got.dispatches),
      expected_saved);
  for (size_t i = 1; i < ref.outs.size(); ++i) {
    INFO("consumer ", i);
    expect_same_bytes(ref.outs[i], got.outs[i], stream);
  }
  if (want_norm) {
    INFO("norm output");
    expect_same_bytes(ref.outs[0], got.outs[0], stream);
  }
}

} // namespace

TEST_CASE("fused norm+qmv: one consumer saves one dispatch, bytes identical") {
  if (!compute_available()) {
    return;
  }
  for (Data kind : {Data::normal, Data::outliers, Data::uniform}) {
    for (int n : {2048, 6144}) {
      INFO("n ", n, " data ", static_cast<int>(kind));
      check_case(1, 2048, {n}, bfloat16, kind, false, 1);
    }
  }
}

TEST_CASE("fused norm+qmv: q/k/v consumers share one fused group dispatch") {
  if (!compute_available()) {
    return;
  }
  check_case(1, 2048, {4096, 512}, bfloat16, Data::normal, false, 1);
  check_case(1, 2048, {4096, 512, 512}, bfloat16, Data::outliers, false, 1);
}

TEST_CASE("fused norm+qmv: a requested norm output is materialized") {
  if (!compute_available()) {
    return;
  }
  check_case(1, 2048, {2048}, bfloat16, Data::outliers, true, 1);
  check_case(1, 2048, {2048, 6144}, bfloat16, Data::normal, true, 1);
}

TEST_CASE("fused norm+qmv: unsupported shapes fall back with identical bytes") {
  if (!compute_available()) {
    return;
  }
  // Wider row: the non-Apple tree norm kernel runs, not covered by the fold.
  check_case(1, 4096, {2048}, bfloat16, Data::normal, false, 0);
  // Float16 stays on the per-node path.
  check_case(1, 2048, {2048}, float16, Data::normal, false, 0);
  // More than one row never takes the single-row group.
  check_case(2, 2048, {2048}, bfloat16, Data::normal, false, 0);
}

TEST_CASE("fused norm+qmv: a non-Qmm consumer of the norm output blocks the fold") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  set_compile_mode(CompileMode::disabled);
  array x = make_row(1, 2048, bfloat16, Data::normal, stream);
  array weight = make_weight(2048, bfloat16, stream);
  Linear l = make_linear(2048, 2048, bfloat16, stream);
  auto build = [&](bool leaf) {
    array y = fast::rms_norm(x, weight, 1e-6f, stream);
    if (leaf) {
      y.eval();
      sync_stream(stream);
    }
    array out = project(y, l, stream);
    array side = multiply(y, array(2.0f, bfloat16), stream);
    return std::vector<array>{out, side};
  };
  uint64_t before = counters().vk_compute_dispatches.load();
  auto ref = build(true);
  eval(ref);
  sync_stream(stream);
  uint64_t ref_dispatches = counters().vk_compute_dispatches.load() - before;
  before = counters().vk_compute_dispatches.load();
  auto got = build(false);
  eval(got);
  sync_stream(stream);
  uint64_t dispatches = counters().vk_compute_dispatches.load() - before;
  CHECK_EQ(dispatches, ref_dispatches);
  expect_same_bytes(ref[0], got[0], stream);
  expect_same_bytes(ref[1], got[1], stream);
}
