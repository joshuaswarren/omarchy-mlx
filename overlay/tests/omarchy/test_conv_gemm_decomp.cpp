// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// 1-D conv k-tap GEMM decomposition: rank-1 groups=1 unit-stride
// unit-dilation fp32 convolutions route through k shifted matmuls instead
// of the direct per-output-element conv kernel (22-121 ms -> 1.7-9.3 ms on
// the Kokoro vocoder shapes, receipts/2026-10-01-opcost-microbench). The
// numeric contract: same results as the CPU reference within float32
// accumulation tolerance, including edge rows where fewer taps contribute.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

bool compute_available() {
  if (!gpu::is_available()) {
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return false;
  }
  return true;
}

std::vector<float> flat(const array& value, Stream stream) {
  array copy = astype(value, float32, stream);
  copy.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = copy.data<float>();
  return std::vector<float>(data, data + copy.size());
}

} // namespace

TEST_CASE("conv1d gemm decomposition matches cpu reference") {
  if (!compute_available()) {
    return;
  }
  Stream gpu = new_stream(Device::gpu);
  Stream cpu = new_stream(Device::cpu);
  struct Case {
    int L, ci, co, k, pad;
  };
  std::vector<Case> cases{
      {64, 8, 16, 7, 3},   // typical vocoder block
      {37, 5, 9, 3, 1},    // odd extents
      {5, 4, 4, 7, 3},     // L < k: several taps fully clipped
      {129, 128, 128, 11, 5},
  };
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (const auto& c : cases) {
    std::vector<float> xd(c.L * c.ci);
    std::vector<float> wd(c.co * c.k * c.ci);
    for (auto& v : xd) v = dist(rng);
    for (auto& v : wd) v = 0.1f * dist(rng);

    array x(xd.begin(), Shape{1, c.L, c.ci}, float32);
    array w(wd.begin(), Shape{c.co, c.k, c.ci}, float32);

    array got = conv1d(x, w, /*stride=*/1, /*padding=*/c.pad, Stream(gpu));
    array want = conv1d(x, w, /*stride=*/1, /*padding=*/c.pad, Stream(cpu));
    auto got_v = flat(got, Stream(gpu));
    auto want_v = flat(want, Stream(cpu));
    CHECK_EQ(got_v.size(), want_v.size());
    double max_err = 0.0;
    for (size_t i = 0; i < want_v.size() && i < got_v.size(); ++i) {
      max_err = std::max(
          max_err,
          static_cast<double>(std::abs(got_v[i] - want_v[i])));
    }
    CHECK_MESSAGE(
        max_err < 1e-3,
        "conv1d gemm decomposition max_err ",
        max_err,
        " for L=",
        c.L,
        " ci=",
        c.ci,
        " co=",
        c.co,
        " k=",
        c.k,
        " pad=",
        c.pad);
  }
}

TEST_CASE("conv1d gemm decomposition agrees with direct gpu path") {
  if (!compute_available()) {
    return;
  }
  Stream gpu = new_stream(Device::gpu);
  // batch >= 2 bypasses the decomposition (batch==1 guard) and rides the
  // general conv.comp kernel: an independent in-backend reference.
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  int L = 96, ci = 16, co = 16, k = 7, pad = 3;
  std::vector<float> xd(2 * L * ci);
  std::vector<float> wd(co * k * ci);
  for (auto& v : xd) v = dist(rng);
  for (auto& v : wd) v = 0.1f * dist(rng);
  array x(xd.begin(), Shape{2, L, ci}, float32);
  array w(wd.begin(), Shape{co, k, ci}, float32);
  // Same batch-0 rows through both paths.
  auto x1 = slice(x, {0, 0, 0}, {1, L, ci}, Stream(gpu));
  array fast = conv1d(x1, w, 1, pad, Stream(gpu));
  array ref = slice(
      conv1d(x, w, 1, pad, Stream(gpu)), {0, 0, 0}, {1, L, co}, Stream(gpu));
  auto got = flat(fast, Stream(gpu));
  auto want = flat(ref, Stream(gpu));
  double max_err = 0.0;
  for (size_t i = 0; i < want.size() && i < got.size(); ++i) {
    max_err = std::max(
        max_err, static_cast<double>(std::abs(got[i] - want[i])));
  }
  CHECK(max_err < 1e-3);
}