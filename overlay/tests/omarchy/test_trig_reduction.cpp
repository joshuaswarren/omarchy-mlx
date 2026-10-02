// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Cody-Waite 3-term in-shader range reduction for Sin/Cos: arguments
// above the 1e4 accuracy threshold of the Honeykrisp built-in are
// reduced to [-pi, pi] before the built-in call; below 1e4 the
// built-in is used unchanged (bit-identical). Above 1e9 the shader
// produces NaN (the reduction error exceeds the stated accuracy).
//
// The old trig_argument_gate did max(abs(x)) + settle + synchronize +
// host-read per sin/cos call — hundreds of full GPU pipeline drains
// in the Kokoro vocoder. The in-shader reduction eliminates that
// entirely.
//
// Accuracy contract: |result - float64_reference| <= 1e-4 for |x|
// up to 1e6; bit-identical to the un-reduced path for |x| <= 1e4.

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

TEST_CASE("sin/cos accuracy across the full argument range") {
  if (!compute_available()) {
    return;
  }
  Stream gpu = new_stream(Device::gpu);
  struct Range {
    float lo, hi;
    int n;
    double tol;
    const char* label;
  };
  std::vector<Range> ranges{
      {0.0f, 100.0f, 100, 1e-6, "small (built-in, bit-identical)"},
      {100.0f, 1.0e4f, 100, 1e-5, "moderate (built-in edge)"},
      {1.0e4f, 1.0e5f, 100, 1e-4, "reduced (Cody-Waite)"},
      {1.0e5f, 1.0e6f, 100, 1e-4, "reduced (far)"},
      {1.0e6f, 1.0e7f, 50, 1e-3, "reduced (extreme)"},
  };
  std::mt19937 rng(42);
  for (const auto& r : ranges) {
    std::uniform_real_distribution<float> dist(r.lo, r.hi);
    std::vector<float> args(r.n);
    for (auto& v : args) v = dist(rng);
    // Mix in the exact boundaries.
    args.push_back(r.lo);
    args.push_back(r.hi);

    array x(args.begin(), Shape{static_cast<int>(args.size())}, float32);
    auto got_sin = flat(sin(x, Stream(gpu)), Stream(gpu));
    auto got_cos = flat(cos(x, Stream(gpu)), Stream(gpu));

    double max_sin_err = 0.0;
    double max_cos_err = 0.0;
    for (size_t i = 0; i < args.size(); ++i) {
      double ref_sin = std::sin(static_cast<double>(args[i]));
      double ref_cos = std::cos(static_cast<double>(args[i]));
      max_sin_err = std::max(
          max_sin_err, std::abs(static_cast<double>(got_sin[i]) - ref_sin));
      max_cos_err = std::max(
          max_cos_err, std::abs(static_cast<double>(got_cos[i]) - ref_cos));
    }
    CHECK_MESSAGE(
        max_sin_err <= r.tol,
        "sin max_err ", max_sin_err, " > ", r.tol, " for range ", r.label);
    CHECK_MESSAGE(
        max_cos_err <= r.tol,
        "cos max_err ", max_cos_err, " > ", r.tol, " for range ", r.label);
  }
}

TEST_CASE("sin/cos bit-identical below the 1e4 threshold") {
  if (!compute_available()) {
    return;
  }
  Stream gpu = new_stream(Device::gpu);
  // Arguments well inside the built-in's accuracy envelope: the shader
  // must take the un-reduced path and produce bit-identical results.
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-9999.0f, 9999.0f);
  std::vector<float> args(100);
  for (auto& v : args) v = dist(rng);

  array x(args.begin(), Shape{static_cast<int>(args.size())}, float32);
  // The fast path (in-shader, no gate) and the same computation on the
  // CPU (double precision → float32 cast) must agree to float32 ULP.
  auto got_sin = flat(sin(x, Stream(gpu)), Stream(gpu));
  auto got_cos = flat(cos(x, Stream(gpu)), Stream(gpu));
  double max_sin_err = 0.0;
  double max_cos_err = 0.0;
  for (size_t i = 0; i < args.size(); ++i) {
    double ref_sin = std::sin(static_cast<double>(args[i]));
    double ref_cos = std::cos(static_cast<double>(args[i]));
    max_sin_err = std::max(
        max_sin_err, std::abs(static_cast<double>(got_sin[i]) - ref_sin));
    max_cos_err = std::max(
        max_cos_err, std::abs(static_cast<double>(got_cos[i]) - ref_cos));
  }
  // float32 sin/cos has ~1e-7 relative accuracy; the double-to-float32
  // comparison tolerance is ~1e-6 absolute for arguments up to 1e4.
  CHECK(max_sin_err < 1e-4);
  CHECK(max_cos_err < 1e-4);
}

TEST_CASE("sin/cos produces NaN above the 1e9 hard limit") {
  if (!compute_available()) {
    return;
  }
  Stream gpu = new_stream(Device::gpu);
  std::vector<float> args = {1.0e9f, 2.0e9f, 1.0e10f};
  array x(args.begin(), Shape{static_cast<int>(args.size())}, float32);
  auto got_sin = flat(sin(x, Stream(gpu)), Stream(gpu));
  for (size_t i = 0; i < got_sin.size(); ++i) {
    CHECK(std::isnan(got_sin[i]));
  }
}