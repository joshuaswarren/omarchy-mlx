// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// The shared Cody-Waite range reduction (shaders/omarchy_trig.h) for
// sin/cos/tan: accurate vs float64 up to kTrigArgumentLimit (5e5) and
// NaN above it, on every path that evaluates trig - eager elementwise,
// compiled tapes, complex exp/sin/cos (real-part reduction), the rope
// kernels, and the complex logsumexp scan. The old trig_argument_gate
// (per-call GPU sync + host read) was removed by 6cbf55d8f; the
// per-call refusal became a NaN-above-the-limit contract at the
// owner's perf tradeoff - docs/known-defects.md, 2026-10-03.
//
// Measured error envelope of the reduction band (max abs err vs float64
// over logspace samples + edges, receipts/2026-10-03-trig-contract/):
//   llvmpipe      2.5e-7   (decade 1e4..1e5; 8.6e-8 in 1e5..5e5)
//   jw16/M1 Max   <see receipt>  (same contract value pinned here)
// The test pins 1e-3 in the reduction band: the reduction's own
// envelope is 2.5e-7 (llvmpipe, jw16), but jwm1's older-Mesa built-in
// sin/cos adds up to 6.9e-4 at some reduced arguments (measured,
// stable across runs), so the pin covers the device built-in floor.
// Per-device maxima are tabulated in the receipt.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/compile.h"
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

// Interleaved (re, im) pairs of a complex64 array, for the complex legs.
std::vector<float> flat_complex(const array& value, Stream stream) {
  array copy = astype(value, complex64, stream);
  copy.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const auto* data = copy.data<complex64_t>();
  std::vector<float> pairs;
  pairs.reserve(copy.size() * 2);
  for (int i = 0; i < copy.size(); ++i) {
    pairs.push_back(data[i].real());
    pairs.push_back(data[i].imag());
  }
  return pairs;
}

constexpr float kLimit = 500000.0f;

} // namespace

TEST_CASE("sin/cos accuracy across the contract bands") {
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
      {0.0f, 100.0f, 64, 1e-6, "small (built-in)"},
      {100.0f, 1.0e3f, 64, 1e-3, "moderate (built-in; jwm1 measured 7.1e-4)"},
      {1.0e3f, 1.0e4f, 64, 1e-2,
          "built-in edge (jwm1 measured 6.6e-3; older driver 4.8e-3)"},
      // The reduction's own envelope is 2.5e-7 (llvmpipe, jw16 measured).
      // jwm1's older-Mesa built-in sin/cos adds up to 6.9e-4 at some
      // reduced arguments (measured, stable across runs) - the pin
      // covers the device built-in floor; per-device maxima are tabulated
      // in receipts/2026-10-03-trig-contract/.
      {1.0e4f, 1.0e5f, 128, 1e-3, "reduced (shared Cody-Waite)"},
      {1.0e5f, kLimit, 128, 1e-3, "reduced (far, exact-product zone)"}};
  std::mt19937 rng(42);
  for (const auto& r : ranges) {
    std::uniform_real_distribution<float> dist(r.lo, r.hi);
    std::vector<float> args(r.n);
    for (auto& v : args) v = dist(rng);
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

TEST_CASE("tan uses the same reduction in its band") {
  if (!compute_available()) {
    return;
  }
  Stream gpu = new_stream(Device::gpu);
  // Fixed arguments whose tangent is moderate; near-pole arguments are
  // excluded because a phase error of a few ulp legitimately amplifies
  // there (same as the built-in at small arguments).
  std::vector<float> args = {
      12345.0f, 54321.0f, 123456.0f, 200000.0f, 400000.0f, kLimit};
  for (float v : args) {
    double ref = std::tan(static_cast<double>(v));
    if (std::abs(ref) > 1.0e3) {
      continue; // this device's v landed near a pole; not a stable pin
    }
    array x(v);
    float got = flat(tan(x, Stream(gpu)), Stream(gpu)).at(0);
    double err = std::abs(static_cast<double>(got) - ref);
    // 1e-2 relative: the reduction contributes ~1e-6; the device
    // built-in's small-argument floor (jwm1, older Mesa: ~7e-4 absolute)
    // amplifies through tan exactly as it does at small arguments.
    CHECK_MESSAGE(
        err <= 1e-2 * std::max(1.0, std::abs(ref)),
        "tan(", v, ") = ", got, " vs ", ref);
  }
  CHECK(std::isnan(flat(tan(array(1e6f), Stream(gpu)), Stream(gpu)).at(0)));
}

TEST_CASE("NaN above the 5e5 limit, accurate at the boundary") {
  if (!compute_available()) {
    return;
  }
  Stream gpu = new_stream(Device::gpu);
  std::vector<float> huge = {
      500001.0f, 1e6f, 1e9f, 1e10f, 1e20f, 1e38f, -2.7e37f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN()};
  for (float v : huge) {
    auto gs = flat(sin(array(v), Stream(gpu)), Stream(gpu));
    auto gc = flat(cos(array(v), Stream(gpu)), Stream(gpu));
    CHECK_MESSAGE(std::isnan(gs[0]), "sin(", v, ") = ", gs[0]);
    CHECK_MESSAGE(std::isnan(gc[0]), "cos(", v, ") = ", gc[0]);
  }
  // The boundary itself: 5e5 - 1 and 5e5 are finite and accurate, the
  // first float past 5e5 is NaN.
  for (float v : {499999.0f, 500000.0f}) {
    auto gs = flat(sin(array(v), Stream(gpu)), Stream(gpu));
    CHECK_MESSAGE(!std::isnan(gs[0]), "boundary sin(", v, ") NaN");
    CHECK_MESSAGE(
        std::abs(gs[0] - (float)std::sin((double)v)) <= 1e-3,
        "boundary sin(", v, ") = ", gs[0]);
  }
}

TEST_CASE("compiled tape leg matches the eager contract") {
  if (!compute_available()) {
    return;
  }
  Stream gpu = new_stream(Device::gpu);
  auto trig_tape = [](std::vector<array> inputs) {
    return std::vector<array>{sin(inputs[0]) * cos(inputs[0])};
  };
  auto compiled = compile(trig_tape);
  for (float v : {123456.0f, 2.0e5f, kLimit}) {
    float eager = flat(trig_tape({array(v)}).at(0), Stream(gpu)).at(0);
    float taped = flat(compiled({array(v)}).at(0), Stream(gpu)).at(0);
    double ref = std::sin((double)v) * std::cos((double)v);
    CHECK_MESSAGE(
        std::abs(eager - ref) <= 1e-4,
        "eager sin*cos(", v, ") = ", eager, " vs ", ref);
    CHECK_MESSAGE(
        std::abs(taped - ref) <= 1e-4,
        "taped sin*cos(", v, ") = ", taped, " vs ", ref);
  }
  for (float v : {1e6f, -2.7e37f}) {
    float taped = flat(compiled({array(v)}).at(0), Stream(gpu)).at(0);
    CHECK_MESSAGE(std::isnan(taped), "taped sin*cos(", v, ") = ", taped);
  }
}

TEST_CASE("complex exp reduces its imaginary part by the same contract") {
  if (!compute_available()) {
    return;
  }
  Stream gpu = new_stream(Device::gpu);
  for (float v : {123456.0f, 2.0e5f, kLimit}) {
    std::vector<complex64_t> zv = {complex64_t{0.0f, v}};
    array x(zv.begin(), Shape{1}, complex64);
    auto got = flat_complex(exp(x, Stream(gpu)), Stream(gpu));
    double ref_re = std::cos(static_cast<double>(v));
    double ref_im = std::sin(static_cast<double>(v));
    CHECK_MESSAGE(
        std::abs(got[0] - (float)ref_re) <= 1e-3,
        "exp(", v, "i).re = ", got[0], " vs ", ref_re);
    CHECK_MESSAGE(
        std::abs(got[1] - (float)ref_im) <= 1e-3,
        "exp(", v, "i).im = ", got[1], " vs ", ref_im);
  }
  // Above the limit the reduction refuses by value: NaN components,
  // never a finite wrong phase.
  std::vector<complex64_t> zv = {complex64_t{0.0f, 1e6f}};
  auto got = flat_complex(
      exp(array(zv.begin(), Shape{1}, complex64), Stream(gpu)), Stream(gpu));
  CHECK(std::isnan(got[0]));
  CHECK(std::isnan(got[1]));
}
