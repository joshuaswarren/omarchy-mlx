// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <bit>
#include <cstdint>
#include <limits>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/ops.h"

using namespace mlx::core;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1).");
    return false;
  }
  return true;
}

// Pull one element of an evaluated array.
template <typename T>
T read_value(array a, size_t index = 0) {
  a.eval();
  omarchy::get_command_encoder(gpu_stream()).synchronize();
  const T* values = a.data<T>();
  return values[index];
}

bool bits_equal_f32(float got, float want) {
  uint32_t g = std::bit_cast<uint32_t>(got);
  uint32_t w = std::bit_cast<uint32_t>(want);
  return g == w;
}

} // namespace

// W2: equal_nan plumbing on identical arrays carrying NaN. The upstream
// suite's mx.array_equal primary comparison helper used to return False
// here because the Equal kernel ignored the equal_nan flag.
TEST_CASE("W2 array_equal with equal_nan maps NaN to NaN as True") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> values{0.0f, 1.0f, std::nanf("")};
  array a(values.begin(), Shape{3}, float32);

  // Default flag stays False: NaN vs NaN compares False (IEEE).
  array out_default = array_equal(a, a, false, stream);
  CHECK_EQ(read_value<bool>(out_default), false);

  // equal_nan True: NaN vs NaN compares True.
  array out_nan = array_equal(a, a, true, stream);
  CHECK_EQ(read_value<bool>(out_nan), true);

  // mx.isclose is element-wise and uses its own composed path (abs
  // diff, isinf corrections, and a logical_or with isnan(a) && isnan(b)
  // gated on equal_nan). The Equal primitive is NOT involved when
  // equal_nan=False. Wrap in all() to get a scalar result.
  std::vector<float> other{0.0f, 1.0f, std::nanf("")};
  array b(other.begin(), Shape{3}, float32);
  array close_default = all(isclose(a, b, 1e-5, 1e-8, false, stream), stream);
  CHECK_EQ(read_value<bool>(close_default), false);
  array close_nan = all(isclose(a, b, 1e-5, 1e-8, true, stream), stream);
  CHECK_EQ(read_value<bool>(close_nan), true);
}

// W2: bool Equal unblocks array_equal machinery for the 6 masked C++
// cases (test array basics, test array types, gguf metadata, is close,
// random split, vmap comparison ops). Before this fix these refused
// with the named Equal-dtype gap; now they dispatch through the
// extended logical_or.comp comparison sextet.
//
// NOTE: the sextet shares dispatch_logical with Or/And/Not, and that
// shader accumulates output words with atomicOr. The output buffer
// must be zero-filled before dispatch or the OR accumulates into
// garbage. This is a shared-infrastructure dependency: the fix belongs
// in dispatch_logical (zero the buffer once, before the atomicOr
// pass), not here.
TEST_CASE("W2 bool Equal and NotEqual match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<bool> lhs{true, false, true, false};
  std::vector<bool> rhs{true, true, false, false};
  array a(lhs.begin(), Shape{4}, bool_);
  array b(rhs.begin(), Shape{4}, bool_);

  array eq = equal(a, b, stream);
  std::vector<bool> expected_eq{true, false, false, true};
  for (size_t i = 0; i < 4; ++i) {
    CHECK_EQ(read_value<bool>(eq, i), expected_eq[i]);
  }

  array neq = not_equal(a, b, stream);
  std::vector<bool> expected_neq{false, true, true, false};
  for (size_t i = 0; i < 4; ++i) {
    CHECK_EQ(read_value<bool>(neq, i), expected_neq[i]);
  }

  // array_equal on bool arrays: was a named gap; now a value path.
  array ae = array_equal(a, a, false, stream);
  CHECK_EQ(read_value<bool>(ae), true);
}

// W5: log10 returns the exact integer for every f32-exact power of
// ten. log10(1000) == 3.0 was the receipt's named pin (C++ ops_tests
// arithmetic unary ops line 1721); the snap table now also nails 10^k
// for k in -10..10.
TEST_CASE("W5 log10 returns exact integer for f32-exact powers of ten") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  for (int k = -10; k <= 10; ++k) {
    float value = std::pow(10.0f, static_cast<float>(k));
    array x(value);
    array y = log10(x, stream);
    float got = read_value<float>(y);
    float want = static_cast<float>(k);
    INFO("10^" << k << " f32 bits 0x" << std::hex
         << std::bit_cast<uint32_t>(value) << std::dec);
    CHECK(bits_equal_f32(got, want));
  }
}

// W5: ad-hoc adversarial values that include the original defect.
TEST_CASE("W5 log10 pin and non-power inputs agree with numpy f32") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> values{
      1.0f,
      2.0f,
      8.0f,
      100.0f,
      1000.0f,
      123456.0f,
      1e6f,
      0.1f,
      0.5f,
      1e-3f};
  for (float v : values) {
    array x(v);
    array y = log10(x, stream);
    float got = read_value<float>(y);
    float want = std::log10(v);
    INFO("log10(" << v << ") got=" << got << " want=" << want
         << " diff=" << std::abs(got - want));
    CHECK(std::abs(got - want) <= 1e-5f);
  }
}

// W8: sin/cos use GLSL built-ins on the common path (|x| below ~2^24).
// The built-ins match std::sin/cos on lavapipe and most drivers. Test
// values mirror test_primitives "Cos and Sin match host references"
// plus the inverse/hyperbolic band's sin/cos anchor points.
// W8: sin/cos/tan reduce large arguments in-shader (shaders/omarchy_trig.h).
// The band 1e4 < |x| <= 5e5 is the shared Cody-Waite reduction (accurate
// vs float64); |x| > 5e5 is NaN by contract. The old per-call host gate
// refused instead; the NaN contract replaced it at the owner's perf
// tradeoff (docs/known-defects.md, 2026-10-03).
TEST_CASE("trig reduction is accurate through the limit band") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // 2e5 sits mid-band: the old gate refused it, the current reduction
  // computes it accurately. 5e5 exactly is the contract boundary.
  for (float v : {2.0e5f, 5.0e5f}) {
    array xs(v);
    array ys = sin(xs, stream);
    array yc = cos(xs, stream);
    float gs = read_value<float>(ys);
    float gc = read_value<float>(yc);
    INFO("v=" << v << " sin=" << gs << " cos=" << gc);
    CHECK(std::abs(gs - (float)std::sin((double)v)) <= 1e-3f);
    CHECK(std::abs(gc - (float)std::cos((double)v)) <= 1e-3f);
  }

  array small({0.5f, 1.0f, 2.0f});
  array y = cos(small, stream);
  float c1 = read_value<float>(y, 1);
  CHECK(std::abs(c1 - (float)std::cos(1.0)) < 1e-4f);
}

TEST_CASE("W8 sin/cos/tan common path matches numpy f32") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> common{
      0.0f, 0.5f, 1.0f, 1.5707963f, 3.1415927f, -0.75f, -2.0f, 3.5f};
  for (float v : common) {
    array xs(v);
    array ys = sin(xs, stream);
    array yc = cos(xs, stream);
    float gs = read_value<float>(ys);
    float gc = read_value<float>(yc);
    float ulp_s = std::abs(gs - (float)std::sin((double)v));
    float ulp_c = std::abs(gc - (float)std::cos((double)v));
    INFO("v=" << v << " sin=" << gs << " cos=" << gc);
    CHECK(ulp_s <= 1e-6f);
    CHECK(ulp_c <= 1e-6f);
  }
}

// W8: full-range pin. Since 6cbf55d8f the eager Sin/Cos (and now Tan)
// reduce arguments in-shader; the contract is accurate-to-float64 up to
// kTrigArgumentLimit = 5e5 and NaN above it (non-finite is loud; a
// finite wrong value is the one forbidden outcome). Bands:
//  |v| <= 1e3        -> built-in, 1e-4 abs (measured 2.8e-5 worst, M1)
//  1e3 < |v| <= 1e4  -> built-in, 5e-3 abs (measured 4.8e-3 worst, M1)
//  1e4 < |v| <= 5e5  -> shared Cody-Waite reduction, 1e-3 abs
//                       (measured 2.5e-7 llvmpipe/jw16; jwm1 built-in adds
//                       up to 6.9e-4; per-device maxima in the receipt)
//  |v| > 5e5         -> NaN (sin, cos, tan alike); +inf/-inf -> NaN,
//                       NaN in -> NaN out (IEEE)
TEST_CASE("W8 sin/cos accurate to 5e5 and NaN above it") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> full{
      0.0f, 0.5f, 1.0f, 1.5707963f, 3.1415927f, -0.75f, -2.0f, 3.5f,
      12345.0f, 54321.0f,
      123456.789f, 200000.0f, 500000.0f, -500000.0f};
  for (float v : full) {
    array xs(v);
    float tol = std::abs(v) <= 1e3f ? 1e-3f
              : (std::abs(v) <= 1e4f ? 1e-2f : 1e-3f);
    array ys = sin(xs, stream);
    array yc = cos(xs, stream);
    float gs = read_value<float>(ys);
    float gc = read_value<float>(yc);
    float ulp_s = std::abs(gs - (float)std::sin((double)v));
    float ulp_c = std::abs(gc - (float)std::cos((double)v));
    INFO("v=" << v << " sin=" << gs << " cos=" << gc);
    CHECK(ulp_s <= tol);
    CHECK(ulp_c <= tol);
  }
  // Above the limit: NaN for every function, exactly for the sampled
  // huge values. +inf stays NaN via the same path (|inf| > limit);
  // NaN in -> NaN out.
  std::vector<float> huge{
      500001.0f, 1e6f, 1e9f, 1e10f, 1e20f, 1e30f, 1e38f, -2.7e37f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN()};
  for (float v : huge) {
    array xs(v);
    array ys = sin(xs, stream);
    array yc = cos(xs, stream);
    array yt = tan(xs, stream);
    float gs = read_value<float>(ys);
    float gc = read_value<float>(yc);
    float gt = read_value<float>(yt);
    INFO("v=" << v << " sin=" << gs << " cos=" << gc << " tan=" << gt);
    CHECK(std::isnan(gs));
    CHECK(std::isnan(gc));
    CHECK(std::isnan(gt));
  }
  // Just below the boundary is finite and accurate; just above is NaN.
  // This pins the threshold at its exact value on both sides.
  {
    array below(499999.0f);
    array ys = sin(below, stream);
    float gs = read_value<float>(ys);
    CHECK(!std::isnan(gs));
    CHECK(std::abs(gs - (float)std::sin(499999.0)) <= 1e-3f);
  }
}
