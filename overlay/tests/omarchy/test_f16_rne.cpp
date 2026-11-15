// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Pins omarchy_f16_bits_rne (shaders/omarchy_half.h), the integer
// float32 -> float16 round-to-nearest-even conversion shared by
// fused_chain, swiglu, fast_trio, qmm_vec, qmm_fma, sdpa_decode_native_p1
// and cast_int, and the f16 rsqrt operand in elementwise.comp.
//
// The header is plain GLSL arithmetic on uint/float, so the CPU cases
// compile the shipped header itself as C++ and compare its bits with
// astype(float16) on the CPU stream and with fixed expected bits: the
// largest finite half, the overflow threshold, the smallest normal, the
// subnormal unit and its half (a tie that goes to zero), ties in the
// normal and subnormal ranges and across the subnormal/normal boundary,
// quiet NaNs with and without payload, and signed zero. A strided sweep
// over the float32 bit space covers the rest. Against the CPU stream every
// non-NaN input must give the same bits and every NaN input a NaN of the
// same sign: the half NaN payload the CPU conversion writes is
// implementation-defined (0x7D00 on aarch64 MLX, 0x7E00 with F16C), so
// the quiet-NaN bits are pinned by the fixed table only.
//
// The GPU cases run the same inputs through a strided float32 -> float16
// astype, which dispatches cast_int.comp (the dtype-converting general
// copy), and the f16 rsqrt route. They skip without a Vulkan device. On
// the GPU a NaN input must give a quiet half NaN; the GPU does not carry
// float32 NaN sign or payload bits through to the conversion.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

namespace omarchy_glsl {
// The GLSL built-ins the header uses, with their GLSL semantics.
using uint = std::uint32_t;
struct vec2 {
  float x;
  float y;
};
inline uint floatBitsToUint(float value) {
  uint bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
inline float half_bits_to_float(uint h) {
  uint sign = (h & 0x8000u) << 16u;
  uint exponent = (h >> 10u) & 0x1Fu;
  uint mantissa = h & 0x3FFu;
  float magnitude;
  if (exponent == 0u) {
    magnitude = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 31u) {
    magnitude = mantissa ? NAN : INFINITY;
  } else {
    magnitude = std::ldexp(static_cast<float>(mantissa | 0x400u),
                           static_cast<int>(exponent) - 25);
  }
  uint bits = floatBitsToUint(magnitude) | sign;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}
inline vec2 unpackHalf2x16(uint packed) {
  return {half_bits_to_float(packed & 0xFFFFu),
          half_bits_to_float(packed >> 16u)};
}
#include "mlx/backend/omarchy/shaders/omarchy_half.h"
} // namespace omarchy_glsl

using namespace mlx::core;

namespace {

struct Case {
  uint32_t f32_bits;
  uint16_t f16_bits;
  const char* name;
};

// Expected bits are IEEE 754 binary16 round-to-nearest-even.
const std::vector<Case>& cases() {
  static const std::vector<Case> table = {
      {0x477FE000u, 0x7BFFu, "65504 (largest finite)"},
      {0x477FEF00u, 0x7BFFu, "65519 (below the overflow threshold)"},
      {0x477FF000u, 0x7C00u, "65520 (overflow threshold, tie to inf)"},
      {0xC77FF000u, 0xFC00u, "-65520"},
      {0x47800000u, 0x7C00u, "65536"},
      {0x38800000u, 0x0400u, "2^-14 (smallest normal)"},
      {0x387FF000u, 0x0400u, "2^-14 - 2^-25 (tie across the boundary)"},
      {0x33800000u, 0x0001u, "2^-24 (subnormal unit)"},
      {0x33000000u, 0x0000u, "2^-25 (tie to even zero)"},
      {0x33000001u, 0x0001u, "just above 2^-25"},
      {0xB3000000u, 0x8000u, "-2^-25"},
      {0x33C00000u, 0x0002u, "1.5 * 2^-24 (subnormal tie up to even)"},
      {0x34200000u, 0x0002u, "2.5 * 2^-24 (subnormal tie down to even)"},
      {0x3F801000u, 0x3C00u, "1 + 2^-11 (tie down to even)"},
      {0x3F803000u, 0x3C02u, "1 + 3 * 2^-11 (tie up to even)"},
      {0x3F801001u, 0x3C01u, "just above 1 + 2^-11"},
      {0x3F800FFFu, 0x3C00u, "just below 1 + 2^-11"},
      {0x7FC00000u, 0x7E00u, "quiet NaN"},
      {0xFFC00000u, 0xFE00u, "negative quiet NaN"},
      {0x7FC02000u, 0x7E01u, "quiet NaN with payload"},
      {0x80000000u, 0x8000u, "-0"},
      {0x00000000u, 0x0000u, "+0"},
      {0x7F800000u, 0x7C00u, "inf"},
      {0xFF800000u, 0xFC00u, "-inf"},
      {0x00000001u, 0x0000u, "smallest float32 denormal"},
      {0x3EAAAAABu, 0x3555u, "1/3"},
  };
  return table;
}

float bits_to_float(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::vector<float> case_values() {
  std::vector<float> values;
  for (const auto& c : cases()) {
    values.push_back(bits_to_float(c.f32_bits));
  }
  return values;
}

std::vector<uint16_t> half_bits(array value, const Stream& stream) {
  value = contiguous(value, false, stream);
  value.eval();
  std::vector<uint16_t> out(value.size());
  std::memcpy(out.data(), value.data<float16_t>(), out.size() * 2);
  return out;
}

bool is_half_nan(uint32_t h) {
  return (h & 0x7C00u) == 0x7C00u && (h & 0x03FFu) != 0u;
}

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

} // namespace

TEST_CASE("omarchy_f16_bits_rne matches the fixed IEEE half bits") {
  for (const auto& c : cases()) {
    INFO(c.name);
    CHECK_EQ(omarchy_glsl::omarchy_f16_bits_rne(bits_to_float(c.f32_bits)),
             uint32_t(c.f16_bits));
  }
}

TEST_CASE("omarchy_f16_bits_rne matches CPU astype(float16)") {
  Stream cpu = default_stream(Device::cpu);
  auto values = case_values();
  auto ref = half_bits(astype(
      array(values.data(), {int(values.size())}, float32), float16, cpu), cpu);
  for (size_t i = 0; i < values.size(); ++i) {
    INFO(cases()[i].name);
    if (std::isnan(values[i])) {
      CHECK(is_half_nan(ref[i]));
      CHECK_EQ(ref[i] & 0x8000u, cases()[i].f16_bits & 0x8000u);
      continue;
    }
    CHECK_EQ(omarchy_glsl::omarchy_f16_bits_rne(values[i]), uint32_t(ref[i]));
    CHECK_EQ(ref[i], cases()[i].f16_bits);
  }

  // Strided sweep over the float32 bit space (odd stride, so every
  // exponent and both signs are visited with varied low bits).
  std::vector<float> sweep;
  for (uint64_t bits = 0; bits <= 0xFFFFFFFFull; bits += 4099u) {
    sweep.push_back(bits_to_float(static_cast<uint32_t>(bits)));
  }
  auto sweep_ref = half_bits(astype(
      array(sweep.data(), {int(sweep.size())}, float32), float16, cpu), cpu);
  size_t mismatches = 0;
  size_t nans = 0;
  for (size_t i = 0; i < sweep.size(); ++i) {
    uint32_t got = omarchy_glsl::omarchy_f16_bits_rne(sweep[i]);
    if (std::isnan(sweep[i])) {
      ++nans;
      CHECK(is_half_nan(got));
      CHECK(is_half_nan(sweep_ref[i]));
      CHECK_EQ(got & 0x8000u, uint32_t(sweep_ref[i] & 0x8000u));
      continue;
    }
    if (got != sweep_ref[i]) {
      if (mismatches < 8) {
        INFO("f32 bits ", omarchy_glsl::floatBitsToUint(sweep[i]));
        CHECK_EQ(got, uint32_t(sweep_ref[i]));
      }
      ++mismatches;
    }
  }
  CHECK_EQ(mismatches, 0u);
  CHECK_GT(nans, 0u);
}

// A strided float32 -> float16 astype takes the dtype-converting general
// copy, which is cast_int.comp; its f16 store is omarchy_f16_bits_rne.
TEST_CASE("strided float32 -> float16 astype on the GPU rounds to nearest even") {
  if (!compute_available()) {
    return;
  }
  Stream gpu = new_stream(Device::gpu);
  auto values = case_values();
  const int n = int(values.size());
  std::vector<float> interleaved;
  for (float v : values) {
    interleaved.push_back(v);
    interleaved.push_back(0.0f);
  }
  array pairs(interleaved.data(), {n, 2}, float32);
  // Column 0 of an {n, 2} array has a row stride of 2, so the astype
  // reads it through the general (strided) copy.
  array column = slice(pairs, {0, 0}, {n, 1}, gpu);
  array out = astype(column, float16, gpu);
  out.eval();
  omarchy::get_command_encoder(gpu).synchronize();
  auto got = half_bits(out, gpu);
  REQUIRE_EQ(got.size(), cases().size());
  for (size_t i = 0; i < got.size(); ++i) {
    INFO(cases()[i].name, " got ", got[i], " want ", cases()[i].f16_bits);
    if (std::isnan(values[i])) {
      CHECK(is_half_nan(got[i]));
      CHECK_NE(got[i] & 0x0200u, 0u);
      continue;
    }
    CHECK_EQ(got[i], cases()[i].f16_bits);
  }
}

// The f16 rsqrt route widens the half operand through a runtime-zero OR
// (elementwise.comp); exact-power inputs and specials pin its bits.
TEST_CASE("f16 rsqrt on the GPU keeps the operand bits") {
  if (!compute_available()) {
    return;
  }
  Stream gpu = new_stream(Device::gpu);
  Stream cpu = default_stream(Device::cpu);
  // value, expected rsqrt as half bits
  const std::vector<std::pair<uint16_t, uint16_t>> exact = {
      {0x4400u, 0x3800u}, // 4 -> 0.5
      {0x3400u, 0x4000u}, // 0.25 -> 2
      {0x4C00u, 0x3400u}, // 16 -> 0.25
      {0x0400u, 0x5800u}, // 2^-14 -> 128
      {0x0001u, 0x6C00u}, // 2^-24 -> 4096
      {0x0000u, 0x7C00u}, // +0 -> inf
      {0x7C00u, 0x0000u}, // inf -> 0
  };
  std::vector<float16_t> input;
  for (auto [in_bits, unused] : exact) {
    float16_t h;
    std::memcpy(&h, &in_bits, 2);
    input.push_back(h);
  }
  const uint16_t nan_bits = 0x7E00u;
  const uint16_t near_bits[] = {0x4200u, 0x7BFFu, 0x3C01u}; // 3, 65504, 1+ulp
  float16_t h;
  std::memcpy(&h, &nan_bits, 2);
  input.push_back(h);
  for (uint16_t b : near_bits) {
    std::memcpy(&h, &b, 2);
    input.push_back(h);
  }
  array x(input.data(), {int(input.size())}, float16);
  array on_gpu = rsqrt(x, gpu);
  on_gpu.eval();
  omarchy::get_command_encoder(gpu).synchronize();
  auto got = half_bits(on_gpu, gpu);
  auto ref = half_bits(rsqrt(x, cpu), cpu);
  for (size_t i = 0; i < exact.size(); ++i) {
    INFO("input bits ", exact[i].first, " got ", got[i]);
    CHECK_EQ(got[i], exact[i].second);
    CHECK_EQ(ref[i], exact[i].second);
  }
  CHECK_EQ(got[exact.size()] & 0x7C00u, 0x7C00u);
  CHECK_NE(got[exact.size()] & 0x03FFu, 0u);
  for (size_t i = exact.size() + 1; i < got.size(); ++i) {
    INFO("input index ", i, " got ", got[i], " cpu ", ref[i]);
    int diff = int(got[i]) - int(ref[i]);
    CHECK(std::abs(diff) <= 1);
  }
}
