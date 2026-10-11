// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Large-binding chunk split tests: flat-index elementwise and copy
// dispatches whose whole-array storage bindings would exceed the
// device's maxStorageBufferRange are issued in chunks along the flat
// element index (see dispatch_compute_chunked in encoder.h). The knob
// MLX_OMARCHY_MAX_STORAGE_RANGE lowers the effective limit, so a CPU
// Vulkan device exercises the splitter with small arrays.
//
// Two modes, selected by that env in the environment:
//   limited   (env set, e.g. 16384): every case's array exceeds the
//             limit, so the chunked path runs; each result is compared
//             bit-exact against the same op on a CPU stream, and the
//             strided broadcast case must keep the encoder's named
//             refusal.
//   unlimited (env unset): the identical cases run single-dispatch.
// Each case prints ONE stable digest line:
//   BIGBIND digest case=<name> fnv=<16 hex> items=<N>
// A receipt diffs the digest lines between the limited and unlimited
// runs: equal digests prove the chunked path is bit-identical to the
// single-dispatch path on this device.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

uint64_t fnv1a(const void* data, size_t bytes) {
  const auto* p = static_cast<const uint8_t*>(data);
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < bytes; ++i) {
    h = (h ^ p[i]) * 1099511628211ull;
  }
  return h;
}

void digest(const char* name, const array& a) {
  uint64_t h = fnv1a(a.data<uint8_t>(), a.nbytes());
  char hex[17];
  snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(h));
  std::cout << "BIGBIND digest case=" << name << " fnv=" << hex
            << " items=" << a.size() << "\n";
  std::cout.flush();
}

// Fills with a deterministic pseudo-random bit pattern in the low
// mantissa bits: finite normals, no NaN or inf, identical across runs.
template <typename T>
std::vector<T> pattern(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<T> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = static_cast<T>(rng());
  }
  return v;
}

// Element-wise bit-exact comparison through the arrays' raw bytes.
// Both sides come from the same op with the same inputs, so any byte
// difference is a correctness failure; sizes and dtypes match by
// construction.
void require_same_bytes(const array& gpu, const array& cpu) {
  REQUIRE(gpu.nbytes() == cpu.nbytes());
  REQUIRE(
      std::memcmp(gpu.data<uint8_t>(), cpu.data<uint8_t>(), gpu.nbytes()) ==
      0);
}

bool omarchy_gpu_available() {
  return gpu::is_available();
}

bool limit_forced() {
  const char* env = std::getenv("MLX_OMARCHY_MAX_STORAGE_RANGE");
  return env != nullptr && env[0] != '\0';
}

constexpr int kOdd = 44441;  // not a multiple of 4 or of any chunk

// Deterministic finite floats in [-15.7, 15.7]: no NaN or inf, so both
// streams must produce identical bytes for identical elementwise math.
std::vector<float> f32_data(int n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) {
    v[i] = static_cast<float>(rng() % 2001 - 1000) / 64.0f;
  }
  return v;
}

// bf16 bit patterns in [0.5, 1): normal numbers, deterministic mantissas.
std::vector<uint16_t> bf16_data(int n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint16_t> v(n);
  for (int i = 0; i < n; ++i) {
    v[i] = static_cast<uint16_t>(0x3F00u | (rng() & 0x7Fu));
  }
  return v;
}

} // namespace

TEST_CASE("astype flat casts survive a lowered storage range") {
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  Stream gs = new_stream(Device::gpu);
  Stream cs = new_stream(Device::cpu);

  SUBCASE("bf16 to f32 (the full-readout cast)") {
    auto raw = pattern<uint16_t>(kOdd, 1);
    array in(raw.data(), Shape{kOdd}, bfloat16);
    array g = astype(in, float32, gs);
    eval(g);
    array r = astype(in, float32, cs);
    eval(r);
    require_same_bytes(g, r);
    digest("astype-bf16-f32", g);
  }
  SUBCASE("f32 to bf16") {
    auto raw = f32_data(kOdd, 2);
    array in(raw.data(), Shape{kOdd}, float32);
    array g = astype(in, bfloat16, gs);
    eval(g);
    array r = astype(in, bfloat16, cs);
    eval(r);
    require_same_bytes(g, r);
    digest("astype-f32-bf16", g);
  }
  SUBCASE("bool to f32 (word-window source kernel)") {
    std::vector<uint8_t> bits(kOdd);
    for (int i = 0; i < kOdd; ++i) {
      bits[i] = (i * 2654435761u) % 7 < 3;
    }
    array in(bits.data(), Shape{kOdd}, bool_);
    array g = astype(in, float32, gs);
    eval(g);
    array r = astype(in, float32, cs);
    eval(r);
    require_same_bytes(g, r);
    digest("astype-bool-f32", g);
  }
  SUBCASE("i32 to f16") {
    auto raw = pattern<uint32_t>(kOdd, 3);
    array in(raw.data(), Shape{kOdd}, int32);
    array g = astype(in, float16, gs);
    eval(g);
    array r = astype(in, float16, cs);
    eval(r);
    require_same_bytes(g, r);
    digest("astype-i32-f16", g);
  }
  SUBCASE("i64 to f32 (8-byte source)") {
    std::vector<int64_t> raw(kOdd);
    for (int i = 0; i < kOdd; ++i) {
      raw[i] = static_cast<int64_t>(i) * 6180339887 + 7;
    }
    array in(raw.data(), Shape{kOdd}, int64);
    array g = astype(in, float32, gs);
    eval(g);
    array r = astype(in, float32, cs);
    eval(r);
    require_same_bytes(g, r);
    digest("astype-i64-f32", g);
  }
}

TEST_CASE("elementwise families survive a lowered storage range") {
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  Stream gs = new_stream(Device::gpu);
  Stream cs = new_stream(Device::cpu);

  SUBCASE("f32 add (general kernel)") {
    auto a = f32_data(kOdd, 4);
    auto b = f32_data(kOdd, 5);
    array ga(a.data(), Shape{kOdd}, float32);
    array gb(b.data(), Shape{kOdd}, float32);
    array g = add(ga, gb, gs);
    eval(g);
    array ra(a.data(), Shape{kOdd}, float32);
    array rb(b.data(), Shape{kOdd}, float32);
    array r = add(ra, rb, cs);
    eval(r);
    require_same_bytes(g, r);
    digest("add-f32", g);
  }
  SUBCASE("bf16 add (vec4 fast path, count multiple of 4)") {
    constexpr int n = kOdd - 1;  // 44440, multiple of 4
    auto a = bf16_data(n, 6);
    auto b = bf16_data(n, 7);
    array ga(a.data(), Shape{n}, bfloat16);
    array gb(b.data(), Shape{n}, bfloat16);
    array g = add(ga, gb, gs);
    eval(g);
    array ra(a.data(), Shape{n}, bfloat16);
    array rb(b.data(), Shape{n}, bfloat16);
    array r = add(ra, rb, cs);
    eval(r);
    require_same_bytes(g, r);
    digest("add-bf16-vec", g);
  }
  SUBCASE("f16 negative (unary general kernel)") {
    constexpr int n = 2 * kOdd + 1;
    std::mt19937 rng(8);
    std::vector<uint16_t> a(n);
    for (int i = 0; i < n; ++i) {
      a[i] = static_cast<uint16_t>(0x3C00u | (rng() & 0x3FFu));
    }
    array in(a.data(), Shape{n}, float16);
    array g = negative(in, gs);
    eval(g);
    array r = negative(in, cs);
    eval(r);
    require_same_bytes(g, r);
    digest("neg-f16", g);
  }
  SUBCASE("i32 add (int kernel)") {
    auto a = pattern<uint32_t>(kOdd, 9);
    auto b = pattern<uint32_t>(kOdd, 10);
    array ga(a.data(), Shape{kOdd}, int32);
    array gb(b.data(), Shape{kOdd}, int32);
    array g = add(ga, gb, gs);
    eval(g);
    array ra(a.data(), Shape{kOdd}, int32);
    array rb(b.data(), Shape{kOdd}, int32);
    array r = add(ra, rb, cs);
    eval(r);
    require_same_bytes(g, r);
    digest("int-add-i32", g);
  }
  SUBCASE("greater compare (bool word-packed out)") {
    auto a = f32_data(kOdd, 11);
    auto b = f32_data(kOdd, 12);
    array ga(a.data(), Shape{kOdd}, float32);
    array gb(b.data(), Shape{kOdd}, float32);
    array g = greater(ga, gb, gs);
    eval(g);
    array ra(a.data(), Shape{kOdd}, float32);
    array rb(b.data(), Shape{kOdd}, float32);
    array r = greater(ra, rb, cs);
    eval(r);
    require_same_bytes(g, r);
    digest("cmp-gt-f32", g);
  }
  SUBCASE("complex add (8-byte items)") {
    constexpr int n = 11111;
    auto re = f32_data(n, 13);
    auto im = f32_data(n, 14);
    std::vector<float> interleaved(2 * n);
    for (int i = 0; i < n; ++i) {
      interleaved[2 * i] = re[i];
      interleaved[2 * i + 1] = im[i];
    }
    std::vector<float> interleaved_b(2 * n);
    for (int i = 0; i < n; ++i) {
      interleaved_b[2 * i] = im[i];
      interleaved_b[2 * i + 1] = re[i];
    }
    array ga(interleaved.data(), Shape{n}, complex64);
    array gb(interleaved_b.data(), Shape{n}, complex64);
    array g = add(ga, gb, gs);
    eval(g);
    array ra(interleaved.data(), Shape{n}, complex64);
    array rb(interleaved_b.data(), Shape{n}, complex64);
    array r = add(ra, rb, cs);
    eval(r);
    require_same_bytes(g, r);
    digest("complex-add", g);
  }
  SUBCASE("trailing broadcast add (modulo operand, flat out)") {
    constexpr int rows = 512;
    constexpr int cols = 871;
    auto a = f32_data(rows * cols, 15);
    auto b = f32_data(cols, 16);
    array ga(a.data(), Shape{rows, cols}, float32);
    array gb(b.data(), Shape{cols}, float32);
    array g = add(ga, gb, gs);
    eval(g);
    array ra(a.data(), Shape{rows, cols}, float32);
    array rb(b.data(), Shape{cols}, float32);
    array r = add(ra, rb, cs);
    eval(r);
    require_same_bytes(g, r);
    digest("add-broadcast", g);
  }
  SUBCASE("i32 arange (chunk-shifted start)") {
    constexpr int n = 130003;
    array g = arange(7, 7 + 3 * n, 3, int32, gs);
    eval(g);
    array r = arange(7, 7 + 3 * n, 3, int32, cs);
    eval(r);
    require_same_bytes(g, r);
    digest("arange-i32", g);
  }
  SUBCASE("i64 arange (high start bits)") {
    constexpr int n = 100003;
    int64_t start = 1ll << 33;
    array g = arange(start, start + 5 * n, 5, int64, gs);
    eval(g);
    array r = arange(start, start + 5 * n, 5, int64, cs);
    eval(r);
    require_same_bytes(g, r);
    digest("arange-i64", g);
  }
}

TEST_CASE("strided broadcast keeps the oversized-binding refusal") {
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  if (!limit_forced()) {
    std::cout << "Skipping: limit not forced (single-dispatch run)\n";
    return;
  }
  // A transposed operand is gapless strided (dims > 0 in the shader):
  // the chunk splitter cannot shift it, so an oversized binding must
  // still raise the encoder's named refusal rather than split.
  Stream gs = new_stream(Device::gpu);
  auto a = pattern<uint32_t>(1024 * 1024, 17);
  auto b = pattern<uint32_t>(1024 * 1024, 19);
  array src(b.data(), Shape{1024, 1024}, float32);
  array lhs = transpose(src);
  array ga(a.data(), Shape{1024, 1024}, float32);
  bool threw = false;
  std::string message;
  try {
    array g = add(lhs, ga, gs);
    eval(g);
  } catch (const std::exception& e) {
    threw = true;
    message = e.what();
  }
  REQUIRE(threw);
  CHECK(message.find("One binding cannot address that much memory") !=
        std::string::npos);
  std::cout << "BIGBIND strided refusal message=" << message << "\n";
}
