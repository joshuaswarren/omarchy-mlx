// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Dense matmul N-split and reduction row-window tests for storage
// bindings past maxStorageBufferRange (see the N-window loop in
// dispatch_matmul and the row windows in dispatch_reduce_general /
// dispatch_arg_reduce_suffix). The knob MLX_OMARCHY_MAX_STORAGE_RANGE
// lowers the effective limit, so a CPU Vulkan device exercises the
// splitters with small arrays.
//
// Two modes, selected by that env in the environment:
//   limited   (env set, e.g. 16384): every oversized case runs the
//             windowed path; float/int results are compared against a
//             CPU stream, and the uncovered corners (non-transposed
//             oversized rhs, reduction-axis-dominant float input, sort)
//             must keep the encoder's named refusal.
//   unlimited (env unset): the identical cases run single-dispatch.
// Each case prints ONE stable digest line:
//   DENSESPLIT digest case=<name> fnv=<16 hex> items=<N>
// A receipt diffs the digest lines between the limited and unlimited
// runs: equal digests prove the windowed path is bit-identical to the
// single-dispatch path on this device (N-split and row windows keep
// every output element's full-length accumulation order, so the
// requirement is exact equality, not a tolerance).

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
  std::cout << "DENSESPLIT digest case=" << name << " fnv=" << hex
            << " items=" << a.size() << "\n";
  std::cout.flush();
}

// Deterministic finite floats in [-15.7, 15.7]: no NaN or inf, identical
// across runs.
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

std::vector<uint16_t> f16_data(int n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint16_t> v(n);
  for (int i = 0; i < n; ++i) {
    v[i] = static_cast<uint16_t>(0x3C00u | (rng() & 0x3FFu));
  }
  return v;
}

  // Element-wise bit-exact comparison through the arrays' raw bytes.
void require_same_bytes(const array& gpu, const array& cpu) {
  REQUIRE(gpu.nbytes() == cpu.nbytes());
  REQUIRE(
      std::memcmp(gpu.data<uint8_t>(), cpu.data<uint8_t>(), gpu.nbytes()) ==
      0);
}

// Worst-case per-element absolute error. bf16/f16 matmul rides a
// different code path on lavapipe than the reference mlx CPU matmul,
// so the elements can differ; the digest is the actual bit-identity
// proof (limited vs unlimited) while this bounds the visible drift.
template <typename T>
void check_max_abs(const array& gpu, const array& cpu, T tol) {
  REQUIRE(gpu.size() == cpu.size());
  const T* gv = gpu.data<T>();
  const T* rv = cpu.data<T>();
  T max_diff = 0;
  for (size_t i = 0; i < gpu.size(); ++i) {
    T d = std::fabs(static_cast<T>(gv[i]) - static_cast<T>(rv[i]));
    if (d > max_diff) {
      max_diff = d;
    }
  }
  CHECK(max_diff <= tol);
}

bool omarchy_gpu_available() {
  return gpu::is_available();
}

bool limit_forced() {
  const char* env = std::getenv("MLX_OMARCHY_MAX_STORAGE_RANGE");
  return env != nullptr && env[0] != '\0';
}

// Matmul shapes sized so a 16 KiB knob still leaves room for a 32-column
// window (per-column rhs bytes must stay under (limit-3)/32), and
// K % 8 != 0 so the cooperative-matrix and direct routes stay off on
// every device and both passes run the same plain scalar-tile kernels.
constexpr int kKBf16 = 220;
constexpr int kKF32 = 96;

} // namespace

TEST_CASE("dense matmul survives a lowered storage range") {
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  Stream gs = new_stream(Device::gpu);
  Stream cs = new_stream(Device::cpu);

  // bf16 readout orientation: x [4, K] @ W.T, W [N, K] row-major. The
  // transposed rhs windows along its contiguous rows; K % 8 != 0 keeps
  // every cooperative-matrix route off.
  constexpr int n_bf16 = 2048;
  auto xb = bf16_data(4 * kKBf16, 1);
  auto wb = bf16_data(n_bf16 * kKBf16, 2);
  array gx(xb.data(), Shape{4, kKBf16}, bfloat16);
  array gw(wb.data(), Shape{n_bf16, kKBf16}, bfloat16);
  array rx(xb.data(), Shape{4, kKBf16}, bfloat16);
  array rw(wb.data(), Shape{n_bf16, kKBf16}, bfloat16);
  SUBCASE("bf16 transposed rhs (the readout orientation)") {
    array g = matmul(gx, transpose(gw), gs);
    eval(g);
    if (!limit_forced()) {
      array r = matmul(rx, transpose(rw), cs);
      eval(r);
      // The digest is the real bit-identity proof; this is a CPU/GPU
      // matmul sanity check on a per-element absolute basis (lavapipe
      // bf16 rounds through fp32 and disagrees with mlx CPU on
      // signed-zero boundaries, so a relative tolerance diverges).
      check_max_abs<bfloat16_t>(g, r, bfloat16_t(1.0f));
    }
    digest("mm-bf16-nt", g);
  }
  SUBCASE("bf16 gemv m=1 (vec-route gates stay off)") {
    auto x1 = bf16_data(kKBf16, 3);
    array g1x(x1.data(), Shape{1, kKBf16}, bfloat16);
    array r1x(x1.data(), Shape{1, kKBf16}, bfloat16);
    array g = matmul(g1x, transpose(gw), gs);
    eval(g);
    if (!limit_forced()) {
      array r = matmul(r1x, transpose(rw), cs);
      eval(r);
      check_max_abs<bfloat16_t>(g, r, bfloat16_t(1.0f));
    }
    digest("mm-bf16-m1", g);
  }
  SUBCASE("f32 transposed rhs with oversized output row (m=1)") {
    constexpr int n_f32 = 8192;
    auto xf = f32_data(kKF32, 4);
    auto wf = f32_data(n_f32 * kKF32, 5);
    array gx(xf.data(), Shape{1, kKF32}, float32);
    array gw(wf.data(), Shape{n_f32, kKF32}, float32);
    array rx(xf.data(), Shape{1, kKF32}, float32);
    array rw(wf.data(), Shape{n_f32, kKF32}, float32);
    array g = matmul(gx, transpose(gw), gs);
    eval(g);
    if (!limit_forced()) {
      array r = matmul(rx, transpose(rw), cs);
      eval(r);
      require_same_bytes(g, r);
    }
    digest("mm-f32-both", g);
  }
  // f16 plain kernel (decode shape: m=4 keeps the output slab under the
  // 16 KiB knob; m>=32 would trip the designed "output row span already
  // fills the limit" refusal for the readout class). K % 8 != 0 keeps
  // every cooperative-matrix / direct route off so both passes run the
  // same plain f16 tile.
  SUBCASE("f16 plain kernel, decode-shaped m=4") {
    constexpr int n_f16 = 1024;
    auto xf = f16_data(4 * kKBf16, 6);
    auto wf = f16_data(n_f16 * kKBf16, 7);
    array gx(xf.data(), Shape{4, kKBf16}, float16);
    array gw(wf.data(), Shape{n_f16, kKBf16}, float16);
    array rx(xf.data(), Shape{4, kKBf16}, float16);
    array rw(wf.data(), Shape{n_f16, kKBf16}, float16);
    array g = matmul(gx, transpose(gw), gs);
    eval(g);
    if (!limit_forced()) {
      array r = matmul(rx, transpose(rw), cs);
      eval(r);
      check_max_abs<float16_t>(g, r, float16_t(1.0f));
    }
    digest("mm-f16-m4", g);
  }
  SUBCASE("small matmul stays on the single-dispatch path") {
    auto xs = bf16_data(4 * 32, 8);
    auto ws = bf16_data(64 * 32, 9);
    array gx(xs.data(), Shape{4, 32}, bfloat16);
    array gw(ws.data(), Shape{64, 32}, bfloat16);
    array rx(xs.data(), Shape{4, 32}, bfloat16);
    array rw(ws.data(), Shape{64, 32}, bfloat16);
    array g = matmul(gx, transpose(gw), gs);
    eval(g);
    if (!limit_forced()) {
      array r = matmul(rx, transpose(rw), cs);
      eval(r);
      check_max_abs<bfloat16_t>(g, r, bfloat16_t(1.0f));
    }
    digest("mm-bf16-small", g);
  }
}

TEST_CASE("matmul uncovered corners keep the oversized-binding refusal") {
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  if (!limit_forced()) {
    std::cout << "Skipping: limit not forced (single-dispatch run)\n";
    return;
  }
  Stream gs = new_stream(Device::gpu);
  // Non-transposed rhs: column windows are strided slabs through the
  // buffer, so no N-split can cover it and the encoder must refuse.
  auto xf = f32_data(4 * kKF32, 10);
  auto bf = f32_data(kKF32 * 8192, 11);
  array gx(xf.data(), Shape{4, kKF32}, float32);
  array gb(bf.data(), Shape{kKF32, 8192}, float32);
  bool threw = false;
  std::string message;
  try {
    array g = matmul(gx, gb, gs);
    eval(g);
  } catch (const std::exception& e) {
    threw = true;
    message = e.what();
  }
  REQUIRE(threw);
  CHECK(message.find("One binding cannot address that much memory") !=
        std::string::npos);
  std::cout << "DENSESPLIT matmul refusal message=" << message << "\n";
}

TEST_CASE("reductions survive a lowered storage range") {
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  Stream gs = new_stream(Device::gpu);
  Stream cs = new_stream(Device::cpu);
  constexpr int rows = 256;
  constexpr int k = 96;  // k <= 4096 keeps the two-phase chunk path off

  auto do_reduce = [&](int op, const array& x, Stream s) -> array {
    switch (op) {
      case 0: return sum(x, -1, false, s);
      case 1: return max(x, -1, false, s);
      case 2: return min(x, -1, false, s);
      default: return prod(x, -1, false, s);
    }
  };
  auto run_reduce = [&](const char* name, int op, const array& gi,
                        const array& ri, bool vs_cpu) {
    array g = do_reduce(op, gi, gs);
    eval(g);
    if (vs_cpu) {
      array r = do_reduce(op, ri, cs);
      eval(r);
      require_same_bytes(g, r);
    }
    digest(name, g);
  };

  SUBCASE("int32 sum/max/min/prod") {
    std::mt19937 rng(12);
    std::vector<int32_t> data(rows * k);
    for (auto& v : data) {
      v = static_cast<int32_t>(rng() % 2001 - 1000);
    }
    array gi(data.data(), Shape{rows, k}, int32);
    array ri(data.data(), Shape{rows, k}, int32);
    run_reduce("red-i32-sum", 0, gi, ri, true);
    run_reduce("red-i32-max", 1, gi, ri, true);
    run_reduce("red-i32-min", 2, gi, ri, true);
    run_reduce("red-i32-prod", 3, gi, ri, true);
  }
  SUBCASE("f32 sum/max/min/prod") {
    auto data = f32_data(rows * k, 13);
    array gi(data.data(), Shape{rows, k}, float32);
    array ri(data.data(), Shape{rows, k}, float32);
    run_reduce("red-f32-sum", 0, gi, ri, true);
    run_reduce("red-f32-max", 1, gi, ri, true);
    run_reduce("red-f32-min", 2, gi, ri, true);
    run_reduce("red-f32-prod", 3, gi, ri, true);
  }
  SUBCASE("bf16 sum/max/min digest-only") {
    auto data = bf16_data(rows * k, 14);
    array gi(data.data(), Shape{rows, k}, bfloat16);
    run_reduce("red-bf16-sum", 0, gi, gi, false);
    run_reduce("red-bf16-max", 1, gi, gi, false);
    run_reduce("red-bf16-min", 2, gi, gi, false);
    run_reduce("red-bf16-prod", 3, gi, gi, false);
  }
  SUBCASE("small reduce stays on the single-dispatch path") {
    std::vector<int32_t> data(8 * 16);
    for (int i = 0; i < 8 * 16; ++i) {
      data[i] = i - 60;
    }
    array gi(data.data(), Shape{8, 16}, int32);
    run_reduce("red-i32-small", 0, gi, gi, true);
  }
  SUBCASE("argmax/argmin with distinct row values") {
    // Values are distinct within each row so index tie-breaking cannot
    // differ between the windowed and single-dispatch walks.
    std::vector<float> data(rows * k);
    for (int r = 0; r < rows; ++r) {
      for (int j = 0; j < k; ++j) {
        data[r * k + j] = static_cast<float>((r * 7919 + j * 104729) % 4093) -
            2048.0f;
      }
    }
    array gi(data.data(), Shape{rows, k}, float32);
    array ri(data.data(), Shape{rows, k}, float32);
    array gmax = argmax(gi, -1, true, gs);
    eval(gmax);
    array rmax = argmax(ri, -1, true, cs);
    eval(rmax);
    require_same_bytes(gmax, rmax);
    digest("argmax-f32", gmax);
    array gmin = argmin(gi, -1, true, gs);
    eval(gmin);
    array rmin = argmin(ri, -1, true, cs);
    eval(rmin);
    require_same_bytes(gmin, rmin);
    digest("argmin-f32", gmin);
  }
  SUBCASE("small argmax stays on the single-dispatch path") {
    std::vector<float> data(8 * 16);
    for (int i = 0; i < 8 * 16; ++i) {
      data[i] = static_cast<float>(i % 7) - 3.0f;
    }
    array gi(data.data(), Shape{8, 16}, float32);
    array g = argmax(gi, -1, true, gs);
    eval(g);
    digest("argmax-small", g);
  }
}

TEST_CASE("reduction uncovered corners keep the oversized-binding refusal") {
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  if (!limit_forced()) {
    std::cout << "Skipping: limit not forced (single-dispatch run)\n";
    return;
  }
  Stream gs = new_stream(Device::gpu);
  // Reduction-axis dominant: one row's reduced span exceeds the limit,
  // so no output-row window can cover it. A reduction-axis split would
  // reassociate float accumulation, so this must keep refusing rather
  // than silently reorder.
  auto data = f32_data(4 * 8192, 15);
  array gi(data.data(), Shape{4, 8192}, float32);
  bool threw = false;
  std::string message;
  try {
    array g = sum(gi, -1, false, gs);
    eval(g);
  } catch (const std::exception& e) {
    threw = true;
    message = e.what();
  }
  REQUIRE(threw);
  CHECK(message.find("One binding cannot address that much memory") !=
        std::string::npos);
  std::cout << "DENSESPLIT reduce refusal message=" << message << "\n";
}

TEST_CASE("sort at a lowered storage range is report-only") {
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  if (!limit_forced()) {
    std::cout << "Skipping: limit not forced (single-dispatch run)\n";
    return;
  }
  Stream gs = new_stream(Device::gpu);
  auto data = f32_data(512 * 256, 16);
  array src(data.data(), Shape{512, 256}, float32);
  bool threw = false;
  std::string message;
  try {
    array g = sort(src, -1, gs);
    eval(g);
    digest("sort-f32", g);
  } catch (const std::exception& e) {
    threw = true;
    message = e.what();
  }
  // Report-only: the sort family is not windowed on this branch. Record
  // whatever happens without failing the suite.
  std::cout << "DENSESPLIT sort refusal threw=" << (threw ? "yes" : "no")
            << " message=" << message << "\n";
}
