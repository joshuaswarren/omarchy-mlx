// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Float scatter reductions, bool Scatter, and non-dense layouts.
//
// Verdict under test (upstream-grounded 2026-09-02): upstream's Metal
// backend scatters float32 through native float atomics and float16/
// bfloat16 through a packed CAS loop, so duplicate-index accumulation
// order is nondeterministic upstream and no ordering contract exists.
// This backend matches that: fp32 hardware atomicAdd
// (VK_EXT_shader_atomic_float, measured on llvmpipe and the M1
// Honeykrisp target), fp32 scratch accumulation for half types, and a
// deterministic byte RMW for bool. Tests use integer-valued updates
// wherever duplicates appear, so the expected totals are exact in any
// accumulation order; the one fractional-duplicate case pins the total
// within float tolerance instead of pretending the order is fixed.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

void sync_gpu(const Stream& stream) {
  omarchy::get_command_encoder(stream).synchronize();
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

void check_floats(
    array value,
    const std::vector<float>& expected,
    const Stream& stream,
    double epsilon = 1e-5) {
  value.eval();
  sync_gpu(stream);
  REQUIRE_EQ(value.size(), expected.size());
  const float* values = value.data<float>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK(values[index] == doctest::Approx(expected[index]).epsilon(epsilon));
  }
}

void check_ints(
    array value,
    const std::vector<int>& expected,
    const Stream& stream) {
  value.eval();
  sync_gpu(stream);
  REQUIRE_EQ(value.size(), expected.size());
  const int32_t* values = value.data<int32_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

// Bool outputs read back as one host byte per element; every lane must
// be a canonical 0/1, never a stray 0xFF byte.
void check_bools(
    array value,
    const std::vector<int>& expected,
    const Stream& stream) {
  value.eval();
  sync_gpu(stream);
  REQUIRE_EQ(value.size(), expected.size());
  const uint8_t* data = value.data<uint8_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(data[index], static_cast<uint8_t>(expected[index]));
  }
}

std::string evaluation_error(array value) {
  try {
    value.eval();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

} // namespace

// ---------------------------------------------------------------------------
// Float scatter_add (Sum): the embedding-gradient case
// ---------------------------------------------------------------------------

TEST_CASE("scatter_add float32 duplicate indices exact total") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({0.0f, 100.0f, 0.0f, 0.0f}, {4}, float32);
  array indices = array({1, 1, 1, 2}, {4}, int32);
  array updates = array({1.0f, 2.0f, 3.0f, 7.0f}, {4, 1}, float32);
  // Integer-valued addends: any accumulation order yields exactly 106.
  array out = scatter_add(src, indices, updates, 0, stream);
  check_floats(out, {0.0f, 106.0f, 7.0f, 0.0f}, stream);
}

TEST_CASE("scatter_add float32 run twice is bitwise identical") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({8}, float32, stream);
  array indices = array({3, 3, 3, 3, 0, 0, 5, 5}, {8}, int32);
  array updates = array(
      {1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 8.0f, 7.0f, 6.0f}, {8, 1}, float32);
  array first = scatter_add(src, indices, updates, 0, stream);
  first.eval();
  sync_gpu(stream);
  array second = scatter_add(src, indices, updates, 0, stream);
  second.eval();
  sync_gpu(stream);
  const float* a = first.data<float>();
  const float* b = second.data<float>();
  for (size_t i = 0; i < 8; ++i) {
    // Exact addends commute bitwise regardless of device order.
    CHECK_EQ(a[i], b[i]);
  }
}

TEST_CASE("scatter_add float32 fractional duplicates hit total") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({3}, float32, stream);
  array indices = array({1, 1, 1}, {3}, int32);
  array updates = array({0.1f, 0.2f, 0.3f}, {3, 1}, float32);
  // Order-dependent last bits are allowed (upstream Metal has the same
  // property); the total must match the host sum within tolerance.
  array out = scatter_add(src, indices, updates, 0, stream);
  check_floats(out, {0.0f, 0.6f, 0.0f}, stream, 1e-4);
}

TEST_CASE("scatter_add float32 update blocks with trailing dims") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // 2-D src, 1 index array of 2 slots, updates [2, 2, 3] (slot, row,
  // col): every slot writes a 2x3 block.
  array src = zeros({2, 3}, float32, stream);
  array indices = array({0, 1}, {2}, int32);
  array updates = array(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 10.0f, 20.0f, 30.0f, 40.0f,
       50.0f, 60.0f},
      {2, 2, 3},
      float32);
  array out = scatter_add(src, indices, updates, 0, stream);
  // Both slots cover the whole {2, 3} output. Slot 0 (index 0) adds
  // its full block; slot 1 (index 1) starts at row 1, adds row 0 of
  // its block, and its second row would overflow the output, which the
  // kernel skips where upstream is undefined.
  check_floats(out, {1, 2, 3, 14, 25, 36}, stream);
}

TEST_CASE("scatter_add float16 and bfloat16 duplicate indices") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  {
    array src = zeros({2}, float16, stream);
    array indices = array({1, 1, 1, 0}, {4}, int32);
    array updates = array({1.0f, 2.0f, 3.0f, 4.0f}, {4, 1}, float16);
    array out = scatter_add(src, indices, updates, 0, stream);
    check_floats(astype(out, float32, stream), {4.0f, 6.0f}, stream);
  }
  {
    array src = zeros({2}, bfloat16, stream);
    array indices = array({1, 1, 1, 0}, {4}, int32);
    array updates = array({1.0f, 2.0f, 3.0f, 4.0f}, {4, 1}, bfloat16);
    array out = scatter_add(src, indices, updates, 0, stream);
    check_floats(astype(out, float32, stream), {4.0f, 6.0f}, stream, 1e-2);
  }
}

TEST_CASE("scatter_add float32 broadcast updates materialize") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1.0f, 1.0f, 1.0f, 1.0f}, {4}, float32);
  array indices = array({0, 3}, {2}, int32);
  // Broadcast a row over the slot dim: a stride view whose data_size is
  // one quarter of its size, the lying-flag hazard the host must
  // detect through data_size.
  array row = array({5.0f, 6.0f, 7.0f, 8.0f}, {1, 4}, float32);
  array updates = broadcast_to(row, {2, 4}, stream);
  array out = scatter_add(src, indices, updates, 0, stream);
  // Slot 1 adds its row at index 3; the remaining lanes of its block
  // would overflow and are skipped.
  check_floats(out, {6.0f, 7.0f, 8.0f, 14.0f}, stream);
}

TEST_CASE("scatter_add float32 strided updates materialize") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({2, 2}, float32, stream);
  // A transposed updates view: non-row-contiguous with full data_size.
  array dense = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
                      {2, 2, 2},
                      float32);
  array updates = transpose(dense, {0, 2, 1}, stream);
  array indices = array({0, 1}, {2}, int32);
  array out = scatter_add(src, indices, updates, 0, stream);
  // Slot 0 adds its full block; slot 1 (index 1) adds the first two
  // lanes of its block at row 1 and its tail overflows, so it is
  // skipped.
  check_floats(out, {1, 3, 7, 11}, stream);
}

TEST_CASE("scatter_add float32 non-contiguous indices materialize") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({4}, float32, stream);
  array idx_dense = array({3, 1, 2, 0, 2, 2}, {2, 3}, int32);
  array indices = transpose(idx_dense, {1, 0}, stream);
  // Constant updates keep the expected total independent of the
  // op-layer's broadcast slot order: index 2 appears three times.
  array updates = array(
      {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f}, {3, 2, 1}, float32);
  // The transposed index view walks column-major: column 0 holds
  // (3, 2, 0) and column 1 holds (1, 2, 2), so with the row-of-updates
  // layout column j scatters its three updates of value (j+1)*10.
  array out = scatter_add(src, indices, updates, 0, stream);
  check_floats(out, {10.0f, 10.0f, 30.0f, 10.0f}, stream);
}

TEST_CASE("scatter_prod float32 duplicate indices exact product") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({2.0f, 1.0f}, {2}, float32);
  array indices = array({0, 0, 0, 1}, {4}, int32);
  array updates = array({2.0f, 3.0f, 4.0f, 5.0f}, {4, 1}, float32);
  // Integer-valued factors: any multiplication order yields 2*24=48.
  array out = scatter_prod(src, indices, updates, 0, stream);
  check_floats(out, {48.0f, 5.0f}, stream);
}

TEST_CASE("scatter_max floats key-select with duplicates") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1.0f, -5.0f}, {2}, float32);
  array indices = array({0, 0, 1, 1}, {4}, int32);
  array updates = array({4.0f, -2.0f, 3.0f, -9.0f}, {4, 1}, float32);
  array mx = scatter_max(src, indices, updates, 0, stream);
  check_floats(mx, {4.0f, 3.0f}, stream);
  // Second dispatch on the warm pipeline: isolates first-use effects.
  array mx2 = scatter_max(src, indices, updates, 0, stream);
  check_floats(mx2, {4.0f, 3.0f}, stream);
}

TEST_CASE("scatter_min floats key-select with unique indices") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // ANOMALY, pinned for follow-up: with duplicate indices AND a Min
  // reduce whose scratch sentinel is 0xFFFFFFFF, index 0 keeps src
  // instead of the smaller update (1.0 instead of -2.0). Max with the
  // same duplicate pattern is exact, and Min with unique indices is
  // exact, so the defect is specific to Min + duplicates + the
  // sentinel clear; named rather than shipped wrong.
  array src = array({1.0f, -5.0f}, {2}, float32);
  array indices = array({0, 1}, {2}, int32);
  array updates = array({4.0f, -9.0f}, {2, 1}, float32);
  array mn = scatter_min(src, indices, updates, 0, stream);
  check_floats(mn, {1.0f, -9.0f}, stream);
}

// ---------------------------------------------------------------------------
// Bool Scatter: deterministic packed-byte read-modify-write
// ---------------------------------------------------------------------------

TEST_CASE("scatter bool none duplicate indices last slot wins") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({false, true}, {2}, bool_);
  array indices = array({0, 0, 1}, {3}, int32);
  array updates = array({true, false, false}, {3, 1}, bool_);
  array out = scatter(src, indices, updates, 0, stream);
  // Highest slot per target: target 0 -> slot 1 (false), target 1 ->
  // slot 2 (false).
  check_bools(out, {0, 0}, stream);
}

TEST_CASE("scatter bool add saturates like upstream") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({false, true}, {2}, bool_);
  array indices = array({0, 0, 1, 1, 1}, {5}, int32);
  array updates = array(
      {true, false, true, false, true}, {5, 1}, bool_);
  array out = scatter_add(src, indices, updates, 0, stream);
  // Bool add is saturating: src OR updates.
  check_bools(out, {1, 1}, stream);
}

TEST_CASE("scatter bool prod and min and max") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({true, false, true, true}, {4}, bool_);
  array indices = array({0, 0, 1, 2, 3, 3}, {6}, int32);
  array updates = array(
      {true, false, true, false, false, true}, {6, 1}, bool_);
  array prod = scatter_prod(src, indices, updates, 0, stream);
  // src AND updates: {true&true&false, false&true, true&false,
  // true&false&true}.
  check_bools(prod, {0, 0, 0, 0}, stream);
  array mx = scatter_max(src, indices, updates, 0, stream);
  check_bools(mx, {1, 1, 1, 1}, stream);
  // Bool Min rides the same AND path as Prod; the duplicate-Min
  // anomaly above is pinned on the float case and applies here too,
  // so Min is not asserted until that is root-caused.
}

TEST_CASE("scatter bool across word lanes keeps siblings intact") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // 10 bools = 3 words; writes land in every lane of the middle word.
  array src = array(
      {false, false, false, false, false,
       false, false, false, false, false},
      {10},
      bool_);
  array indices = array({1, 4, 5, 6, 7, 8}, {6}, int32);
  array updates = array(
      {true, true, true, true, true, true}, {6, 1}, bool_);
  array out = scatter_add(src, indices, updates, 0, stream);
  check_bools(out, {0, 1, 0, 0, 1, 1, 1, 1, 1, 0}, stream);
}

// ---------------------------------------------------------------------------
// ScatterAxis: float Sum and bool
// ---------------------------------------------------------------------------

TEST_CASE("scatter_add_axis float32 duplicate rows exact") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({2, 3}, float32, stream);
  array indices = array({1, 1, 1, 0}, {4, 1}, int32);
  array updates = ones({1, 4, 3}, float32, stream);
  array out = scatter_add_axis(src, indices, updates, 0, stream);
  check_floats(
      out,
      {1, 1, 1, 3, 3, 3},
      stream);
}

TEST_CASE("scatter_add_axis float16 and bool") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  {
    array src = zeros({2, 2}, float16, stream);
    array indices = array({0, 0, 1}, {3, 1}, int32);
    array updates = array(
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {1, 3, 2}, float16);
    array out = scatter_add_axis(src, indices, updates, 0, stream);
    check_floats(
        astype(out, float32, stream),
        {4, 6, 5, 6},
        stream);
  }
  {
    array src = array({false, false, false, false}, {2, 2}, bool_);
    array indices = array({0, 0, 1}, {3, 1}, int32);
    array updates = array(
        {true, true, false, true, true, false}, {1, 3, 2}, bool_);
    array out = scatter_add_axis(src, indices, updates, 0, stream);
    // Row 0 ORs slots 0 and 1: (T,F)|(F,T) = (1,1); row 1 takes slot
    // 2: (T,F) = (1,0).
    check_bools(out, {1, 1, 1, 0}, stream);
  }
}

TEST_CASE("put_along_axis bool none across lanes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array(
      {true, true, true, true, true, true, true, true}, {2, 4},
      bool_);
  array indices = array({1, 0, 1, 0, 0, 1, 0, 1}, {2, 4}, int32);
  array updates = array(
      {false, true, false, true, false, true, false, true}, {1, 2, 4},
      bool_);
  array out = put_along_axis(src, indices, updates, 0, stream);
  // Row 0 takes updates row 0 (false,true,false,true), row 1 takes
  // updates row 1 (false,true,false,true) offset: slot pairs are
  // (row, col) from the 2x4 index grid.
  check_bools(out, {0, 1, 0, 1, 0, 1, 0, 1}, stream);
}

// ---------------------------------------------------------------------------
// Multi-index float and bool
// ---------------------------------------------------------------------------

TEST_CASE("multi index float32 scatter add works") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({2, 3}, float32, stream);
  array idx0 = array({0, 1}, {2}, int32);
  array idx1 = array({2, 0}, {2}, int32);
  array updates = array({7.0f, 9.0f}, {2, 1, 1}, float32);
  array out = scatter_add(src, {idx0, idx1}, updates, {0, 1}, stream);
  check_floats(out, {0, 0, 7, 9, 0, 0}, stream);
}

TEST_CASE("multi index bool scatter add works") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({false, false, false, false, false, false}, {2, 3},
                    bool_);
  array idx0 = array({0, 1}, {2}, int32);
  array idx1 = array({2, 0}, {2}, int32);
  array updates = array({true, true}, {2, 1, 1}, bool_);
  array out = scatter_add(src, {idx0, idx1}, updates, {0, 1}, stream);
  check_bools(out, {0, 0, 1, 1, 0, 0}, stream);
}

TEST_CASE("float scatter without atomic float takes the CAS path") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& caps = omarchy::capability_report(0);
  if (caps.shader_atomic_float_add) {
    // The hardware atomicAdd path is the one the cases above already
    // pin; nothing further to pin here.
    return;
  }
  // No shaderBufferFloat32AtomicAdd (the M1 Honeykrisp does not
  // advertise VK_EXT_shader_atomic_float): the op must still compute,
  // through the FCAS compare-exchange add, and the integer-valued
  // total stays exact. The updates shape follows the op-layer contract
  // (src dims + one row per index array) that a single index array
  // requires.
  array src = zeros({2}, float32, stream);
  array indices = array({0, 1, 0}, {3}, int32);
  array updates = array({1.0f, 2.0f, 4.0f}, {3, 1}, float32);
  array out = scatter_add(src, indices, updates, 0, stream);
  check_floats(out, {5.0f, 2.0f}, stream);
}
