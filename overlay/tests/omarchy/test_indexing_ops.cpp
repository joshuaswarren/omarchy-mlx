// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Wave 5 indexing and scatter: GatherAxis (take_along_axis), Scatter and
// ScatterAxis (put_along_axis / scatter_add_axis), MaskedScatter, and the
// ArgPartition paths. Every implemented mode checks exact host-computed
// values; deliberately unsupported paths pin their named errors.
//
// The wave-5 gates on these paths cited a device defect ("multi-slot
// dispatches only land the first slot write"). Root-caused 2026-09-02:
// the defect was our own address math (the host update-dim table read
// the index-prefix dims of updates, and the axis shaders decomposed t
// with a remainder taken against the wrong extent). The value tests
// below on 3-D axis-1 tensors fail if EITHER half of the shader slot
// decomposition (slot divisor or remainder) is reverted, and the
// multi-slot and update-block scatter tests fail if the host table
// reverts, so the pair cannot silently drift apart again.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
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
  // Retained views share gapped buffers: read logical C order only.
  auto dense = contiguous(value);
  dense.eval();
  sync_gpu(stream);
  REQUIRE_EQ(dense.size(), expected.size());
  const float* values = dense.data<float>();
  for (size_t index = 0; index < expected.size(); ++index) {
    INFO("index ", index, " got ", values[index], " want ", expected[index]);
    CHECK(values[index] == doctest::Approx(expected[index]).epsilon(epsilon));
  }
}

void check_ints(
    array value,
    const std::vector<int>& expected,
    const Stream& stream) {
  auto dense = contiguous(value);
  dense.eval();
  sync_gpu(stream);
  REQUIRE_EQ(dense.size(), expected.size());
  const int32_t* values = dense.data<int32_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    INFO("index ", index, " got ", values[index], " want ", expected[index]);
    CHECK_EQ(values[index], expected[index]);
  }
}

void check_i64(
    array value,
    const std::vector<int64_t>& expected,
    const Stream& stream) {
  auto dense = contiguous(value);
  dense.eval();
  sync_gpu(stream);
  REQUIRE_EQ(dense.size(), expected.size());
  const int64_t* values = dense.data<int64_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    INFO("index ", index, " got ", values[index], " want ", expected[index]);
    CHECK_EQ(values[index], expected[index]);
  }
}

void check_complex(
    array value,
    const std::vector<complex64_t>& expected,
    const Stream& stream) {
  auto dense = contiguous(value);
  dense.eval();
  sync_gpu(stream);
  REQUIRE_EQ(dense.size(), expected.size());
  const complex64_t* values = dense.data<complex64_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    INFO("index ", index);
    CHECK_EQ(values[index].real(), expected[index].real());
    CHECK_EQ(values[index].imag(), expected[index].imag());
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

// Deterministic pseudo-random floats in [0, 1) for reproducible tests.
float lcg(uint64_t& state) {
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<float>((state >> 40) & 0xFFFFFF) /
      static_cast<float>(0x1000000);
}

} // namespace

// ---------------------------------------------------------------------------
// GatherAxis / take_along_axis
// ---------------------------------------------------------------------------

TEST_CASE("take_along_axis gathers along axis 1 with int32 indices") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array(
      {10.0f, 11.0f, 12.0f, 13.0f,
       20.0f, 21.0f, 22.0f, 23.0f,
       30.0f, 31.0f, 32.0f, 33.0f},
      {3, 4},
      float32);
  array indices = array({3, 0, 1, 1, 2, 0}, {3, 2}, int32);
  array out = take_along_axis(src, indices, 1, stream);
  CHECK_EQ(out.dtype(), float32);
  CHECK_EQ(out.shape(), Shape({3, 2}));
  check_floats(out, {13, 10, 21, 21, 32, 30}, stream);
}

TEST_CASE("take_along_axis preserves int64 table payloads") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array(
      {int64_t(9007199254740993LL), int64_t(-9007199254740995LL),
       int64_t(4611686018427387907LL), int64_t(-4611686018427387901LL),
       int64_t(17), int64_t(-19)},
      {2, 3},
      int64);
  array indices = array({2, 0, 1, 2}, {2, 2}, int32);
  check_i64(
      take_along_axis(src, indices, 1, stream),
      {int64_t(4611686018427387907LL), int64_t(9007199254740993LL),
       int64_t(17), int64_t(-19)},
      stream);
}

TEST_CASE("take preserves int64 table payloads") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array(
      {int64_t(9007199254740993LL), int64_t(-9007199254740995LL),
       int64_t(4611686018427387907LL)},
      {3},
      int64);
  array indices = array({2, 0, 1}, {3}, int32);
  check_i64(
      take(src, indices, 0, stream),
      {int64_t(4611686018427387907LL), int64_t(9007199254740993LL),
       int64_t(-9007199254740995LL)},
      stream);
}

TEST_CASE("take_along_axis wraps negative indices") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1.0f, 2.0f, 3.0f}, {1, 3}, float32);
  array indices = array({-1, -2, -3}, {1, 3}, int32);
  array out = take_along_axis(src, indices, 1, stream);
  check_floats(out, {3, 2, 1}, stream);
}

TEST_CASE("take_along_axis gathers along axis 0 with trailing dims") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The axis-0 case has pre dims of zero and post dims of three, the
  // mirror of the axis-1 3-D case below; together they pin both walks
  // of the non-axis stride tables.
  array src = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  array indices = array({1, 0, 1}, {1, 3}, uint32);
  array out = take_along_axis(src, indices, 0, stream);
  CHECK_EQ(out.dtype(), float32);
  check_floats(out, {4, 2, 6}, stream);
}

TEST_CASE("take_along_axis gathers int64 indices along axis 0") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // int64 indices arrive as two little-endian index words per slot;
  // the axis-0 walk must stride the word buffer, not the slot buffer.
  array src = array({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, float32);
  array indices = array({int64_t(0), int64_t(1)}, {1, 2}, int64);
  array out = take_along_axis(src, indices, 0, stream);
  check_floats(out, {5, 8}, stream);
}

TEST_CASE("take_along_axis out-of-range indices read zero (documented)") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Upstream leaves out-of-range gather reads undefined; this backend
  // documents a zero read, the same rule the axis-0 row gather uses.
  array src = array({7.0f, 8.0f, 9.0f}, {1, 3}, float32);
  array indices = array({1, 3, 1}, {1, 3}, int32);
  array out = take_along_axis(src, indices, 1, stream);
  check_floats(out, {8, 0, 8}, stream);
}

TEST_CASE("take_along_axis handles float16 and bfloat16 sources") {
  if (!compute_available()) {
    return;
  }
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16/BF16 storage features.");
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, float32);
  array indices = array({1, 0, 0, 1}, {2, 2}, int32);
  array f16 =
      take_along_axis(astype(src, float16, stream), indices, 1, stream);
  CHECK_EQ(f16.dtype(), float16);
  check_floats(astype(f16, float32, stream), {2, 1, 3, 4}, stream);
  array bf16 =
      take_along_axis(astype(src, bfloat16, stream), indices, 1, stream);
  CHECK_EQ(bf16.dtype(), bfloat16);
  check_floats(astype(bf16, float32, stream), {2, 1, 3, 4}, stream, 1e-2);
}

TEST_CASE("take_along_axis broadcasts the source on non-axis dims") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // src [4, 1] broadcasts against indices [4, 3]: every row reads the
  // same single column value through the explicit-stride path.
  array src = array({1.0f, 2.0f, 3.0f, 4.0f}, {4, 1}, float32);
  array indices = array({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {4, 3}, int32);
  array out = take_along_axis(src, indices, 1, stream);
  CHECK_EQ(out.shape(), Shape({4, 3}));
  check_floats(out, {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4}, stream);
}

TEST_CASE("take_along_axis gathers 3-D axis 1 with trailing dims") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // This is the shape class the wave-5 gate refused: a nonzero axis
  // with trailing dims. It fails if either half of the shader slot
  // decomposition reverts (see the file comment).
  array src = array(
      {100.0f, 101.0f, 102.0f, 103.0f, 110.0f, 111.0f, 112.0f, 113.0f,
       120.0f, 121.0f, 122.0f, 123.0f, 200.0f, 201.0f, 202.0f, 203.0f,
       210.0f, 211.0f, 212.0f, 213.0f, 220.0f, 221.0f, 222.0f, 223.0f},
      {2, 3, 4},
      float32);
  array indices = array(
      {0, 1, 2, 1, 2, 2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1}, {2, 2, 4}, int32);
  array out = take_along_axis(src, indices, 1, stream);
  // out[i][ax][p] = src[i][indices[i][ax][p]][p].
  check_floats(
      out,
      {100, 111, 122, 113, 120, 121, 112, 103,
       210, 211, 202, 203, 200, 201, 212, 213},
      stream);
}

TEST_CASE("take_along_axis transports complex values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array(
      {complex64_t{1, 2}, complex64_t{3, 4}, complex64_t{5, 6},
       complex64_t{7, 8}},
      {2, 2},
      complex64);
  array indices = array({1, 0, 0, 1}, {2, 2}, int32);
  check_complex(
      take_along_axis(src, indices, 1, stream),
      {{3, 4}, {1, 2}, {5, 6}, {7, 8}},
      stream);
}

TEST_CASE("gather uses metadata for four index arrays and complex values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<complex64_t> values;
  for (int i = 0; i < 24; ++i) {
    values.emplace_back(float(i), float(-i));
  }
  array table(values.data(), Shape{2, 3, 2, 2}, complex64);
  array i0 = array({1, 0}, {2}, int32);
  array i1 = array({2, 1}, {2}, int32);
  array i2 = array({0, 1}, {2}, int32);
  array i3 = array({1, 0}, {2}, int32);
  array out = gather(
      table,
      {i0, i1, i2, i3},
      {0, 1, 2, 3},
      {1, 1, 1, 1},
      stream);
  CHECK_EQ(out.shape(), Shape({2, 1, 1, 1, 1}));
  check_complex(out, {{21, -21}, {6, -6}}, stream);
}

TEST_CASE("take_along_axis rejects rank beyond the four-slot table") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The non-axis stride tables cap at four entries; a 6-D gather has
  // five non-axis dims and keeps the named rejection.
  std::vector<float> raw(64, 1.0f);
  array src = array(raw.data(), Shape({2, 2, 2, 2, 2, 2}), float32);
  std::vector<int32_t> idx(32, 0);
  array indices = array(idx.data(), Shape({2, 2, 2, 2, 2, 1}), int32);
  std::string error =
      evaluation_error(take_along_axis(src, indices, 5, stream));
  CHECK(error.find("[omarchy] Take") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Scatter: None / Sum / Prod / Max / Min over one index array
// ---------------------------------------------------------------------------

TEST_CASE("scatter without indices applies update blocks and reducers") {
  if (!compute_available()) return;
  auto stream = gpu_stream();
  std::vector<array> indices;
  std::vector<int> axes;
  check_ints(scatter_max(array(1), indices, array(2), axes, stream), {2}, stream);
  array src({1, 5, -2, 8, 0});
  array updates({3, 4, -7, -1, 6});
  check_ints(scatter(src, indices, updates, axes, stream), {3, 4, -7, -1, 6}, stream);
  check_ints(scatter_add(src, indices, updates, axes, stream), {4, 9, -9, 7, 6}, stream);
  check_ints(scatter_prod(src, indices, updates, axes, stream), {3, 20, 14, -8, 0}, stream);
  check_ints(scatter_max(src, indices, updates, axes, stream), {3, 5, -2, 8, 6}, stream);
  check_ints(scatter_min(src, indices, updates, axes, stream), {1, 4, -7, -1, 0}, stream);
  array matrix({1, 2, 3, 4, 5, 6}, {2, 3});
  array block({10, 20, 30, 40}, {2, 2});
  check_ints(scatter(matrix, indices, block, axes, stream), {10, 20, 3, 30, 40, 6}, stream);
  auto empty = zeros({0}, int32, stream);
  check_ints(scatter_add(src, indices, empty, axes, stream), {1, 5, -2, 8, 0}, stream);
  array real_src({2.0f, 3.0f});
  array real_updates({5.0f, 4.0f});
  check_floats(scatter_add(real_src, indices, real_updates, axes, stream), {7, 7}, stream);
  check_floats(scatter_prod(real_src, indices, real_updates, axes, stream), {10, 12}, stream);
  std::vector<uint8_t> bit_values{0, 1, 1, 0, 1};
  std::vector<uint8_t> update_values{1, 0, 1, 1, 0};
  array bits(bit_values.begin(), Shape{5}, bool_);
  array bit_updates(update_values.begin(), Shape{5}, bool_);
  check_ints(astype(scatter(bits, indices, bit_updates, axes, stream), int32, stream),
             {1, 0, 1, 1, 0}, stream);
  check_ints(astype(scatter_add(bits, indices, bit_updates, axes, stream), int32, stream),
             {1, 1, 1, 1, 1}, stream);
  check_ints(astype(scatter_prod(bits, indices, bit_updates, axes, stream), int32, stream),
             {0, 0, 1, 0, 0}, stream);
}

TEST_CASE("scatter none writes every slot of an eight-slot permutation") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Eight slots, distinct in-range indices, updates as [8, 1] blocks.
  // The wave-5 misroute landed slot t at index[t] + t; every slot must
  // land at index[t] instead.
  array src = zeros({16}, float32, stream);
  array indices = array({3, 7, 0, 5, 1, 6, 2, 4}, {8}, int32);
  array updates = array(
      {100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f},
      {8, 1},
      float32);
  array out = scatter(src, std::vector<array>{indices}, updates, {0}, stream);
  check_floats(
      out,
      {102, 104, 106, 100, 107, 103, 105, 101, 0, 0, 0, 0, 0, 0, 0, 0},
      stream);
}

TEST_CASE("scatter none resolves duplicate indices last-write-wins") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The two-phase rank replay must carry phase 0 ranks into phase 1
  // (the encoder inserts a memory barrier between dispatches) and the
  // highest slot wins, matching the sequential CPU loop.
  array src = zeros({4}, float32, stream);
  array indices = array({2, 2, 2}, {3}, int32);
  array updates = array({10.0f, 20.0f, 30.0f}, {3, 1}, float32);
  array out = scatter(src, std::vector<array>{indices}, updates, {0}, stream);
  check_floats(out, {0, 0, 30, 0}, stream);
}

TEST_CASE("scatter none writes integer data") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1, 2, 3, 4}, {4}, int32);
  array indices = array({0, 2}, {2}, uint32);
  array updates = array({9, 8}, {2, 1}, int32);
  array out = scatter(src, std::vector<array>{indices}, updates, {0}, stream);
  check_ints(out, {9, 2, 8, 4}, stream);
}

TEST_CASE("scatter none writes update blocks across every element") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Two slots, each carrying a [1, 2] update block onto a [3, 2]
  // output: the kernel dispatches over update elements, so both
  // elements of both blocks land. A slot-only dispatch would drop the
  // second element of each block.
  array src = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2}, float32);
  array indices = array({2, 0}, {2}, int32);
  array updates = array({-1.0f, -2.0f, -3.0f, -4.0f}, {2, 1, 2}, float32);
  array out = scatter(src, std::vector<array>{indices}, updates, {0}, stream);
  // Row 1 keeps its source values {3, 4}: neither slot targets it.
  check_floats(out, {-3, -4, 3, 4, -1, -2}, stream);
}

TEST_CASE("scatter none writes 16-bit float copies") {
  if (!compute_available()) {
    return;
  }
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16/BF16 storage features.");
    return;
  }
  Stream stream = gpu_stream();
  array src = array({10.0f, 20.0f, 30.0f, 40.0f}, {4}, float32);
  array indices = array({1, 3}, {2}, int32);
  array updates = array({-1.0f, -3.0f}, {2, 1}, float32);
  array f16 = scatter(
      astype(src, float16, stream),
      std::vector<array>{indices},
      astype(updates, float16, stream),
      {0},
      stream);
  CHECK_EQ(f16.dtype(), float16);
  check_floats(astype(f16, float32, stream), {10, -1, 30, -3}, stream);
  array bf16 = scatter(
      astype(src, bfloat16, stream),
      std::vector<array>{indices},
      astype(updates, bfloat16, stream),
      {0},
      stream);
  check_floats(astype(bf16, float32, stream), {10, -1, 30, -3}, stream, 1e-2);
}

TEST_CASE("scatter max keeps the largest update across duplicates") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({5.0f, 1.0f, 5.0f}, {3}, float32);
  array indices = array({1, 1, 2}, {3}, int32);
  array updates = array({7.0f, 3.0f, 2.0f}, {3, 1}, float32);
  array out =
      scatter_max(src, std::vector<array>{indices}, updates, {0}, stream);
  check_floats(out, {5, 7, 5}, stream);
}

TEST_CASE("scatter max keys negative values correctly") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // All-negative keys exercise the biased monotone key map.
  array src = array({5.0f, -1.0f, 5.0f}, {3}, float32);
  array indices = array({1, 1}, {2}, int32);
  array updates = array({-7.0f, -3.0f}, {2, 1}, float32);
  array out =
      scatter_max(src, std::vector<array>{indices}, updates, {0}, stream);
  check_floats(out, {5, -1, 5}, stream);
}

TEST_CASE("scatter min keeps the smallest update across duplicates") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({5.0f, 9.0f, 5.0f}, {3}, float32);
  array indices = array({1, 1, 2}, {3}, int32);
  array updates = array({2.0f, 8.0f, 7.0f}, {3, 1}, float32);
  array out =
      scatter_min(src, std::vector<array>{indices}, updates, {0}, stream);
  check_floats(out, {5, 2, 5}, stream);
}

TEST_CASE("scatter max handles integer data with duplicates") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({-5, 1, -5}, {3}, int32);
  array indices = array({0, 0, 2}, {3}, int32);
  array updates = array({3, -9, 7}, {3, 1}, int32);
  array out =
      scatter_max(src, std::vector<array>{indices}, updates, {0}, stream);
  check_ints(out, {3, 1, 7}, stream);
}

TEST_CASE("scatter skips out-of-range indices (documented)") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Upstream leaves out-of-range scatter writes undefined; this
  // backend documents the skip, matching ScatterAxis. Negative indices
  // are out of range here; gather wraps them, scatter does not.
  array src = array({1.0f, 2.0f, 3.0f}, {3}, float32);
  array indices = array({-1, -3}, {2}, int32);
  array updates = array({-1.0f, -3.0f}, {2, 1}, float32);
  array out =
      scatter_max(src, std::vector<array>{indices}, updates, {0}, stream);
  check_floats(out, {1, 2, 3}, stream);
}

TEST_CASE("scatter max drops out-of-range slots in mixed updates") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Slot 1 targets index 3 of a 3-wide axis: out of range, skipped.
  array src = array({1.0f, 2.0f, 3.0f}, {3}, float32);
  array indices = array({1, 3, 1}, {3}, int32);
  array updates = array({5.0f, -2.0f, 6.0f}, {3, 1}, float32);
  array out =
      scatter_max(src, std::vector<array>{indices}, updates, {0}, stream);
  check_floats(out, {1, 6, 3}, stream);
}

TEST_CASE("scatter prod multiplies integer updates exactly") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Wrap-around integer multiply is associative and commutative, so
  // the CAS-mul replay is deterministic under any interleaving.
  array src = array({1, 2, 3, 4}, {4}, int32);
  array indices = array({1, 1, 3}, {3}, int32);
  array updates = array({5, 6, 7}, {3, 1}, int32);
  array out =
      scatter_prod(src, std::vector<array>{indices}, updates, {0}, stream);
  check_ints(out, {1, 60, 3, 28}, stream);
}

TEST_CASE("scatter float Sum and Prod compute without atomic float; multi-index stays rejected") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1.0f, 2.0f, 3.0f}, {3}, float32);
  array indices = array({0, 1}, {2}, int32);
  array updates = array({5.0f, 6.0f}, {2, 1}, float32);

  // Float Sum and Prod compute even without VK_EXT_shader_atomic_float:
  // the FCAS compare-exchange replay landed in 959c7a0. A device that
  // lacks the extension (the M1 Honeykrisp) gets the integer-exact
  // values instead of a refusal; this guard engages only there. The
  // atomic-float value coverage lives in omarchy_scatter_determinism_tests.
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_atomic_float_add) {
    array added =
        scatter_add(src, std::vector<array>{indices}, updates, {0}, stream);
    check_floats(added, {6.0f, 8.0f, 3.0f}, stream);
    array product =
        scatter_prod(src, std::vector<array>{indices}, updates, {0}, stream);
    check_floats(product, {5.0f, 12.0f, 3.0f}, stream);
  }

  // Three index arrays run the triple-index kernel: one slot per
  // batch axis, last-write-wins on duplicates like the CPU order.
  array src3d = array(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 2, 2}, float32);
  array i0 = array({0}, {1}, int32);
  array i1 = array({0}, {1}, int32);
  array i2 = array({0}, {1}, int32);
  array one3 = array({9.0f}, {1, 1, 1, 1}, float32);
  array triple = scatter(
      src3d, std::vector<array>{i0, i1, i2}, one3, {0, 1, 2}, stream);
  check_floats(triple, {9.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
               stream);

  // The bool scatter pin and the float16 Sum pin retired: ScatterBool
  // and the FADD float16 kernel are implemented, and their value
  // coverage lives in omarchy_scatter_determinism_tests.
}

TEST_CASE("triple-index float scatter reductions and complex scatter add") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = ones({2, 2, 2}, float32, stream);
  array i0 = array({1, 1}, {2}, int32);
  array i1 = array({0, 0}, {2}, int32);
  array i2 = array({1, 1}, {2}, int32);
  array updates = array({2.0f, 3.0f}, {2, 1, 1, 1}, float32);
  check_floats(
      scatter_add(src, {i0, i1, i2}, updates, {0, 1, 2}, stream),
      {1, 1, 1, 1, 1, 6, 1, 1},
      stream);
  check_floats(
      scatter_prod(src, {i0, i1, i2}, updates, {0, 1, 2}, stream),
      {1, 1, 1, 1, 1, 6, 1, 1},
      stream);

  array complex_src = array(
      {complex64_t{1, 1}, complex64_t{2, 2}, complex64_t{3, 3}},
      {3},
      complex64);
  array complex_indices = array({-2, -2}, {2}, int32);
  array complex_updates = array(
      {complex64_t{4, 5}, complex64_t{6, 7}},
      {2, 1},
      complex64);
  check_complex(
      scatter_add(
          complex_src,
          {complex_indices},
          complex_updates,
          {0},
          stream),
      {{1, 1}, {12, 14}, {3, 3}},
      stream);
}

TEST_CASE("general scatter packs four broadcast index arrays") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({2, 3, 4, 5}, float32, stream);
  Shape index_shape{2, 2, 3};
  std::vector<int> zeros_index(12, 0);
  std::vector<int> axis_one{0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1};
  std::vector<int> axis_two{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  std::vector<int> axis_three(12, 4);
  std::vector<array> indices{
      array(zeros_index.data(), index_shape, int32),
      array(axis_one.data(), index_shape, int32),
      array(axis_two.data(), index_shape, int32),
      array(axis_three.data(), index_shape, int32)};
  array updates = ones({2, 2, 3, 1, 1, 1, 1}, float32, stream);
  std::vector<float> expected(120, 0.0f);
  for (int axis_one = 0; axis_one < 2; ++axis_one) {
    for (int axis_two = 0; axis_two < 3; ++axis_two) {
      expected[axis_one * 20 + axis_two * 5 + 4] = 1.0f;
    }
  }
  check_floats(
      scatter(src, indices, updates, {0, 1, 2, 3}, stream), expected, stream);
}

TEST_CASE("general scatter wraps signed negative indices and skips out of range") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({2, 2, 2, 2}, float32, stream);
  array updates = array({5.0f, 6.0f, 7.0f}, {3, 1, 1, 1, 1}, float32);
  std::vector<float> expected(16, 0.0f);
  expected.front() = 7.0f;
  expected.back() = 5.0f;

  std::vector<array> indices_i32{
      array({-1, -3, 0}, {3}, int32),
      array({-1, 0, 0}, {3}, int32),
      array({-1, 0, 0}, {3}, int32),
      array({-1, 0, 0}, {3}, int32)};
  check_floats(
      scatter(src, indices_i32, updates, {0, 1, 2, 3}, stream),
      expected,
      stream);

  std::vector<array> indices_i64{
      array({int64_t(-1), int64_t(-3), int64_t(0)}, {3}, int64),
      array({int64_t(-1), int64_t(0), int64_t(0)}, {3}, int64),
      array({int64_t(-1), int64_t(0), int64_t(0)}, {3}, int64),
      array({int64_t(-1), int64_t(0), int64_t(0)}, {3}, int64)};
  check_floats(
      scatter(src, indices_i64, updates, {0, 1, 2, 3}, stream),
      expected,
      stream);
}

// ---------------------------------------------------------------------------
// Scatter with two index arrays (multi-index): one index array per axis,
// None / Sum / Max / Min, against hand-computed host references.
// ---------------------------------------------------------------------------

TEST_CASE("multi-index scatter none addresses two axes with update blocks") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // 3x4 output; slots target (2,1) and (0,2); each carries a [1,2]
  // update block that walks the output's trailing-dim strides.
  array src = zeros({3, 4}, float32, stream);
  array rows = array({2, 0}, {2}, int32);
  array cols = array({1, 2}, {2}, int32);
  array updates = array({10.0f, 11.0f, 20.0f, 21.0f}, {2, 1, 2}, float32);
  array out = scatter(src, std::vector<array>{rows, cols}, updates, {0, 1}, stream);
  check_floats(
      out,
      {0, 0, 20, 21, 0, 0, 0, 0, 0, 10, 11, 0},
      stream);
}

TEST_CASE("multi-index scatter none resolves duplicates last write wins") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Slots 0 and 1 both target (1,2); the highest slot must win, and
  // slot 2 lands at (0,3).
  array src = zeros({2, 4}, float32, stream);
  array rows = array({1, 1, 0}, {3}, int32);
  array cols = array({2, 2, 3}, {3}, int32);
  array updates = array({10.0f, 20.0f, 30.0f}, {3, 1, 1}, float32);
  array out = scatter(src, std::vector<array>{rows, cols}, updates, {0, 1}, stream);
  check_floats(out, {0, 0, 0, 30, 0, 0, 20, 0}, stream);
}

TEST_CASE("multi-index scatter max and min keep extreme updates") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Duplicate target (1,1) twice, plus one clean slot at (0,1): pins
  // the src-combine step of the key finalize for both extremes.
  array src = array({3.0f, 1.0f, 5.0f, 7.0f}, {2, 2}, float32);
  array rows = array({0, 1, 1}, {3}, int32);
  array cols = array({1, 1, 1}, {3}, int32);
  array max_upd = array({5.0f, 9.0f, 2.0f}, {3, 1, 1}, float32);
  array max_out =
      scatter_max(src, std::vector<array>{rows, cols}, max_upd, {0, 1}, stream);
  // Duplicate target (1,1): max{7,9,2} = 9; clean slot keeps src * update max.
  check_floats(max_out, {3, 5, 5, 9}, stream);
  array min_out =
      scatter_min(src, std::vector<array>{rows, cols}, max_upd, {0, 1}, stream);
  // Same slots under min: min{7,9,2} = 2; (0,1) keeps src 1.
  check_floats(min_out, {3, 1, 5, 2}, stream);
}

TEST_CASE("multi-index scatter add accumulates integer duplicates") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({2, 3}, int32, stream);
  array rows = array({0, 1, 1}, {3}, int32);
  array cols = array({1, 1, 0}, {3}, int32);
  array updates = array({10, 20, 30}, {3, 1, 1}, int32);
  array out =
      scatter_add(src, std::vector<array>{rows, cols}, updates, {0, 1}, stream);
  check_ints(out, {0, 10, 0, 30, 20, 0}, stream);
}

TEST_CASE("multi-index scatter skips slots with any out-of-range axis") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Slot 1 names column 9 of a 4-wide axis: the whole slot is skipped,
  // no partial write from its in-range row index.
  array src = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 4}, float32);
  array rows = array({0, 1, 1}, {3}, int32);
  array cols = array({1, 9, 3}, {3}, int32);
  array updates = array({-1.0f, -2.0f, -3.0f}, {3, 1, 1}, float32);
  array out = scatter(src, std::vector<array>{rows, cols}, updates, {0, 1}, stream);
  check_floats(out, {1, -1, 3, 4, 5, 6, 7, -3}, stream);
}

TEST_CASE("multi-index scatter writes int32 data through uint32 words") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1, 2, 3, 4, 5, 6}, {2, 3}, int32);
  array rows = array({1, 0}, {2}, int32);
  array cols = array({2, 0}, {2}, int32);
  array updates = array({-7, -8}, {2, 1, 1}, int32);
  array out = scatter(src, std::vector<array>{rows, cols}, updates, {0, 1}, stream);
  check_ints(out, {-8, 2, 3, 4, 5, -7}, stream);
}

TEST_CASE("multi-index scatter decodes int64 index pairs per axis") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // int64 indices read as two little-endian words per slot, for both
  // arrays independently.
  array src = zeros({3, 3}, float32, stream);
  array rows = array({0, 2, 1}, {3}, int64);
  array cols = array({1, 0, 2}, {3}, int64);
  array updates = array({10.0f, 20.0f, 30.0f}, {3, 1, 1}, float32);
  array out = scatter(src, std::vector<array>{rows, cols}, updates, {0, 1}, stream);
  check_floats(out, {0, 10, 0, 0, 0, 30, 20, 0, 0}, stream);
}


// ---------------------------------------------------------------------------
// ScatterAxis: put_along_axis and scatter_add_axis
// ---------------------------------------------------------------------------

TEST_CASE("put_along_axis writes 2-D axis 1 slots") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({2, 3}, float32, stream);
  array indices = array({0, 1, 2, 2, 0, 1}, {2, 3}, int32);
  array values = array({10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f}, {2, 3},
                       float32);
  array out = put_along_axis(src, indices, values, 1, stream);
  check_floats(out, {10, 11, 12, 21, 22, 20}, stream);
}

TEST_CASE("put_along_axis writes 3-D axis 1 with duplicates and skips") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Nonzero axis with pre dims, multiple index rows, and trailing
  // dims: this is the shape class the wave-5 gate refused and it fails
  // if either half of the shader slot decomposition reverts (see the
  // file comment). Negative indices are out of range and skip.
  array src = zeros({2, 3, 4}, float32, stream);
  array indices = array(
      {0, -1, 2, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1}, {2, 2, 4}, int32);
  array values = array(
      {10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f,
       30.0f, 31.0f, 32.0f, 33.0f, 40.0f, 41.0f, 42.0f, 43.0f},
      {2, 2, 4},
      float32);
  array out = put_along_axis(src, indices, values, 1, stream);
  // Duplicates resolve to the highest flat slot (the rank replay);
  // [0][2][1] keeps 21 (slot 5) over the skipped -1 slot, and row 1's
  // collisions land on the later writer.
  check_floats(
      out,
      {10, 0, 22, 23, 0, 0, 0, 13, 20, 21, 12, 0,
       40, 41, 32, 33, 30, 31, 42, 43, 0, 0, 0, 0},
      stream);
}

TEST_CASE("scatter_add_axis accumulates integer duplicates") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // atomicAdd across duplicate indices: row 0 index 1 receives both 2
  // and 3; the result is deterministic because integer addition is
  // associative.
  array src = zeros({2, 3}, int32, stream);
  array indices = array({0, 1, 1, 2, 0, 1}, {2, 3}, int32);
  array values = array({1, 2, 3, 4, 5, 6}, {2, 3}, int32);
  array out = scatter_add_axis(src, indices, values, 1, stream);
  check_ints(out, {1, 5, 0, 5, 6, 4}, stream);
}

TEST_CASE("scatter_add_axis float Sum computes on the atomic and CAS paths") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The axis Sum computes with shaderBufferFloat32AtomicAdd and, since
  // 959c7a0, also through the FCAS compare-exchange replay on devices
  // without the extension (the M1 Honeykrisp). Both paths reach the
  // integer-exact duplicate-index total.
  array src = array({0.0f, 0.0f}, {2}, float32);
  array indices = array({0, 0}, {2}, int32);
  array values = array({1.0f, 2.0f}, {2}, float32);
  array out = scatter_add_axis(src, indices, values, 0, stream);
  check_floats(out, {3.0f, 0.0f}, stream);
}

TEST_CASE("scatter_add_axis accumulates both complex components") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array(
      {complex64_t{1, 1}, complex64_t{2, 2}}, {2}, complex64);
  array indices =
      array({int64_t(-2), int64_t(-2)}, {2}, int64);
  array updates = array(
      {complex64_t{3, 4}, complex64_t{5, 6}}, {2}, complex64);
  check_complex(
      scatter_add_axis(src, indices, updates, 0, stream),
      {{9, 11}, {2, 2}},
      stream);
}

// ---------------------------------------------------------------------------
// MaskedScatter
// ---------------------------------------------------------------------------

TEST_CASE("masked_scatter fills true positions from the source in order") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array dst = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  array mask = array({0, 1, 1, 0, 1, 0}, {2, 3}, bool_);
  array value = array({-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f}, {6},
                      float32);
  array out = masked_scatter(dst, mask, value, stream);
  // Row 0: positions 1, 2 get -1, -2; row 1: position 1 gets -4, the
  // first element of row 1's own source segment.
  check_floats(out, {1, -1, -2, 4, -3, 6}, stream);
}

TEST_CASE("masked_scatter materializes a broadcast source slice") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array dst = zeros({2, 3}, float32, stream);
  array mask = array({1, 0}, {2}, bool_);
  array src = array({2.0f, 3.0f, 4.0f}, {3}, float32);
  check_floats(
      masked_scatter(dst, mask, src, stream),
      {2, 3, 4, 0, 0, 0},
      stream);
}

TEST_CASE("masked_scatter ignores surplus source elements") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array dst = array({0.0f, 0.0f, 0.0f}, {3}, float32);
  array mask = array({1, 0, 1}, {3}, bool_);
  array src = array({9.0f, 8.0f, 7.0f, 6.0f}, {4}, float32);
  array out = masked_scatter(dst, mask, src, stream);
  check_floats(out, {9, 0, 8}, stream);
}

TEST_CASE("masked_scatter broadcasts a scalar value") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array dst = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, float32);
  array mask = array({0, 1, 1, 0}, {2, 2}, bool_);
  array value = array(-7.0f);
  array out = masked_scatter(dst, mask, value, stream);
  // Row 0: position 1 gets -7; row 1: position 0 gets -7.
  check_floats(out, {1, -7, -7, 4}, stream);
}

TEST_CASE("masked_scatter skips writes when the source runs out") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Three true positions but only two source elements: the third write
  // is skipped, matching the upstream Metal kernel behavior.
  array dst = array({1.0f, 2.0f, 3.0f}, {3}, float32);
  array mask = array({1, 1, 1}, {3}, bool_);
  array src = array({-1.0f, -2.0f}, {2}, float32);
  array out = masked_scatter(dst, mask, src, stream);
  check_floats(out, {-1, -2, 3}, stream);
}

TEST_CASE("masked_scatter carries the scan across chunks and rows") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Two rows of 600 elements (three 256-wide chunks each) with every
  // other position true: 300 writes per row must chain through the
  // shared-memory carry across chunk boundaries without dropping or
  // reordering a single assignment.
  const int rows = 2;
  const int cols = 600;
  std::vector<float> dst_data(rows * cols, 0.0f);
  array dst = array(dst_data.data(), Shape({rows, cols}), float32);
  std::vector<unsigned char> bytes(rows * cols, 0);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; c += 2) {
      bytes[r * cols + c] = 1;
    }
  }
  array mask = array(bytes.data(), Shape({rows, cols}), bool_);
  // With a 2-D mask the op layer supplies a 1-D value and expands a
  // leading batch dim, so the primitive sees ONE row: the j-th true
  // position in flat mask order consumes value[j].
  std::vector<float> value_data(rows * cols, 0.0f);
  for (int k = 0; k < rows * cols; ++k) {
    value_data[k] = -static_cast<float>(k);
  }
  array value = array(value_data.data(), Shape({rows * cols}), float32);
  array out = masked_scatter(dst, mask, value, stream);
  std::vector<float> expected(rows * cols, 0.0f);
  for (int k = 0; k < rows * cols / 2; ++k) {
    expected[2 * k] = -static_cast<float>(k);
  }
  check_floats(out, expected, stream);
}

TEST_CASE("masked_scatter supports int32 and float16 data") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array dst = array({1, 2, 3, 4}, {4}, int32);
  array mask = array({1, 0, 0, 1}, {4}, bool_);
  array src = array({-1, -4}, {2}, int32);
  array out = masked_scatter(dst, mask, src, stream);
  check_ints(out, {-1, 2, 3, -4}, stream);

  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 storage features.");
    return;
  }
  array f16dst = astype(array({1.0f, 2.0f}, {2}, float32), float16, stream);
  array f16mask = array({0, 1}, {2}, bool_);
  array f16src = astype(array({5.0f}, {1}, float32), float16, stream);
  array f16out = masked_scatter(f16dst, f16mask, f16src, stream);
  check_floats(astype(f16out, float32, stream), {1, 5}, stream);
}

TEST_CASE("masked_scatter broadcast mask fills whole leading rows") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // A [2] mask over a [2, 2] array masks leading rows (upstream expands
  // it to [2, 1] and broadcasts), and a scalar value fills every masked
  // position.
  array dst = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, float32);
  array mask = array({1, 0}, {2}, bool_);
  array value = array(-7.0f);
  check_floats(masked_scatter(dst, mask, value, stream), {-7, -7, 3, 4}, stream);
}

// ---------------------------------------------------------------------------
// Wide-row ArgPartition
// ---------------------------------------------------------------------------


TEST_CASE("argpartition small rows partition exactly on the bitonic path") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // kth = 0: the smallest value lands at position 0.
  {
    std::vector<float> row = {3.0f, -1.0f, 2.0f, -1.0f, 0.0f};
    array a = array(row.data(), Shape({1, 5}), float32);
    array out = argpartition(a, 0, -1, stream);
    out.eval();
    sync_gpu(stream);
    const uint32_t* indices = out.data<uint32_t>();
    CHECK_EQ(row[indices[0]], -1.0f);
  }

  // kth = N-1: the largest value lands at position N-1.
  {
    std::vector<float> row = {3.0f, -1.0f, 2.0f, 9.0f, 0.0f};
    array a = array(row.data(), Shape({1, 5}), float32);
    array out = argpartition(a, 4, -1, stream);
    out.eval();
    sync_gpu(stream);
    const uint32_t* indices = out.data<uint32_t>();
    CHECK_EQ(row[indices[4]], 9.0f);
  }

  // All-equal rows: every position holds the tied value.
  {
    std::vector<float> row(300, 1.5f);
    array a = array(row.data(), Shape({1, 300}), float32);
    array out = argpartition(a, 150, -1, stream);
    out.eval();
    sync_gpu(stream);
    const uint32_t* indices = out.data<uint32_t>();
    for (int i = 0; i < 300; ++i) {
      CHECK_EQ(row[indices[i]], 1.5f);
    }
  }
}

TEST_CASE("argpartition partitions over a non-suffix axis") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a = array({4.0f, 1.0f, 2.0f, 3.0f}, {2, 2}, float32);
  // kth = 1 along axis 0: the full-sort redirect makes the per-column
  // index output equal argsort: column 0 sorts {4, 2} to {1, 0} and
  // column 1 sorts {1, 3} to {0, 1}.
  array out = argpartition(a, 1, 0, stream);
  out.eval();
  sync_gpu(stream);
  const uint32_t* v = out.data<uint32_t>();
  CHECK_EQ(v[0], 1u);
  CHECK_EQ(v[1], 0u);
  CHECK_EQ(v[2], 0u);
  CHECK_EQ(v[3], 1u);
}

TEST_CASE("argpartition wide rows partition exactly") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Any row length now sorts: the wide-row sort path carries
  // ArgPartition's full-sort redirect past the old 1024 cap, so this
  // value test replaced the refusal pin in the same change.
  std::vector<float> row(2000);
  for (int i = 0; i < 2000; ++i) {
    row[i] = float((i * 48271) % 2009) - 1000.0f;
  }
  array a = array(row.data(), Shape({1, 2000}), float32);
  // kth = 0: position 0 names the minimum's index.
  {
    array out = argpartition(a, 0, -1, stream);
    out.eval();
    sync_gpu(stream);
    const uint32_t* indices = out.data<uint32_t>();
    float smallest = *std::min_element(row.begin(), row.end());
    CHECK_EQ(row[indices[0]], smallest);
  }
  // kth = N-1: position N-1 names the maximum's index.
  {
    array out = argpartition(a, 1999, -1, stream);
    out.eval();
    sync_gpu(stream);
    const uint32_t* indices = out.data<uint32_t>();
    float largest = *std::max_element(row.begin(), row.end());
    CHECK_EQ(row[indices[1999]], largest);
  }
  // kth = 1000: the partition property must hold exactly, like the
  // CPU reference: nothing right of kth is smaller than value[kth],
  // nothing left of kth is larger.
  {
    array out = argpartition(a, 1000, -1, stream);
    out.eval();
    sync_gpu(stream);
    const uint32_t* indices = out.data<uint32_t>();
    float pivot = row[indices[1000]];
    for (int i = 0; i < 1000; ++i) {
      CHECK(row[indices[i]] <= pivot);
    }
    for (int i = 1001; i < 2000; ++i) {
      CHECK(row[indices[i]] >= pivot);
    }
  }
}

// ---------------------------------------------------------------------------
// Attn128: small-k wide-row value Partition (the mx.topk route)
// ---------------------------------------------------------------------------
//
// partition(a, kth, -1) on long float rows with a small tail k routes to a
// one-dispatch radix selection instead of the pad-chunk-merge sort. The
// tail k values must be bit-identical to the sort path's for finite rows;
// ties (NaN, the two zeros) resolve by the order-preserving sign-flip key
// map the kernel documents, which the references below replicate exactly.

namespace {

uint32_t host_key(uint32_t bits) {
  return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}
uint32_t host_unkey(uint32_t key) {
  return (key & 0x80000000u) != 0u ? (key ^ 0x80000000u) : ~key;
}
uint32_t float_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return bits;
}
float bits_float(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}
// Round-to-nearest-even f32 -> bf16 (matches the shader's RNE bf16_store).
uint16_t f32_to_bf16_rne(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  if (std::isnan(value)) {
    uint32_t nan_payload = (bits >> 16) | 0x40u;
    return static_cast<uint16_t>(nan_payload);
  }
  // cast.comp's RNE: add 0x7fff plus the kept lsb, then drop 16 bits.
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
// Exact f16 widening; bf16 is a 16-bit left shift.
float f16_bits_to_float(uint16_t half) {
  uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
  uint32_t exp = (half >> 10) & 0x1fu;
  uint32_t man = half & 0x3ffu;
  uint32_t bits;
  if (exp == 0u && man == 0u) {
    bits = sign;
  } else if (exp == 0u) {
    int e = -1;
    uint32_t m = man;
    do {
      ++e;
      m <<= 1;
    } while ((m & 0x400u) == 0u);
    bits = sign | ((127u - 15u - static_cast<uint32_t>(e)) << 23) |
        ((m & 0x3ffu) << 13);
  } else if (exp == 31u) {
    bits = sign | 0x7f800000u | (man << 13);
  } else {
    bits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
  }
  return bits_float(bits);
}

uint64_t smallk_dispatches(
    const std::function<array()>& step,
    const Stream& stream) {
  array out = step();
  out.eval();
  sync_gpu(stream);
  uint64_t before = omarchy::trace::counters().vk_compute_dispatches.load();
  out = step();
  out.eval();
  sync_gpu(stream);
  return omarchy::trace::counters().vk_compute_dispatches.load() - before;
}

// The documented tie rule, on the host: ascending key order.
std::vector<uint32_t> host_tail_bits(
    const std::vector<uint32_t>& widened_bits,
    int k) {
  std::vector<uint32_t> keys(widened_bits.size());
  for (size_t i = 0; i < widened_bits.size(); ++i) {
    keys[i] = host_key(widened_bits[i]);
  }
  std::vector<size_t> order(widened_bits.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(
      order.begin(), order.end(), [&](size_t a, size_t b) {
        return keys[a] < keys[b];
      });
  std::vector<uint32_t> tail;
  tail.reserve(k);
  for (int i = 0; i < k; ++i) {
    tail.push_back(widened_bits[order[order.size() - static_cast<size_t>(k) + i]]);
  }
  // tail is in ascending key order; the route stores exactly these bits.
  return tail;
}

void require_tail_bits(
    const array& out,
    int kth,
    const std::vector<uint32_t>& expected_tail_bits,
    const Stream& stream,
    const char* what) {
  array dense = contiguous(out);
  dense.eval();
  sync_gpu(stream);
  const uint32_t* words = dense.data<uint32_t>();
  for (int i = 0; i < kth; ++i) {
    // Leading region: the threshold fill, always <= the first tail value.
    CHECK_MESSAGE(
        host_key(words[i]) <= host_key(words[kth]),
        what,
        " leading slot ",
        i,
        " exceeds the threshold");
  }
  for (size_t i = 0; i < expected_tail_bits.size(); ++i) {
    CHECK_MESSAGE(
        words[kth + i] == expected_tail_bits[i],
        what,
        " tail slot ",
        i,
        ": got ",
        words[kth + i],
        " want ",
        expected_tail_bits[i]);
  }
}

std::vector<float> smallk_pattern(size_t count, uint32_t seed) {
  std::vector<float> values;
  values.reserve(count);
  uint32_t state = seed;
  for (size_t index = 0; index < count; ++index) {
    state = state * 1664525u + 1013904223u;
    values.push_back(
        static_cast<float>(static_cast<double>(state % 20000u) / 10000.0) -
        1.0f);
  }
  return values;
}

} // namespace

TEST_CASE("wide-row small-k partition selects the exact tail in one dispatch") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int n = 4096;
  auto row = smallk_pattern(n, 101);
  for (int k : {1, 50, 256}) {
    CAPTURE(k);
    array a = array(row.begin(), Shape{1, n}, float32);
    int kth = n - k;
    uint64_t dispatches = smallk_dispatches(
        [&] { return partition(a, kth, -1, stream); }, stream);
    CHECK_EQ(dispatches, 1);
    auto widened = std::vector<uint32_t>(n);
    for (int i = 0; i < n; ++i) {
      widened[i] = float_bits(row[i]);
    }
    require_tail_bits(
        partition(a, kth, -1, stream), kth, host_tail_bits(widened, k), stream,
        "f32 tail");
  }
}

TEST_CASE("wide-row small-k partition covers rows, ties, and 16-bit dtypes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int n = 2048;
  const int k = 5;
  const int kth = n - k;

  // Three rows, f32, with heavy duplicates and the k boundary inside a tie
  // run, plus infinities.
  std::vector<float> rows;
  for (int r = 0; r < 3; ++r) {
    auto row = smallk_pattern(n, 202 + r);
    for (int i = 0; i < k + 3; ++i) {
      row[i] = 7.5f; // duplicate values straddling the boundary
    }
    row[10] = std::numeric_limits<float>::infinity();
    rows.insert(rows.end(), row.begin(), row.end());
  }
  array a = array(rows.begin(), Shape{3, n}, float32);
  uint64_t dispatches = smallk_dispatches(
      [&] { return partition(a, kth, -1, stream); }, stream);
  CHECK_EQ(dispatches, 1);
  std::vector<uint32_t> expected;
  for (int r = 0; r < 3; ++r) {
    std::vector<uint32_t> widened;
    widened.reserve(n);
    for (int i = 0; i < n; ++i) {
      widened.push_back(float_bits(rows[r * n + i]));
    }
    auto tail = host_tail_bits(widened, k);
    expected.insert(expected.end(), tail.begin(), tail.end());
  }
  {
    array dense = contiguous(partition(a, kth, -1, stream));
    dense.eval();
    sync_gpu(stream);
    const uint32_t* words = dense.data<uint32_t>();
    for (int r = 0; r < 3; ++r) {
      for (int i = 0; i < kth; ++i) {
        CHECK(host_key(words[r * n + i]) <= host_key(words[r * n + kth]));
      }
      for (int i = 0; i < k; ++i) {
        CHECK_EQ(words[r * n + kth + i], expected[r * k + i]);
      }
    }
  }

  // 16-bit rows (bf16 and f16). Input comes through astype - never
  // array(uint16_t*, shape, bfloat16), whose init() VALUE-converts each
  // uint16 (0xBF00 became 48896); that ctor was the whole 2026-09-30
  // "bf16 defect". The reference reads back the device-held input words,
  // widens them exactly, stable-sorts by the documented key and takes the
  // INPUT words at the top-k indices, so no host narrowing can disagree.
  // 16-bit rows take the one-dispatch selection on every part (verified
  // on G14C and G13C; the dispatches == 1 asserts below are the route
  // proof per chip).
  constexpr bool kSixteenBitSelects = true;
  std::vector<float> rows16;
  for (int r = 0; r < 2; ++r) {
    auto row = smallk_pattern(n, 303 + r);
    rows16.insert(rows16.end(), row.begin(), row.end());
  }
  array rows16_f32 = array(rows16.begin(), Shape{2, n}, float32);
  for (Dtype dtype : {bfloat16, float16}) {
    CAPTURE(dtype == bfloat16 ? "bf16" : "f16");
    array x = contiguous(astype(rows16_f32, dtype, stream), false, stream);
    x.eval();
    sync_gpu(stream);
    const uint16_t* in_words = x.data<uint16_t>();
    auto widen = [&](uint16_t w) {
      return dtype == bfloat16 ? static_cast<uint32_t>(w) << 16
                               : float_bits(f16_bits_to_float(w));
    };
    uint64_t dispatches = smallk_dispatches(
        [&] { return partition(x, kth, -1, stream); }, stream);
    if (kSixteenBitSelects) {
      CHECK_EQ(dispatches, 1);
    } else {
      CHECK(dispatches > 1);
    }
    array dense = contiguous(partition(x, kth, -1, stream));
    dense.eval();
    sync_gpu(stream);
    const uint16_t* words = dense.data<uint16_t>();
    for (int r = 0; r < 2; ++r) {
      std::vector<uint32_t> keys(n);
      for (int i = 0; i < n; ++i) {
        keys[i] = host_key(widen(in_words[r * n + i]));
      }
      std::vector<int> order(n);
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return keys[a] < keys[b];
      });
      for (int i = 0; i < k; ++i) {
        CHECK_MESSAGE(
            words[r * n + kth + i] == in_words[r * n + order[n - k + i]],
            "tail r=", r, " i=", i, " got=", words[r * n + kth + i],
            " want=", in_words[r * n + order[n - k + i]]);
      }
      uint32_t first = host_key(widen(words[r * n + kth]));
      for (int i = 0; i < kth; ++i) {
        CHECK(host_key(widen(words[r * n + i])) <= first);
      }
    }
  }
}

TEST_CASE("wide-row small-k partition places NaNs and zero signs by the key map") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int n = 2048;
  const int k = 6;
  const int kth = n - k;
  auto row = smallk_pattern(n, 404);
  row[0] = std::numeric_limits<float>::quiet_NaN();
  row[1] = -0.0f;
  row[2] = 0.0f;
  row[3] = std::numeric_limits<float>::infinity();
  array a = array(row.begin(), Shape{1, n}, float32);
  uint64_t dispatches = smallk_dispatches(
      [&] { return partition(a, kth, -1, stream); }, stream);
  CHECK_EQ(dispatches, 1);
  std::vector<uint32_t> widened(n);
  for (int i = 0; i < n; ++i) {
    widened[i] = float_bits(row[i]);
  }
  // NaN keys rank above +inf, the two zeros separate (-0.0 below +0.0);
  // the tail stores the selected values' exact bits.
  require_tail_bits(
      partition(a, kth, -1, stream), kth, host_tail_bits(widened, k), stream,
      "NaN tail");
}

TEST_CASE("wide-row shapes outside the small-k selection keep the full sort") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int n = 2048;
  auto row = smallk_pattern(n, 505);

  // kth = 0 asks for the whole row (k = n > 256): the sort path answers and
  // sorts the row exactly.
  {
    array a = array(row.begin(), Shape{1, n}, float32);
    uint64_t dispatches = smallk_dispatches(
        [&] { return partition(a, 0, -1, stream); }, stream);
    CHECK(dispatches > 1);
    array dense = contiguous(partition(a, 0, -1, stream));
    dense.eval();
    sync_gpu(stream);
    const uint32_t* words = dense.data<uint32_t>();
    auto sorted = row;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < n; ++i) {
      CHECK_EQ(words[i], float_bits(sorted[i]));
    }
  }

  // More than 256 rows also keeps the sort path, with its exact output.
  {
    std::vector<float> grid;
    for (int r = 0; r < 257; ++r) {
      auto row_r = smallk_pattern(n, 606 + r);
      grid.insert(grid.end(), row_r.begin(), row_r.end());
    }
    array a = array(grid.begin(), Shape{257, n}, float32);
    uint64_t dispatches = smallk_dispatches(
        [&] { return partition(a, n - 4, -1, stream); }, stream);
    CHECK(dispatches > 1);
    array dense = contiguous(partition(a, n - 4, -1, stream));
    dense.eval();
    sync_gpu(stream);
    const uint32_t* words = dense.data<uint32_t>();
    for (int r = 0; r < 257; ++r) {
      auto row_r = smallk_pattern(n, 606 + r);
      auto sorted = row_r;
      std::sort(sorted.begin(), sorted.end());
      for (int i = 0; i < 4; ++i) {
        CHECK_EQ(
            words[r * n + n - 4 + i], float_bits(sorted[n - 4 + i]));
      }
    }
  }

  // Integer rows keep the sort path too (the selection route is float
  // only), with the exact sorted row.
  {
    std::vector<int32_t> irow(n);
    for (int i = 0; i < n; ++i) {
      irow[i] = static_cast<int32_t>((i * 2654435761u) % 1000u) - 500;
    }
    array a = array(irow.begin(), Shape{1, n}, int32);
    uint64_t dispatches = smallk_dispatches(
        [&] { return partition(a, n - 8, -1, stream); }, stream);
    CHECK(dispatches > 1);
    array dense = contiguous(partition(a, n - 8, -1, stream));
    dense.eval();
    sync_gpu(stream);
    const int32_t* words = dense.data<int32_t>();
    auto sorted = irow;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < 8; ++i) {
      CHECK_EQ(words[n - 8 + i], sorted[n - 8 + i]);
    }
  }
}
