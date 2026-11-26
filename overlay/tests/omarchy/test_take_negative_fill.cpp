// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Regression tests for two silent wrong-value defects found by the
// 2026-09-02 upstream suite run (receipts/2026-09-02-upstream-suite-coverage.md
// defects W3 and W4; the catching upstream cases are ops_tests "test take"
// line 2300 and "test full_like" line 3173).
//
// W3: the axis-0 row gather (shaders/gather_rows.comp) treated a negative
// index as out-of-range and wrote a zero row. Negative indices now wrap
// like upstream offset_neg_idx: -1 reads the last row and -axis_size
// reads row 0. Only an index outside [-axis_size, axis_size) keeps the
// documented zero fill; upstream leaves such reads undefined, and the
// zero row is this backend's documented deviation. Reverting the wrap
// fails every negative-index check below.
//
// W4: the flat Vector/dtype-converting copy in copy.cpp read a stride-0
// broadcast view past its one-element buffer, so a fill with an array
// value and a dtype cast returned [v, 0, 0, ...] (and could segfault on
// the out-of-bounds read). copy_gpu now materializes any input whose
// data_size() differs from size() before a flat copy; reverting that
// condition fails the full_like checks below.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <iostream>
#include <functional>
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

void check_ints(array value, const std::vector<int>& expected) {
  value.eval();
  REQUIRE_EQ(value.size(), expected.size());
  const int32_t* values = value.data<int32_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

void check_floats(
    array value,
    const std::vector<float>& expected,
    double epsilon = 1e-5) {
  value.eval();
  REQUIRE_EQ(value.size(), expected.size());
  const float* values = value.data<float>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK(values[index] == doctest::Approx(expected[index]).epsilon(epsilon));
  }
}

void expect_named_error(const std::function<void()>& body, const char* text) {
  bool raised = false;
  try {
    body();
  } catch (const std::exception& error) {
    raised = true;
    CHECK(std::string(error.what()).find(text) != std::string::npos);
  }
  CHECK(raised);
}

} // namespace

TEST_CASE("take wraps negative indices along axis 0 for int32") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a = array({1, 2, 3, 4}, {2, 2}, int32);

  // Scalar negative index: the upstream case at ops_tests line 2300.
  check_ints(take(a, array(-1), 0, stream), {3, 4});
  // The boundary index -axis_size wraps to row 0.
  check_ints(take(a, array(-2), 0, stream), {1, 2});
  // Array negative index.
  check_ints(take(a, array({-1}, int32), 0, stream), {3, 4});
  // Mixed positive and negative.
  check_ints(take(a, array({0, -1}, int32), 0, stream), {1, 2, 3, 4});
  check_ints(take(a, array({-2, -1}, int32), 0, stream), {1, 2, 3, 4});
  check_ints(take(a, array({-1, 0}, int32), 0, stream), {3, 4, 1, 2});

  // Genuinely out of range. Upstream has no bounds check (offset_neg_idx
  // alone), so those reads are undefined there; the documented omarchy
  // deviation is a zero row and must keep applying after the wrap.
  check_ints(take(a, array({2}, int32), 0, stream), {0, 0});
  check_ints(take(a, array({-3}, int32), 0, stream), {0, 0});
  sync_gpu(stream);
}

TEST_CASE("take wraps negative indices for float32 and float16 tables") {
  if (!compute_available()) {
    return;
  }
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 storage features.");
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, float32);

  array f32 = take(src, array({-1}, int32), 0, stream);
  CHECK_EQ(f32.dtype(), float32);
  check_floats(f32, {3, 4});
  check_floats(take(src, array(-2), 0, stream), {1, 2});

  array f16 = take(astype(src, float16, stream), array({0, -1}, int32), 0, stream);
  CHECK_EQ(f16.dtype(), float16);
  check_floats(astype(f16, float32, stream), {1, 2, 3, 4});
  sync_gpu(stream);
}

TEST_CASE("take wraps negative int64 indices") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a = array({1, 2, 3, 4}, {2, 2}, int32);

  check_ints(take(a, array({-1}, int64), 0, stream), {3, 4});
  check_ints(take(a, array({-2, -1}, int64), 0, stream), {1, 2, 3, 4});
  // Below -2^31 an int64 index has a high word that is neither 0 nor
  // 0xFFFFFFFF; it must stay out of range rather than wrap by the low
  // word alone (which would land at table_rows - 1).
  std::initializer_list<int64_t> far_below = {
      static_cast<int64_t>(-4294967297LL)};
  check_ints(take(a, array(far_below, {1}, int64), 0, stream), {0, 0});
  // Above 2^31 likewise stays out of range.
  std::initializer_list<int64_t> far_above = {
      static_cast<int64_t>(4294967297LL)};
  check_ints(take(a, array(far_above, {1}, int64), 0, stream), {0, 0});
  sync_gpu(stream);
}

TEST_CASE("take along non-suffix axes keeps the zero-fill rule") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a = array({1, 2, 3, 4}, {2, 2}, int32);
  // Axis 1 (and negative -1 on a 2-D array) gathers columns; an
  // out-of-range index reads zero, the same documented deviation the
  // axis-0 row gather makes.
  check_ints(take(a, array({0}, int32), 1, stream), {1, 3});
  check_ints(take(a, array({0}, int32), -1, stream), {1, 3});
  check_ints(take(a, array({7}, int32), 1, stream), {0, 0});
  sync_gpu(stream);
}

TEST_CASE("take_along_axis wraps negative indices at the -axis_size boundary") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // gather_axis.comp already wrapped negatives when take_along_axis
  // landed; pin the family boundary so both gather shaders stay
  // consistent. Indices must already be dense so the GatherAxis
  // host's data_size == size check passes and the shader actually
  // runs the wrap code.
  array src = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  array indices =
      array({-3, -2, -1, -1, -2, -3}, {2, 3}, int32);
  array out = take_along_axis(src, indices, 1, stream);
  CHECK_EQ(out.shape(), Shape({2, 3}));
  check_floats(out, {1, 2, 3, 6, 5, 4});
  sync_gpu(stream);
}

TEST_CASE("full_like array fill with dtype cast fills every element") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  set_default_device(Device::gpu);

  // The exact upstream case: int16 base, float32 array value, float16
  // out. The old behavior returned [7.5, 0, 0].
  array base_i16 = array({1, 2, 3}, {3}, int16);
  array to_f16 = full_like(base_i16, array(7.5f), float16, stream);
  CHECK_EQ(to_f16.dtype(), float16);
  check_floats(astype(to_f16, float32, stream), {7.5, 7.5, 7.5});

  // Float to int: the whole array truncates to the fill value.
  array base_f32 = array({1.0f, 2.0f, 3.0f}, {3}, float32);
  array to_i32 = full_like(base_f32, array(2.9f), int32, stream);
  CHECK_EQ(to_i32.dtype(), int32);
  check_ints(to_i32, {2, 2, 2});

  // Int to float with a multi-dimensional base larger than one
  // workgroup row.
  array base_i32 = astype(
      array({1, 2, 3, 4, 5, 6}, {2, 3}, int32), int32, stream);
  array wide = full_like(base_i32, array(7.5f), float16, stream);
  CHECK_EQ(wide.dtype(), float16);
  CHECK_EQ(wide.shape(), Shape({2, 3}));
  check_floats(
      astype(reshape(wide, {6}, stream), float32, stream),
      {7.5, 7.5, 7.5, 7.5, 7.5, 7.5});

  // Widening int to float.
  array to_f32 = full_like(base_i16, array(5, int32), float32, stream);
  CHECK_EQ(to_f32.dtype(), float32);
  check_floats(to_f32, {5, 5, 5});

  // A thousand elements crosses several workgroups; every element must
  // be filled, not just the first.
  Shape big_shape({1000});
  array big = full(Shape(big_shape), array(7.5f), float16, stream);
  array big_f32 = astype(big, float32, stream);
  big_f32.eval();
  sync_gpu(stream);
  REQUIRE_EQ(big_f32.size(), 1000u);
  const float* big_values = big_f32.data<float>();
  for (size_t index = 0; index < big_f32.size(); ++index) {
    CHECK(big_values[index] == doctest::Approx(7.5).epsilon(1e-5));
  }
}

TEST_CASE("full with an array value and dtype cast fills every element") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // full and full_like share the astype-then-Full graph, so the same
  // flat-copy defect expressed through full in C++.
  array out = full(Shape({3}), array(7.5f), float16, stream);
  CHECK_EQ(out.dtype(), float16);
  check_floats(astype(out, float32, stream), {7.5, 7.5, 7.5});
  sync_gpu(stream);
}

TEST_CASE("full_like same-dtype array fill still fills every element") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The same-dtype path never casts; guard it so the materialization
  // change cannot regress the working scalar-fill route.
  array base_f32 = array({1.0f, 2.0f, 3.0f}, {3}, float32);
  array same = full_like(base_f32, array(7.5f), float32, stream);
  CHECK_EQ(same.dtype(), float32);
  check_floats(same, {7.5, 7.5, 7.5});
  sync_gpu(stream);
}

// Rank collapse: upstream MLX allows any table rank in take; the
// kernel packs four dims, so a dense row-contiguous rank-5+ table
// folds to [prefix, gathered, suffix] on the host. The Qwen2-VL
// vision path gathers a [1,1,T,16,80] float16 table along axis 2
// (A9 row: "Take rank ... shape=[1,1,2016,16,80]"). Before the
// collapse these refused by name.
TEST_CASE("take collapses rank-5 float16, bool, and rank-6 int tables") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Qwen2-VL pattern, scaled down: table [6,16,80] f16, axis 0.
  // 1-D index keeps the assertion surface independent of the MLX
  // array(init_list, shape, dtype) overload that over-resolved the
  // multi-leading-dim initializer used in earlier drafts.
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 storage features.");
    return;
  }
  std::vector<float> vl_values(6 * 16 * 80);
  for (size_t i = 0; i < vl_values.size(); ++i) {
    vl_values[i] = static_cast<float>(i % 97) * 0.25f;
  }
  array vl(vl_values.begin(), Shape({6, 16, 80}), float32);
  array vl16 = astype(vl, float16, stream);
  std::vector<int32_t> f16_idx{5, 0, 3};
  array idx = array(f16_idx.begin(), Shape{3}, int32);
  array out = take(vl16, idx, 0, stream);
  CHECK_EQ(out.dtype(), float16);
  CHECK_EQ(out.shape(), Shape({3, 16, 80}));
  std::vector<float> expected;
  expected.reserve(3 * 16 * 80);
  for (int t : {5, 0, 3}) {
    for (int r = 0; r < 16; ++r) {
      for (int c = 0; c < 80; ++c) {
        size_t off = (static_cast<size_t>(t) * 16 + r) * 80 + c;
        expected.push_back(vl_values[off]);
      }
    }
  }
  check_floats(astype(out, float32, stream), expected);

  // bool rank 5, axis 3: prefix [2,3,4] folds to 24, suffix [6,7].
  std::vector<uint8_t> bvals(2 * 3 * 4 * 5 * 6 * 7);
  for (size_t i = 0; i < bvals.size(); ++i) {
    bvals[i] = (i % 3 == 0) ? uint8_t{1} : uint8_t{0};
  }
  array b5(bvals.begin(), Shape({2, 3, 4, 5, 6, 7}), bool_);
  std::vector<int32_t> bidx_v{4, 0, 2, 1};
  array bidx = array(bidx_v.begin(), Shape{4}, int32);
  array bout = take(b5, bidx, 3, stream);
  CHECK_EQ(bout.dtype(), bool_);
  CHECK_EQ(bout.shape(), Shape({2, 3, 4, 4, 6, 7}));
  auto bdense = contiguous(bout);
  bdense.eval();
  sync_gpu(stream);
  REQUIRE_EQ(bdense.size(), 4u * 2 * 3 * 4 * 6 * 7);
  const uint8_t* bv = bdense.data<uint8_t>();
  size_t at = 0;
  // Table [2,3,4,5,6,7] axis 3, row-contiguous strides s5=1,
  // s4=7, s3=42, s2=210, s1=1260, s0=3780. Prefix flat p (0..23)
  // stride s2=210; suffix t (0..41) stride 1.
  for (int k : {4, 0, 2, 1}) {
    for (int pflat = 0; pflat < 24; ++pflat) {
      for (int sfx = 0; sfx < 42; ++sfx) {
        size_t off = static_cast<size_t>(pflat) * 210 + k * 42 + sfx;
        INFO("bool rank5 k=", k, " flat=", at, " got ", int(bv[at]),
             " want ", int(bvals[off]));
        CHECK_EQ(int(bv[at]), int(bvals[off]));
        ++at;
      }
    }
  }
  sync_gpu(stream);

  // int32 rank 6, axis 3: prefix [2,2,2] folds to 8, suffix [2,2] to 4.
  std::vector<int> i6vals(64);
  for (int i = 0; i < 64; ++i) {
    i6vals[i] = i;
  }
  array i6 = array(i6vals.begin(), Shape({2, 2, 2, 2, 2, 2}), int32);
  std::vector<int32_t> iidx_v{1, 0};
  array iidx = array(iidx_v.begin(), Shape{2}, int32);
  array iout = take(i6, iidx, 3, stream);
  CHECK_EQ(iout.dtype(), int32);
  CHECK_EQ(iout.shape(), Shape({2, 2, 2, 2, 2, 2}));
  std::vector<int> iexp;
  for (int k : {1, 0}) {
    for (int pflat = 0; pflat < 8; ++pflat) {
      for (int sfx = 0; sfx < 4; ++sfx) {
        iexp.push_back(pflat * 8 + k * 4 + sfx);
      }
    }
  }
  check_ints(iout, iexp);
  sync_gpu(stream);
}
