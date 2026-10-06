// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Regression tests for the bool Take / TakeAlongAxis path the Omarchy
// Vulkan backend shipped without. oMLX DeepSeek-OCR row A10
// (jw16 /var/tmp/omlxmodal-a10/server.log lines 60, 147, 251) hit
// `[omarchy] Take dtype is not implemented ... (dtype=bool,
// shape=[1,599])` from the `np.where(images_seq_mask)` gather that
// `mlx_vlm.models.deepseekocr` calls on a bool mask at row 300. The
// packed-bool gather_take.comp USE_BOOL path is the dispatched kernel
// once the dtype branch in `Gather::eval_gpu` is taken; the existing
// float/word/halfword/halfword/i64/complex branch is the documented
// pattern, so the bool entry mirrors it (gate, kernel selection, no
// CPU dispatch).
//
// TakeBool reads the bool table as packed 32-bit words, extracts the
// bool bit at the indexed byte lane through BYTE_AT, and OR-merges
// the result into the output word. TakeAlongAxisBool rides the same
// packed storage through the gather_axis path. Both must stay bit-
// exact against the host-computed values (the bool test fixtures use
// uint8 carrier arrays, the same convention the bool test fixtures
// already use).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <functional>
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

void check_bool(
    array value,
    const std::vector<uint8_t>& expected,
    const Stream& stream) {
  auto dense = contiguous(value);
  dense.eval();
  sync_gpu(stream);
  REQUIRE_EQ(dense.size(), expected.size());
  const uint8_t* values = dense.data<uint8_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    INFO("index ", index, " got ", int(values[index]),
         " want ", int(expected[index]));
    CHECK_EQ(int(values[index]), int(expected[index]));
  }
}

} // namespace

// The exact DeepSeek-OCR row-A10 reproducer: a [1, 5] bool mask taken
// with an int32 index array. Before the fix the dispatch throws
// `[omarchy] Take dtype is not implemented ... (dtype=bool,
// shape=[1,599])`; after the fix the dispatched TakeBool kernel
// returns the indexed bool lanes bit-exactly.
TEST_CASE("take with a bool table and int32 indices (DeepSeek-OCR row A10)") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array mask = astype(
      array({0.0f, 1.0f, 0.0f, 1.0f, 1.0f}, {1, 5}, float32),
      bool_,
      stream);
  // The A10 shape: one index along axis 0 of a [1, N] mask gives a
  // [1, N] row copy.
  array out = take(mask, array({0}, {1}, int32), 0, stream);
  CHECK_EQ(out.dtype(), bool_);
  CHECK_EQ(out.shape(), Shape({1, 5}));
  std::vector<uint8_t> expected = {0, 1, 0, 1, 1};
  check_bool(out, expected, stream);

  // Three indices give three row copies.
  array three = take(mask, array({0, 0, 0}, {3}, int32), 0, stream);
  CHECK_EQ(three.shape(), Shape({3, 5}));
  std::vector<uint8_t> expected_three;
  for (int row = 0; row < 3; ++row) {
    expected_three.insert(
        expected_three.end(), expected.begin(), expected.end());
  }
  check_bool(three, expected_three, stream);
}

// The A10 1xN row shape over a wider row: 13 bools span three words
// (4+4+4) plus a one-byte tail word. Axis-0 extent is 1, so index 0
// and wrapped -1 read the row; anything else is the documented
// out-of-range zero row.
TEST_CASE("take on a packed 1x13 bool mask covers word and tail lane") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array mask = astype(
      array({1.0f,
             0.0f,
             0.0f,
             0.0f,
             0.0f,
             0.0f,
             0.0f,
             1.0f,
             0.0f,
             0.0f,
             0.0f,
             1.0f,
             0.0f},
            {1, 13},
            float32),
      bool_,
      stream);
  std::vector<uint8_t> base = {1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0};
  std::vector<uint8_t> zeros(13, 0);

  // The A10 shape: one index, [1, 13] row copy.
  array out = take(mask, array({0}, {1}, int32), 0, stream);
  CHECK_EQ(out.dtype(), bool_);
  CHECK_EQ(out.shape(), Shape({1, 13}));
  check_bool(out, base, stream);

  // Wrap (-1 -> row 0), valid 0, and two out-of-range zero rows.
  // A [1, 4] index keeps the leading mask dim: out is [1, 4, 13].
  array mixed = take(mask, array({-1, 0, 5, 13}, {1, 4}, int32), 0, stream);
  CHECK_EQ(mixed.shape(), Shape({1, 4, 13}));
  std::vector<uint8_t> expected_mixed;
  for (const auto* row : {&base, &base, &zeros, &zeros}) {
    expected_mixed.insert(
        expected_mixed.end(), row->begin(), row->end());
  }
  check_bool(mixed, expected_mixed, stream);
}

// The same dtype contract through take_along_axis: a 2-D bool source
// and a 2-D int32 index, axis 1.
TEST_CASE("take_along_axis on a bool source keeps bool dtype and values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = astype(
      array({0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}, {2, 3}, float32),
      bool_,
      stream);
  array indices = array({2, 0, 1, 0, 2, 0}, {2, 3}, int32);
  array out = take_along_axis(src, indices, 1, stream);
  CHECK_EQ(out.dtype(), bool_);
  std::vector<uint8_t> expected = {1, 0, 1, 0, 0, 0};
  check_bool(out, expected, stream);
}

// Negative indices still wrap to the same bool lane, the documented
// deviation the float take already provides.
TEST_CASE("take with bool table wraps negative indices") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array mask = astype(
      array({0.0f, 1.0f, 1.0f, 0.0f}, {2, 2}, float32), bool_, stream);
  array out = take(mask, array({-1, -2}, int32), 0, stream);
  CHECK_EQ(out.dtype(), bool_);
  std::vector<uint8_t> expected = {1, 0, 0, 1};
  check_bool(out, expected, stream);
}

// Out-of-range indices read zero, the same documented deviation the
// float / int tables already pin. The bool "zero" is a zero byte.
TEST_CASE("take with bool table reads zero on out-of-range indices") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array mask = astype(
      array({0.0f, 1.0f, 1.0f, 0.0f}, {2, 2}, float32), bool_, stream);
  array out = take(mask, array({2, -3}, int32), 0, stream);
  CHECK_EQ(out.dtype(), bool_);
  std::vector<uint8_t> expected = {0, 0, 0, 0};
  check_bool(out, expected, stream);
}
