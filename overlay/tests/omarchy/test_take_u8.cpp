// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Regression tests for uint8/int8 Take on the Omarchy Vulkan backend.
// The LTX-2 video-VAE decode on a 16-inch M1 Max (G13C) hit `[omarchy] Take dtype is
// not implemented ... (dtype=uint8, shape=[1,3,320,512])` from the
// tiled_decode loop taking a scalar position out of a uint8 chunk. The
// bool Take fix (test_take_bool.cpp) already rides a 1-byte packed-word
// route: BYTE_AT lane extraction, atomicOr byte merge, pre-zeroed
// output. uint8/int8 are the same 1-byte storage, so the fix dispatches
// them through a byte variant of the same shader; the tests below pin
// bit-exact byte copies (values above 1 must survive verbatim, which
// the bool normalization would collapse, and int8 signs must survive).
//
// Every implemented mode checks exact host-computed values. Before the
// fix each take below throws the named "Take dtype" refusal at eval and
// the case fails; after the fix every case is bit-exact.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <iostream>
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

void check_u8(
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

void check_i8(
    array value,
    const std::vector<int8_t>& expected,
    const Stream& stream) {
  auto dense = contiguous(value);
  dense.eval();
  sync_gpu(stream);
  REQUIRE_EQ(dense.size(), expected.size());
  const int8_t* values = dense.data<int8_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    INFO("index ", index, " got ", int(values[index]),
         " want ", int(expected[index]));
    CHECK_EQ(int(values[index]), int(expected[index]));
  }
}

void check_floats(
    array value,
    const std::vector<float>& expected,
    const Stream& stream,
    double epsilon = 1e-5) {
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

} // namespace

// Bytes above 1 must copy verbatim; the bool normalization (!= 0) on
// the same packed-word route would collapse 250 / 255 / 128 to 1.
TEST_CASE("take copies a uint8 table byte-exactly with int32 indices") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<uint8_t> raw = {7, 250, 3, 255, 128, 1};
  array table(raw.data(), {6}, uint8);
  array out = take(table, array({5, 1, 1}, {3}, int32), 0, stream);
  CHECK_EQ(out.dtype(), uint8);
  CHECK_EQ(out.shape(), Shape({3}));
  check_u8(out, {1, 250, 250}, stream);
}

// Signed bytes keep their sign through the gather; the storage byte is
// copied, so -100 arrives as -100, not 156.
TEST_CASE("take preserves negative int8 values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<int8_t> raw = {-100, 3, -1, 127};
  array table(raw.data(), {4}, int8);
  array out = take(table, array({0, 3}, {2}, int32), 0, stream);
  CHECK_EQ(out.dtype(), int8);
  check_i8(out, {int8_t(-100), int8_t(127)}, stream);
}

// Negative indices wrap like upstream offset_neg_idx; an index outside
// [-dim, dim) reads zero, the documented deviation the float and int
// table tests already pin.
TEST_CASE("take on a 2-D uint8 table wraps negatives and zeroes out-of-range") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<uint8_t> raw = {10, 11, 12, 20, 21, 22};
  array table(raw.data(), {2, 3}, uint8);
  array out = take(table, array({1, -1, 2}, {3}, int32), 0, stream);
  CHECK_EQ(out.shape(), Shape({3, 3}));
  check_u8(
      out,
      {20, 21, 22, 20, 21, 22, 0, 0, 0},
      stream);
}

// The 8-bit index modes the decode already supports (uint32, int64)
// ride the same byte table.
TEST_CASE("take on a uint8 table accepts uint32 and int64 index arrays") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<uint8_t> raw = {4, 5, 6, 7};
  array table(raw.data(), {4}, uint8);
  check_u8(
      take(table, array({2, 0}, {2}, uint32), 0, stream),
      {6, 4},
      stream);
  check_u8(
      take(table,
           array({int64_t(1), int64_t(3)}, {2}, int64),
           0,
           stream),
      {5, 7},
      stream);
}

// The LTX-2 video-VAE tiled_decode row: a scalar index on a 5-D uint8
// chunk drops the gathered axis. A 0-d index array is the dynamic form
// that dispatches Take (the static `t[:, :, 1]` syntax lowers to Slice,
// which already passes, so the scalar Take is the crashing path).
//
// The G13C run of the first landing attempt corrupted exactly this
// case (stale bytes came back; the five simpler cases passed), so the
// case carries four discriminators that isolate a G13C failure to one
// layer: the same take through the float kernel (byte storage vs
// collapse/scalar path), the sibling bool kernel (the landed TakeBool
// code on the same shape), a plain 72-byte u8 copy roundtrip (a broken
// u8 readback at this size would masquerade as a kernel failure), and
// a float32 readback of the take output (bytes right as floats means
// the u8 readback is the broken layer).
TEST_CASE("scalar-index take drops the axis on a 5-D uint8 table") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // [1,3,2,4,6] is 144 elements; the raw-pointer array ctor does not
  // size-check the host buffer, so the table vector must carry the
  // full 144 bytes (a 36-byte buffer here reads heap garbage past its
  // end and the gather faithfully returns it — the bug behind the
  // first G13C run's "corruption").
  std::vector<uint8_t> raw(144);
  for (size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<uint8_t>((i * 7 + 1) & 0xFF);
  }
  array table(raw.data(), {1, 3, 2, 4, 6}, uint8);
  array out = take(table, array(1), 2, stream);
  CHECK_EQ(out.dtype(), uint8);
  CHECK_EQ(out.shape(), Shape({1, 3, 4, 6}));
  std::vector<uint8_t> expected;
  for (int b = 0; b < 3; ++b) {
    for (int p = 0; p < 24; ++p) {
      int flat = (b * 2 + 1) * 24 + p;
      expected.push_back(static_cast<uint8_t>((flat * 7 + 1) & 0xFF));
    }
  }
  check_u8(out, expected, stream);

  // Discriminator 1: the same 5-D scalar take through the float kernel
  // (collapse + scalar indexing without byte storage). Pass here + fail
  // above isolates the byte storage path.
  array table_f = astype(table, float32, stream);
  std::vector<float> table_f_expected;
  for (uint8_t byte : raw) {
    table_f_expected.push_back(static_cast<float>(byte));
  }
  std::vector<float> scalar_f_expected;
  for (int b = 0; b < 3; ++b) {
    for (int p = 0; p < 24; ++p) {
      scalar_f_expected.push_back(static_cast<float>(expected[b * 24 + p]));
    }
  }
  check_floats(
      take(table_f, array(1), 2, stream),
      scalar_f_expected,
      stream);

  // Discriminator 2: the sibling bool kernel (the landed TakeBool code)
  // on the same 5-D collapsed shape. If bool corrupts too, the byte-lane
  // merge on G13C is broken for the pre-existing kernel as well.
  array table_b = astype(table, bool_, stream);
  std::vector<uint8_t> expected_b;
  for (uint8_t byte : expected) {
    expected_b.push_back(byte ? 1 : 0);
  }
  check_u8(take(table_b, array(1), 2, stream), expected_b, stream);

  // Discriminator 3: plain u8 GPU copy roundtrip at 72 bytes (the take
  // output size; a broken u8 readback at this size would masquerade as
  // a kernel failure).
  std::vector<uint8_t> rt(72);
  for (size_t i = 0; i < rt.size(); ++i) {
    rt[i] = static_cast<uint8_t>((i * 3 + 7) & 0xFF);
  }
  array rt_table(rt.data(), {72}, uint8);
  check_u8(contiguous(rt_table, false, stream), rt, stream);

  // Discriminator 4: read the same take output back as float32. If the
  // bytes are right as floats while check_u8 failed, the u8 readback
  // path is the broken layer.
  array out_f = astype(out, float32, stream);
  std::vector<float> expected_f(expected.begin(), expected.end());
  check_floats(out_f, expected_f, stream);

  // A length-1 index keeps the axis instead.
  array kept = take(table, array({1}, {1}, int32), 2, stream);
  CHECK_EQ(kept.shape(), Shape({1, 3, 1, 4, 6}));
  check_u8(kept, expected, stream);
}

// Unaligned byte counts exercise the descriptor boundary the word
// transport rides: a 5-byte table reads input word (in_elem >> 2) --
// index 4 pulls word 1, whose bytes 5..7 sit past the logical table --
// and a 3-byte output merges all its bytes into output word 0 through
// atomicOr, one byte past the logical end. binding() rounds ranges up
// to word granularity so both accesses stay inside the bound; bytes
// past a logical end are never extracted into a result (BYTE_AT reads
// only valid lanes) and unclaimed output lanes stay at the host
// pre-zero fill. Expected bytes computed by hand.
TEST_CASE("take copies unaligned byte tables and writes unaligned outputs") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // 5-byte uint8 table (nbytes % 4 == 1), indices [2, 0, 4]:
  // out = {t[2], t[0], t[4]} = {30, 10, 50}, a 3-byte output
  // (nbytes % 4 == 3) landing entirely inside word 0.
  std::vector<uint8_t> raw = {10, 20, 30, 40, 50};
  array table(raw.data(), {5}, uint8);
  array out = take(table, array({2, 0, 4}, {3}, int32), 0, stream);
  CHECK_EQ(out.dtype(), uint8);
  CHECK_EQ(out.shape(), Shape({3}));
  check_u8(out, {30, 10, 50}, stream);

  // The matching int8 variant: index 2 must return the most-negative
  // byte verbatim ({-3, 127, -128, 1, 0}[{2, 0, 4}] = {-128, -3, 0}),
  // not its unsigned echo.
  std::vector<int8_t> sraw = {-3, 127, -128, 1, 0};
  array stable(sraw.data(), {5}, int8);
  array sout = take(stable, array({2, 0, 4}, {3}, int32), 0, stream);
  CHECK_EQ(sout.dtype(), int8);
  check_i8(sout, {int8_t(-128), int8_t(-3), int8_t(0)}, stream);
}

// Multi-index gather packs the broadcast indices through the metadata
// transport; the byte table must survive the same walk.
TEST_CASE("gather with two index arrays copies a uint8 table") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<uint8_t> raw = {1, 2, 3, 4, 5, 6};
  array table(raw.data(), {2, 3}, uint8);
  array i0 = array({1, 0}, {2}, int32);
  array i1 = array({2, 0}, {2}, int32);
  array out = gather(table, {i0, i1}, {0, 1}, {1, 1}, stream);
  CHECK_EQ(out.dtype(), uint8);
  CHECK_EQ(out.shape(), Shape({2, 1, 1}));
  check_u8(out, {6, 1}, stream);
}
