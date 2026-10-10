// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// EXL3 trellis decode tests (M3a): the GPU kernel against the op's CPU
// reference (the bit-exact expert_exl3.zig mirror in mlx/fast.cpp), then
// the M3a acceptance GEMV gate. Reference lineage: the M1 lavapipe oracle
// (sushi fork exl3-vk-decode 94715e0) proved the algorithm bit-exact
// against the Zig reference on 294912/294912 elements; this test pins the
// omarchy port to that same reference.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/device.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

struct Case {
  int in_f;
  int out_f;
  int bits;
  int window;
};

// Full-range trellis words: every codeword pattern, masked in the decoder.
std::vector<uint16_t> gen_trellis(size_t n, std::mt19937& rng) {
  std::vector<uint16_t> v(n);
  for (auto& x : v) {
    x = static_cast<uint16_t>(rng() & 0xFFFF);
  }
  return v;
}

// Edge scales first (+-0, min subnormal, max subnormal, min normal, +-1,
// +-2, max finite), then arbitrary normals.
std::vector<uint16_t> gen_scales(size_t n, std::mt19937& rng) {
  static const uint16_t edges[] = {
      0x0000, 0x8000, 0x0001, 0x03FF, 0x8400, 0x3C00, 0xBC00, 0x4000,
      0x7BFF, 0x3555};
  std::vector<uint16_t> v(n);
  for (size_t i = 0; i < n; i++) {
    v[i] = (i < sizeof(edges) / sizeof(edges[0]))
        ? edges[i]
        : static_cast<uint16_t>(0x3800 | (rng() & 0x3FF) |
                                ((rng() & 1) << 15));
  }
  return v;
}

// GPU-comparison gate: the kernel is shift-heavy integer decode, the exact
// class stock Mesa (< 26.2.4) miscompiles (the backend records a
// stock-driver warning at init). Skip the GPU legs where the warning is
// recorded; they run on Honeykrisp devices and are exercised on old
// llvmpipe only through the raw-Vulkan oracle harness.
bool qualifying_gpu() {
  if (!gpu::is_available()) {
    skip("no omarchy GPU device");
    return false;
  }
  auto info = gpu::device_info(0);
  auto warn = info.find("stock_driver_warning");
  if (warn != info.end()) {
    if (auto* w = std::get_if<std::string>(&warn->second); w && !w->empty()) {
      skip(w->c_str());
      return false;
    }
  }
  return true;
}

} // namespace

// Debug bisect companion (MLX_OMARCHY_EXL3_DEBUG_DUMP=<dir>): the GPU side
// dumps dec16/tmp/out inside Exl3Decode::eval_gpu. For every GPU call this
// writes the matching CPU reference stages — dec16 (f16-bit tile decode,
// uint32 words), tmp (row H128 + suh), out (f32) — plus a manifest line.
// exl3_dump_compare.py pairs call= and ref= entries by (dims, rate, dtype)
// and prints per-stage first-mismatch indices.
void dump_reference_stages(
    const std::vector<uint16_t>& tv,
    const std::vector<uint16_t>& su,
    const std::vector<uint16_t>& sv,
    int in_f,
    int out_f,
    int bits,
    int window,
    Dtype dt) {
  const char* dir = std::getenv("MLX_OMARCHY_EXL3_DEBUG_DUMP");
  if (dir == nullptr || dir[0] == '\0') {
    return;
  }
  static uint32_t ref_index = 0;
  uint32_t idx = ref_index++;
  size_t blocks =
      static_cast<size_t>(in_f / 128) * static_cast<size_t>(out_f / 128);
  std::vector<float> w(static_cast<size_t>(in_f) * out_f);
  std::vector<uint32_t> dec16(blocks * 16384);
  std::vector<float> mid(blocks * 16384);
  fast::exl3_reconstruct_stages(
      tv.data(), su.data(), sv.data(), in_f, out_f, bits, window, w.data(),
      dec16.data(), mid.data());
  const char* dtype_tag =
      (dt == float32) ? "f32" : (dt == float16) ? "f16" : "bf16";
  auto write_all = [&](const std::string& path, const void* data,
                       size_t bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  };
  std::string base = std::string(dir) + "/ref" + std::to_string(idx);
  write_all(base + "_dec16_u32.bin", dec16.data(), dec16.size() * 4);
  write_all(base + "_tmp_f32.bin", mid.data(), mid.size() * 4);
  write_all(base + "_out_f32.bin", w.data(), w.size() * 4);
  std::ofstream manifest(std::string(dir) + "/manifest.txt", std::ios::app);
  manifest << "ref=" << idx << " in=" << in_f << " out=" << out_f
           << " bits=" << bits << " window=" << window
           << " dtype=" << dtype_tag << "\n";
}

TEST_CASE("exl3 decode gpu vs cpu reference") {
  if (!qualifying_gpu()) {
    return;
  }
  Stream gs = new_stream(Device::gpu);
  Stream cs = new_stream(Device::cpu);
  std::mt19937 rng(20261010);
  // Rates cover the uniform windows (n = 32, 64, 128), a non-uniform
  // fresh-bits rate (n = 48: 3 fresh bits per weight), n = 96, and the
  // legacy unmasked window 16.
  std::vector<Case> cases{
      {128, 128, 2, 15},
      {256, 128, 3, 15},
      {128, 256, 4, 15},
      {256, 256, 6, 15},
      {384, 256, 8, 15},
      {256, 128, 4, 16}};
  for (const auto& c : cases) {
    INFO("case ", c.in_f, "x", c.out_f, " bits=", c.bits,
         " window=", c.window);
    size_t tiles =
        static_cast<size_t>(c.in_f / 128) * (c.out_f / 128) * 64;
    size_t tlen = tiles * static_cast<size_t>(c.bits) * 16;
    auto tv = gen_trellis(tlen, rng);
    auto su = gen_scales(static_cast<size_t>(c.in_f), rng);
    auto sv = gen_scales(static_cast<size_t>(c.out_f), rng);
    Shape tshape{static_cast<int>(tlen)};
    array trellis(tv.data(), tshape, uint16);
    array suh(su.data(), Shape{c.in_f}, uint16);
    array svh(sv.data(), Shape{c.out_f}, uint16);

    for (Dtype dt : {float32, float16, bfloat16}) {
      array g = fast::exl3_decode(
          trellis, suh, svh, c.in_f, c.out_f, c.bits, c.window, dt, gs);
      eval(g);
      dump_reference_stages(
          tv, su, sv, c.in_f, c.out_f, c.bits, c.window, dt);
      array r = fast::exl3_decode(
          trellis, suh, svh, c.in_f, c.out_f, c.bits, c.window, dt, cs);
      eval(r);
      CHECK(g.shape() == Shape{c.in_f, c.out_f});
      size_t n = g.size();
      if (dt == float32) {
        auto* gp = g.data<float>();
        auto* rp = r.data<float>();
        double max_rel = 0.0;
        size_t bit_diffs = 0;
        for (size_t i = 0; i < n; i++) {
          if (std::memcmp(&gp[i], &rp[i], 4) != 0) {
            bit_diffs++;
            double diff = std::fabs(static_cast<double>(gp[i]) - rp[i]);
            double denom = std::fabs(static_cast<double>(rp[i]));
            if (denom > 0 && diff / denom > max_rel) {
              max_rel = diff / denom;
            }
          }
        }
        CHECK_EQ(bit_diffs, 0);
        CHECK_LT(max_rel, 1e-6);
        if (bit_diffs != 0) {
          std::cout << "  f32 bit_diffs=" << bit_diffs << "/"
                    << n << " max_rel=" << max_rel << "\n";
        }
      } else {
        auto* gp = g.data<uint16_t>();
        auto* rp = r.data<uint16_t>();
        size_t diffs = 0;
        for (size_t i = 0; i < n; i++) {
          if (gp[i] != rp[i]) {
            diffs++;
          }
        }
        CHECK_EQ(diffs, 0);
        if (diffs != 0) {
          std::cout << "  bits bit_diffs=" << diffs << "/" << n << "\n";
        }
      }
    }
  }
}

TEST_CASE("exl3 decode gemv vs host reference") {
  if (!qualifying_gpu()) {
    return;
  }
  Stream gs = new_stream(Device::gpu);
  Stream cs = new_stream(Device::cpu);
  std::mt19937 rng(20261011);
  const int in_f = 256;
  const int out_f = 128;
  const int bits = 4;
  const int window = 15;
  size_t tlen =
      static_cast<size_t>(in_f / 128) * (out_f / 128) * 64 * bits * 16;
  auto tv = gen_trellis(tlen, rng);
  auto su = gen_scales(static_cast<size_t>(in_f), rng);
  auto sv = gen_scales(static_cast<size_t>(out_f), rng);
  array trellis(tv.data(), Shape{static_cast<int>(tlen)}, uint16);
  array suh(su.data(), Shape{in_f}, uint16);
  array svh(sv.data(), Shape{out_f}, uint16);

  // Production route: f16 weights on the GPU stream, x in f32.
  array w16 = fast::exl3_decode(
      trellis, suh, svh, in_f, out_f, bits, window, float16, gs);
  eval(w16);

  const int rows = 8;
  std::vector<float> xv(static_cast<size_t>(rows) * in_f);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (auto& v : xv) {
    v = dist(rng);
  }
  array x(xv.data(), Shape{rows, in_f}, float32);
  array y = matmul(x, astype(w16, float32, gs), gs);
  eval(y);
  auto* yp = y.data<float>();

  // Host reference in double over the CPU reference decode.
  array r = fast::exl3_decode(
      trellis, suh, svh, in_f, out_f, bits, window, float32, cs);
  eval(r);
  auto* rp = r.data<float>();
  double max_rel = 0.0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < out_f; j++) {
      double acc = 0.0;
      for (int k = 0; k < in_f; k++) {
        acc += static_cast<double>(xv[static_cast<size_t>(i) * in_f + k]) *
            static_cast<double>(rp[static_cast<size_t>(k) * out_f + j]);
      }
      double got = yp[static_cast<size_t>(i) * out_f + j];
      double denom = std::fabs(acc) > 1e-9 ? std::fabs(acc) : 1.0;
      double rel = std::fabs(got - acc) / denom;
      if (rel > max_rel) {
        max_rel = rel;
      }
    }
  }
  std::cout << "GEMV max_rel = " << max_rel << "\n";
  CHECK_LT(max_rel, 1e-3);
}

TEST_CASE("exl3 decode rejects malformed inputs") {
  Stream cs = new_stream(Device::cpu);
  std::vector<uint16_t> tv(64 * 32);
  std::vector<uint16_t> sv(128, 0x3C00);
  array trellis(tv.data(), Shape{static_cast<int>(tv.size())}, uint16);
  array suh(sv.data(), Shape{128}, uint16);
  array svh(sv.data(), Shape{128}, uint16);
  CHECK_THROWS(fast::exl3_decode(
      trellis, suh, svh, 100, 128, 2, 15, float32, cs)); // not %128
  CHECK_THROWS(fast::exl3_decode(
      trellis, suh, svh, 128, 128, 0, 15, float32, cs)); // rate < 16
  CHECK_THROWS(fast::exl3_decode(
      trellis, suh, svh, 128, 128, 2, 7, float32, cs)); // window < 8
}
