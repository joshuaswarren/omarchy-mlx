// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Trio RMSNorm (TRIO_NORM) for float16 AND bfloat16. The bf16 body
// keeps the float32 reduction and rounds once at the store, the same
// contract as fast_norm.comp's bf16 form, so the fused dispatch must
// match the standalone FastRmsNormBF16 dispatch bit for bit. The two
// dispatches are one-for-one (same binding count and grid), so unlike
// the RoPE pair there is no dispatch-count delta that could prove the
// fused kernel ran; the honest assertion here is the bit contract
// itself plus the fact that a wrong or missing bf16 kernel fails
// pipeline creation loudly rather than producing near-miss bits.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/compile.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

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

void sync_stream(const Stream& stream) {
  mlx::core::omarchy::get_command_encoder(stream).synchronize();
}

void set_trio(const char* value) {
  setenv("MLX_OMARCHY_FUSED_TRIO", value, 1);
}

void expect_bits_equal(const array& a, const array& b) {
  REQUIRE_EQ(a.size(), b.size());
  const uint16_t* pa = a.data<uint16_t>();
  const uint16_t* pb = b.data<uint16_t>();
  for (size_t i = 0; i < a.size(); ++i) {
    INFO("bit mismatch at ", i, " trio=0x", std::hex, pa[i], " solo=0x", pb[i]);
    CHECK_EQ(pa[i], pb[i]);
  }
}

// One rms_norm row batch, evaluated twice: TRIO=0 (standalone kernel)
// and TRIO=1 (trio norm), same input bytes both arms.
void check_norm_case(Dtype dtype, bool scalar_weight) {
  Stream stream = gpu_stream();
  set_compile_mode(CompileMode::disabled);

  array x = astype(
      random::normal(Shape{3, 64}, float32, std::nullopt, stream),
      dtype,
      stream);
  x.eval();
  array w = astype(
      scalar_weight ? random::normal(Shape{1}, float32, std::nullopt, stream)
                    : random::normal(Shape{64}, float32, std::nullopt, stream),
      dtype,
      stream);
  w.eval();
  sync_stream(stream);

  set_trio("0");
  array solo = fast::rms_norm(x, w, 1e-6f, stream);
  eval(std::vector<array>{solo});
  sync_stream(stream);

  set_trio("1");
  array trio = fast::rms_norm(x, w, 1e-6f, stream);
  eval(std::vector<array>{trio});
  sync_stream(stream);

  REQUIRE_EQ(trio.shape(), solo.shape());
  expect_bits_equal(trio, solo);
}

} // namespace

TEST_CASE("bf16 trio rms norm is bit-exact against the standalone kernel") {
  if (!compute_available()) {
    return;
  }
  setenv("MLX_OMARCHY_ALLOW_NON_APPLE", "1", 0);
  check_norm_case(bfloat16, false);
}

TEST_CASE("f16 trio rms norm stays bit-exact") {
  if (!compute_available()) {
    return;
  }
  setenv("MLX_OMARCHY_ALLOW_NON_APPLE", "1", 0);
  check_norm_case(float16, false);
}
