// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// Wave 6: the matmul family. BlockMaskedMM, GatherMM, SegmentedMM,
// GatherQMM, GatherQQMM, and QQMatmul, each valued against a host
// double-precision reference on small shapes, one edge-tile shape per op,
// and named-error pins for the unsupported quantization modes.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/cpu/device_info.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/matmul_direct_select.h"
#include "mlx/device.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

// The no-progress submit watchdog (device.cpp wait_for_timeline_progress)
// sizes its hang interval for hardware: a single-increment timeline wait
// shows no counter motion until its submit completes, so any dispatch that
// legitimately runs longer than MLX_OMARCHY_HANG_NO_PROGRESS_NS is
// misdiagnosed as a hung device. On hardware each prefill submit stays
// under the 10 s default; on the Mesa llvmpipe software rasterizer this
// repo develops against, one 4864x4864 prefill-shape quantized matmul
// takes tens of seconds, so the default window kills the tile case's
// jumbo sweep (and trips a teardown crash on the error path) even though
// the kernel is correct. Provision the documented env override before the
// first wait caches it (device.cpp env_ns_override is static-cached on
// first use): only when the driver is llvmpipe, only when the operator
// has not set a window already. Hardware keeps the 10 s default, and the
// MLX_OMARCHY_MAX_WALL_NS ceiling (30 min default) stays authoritative.
void provision_software_device_watchdog() {
  if (std::getenv("MLX_OMARCHY_HANG_NO_PROGRESS_NS") != nullptr) {
    return;
  }
  if (!gpu::is_available()) {
    return;
  }
  const auto& caps = omarchy::device(0).capabilities();
  if (caps.driver_id != VK_DRIVER_ID_MESA_LLVMPIPE) {
    return;
  }
  std::cout
      << "[provenance] llvmpipe detected: raising submit no-progress"
      << " window to 900 s for jumbo prefill-shape dispatches\n";
  setenv("MLX_OMARCHY_HANG_NO_PROGRESS_NS", "900000000000", 1);
}

} // namespace

int main(int argc, char** argv) {
  provision_software_device_watchdog();
  doctest::Context ctx;
  ctx.applyCommandLine(argc, argv);
  return ctx.run();
}

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
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return false;
  }
  return true;
}

bool float16_available() {
  const auto& capabilities = omarchy::device(0).capabilities();
  return capabilities.shader_float16 &&
      capabilities.storage_buffer_16bit_access;
}

std::string evaluation_error(array value) {
  try {
    value.eval();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

std::vector<float> readback_f32(const Stream& stream, array value) {
  value = astype(value, float32, stream);
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = value.data<float>();
  return std::vector<float>(data, data + value.size());
}

void expect_close(
    const std::vector<float>& device,
    const std::vector<float>& expected,
    double epsilon) {
  REQUIRE_EQ(device.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK(device[index] == doctest::Approx(expected[index]).epsilon(epsilon));
  }
}

// fp32 accumulation against a double reference carries a small
// absolute error even where cancellation makes the expected value
// near zero; epsilon alone is pure relative, so add an atol floor.
// Any wrong scale, code, or index misses by O(1) and still fails.
void expect_close_tol(
    const std::vector<float>& device,
    const std::vector<float>& expected,
    double atol,
    double rtol) {
  REQUIRE_EQ(device.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    // This doctest's rule is |a - b| <= eps * (scale + max(|a|, |b|)),
    // so scale(atol / rtol) makes the tolerance atol + rtol * max.
    CHECK(device[index] ==
        doctest::Approx(expected[index]).epsilon(rtol).scale(atol / rtol));
  }
}

double host_at(const std::vector<float>& values, size_t index) {
  return static_cast<double>(values[index]);
}

using cdouble = std::complex<double>;

array complex_array(const std::vector<cdouble>& values, Shape shape) {
  std::vector<complex64_t> host(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    host[i] = complex64_t(
        static_cast<float>(values[i].real()),
        static_cast<float>(values[i].imag()));
  }
  return array(host.begin(), std::move(shape), complex64);
}

std::vector<cdouble> readback_complex(const Stream& stream, array value) {
  value = contiguous(value, false, stream);
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const complex64_t* data = value.data<complex64_t>();
  std::vector<cdouble> out(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    out[i] = {data[i].real(), data[i].imag()};
  }
  return out;
}

void expect_complex_close(
    const std::vector<cdouble>& device,
    const std::vector<cdouble>& expected,
    double atol,
    double rtol) {
  REQUIRE_EQ(device.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    INFO("index ", index, " got ", device[index], " expected ", expected[index]);
    CHECK(std::abs(device[index] - expected[index]) <=
        atol + rtol * std::abs(expected[index]));
  }
}

// Host affine quantizer matching the upstream affine_quantize kernel:
// the abs-dominant endpoint picks the scale sign, the q0 refinement pins
// that endpoint exactly, and clipped rounded codes pack LSB-first,
// 32/bits values per uint32 word.
struct HostQuantizedWeights {
  std::vector<uint32_t> words;
  std::vector<float> scales;
  std::vector<float> biases;
};

// Round values through a 16-bit dtype on the device and read the exact
// 16-bit values back, so host references see what the kernel sees.
std::vector<float> round_trip(
    const Stream& stream,
    const std::vector<float>& values,
    Dtype dtype) {
  array device(values.begin(), Shape{static_cast<int>(values.size())}, float32);
  return readback_f32(
      stream, astype(astype(device, dtype, stream), float32, stream));
}

HostQuantizedWeights host_affine_quantize(
    const std::vector<float>& matrix,
    int rows,
    int cols,
    int group_size,
    int bits) {
  HostQuantizedWeights result;
  int groups = cols / group_size;
  int pack = 32 / bits;
  int words_per_row = cols / pack;
  float n_bins = static_cast<float>((1 << bits) - 1);
  result.words.assign(static_cast<size_t>(rows) * words_per_row, 0);
  result.scales.resize(static_cast<size_t>(rows) * groups);
  result.biases.resize(static_cast<size_t>(rows) * groups);
  for (int row = 0; row < rows; ++row) {
    for (int group = 0; group < groups; ++group) {
      float w_max = -std::numeric_limits<float>::infinity();
      float w_min = std::numeric_limits<float>::infinity();
      for (int i = 0; i < group_size; ++i) {
        float value = matrix[row * cols + group * group_size + i];
        w_max = std::max(w_max, value);
        w_min = std::min(w_min, value);
      }
      bool min_dominant = std::abs(w_min) > std::abs(w_max);
      float scale = std::max((w_max - w_min) / n_bins, 1e-7f);
      if (!min_dominant) {
        scale = -scale;
      }
      float edge = min_dominant ? w_min : w_max;
      float q0 = std::round(edge / scale);
      if (q0 != 0.0f) {
        scale = edge / q0;
      }
      float bias = (q0 == 0.0f) ? 0.0f : edge;
      result.scales[row * groups + group] = scale;
      result.biases[row * groups + group] = bias;
      for (int i = 0; i < group_size; ++i) {
        float value = matrix[row * cols + group * group_size + i];
        float q = std::clamp(std::round((value - bias) / scale), 0.0f, n_bins);
        uint32_t code = static_cast<uint32_t>(q);
        int col = group * group_size + i;
        result.words[row * words_per_row + col / pack] |=
            code << ((col % pack) * bits);
      }
    }
  }
  return result;
}

// Host dot in double precision: x row m against dequantized w row n,
// dequant = q * scale + bias.
std::vector<float> host_quantized_matmul(
    const HostQuantizedWeights& w,
    const std::vector<float>& x,
    int m,
    int n,
    int k,
    int group_size,
    int bits) {
  int pack = 32 / bits;
  int words_per_row = k / pack;
  int groups = k / group_size;
  uint32_t mask = (1u << bits) - 1u;
  std::vector<float> out(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    for (int column = 0; column < n; ++column) {
      double acc = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        uint32_t code =
            (w.words[column * words_per_row + inner / pack] >>
             ((inner % pack) * bits)) &
            mask;
        double dequant = static_cast<double>(code) *
                w.scales[column * groups + inner / group_size] +
            w.biases[column * groups + inner / group_size];
        acc += host_at(x, row * k + inner) * dequant;
      }
      out[row * n + column] = static_cast<float>(acc);
    }
  }
  return out;
}

// The GatherQQMM / QQMatmul affine contract carries no biases: the
// dequantized value is q * scale only.
std::vector<float> host_scale_only_quantized_matmul(
    const HostQuantizedWeights& w,
    const std::vector<float>& x,
    int m,
    int n,
    int k,
    int group_size,
    int bits) {
  HostQuantizedWeights zero_bias = w;
  std::fill(zero_bias.biases.begin(), zero_bias.biases.end(), 0.0f);
  return host_quantized_matmul(zero_bias, x, m, n, k, group_size, bits);
}

// BlockMaskedMM reference: a masked block contributes nothing, so the
// element-level filter below is exact upstream block semantics including
// partial edge blocks; the out mask multiplies whole blocks afterwards.
std::vector<float> block_masked_reference(
    const std::vector<float>& a, // (batch, m, k)
    const std::vector<float>& b, // (batch, k, n)
    const std::vector<uint8_t>& lhs_mask,
    const std::vector<uint8_t>& rhs_mask,
    const std::vector<float>* out_mask, // nullptr when absent
    int batch,
    int m,
    int k,
    int n,
    int bs) {
  int tm = (m + bs - 1) / bs;
  int tk = (k + bs - 1) / bs;
  int tn = (n + bs - 1) / bs;
  std::vector<float> expected(static_cast<size_t>(batch) * m * n, 0.0f);
  for (int bt = 0; bt < batch; ++bt) {
    for (int r = 0; r < m; ++r) {
      for (int c = 0; c < n; ++c) {
        double acc = 0.0;
        for (int inner = 0; inner < k; ++inner) {
          bool a_keep =
              lhs_mask[(static_cast<size_t>(bt) * tm + r / bs) * tk +
                       inner / bs];
          bool b_keep =
              rhs_mask[(static_cast<size_t>(bt) * tk + inner / bs) * tn +
                       c / bs];
          if (a_keep && b_keep) {
            acc += host_at(a, (static_cast<size_t>(bt) * m + r) * k + inner) *
                host_at(b, (static_cast<size_t>(bt) * k + inner) * n + c);
          }
        }
        float value = static_cast<float>(acc);
        if (out_mask) {
          value *= (*out_mask)[(static_cast<size_t>(bt) * tm + r / bs) * tn +
                               c / bs];
        }
        expected[(static_cast<size_t>(bt) * m + r) * n + c] = value;
      }
    }
  }
  return expected;
}

// Plain double-precision matmul over stacked operand values.
std::vector<float> host_matmul(
    const std::vector<float>& a, // (m, k) or (batch, m, k)
    const std::vector<float>& b, // (k, n) or (batch, k, n)
    int batch,
    int m,
    int k,
    int n) {
  std::vector<float> out(static_cast<size_t>(batch) * m * n);
  for (int bt = 0; bt < batch; ++bt) {
    for (int r = 0; r < m; ++r) {
      for (int c = 0; c < n; ++c) {
        double acc = 0.0;
        for (int inner = 0; inner < k; ++inner) {
          acc += host_at(a, (static_cast<size_t>(bt) * m + r) * k + inner) *
              host_at(b, (static_cast<size_t>(bt) * k + inner) * n + c);
        }
        out[(static_cast<size_t>(bt) * m + r) * n + c] =
            static_cast<float>(acc);
      }
    }
  }
  return out;
}

} // namespace

TEST_CASE("single-row matmul covers columns beyond the dispatch limit") {
  if (!compute_available()) {
    return;
  }
  const int n = 32 * 65535 + 1;
  Stream stream = gpu_stream();
  std::vector<float> values(n);
  for (int i = 0; i < n; ++i) {
    values[i] = float(1 + i % 7);
  }
  auto got = readback_f32(stream, matmul(
      array({2.5f}, Shape{1, 1}),
      array(values.begin(), Shape{1, n}, float32), stream));
  REQUIRE_EQ(got.size(), values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    CHECK_EQ(got[i], 2.5f * values[i]);
  }
}

TEST_CASE("dense matmul supports complex64 values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  uint64_t dispatches_before =
      omarchy::trace::counters().vk_compute_dispatches.load();
  std::vector<cdouble> a_values{
      {1.0, 2.0}, {-0.5, 0.25}, {2.0, -1.0},
      {0.75, -0.5}, {1.5, 0.0}, {-1.0, 1.25}};
  std::vector<cdouble> b_values{
      {0.5, -1.0}, {1.0, 0.25},
      {-2.0, 0.5}, {0.75, -0.25},
      {1.25, 1.5}, {-0.5, 2.0}};
  std::vector<cdouble> expected(4);
  for (int row = 0; row < 2; ++row) {
    for (int column = 0; column < 2; ++column) {
      cdouble sum{};
      for (int inner = 0; inner < 3; ++inner) {
        sum += a_values[row * 3 + inner] * b_values[inner * 2 + column];
      }
      expected[row * 2 + column] = sum;
    }
  }
  array out = matmul(
      complex_array(a_values, Shape{2, 3}),
      complex_array(b_values, Shape{3, 2}),
      stream);
  expect_complex_close(readback_complex(stream, out), expected, 1e-6, 1e-5);
  CHECK(omarchy::trace::counters().vk_compute_dispatches.load() >
      dispatches_before);
}

TEST_CASE("dense matmul supports general bias rank and half values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  SUBCASE("general AddMM broadcast") {
    std::vector<float> a_values(2 * 3 * 4);
    std::vector<float> b_values(2 * 4 * 5);
    std::vector<float> c_values(2 * 5);
    for (size_t i = 0; i < a_values.size(); ++i) {
      a_values[i] = static_cast<float>(static_cast<int>(i % 9) - 4) / 8.0f;
    }
    for (size_t i = 0; i < b_values.size(); ++i) {
      b_values[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 8.0f;
    }
    for (size_t i = 0; i < c_values.size(); ++i) {
      c_values[i] = static_cast<float>(i + 1) / 16.0f;
    }
    std::vector<float> expected(2 * 3 * 5);
    for (int batch = 0; batch < 2; ++batch) {
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 5; ++column) {
          double sum = 0.0;
          for (int inner = 0; inner < 4; ++inner) {
            sum += host_at(a_values, (batch * 3 + row) * 4 + inner) *
                host_at(b_values, (batch * 4 + inner) * 5 + column);
          }
          expected[(batch * 3 + row) * 5 + column] =
              static_cast<float>(0.5 * sum + 2.0 * c_values[batch * 5 + column]);
        }
      }
    }
    array a(a_values.begin(), Shape{2, 3, 4}, float32);
    array b(b_values.begin(), Shape{2, 4, 5}, float32);
    array c(c_values.begin(), Shape{2, 1, 5}, float32);
    expect_close(
        readback_f32(stream, addmm(c, a, b, 0.5f, 2.0f, stream)),
        expected,
        1e-6);
  }

  SUBCASE("rank six broadcasted matrix vector") {
    std::vector<float> a_values(2 * 2 * 3 * 4);
    std::vector<float> b_values(2 * 2 * 2 * 4);
    for (size_t i = 0; i < a_values.size(); ++i) {
      a_values[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 8.0f;
    }
    for (size_t i = 0; i < b_values.size(); ++i) {
      b_values[i] = static_cast<float>(static_cast<int>(i % 5) - 2) / 4.0f;
    }
    std::vector<float> expected;
    expected.reserve(2 * 2 * 2 * 3);
    for (int batch0 = 0; batch0 < 2; ++batch0) {
      for (int batch2 = 0; batch2 < 2; ++batch2) {
        for (int batch3 = 0; batch3 < 2; ++batch3) {
          for (int row = 0; row < 3; ++row) {
            double sum = 0.0;
            for (int inner = 0; inner < 4; ++inner) {
              size_t a_index = ((batch0 * 2 + batch2) * 3 + row) * 4 + inner;
              size_t b_index =
                  ((batch0 * 2 + batch2) * 2 + batch3) * 4 + inner;
              sum += host_at(a_values, a_index) * host_at(b_values, b_index);
            }
            expected.push_back(static_cast<float>(sum));
          }
        }
      }
    }
    array a(a_values.begin(), Shape{2, 1, 2, 1, 3, 4}, float32);
    array b_column(b_values.begin(), Shape{2, 1, 2, 2, 1, 4}, float32);
    array b = swapaxes(b_column, -1, -2, stream);
    expect_close(readback_f32(stream, matmul(a, b, stream)), expected, 1e-6);
  }

  SUBCASE("float16 transposed lhs and wide AddMM") {
    std::vector<float> a_storage{
        0.25f, -0.5f, 0.75f, 1.0f, -0.25f, 0.5f};
    std::vector<float> b_values{
        0.5f, -0.25f, 1.0f, 0.75f,
        -1.0f, 0.5f, 0.25f, -0.5f};
    std::vector<float> c_values{0.125f, -0.25f, 0.5f, 0.75f};
    array a_dense(a_storage.begin(), Shape{2, 3}, float32);
    array a = astype(swapaxes(a_dense, -1, -2, stream), float16, stream);
    array b = astype(
        array(b_values.begin(), Shape{2, 4}, float32), float16, stream);
    array c = astype(array(c_values.begin(), Shape{4}, float32), float16, stream);
    std::vector<float> expected(3 * 4);
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 4; ++column) {
        double sum = 0.0;
        for (int inner = 0; inner < 2; ++inner) {
          sum += host_at(a_storage, inner * 3 + row) *
              host_at(b_values, inner * 4 + column);
        }
        expected[row * 4 + column] =
            static_cast<float>(sum + c_values[column]);
      }
    }
    expect_close(
        readback_f32(stream, addmm(c, a, b, 1.0f, 1.0f, stream)),
        expected,
        1e-5);
  }
}

// Dense f32 matmul across the shapes the 8x8x8 cooperative-matrix gate
// accepts (dispatch_matmul, primitives.cpp) plus the shapes it must
// refuse. On a device reporting cooperative_matrix_f32_8 (Honeykrisp
// honeykrisp-coopmat branch, AGX_SIMDMAT=1) the gated cases run
// MatmulF32Coopmat; on llvmpipe and stock Mesa every case runs the 16x16
// tile. Operands are small integers so both kernels are exact and the
// comparison is equality, not a tolerance.
TEST_CASE("register-blocked f16 matmul matches the 16x16 tile bit for bit") {
  if (!compute_available()) {
    return;
  }
  if (!float16_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The 64x64 register-blocked kernel takes matrix_m >= 32; the same
  // product over a 16-row slice of A runs the 16x16 tile. Each output
  // visits k in the same ascending order with the same zero padding
  // to a multiple of 16, so the stored f16 bits must agree exactly.
  // Shapes follow the prefill attention matmuls: probs @ v (k tail
  // 1052 % 16 = 12, n = 64) and q @ k^T (k = 64, transposed b, n
  // tail), both with a batch of three.
  struct Case {
    int m;
    int k;
    int n;
    bool b_transposed;
  };
  auto bits = [](array x) {
    eval(x);
    return std::vector<uint16_t>(
        x.data<uint16_t>(), x.data<uint16_t>() + x.size());
  };
  for (const Case& c : {Case{40, 1052, 64, false}, Case{70, 64, 70, true},
                        Case{32, 16, 128, false}}) {
    Shape a_shape{3, c.m, c.k};
    Shape b_shape = c.b_transposed ? Shape{3, c.n, c.k} : Shape{3, c.k, c.n};
    std::vector<float> av(3 * c.m * c.k);
    std::vector<float> bv(3 * c.k * c.n);
    for (size_t i = 0; i < av.size(); ++i) {
      av[i] = std::sin(0.13f * static_cast<float>(i)) * 0.5f;
    }
    for (size_t i = 0; i < bv.size(); ++i) {
      bv[i] = std::cos(0.29f * static_cast<float>(i)) * 0.5f;
    }
    array a = astype(array(av.begin(), a_shape, float32), float16, stream);
    array b = astype(array(bv.begin(), b_shape, float32), float16, stream);
    if (c.b_transposed) {
      b = swapaxes(b, 1, 2, stream);
    }
    array full = matmul(a, b, stream);
    array head = matmul(slice(a, {0, 0, 0}, {3, 16, c.k}, stream), b, stream);
    auto full_bits = bits(full);
    auto head_bits = bits(head);
    size_t mismatches = 0;
    for (int batch = 0; batch < 3; ++batch) {
      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < c.n; ++col) {
          mismatches += full_bits[(batch * c.m + row) * c.n + col] !=
              head_bits[(batch * 16 + row) * c.n + col];
        }
      }
    }
    INFO("m=" << c.m << " k=" << c.k << " n=" << c.n
              << " bT=" << c.b_transposed);
    CHECK_EQ(mismatches, 0u);
    // Host reference under the f32-order anchor bound, so a shared
    // wrong answer cannot pass both kernels.
    array wide = astype(full, float32, stream);
    eval(wide);
    const float* got = wide.data<float>();
    float max_err = 0.0f;
    for (int batch = 0; batch < 3; ++batch) {
      for (int row = 0; row < c.m; ++row) {
        for (int col = 0; col < c.n; ++col) {
          double sum = 0.0;
          for (int inner = 0; inner < c.k; ++inner) {
            float x = float16_t(av[(batch * c.m + row) * c.k + inner]);
            float y = float16_t(
                c.b_transposed ? bv[(batch * c.n + col) * c.k + inner]
                               : bv[(batch * c.k + inner) * c.n + col]);
            sum += static_cast<double>(x) * y;
          }
          max_err = std::max(
              max_err,
              std::abs(got[(batch * c.m + row) * c.n + col] -
                       static_cast<float>(sum)));
        }
      }
    }
    CHECK_LT(max_err, 0.05f);
  }
}

TEST_CASE("direct matmul route: first matching row wins, else the shipped route") {
  using omarchy::ComputeKernel;
  using omarchy::DirectMatmulRoute;
  const DirectMatmulRoute shipped{ComputeKernel::MatmulDirectF16Nt, 64u};
  const DirectMatmulRoute wide{ComputeKernel::MatmulDirectF16Nn, 128u};
  const DirectMatmulRoute any_g13{ComputeKernel::MatmulDirectF16Tn, 64u};
  const std::array<omarchy::DirectMatmulRow, 2> rows{{
      {"G13G", float16, false, true, 128u, 64u, wide},
      {"G13", float16, false, true, 32u, 32u, any_g13},
  }};
  auto pick = [&](std::string_view device, Dtype dtype, bool a_t, bool b_t,
                  uint32_t m, uint32_t n) {
    return omarchy::select_direct_matmul_route(
        rows, device, dtype, a_t, b_t, m, n, shipped);
  };
  const auto m1 = pick("Apple M1 (G13G B1)", float16, false, true, 512u, 4096u);
  CHECK(m1.kernel == wide.kernel);
  CHECK_EQ(m1.tile_n, 128u);
  // Below the first row's m or n floor, the next matching row takes over.
  CHECK(pick("Apple M1 (G13G B1)", float16, false, true, 127u, 4096u).kernel ==
        any_g13.kernel);
  CHECK(pick("Apple M1 (G13G B1)", float16, false, true, 512u, 63u).kernel ==
        any_g13.kernel);
  // The chip key is a device-name substring.
  CHECK(pick("Apple M1 Max (G13C C0)", float16, false, true, 512u, 4096u).kernel ==
        any_g13.kernel);
  // Another chip, dtype, or orientation keeps the shipped route. bf16 and
  // f16 are distinct keys although both are 16-bit.
  const auto g14 = pick("Apple M2 (G14G B1)", float16, false, true, 512u, 4096u);
  CHECK(g14.kernel == shipped.kernel);
  CHECK_EQ(g14.tile_n, 64u);
  CHECK(pick("Apple M1 (G13G B1)", float32, false, true, 512u, 4096u).kernel ==
        shipped.kernel);
  CHECK(pick("Apple M1 (G13G B1)", bfloat16, false, true, 512u, 4096u).kernel ==
        shipped.kernel);
  CHECK(pick("Apple M1 (G13G B1)", float16, false, false, 512u, 4096u).kernel ==
        shipped.kernel);
  CHECK(pick("Apple M1 (G13G B1)", float16, true, true, 512u, 4096u).kernel ==
        shipped.kernel);
  // A 128-wide route never applies below n = 64, even if its row allows it.
  const std::array<omarchy::DirectMatmulRow, 1> loose{{
      {"G13G", float16, false, true, 32u, 32u, wide},
  }};
  CHECK(omarchy::select_direct_matmul_route(
            loose, "Apple M1 (G13G B1)", float16, false, true, 512u, 48u,
            shipped).kernel == shipped.kernel);
  CHECK_EQ(omarchy::select_direct_matmul_route(
               loose, "Apple M1 (G13G B1)", float16, false, true, 512u, 64u,
               shipped).tile_n, 128u);
}

TEST_CASE("direct matmul route: the shipped rows apply only on the measured chips") {
  using omarchy::ComputeKernel;
  const omarchy::DirectMatmulRoute shipped{ComputeKernel::Count, 0u};
  auto kernel = [&](std::string_view device, Dtype dtype, bool a_t, bool b_t) {
    return omarchy::select_direct_matmul_route(
        omarchy::kDirectMatmulRows, device, dtype, a_t, b_t, 4096u, 4096u,
        shipped).kernel;
  };
  for (std::string_view chip : {"Apple M1 (G13G B1)", "Apple M1 Max (G13C C0)"}) {
    CHECK(kernel(chip, float16, false, true) == ComputeKernel::MatmulDirectF16NtK4S8);
    CHECK(kernel(chip, float16, true, false) == ComputeKernel::MatmulDirectF16TnWS8);
    CHECK(kernel(chip, float16, false, false) == shipped.kernel);
    CHECK(kernel(chip, bfloat16, true, false) == ComputeKernel::MatmulDirectBF16TnWS8);
    CHECK(kernel(chip, float32, false, true) == ComputeKernel::MatmulDirectF32NtK4S8);
  }
  // Per-chip rows (MatmulGap H15, H16).
  CHECK(kernel("Apple M1 Max (G13C C0)", bfloat16, false, true) ==
        ComputeKernel::MatmulDirectBF16NtK4S8);
  CHECK(kernel("Apple M1 (G13G B1)", bfloat16, false, true) ==
        ComputeKernel::MatmulDirectBF16Nt);
  // No G13G f32 a @ b row: it lost 8 to 42 % at k 2560 and at n 9728 (MatmulGap H36).
  CHECK(kernel("Apple M1 (G13G B1)", float32, false, false) == shipped.kernel);
  CHECK(kernel("Apple M1 (G13G B1)", float32, true, false) ==
        ComputeKernel::MatmulDirectF32TnWS8);
  CHECK(kernel("Apple M1 Max (G13C C0)", float32, false, false) == shipped.kernel);
  CHECK(kernel("Apple M1 Max (G13C C0)", float32, true, false) == shipped.kernel);
  // M2 Max (device name per receipts/2026-10-04-hwprobe-device-info): three
  // rows (MatmulGap H20); every other cell keeps the shipped route.
  constexpr std::string_view g14c = "Apple M2 Max (G14C B1)";
  CHECK(kernel(g14c, bfloat16, true, false) ==
        ComputeKernel::MatmulDirectBF16TnWS8);
  CHECK(kernel(g14c, bfloat16, false, true) ==
        ComputeKernel::MatmulDirectBF16NtK4S8);
  CHECK(kernel(g14c, float16, false, true) ==
        ComputeKernel::MatmulDirectF16NtK2S8);
  for (Dtype dtype : {float16, bfloat16, float32}) {
    for (bool a_t : {false, true}) {
      for (bool b_t : {false, true}) {
        bool row = (dtype == bfloat16 && a_t != b_t) ||
            (dtype == float16 && !a_t && b_t);
        if (!row) {
          CHECK(kernel(g14c, dtype, a_t, b_t) == shipped.kernel);
        }
      }
    }
  }
}

TEST_CASE("direct matmul route: every row starts at its measured m floor") {
  using omarchy::ComputeKernel;
  const omarchy::DirectMatmulRoute shipped{ComputeKernel::Count, 0u};
  struct Floor {
    std::string_view device;
    Dtype dtype;
    bool a_t;
    bool b_t;
    uint32_t m;
    ComputeKernel kernel;
  };
  const Floor floors[] = {
      {"Apple M1 (G13G B1)", float16, false, true, 512u, ComputeKernel::MatmulDirectF16NtK4S8},
      {"Apple M1 (G13G B1)", float16, true, false, 1024u, ComputeKernel::MatmulDirectF16TnWS8},
      {"Apple M1 (G13G B1)", bfloat16, true, false, 512u, ComputeKernel::MatmulDirectBF16TnWS8},
      {"Apple M1 (G13G B1)", bfloat16, false, true, 4096u, ComputeKernel::MatmulDirectBF16Nt},
      {"Apple M1 (G13G B1)", float32, false, true, 512u, ComputeKernel::MatmulDirectF32NtK4S8},
      {"Apple M1 (G13G B1)", float32, true, false, 512u, ComputeKernel::MatmulDirectF32TnWS8},
      {"Apple M1 Max (G13C C0)", float16, false, true, 512u, ComputeKernel::MatmulDirectF16NtK4S8},
      {"Apple M1 Max (G13C C0)", float16, true, false, 4096u, ComputeKernel::MatmulDirectF16TnWS8},
      {"Apple M1 Max (G13C C0)", bfloat16, true, false, 4096u, ComputeKernel::MatmulDirectBF16TnWS8},
      {"Apple M1 Max (G13C C0)", bfloat16, false, true, 512u, ComputeKernel::MatmulDirectBF16NtK4S8},
      {"Apple M1 Max (G13C C0)", float32, false, true, 4096u, ComputeKernel::MatmulDirectF32NtK4S8},
      {"Apple M2 Max (G14C B1)", float16, false, true, 1024u, ComputeKernel::MatmulDirectF16NtK2S8},
      {"Apple M2 Max (G14C B1)", bfloat16, false, true, 512u, ComputeKernel::MatmulDirectBF16NtK4S8},
      {"Apple M2 Max (G14C B1)", bfloat16, true, false, 4096u, ComputeKernel::MatmulDirectBF16TnWS8},
  };
  for (const auto& f : floors) {
    CAPTURE(f.device);
    CAPTURE(f.m);
    auto route = [&](uint32_t m, uint32_t n) {
      return omarchy::select_direct_matmul_route(
          omarchy::kDirectMatmulRows, f.device, f.dtype, f.a_t, f.b_t, m, n,
          shipped).kernel;
    };
    CHECK(route(f.m, 4096u) == f.kernel);
    CHECK(route(f.m * 2u, 4096u) == f.kernel);
    CHECK(route(f.m - 1u, 4096u) == shipped.kernel);
    CHECK(route(f.m, 4095u) == shipped.kernel);
  }
  // One floors entry per table row: a row added without its entry fails here.
  CHECK_EQ(std::size(floors), omarchy::kDirectMatmulRows.size());
}

TEST_CASE("direct cooperative-matrix matmul matches the 16-row slices in every orientation") {
  if (!compute_available()) {
    return;
  }
  if (!float16_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // On a cooperative-matrix device, f16 and f32 with m >= 32 take
  // MatmulDirect, and bf16 does too where the bf16 shape exists
  // (cooperative_matrix_bf16_8); every 16-row slice of A runs the 16x16
  // tile (f16, bf16) or the staged coopmat tile (f32). All keep one f32
  // accumulator over ascending k on the same operand values, so every
  // stored bit must agree - including the edge tiles that shift back to
  // m - 32 / n - 32 and recompute their neighbour's outputs.
  struct Case {
    int m;
    int k;
    int n;
    bool a_t;
    bool b_t;
  };
  auto bytes = [](array x) {
    eval(x);
    const uint8_t* p = x.data<uint8_t>();
    return std::vector<uint8_t>(p, p + x.nbytes());
  };
  std::vector<Dtype> dtypes{float16, float32};
  if (omarchy::device(0).capabilities().cooperative_matrix_bf16_8) {
    dtypes.push_back(bfloat16);
  }
  for (Dtype dtype : dtypes) {
    // The first five shapes are small. The last seven are above the route-row
    // floors with n = 4096 and m not a multiple of 64 (520, 1030, 2054, 4100), so
    // on a chip with rows the row kernels run their partial row tiles; k = 32
    // keeps the slice reference cheap. Partial column tiles (n = 4098) are
    // covered by the shape sweep in the receipt, not here.
    for (const Case& c :
         {Case{142, 88, 200, false, false}, Case{142, 40, 72, false, true},
          Case{142, 64, 72, true, false}, Case{142, 16, 34, true, true},
          Case{32, 8, 32, false, false}, Case{520, 32, 4096, false, true},
          Case{520, 32, 4096, true, false}, Case{520, 32, 4096, false, false},
          Case{1030, 32, 4096, false, true}, Case{1030, 32, 4096, true, false},
          Case{2054, 32, 4096, false, false}, Case{4100, 32, 4096, true, false}}) {
      std::vector<float> av(2 * c.m * c.k);
      std::vector<float> bv(2 * c.k * c.n);
      for (size_t i = 0; i < av.size(); ++i) {
        av[i] = std::sin(0.37f * static_cast<float>(i));
      }
      for (size_t i = 0; i < bv.size(); ++i) {
        bv[i] = std::cos(0.23f * static_cast<float>(i));
      }
      array a = astype(
          array(av.begin(), c.a_t ? Shape{2, c.k, c.m} : Shape{2, c.m, c.k},
                float32),
          dtype, stream);
      array b = astype(
          array(bv.begin(), c.b_t ? Shape{2, c.n, c.k} : Shape{2, c.k, c.n},
                float32),
          dtype, stream);
      if (c.a_t) {
        a = swapaxes(a, 1, 2, stream);
      }
      if (c.b_t) {
        b = swapaxes(b, 1, 2, stream);
      }
      const size_t row_bytes = static_cast<size_t>(c.n) * size_of(dtype);
      auto full = bytes(matmul(a, b, stream));
      size_t mismatched_rows = 0;
      for (int row0 = 0; row0 < c.m; row0 += 16) {
        int rows = std::min(16, c.m - row0);
        auto part = bytes(matmul(
            slice(a, {0, row0, 0}, {2, row0 + rows, c.k}, stream), b,
            stream));
        for (int batch = 0; batch < 2; ++batch) {
          for (int row = 0; row < rows; ++row) {
            mismatched_rows += std::memcmp(
                full.data() + (batch * c.m + row0 + row) * row_bytes,
                part.data() + (batch * rows + row) * row_bytes,
                row_bytes) != 0;
          }
        }
      }
      INFO("dtype=" << dtype << " m=" << c.m << " k=" << c.k << " n=" << c.n
                    << " aT=" << c.a_t << " bT=" << c.b_t);
      CHECK_EQ(mismatched_rows, 0u);
    }
  }
}

TEST_CASE("dense f32 matmul matches host across coopmat-gated shapes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& caps = omarchy::device(0).capabilities();
  const bool coopmat_device =
      caps.cooperative_matrix_f32_8 && caps.subgroup_size == 32;
  std::cout << "[provenance] cooperative_matrix_f32_8="
            << (caps.cooperative_matrix_f32_8 ? 1 : 0)
            << " subgroup_size=" << caps.subgroup_size << " -> gated shapes run "
            << (coopmat_device ? "MatmulF32Coopmat" : "MatmulF32 (16x16 tile)")
            << "\n";

  std::mt19937 rng(20260908);
  std::uniform_int_distribution<int> small(-4, 4);
  auto integers = [&](size_t count) {
    std::vector<float> values(count);
    for (auto& v : values) {
      v = static_cast<float>(small(rng));
    }
    return values;
  };
  // Operand a is (batch, m, k) row-major on the host; the device view is
  // either that array or a transposed view of its (batch, k, m) copy so
  // the kernel sees the column-major flag with the same values.
  auto operand = [&](const std::vector<float>& values,
                     int batch,
                     int rows,
                     int cols,
                     bool transposed) {
    if (!transposed) {
      return array(values.begin(), Shape{batch, rows, cols}, float32);
    }
    std::vector<float> swapped(values.size());
    for (int bt = 0; bt < batch; ++bt) {
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          swapped[(static_cast<size_t>(bt) * cols + c) * rows + r] =
              values[(static_cast<size_t>(bt) * rows + r) * cols + c];
        }
      }
    }
    return transpose(
        array(swapped.begin(), Shape{batch, cols, rows}, float32),
        {0, 2, 1},
        stream);
  };
  auto run = [&](const char* label,
                 int batch,
                 int m,
                 int k,
                 int n,
                 bool a_t,
                 bool b_t,
                 bool gated) {
    INFO(label, " batch=", batch, " m=", m, " k=", k, " n=", n,
         " a_t=", a_t, " b_t=", b_t, " path=",
         (gated && coopmat_device ? "coopmat" : "tiled"));
    auto a_values = integers(static_cast<size_t>(batch) * m * k);
    auto b_values = integers(static_cast<size_t>(batch) * k * n);
    std::vector<float> expected = host_matmul(a_values, b_values, batch, m, k, n);
    std::vector<float> got = readback_f32(
        stream,
        matmul(
            operand(a_values, batch, m, k, a_t),
            operand(b_values, batch, k, n, b_t),
            stream));
    REQUIRE_EQ(got.size(), expected.size());
    float max_abs_err = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
      REQUIRE(std::isfinite(got[i]));
      max_abs_err = std::max(max_abs_err, std::fabs(got[i] - expected[i]));
    }
    std::cout << "[matmul-f32] " << label << " " << batch << "x" << m << "x"
              << k << "x" << n << (a_t ? " aT" : " a") << (b_t ? " bT" : " b")
              << " path=" << (gated && coopmat_device ? "coopmat" : "tiled")
              << " max_abs_err=" << max_abs_err << "\n";
    CHECK_EQ(max_abs_err, 0.0f);
  };

  struct Case {
    const char* label;
    int batch, m, k, n;
    bool gated;
  };
  const Case cases[] = {
      {"cube8", 1, 8, 8, 8, true},
      {"cube64", 1, 64, 64, 64, true},
      {"cube128", 1, 128, 128, 128, true},
      {"batched4x32", 4, 32, 32, 32, true},
      {"odd17x23x19", 1, 17, 23, 19, false},
  };
  for (const auto& c : cases) {
    for (bool a_t : {false, true}) {
      for (bool b_t : {false, true}) {
        run(c.label, c.batch, c.m, c.k, c.n, a_t, b_t, c.gated);
      }
    }
  }

  // Nonzero element offsets through slices of a larger parent. A slice at
  // a 16-byte aligned offset with 4-aligned row gaps keeps the coopmat
  // gate; a slice one column over breaks the alignment and must take the
  // tiled kernel on every device.
  {
    const int m = 32, k = 40, n = 24;
    const int pad_rows = 8, pad_cols = 12;
    auto parent_a = integers(static_cast<size_t>(m + pad_rows) * (k + pad_cols));
    auto parent_b = integers(static_cast<size_t>(k + pad_rows) * (n + pad_cols));
    array pa(parent_a.begin(), Shape{m + pad_rows, k + pad_cols}, float32);
    array pb(parent_b.begin(), Shape{k + pad_rows, n + pad_cols}, float32);
    for (int column0 : {4, 5}) {
      const bool gated = column0 % 4 == 0;
      INFO("slice column offset ", column0, " path=",
           (gated && coopmat_device ? "coopmat" : "tiled"));
      std::vector<float> a_values(static_cast<size_t>(m) * k);
      std::vector<float> b_values(static_cast<size_t>(k) * n);
      for (int r = 0; r < m; ++r) {
        for (int c = 0; c < k; ++c) {
          a_values[static_cast<size_t>(r) * k + c] = parent_a
              [static_cast<size_t>(r + pad_rows) * (k + pad_cols) + c + column0];
        }
      }
      for (int r = 0; r < k; ++r) {
        for (int c = 0; c < n; ++c) {
          b_values[static_cast<size_t>(r) * n + c] = parent_b
              [static_cast<size_t>(r + pad_rows) * (n + pad_cols) + c + column0];
        }
      }
      std::vector<float> expected = host_matmul(a_values, b_values, 1, m, k, n);
      array a = slice(pa, {pad_rows, column0}, {pad_rows + m, column0 + k}, stream);
      array b = slice(pb, {pad_rows, column0}, {pad_rows + k, column0 + n}, stream);
      std::vector<float> got = readback_f32(stream, matmul(a, b, stream));
      REQUIRE_EQ(got.size(), expected.size());
      float max_abs_err = 0.0f;
      for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(std::isfinite(got[i]));
        max_abs_err = std::max(max_abs_err, std::fabs(got[i] - expected[i]));
      }
      std::cout << "[matmul-f32] slice col" << column0 << " " << m << "x" << k
                << "x" << n << " path="
                << (gated && coopmat_device ? "coopmat" : "tiled")
                << " max_abs_err=" << max_abs_err << "\n";
      CHECK_EQ(max_abs_err, 0.0f);
    }
  }
}

// Round-to-nearest-even bf16 quantization of an f32 value: the exact
// arithmetic of the kernel drain's bf16_store (NaN quiet bit included),
// so a host f32 sum can be folded onto the bf16 grid the way the shader
// folds it.
float host_bf16_round(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if (std::isnan(value)) {
    bits = (bits >> 16) | 0x40u;
  } else {
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
  }
  bits <<= 16;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// Double-precision host matmul over (batch, m, k) x (batch, k, n), k
// innermost per row for cache locality, rows split across threads so the
// 1053x4864x4864 model shapes stay inside a single GPU-lock window.
std::vector<float> host_matmul_bf16_reference(
    const std::vector<float>& a, // (batch, m, k)
    const std::vector<float>& b, // (batch, k, n)
    int batch,
    int m,
    int k,
    int n) {
  std::vector<float> out(static_cast<size_t>(batch) * m * n, 0.0f);
  unsigned threads = std::thread::hardware_concurrency();
  if (threads == 0u) {
    threads = 4u;
  }
  threads = std::min(threads, static_cast<unsigned>(m));
  auto worker = [&](unsigned t) {
    size_t row_begin = static_cast<size_t>(m) * t / threads;
    size_t row_end = static_cast<size_t>(m) * (t + 1u) / threads;
    for (int bt = 0; bt < batch; ++bt) {
      for (size_t r = row_begin; r < row_end; ++r) {
        std::vector<double> acc(n, 0.0);
        const float* a_row =
            &a[(static_cast<size_t>(bt) * m + r) * k];
        for (int inner = 0; inner < k; ++inner) {
          double av = a_row[inner];
          const float* b_row = &b[(static_cast<size_t>(bt) * k + inner) * n];
          for (int c = 0; c < n; ++c) {
            acc[c] += av * b_row[c];
          }
        }
        for (int c = 0; c < n; ++c) {
          out[(static_cast<size_t>(bt) * m + r) * n + c] =
              static_cast<float>(acc[c]);
        }
      }
    }
  };
  std::vector<std::thread> pool;
  for (unsigned t = 1u; t < threads; ++t) {
    pool.emplace_back(worker, t);
  }
  worker(0u);
  for (auto& thread : pool) {
    thread.join();
  }
  return out;
}

// Dense bf16 matmul against an independent host reference, proving EVERY
// output row and column of the staged 32x32 cooperative-matrix kernel
// (MatmulBF16Coopmat) across tile tails, the Qwen2.5-0.5B prefill shapes,
// all four transpose orientations, and the even-element offset/batch
// gates. Operands are small integers: exactly representable in bf16 with
// f32-exact partial sums (|sum| < 2^24), so every accumulation order -
// coopmat 8-wide blocks and the 16x16 tiled fallback alike - lands on
// the same f32 value and the comparison is exact equality after one
// round-to-nearest-even bf16 quantization. An earlier verification of
// this kernel passed while checking row 0 only, but the drain then wrote
// only 4 rows of each 8-row block and left the rest stale, so this case
// pins the full matrix and asserts each expected row carries a nonzero
// element: an unwritten or stale row cannot pass.
TEST_CASE("dense bf16 matmul matches host on every row across coopmat shapes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& caps = omarchy::device(0).capabilities();
  const bool coopmat_device =
      caps.cooperative_matrix_f32_8 && caps.subgroup_size == 32;
  std::cout << "[provenance] cooperative_matrix_f32_8="
            << (caps.cooperative_matrix_f32_8 ? 1 : 0)
            << " subgroup_size=" << caps.subgroup_size << "\n";

  std::mt19937 rng(20260908);
  std::uniform_int_distribution<int> small(-4, 4);
  auto integers = [&](size_t count) {
    std::vector<float> values(count);
    for (auto& v : values) {
      v = static_cast<float>(small(rng));
    }
    return values;
  };
  // Operand a is (batch, m, k) row-major on the host; the transposed
  // device view carries the column-major flag with the same values.
  auto bf16_operand = [&](const std::vector<float>& values,
                          int batch,
                          int rows,
                          int cols,
                          bool transposed) {
    if (!transposed) {
      return astype(
          array(values.begin(), Shape{batch, rows, cols}, float32),
          bfloat16,
          stream);
    }
    std::vector<float> swapped(values.size());
    for (int bt = 0; bt < batch; ++bt) {
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          swapped[(static_cast<size_t>(bt) * cols + c) * rows + r] =
              values[(static_cast<size_t>(bt) * rows + r) * cols + c];
        }
      }
    }
    return transpose(
        astype(
            array(swapped.begin(), Shape{batch, cols, rows}, float32),
            bfloat16,
            stream),
        {0, 2, 1},
        stream);
  };
  // One host reference per VALUE array pair; every orientation of the
  // same shape reuses it, then each device result is compared over the
  // full matrix with per-row diagnostics.
  auto run = [&](const char* label,
                 int batch,
                 int m,
                 int k,
                 int n,
                 const std::vector<float>& a_values,
                 const std::vector<float>& b_values) {
    std::vector<float> expected(
        static_cast<size_t>(batch) * m * n);
    {
      auto sums =
          host_matmul_bf16_reference(a_values, b_values, batch, m, k, n);
      for (size_t i = 0; i < sums.size(); ++i) {
        expected[i] = host_bf16_round(sums[i]);
      }
    }
    size_t dead_rows = 0;
    for (size_t i = 0; i < expected.size(); i += static_cast<size_t>(n)) {
      float row_max = 0.0f;
      for (int c = 0; c < n; ++c) {
        row_max = std::max(row_max, std::fabs(expected[i + c]));
      }
      if (row_max == 0.0f) {
        ++dead_rows;
      }
    }
    CHECK_EQ(dead_rows, size_t{0});
    for (bool a_t : {false, true}) {
      for (bool b_t : {false, true}) {
        std::vector<float> got = readback_f32(
            stream,
            matmul(
                bf16_operand(a_values, batch, m, k, a_t),
                bf16_operand(b_values, batch, k, n, b_t),
                stream));
        REQUIRE_EQ(got.size(), expected.size());
        float max_abs_err = 0.0f;
        size_t bad_rows = 0;
        for (size_t base = 0, i = 0; base < expected.size();
             base += static_cast<size_t>(n)) {
          float row_err = 0.0f;
          for (int c = 0; c < n; ++c, ++i) {
            REQUIRE(std::isfinite(got[i]));
            row_err = std::max(row_err, std::fabs(got[i] - expected[i]));
          }
          if (row_err > 0.0f) {
            ++bad_rows;
            if (bad_rows <= 4u) {
              std::cout << "  [row-mismatch] " << label << " row="
                        << (base / static_cast<size_t>(n))
                        << " max_err=" << row_err << "\n";
            }
          }
          max_abs_err = std::max(max_abs_err, row_err);
        }
        std::cout << "[matmul-bf16] " << label << " " << batch << "x" << m
                  << "x" << k << "x" << n << (a_t ? " aT" : " a")
                  << (b_t ? " bT" : " b")
                  << " max_abs_err=" << max_abs_err << "\n";
        CHECK_EQ(max_abs_err, 0.0f);
      }
    }
  };

  // Tile tails: m tails, n tails across two tile columns, single and
  // multi k steps, and the batch unravel. m8 keeps every row of the one
  // 8-row block honest - rows 4..7 were exactly the old drain's
  // casualties. K is a multiple of 8 and n stays even (bf16 word
  // packing), matching the gate.
  struct TailCase {
    const char* label;
    int batch, m, k, n;
  };
  const TailCase tails[] = {
      {"tail-m8", 1, 8, 8, 8},
      {"tail-m9", 1, 9, 8, 8},
      {"tail-m33-n40", 1, 33, 8, 40},
      {"tile32", 1, 32, 32, 32},
      {"tail-m17-k24-n34", 1, 17, 24, 34},
      {"batch2", 2, 40, 8, 40},
      // m >= 32 with a k % 16 == 8 tail and k < 16: on coopmat devices
      // these route to MatmulBF16Coopmat and so exercise its 16-wide
      // staging's final 8-wide step and its k / 16 == 0 degenerate loop.
      {"tail-m32-k24-n34", 1, 32, 24, 34},
      {"tail-m33-k40-n18", 1, 33, 40, 18},
      {"tail-m48-k8-n16", 1, 48, 8, 16},
  };
  for (const auto& t : tails) {
    run(
        t.label,
        t.batch,
        t.m,
        t.k,
        t.n,
        integers(static_cast<size_t>(t.batch) * t.m * t.k),
        integers(static_cast<size_t>(t.batch) * t.k * t.n));
  }

  // Gate refusals must take the 16x16 tiled kernel on every device.
  run("single-row", 1, 1, 8, 8, integers(8), integers(64));
  run("k12", 1, 17, 12, 34, integers(17 * 12), integers(12 * 34));
  run("odd-n", 1, 17, 8, 17, integers(17 * 8), integers(8 * 17));

  // Qwen2.5-0.5B prefill shapes: token counts M in {30, 262, 1053} against
  // hidden/intermediate K and N in {896, 4864}, every combination, all
  // four orientations. On the coopmat device these run MatmulBF16Coopmat;
  // the tiled fallback needs minutes per jumbo run on llvmpipe, so
  // software devices prove the shape plumbing on the smallest combination
  // only and the full sweep is a hardware leg.
  for (int m : {30, 262, 1053}) {
    for (int k : {896, 4864}) {
      for (int n : {896, 4864}) {
        if (!coopmat_device && (m != 30 || k != 896 || n != 896)) {
          continue;
        }
        run(
            "model",
            1,
            m,
            k,
            n,
            integers(static_cast<size_t>(m) * k),
            integers(static_cast<size_t>(k) * n));
      }
    }
  }

  // Nonzero element offsets through slices of a larger bf16 parent. An
  // even slice offset with even row gaps keeps the coopmat gate; one
  // column over makes the element offset odd and must take the tiled
  // kernel on every device.
  {
    const int m = 32, k = 40, n = 24;
    const int pad_rows = 8, pad_cols = 12;
    auto parent_a_values = integers(static_cast<size_t>(m + pad_rows) * (k + pad_cols));
    auto parent_b_values = integers(static_cast<size_t>(k + pad_rows) * (n + pad_cols));
    for (int column0 : {4, 5}) {
      std::vector<float> a_values(static_cast<size_t>(m) * k);
      std::vector<float> b_values(static_cast<size_t>(k) * n);
      for (int r = 0; r < m; ++r) {
        for (int c = 0; c < k; ++c) {
          a_values[static_cast<size_t>(r) * k + c] = parent_a_values
              [static_cast<size_t>(r + pad_rows) * (k + pad_cols) + c + column0];
        }
      }
      for (int r = 0; r < k; ++r) {
        for (int c = 0; c < n; ++c) {
          b_values[static_cast<size_t>(r) * n + c] = parent_b_values
              [static_cast<size_t>(r + pad_rows) * (n + pad_cols) + c + column0];
        }
      }
      std::vector<float> expected(static_cast<size_t>(m) * n);
      {
        auto sums = host_matmul_bf16_reference(a_values, b_values, 1, m, k, n);
        for (size_t i = 0; i < sums.size(); ++i) {
          expected[i] = host_bf16_round(sums[i]);
        }
      }
      array parent_a = astype(
          array(parent_a_values.begin(),
                Shape{m + pad_rows, k + pad_cols},
                float32),
          bfloat16,
          stream);
      array parent_b = astype(
          array(parent_b_values.begin(),
                Shape{k + pad_rows, n + pad_cols},
                float32),
          bfloat16,
          stream);
      array a = slice(
          parent_a, {pad_rows, column0}, {pad_rows + m, column0 + k}, stream);
      array b = slice(
          parent_b, {pad_rows, column0}, {pad_rows + k, column0 + n}, stream);
      std::vector<float> got = readback_f32(stream, matmul(a, b, stream));
      REQUIRE_EQ(got.size(), expected.size());
      float max_abs_err = 0.0f;
      for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(std::isfinite(got[i]));
        max_abs_err = std::max(max_abs_err, std::fabs(got[i] - expected[i]));
      }
      std::cout << "[matmul-bf16] slice col" << column0 << " " << m << "x" << k
                << "x" << n
                << " max_abs_err=" << max_abs_err << "\n";
      CHECK_EQ(max_abs_err, 0.0f);
    }
  }

  // Fractional bf16 operands (pre-quantized to the bf16 grid so host and
  // device widen identical values): accumulation order now rounds in f32,
  // so the check moves from exact equality to the documented anchor bound
  // |got - exact| <= P*(k_steps*4.5*2^-23) + ulp_bf16 amplification, with
  // P the largest partial-sum magnitude. The recorded err against the
  // exact double reference is the evidence; the bound is never tuned to
  // pass.
  {
    const int m = 33, k = 24, n = 40;
    std::uniform_real_distribution<float> frac(-2.0f, 2.0f);
    auto grid = [&](size_t count) {
      std::vector<float> values(count);
      for (auto& v : values) {
        v = host_bf16_round(frac(rng));
      }
      return values;
    };
    auto a_values = grid(static_cast<size_t>(m) * k);
    auto b_values = grid(static_cast<size_t>(k) * n);
    auto sums = host_matmul_bf16_reference(a_values, b_values, 1, m, k, n);
    std::vector<float> expected(sums.size());
    float partial_max = 0.0f;
    for (size_t i = 0; i < sums.size(); ++i) {
      partial_max = std::max(partial_max, std::fabs(sums[i]));
      expected[i] = host_bf16_round(sums[i]);
    }
    const float bound = partial_max *
            (static_cast<float>(k / 8) * 4.5f * 0x1p-23f + 2.0f / 256.0f) +
        2.0f / 256.0f;
    std::vector<float> got = readback_f32(
        stream,
        matmul(
            bf16_operand(a_values, 1, m, k, false),
            bf16_operand(b_values, 1, k, n, false),
            stream));
    REQUIRE_EQ(got.size(), expected.size());
    float max_abs_err = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
      REQUIRE(std::isfinite(got[i]));
      max_abs_err = std::max(max_abs_err, std::fabs(got[i] - expected[i]));
    }
    std::cout << "[matmul-bf16] fractional 33x24x40 bound=" << bound
              << " max_abs_err=" << max_abs_err << " partial_max="
              << partial_max << "\n";
    CHECK(max_abs_err <= bound);
  }
}

// The dense BF16 decode GEMV (packed 4-wide bf16 loads, 128-wide k stripes
// per subgroup, five shuffle-down reduction levels) changed the f32
// accumulation order of every single-row projection, so the generated-token
// digests moved to new per-driver pins (recorded in
// receipts/2026-09-10-bf16-decode-gemv-land). This case pins the accuracy
// the digests no longer carry: the device result must stay inside the
// documented f32 accumulation bound against the float64 host reference on
// every Qwen2.5-0.5B decode projection shape that selects the kernel
// (m=1, k % 128 == 0, n % 4 == 0, transposed weight), on the deep-
// cancellation regime the projections are known to hit, and on the
// fallback shapes that must keep the general tile kernel. The bound is the
// existing fractional-case anchor recomputed for the wider step count
// (k/8 tile steps vs k/32 lane steps + 5 tree levels, so the tile count
// stays the conservative envelope on every device); it is never tuned to
// pass. No digest appears here on purpose.
TEST_CASE("single-row bf16 decode matmul stays inside the f64 bound") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& caps = omarchy::device(0).capabilities();
  const bool subgroup_kernel =
      caps.subgroup_size == 32u &&
      (caps.subgroup_operations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) != 0;
  std::cout << "[provenance] subgroup_size=" << caps.subgroup_size
            << " decode_gemv_path="
            << (subgroup_kernel ? "subgroup-shuffle" : "tile-fallback")
            << "\n";

  std::mt19937 rng(20260910);
  std::uniform_real_distribution<float> frac(-2.0f, 2.0f);
  // Operands pre-quantized to the bf16 grid so host and device widen
  // identical values; the float64 reference then sums exact products.
  auto grid = [&](size_t count) {
    std::vector<float> values(count);
    for (auto& v : values) {
      v = host_bf16_round(frac(rng));
    }
    return values;
  };
  // One check per shape. Weights live in the real [n, k] transposed-operand
  // layout (no host-side swap copy, lm_head included); the f64 reference
  // reads the same flat buffer in [k, n] order. Records the measured error
  // so the receipt can quote it as evidence.
  auto check = [&](const char* label,
                   int k,
                   int n,
                   const std::vector<float>& a_values,
                   const std::vector<float>& w_nt) {
    std::vector<float> sums(n, 0.0f);
    float partial_max = 0.0f;
    for (int c = 0; c < n; ++c) {
      double acc = 0.0;
      for (int r = 0; r < k; ++r) {
        acc += static_cast<double>(a_values[r]) *
            w_nt[(static_cast<size_t>(c) * k) + r];
      }
      sums[c] = static_cast<float>(acc);
      partial_max = std::max(partial_max, std::fabs(sums[c]));
    }
    // fma rounding envelope (4.5 * 2^-23 per step) over the larger of the
    // two shipped step counts (k/8 tile steps vs k/32 lane steps + 5 tree
    // levels), plus one bf16 ulp pair for the kernel's final
    // round-to-nearest-even store and one outside for readback.
    const float steps = static_cast<float>(k / 8 + 5);
    const float bound = partial_max *
            (steps * 4.5f * 0x1p-23f + 2.0f / 256.0f) +
        2.0f / 256.0f;
    array a = astype(
        array(a_values.begin(), Shape{1, k}, float32), bfloat16, stream);
    array b = transpose(
        astype(
            array(w_nt.begin(), Shape{n, k}, float32),
            bfloat16,
            stream),
        {1, 0},
        stream);
    std::vector<float> got = readback_f32(stream, matmul(a, b, stream));
    REQUIRE_EQ(got.size(), sums.size());
    float max_abs_err = 0.0f;
    for (size_t i = 0; i < sums.size(); ++i) {
      REQUIRE(std::isfinite(got[i]));
      max_abs_err = std::max(max_abs_err, std::fabs(got[i] - sums[i]));
    }
    std::cout << "[matmul-bf16-decode] " << label << " 1x" << k << "x" << n
              << " bT bound=" << bound
              << " max_abs_err_vs_f64=" << max_abs_err
              << " partial_max=" << partial_max << "\n";
    CHECK(max_abs_err <= bound);
  };

  // Every real decode projection selects the new kernel on a subgroup-32
  // device (k and n all satisfy the gate) and the tile kernel on
  // llvmpipe; the bound must hold under either order.
  const struct {
    const char* label;
    int k, n;
  } projections[] = {
      {"q_proj", 896, 896},
      {"k_proj", 896, 128},
      {"v_proj", 896, 128},
      {"o_proj", 896, 896},
      {"gate_proj", 896, 4864},
      {"down_proj", 4864, 896},
      {"lm_head", 896, 151936},
  };
  for (const auto& p : projections) {
    check(
        p.label,
        p.k,
        p.n,
        grid(p.k),
        grid(static_cast<size_t>(p.k) * p.n));
  }

  // Deep cancellation: weight columns alternate sign so partial sums grow
  // to k * c^2 while the exact result stays near zero - the regime where
  // macOS Metal rounds real projection elements to zero and this backend
  // stays RNE(f64)-exact (receipts/2026-09-10-bf16-decode-gemv-requal).
  {
    const int k = 896, n = 128;
    std::vector<float> a_values = grid(k);
    std::vector<float> w_nt(static_cast<size_t>(k) * n);
    for (int c = 0; c < n; ++c) {
      for (int r = 0; r < k; ++r) {
        w_nt[(static_cast<size_t>(c) * k) + r] =
            (r % 2 == 0 ? 1.0f : -1.0f) * a_values[r];
      }
    }
    check("cancel896x128", k, n, a_values, w_nt);
  }

  // Gate refusals must keep the general tile kernel everywhere: k not a
  // multiple of 128, n not a multiple of 4. Fractional operands keep the
  // f64 anchor honest on the fallback path too.
  check("fallback-k832", 832, 896, grid(832), grid(832 * 896));
  check("fallback-n897", 896, 897, grid(896), grid(896 * 897));
}

TEST_CASE("block masked mm zeroes and scales blocks") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(11);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::uniform_int_distribution<int> bit(0, 1);
  constexpr int bs = 32;

  auto build_masks = [&](int batch, int tm, int tk, int tn) {
    std::vector<uint8_t> lhs(static_cast<size_t>(batch) * tm * tk);
    std::vector<uint8_t> rhs(static_cast<size_t>(batch) * tk * tn);
    for (auto& value : lhs) {
      value = static_cast<uint8_t>(bit(gen));
    }
    for (auto& value : rhs) {
      value = static_cast<uint8_t>(bit(gen));
    }
    return std::make_pair(lhs, rhs);
  };

  // Operand masks, batched: 4 inputs, bool grids with both keep and
  // drop blocks present.
  {
    int batch = 2, m = 48, k = 64, n = 48;
    int tm = 2, tk = 2, tn = 2;
    auto [lhs_v, rhs_v] = build_masks(batch, tm, tk, tn);
    std::vector<float> a_values(static_cast<size_t>(batch) * m * k);
    std::vector<float> b_values(static_cast<size_t>(batch) * k * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    array a(a_values.begin(), Shape{batch, m, k}, float32);
    array b(b_values.begin(), Shape{batch, k, n}, float32);
    array mask_lhs(lhs_v.begin(), Shape{batch, tm, tk}, bool_);
    array mask_rhs(rhs_v.begin(), Shape{batch, tk, tn}, bool_);
    array out =
        block_masked_mm(a, b, bs, std::nullopt, mask_lhs, mask_rhs, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> expected = block_masked_reference(
        a_values, b_values, lhs_v, rhs_v, nullptr, batch, m, k, n, bs);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // Bool masks with transposed block axes and a broadcast batch must be
  // normalized in logical order before their flat bool-to-float cast.
  {
    int batch = 4, m = 64, k = 96, n = 64;
    int tm = 2, tk = 3, tn = 2;
    std::vector<float> a_values(static_cast<size_t>(batch) * m * k);
    std::vector<float> b_values(static_cast<size_t>(batch) * k * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }

    std::vector<uint8_t> lhs_base{1, 0, 0, 1, 1, 0};
    std::vector<uint8_t> rhs_base{1, 0, 1, 0, 1, 1};
    std::vector<uint8_t> out_base{1, 0, 1, 1};
    std::vector<uint8_t> lhs_one{1, 0, 1, 0, 1, 0};
    std::vector<uint8_t> rhs_one{1, 0, 0, 1, 1, 1};
    std::vector<uint8_t> out_one{1, 1, 0, 1};
    std::vector<uint8_t> lhs_v;
    std::vector<uint8_t> rhs_v;
    std::vector<float> out_v;
    for (int i = 0; i < batch; ++i) {
      lhs_v.insert(lhs_v.end(), lhs_one.begin(), lhs_one.end());
      rhs_v.insert(rhs_v.end(), rhs_one.begin(), rhs_one.end());
      out_v.insert(out_v.end(), out_one.begin(), out_one.end());
    }

    array a(a_values.begin(), Shape{batch, m, k}, float32);
    array b(b_values.begin(), Shape{batch, k, n}, float32);
    array lhs = broadcast_to(
        swapaxes(
            array(lhs_base.begin(), Shape{1, tk, tm}, bool_), -1, -2, stream),
        Shape{batch, tm, tk},
        stream);
    array rhs = broadcast_to(
        swapaxes(
            array(rhs_base.begin(), Shape{1, tn, tk}, bool_), -1, -2, stream),
        Shape{batch, tk, tn},
        stream);
    array out_mask = broadcast_to(
        swapaxes(
            array(out_base.begin(), Shape{1, tn, tm}, bool_), -1, -2, stream),
        Shape{batch, tm, tn},
        stream);
    array out = block_masked_mm(a, b, bs, out_mask, lhs, rhs, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> expected = block_masked_reference(
        a_values, b_values, lhs_v, rhs_v, &out_v, batch, m, k, n, bs);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // Float32 out mask only, unbatched: block factors 0.0, 0.5, 1.0, and
  // 2.0 exercise zeroing, attenuation, identity, and gain.
  {
    int m = 48, k = 64, n = 48;
    int tm = 2, tn = 2;
    std::vector<float> a_values(static_cast<size_t>(m) * k);
    std::vector<float> b_values(static_cast<size_t>(k) * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    std::vector<uint8_t> lhs_keep(static_cast<size_t>(tm) * 2, 1);
    std::vector<uint8_t> rhs_keep(static_cast<size_t>(2) * tn, 1);
    std::vector<float> out_mask_values{0.0f, 0.5f, 1.0f, 2.0f};
    array a(a_values.begin(), Shape{m, k}, float32);
    array b(b_values.begin(), Shape{k, n}, float32);
    array mask_out(out_mask_values.begin(), Shape{tm, tn}, float32);
    array out = block_masked_mm(
        a, b, bs, mask_out, std::nullopt, std::nullopt, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> expected = block_masked_reference(
        a_values,
        b_values,
        lhs_keep,
        rhs_keep,
        &out_mask_values,
        1,
        m,
        k,
        n,
        bs);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // All five inputs: batched operand masks plus a bool out mask.
  {
    int batch = 2, m = 48, k = 64, n = 48;
    int tm = 2, tk = 2, tn = 2;
    auto [lhs_v, rhs_v] = build_masks(batch, tm, tk, tn);
    std::vector<uint8_t> out_v(static_cast<size_t>(batch) * tm * tn);
    for (auto& value : out_v) {
      value = static_cast<uint8_t>(bit(gen));
    }
    std::vector<float> a_values(static_cast<size_t>(batch) * m * k);
    std::vector<float> b_values(static_cast<size_t>(batch) * k * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    array a(a_values.begin(), Shape{batch, m, k}, float32);
    array b(b_values.begin(), Shape{batch, k, n}, float32);
    array mask_out(out_v.begin(), Shape{batch, tm, tn}, bool_);
    array mask_lhs(lhs_v.begin(), Shape{batch, tm, tk}, bool_);
    array mask_rhs(rhs_v.begin(), Shape{batch, tk, tn}, bool_);
    array out =
        block_masked_mm(a, b, bs, mask_out, mask_lhs, mask_rhs, stream);
    REQUIRE(evaluation_error(out).empty());
    // Bool out mask: false zeroes the block, true multiplies by 1.0.
    std::vector<float> out_scale(out_v.begin(), out_v.end());
    std::vector<float> expected = block_masked_reference(
        a_values, b_values, lhs_v, rhs_v, &out_scale, batch, m, k, n, bs);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // Edge tiles: 33/65/17 leaves partial 32-wide blocks on every axis.
  {
    int m = 33, k = 65, n = 17;
    int tm = 2, tk = 3, tn = 1;
    auto [lhs_v, rhs_v] = build_masks(1, tm, tk, tn);
    std::vector<float> a_values(static_cast<size_t>(m) * k);
    std::vector<float> b_values(static_cast<size_t>(k) * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    array a(a_values.begin(), Shape{m, k}, float32);
    array b(b_values.begin(), Shape{k, n}, float32);
    array mask_lhs(lhs_v.begin(), Shape{tm, tk}, bool_);
    array mask_rhs(rhs_v.begin(), Shape{tk, tn}, bool_);
    array out =
        block_masked_mm(a, b, bs, std::nullopt, mask_lhs, mask_rhs, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> expected = block_masked_reference(
        a_values, b_values, lhs_v, rhs_v, nullptr, 1, m, k, n, bs);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // Named error: upstream restricts BlockMaskedMM to float32.
  {
    std::vector<double> wide(16, 0.5);
    array a(wide.begin(), Shape{4, 4}, float64);
    array b(wide.begin(), Shape{4, 4}, float64);
    std::string error;
    try {
      array out = block_masked_mm(
          a, b, bs, std::nullopt, std::nullopt, std::nullopt, stream);
      error = evaluation_error(out);
    } catch (const std::exception& caught) {
      error = caught.what();
    }
    bool f64_pinned = error.find("float64") != std::string::npos ||
        error.find("BlockMaskedMM") != std::string::npos;
    CHECK(f64_pinned);
  }
}

TEST_CASE("gather mm gathers operand matrices") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(23);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::uniform_int_distribution<uint32_t> index_a(0, 2);
  std::uniform_int_distribution<uint32_t> index_b(0, 3);

  // Batched both sides: lhs (2,3) and rhs (2,3) index a (3,M,K) and
  // b (4,K,N); the reference walks the flat index shape row-major.
  {
    int batch_a = 3, batch_b = 4, m = 6, k = 5, n = 7;
    std::vector<float> a_values(static_cast<size_t>(batch_a) * m * k);
    std::vector<float> b_values(static_cast<size_t>(batch_b) * k * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    std::vector<uint32_t> lhs_v(6);
    std::vector<uint32_t> rhs_v(6);
    for (auto& value : lhs_v) {
      value = index_a(gen);
    }
    for (auto& value : rhs_v) {
      value = index_b(gen);
    }
    array a(a_values.begin(), Shape{batch_a, m, k}, float32);
    array b(b_values.begin(), Shape{batch_b, k, n}, float32);
    array lhs(lhs_v.begin(), Shape{2, 3}, uint32);
    array rhs(rhs_v.begin(), Shape{2, 3}, uint32);
    array out = gather_mm(a, b, lhs, rhs, /*sorted_indices=*/false, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> a_view = a_values;
    std::vector<float> b_view = b_values;
    std::vector<float> expected;
    expected.reserve(6 * static_cast<size_t>(m) * n);
    // Expand the gathered stacks so host_matmul walks batch positions.
    std::vector<float> ga(6 * static_cast<size_t>(m) * k);
    std::vector<float> gb(6 * static_cast<size_t>(k) * n);
    for (size_t i = 0; i < 6; ++i) {
      for (int r = 0; r < m; ++r) {
        for (int inner = 0; inner < k; ++inner) {
          ga[i * m * k + static_cast<size_t>(r) * k + inner] =
              a_view[(static_cast<size_t>(lhs_v[i]) * m + r) * k + inner];
        }
      }
      for (int inner = 0; inner < k; ++inner) {
        for (int c = 0; c < n; ++c) {
          gb[i * k * n + static_cast<size_t>(inner) * n + c] =
              b_view[(static_cast<size_t>(rhs_v[i]) * k + inner) * n + c];
        }
      }
    }
    expected = host_matmul(ga, gb, 6, m, k, n);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // 2D left operand with omitted lhs indices: they default to zero, so
  // every output draws from the single a matrix while rhs gathers b.
  {
    int batch_b = 4, m = 6, k = 5, n = 7;
    std::vector<float> a_values(static_cast<size_t>(m) * k);
    std::vector<float> b_values(static_cast<size_t>(batch_b) * k * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    std::vector<uint32_t> rhs_v{1u, 3u, 0u};
    array a(a_values.begin(), Shape{m, k}, float32);
    array b(b_values.begin(), Shape{batch_b, k, n}, float32);
    array rhs(rhs_v.begin(), Shape{3}, uint32);
    // Default lhs indices compose arange(total, uint32) at the op layer.
    array out = gather_mm(a, b, std::nullopt, rhs, false, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> ga(3 * static_cast<size_t>(m) * k);
    std::vector<float> gb(3 * static_cast<size_t>(k) * n);
    for (size_t i = 0; i < 3; ++i) {
      std::copy(a_values.begin(), a_values.end(), ga.begin() + i * m * k);
      for (int inner = 0; inner < k; ++inner) {
        for (int c = 0; c < n; ++c) {
          gb[i * k * n + static_cast<size_t>(inner) * n + c] =
              b_values[(static_cast<size_t>(rhs_v[i]) * k + inner) * n + c];
        }
      }
    }
    std::vector<float> expected = host_matmul(ga, gb, 3, m, k, n);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // Edge tiles: 5/3/7 sits off every 16-wide matmul tile boundary.
  {
    int batch_a = 2, batch_b = 2, m = 5, k = 3, n = 7;
    std::vector<float> a_values(static_cast<size_t>(batch_a) * m * k);
    std::vector<float> b_values(static_cast<size_t>(batch_b) * k * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    std::vector<uint32_t> lhs_v{1u, 0u, 1u, 1u};
    std::vector<uint32_t> rhs_v{0u, 1u, 1u, 0u};
    array a(a_values.begin(), Shape{batch_a, m, k}, float32);
    array b(b_values.begin(), Shape{batch_b, k, n}, float32);
    array lhs(lhs_v.begin(), Shape{4}, uint32);
    array rhs(rhs_v.begin(), Shape{4}, uint32);
    array out = gather_mm(a, b, lhs, rhs, false, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> ga(4 * static_cast<size_t>(m) * k);
    std::vector<float> gb(4 * static_cast<size_t>(k) * n);
    for (size_t i = 0; i < 4; ++i) {
      for (int r = 0; r < m; ++r) {
        for (int inner = 0; inner < k; ++inner) {
          ga[i * m * k + static_cast<size_t>(r) * k + inner] =
              a_values[(static_cast<size_t>(lhs_v[i]) * m + r) * k + inner];
        }
      }
      for (int inner = 0; inner < k; ++inner) {
        for (int c = 0; c < n; ++c) {
          gb[i * k * n + static_cast<size_t>(inner) * n + c] =
              b_values[(static_cast<size_t>(rhs_v[i]) * k + inner) * n + c];
        }
      }
    }
    std::vector<float> expected = host_matmul(ga, gb, 4, m, k, n);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // A transposed right operand (a transpose view, not a dense stack)
  // takes the materialization path and still gathers correctly.
  {
    int batch_b = 3, m = 6, k = 5, n = 7;
    std::vector<float> a_values(static_cast<size_t>(m) * k);
    std::vector<float> w_values(static_cast<size_t>(batch_b) * n * k);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : w_values) {
      value = dist(gen);
    }
    array a(a_values.begin(), Shape{m, k}, float32);
    // w has shape (batch_b, n, k); its transpose is (batch_b, k, n).
    array w(w_values.begin(), Shape{batch_b, n, k}, float32);
    array b = transpose(w, {0, 2, 1}, stream);
    std::vector<uint32_t> rhs_v{2u, 0u};
    array rhs(rhs_v.begin(), Shape{2}, uint32);
    array out = gather_mm(a, b, std::nullopt, rhs, false, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> gb(2 * static_cast<size_t>(k) * n);
    for (size_t i = 0; i < 2; ++i) {
      // b[i] = transpose(w[rhs[i]]): element (inner, c) is
      // w[rhs[i]][c][inner].
      for (int inner = 0; inner < k; ++inner) {
        for (int c = 0; c < n; ++c) {
          gb[i * k * n + static_cast<size_t>(inner) * n + c] =
              w_values[(static_cast<size_t>(rhs_v[i]) * n + c) * k + inner];
        }
      }
    }
    std::vector<float> ga(2 * static_cast<size_t>(m) * k);
    for (size_t i = 0; i < 2; ++i) {
      std::copy(a_values.begin(), a_values.end(), ga.begin() + i * m * k);
    }
    std::vector<float> expected = host_matmul(ga, gb, 2, m, k, n);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // f16 activations against the same double reference over f16 inputs.
  if (float16_available()) {
    int batch_a = 2, batch_b = 2, m = 6, k = 5, n = 7;
    std::vector<float> a_values(static_cast<size_t>(batch_a) * m * k);
    std::vector<float> b_values(static_cast<size_t>(batch_b) * k * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    std::vector<float> a_f16 = round_trip(stream, a_values, float16);
    std::vector<float> b_f16 = round_trip(stream, b_values, float16);
    array a(a_f16.begin(), Shape{batch_a, m, k}, float16);
    array b(b_f16.begin(), Shape{batch_b, k, n}, float16);
    std::vector<uint32_t> lhs_v{0u, 1u};
    std::vector<uint32_t> rhs_v{1u, 0u};
    array lhs(lhs_v.begin(), Shape{2}, uint32);
    array rhs(rhs_v.begin(), Shape{2}, uint32);
    array out = gather_mm(a, b, lhs, rhs, false, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> ga(2 * static_cast<size_t>(m) * k);
    std::vector<float> gb(2 * static_cast<size_t>(k) * n);
    for (size_t i = 0; i < 2; ++i) {
      for (int r = 0; r < m; ++r) {
        for (int inner = 0; inner < k; ++inner) {
          ga[i * m * k + static_cast<size_t>(r) * k + inner] =
              a_f16[(static_cast<size_t>(lhs_v[i]) * m + r) * k + inner];
        }
      }
      for (int inner = 0; inner < k; ++inner) {
        for (int c = 0; c < n; ++c) {
          gb[i * k * n + static_cast<size_t>(inner) * n + c] =
              b_f16[(static_cast<size_t>(rhs_v[i]) * k + inner) * n + c];
        }
      }
    }
    std::vector<float> expected = host_matmul(ga, gb, 2, m, k, n);
    expect_close(readback_f32(stream, out), expected, 2e-2);
  }
}

TEST_CASE("segmented mm writes per-segment contractions") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(37);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  // Segments over K=33: full span, two sub-spans, an empty segment, and
  // an inverted segment; empties must write zeros.
  {
    int m = 6, k = 33, n = 5;
    std::vector<float> a_values(static_cast<size_t>(m) * k);
    std::vector<float> b_values(static_cast<size_t>(k) * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    std::vector<uint32_t> segments_v{
        0u, 33u, // full
        0u, 16u, // low sub-span
        16u, 33u, // high sub-span (off-tile boundary)
        7u, 7u, // empty
        20u, 5u, // inverted
    };
    array a(a_values.begin(), Shape{m, k}, float32);
    array b(b_values.begin(), Shape{k, n}, float32);
    array segments(segments_v.begin(), Shape{5, 2}, uint32);
    array out = segmented_mm(a, b, segments, stream);
    REQUIRE(evaluation_error(out).empty());
    REQUIRE_EQ(out.shape(), Shape{5, m, n});
    std::vector<float> expected(static_cast<size_t>(5) * m * n, 0.0f);
    for (int s = 0; s < 5; ++s) {
      uint32_t k_start = segments_v[2 * s];
      uint32_t k_end = segments_v[2 * s + 1];
      if (k_end <= k_start) {
        continue; // zeros stay
      }
      for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
          double acc = 0.0;
          for (uint32_t inner = k_start; inner < k_end; ++inner) {
            acc += host_at(a_values, static_cast<size_t>(r) * k + inner) *
                host_at(b_values, static_cast<size_t>(inner) * n + c);
          }
          expected[(static_cast<size_t>(s) * m + r) * n + c] =
              static_cast<float>(acc);
        }
      }
    }
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // Transposed operand views take the materialization path; segments
  // stay non-aligned to the 16-wide tile.
  {
    int m = 6, k = 33, n = 5;
    std::vector<float> a_values(static_cast<size_t>(k) * m);
    std::vector<float> b_values(static_cast<size_t>(n) * k);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    // at has shape (k, m); its transpose is (m, k). bt has shape (n, k);
    // its transpose is (k, n).
    array at(a_values.begin(), Shape{k, m}, float32);
    array bt(b_values.begin(), Shape{n, k}, float32);
    array a = transpose(at, {1, 0}, stream);
    array b = transpose(bt, {1, 0}, stream);
    std::vector<uint32_t> segments_v{
        0u, 1u,
        5u, 32u,
        31u, 33u,
    };
    array segments(segments_v.begin(), Shape{3, 2}, uint32);
    array out = segmented_mm(a, b, segments, stream);
    REQUIRE(evaluation_error(out).empty());
    REQUIRE_EQ(out.shape(), Shape{3, m, n});
    std::vector<float> expected(static_cast<size_t>(3) * m * n, 0.0f);
    for (int s = 0; s < 3; ++s) {
      uint32_t k_start = segments_v[2 * s];
      uint32_t k_end = segments_v[2 * s + 1];
      for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
          double acc = 0.0;
          for (uint32_t inner = k_start; inner < k_end; ++inner) {
            acc += host_at(a_values, static_cast<size_t>(inner) * m + r) *
                host_at(b_values, static_cast<size_t>(c) * k + inner);
          }
          expected[(static_cast<size_t>(s) * m + r) * n + c] =
              static_cast<float>(acc);
        }
      }
    }
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // f16 activations against the double reference.
  if (float16_available()) {
    int m = 4, k = 20, n = 3;
    std::vector<float> a_values(static_cast<size_t>(m) * k);
    std::vector<float> b_values(static_cast<size_t>(k) * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    std::vector<float> a_f16 = round_trip(stream, a_values, float16);
    std::vector<float> b_f16 = round_trip(stream, b_values, float16);
    array a(a_f16.begin(), Shape{m, k}, float16);
    array b(b_f16.begin(), Shape{k, n}, float16);
    std::vector<uint32_t> segments_v{0u, 20u, 3u, 17u, 17u, 20u};
    array segments(segments_v.begin(), Shape{3, 2}, uint32);
    array out = segmented_mm(a, b, segments, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> expected(static_cast<size_t>(3) * m * n, 0.0f);
    for (int s = 0; s < 3; ++s) {
      uint32_t k_start = segments_v[2 * s];
      uint32_t k_end = segments_v[2 * s + 1];
      for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
          double acc = 0.0;
          for (uint32_t inner = k_start; inner < k_end; ++inner) {
            acc += host_at(a_f16, static_cast<size_t>(r) * k + inner) *
                host_at(b_f16, static_cast<size_t>(inner) * n + c);
          }
          expected[(static_cast<size_t>(s) * m + r) * n + c] =
              static_cast<float>(acc);
        }
      }
    }
    expect_close(readback_f32(stream, out), expected, 2e-2);
  }

  // Named error: upstream rejects batched segmented_mm at the op layer.
  {
    std::vector<float> values(2 * 4 * 4, 1.0f);
    array a(values.begin(), Shape{2, 4, 4}, float32);
    array b(values.begin(), Shape{2, 4, 4}, float32);
    std::vector<uint32_t> segments_v{0u, 4u};
    array segments(segments_v.begin(), Shape{1, 2}, uint32);
    std::string error;
    try {
      array out = segmented_mm(a, b, segments, stream);
      error = evaluation_error(out);
    } catch (const std::exception& caught) {
      error = caught.what();
    }
    CHECK(error.find("segmented_mm") != std::string::npos);
  }
}

TEST_CASE("gather qmm gathers experts with scales and biases") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(41);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::uniform_int_distribution<uint32_t> index_x(0, 1);
  std::uniform_int_distribution<uint32_t> index_w(0, 2);

  auto run_case = [&](int experts,
                      int x_batch,
                      int m,
                      int k,
                      int n,
                      int group_size,
                      int bits,
                      bool with_lhs) {
    CAPTURE(experts);
    CAPTURE(m);
    CAPTURE(n);
    CAPTURE(k);
    CAPTURE(group_size);
    CAPTURE(bits);
    int groups = k / group_size;
    int pack = 32 / bits;
    int words_per_row = k / pack;
    std::vector<HostQuantizedWeights> host_w;
    // uint32, not float: these are raw packed code words. Held in a
    // float vector, every word >= 2^24 is silently rounded by the
    // element conversion (low eight bits destroyed, rounding carry into
    // bit 8), and the uint32 array constructor copies the rounded
    // values verbatim - the kernel then decodes wrong codes.
    std::vector<uint32_t> w_all;
    std::vector<float> scales_all;
    std::vector<float> biases_all;
    for (int e = 0; e < experts; ++e) {
      std::vector<float> matrix(static_cast<size_t>(n) * k);
      for (auto& value : matrix) {
        value = dist(gen);
      }
      HostQuantizedWeights host =
          host_affine_quantize(matrix, n, k, group_size, bits);
      host_w.push_back(host);
      w_all.insert(w_all.end(), host.words.begin(), host.words.end());
      scales_all.insert(scales_all.end(), host.scales.begin(),
                        host.scales.end());
      biases_all.insert(biases_all.end(), host.biases.begin(),
                        host.biases.end());
    }
    std::vector<std::vector<float>> x_batches;
    std::vector<float> x_all;
    for (int b = 0; b < x_batch; ++b) {
      std::vector<float> matrix(static_cast<size_t>(m) * k);
      for (auto& value : matrix) {
        value = dist(gen);
      }
      x_batches.push_back(matrix);
      x_all.insert(x_all.end(), matrix.begin(), matrix.end());
    }
    int positions = with_lhs ? 6 : x_batch;
    std::vector<uint32_t> lhs_v(positions);
    std::vector<uint32_t> rhs_v(positions);
    for (auto& value : lhs_v) {
      value = index_x(gen) % static_cast<uint32_t>(std::max(x_batch, 1));
    }
    for (auto& value : rhs_v) {
      value = index_w(gen) % static_cast<uint32_t>(experts);
    }
    array w_words(
        w_all.begin(),
        Shape{experts, n, words_per_row},
        uint32);
    array scales(
        scales_all.begin(), Shape{experts, n, groups}, float32);
    array biases(
        biases_all.begin(), Shape{experts, n, groups}, float32);
    array x(
        x_all.begin(),
        Shape{x_batch, m, k},
        float32);

    // gather_qmm returns a fresh array in both branches; hold it in an
    // optional so one declaration serves both.
    std::optional<array> out_holder;
    if (with_lhs) {
      array lhs(lhs_v.begin(), Shape{2, 3}, uint32);
      array rhs(rhs_v.begin(), Shape{2, 3}, uint32);
      out_holder = gather_qmm(
          x,
          w_words,
          scales,
          biases,
          lhs,
          rhs,
          /*transpose=*/true,
          group_size,
          bits,
          "affine",
          std::nullopt, /*global_scale*/
          /*sorted_indices=*/false,
          stream);
    } else {
      // Explicit zero lhs indices exercise the x-side broadcast without
      // the op layer's default path, which composes arange(uint32) --
      // a wave-1 Arange gap this backend names separately.
      std::vector<uint32_t> zeros(x_batch, 0u);
      array lhs0(zeros.begin(), Shape{x_batch}, uint32);
      array rhs(rhs_v.begin(), Shape{x_batch}, uint32);
      out_holder = gather_qmm(
          x,
          w_words,
          scales,
          biases,
          lhs0,
          rhs,
          true,
          group_size,
          bits,
          "affine",
          std::nullopt,
          false,
          stream);
    }
    array out = *out_holder;
    REQUIRE(evaluation_error(out).empty());
    REQUIRE_EQ(out.shape(-2), m);
    REQUIRE_EQ(out.shape(-1), n);
    // Reference: per position p, x[lhs[p]] contracted against the
    // affine dequant of the packed codes of expert rhs[p], in double
    // precision over the exact host words, scales, and biases the
    // device buffers carry.
    std::vector<float> expected(
        static_cast<size_t>(lhs_v.size()) * m * n, 0.0f);
    for (size_t p = 0; p < lhs_v.size(); ++p) {
      std::vector<float> piece = host_quantized_matmul(
          host_w[rhs_v[p]],
          // The non-lhs branch passes an all-zero index container to
          // the device (x-side broadcast), so the reference must
          // contract row 0 there too - not the drawn lhs_v[p].
          x_batches[with_lhs ? lhs_v[p] : 0u],
          m,
          n,
          k,
          group_size,
          bits);
      std::copy(
          piece.begin(),
          piece.end(),
          expected.begin() +
              static_cast<ptrdiff_t>(p) * static_cast<ptrdiff_t>(m) * n);
    }
    expect_close_tol(readback_f32(stream, out), expected, 1e-4, 1e-3);
  };

  run_case(3, 2, 5, 192, 37, 64, 4, true);
  run_case(3, 2, 5, 192, 37, 64, 4, false);
  run_case(2, 2, 1, 128, 16, 32, 8, true);
  run_case(3, 1, 7, 192, 37, 64, 8, true);

  // Named errors. Each shape is crafted to clear the op layer's shape
  // validation so the rejection provably comes from this backend: the
  // op layer's bit/group equality runs first, so the packed weight and
  // scale shapes must satisfy it even for the rejected parameters.
  {
    // mxfp4 mode: scales must be uint8 and biases absent at the op
    // layer; the primitive then rejects the mode itself.
    std::vector<float> x_values(2 * 64, 0.25f);
    array x(x_values.begin(), Shape{2, 64}, float32);
    std::vector<uint32_t> w_values(8 * 8, 0u);
    array w_words(w_values.begin(), Shape{8, 8}, uint32);
    std::vector<uint8_t> fp_scales(8 * 2, 127u);
    array scales8(fp_scales.begin(), Shape{8, 2}, uint8);
    std::vector<uint32_t> idx0{0u};
    array rhs(idx0.begin(), Shape{1}, uint32);
    array lhs0(idx0.begin(), Shape{1}, uint32);
    std::string mode_error = evaluation_error(gather_qmm(
        x,
        w_words,
        scales8,
        std::nullopt,
        lhs0,
        rhs,
        true,
        std::nullopt,
        std::nullopt,
        "mxfp4",
        std::nullopt,
        false,
        stream));
    // mxfp4 gather computes now: zero codes decode to 0.0 under any e8m0
    // scale, so the gathered product is all zeros.
    REQUIRE(mode_error.empty());
    array fp_out = gather_qmm(
        x, w_words, scales8, std::nullopt, lhs0, rhs, true, std::nullopt,
        std::nullopt, "mxfp4", std::nullopt, false, stream);
    std::vector<float> fp_values = readback_f32(stream, fp_out);
    REQUIRE_EQ(fp_values.size(), static_cast<size_t>(2 * 8));
    for (float v : fp_values) {
      CHECK(v == 0.0f);
    }

    // Non-transposed weights are a value path now: w reads as (k, n)
    // with k == 64, groups along n. Codes are zero and every
    // scale/bias is 0.5, so deq[k][n] == 0.5 and each output element
    // is 64 * (0.25 * 0.5) == 8.
    std::vector<uint32_t> w_nt(64 * 8, 0u);
    array w_nt_words(w_nt.begin(), Shape{64, 8}, uint32);
    std::vector<float> nt_params(64 * 2, 0.5f);
    array nt_scales(nt_params.begin(), Shape{64, 2}, float32);
    array nt_biases(nt_params.begin(), Shape{64, 2}, float32);
    array nt_out = gather_qmm(
        x,
        w_nt_words,
        nt_scales,
        nt_biases,
        lhs0,
        rhs,
        false,
        32,
        4,
        "affine",
        std::nullopt, false,
        stream);
    REQUIRE(evaluation_error(nt_out).empty());
    REQUIRE_EQ(nt_out.shape(), Shape{1, 2, 64});
    std::vector<float> nt_expected(2 * 64, 64.0f * 0.25f * 0.5f);
    expect_close_tol(readback_f32(stream, nt_out), nt_expected, 1e-5, 1e-4);

    // bits=2: packed width and scales must agree through the op layer
    // equality, so k = 8*32/2 = 128 with one scale per 128 columns.
    std::vector<float> x2_values(2 * 128, 0.25f);
    array x2(x2_values.begin(), Shape{2, 128}, float32);
    std::vector<float> one_param(8 * 1, 0.5f);
    array scales_one(one_param.begin(), Shape{8, 1}, float32);
    array biases_one(one_param.begin(), Shape{8, 1}, float32);
    std::string bits_error = evaluation_error(gather_qmm(
        x2,
        w_words,
        scales_one,
        biases_one,
        lhs0,
        rhs,
        true,
        128,
        2,
        "affine",
        std::nullopt, false,
        stream));
    CHECK(bits_error.find("GatherQMM bits") != std::string::npos);

    // group_size=128 is a value path now: k = 32*32/4 = 256 with two
    // scales per row. Codes are zero and every scale/bias is 0.5, so
    // each output element is 256 * (0.25 * 0.5) == 32.
    std::vector<uint32_t> w_wide(8 * 32, 0u);
    array w_wide_words(w_wide.begin(), Shape{8, 32}, uint32);
    std::vector<float> two_params(8 * 2, 0.5f);
    array scales_two(two_params.begin(), Shape{8, 2}, float32);
    array biases_two(two_params.begin(), Shape{8, 2}, float32);
    std::vector<float> x4_values(2 * 256, 0.25f);
    array x4(x4_values.begin(), Shape{2, 256}, float32);
    array wide_out = gather_qmm(
        x4,
        w_wide_words,
        scales_two,
        biases_two,
        lhs0,
        rhs,
        true,
        128,
        4,
        "affine",
        std::nullopt, false,
        stream);
    REQUIRE(evaluation_error(wide_out).empty());
    REQUIRE_EQ(wide_out.shape(), Shape{1, 2, 8});
    std::vector<float> wide_expected(16, 256.0f * 0.25f * 0.5f);
    expect_close_tol(readback_f32(stream, wide_out), wide_expected, 1e-5, 1e-4);
  }
}

TEST_CASE("gather qmm subgroup kernel matches scalar at decode shapes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(131);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::uniform_int_distribution<uint32_t> index_w(0, 2);

  // One case per (dtype, transpose, k edge): decode m == 1 with the
  // routed-expert gather; k 128 divides the 256-lane K split, k 96
  // exercises the tail guard, and the non-transposed layout runs the
  // other weight-routing branch.
  auto run_case_shape = [&](Dtype dtype, bool transpose, int k, int experts,
                            int index_count, int n) {
    const bool bf16 = dtype == bfloat16;
    const int m = 1;
    const int group_size = 64;
    const int bits = 4;
    const int groups = k / group_size;
    const int pack = 32 / bits;
    const int words_per_row = k / pack;
    std::vector<HostQuantizedWeights> host_w;
    std::vector<uint32_t> w_all;
    std::vector<float> scales_all;
    std::vector<float> biases_all;
    for (int e = 0; e < experts; ++e) {
      std::vector<float> matrix(static_cast<size_t>(n) * k);
      for (auto& value : matrix) {
        value = dist(gen);
      }
      HostQuantizedWeights host =
          host_affine_quantize(matrix, n, k, group_size, bits);
      host_w.push_back(host);
      w_all.insert(w_all.end(), host.words.begin(), host.words.end());
      scales_all.insert(
          scales_all.end(), host.scales.begin(), host.scales.end());
      biases_all.insert(
          biases_all.end(), host.biases.begin(), host.biases.end());
    }
    std::vector<std::vector<float>> x_batches;
    std::vector<float> x_all;
    for (int b = 0; b < index_count; ++b) {
      std::vector<float> matrix(static_cast<size_t>(m) * k);
      for (auto& value : matrix) {
        value = dist(gen);
      }
      x_batches.push_back(matrix);
      x_all.insert(x_all.end(), matrix.begin(), matrix.end());
    }
    std::vector<uint32_t> rhs_v(index_count);
    for (auto& value : rhs_v) {
      value = index_w(gen) % static_cast<uint32_t>(experts);
    }
    array w_words(w_all.begin(), Shape{experts, n, words_per_row}, uint32);
    array scales =
        astype(array(scales_all.begin(), Shape{experts, n, groups}, float32),
               dtype,
               stream);
    array biases =
        astype(array(biases_all.begin(), Shape{experts, n, groups}, float32),
               dtype,
               stream);
    array x = astype(
        array(x_all.begin(), Shape{index_count, m, k}, float32), dtype, stream);
    std::vector<uint32_t> lhs_v(index_count);
    for (auto& value : lhs_v) {
      value = index_w(gen) % static_cast<uint32_t>(index_count);
    }
    array lhs(lhs_v.begin(), Shape{index_count}, uint32);
    array rhs(rhs_v.begin(), Shape{index_count}, uint32);

    std::vector<float> expected(index_count * m * n, 0.0f);
    for (int b = 0; b < index_count; ++b) {
      std::vector<float> piece = host_quantized_matmul(
          host_w[rhs_v[b]], x_batches[lhs_v[b]], m, n, k, group_size, bits);
      std::copy(
          piece.begin(),
          piece.end(),
          expected.begin() + static_cast<ptrdiff_t>(b) * m * n);
    }

    auto run_once = [&](bool sub) {
      if (sub) {
        unsetenv("MLX_OMARCHY_GATHER_QMM_SUB");
      } else {
        setenv("MLX_OMARCHY_GATHER_QMM_SUB", "0", 1);
      }
      array out = gather_qmm(
          x,
          w_words,
          scales,
          biases,
        lhs,
          rhs,
          transpose,
          group_size,
          bits,
          "affine",
          std::nullopt,
          false,
          stream);
      REQUIRE(evaluation_error(out).empty());
      return readback_f32(stream, out);
    };

    std::vector<float> scalar_out = run_once(false);
    unsetenv("MLX_OMARCHY_GATHER_QMM_SUB");
    std::vector<float> sub_out = run_once(true);
    unsetenv("MLX_OMARCHY_GATHER_QMM_SUB");
    // Scale-aware numerics metric (docs/numerics-gate.md spirit): both
    // arms anchor to the fp64-accumulated host reference with rel-L2
    // over the whole output and max-abs error normalized by the
    // reference's max magnitude - elementwise relative error explodes
    // on bf16 outputs of k-large sums that pass near zero. The Sub arm
    // must be within 1.5x the scalar arm's rel-L2 (equal-or-better with
    // slack), and the scalar arm must clear its own absolute bound with
    // margin. Both errors print per case so one run names who fails.
    double ref_l2 = 0.0, ref_max = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
      ref_l2 += double(expected[i]) * expected[i];
      ref_max = std::max(ref_max, std::abs(double(expected[i])));
    }
    ref_l2 = std::sqrt(ref_l2);
    auto rel_l2 = [&](const std::vector<float>& v) {
      double acc = 0.0, mx = 0.0;
      for (size_t i = 0; i < v.size(); ++i) {
        double d = double(v[i]) - expected[i];
        acc += d * d;
        mx = std::max(mx, std::abs(d));
      }
      return std::pair<double, double>{std::sqrt(acc) / ref_l2, mx / ref_max};
    };
    auto [s_l2, s_max] = rel_l2(scalar_out);
    auto [b_l2, b_max] = rel_l2(sub_out);
    double scalar_bound = bf16 ? 0.05 : 0.01;
    std::cout << "[gather-qmm-sub] dtype=" << (bf16 ? "bf16" : "f16") << " transpose=" << transpose
              << " k=" << k << " experts=" << experts
              << " index_count=" << index_count << " n=" << n
              << " scalar: relL2=" << s_l2 << " maxabs/ref=" << s_max
              << " | sub: relL2=" << b_l2 << " maxabs/ref=" << b_max
              << " | count=" << expected.size() << std::endl;
    CHECK(s_l2 <= scalar_bound);
    CHECK(s_max <= scalar_bound * 2.0);
    CHECK(b_l2 <= 1.5 * s_l2 + 1e-9);
    CHECK(b_max <= 1.5 * s_max + 1e-9);
    // Direct sub-versus-scalar check (PR review): the 1.5x gate above sits on
    // the output-rounding floor, so it cannot see a small extra error. At
    // least 99 % of the elements must agree within one ulp of the output
    // dtype and no element may differ by more than 2e-3 of the reference
    // maximum (near-zero outputs after cancellation can move more than one
    // ulp, so it is not an every-element ulp rule).
    {
      const int mantissa = bf16 ? 7 : 10;
      size_t within = 0;
      double worst = 0.0;
      for (size_t i = 0; i < sub_out.size(); ++i) {
        const double a = sub_out[i];
        const double b = scalar_out[i];
        const double diff = std::abs(a - b);
        worst = std::max(worst, diff);
        int exponent = 0;
        std::frexp(std::max(std::abs(a), std::abs(b)), &exponent);
        const double ulp = std::ldexp(1.0, exponent - 1 - mantissa);
        within += diff <= ulp ? 1 : 0;
      }
      const double fraction = double(within) / double(sub_out.size());
      std::cout << "[gather-qmm-sub] sub vs scalar: within 1 ulp " << fraction
                << " max abs diff / ref max " << worst / ref_max << std::endl;
      CHECK(fraction >= 0.99);
      CHECK(worst <= 2e-3 * ref_max);
    }
    // z-chunk boundary: with index_count x n > 65535, outputs at flat
    // indices >= 65535 come from the z>=1 workgroup chunks - they must
    // track the reference, not zeros/garbage.
    if (index_count * n > 65535) {
      size_t boundary = 65535;
      double ref_b = 0.0, acc = 0.0;
      for (size_t i = boundary; i < sub_out.size(); ++i) {
        double d = double(sub_out[i]) - expected[i];
        acc += d * d;
        ref_b += double(expected[i]) * expected[i];
      }
      double tail_l2 = std::sqrt(acc) / std::sqrt(ref_b);
      std::cout << "[gather-qmm-sub] z-chunk tail relL2=" << tail_l2
                << std::endl;
      CHECK(tail_l2 <= 1.5 * s_l2 + 1e-9);
    }
  };

  // Only bf16 and f16 transposed configs: they cover the Sub selector's
  // proven class (4-bit, group-64, transposed, m==1). f32 configs are NOT
  // covered by this test (the f32 non-transposed authoring used an
  // invalid weight layout [E, N, K/pack] and the f32 transposed m==1
  // paths are a separate investigation - see the lane receipt).
  // fp16 joined the Sub selector's class with MatmulGap H45: the same
  // shapes, the same gate (sub within 1.5x of the scalar arm against the
  // fp64 host reference).
  for (Dtype dtype : {bfloat16, float16}) {
    run_case_shape(dtype, true, 128, 3, 2, 64);
    run_case_shape(dtype, true, 192, 3, 2, 64);   // k below one 256-lane stride: tail guard
    // GLM-4.5-Air routed shapes: B=1 gate/up gather (index_count 8 x
    // n 1408 = 11,264 workgroups) and a count above the 65535 per-dimension
    // workgroup limit (48 x 1408 = 67,584 -> z-chunked dispatch).
    run_case_shape(dtype, true, 128, 3, 8, 1408);
    run_case_shape(dtype, true, 2048, 3, 8, 1408);
    run_case_shape(dtype, true, 128, 48, 48, 1408);
  }
}

TEST_CASE("gather qqmm dequants with scales only") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(53);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  auto run_case = [&](int m, int k, int n, int group_size, int bits) {
    CAPTURE(m);
    CAPTURE(n);
    CAPTURE(k);
    int experts = 3;
    int x_batch = 2;
    int groups = k / group_size;
    int pack = 32 / bits;
    int words_per_row = k / pack;
    std::vector<HostQuantizedWeights> host_w;
    // uint32, not float - same packed-words hazard as gather_qmm above.
    std::vector<uint32_t> w_all;
    std::vector<float> scales_all;
    for (int e = 0; e < experts; ++e) {
      std::vector<float> matrix(static_cast<size_t>(n) * k);
      for (auto& value : matrix) {
        value = dist(gen);
      }
      HostQuantizedWeights host =
          host_affine_quantize(matrix, n, k, group_size, bits);
      host_w.push_back(host);
      w_all.insert(w_all.end(), host.words.begin(), host.words.end());
      scales_all.insert(
          scales_all.end(), host.scales.begin(), host.scales.end());
    }
    std::vector<std::vector<float>> x_batches;
    std::vector<float> x_all;
    for (int b = 0; b < x_batch; ++b) {
      std::vector<float> matrix(static_cast<size_t>(m) * k);
      for (auto& value : matrix) {
        value = dist(gen);
      }
      x_batches.push_back(matrix);
      x_all.insert(x_all.end(), matrix.begin(), matrix.end());
    }
    std::vector<uint32_t> lhs_v{0u, 1u, 1u, 0u};
    std::vector<uint32_t> rhs_v{2u, 0u, 1u, 2u};
    array w_words(
        w_all.begin(), Shape{experts, n, words_per_row}, uint32);
    array scales(scales_all.begin(), Shape{experts, n, groups}, float32);
    array x(x_all.begin(), Shape{x_batch, m, k}, float32);
    array lhs(lhs_v.begin(), Shape{2, 2}, uint32);
    array rhs(rhs_v.begin(), Shape{2, 2}, uint32);
    array out = gather_qqmm(
        x,
        w_words,
        scales,
        lhs,
        rhs,
        group_size,
        bits,
        "affine",
        std::nullopt,
        std::nullopt,
        false,
        stream);
    REQUIRE(evaluation_error(out).empty());
    REQUIRE_EQ(out.shape(-2), m);
    REQUIRE_EQ(out.shape(-1), n);
    // Reference: same contraction as GatherQMM but the GatherQQMM
    // affine contract drops the bias, so dequant is q * scale only.
    std::vector<float> expected(4 * static_cast<size_t>(m) * n, 0.0f);
    for (size_t p = 0; p < lhs_v.size(); ++p) {
      std::vector<float> piece = host_scale_only_quantized_matmul(
          host_w[rhs_v[p]],
          x_batches[lhs_v[p]],
          m,
          n,
          k,
          group_size,
          bits);
      std::copy(
          piece.begin(),
          piece.end(),
          expected.begin() +
              static_cast<ptrdiff_t>(p) * static_cast<ptrdiff_t>(m) * n);
    }
    expect_close_tol(readback_f32(stream, out), expected, 1e-4, 1e-3);
  };

  run_case(5, 192, 37, 64, 4);
  run_case(1, 128, 16, 32, 8);

  // Named errors: a non-affine mode outside its group-size family and
  // affine float weights keep named tags; mxfp4 at its own group size
  // computes (covered by the fp qqmm test case below).
  {
    int k = 64, n = 8;
    std::vector<float> matrix(static_cast<size_t>(n) * k, 0.5f);
    auto host = host_affine_quantize(matrix, n, k, 64, 4);
    array w_words(host.words.begin(), Shape{n, k / 8}, uint32);
    array w_float(matrix.begin(), Shape{n, k}, float32);
    array scales(host.scales.begin(), Shape{n, 1}, float32);
    std::vector<float> x_values(static_cast<size_t>(2) * k, 0.25f);
    array x(x_values.begin(), Shape{2, k}, float32);
    std::vector<uint32_t> idx_v{0u};
    array idx(idx_v.begin(), Shape{1}, uint32);

    std::string mode_error = evaluation_error(gather_qqmm(
        x, w_words, scales, idx, idx, 64, 4, "mxfp4", std::nullopt,
        std::nullopt, false, stream));
    CHECK(mode_error.find("GatherQQMM group size") != std::string::npos);

    std::string weight_error = evaluation_error(gather_qqmm(
        x, w_float, scales, idx, idx, 64, 4, "affine", std::nullopt,
        std::nullopt, false, stream));
    CHECK(weight_error.find("GatherQQMM") != std::string::npos);
  }
}

TEST_CASE("qq matmul matches scale-only quantized and float paths") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(67);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  // Quantized weights: scale-only affine dequant, no activation
  // quantization on this backend.
  {
    int m = 7, k = 128, n = 37;
    for (auto [group_size, bits] : std::vector<std::pair<int, int>>{
             {64, 4}, {32, 8}}) {
      int groups = k / group_size;
      int pack = 32 / bits;
      int words_per_row = k / pack;
      std::vector<float> matrix(static_cast<size_t>(n) * k);
      for (auto& value : matrix) {
        value = dist(gen);
      }
      HostQuantizedWeights host =
          host_affine_quantize(matrix, n, k, group_size, bits);
      std::vector<float> x_values(static_cast<size_t>(m) * k);
      for (auto& value : x_values) {
        value = dist(gen);
      }
      array w_words(
          host.words.begin(), Shape{n, words_per_row}, uint32);
      array scales(host.scales.begin(), Shape{n, groups}, float32);
      array x(x_values.begin(), Shape{m, k}, float32);
      array out = qqmm(x, w_words, scales, group_size, bits, "affine",
                       std::nullopt, std::nullopt, stream);
      REQUIRE(evaluation_error(out).empty());
      REQUIRE_EQ(out.shape(), Shape{m, n});
      std::vector<float> expected = host_scale_only_quantized_matmul(
          host, x_values, m, n, k, group_size, bits);
      expect_close(readback_f32(stream, out), expected, 1e-3);
    }
  }

  // Float weights: the un-quantized product x @ w.T against the host.
  {
    int m = 7, k = 20, n = 37;
    std::vector<float> matrix(static_cast<size_t>(n) * k);
    std::vector<float> x_values(static_cast<size_t>(m) * k);
    for (auto& value : matrix) {
      value = dist(gen);
    }
    for (auto& value : x_values) {
      value = dist(gen);
    }
    array w(matrix.begin(), Shape{n, k}, float32);
    array x(x_values.begin(), Shape{m, k}, float32);
    // Float-weight qqmm needs x @ w.T; the backend names the rejection
    // (no transposed view inside an eval). Pin it.
    array out = qqmm(x, w, std::nullopt, 64, 4, "affine", std::nullopt,
                     std::nullopt, stream);
    std::string weight_error = evaluation_error(out);
    CHECK(weight_error.find("QQMatmul weight dtype") != std::string::npos);

    (void)m;
    (void)k;
    (void)n;
  }

  // Named error: a non-affine mode outside its group-size family keeps
  // a named tag; mxfp4 at its own group size computes (covered by the
  // fp qqmm test case below).
  {
    std::vector<float> matrix(4 * 32, 0.5f);
    array w(matrix.begin(), Shape{4, 32}, float32);
    std::vector<float> x_values(2 * 32, 0.25f);
    array x(x_values.begin(), Shape{2, 32}, float32);
    std::string mode_error = evaluation_error(
        qqmm(x, w, std::nullopt, 64, 4, "mxfp4", std::nullopt, std::nullopt,
             stream));
    CHECK(mode_error.find("QQMatmul group size") != std::string::npos);
  }
}

// ---------------------------------------------------------------------------
// fp qqmm: host bit-reference for mxfp4 / mxfp8 / nvfp4.
//
// The conversions transcribe quantize.comp / dequant.comp (which mirror
// the pinned Metal fp4.h / fp8.h bit for bit): e4m3 scale encode is
// round-to-nearest-even with saturation at 448, round-up e8m0 keeps a
// group maximum representable, fp4 e2m1 uses the threshold table, and
// the nvfp4 global scale rides in as the 2688 / global_scale encode
// factor on both the scale byte and the element domain. The expected
// products fold dequant(quant(x)) against dequant(quant(w)) in double
// precision - exactly the upstream python contract
// (test_quantized.py::test_qqmm / test_gather_qqmm): a wrong scale, a
// wrong code, or raw-float activations against dequantized weights all
// miss by O(1).
namespace {

uint32_t host_bits_of(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float host_float_of(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float host_half_bits_to_float(uint32_t h) {
  uint32_t sign = (h & 0x8000u) << 16u;
  uint32_t exponent = (h >> 10u) & 0x1Fu;
  uint32_t mantissa = h & 0x3FFu;
  float value;
  if (exponent == 0u) {
    value = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 31u) {
    value = mantissa == 0u
        ? std::numeric_limits<float>::infinity()
        : std::numeric_limits<float>::quiet_NaN();
  } else {
    value = std::ldexp(
        static_cast<float>(mantissa | 0x400u),
        static_cast<int>(exponent) - 25);
  }
  return sign != 0u ? -value : value;
}

uint8_t host_fp8_encode(float value) {
  const uint32_t fp8_max = 543u << 21u;
  const uint32_t denorm_mask = 141u << 23u;
  uint32_t f_bits = host_bits_of(value);
  uint32_t sign = f_bits & 0x80000000u;
  f_bits ^= sign;

  uint32_t f_bits_low = host_bits_of(
      host_float_of(f_bits) + host_float_of(denorm_mask));
  uint32_t result_low = (f_bits_low - denorm_mask) & 0xFFu;

  uint32_t mant_odd = (f_bits >> 20u) & 1u;
  uint32_t f_bits_high = f_bits + (((7u - 127u) << 23u) + 0x7FFFFu);
  f_bits_high += mant_odd;
  uint32_t result_high = (f_bits_high >> 20u) & 0xFFu;

  uint32_t result = (f_bits < (121u << 23u)) ? result_low : result_high;
  result = (f_bits >= fp8_max) ? 0x7Eu : result;
  return static_cast<uint8_t>(result | (sign >> 24u));
}

float host_fp8_decode(uint32_t bits) {
  float magnitude = host_half_bits_to_float((bits & 127u) << 7u) * 256.0f;
  return (bits & 128u) != 0u ? -magnitude : magnitude;
}

uint8_t host_e8m0_encode(float x) {
  if (std::isnan(x) || std::isinf(x)) {
    return 0xFF;
  }
  if (x <= 0.0f) {
    return 0x00;
  }
  int n = static_cast<int>(std::round(std::log2(x)));
  n = std::max(-127, std::min(127, n));
  uint8_t bits = static_cast<uint8_t>(n + 127);
  float decoded = host_float_of(static_cast<uint32_t>(
      bits == 0 ? 0x400000u : static_cast<uint32_t>(bits) << 23u));
  if (bits < 0xFE && decoded < x) {
    bits += 1;
  }
  return bits;
}

float host_e8m0_decode(uint8_t bits) {
  return host_float_of(bits == 0
          ? 0x400000u
          : static_cast<uint32_t>(bits) << 23u);
}

uint8_t host_fp4_encode(float x) {
  if (std::isnan(x)) {
    return 0x7;
  }
  uint8_t sign = std::signbit(x) ? 0x8 : 0x0;
  float a = std::abs(x);
  uint8_t m;
  if (a > 5.0f) {
    m = 0x7;
  } else if (a >= 3.5f) {
    m = 0x6;
  } else if (a > 2.5f) {
    m = 0x5;
  } else if (a >= 1.75f) {
    m = 0x4;
  } else if (a > 1.25f) {
    m = 0x3;
  } else if (a >= 0.75f) {
    m = 0x2;
  } else if (a > 0.25f) {
    m = 0x1;
  } else {
    m = 0x0;
  }
  return m | sign;
}

float host_fp4_decode(uint8_t code) {
  static const float lut[8] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  float magnitude = lut[code & 7u];
  return (code & 8u) != 0u ? -magnitude : magnitude;
}

// dequant(quant(x)) across both kernels: one scale byte per group
// (fp8 e4m3 at group 16 carrying the 2688 / global_scale encode
// factor, round-up e8m0 at group 32), elements e2m1 / e4m3.
std::vector<float> host_fp_fake_quantize(
    const std::vector<float>& x,
    int group_size,
    int bits,
    const float* global_scale) {
  float maxval = bits == 8 ? 448.0f : 6.0f;
  float global_enc =
      global_scale != nullptr ? 2688.0f / *global_scale : 1.0f;
  float global_dec =
      global_scale != nullptr ? *global_scale / 2688.0f : 1.0f;
  std::vector<float> out(x.size(), 0.0f);
  size_t groups = x.size() / static_cast<size_t>(group_size);
  for (size_t g = 0; g < groups; ++g) {
    size_t base = g * static_cast<size_t>(group_size);
    float amax = 0.0f;
    for (int lane = 0; lane < group_size; ++lane) {
      amax = std::max(amax, std::abs(x[base + lane]));
    }
    float scale_dec = amax / maxval;
    uint8_t q_scale;
    float decoded;
    if (group_size == 16) {
      scale_dec *= global_enc;
      q_scale = host_fp8_encode(scale_dec);
      decoded = host_fp8_decode(q_scale);
    } else {
      q_scale = host_e8m0_encode(scale_dec);
      decoded = host_e8m0_decode(q_scale);
    }
    float inv = (decoded == 0.0f) ? 0.0f : global_enc / decoded;
    for (int lane = 0; lane < group_size; ++lane) {
      float value = x[base + lane] * inv;
      float code = bits == 8
          ? host_fp8_decode(host_fp8_encode(value))
          : host_fp4_decode(host_fp4_encode(value));
      out[base + lane] = decoded * global_dec * code;
    }
  }
  return out;
}

// Packed codes plus scale bytes exactly as the quantize kernels store
// them: fp4 pairs two elements per byte (low element low nibble), fp8
// one byte per element, bytes LSB-first inside uint32 words.
struct HostFpWeights {
  std::vector<uint32_t> words;
  std::vector<uint8_t> scales;
};

HostFpWeights host_fp_quantize_packed(
    const std::vector<float>& w,
    int rows,
    int cols,
    int group_size,
    int bits,
    const float* global_scale) {
  float maxval = bits == 8 ? 448.0f : 6.0f;
  float global_enc =
      global_scale != nullptr ? 2688.0f / *global_scale : 1.0f;
  int groups_per_row = cols / group_size;
  HostFpWeights result;
  result.words.assign(static_cast<size_t>(rows) * (cols * bits / 32), 0u);
  result.scales.assign(static_cast<size_t>(rows) * groups_per_row, 0u);
  for (int row = 0; row < rows; ++row) {
    for (int g = 0; g < groups_per_row; ++g) {
      size_t base = static_cast<size_t>(row) * cols +
          static_cast<size_t>(g) * group_size;
      float amax = 0.0f;
      for (int lane = 0; lane < group_size; ++lane) {
        amax = std::max(amax, std::abs(w[base + lane]));
      }
      float scale_dec = amax / maxval;
      uint8_t q_scale;
      if (group_size == 16) {
        scale_dec *= global_enc;
        q_scale = host_fp8_encode(scale_dec);
      } else {
        q_scale = host_e8m0_encode(scale_dec);
      }
      result.scales[static_cast<size_t>(row) * groups_per_row + g] =
          q_scale;
      float decoded = group_size == 16
          ? host_fp8_decode(q_scale)
          : host_e8m0_decode(q_scale);
      float inv = (decoded == 0.0f) ? 0.0f : global_enc / decoded;
      for (int lane = 0; lane < group_size; ++lane) {
        float value = w[base + lane] * inv;
        uint8_t code = bits == 8 ? host_fp8_encode(value)
                                 : host_fp4_encode(value);
        size_t flat = base + lane;
        if (bits == 8) {
          result.words[flat / 4u] |=
              static_cast<uint32_t>(code) << ((flat % 4u) * 8u);
        } else {
          result.words[flat / 8u] |=
              static_cast<uint32_t>(code) << ((flat % 8u) * 4u);
        }
      }
    }
  }
  return result;
}

std::vector<float> host_fp_dequantize_packed(
    const HostFpWeights& packed,
    int rows,
    int cols,
    int group_size,
    int bits,
    const float* global_scale) {
  float global_dec =
      global_scale != nullptr ? *global_scale / 2688.0f : 1.0f;
  int groups_per_row = cols / group_size;
  std::vector<float> out(static_cast<size_t>(rows) * cols, 0.0f);
  for (int row = 0; row < rows; ++row) {
    for (int g = 0; g < groups_per_row; ++g) {
      uint8_t q_scale =
          packed.scales[static_cast<size_t>(row) * groups_per_row + g];
      float scale = group_size == 16
          ? host_fp8_decode(q_scale)
          : host_e8m0_decode(q_scale);
      scale *= global_dec;
      size_t base = static_cast<size_t>(row) * cols +
          static_cast<size_t>(g) * group_size;
      for (int lane = 0; lane < group_size; ++lane) {
        size_t flat = base + lane;
        uint8_t code;
        if (bits == 8) {
          code = (packed.words[flat / 4u] >> ((flat % 4u) * 8u)) & 0xFFu;
        } else {
          code = (packed.words[flat / 8u] >> ((flat % 8u) * 4u)) & 0xFu;
        }
        float value = bits == 8 ? host_fp8_decode(code)
                                : host_fp4_decode(code);
        out[flat] = scale * value;
      }
    }
  }
  return out;
}

std::vector<float> host_fp_matmul(
    const std::vector<float>& x_hat,
    const std::vector<float>& w_hat,
    int m,
    int n,
    int k) {
  std::vector<float> out(static_cast<size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double acc = 0.0;
      for (int i = 0; i < k; ++i) {
        acc += static_cast<double>(x_hat[static_cast<size_t>(row) * k + i]) *
            static_cast<double>(w_hat[static_cast<size_t>(col) * k + i]);
      }
      out[static_cast<size_t>(row) * n + col] = static_cast<float>(acc);
    }
  }
  return out;
}

} // namespace

TEST_CASE("qqmm fp modes fake-quantize the activation") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(97);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  // Exact quant/dequant reference values through the product: an
  // identity weight exposes each fake-quantized activation element as
  // one output entry, threshold-boundary inputs included.
  {
    int k = 32, m = 1;
    std::vector<float> x(k, 0.0f);
    x[0] = 0.25f;
    x[1] = 0.26f;
    x[2] = 0.74f;
    x[3] = 0.76f;
    x[4] = 1.75f;
    x[5] = 2.5f;
    x[6] = 5.0f;
    x[7] = 6.0f;
    x[8] = -0.25f;
    x[9] = -0.5f;
    x[10] = -1.5f;
    x[11] = -6.0f;
    for (int i = 12; i < k; ++i) {
      x[i] = std::ldexp(1.0f, -3 + (i % 5)) * ((i % 2) == 0 ? 1.0f : -1.0f);
    }
    std::vector<float> w(static_cast<size_t>(k) * k, 0.0f);
    for (int i = 0; i < k; ++i) {
      w[static_cast<size_t>(i) * k + i] = 1.0f;
    }
    auto x_hat = host_fp_fake_quantize(x, 32, 4, nullptr);
    auto packed = host_fp_quantize_packed(w, k, k, 32, 4, nullptr);
    auto w_hat = host_fp_dequantize_packed(packed, k, k, 32, 4, nullptr);
    auto expected = host_fp_matmul(x_hat, w_hat, m, k, k);

    array w_words(packed.words.begin(), Shape{k, k * 4 / 32}, uint32);
    array w_scales(packed.scales.begin(), Shape{k, k / 32}, uint8);
    array x_arr(x.begin(), Shape{m, k}, float32);
    array out = qqmm(
        x_arr,
        w_words,
        w_scales,
        32,
        4,
        "mxfp4",
        std::nullopt,
        std::nullopt,
        stream);
    REQUIRE(evaluation_error(out).empty());
    REQUIRE_EQ(out.shape(), Shape{m, k});
    expect_close_tol(readback_f32(stream, out), expected, 1e-5, 1e-4);
  }

  // Un-gathered random products, quantized weights, all three modes;
  // nvfp4 binds both global scales and requires the HGS correction.
  {
    int m = 2, n = 5;
    for (auto [mode, group_size, bits, use_gs] :
        std::vector<std::tuple<const char*, int, int, bool>>{
            {"mxfp8", 32, 8, false},
            {"mxfp4", 32, 4, false},
            {"nvfp4", 16, 4, true}}) {
      CAPTURE(mode);
      int k = group_size * 4;
      std::vector<float> x_values(static_cast<size_t>(m) * k);
      for (auto& value : x_values) {
        value = dist(gen);
      }
      std::vector<float> w_values(static_cast<size_t>(n) * k);
      for (auto& value : w_values) {
        value = dist(gen);
      }
      float global_scale_x = 0.0f;
      float global_scale_w = 0.0f;
      for (float value : x_values) {
        global_scale_x = std::max(global_scale_x, std::abs(value));
      }
      for (float value : w_values) {
        global_scale_w = std::max(global_scale_w, std::abs(value));
      }
      const float* gs_x = use_gs ? &global_scale_x : nullptr;
      const float* gs_w = use_gs ? &global_scale_w : nullptr;

      auto x_hat = host_fp_fake_quantize(x_values, group_size, bits, gs_x);
      auto packed =
          host_fp_quantize_packed(w_values, n, k, group_size, bits, gs_w);
      auto w_hat =
          host_fp_dequantize_packed(packed, n, k, group_size, bits, gs_w);
      auto expected = host_fp_matmul(x_hat, w_hat, m, n, k);

      array x_arr(x_values.begin(), Shape{m, k}, float32);
      array w_words(packed.words.begin(), Shape{n, k * bits / 32}, uint32);
      array w_scales(
          packed.scales.begin(), Shape{n, k / group_size}, uint8);
      array gs_x_arr(global_scale_x);
      array gs_w_arr(global_scale_w);
      array out = qqmm(
          x_arr,
          w_words,
          w_scales,
          group_size,
          bits,
          mode,
          use_gs ? std::optional<array>(gs_x_arr) : std::nullopt,
          use_gs ? std::optional<array>(gs_w_arr) : std::nullopt,
          stream);
      REQUIRE(evaluation_error(out).empty());
      REQUIRE_EQ(out.shape(), Shape{m, n});
      expect_close_tol(readback_f32(stream, out), expected, 1e-4, 1e-3);
    }
  }

  // Gathered fp qqmm across batch slices with expert weights; the
  // gathered product must use the same fake-quantized activation. The
  // nvfp4 w global scale is one float32 scalar for the whole tensor
  // (the pinned upstream python contract), pinned here to 1.0f so the
  // packed scale bytes carry the 2688 encode factor and the HGS output
  // correction divides it back out against the reference, which
  // applies the same factor per group.
  {
    int experts = 2, x_batch = 2, m = 3, n = 5;
    for (auto [mode, group_size, bits, use_gs] :
        std::vector<std::tuple<const char*, int, int, bool>>{
            {"mxfp8", 32, 8, false},
            {"nvfp4", 16, 4, true}}) {
      CAPTURE(mode);
      int k = group_size * 4;
      float global_scale_w = 1.0f;
      const float* gs_w = use_gs ? &global_scale_w : nullptr;

      std::vector<std::vector<float>> x_batches;
      std::vector<float> x_all;
      for (int b = 0; b < x_batch; ++b) {
        std::vector<float> matrix(static_cast<size_t>(m) * k);
        for (auto& value : matrix) {
          value = dist(gen);
        }
        x_batches.push_back(matrix);
        x_all.insert(x_all.end(), matrix.begin(), matrix.end());
      }
      float global_scale_x = 0.0f;
      for (float value : x_all) {
        global_scale_x = std::max(global_scale_x, std::abs(value));
      }
      const float* gs_x = use_gs ? &global_scale_x : nullptr;

      std::vector<HostFpWeights> packed_experts;
      std::vector<uint32_t> w_all;
      std::vector<uint8_t> scales_all;
      for (int e = 0; e < experts; ++e) {
        std::vector<float> matrix(static_cast<size_t>(n) * k);
        for (auto& value : matrix) {
          value = dist(gen);
        }
        auto packed = host_fp_quantize_packed(matrix, n, k, group_size, bits, gs_w);
        packed_experts.push_back(packed);
        w_all.insert(w_all.end(), packed.words.begin(), packed.words.end());
        scales_all.insert(
            scales_all.end(), packed.scales.begin(), packed.scales.end());
      }

      std::vector<uint32_t> lhs_v{0u, 1u, 0u};
      std::vector<uint32_t> rhs_v{1u, 0u, 1u};
      array x_arr(x_all.begin(), Shape{x_batch, m, k}, float32);
      array w_words(
          w_all.begin(), Shape{experts, n, k * bits / 32}, uint32);
      array w_scales(
          scales_all.begin(), Shape{experts, n, k / group_size}, uint8);
      array lhs(lhs_v.begin(), Shape{3}, uint32);
      array rhs(rhs_v.begin(), Shape{3}, uint32);
      array gs_x_arr(global_scale_x);
      array gs_w_arr(global_scale_w);
      array out = gather_qqmm(
          x_arr,
          w_words,
          w_scales,
          lhs,
          rhs,
          group_size,
          bits,
          mode,
          use_gs ? std::optional<array>(gs_x_arr) : std::nullopt,
          use_gs ? std::optional<array>(gs_w_arr) : std::nullopt,
          false,
          stream);
      REQUIRE(evaluation_error(out).empty());
      REQUIRE_EQ(out.shape(), Shape{3, m, n});

      std::vector<float> expected(3 * static_cast<size_t>(m) * n, 0.0f);
      for (size_t p = 0; p < lhs_v.size(); ++p) {
        auto xb = host_fp_fake_quantize(
            x_batches[lhs_v[p]], group_size, bits, gs_x);
        auto wh = host_fp_dequantize_packed(
            packed_experts[rhs_v[p]], n, k, group_size, bits, gs_w);
        auto piece = host_fp_matmul(xb, wh, m, n, k);
        std::copy(
            piece.begin(),
            piece.end(),
            expected.begin() +
                static_cast<ptrdiff_t>(p) * static_cast<ptrdiff_t>(m) * n);
      }
      expect_close_tol(readback_f32(stream, out), expected, 1e-4, 1e-3);
    }
  }

  // Float weights: the backend packs w with the same kernels it backs
  // mx.quantize with, so the product still contracts quantized-x codes
  // against quantized-w codes.
  {
    int m = 2, k = 32, n = 4;
    std::vector<float> x_values(static_cast<size_t>(m) * k);
    for (auto& value : x_values) {
      value = dist(gen);
    }
    std::vector<float> w_values(static_cast<size_t>(n) * k);
    for (auto& value : w_values) {
      value = dist(gen);
    }
    auto x_hat = host_fp_fake_quantize(x_values, 32, 8, nullptr);
    auto packed = host_fp_quantize_packed(w_values, n, k, 32, 8, nullptr);
    auto w_hat = host_fp_dequantize_packed(packed, n, k, 32, 8, nullptr);
    auto expected = host_fp_matmul(x_hat, w_hat, m, n, k);

    array x_arr(x_values.begin(), Shape{m, k}, float32);
    array w_arr(w_values.begin(), Shape{n, k}, float32);
    array out = qqmm(
        x_arr,
        w_arr,
        std::nullopt,
        32,
        8,
        "mxfp8",
        std::nullopt,
        std::nullopt,
        stream);
    REQUIRE(evaluation_error(out).empty());
    REQUIRE_EQ(out.shape(), Shape{m, n});
    expect_close_tol(readback_f32(stream, out), expected, 1e-4, 1e-3);
  }

  // The nvfp4 global_scale_w reaches the kernel as a scalar VIEW whose
  // storage offset inside its buffer is nonzero (a slice of a wider
  // array): the HGS binding pins the whole buffer at word zero, so the
  // offset must ride through aux_size like the quantize/dequantize
  // global-scale siblings, or the correction reads the wrong word.
  {
    int m = 2, k = 64, n = 4;
    std::vector<float> x_values(static_cast<size_t>(m) * k);
    for (auto& value : x_values) {
      value = dist(gen);
    }
    std::vector<float> w_values(static_cast<size_t>(n) * k);
    for (auto& value : w_values) {
      value = dist(gen);
    }
    float global_scale_x = 0.0f;
    float global_scale_w = 0.0f;
    for (float value : x_values) {
      global_scale_x = std::max(global_scale_x, std::abs(value));
    }
    for (float value : w_values) {
      global_scale_w = std::max(global_scale_w, std::abs(value));
    }
    auto x_hat =
        host_fp_fake_quantize(x_values, 16, 4, &global_scale_x);
    auto packed =
        host_fp_quantize_packed(w_values, n, k, 16, 4, &global_scale_w);
    auto w_hat =
        host_fp_dequantize_packed(packed, n, k, 16, 4, &global_scale_w);
    auto expected = host_fp_matmul(x_hat, w_hat, m, n, k);

    // The real scale sits at element 3 of a wider evaluated array.
    std::vector<float> padding{9.0f, 8.0f, 7.0f, global_scale_w, 5.0f};
    array flat(padding.begin(), Shape{5}, float32);
    array gs_w_view = slice(flat, Shape{3}, Shape{4});
    gs_w_view.eval();
    REQUIRE_EQ(gs_w_view.size(), 1);
    array x_arr(x_values.begin(), Shape{m, k}, float32);
    array w_words(packed.words.begin(), Shape{n, k * 4 / 32}, uint32);
    array w_scales(packed.scales.begin(), Shape{n, k / 16}, uint8);
    array gs_x_arr(global_scale_x);
    array out = qqmm(
        x_arr,
        w_words,
        w_scales,
        16,
        4,
        "nvfp4",
        std::optional<array>(gs_x_arr),
        std::optional<array>(gs_w_view),
        stream);
    REQUIRE(evaluation_error(out).empty());
    REQUIRE_EQ(out.shape(), Shape{m, n});
    expect_close_tol(readback_f32(stream, out), expected, 1e-4, 1e-3);
  }

  // Named errors: a noncanonical mode/group/bits combo keeps its mode
  // tag instead of silently misreading the mode-fixed scale stream.
  {
    int k = 32;
    std::vector<float> matrix(static_cast<size_t>(4) * k, 0.5f);
    auto host = host_fp_quantize_packed(matrix, 4, k, 32, 4, nullptr);
    array w_words(host.words.begin(), Shape{4, k * 4 / 32}, uint32);
    array w_scales(host.scales.begin(), Shape{4, k / 32}, uint8);
    std::vector<float> x_values(static_cast<size_t>(2) * k, 0.25f);
    array x(x_values.begin(), Shape{2, k}, float32);

    // nvfp4 labelled but packed at group 32.
    std::string nvfp4_error = evaluation_error(qqmm(
        x, w_words, w_scales, 32, 4, "nvfp4", std::nullopt, std::nullopt,
        stream));
    CHECK(nvfp4_error.find("QQMatmul mode") != std::string::npos);

    // mxfp8 labelled but 4-bit words.
    std::string mxfp8_error = evaluation_error(qqmm(
        x, w_words, w_scales, 32, 4, "mxfp8", std::nullopt, std::nullopt,
        stream));
    CHECK(mxfp8_error.find("QQMatmul mode") != std::string::npos);

    std::vector<uint32_t> idx_v{0u};
    array idx(idx_v.begin(), Shape{1}, uint32);
    std::string gather_error = evaluation_error(gather_qqmm(
        x, w_words, w_scales, idx, idx, 32, 4, "nvfp4", std::nullopt,
        std::nullopt, false, stream));
    CHECK(gather_error.find("GatherQQMM mode") != std::string::npos);
  }
}

// DecodeGemvSubgroup equivalence: with caps.subgroup_size==32 the
// dispatch path in primitives.cpp routes every decode row through the
// new QmmVecSubgroup kernel, so this case is the load-bearing proof
// that the production subgroup path computes the same numbers as the
// CPU double-precision dequant reference across the real decode
// shapes. The microbenchmark in tools/subgroup-bench covers a 32-elem
// float reduction in isolation; this case covers n not divisible by 8
// (workgroup width), n not divisible by 32 (subgroup width), group
// sizes 32 and 64, bits 4 and 8, and f32 / f16 / bf16 storage. bf16
// and f16 store through the same f32 accumulator inside qmm_vec.comp,
// so the reduction order determines the f32 sum bit-for-bit; storage
// quantization then matches both kernels. Where subgroup reduction
// order legitimately changes rounding, the tolerance is one f32 ulp
// at the reduction plus the storage dtype's rtol after STORE_VALUE.
//
// Tree-vs-subgroup bit-exact comparison cannot be expressed through
// the public dispatch API today (the encoder's combined
// scales+biases buffer is private). The microbenchmark covers that
// reduction-order comparison at the kernel level on identical
// 32-element float sums; this test covers the end-to-end shape
// coverage the bench does not.
//
// A red subgroup-vs-host result on the M1 ends the A/B regardless of
// speed, per the assignment.
TEST_CASE("qmm_vec subgroup dispatch matches host reference across decode shapes") {
  if (!compute_available()) {
    return;
  }
  const auto& caps = omarchy::device(0).capabilities();
  bool subgroup_ready = caps.subgroup_size == 32u &&
      (caps.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
  if (!subgroup_ready) {
    skip("subgroup size != 32 or no ARITHMETIC; subgroup variant unrunnable on this device");
    return;
  }
  Stream stream = gpu_stream();

  // Shapes: n=1 is a single-element reduction; n=7 is under one
  // workgroup width (COLUMNS_PER_GROUP=8); n=9 crosses one workgroup
  // boundary; n=37 crosses multiple workgroups with a partial last
  // workgroup; n=64 is exactly two workgroups; n=256 is large. k is
  // always 3*group_size so lane boundaries land mid-slot (a known
  // sensitive boundary for the reduction step).
  std::vector<int> n_shapes{1, 7, 9, 37, 64, 256};
  std::vector<std::pair<int, int>> qbits{{64, 4}, {32, 4}, {64, 8}, {32, 8}};
  std::vector<Dtype> dtypes{float32};
  if (float16_available()) {
    dtypes.push_back(float16);
  }
  dtypes.push_back(bfloat16);

  std::mt19937 gen(91);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  for (auto [group_size, bits] : qbits) {
    int k = group_size * 3;
    int pack = 32 / bits;
    int words_per_row = k / pack;
    int groups_per_row = k / group_size;
    for (auto dtype : dtypes) {
      for (int n : n_shapes) {
        std::vector<float> matrix(static_cast<size_t>(n) * k);
        for (auto& v : matrix) v = dist(gen);
        std::vector<float> x_values(k);
        for (auto& v : x_values) v = dist(gen);
        HostQuantizedWeights host =
            host_affine_quantize(matrix, n, k, group_size, bits);
        // Round-trip scales/biases through the storage dtype the kernel
        // sees, otherwise the host reference uses f32 storage while the
        // device rounds to bf16/f16 and the gap shows up as an
        // apparent equivalence failure unrelated to the reduction.
        std::vector<float> scales_rt = round_trip(stream, host.scales, dtype);
        std::vector<float> biases_rt = round_trip(stream, host.biases, dtype);
        HostQuantizedWeights rounded = host;
        rounded.scales = scales_rt;
        rounded.biases = biases_rt;
        std::vector<float> expected = host_quantized_matmul(
            rounded, x_values, 1, n, k, group_size, bits);

        // With subgroup_ready=true the dispatch path picks
        // QmmVecSubgroup on the M1; this is the production path.
        array x(x_values.begin(), Shape{1, k}, dtype);
        array w_words(host.words.begin(), Shape{n, words_per_row}, uint32);
        array scales(scales_rt.begin(), Shape{n, groups_per_row}, dtype);
        array biases(biases_rt.begin(), Shape{n, groups_per_row}, dtype);
        array out = quantized_matmul(
            x, w_words, scales, biases, true, group_size, bits, "affine",
            stream);
        REQUIRE(evaluation_error(out).empty());
        REQUIRE_EQ(out.shape(), Shape{1, n});
        std::vector<float> device_result = readback_f32(stream, out);

        float max_diff = 0.0f;
        int worst_idx = -1;
        for (int i = 0; i < n; ++i) {
          float d = std::fabs(device_result[i] - expected[i]);
          if (d > max_diff) {
            max_diff = d;
            worst_idx = i;
          }
        }
        // Per-dtype bound derived from reduction depth and
        // accumulator type. The device computes the dot in f32
        // throughout (qmm_vec.comp promotes scales/biases/x to f32
        // via LOAD_VALUE); the host reference computes in f64
        // (host_quantized_matmul). The gap is therefore dominated
        // by f32 vs f64 multiply-add precision across the K
        // elements of one output column, plus cross-lane reduction
        // (32 lanes summed in 5 pairwise-add rounds on both the
        // tree and the subgroupAdd path, each add rounding at
        // most 1 ulp).
        //
        // Per-element ops: scale * q + bias (2 ops) then x * (...)
        // (1 op) = 3 ops per k element. Per-lane accumulates
        // K/32 elements. 32 cross-lane adds. Total f32 multiply-add
        // count: (3 * K/32 + 32). Each op is bounded by 1 ulp at
        // f32 relative to the operand magnitude. Conservative
        // total error vs the f64 host sum:
        //
        //   E_f32 = (3*K/32 + 32) * M_max * 2^-23
        //
        // For bf16/f16 storage, STORE_VALUE rounds the f32 sum to
        // the storage dtype. The half-ulp at the rounded magnitude
        // adds:
        //
        //   E_storage = M_max * 2^-(mantissa_bits + 1)
        //
        // where mantissa_bits = 23 (f32), 10 (f16), 7 (bf16).
        //
        // Total gap bound (no cancellation assumed):
        //
        //   bound = E_f32 + E_storage
        //
        // A kernel defect (wrong shift, wrong lane, init not
        // reset) would produce a gap of order M_max, not M_max *
        // eps. The derived bound is loose enough to permit
        // legitimate reduction-order and storage rounding, tight
        // enough to catch defects.
        //
        // M_max here is the worst-case |device_result[i]| in this
        // run; using a constant derived from the input range (the
        // test uses dist(-2, 2) for matrix and x, with scales and
        // biases from the same range, and the dequantized w row
        // has magnitude at most 2 * (2^(bits-1) - 1) * 2 + 2).
        // That gives |y| up to ~50 worst case; the empirical
        // |device_result| range was 10-43 on the M1 leg. The bound
        // uses the empirical max with a 2x safety factor so a
        // single test run does not falsely reject when the run's
        // M happens to land near the upper edge of the input
        // distribution.
        double m_max_obs =
            *std::max_element(device_result.begin(), device_result.end(),
                [](float a, float b) {
                  return std::fabs(a) < std::fabs(b);
                });
        m_max_obs = std::fabs(m_max_obs);
        double m_max_bound = std::max(m_max_obs * 2.0, 50.0);
        int storage_mantissa_bits = (dtype == float32)
            ? 23
            : (dtype == float16) ? 10 : 7;
        double ops = 3.0 * k / 32.0 + 32.0;
        double e_f32 = ops * m_max_bound * std::ldexp(1.0, -23);
        double e_storage = m_max_bound *
            std::ldexp(1.0, -(storage_mantissa_bits + 1));
        double bound = e_f32 + e_storage;
        // Floor at 1e-6 so f32 storage's tighter precision is not
        // asserted tighter than the derived math (the f32 E_storage
        // term is below 1e-6 at |M|<16, which is the typical
        // case in this test).
        bound = std::max(bound, 1e-6);
        INFO("bits=" << bits << " group_size=" << group_size
             << " dtype=" << dtype << " n=" << n
             << " worst_idx=" << worst_idx
             << " device=" << device_result[worst_idx]
             << " host=" << expected[worst_idx]
             << " diff=" << max_diff
             << " bound=" << bound
             << " m_max_obs=" << m_max_obs
             << " e_f32=" << e_f32
             << " e_storage=" << e_storage);
        CHECK(max_diff <= bound);
      }
    }
  }
}

// Batched prefill: x is [batch, m, k] with DISTINCT token rows per
// batch, quantized weights shared across the batch (the 2D-w serving
// shape). The bf16-activation coopmat route widens x to f32 once
// (CastBF16F32) and the shader must index that f32 view in f32
// elements per batch; a batch term halved there shifts batch b by
// m / 2 rows, so odd batches read the wrong tokens while b % 2 == 0
// lands on batch b/2 (invisible when every batch holds
// identical rows, which is how the defect shipped: batched
// prefill/serving got wrong logits for every row after the first).
TEST_CASE("batched quantized_matmul matches host on every batch row") {
  if (!compute_available()) {
    return;
  }
  const auto& caps = omarchy::device(0).capabilities();
  bool coopmat =
      caps.cooperative_matrix_f32_8 && caps.subgroup_size == 32u;
  if (!coopmat) {
    skip("no cooperative_matrix_f32_8; batched prefill takes the tile route");
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(2026);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  int k = 2048;
  // {batch, m, n} on the dominant 4-bit group-64 transposed layout:
  // the w7B failing shape (2 and 4 batch at T=16, N=6144), odd batch
  // counts, a T that is not a tile multiple (17), the vec route (m=1),
  // and a small-n projection. f16 activations cover the no-cast f16
  // route; bits 8 / group 32 / group 128 take the scalar tile route
  // (the coopmat gate is 4-bit g64 only) and pin its batch math too.
  struct Case {
    int batch, m, n, group_size, bits;
    Dtype dtype;
    const char* label;
  };
  std::vector<Case> cases{
      {2, 16, 6144, 64, 4, bfloat16, "b4-t16-n6144"},
      {4, 16, 6144, 64, 4, bfloat16, "b8-t16-n6144"},
      {3, 16, 2048, 64, 4, bfloat16, "b3-t16-n2048"},
      {8, 4, 2048, 64, 4, bfloat16, "b8-t4-n2048"},
      {5, 17, 2048, 64, 4, bfloat16, "b5-t17-n2048"},
      {2, 64, 2048, 64, 4, bfloat16, "b2-t64-n2048"},
      {2, 1, 2048, 64, 4, bfloat16, "b2-t1-n2048-vec"},
      {4, 16, 16, 64, 4, bfloat16, "b4-t16-n16"},
      // Named route pins (w7Q review): m=16 exercises the M16 FullN
      // build; m=128 at N=2048 gives ceil(m/32)*(N/32) = 256 tile
      // units, past the cores*6 threshold where coopmat_tile_rows
      // picks 32 rows (the earlier m=40/N=256 pin stayed on the M16
      // build: 2*8 = 16 units). The dispatched kernel itself is
      // asserted in the dispatch-pins case below.
      {2, 16, 256, 64, 4, bfloat16, "b2-m16-n256-m16route"},
      {4, 16, 256, 64, 4, bfloat16, "b4-m16-n256-m16route"},
      {2, 128, 2048, 64, 4, bfloat16, "b2-m128-n2048-x32route"},
      {4, 128, 2048, 64, 4, bfloat16, "b4-m128-n2048-x32route"},
  };
  if (float16_available()) {
    cases.push_back({4, 16, 6144, 64, 4, float16, "f16-b4-t16-n6144"});
    cases.push_back({3, 16, 2048, 64, 4, float16, "f16-b3-t16-n2048"});
    cases.push_back({2, 17, 2048, 64, 4, float16, "f16-b2-t17-n2048"});
  }
  cases.push_back({4, 16, 2048, 64, 8, bfloat16, "bits8-tile"});
  cases.push_back({4, 16, 2048, 32, 4, bfloat16, "g32-tile"});
  cases.push_back({4, 16, 2048, 128, 4, bfloat16, "g128-tile"});

  for (const auto& c : cases) {
    CAPTURE(c.label);
    CAPTURE(c.batch);
    CAPTURE(c.m);
    CAPTURE(c.n);
    CAPTURE(c.group_size);
    CAPTURE(c.bits);
    int groups = k / c.group_size;
    int pack = 32 / c.bits;
    int words_per_row = k / pack;

    std::vector<float> matrix(static_cast<size_t>(c.n) * k);
    for (auto& v : matrix) {
      v = dist(gen);
    }
    HostQuantizedWeights host =
        host_affine_quantize(matrix, c.n, k, c.group_size, c.bits);
    std::vector<float> scales_rt = round_trip(stream, host.scales, c.dtype);
    std::vector<float> biases_rt = round_trip(stream, host.biases, c.dtype);
    HostQuantizedWeights rounded = host;
    rounded.scales = scales_rt;
    rounded.biases = biases_rt;

    // DISTINCT token rows per batch (the defect is invisible when
    // every batch repeats one row block).
    std::vector<float> x_flat(static_cast<size_t>(c.batch) * c.m * k);
    for (auto& v : x_flat) {
      v = dist(gen);
    }
    std::vector<float> x_rt = round_trip(stream, x_flat, c.dtype);

    array x(x_rt.begin(), Shape{c.batch, c.m, k}, c.dtype);
    array w_words(host.words.begin(), Shape{c.n, words_per_row}, uint32);
    array scales(scales_rt.begin(), Shape{c.n, groups}, c.dtype);
    array biases(biases_rt.begin(), Shape{c.n, groups}, c.dtype);
    array out = quantized_matmul(
        x, w_words, scales, biases, true, c.group_size, c.bits, "affine",
        stream);
    REQUIRE(evaluation_error(out).empty());
    REQUIRE_EQ(out.shape(), Shape{c.batch, c.m, c.n});
    std::vector<float> device_result = readback_f32(stream, out);

    // Per-batch-row f64 host reference, all batches first: the bound
    // must derive from the REFERENCE magnitude, not the device result
    // (w7Q review - a device-derived m_max widens the bound exactly
    // where a defect inflates the output). The bound follows the same
    // derivation as the qmm_vec case above with the tile route's
    // deeper k chain carried conservatively (one f32 rounding per k
    // element). With dist(-2,2) over k=2048 the reference |y| tail
    // reaches several hundred, where ONE bf16 storage ulp is already
    // ~1.0 - a fixed-magnitude bound flags legitimate storage
    // rounding. A wrong-row defect still misses by O(|y|), two orders
    // above the bound.
    std::vector<std::vector<float>> expected_batches;
    expected_batches.reserve(c.batch);
    double m_max_ref = 0.0;
    for (int b = 0; b < c.batch; ++b) {
      std::vector<float> x_batch(
          x_rt.begin() + static_cast<ptrdiff_t>(b) * c.m * k,
          x_rt.begin() + static_cast<ptrdiff_t>(b + 1) * c.m * k);
      expected_batches.push_back(host_quantized_matmul(
          rounded, x_batch, c.m, c.n, k, c.group_size, c.bits));
      for (float v : expected_batches.back()) {
        m_max_ref = std::max(m_max_ref,
            static_cast<double>(std::fabs(v)));
      }
    }
    double m_max = std::max(m_max_ref * 2.0, 50.0);
    int storage_mantissa_bits = (c.dtype == float32)
        ? 23
        : (c.dtype == float16) ? 10 : 7;
    double ops = static_cast<double>(k);
    double e_f32 = ops * m_max * std::ldexp(1.0, -23);
    double e_storage =
        m_max * std::ldexp(1.0, -(storage_mantissa_bits + 1));
    double bound = std::max(e_f32 + e_storage, 1e-6);

    for (int b = 0; b < c.batch; ++b) {
      const std::vector<float>& expected = expected_batches[b];
      double max_diff = 0.0;
      int worst = -1;
      for (int i = 0; i < c.m * c.n; ++i) {
        double d = std::fabs(static_cast<double>(
                                 device_result[static_cast<size_t>(b) *
                                         c.m * c.n +
                                     i]) -
            expected[i]);
        if (d > max_diff) {
          max_diff = d;
          worst = i;
        }
      }
      INFO("batch=", b, " worst=", worst, " diff=", max_diff,
          " bound=", bound, " got=",
          device_result[static_cast<size_t>(b) * c.m * c.n + worst],
          " want=", expected[worst]);
      CHECK(max_diff <= bound);
    }
  }
}

// Batch independence on the coopmat route: every output row's k chain
// is independent of which batch (and of how many sibling batches) run
// beside it, so [batch, m, k] must be bit-identical to each [1, m, k]
// slice run alone. bf16/f16 widen exactly into f32, so memcmp on the
// f32 readback is storage-bit equality.
TEST_CASE("batched quantized_matmul is bit-identical per batch row alone") {
  if (!compute_available()) {
    return;
  }
  const auto& caps = omarchy::device(0).capabilities();
  bool coopmat =
      caps.cooperative_matrix_f32_8 && caps.subgroup_size == 32u;
  if (!coopmat) {
    skip("no cooperative_matrix_f32_8; batched prefill takes the tile route");
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(409);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  int k = 2048;
  int n = 2048;
  int group_size = 64;
  int bits = 4;
  int groups = k / group_size;
  int pack = 32 / bits;
  int words_per_row = k / pack;
  std::vector<float> matrix(static_cast<size_t>(n) * k);
  for (auto& v : matrix) {
    v = dist(gen);
  }
  HostQuantizedWeights host = host_affine_quantize(matrix, n, k, group_size, bits);
  std::vector<float> scales_rt = round_trip(stream, host.scales, bfloat16);
  std::vector<float> biases_rt = round_trip(stream, host.biases, bfloat16);
  array w_words(host.words.begin(), Shape{n, words_per_row}, uint32);
  array scales(scales_rt.begin(), Shape{n, groups}, bfloat16);
  array biases(biases_rt.begin(), Shape{n, groups}, bfloat16);

  for (int batch : {2, 5}) {
    for (int m : {1, 2, 17, 64}) {
      CAPTURE(batch);
      CAPTURE(m);
      std::vector<float> x_flat(static_cast<size_t>(batch) * m * k);
      for (auto& v : x_flat) {
        v = dist(gen);
      }
      std::vector<float> x_rt = round_trip(stream, x_flat, bfloat16);
      array x(x_rt.begin(), Shape{batch, m, k}, bfloat16);
      array full = quantized_matmul(
          x, w_words, scales, biases, true, group_size, bits, "affine",
          stream);
      REQUIRE(evaluation_error(full).empty());
      std::vector<float> full_bits = readback_f32(stream, full);
      for (int b = 0; b < batch; ++b) {
        std::vector<float> x_one(
            x_rt.begin() + static_cast<ptrdiff_t>(b) * m * k,
            x_rt.begin() + static_cast<ptrdiff_t>(b + 1) * m * k);
        array x1(x_one.begin(), Shape{1, m, k}, bfloat16);
        array alone = quantized_matmul(
            x1, w_words, scales, biases, true, group_size, bits, "affine",
            stream);
        REQUIRE(evaluation_error(alone).empty());
        std::vector<float> alone_bits = readback_f32(stream, alone);
        REQUIRE_EQ(full_bits.size(),
            static_cast<size_t>(batch) * m * n);
        REQUIRE_EQ(alone_bits.size(), static_cast<size_t>(m) * n);
        size_t row_bytes = static_cast<size_t>(m) * n * sizeof(float);
        bool equal = std::memcmp(
            full_bits.data() + static_cast<size_t>(b) * m * n,
            alone_bits.data(),
            row_bytes) == 0;
        INFO("batch=", b, " rows differ from the alone run");
        CHECK(equal);
      }
    }
  }
}

// w7Q review: the batched-prefill pins must prove WHICH compiled build
// ran, not only that some coopmat kernel was numerically right. The
// dispatch trace ("[rtmod] DISPATCH kernel=<id>", ids = the
// ComputeKernel enum) is captured around one call and matched against
// the named enum member, so a route silently drifting to another build
// fails here instead of quietly covering a different shader site.
// Sites covered: qmm_tile's X_F32 x_slice (M16 FullN and 32-row
// FullN builds) and qmm_tile_two's (the TwoN build, which only
// dispatches under MLX_OMARCHY_QMM_TWON=1 with N % 64 == 0).
// qmm_tile_two correctness is asserted against the f64 host
// reference, so this case fails-before on the unfixed shader.
TEST_CASE("coopmat prefill dispatch pins M16, 32-row and TwoN builds") {
  if (!compute_available()) {
    return;
  }
  const auto& caps = omarchy::device(0).capabilities();
  bool coopmat =
      caps.cooperative_matrix_f32_8 && caps.subgroup_size == 32u;
  if (!coopmat) {
    skip("no cooperative_matrix_f32_8; dispatch pins need the coopmat route");
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(614);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  int k = 2048;
  int group_size = 64;
  int bits = 4;
  int groups = k / group_size;
  int words_per_row = k / 8;
  std::vector<float> w_matrix_2048(static_cast<size_t>(2048) * k);
  for (auto& v : w_matrix_2048) {
    v = dist(gen);
  }
  std::vector<float> w_matrix_256(static_cast<size_t>(256) * k);
  for (auto& v : w_matrix_256) {
    v = dist(gen);
  }
  HostQuantizedWeights host_2048 =
      host_affine_quantize(w_matrix_2048, 2048, k, group_size, bits);
  HostQuantizedWeights host_256 =
      host_affine_quantize(w_matrix_256, 256, k, group_size, bits);
  std::vector<float> scales_rt = round_trip(stream, host_2048.scales, bfloat16);
  std::vector<float> biases_rt =
      round_trip(stream, host_2048.biases, bfloat16);
  std::vector<float> scales_rt_256 =
      round_trip(stream, host_256.scales, bfloat16);
  std::vector<float> biases_rt_256 =
      round_trip(stream, host_256.biases, bfloat16);

  auto captured = [&](const std::function<void()>& call) {
    std::fflush(stderr);
    int saved = ::dup(::fileno(stderr));
    FILE* cap = std::fopen("/tmp/qmm_dispatch_cap.log", "w");
    REQUIRE(cap != nullptr);
    ::dup2(::fileno(cap), ::fileno(stderr));
    setenv("MLX_OMARCHY_TRACE_DISPATCH", "1", 1);
    call();
    omarchy::get_command_encoder(stream).synchronize();
    std::fflush(stderr);
    ::dup2(saved, ::fileno(stderr));
    ::close(saved);
    std::fclose(cap);
    unsetenv("MLX_OMARCHY_TRACE_DISPATCH");
    std::ifstream in("/tmp/qmm_dispatch_cap.log");
    std::string text((std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());
    return text;
  };
  auto dispatches = [](const std::string& text,
                           uint32_t kernel_id) {
    return text.find("[rtmod] DISPATCH kernel=" +
                     std::to_string(kernel_id) + " ") !=
        std::string::npos;
  };
  auto seen_kernels = [](const std::string& text) {
    std::string ids;
    for (size_t pos = text.find("kernel=");
        pos != std::string::npos;
        pos = text.find("kernel=", pos + 1)) {
      size_t begin = pos + 7;
      size_t end = text.find_first_not_of("0123456789", begin);
      ids += text.substr(begin, end - begin) + " ";
    }
    return ids;
  };
  auto kernel_of = [](omarchy::ComputeKernel kernel) {
    return static_cast<uint32_t>(kernel);
  };

  struct Pin {
    int m, n;
    const char* env_name;
    omarchy::ComputeKernel expected;
    const char* label;
  };
  // The M16 full-n route defaults to the LDS-padded build on G14C and to the
  // plain build elsewhere (primitives.cpp, default_pad).
  const bool g14c = caps.device_name.find("G14C") != std::string::npos;
  std::vector<Pin> pins{
      {16, 256, nullptr,
          g14c ? omarchy::ComputeKernel::QmmPrefillCoopmatM16BF16X32FullNLdsPad
               : omarchy::ComputeKernel::QmmPrefillCoopmatM16BF16X32FullN,
          g14c ? "M16FullNLdsPad" : "M16FullN"},
      {128, 2048, "MLX_OMARCHY_QMM_NO_RASTER",
          omarchy::ComputeKernel::QmmPrefillCoopmatBF16X32FullN,
          "X32FullN"},
      {128, 2048, "MLX_OMARCHY_QMM_TWON",
          omarchy::ComputeKernel::QmmPrefillCoopmatBF16X32FullNTwoN,
          "TwoN"},
  };

  for (const auto& pin : pins) {
    CAPTURE(pin.label);
    HostQuantizedWeights& host = pin.n == 256 ? host_256 : host_2048;
    std::vector<float>& scales_use = pin.n == 256 ? scales_rt_256 : scales_rt;
    std::vector<float>& biases_use = pin.n == 256 ? biases_rt_256 : biases_rt;
    std::vector<float> x_flat(static_cast<size_t>(2) * pin.m * k);
    for (auto& v : x_flat) {
      v = dist(gen);
    }
    std::vector<float> x_rt = round_trip(stream, x_flat, bfloat16);
    array x(x_rt.begin(), Shape{2, pin.m, k}, bfloat16);
    array w_words(host.words.begin(), Shape{pin.n, words_per_row}, uint32);
    array scales(scales_use.begin(), Shape{pin.n, groups}, bfloat16);
    array biases(biases_use.begin(), Shape{pin.n, groups}, bfloat16);

    const char* old_env = pin.env_name != nullptr
        ? std::getenv(pin.env_name) : nullptr;
    std::string saved_env = old_env != nullptr ? old_env : "";
    if (pin.env_name != nullptr) {
      setenv(pin.env_name, "1", 1);
    }
    std::optional<array> out_holder;
    std::string trace = captured([&] {
      out_holder = quantized_matmul(
          x, w_words, scales, biases, true, group_size, bits, "affine",
          stream);
      out_holder->eval();
    });
    if (pin.env_name != nullptr) {
      if (old_env != nullptr) {
        setenv(pin.env_name, saved_env.c_str(), 1);
      } else {
        unsetenv(pin.env_name);
      }
    }
    uint32_t expected_id = kernel_of(pin.expected);
    bool hit = dispatches(trace, expected_id);
    INFO("pin=", pin.label, " expected kernel id ", expected_id,
        " seen ids: ", seen_kernels(trace));
    CHECK(hit);

    if (std::string(pin.label) == "TwoN") {
      // The TwoN build owns the qmm_tile_two x_slice site: its batched
      // rows must also be numerically right, so this pin fails-before
      // on the unfixed shader.
      REQUIRE(evaluation_error(*out_holder).empty());
      std::vector<float> device_result = readback_f32(stream, *out_holder);
      std::vector<std::vector<float>> expected_batches;
      double m_max_ref = 0.0;
      for (int b = 0; b < 2; ++b) {
        std::vector<float> x_batch(
            x_rt.begin() + static_cast<ptrdiff_t>(b) * pin.m * k,
            x_rt.begin() + static_cast<ptrdiff_t>(b + 1) * pin.m * k);
        expected_batches.push_back(host_quantized_matmul(
            host, x_batch, pin.m, pin.n, k, group_size, bits));
        for (float v : expected_batches.back()) {
          m_max_ref = std::max(m_max_ref,
              static_cast<double>(std::fabs(v)));
        }
      }
      double m_max = std::max(m_max_ref * 2.0, 50.0);
      double bound = k * m_max * std::ldexp(1.0, -23) +
          m_max * std::ldexp(1.0, -8);
      for (int b = 0; b < 2; ++b) {
        double max_diff = 0.0;
        for (int i = 0; i < pin.m * pin.n; ++i) {
          max_diff = std::max(max_diff,
              std::fabs(static_cast<double>(
                            device_result[static_cast<size_t>(b) *
                                    pin.m * pin.n +
                                i]) -
                  expected_batches[b][i]));
        }
        INFO("TwoN batch=", b, " diff=", max_diff, " bound=", bound);
        CHECK(max_diff <= bound);
      }
    }
  }
}

namespace {

struct QmmVecQ4WordGate {
  explicit QmmVecQ4WordGate(bool on) {
    set(on);
  }
  ~QmmVecQ4WordGate() {
    unsetenv("MLX_OMARCHY_QMM_VEC_Q4_WORD");
  }
  void set(bool on) {
    setenv("MLX_OMARCHY_QMM_VEC_Q4_WORD", on ? "1" : "0", 1);
  }
};

// Forces the PrefillQmmTile dispatch gates. Always restores every variable to
// unset, so an aborting REQUIRE cannot leak a switch into later cases.
struct QmmTileGate {
  explicit QmmTileGate(bool on, bool register_block = false) {
    set(on);
    set_register_block(register_block);
  }
  ~QmmTileGate() {
    unsetenv("MLX_OMARCHY_QMM_TILE");
    unsetenv("MLX_OMARCHY_QMM_TILE_RB");
  }
  void set(bool on) {
    setenv("MLX_OMARCHY_QMM_TILE", on ? "1" : "0", 1);
  }
  void set_register_block(bool on) {
    setenv("MLX_OMARCHY_QMM_TILE_RB", on ? "1" : "0", 1);
  }
};

} // namespace

TEST_CASE("qmm_vec packed-word candidate matches baseline and host reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<Dtype> dtypes{float32};
  if (float16_available()) {
    dtypes.push_back(float16);
  }
  dtypes.push_back(bfloat16);

  auto finite_max = [](const std::vector<float>& values, double& max_abs) {
    max_abs = 0.0;
    for (float v : values) {
      if (!std::isfinite(v)) {
        return false;
      }
      max_abs = std::max(max_abs, std::fabs(static_cast<double>(v)));
    }
    return true;
  };
  auto max_diff = [](
      const std::vector<float>& lhs, const std::vector<float>& rhs) {
    double result = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
      result = std::max(
          result, std::fabs(static_cast<double>(lhs[i]) - rhs[i]));
    }
    return result;
  };
  auto storage_mantissa = [](Dtype dtype) {
    return dtype == float32 ? 23 : (dtype == float16 ? 10 : 7);
  };

  const std::vector<std::pair<int, int>> shapes{
      {64, 7}, {896, 9}, {4864, 37}, {8192, 7}};
  for (auto dtype : dtypes) {
    for (auto [k, n] : shapes) {
      INFO("dtype=" << dtype << " n=" << n << " k=" << k);
      std::mt19937 gen(static_cast<unsigned>(k + n * 101));
      std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
      std::vector<float> matrix(static_cast<size_t>(n) * k);
      std::vector<float> x_values(k);
      for (auto& v : matrix) v = dist(gen);
      for (auto& v : x_values) v = dist(gen);

      HostQuantizedWeights host = host_affine_quantize(matrix, n, k, 64, 4);
      HostQuantizedWeights rounded = host;
      rounded.scales = round_trip(stream, host.scales, dtype);
      rounded.biases = round_trip(stream, host.biases, dtype);
      std::vector<float> x_rt = round_trip(stream, x_values, dtype);
      std::vector<float> expected =
          host_quantized_matmul(rounded, x_rt, 1, n, k, 64, 4);

      array x(x_rt.begin(), Shape{1, k}, dtype);
      array w_words(host.words.begin(), Shape{n, k / 8}, uint32);
      array scales(rounded.scales.begin(), Shape{n, k / 64}, dtype);
      array biases(rounded.biases.begin(), Shape{n, k / 64}, dtype);

      QmmVecQ4WordGate gate(false);
      array base = quantized_matmul(
          x, w_words, scales, biases, true, 64, 4, "affine", stream);
      const auto base_error = evaluation_error(base);
      REQUIRE_MESSAGE(base_error.empty(), base_error);
      std::vector<float> base_v = readback_f32(stream, base);

      gate.set(true);
      array candidate = quantized_matmul(
          x, w_words, scales, biases, true, 64, 4, "affine", stream);
      const auto candidate_error = evaluation_error(candidate);
      REQUIRE_MESSAGE(candidate_error.empty(), candidate_error);
      std::vector<float> candidate_v = readback_f32(stream, candidate);

      REQUIRE_EQ(candidate_v.size(), base_v.size());
      REQUIRE_EQ(candidate_v.size(), expected.size());
      double base_max = 0.0;
      double candidate_max = 0.0;
      double host_max = 0.0;
      REQUIRE_MESSAGE(finite_max(base_v, base_max),
          "baseline qmm_vec produced a non-finite output");
      REQUIRE_MESSAGE(finite_max(candidate_v, candidate_max),
          "packed-word qmm_vec produced a non-finite output");
      REQUIRE_MESSAGE(finite_max(expected, host_max),
          "host reference produced a non-finite output");

      double magnitude =
          std::max({base_max, candidate_max, host_max, 1.0}) * 2.0;
      double f32_error = (3.0 * k / 32.0 + 32.0) * magnitude *
          std::ldexp(1.0, -23);
      double storage_error =
          magnitude * std::ldexp(1.0, -(storage_mantissa(dtype) + 1));
      double host_bound = std::max(f32_error + storage_error, 1e-6);
      double pair_bound = 2.0 * host_bound;
      double candidate_host_diff = max_diff(candidate_v, expected);
      double base_host_diff = max_diff(base_v, expected);
      double pair_diff = max_diff(candidate_v, base_v);
      INFO("candidate_host_diff=" << candidate_host_diff
           << " base_host_diff=" << base_host_diff
           << " pair_diff=" << pair_diff
           << " host_bound=" << host_bound
           << " pair_bound=" << pair_bound);
      CHECK(candidate_host_diff <= host_bound);
      CHECK(base_host_diff <= host_bound);
      CHECK(pair_diff <= pair_bound);
    }
  }

  // A row-contiguous f16 slice can start between 16-byte boundaries. The
  // packed activation view must materialize it before its uvec4 loads.
  if (float16_available()) {
    constexpr int view_k = 64;
    constexpr int view_n = 9;
    std::vector<float> matrix(static_cast<size_t>(view_n) * view_k);
    std::vector<float> x_values(view_k);
    for (size_t i = 0; i < matrix.size(); ++i) {
      matrix[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.03125f;
    }
    for (int i = 0; i < view_k; ++i) {
      x_values[i] = static_cast<float>((i * 7) % 23 - 11) * 0.0625f;
    }
    HostQuantizedWeights weights =
        host_affine_quantize(matrix, view_n, view_k, 64, 4);
    weights.scales = round_trip(stream, weights.scales, float16);
    weights.biases = round_trip(stream, weights.biases, float16);
    x_values = round_trip(stream, x_values, float16);
    std::vector<float> parent_values(view_k + 1, 123.0f);
    std::copy(x_values.begin(), x_values.end(), parent_values.begin() + 1);
    array parent = astype(
        array(parent_values.begin(), Shape{1, view_k + 1}, float32),
        float16,
        stream);
    array x_view = slice(parent, {0, 1}, {1, view_k + 1}, stream);
    array x_aligned(x_values.begin(), Shape{1, view_k}, float16);
    array w(
        weights.words.begin(), Shape{view_n, view_k / 8}, uint32);
    array scales(
        weights.scales.begin(), Shape{view_n, view_k / 64}, float16);
    array biases(
        weights.biases.begin(), Shape{view_n, view_k / 64}, float16);
    QmmVecQ4WordGate gate(true);
    auto aligned = readback_f32(stream, quantized_matmul(
        x_aligned, w, scales, biases, true, 64, 4, "affine", stream));
    auto unaligned = readback_f32(stream, quantized_matmul(
        x_view, w, scales, biases, true, 64, 4, "affine", stream));
    CHECK_EQ(unaligned, aligned);
  }

  const int k = 64;
  const int n = 9;
  std::vector<float> matrix(static_cast<size_t>(n) * k, 0.25f);
  std::vector<float> x_values(k, -0.5f);
  HostQuantizedWeights host = host_affine_quantize(matrix, n, k, 64, 8);
  array x(x_values.begin(), Shape{1, k}, float32);
  array w_words(host.words.begin(), Shape{n, k / 4}, uint32);
  array scales(host.scales.begin(), Shape{n, 1}, float32);
  array biases(host.biases.begin(), Shape{n, 1}, float32);
  QmmVecQ4WordGate gate(false);
  std::vector<float> base_v = readback_f32(stream, quantized_matmul(
      x, w_words, scales, biases, true, 64, 8, "affine", stream));
  gate.set(true);
  std::vector<float> gated_v = readback_f32(stream, quantized_matmul(
      x, w_words, scales, biases, true, 64, 8, "affine", stream));
  CHECK_EQ(gated_v, base_v);
}

TEST_CASE("qmm tile covers the large-prefill arithmetic path") {
  if (!compute_available() || !float16_available()) {
    return;
  }
  Stream stream = gpu_stream();
  constexpr int m = 1024;
  constexpr int n = 17;
  constexpr int k = 64;
  std::vector<float> x_values(static_cast<size_t>(m) * k);
  for (int row = 0; row < m; ++row) {
    for (int inner = 0; inner < k; ++inner) {
      x_values[row * k + inner] =
          static_cast<float>((row * 3 + inner * 5) % 17 - 8) * 0.03125f;
    }
  }
  HostQuantizedWeights weights;
  weights.words.resize(static_cast<size_t>(n) * (k / 8));
  weights.scales.assign(n, 0.03125f);
  weights.biases.assign(n, -0.25f);
  for (int column = 0; column < n; ++column) {
    for (int pack = 0; pack < k / 8; ++pack) {
      uint32_t word = 0;
      for (int lane = 0; lane < 8; ++lane) {
        word |= static_cast<uint32_t>((column + pack + lane * 3) & 15)
            << (lane * 4);
      }
      weights.words[column * (k / 8) + pack] = word;
    }
  }
  std::vector<float> x_rounded = round_trip(stream, x_values, float16);
  std::vector<float> expected =
      host_quantized_matmul(weights, x_rounded, m, n, k, 64, 4);
  array x(x_rounded.begin(), Shape{m, k}, float16);
  array w(weights.words.begin(), Shape{n, k / 8}, uint32);
  array scales(weights.scales.begin(), Shape{n, 1}, float16);
  array biases(weights.biases.begin(), Shape{n, 1}, float16);
  QmmTileGate gate(true, true);
  array out = quantized_matmul(
      x, w, scales, biases, true, 64, 4, "affine", stream);
  REQUIRE(evaluation_error(out).empty());
  expect_close_tol(readback_f32(stream, out), expected, 1e-3, 1e-3);
}

// PrefillQmmTile equivalence: the env-gated m-tiled kernel must match
// the general qmm.comp kernel it stands in for at matrix_m > 1. Both
// sides run on device through the public dispatch - the gate selects
// qmm_tile unless MLX_OMARCHY_QMM_TILE is "0" (OFF), selecting qmm.comp.
//
// Bound derivation. Both kernels accumulate x * (scale*q + bias) over
// ascending k into a single f32 accumulator per output element in the
// SAME order, so there is no reduction-depth term between them: the
// residual gap is (a) FMA contraction differences between two shaders
// whose expression shapes differ - at most one f32 ulp per k step
// against the output magnitude, (k + 2) * M * 2^-23 - and (b) the
// final STORE_VALUE rounding, at most one storage-dtype ulp,
// M * 2^-(mantissa + 1). A defect (row/column swap, tail overrun,
// barrier misordering, wrong group index) misses by O(M), not by
// eps-scale dust, because the dequantized weights span both signs.
//
// The host-anchored tail case (m=17, n=37) pins the tile kernel to the
// f64 affine contract itself, so tile cannot drift in step with
// qmm.comp; the f16/bf16 anchor round-trips x, scales, and biases
// through the storage dtype so the reference sees exactly the operand
// rounding the kernel sees. The anchor bound adds the per-element
// dequant (2 ops) and mul-add (1 op) across the sequential k
// reduction: (3k + 1) * M * 2^-23 plus storage rounding, the same
// derivation as the qmm_vec case above.
//
// Every device readback is asserted finite BEFORE the max/abs
// reductions. NaN never wins a `>` comparison, so one NaN in the
// output would leave max_diff at 0 and false-pass the CHECK; +Inf
// would inflate the observed maximum and with it the bound to Inf,
// and Inf <= Inf compares true. The finite gate closes both holes.
//
// The default sweep matrix is bounded: every dtype x bits/group combo
// over m {2, 15, 16, 17, 1023} at the hidden width (n = k = 896, all
// m tail classes), plus the wide 4864 x 4864 gate/up projection at
// both m extremes for a 4-bit/f16 and an 8-bit/bf16 combo. The full
// m x n x k x quant x dtype cross product is opt-in for hardware
// validation runs via MLX_OMARCHY_QMM_TILE_FULL_SWEEP=1; it is the
// same cases through the same helper, unbounded in software-GPU time.
TEST_CASE("qmm tile matches host reference and qmm.comp across prefill shapes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  std::vector<Dtype> dtypes{float32};
  if (float16_available()) {
    dtypes.push_back(float16);
  }
  dtypes.push_back(bfloat16);
  std::vector<std::pair<int, int>> qbits{
      {32, 4}, {64, 4}, {32, 8}, {64, 8}};

  auto storage_mantissa = [](Dtype dtype) {
    return dtype == float32 ? 23 : (dtype == float16 ? 10 : 7);
  };

  // Maximum magnitude over a readback, refusing non-finite values:
  // returns false if any element is NaN or Inf, with the finite max in
  // max_abs either way.
  auto finite_max = [](const std::vector<float>& values, double& max_abs) {
    max_abs = 0.0;
    for (float v : values) {
      if (!std::isfinite(v)) {
        return false;
      }
      max_abs = std::max(max_abs, std::fabs(static_cast<double>(v)));
    }
    return true;
  };

  // One tile-vs-qmm.comp case through the public dispatch: gate ON
  // selects qmm_tile, gate OFF selects qmm.comp, same inputs. Carries
  // the finite guard, the derived bound, and the worst-element report.
  auto run_tile_case = [&](Dtype dtype, int group_size, int bits, int k,
      int n, int m, unsigned seed) {
    const int pack = 32 / bits;
    const int words_per_row = k / pack;
    const int groups_per_row = k / group_size;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> matrix(static_cast<size_t>(n) * k);
    for (auto& v : matrix) {
      v = dist(gen);
    }
    std::vector<float> x_values(static_cast<size_t>(m) * k);
    for (auto& v : x_values) {
      v = dist(gen);
    }
    HostQuantizedWeights host =
        host_affine_quantize(matrix, n, k, group_size, bits);
    HostQuantizedWeights rounded = host;
    rounded.scales = round_trip(stream, host.scales, dtype);
    rounded.biases = round_trip(stream, host.biases, dtype);
    array x(x_values.begin(), Shape{m, k}, dtype);
    array w_words(host.words.begin(), Shape{n, words_per_row}, uint32);
    array scales(rounded.scales.begin(), Shape{n, groups_per_row}, dtype);
    array biases(rounded.biases.begin(), Shape{n, groups_per_row}, dtype);

    QmmTileGate gate(true);
    array out_tile = quantized_matmul(
        x, w_words, scales, biases, true, group_size, bits, "affine",
        stream);
    INFO("dtype=", dtype, " group=", group_size, " bits=", bits,
        " m=", m, " n=", n, " k=", k);
    const auto error = evaluation_error(out_tile);
    REQUIRE_MESSAGE(error.empty(), error);
    std::vector<float> tile_v = readback_f32(stream, out_tile);
    gate.set(false);
    array out_base = quantized_matmul(
        x, w_words, scales, biases, true, group_size, bits, "affine",
        stream);
    std::vector<float> base_v = readback_f32(stream, out_base);

    REQUIRE_EQ(tile_v.size(), base_v.size());
    double tile_max = 0.0;
    double base_max = 0.0;
    REQUIRE_MESSAGE(finite_max(tile_v, tile_max),
        "tile kernel produced a non-finite output");
    REQUIRE_MESSAGE(finite_max(base_v, base_max),
        "qmm kernel produced a non-finite output");
    double m_bound = std::max(std::max(tile_max, base_max) * 2.0, 1.0);
    double bound =
        (static_cast<double>(k) + 2.0) * m_bound * std::ldexp(1.0, -23) +
        m_bound * std::ldexp(1.0, -(storage_mantissa(dtype) + 1));
    bound = std::max(bound, 1e-6);
    double max_diff = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < tile_v.size(); ++i) {
      double d =
          std::fabs(static_cast<double>(tile_v[i]) - base_v[i]);
      if (d > max_diff) {
        max_diff = d;
        worst = i;
      }
    }
    INFO("tile case dtype=" << dtype << " bits=" << bits
         << " group_size=" << group_size << " m=" << m << " n=" << n
         << " k=" << k << " worst=" << worst << " diff=" << max_diff
         << " bound=" << bound);
    CHECK(max_diff <= bound);
  };

  // Host-anchored tail case: m and n straddle one tile boundary each;
  // k = 3 groups puts group boundaries inside every k span the tile
  // crosses. x, scales, and biases are round-tripped through the
  // storage dtype so the host reference sees the kernel's operands.
  for (auto dtype : dtypes) {
    for (auto [group_size, bits] : qbits) {
      const int m = 17;
      const int n = 37;
      const int k = group_size * 3;
      const int pack = 32 / bits;
      const int words_per_row = k / pack;
      const int groups_per_row = k / group_size;
      std::mt19937 gen(17);
      std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
      std::vector<float> matrix(static_cast<size_t>(n) * k);
      for (auto& v : matrix) {
        v = dist(gen);
      }
      std::vector<float> x_values(static_cast<size_t>(m) * k);
      for (auto& v : x_values) {
        v = dist(gen);
      }
      HostQuantizedWeights host =
          host_affine_quantize(matrix, n, k, group_size, bits);
      HostQuantizedWeights rounded = host;
      rounded.scales = round_trip(stream, host.scales, dtype);
      rounded.biases = round_trip(stream, host.biases, dtype);
      std::vector<float> x_rt = round_trip(stream, x_values, dtype);
      std::vector<float> expected =
          host_quantized_matmul(rounded, x_rt, m, n, k, group_size, bits);

      QmmTileGate gate(true);
      array x(x_rt.begin(), Shape{m, k}, dtype);
      array w_words(host.words.begin(), Shape{n, words_per_row}, uint32);
      array scales(rounded.scales.begin(), Shape{n, groups_per_row}, dtype);
      array biases(rounded.biases.begin(), Shape{n, groups_per_row}, dtype);
      array out = quantized_matmul(
          x, w_words, scales, biases, true, group_size, bits, "affine",
          stream);
      REQUIRE(evaluation_error(out).empty());
      REQUIRE_EQ(out.shape(), Shape{m, n});
      std::vector<float> device_result = readback_f32(stream, out);

      double m_max = 0.0;
      REQUIRE_MESSAGE(finite_max(device_result, m_max),
          "tile kernel produced a non-finite output");
      double m_bound = std::max(m_max * 2.0, 1.0);
      double ops = 3.0 * k + 1.0;
      double bound = ops * m_bound * std::ldexp(1.0, -23) +
          m_bound * std::ldexp(1.0, -(storage_mantissa(dtype) + 1));
      bound = std::max(bound, 1e-6);
      double max_diff = 0.0;
      size_t worst = 0;
      for (size_t i = 0; i < expected.size(); ++i) {
        double d = std::fabs(
            static_cast<double>(device_result[i]) - expected[i]);
        if (d > max_diff) {
          max_diff = d;
          worst = i;
        }
      }
      INFO("anchor bits=" << bits << " group_size=" << group_size
           << " dtype=" << dtype << " worst=" << worst
           << " diff=" << max_diff << " bound=" << bound);
      CHECK(max_diff <= bound);
    }
  }

  // Device-vs-device sweep: gate ON must reproduce qmm.comp. Weights
  // quantize once per case; seeds are order-deterministic.
  const bool full_sweep = [] {
    const char* v = std::getenv("MLX_OMARCHY_QMM_TILE_FULL_SWEEP");
    return v != nullptr && v[0] == '1';
  }();
  std::vector<int> m_shapes = full_sweep
      ? std::vector<int>{2, 15, 16, 17, 64, 255, 256, 1023}
      : std::vector<int>{2, 15, 16, 17, 1023};
  std::vector<int> n_shapes =
      full_sweep ? std::vector<int>{64, 896, 4864} : std::vector<int>{896};
  std::vector<int> k_shapes =
      full_sweep ? std::vector<int>{896, 4864} : std::vector<int>{896};
  unsigned seed = 47;
  for (auto dtype : dtypes) {
    for (auto [group_size, bits] : qbits) {
      for (int k : k_shapes) {
        for (int n : n_shapes) {
          for (int m : m_shapes) {
            run_tile_case(dtype, group_size, bits, k, n, m, seed++);
          }
        }
      }
    }
  }
  if (!full_sweep) {
    // The 4864 x 4864 gate/up projection at both m extremes, one
    // 4-bit/f16 and one 8-bit/bf16 combo, keeps the largest real
    // shapes covered without the cross product.
    for (Dtype dtype : {float16, bfloat16}) {
      if (std::find(dtypes.begin(), dtypes.end(), dtype) ==
          dtypes.end()) {
        continue;
      }
      for (int m : {2, 1023}) {
        run_tile_case(dtype, 32, 4, 4864, 4864, m, seed++);
        run_tile_case(dtype, 64, 8, 4864, 4864, m, seed++);
      }
    }
  }
}

TEST_CASE("qmm non-transposed small-k zone matches host reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 shader and storage features.");
    return;
  }

  // Host dot in double precision against a [k, n] non-transposed packed
  // weight: word (row, col / pack) holds the codes along n.
  auto host_nt_matmul = [](const HostQuantizedWeights& w,
                           const std::vector<float>& x,
                           int m,
                           int n,
                           int k,
                           int group_size,
                           int bits) {
    int pack = 32 / bits;
    int words_per_row = n / pack;
    int groups = n / group_size;
    uint32_t mask = (1u << bits) - 1u;
    std::vector<float> out(static_cast<size_t>(m) * n);
    for (int row = 0; row < m; ++row) {
      for (int column = 0; column < n; ++column) {
        double acc = 0.0;
        for (int inner = 0; inner < k; ++inner) {
          uint32_t code =
              (w.words[inner * words_per_row + column / pack] >>
              ((column % pack) * bits)) &
              mask;
          double dequant =
              static_cast<double>(code) *
              w.scales[inner * groups + column / group_size] +
              w.biases[inner * groups + column / group_size];
          acc += host_at(x, row * k + inner) * dequant;
        }
        out[static_cast<size_t>(row) * n + column] = static_cast<float>(acc);
      }
    }
    return out;
  };

  auto finite_max = [](const std::vector<float>& values, double& max_abs) {
    max_abs = 0.0;
    for (float v : values) {
      if (!std::isfinite(v)) {
        return false;
      }
      max_abs = std::max(max_abs, static_cast<double>(std::fabs(v)));
    }
    return true;
  };

  // One non-transposed case: batched=false builds x [m, k] against a
  // 2D weight, batched_shared pairs x [B, m, k] with the shared 2D
  // weight, and batched_paired builds a 3D weight one slice per batch
  // index. The bound mirrors the transposed anchor: fp32 matmul
  // accumulation plus one storage-dtype rounding per output element.
  auto run_nt_case = [&](Dtype dtype,
                         int group_size,
                         int bits,
                         int m,
                         int n,
                         int k,
                         int batch,
                         bool paired,
                         unsigned seed) {
    int storage_mantissa_bits =
        dtype == float32 ? 23 : (dtype == float16 ? 10 : 7);
    int pack = 32 / bits;
    // Non-transposed packing runs along n: w is [K, n * bits / 32]
    // with scales/biases [K, n / group_size]; n must be a group
    // multiple for the shape contract.
    int words_per_row = n / pack;
    int groups = n / group_size;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> matrix(static_cast<size_t>(k) * n);
    std::vector<float> x_values(
        static_cast<size_t>(batch) * m * k);
    for (auto& value : matrix) {
      value = dist(gen);
    }
    for (auto& value : x_values) {
      value = dist(gen);
    }
    HostQuantizedWeights host =
        host_affine_quantize(matrix, k, n, group_size, bits);
    auto round_trip = [&](const std::vector<float>& values) {
      array device(
          values.begin(), Shape{static_cast<int>(values.size())}, float32);
      return readback_f32(
          stream, astype(astype(device, dtype, stream), float32, stream));
    };
    HostQuantizedWeights rounded = host;
    rounded.scales = round_trip(host.scales);
    rounded.biases = round_trip(host.biases);
    std::vector<float> x_rt = round_trip(x_values);
    std::vector<float> expected =
        host_nt_matmul(rounded, x_rt, batch * m, n, k, group_size, bits);

    // A paired 3D weight carries one full copy of the quantized matrix
    // per batch slice; both slices are equal, so the host reference
    // stays the single-matrix product over batch * m rows.
    HostQuantizedWeights paired_words = rounded;
    if (paired) {
      paired_words.words.insert(
          paired_words.words.end(),
          rounded.words.begin(),
          rounded.words.end());
      paired_words.scales.insert(
          paired_words.scales.end(),
          rounded.scales.begin(),
          rounded.scales.end());
      paired_words.biases.insert(
          paired_words.biases.end(),
          rounded.biases.begin(),
          rounded.biases.end());
    }
    auto words_shape = Shape{k, words_per_row};
    auto params_shape = Shape{k, groups};
    if (paired) {
      words_shape.insert(words_shape.begin(), batch);
      params_shape.insert(params_shape.begin(), batch);
    }
    array x(
        x_rt.begin(),
        Shape{batch * m, k},
        dtype);
    array w_words(paired_words.words.begin(), words_shape, uint32);
    array w_scales(paired_words.scales.begin(), params_shape, dtype);
    array w_biases(paired_words.biases.begin(), params_shape, dtype);
    // A rank-3 x rides the broadcast/pairing contracts of the eval; a
    // paired weight needs the matching rank.
    if (batch > 1) {
      x = reshape(x, Shape{batch, m, k}, stream);
    }
    array out = quantized_matmul(
        x, w_words, w_scales, w_biases, /*transpose=*/false, group_size,
        bits, "affine", stream);
    INFO("nt case dtype=" << dtype << " bits=" << bits
         << " group_size=" << group_size << " m=" << m << " n=" << n
         << " k=" << k << " batch=" << batch << " paired=" << paired);
    const auto error = evaluation_error(out);
    REQUIRE_MESSAGE(error.empty(), error);
    REQUIRE_EQ(
        out.shape(),
        batch > 1 ? Shape{batch, m, n} : Shape{m, n});
    std::vector<float> device_values = readback_f32(stream, out);
    REQUIRE_EQ(device_values.size(), expected.size());
    double out_max = 0.0;
    REQUIRE_MESSAGE(
        finite_max(device_values, out_max), "non-finite qmm nt output");
    double m_bound = std::max(out_max * 2.0, 1.0);
    double ops = 3.0 * k + 1.0;
    double bound = ops * m_bound * std::ldexp(1.0, -23) +
        m_bound * std::ldexp(1.0, -(storage_mantissa_bits + 1));
    bound = std::max(bound, 1e-6);
    double max_diff = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
      double d =
          std::fabs(static_cast<double>(device_values[i]) - expected[i]);
      if (d > max_diff) {
        max_diff = d;
        worst = i;
      }
    }
    INFO("nt worst=" << worst << " diff=" << max_diff
         << " bound=" << bound);
    CHECK(max_diff <= bound);
  };

  // The bf16 composed zone: every affine bit width the route serves,
  // group sizes 32/64/128, aligned and straddling m, and the k = 64,
  // n = 2048 projection class the 2B hybrid backward trips on.
  for (auto [group_size, bits] :
       std::vector<std::pair<int, int>>{
           {32, 4}, {64, 4}, {128, 4}, {32, 8}, {64, 8}}) {
    int k = std::min(2 * group_size, 128);
    run_nt_case(bfloat16, group_size, bits, 17, 128, k, 1, false, 101);
    run_nt_case(bfloat16, group_size, bits, 33, 128, 128, 1, false, 103);
  }
  run_nt_case(bfloat16, 64, 4, 32, 2048, 64, 1, false, 107);
  // Rank-3 x: one shared 2D weight broadcast across the batch, and one
  // 3D weight paired slice per batch index.
  run_nt_case(bfloat16, 64, 4, 17, 128, 128, 2, false, 109);
  run_nt_case(bfloat16, 64, 4, 17, 128, 128, 2, true, 113);
  // The routes this change did not touch stay pinned: f16 and f32 keep
  // the direct tile route, and bits 2 bf16 passes it, all inside the
  // same k <= 128 non-transposed zone.
  run_nt_case(float16, 64, 4, 17, 128, 128, 1, false, 127);
  run_nt_case(float32, 64, 4, 17, 128, 128, 1, false, 131);
  run_nt_case(bfloat16, 32, 2, 17, 128, 128, 1, false, 137);
}

TEST_CASE(
    "qmm packed-word register-block prefill matches baseline tile and host") {
  if (!compute_available() || !float16_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // On a cooperative_matrix_f32_8 device the register-block gate selects
  // QmmPrefillCoopmatF16 (a different accumulation order), so the
  // bitwise pin against the 16x16 tile only holds off such a device; the
  // host bound below applies everywhere.
  const auto& caps = omarchy::device(0).capabilities();
  const bool coopmat_device =
      caps.cooperative_matrix_f32_8 && caps.subgroup_size == 32;

  for (auto [m, n, k, seed] : {
           std::tuple{2, 7, 64, 67u},
           std::tuple{15, 17, 192, 69u},
           std::tuple{32, 32, 128, 71u},
           std::tuple{33, 17, 64, 73u}}) {
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr int pack = 32 / bits;
    const int words_per_row = k / pack;
    const int groups_per_row = k / group_size;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> matrix(static_cast<size_t>(n) * k);
    std::vector<float> x_values(static_cast<size_t>(m) * k);
    for (auto& value : matrix) {
      value = dist(gen);
    }
    for (auto& value : x_values) {
      value = dist(gen);
    }

    HostQuantizedWeights weights =
        host_affine_quantize(matrix, n, k, group_size, bits);
    weights.scales = round_trip(stream, weights.scales, float16);
    weights.biases = round_trip(stream, weights.biases, float16);
    x_values = round_trip(stream, x_values, float16);
    const std::vector<float> expected = host_quantized_matmul(
        weights, x_values, m, n, k, group_size, bits);

    array x(x_values.begin(), Shape{m, k}, float16);
    array w_words(weights.words.begin(), Shape{n, words_per_row}, uint32);
    array scales(
        weights.scales.begin(), Shape{n, groups_per_row}, float16);
    array biases(
        weights.biases.begin(), Shape{n, groups_per_row}, float16);

    QmmTileGate gate(true, true);
    array candidate = quantized_matmul(
        x, w_words, scales, biases, true, group_size, bits, "affine", stream);
    INFO("packed-word candidate m=" << m << " n=" << n << " k=" << k);
    const auto candidate_error = evaluation_error(candidate);
    REQUIRE_MESSAGE(candidate_error.empty(), candidate_error);
    const std::vector<float> candidate_values = readback_f32(stream, candidate);

    gate.set_register_block(false);
    array baseline = quantized_matmul(
        x, w_words, scales, biases, true, group_size, bits, "affine", stream);
    const auto baseline_error = evaluation_error(baseline);
    REQUIRE_MESSAGE(baseline_error.empty(), baseline_error);
    const std::vector<float> baseline_values = readback_f32(stream, baseline);

    REQUIRE_EQ(candidate_values.size(), baseline_values.size());
    REQUIRE_EQ(candidate_values.size(), expected.size());
    double max_abs = 1.0;
    double candidate_baseline_max = 0.0;
    for (size_t index = 0; index < expected.size(); ++index) {
      REQUIRE(std::isfinite(candidate_values[index]));
      REQUIRE(std::isfinite(baseline_values[index]));
      REQUIRE(std::isfinite(expected[index]));
      max_abs = std::max({
          max_abs,
          std::fabs(static_cast<double>(candidate_values[index])),
          std::fabs(static_cast<double>(baseline_values[index])),
          std::fabs(static_cast<double>(expected[index]))});
      candidate_baseline_max = std::max(
          candidate_baseline_max,
          std::fabs(static_cast<double>(candidate_values[index]) -
              baseline_values[index]));
      if (!coopmat_device) {
        CHECK_EQ(candidate_values[index], baseline_values[index]);
      }
    }
    const double bound =
        (3.0 * k + 1.0) * (2.0 * max_abs) * std::ldexp(1.0, -23) +
        (2.0 * max_abs) * std::ldexp(1.0, -11);
    double candidate_host_max = 0.0;
    double baseline_host_max = 0.0;
    for (size_t index = 0; index < expected.size(); ++index) {
      const double candidate_error = std::fabs(
          static_cast<double>(candidate_values[index]) - expected[index]);
      const double baseline_error = std::fabs(
          static_cast<double>(baseline_values[index]) - expected[index]);
      candidate_host_max = std::max(candidate_host_max, candidate_error);
      baseline_host_max = std::max(baseline_host_max, baseline_error);
      INFO("index=" << index << " candidate=" << candidate_values[index]
           << " baseline=" << baseline_values[index]
           << " expected=" << expected[index] << " bound=" << bound);
      CHECK(candidate_error <= bound);
      CHECK(baseline_error <= bound);
    }
    std::cout << "[qmm-rb-word] m=" << m << " n=" << n << " k=" << k
              << " candidate_vs_baseline_max=" << candidate_baseline_max
              << " candidate_vs_host_max=" << candidate_host_max
              << " baseline_vs_host_max=" << baseline_host_max
              << " bound=" << bound << "\n";
  }
}

// QmmPrefillCoopmat: the transposed affine 4-bit/group-64 f16 prefill at
// the Qwen2.5-0.5B-4bit shapes (hidden 896, intermediate 4864, kv 128)
// against the f64 host reference, at m spanning one 8-row block, one
// full 32-row tile, and the two README prefill lengths (262, 1053) that
// leave a tail row block. On a cooperative_matrix_f32_8 device with
// subgroup size 32 the default dispatch runs QmmPrefillCoopmatF16; on
// llvmpipe and stock Mesa it runs the register-blocked tile, so the
// case pins both to the same anchor bound as the qmm tile case
// ((3k + 1) f32 ops plus one f16 store rounding against twice the
// output magnitude). Coopmat accumulates the 8-wide products in the
// matrix unit's own order, so there is no bitwise pin against the tile.
TEST_CASE("qmm coopmat prefill matches host reference at Qwen shapes") {
  if (!compute_available() || !float16_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& caps = omarchy::device(0).capabilities();
  const bool coopmat_device =
      caps.cooperative_matrix_f32_8 && caps.subgroup_size == 32;
  std::cout << "[provenance] cooperative_matrix_f32_8="
            << (caps.cooperative_matrix_f32_8 ? 1 : 0)
            << " subgroup_size=" << caps.subgroup_size
            << " -> f16 q4/g64 prefill runs "
            << (coopmat_device ? "QmmPrefillCoopmatF16" : "QmmTileRbF16")
            << "\n";

  constexpr int group_size = 64;
  constexpr int bits = 4;
  unsigned seed = 90u;
  for (auto [k, n] : {std::pair{896, 896}, std::pair{896, 4864},
           std::pair{4864, 896}, std::pair{896, 128}}) {
    const int words_per_row = k / (32 / bits);
    const int groups_per_row = k / group_size;
    std::mt19937 gen(seed++);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> matrix(static_cast<size_t>(n) * k);
    for (auto& value : matrix) {
      value = dist(gen);
    }
    HostQuantizedWeights weights =
        host_affine_quantize(matrix, n, k, group_size, bits);
    weights.scales = round_trip(stream, weights.scales, float16);
    weights.biases = round_trip(stream, weights.biases, float16);
    array w_words(weights.words.begin(), Shape{n, words_per_row}, uint32);
    array scales(
        weights.scales.begin(), Shape{n, groups_per_row}, float16);
    array biases(
        weights.biases.begin(), Shape{n, groups_per_row}, float16);

    for (int m : {8, 32, 262, 1053}) {
      std::vector<float> x_values(static_cast<size_t>(m) * k);
      for (auto& value : x_values) {
        value = dist(gen);
      }
      x_values = round_trip(stream, x_values, float16);
      const std::vector<float> expected = host_quantized_matmul(
          weights, x_values, m, n, k, group_size, bits);
      array x(x_values.begin(), Shape{m, k}, float16);

      QmmTileGate gate(true, true);
      array out = quantized_matmul(
          x, w_words, scales, biases, true, group_size, bits, "affine",
          stream);
      INFO("coopmat prefill m=" << m << " n=" << n << " k=" << k);
      const auto error = evaluation_error(out);
      REQUIRE_MESSAGE(error.empty(), error);
      REQUIRE_EQ(out.shape(), Shape{m, n});
      const std::vector<float> device_values = readback_f32(stream, out);
      REQUIRE_EQ(device_values.size(), expected.size());

      double max_abs = 1.0;
      for (size_t index = 0; index < expected.size(); ++index) {
        REQUIRE(std::isfinite(device_values[index]));
        REQUIRE(std::isfinite(expected[index]));
        max_abs = std::max({
            max_abs,
            std::fabs(static_cast<double>(device_values[index])),
            std::fabs(static_cast<double>(expected[index]))});
      }
      const double bound =
          (3.0 * k + 1.0) * (2.0 * max_abs) * std::ldexp(1.0, -23) +
          (2.0 * max_abs) * std::ldexp(1.0, -11);
      double max_diff = 0.0;
      size_t worst = 0;
      for (size_t index = 0; index < expected.size(); ++index) {
        const double d = std::fabs(
            static_cast<double>(device_values[index]) - expected[index]);
        if (d > max_diff) {
          max_diff = d;
          worst = index;
        }
      }
      INFO("worst=" << worst << " device=" << device_values[worst]
           << " expected=" << expected[worst] << " diff=" << max_diff
           << " bound=" << bound);
      CHECK(max_diff <= bound);
      std::cout << "[qmm-coopmat] kernel="
                << (coopmat_device ? "QmmPrefillCoopmatF16" : "QmmTileRbF16")
                << " m=" << m << " n=" << n << " k=" << k
                << " max_abs_err=" << max_diff
                << " rel_err=" << max_diff / max_abs
                << " bound=" << bound << "\n";
    }
  }
}

// The coopmat gate used to reroute a row-contiguous x view at an odd
// f16-element offset to the register-blocked tile kernel, whose different
// accumulation order silently changes generated tokens under allocator
// pressure (receipts/2026-09-10-qmm-splitk-parity: long-decode-128
// flipped 4cc08910 -> f873dc2b under concurrent GPU load, and a
// device_info() call before model load triggered the same flip). The
// dispatch now stages an unaligned view into an aligned buffer, so the
// same data must produce bit-identical output regardless of the view's
// offset parity. On a device without cooperative-matrix support both
// arms take the tile route and the check is inert; on the M1 fork it
// fails on pre-fix code because the odd arm lands on the tile kernel.
TEST_CASE("qmm coopmat output is bit-identical across x offset alignment") {
  if (!compute_available() || !float16_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& caps = omarchy::device(0).capabilities();
  const bool coopmat_device =
      caps.cooperative_matrix_f32_8 && caps.subgroup_size == 32;
  std::cout << "[provenance] cooperative_matrix_f32_8="
            << (caps.cooperative_matrix_f32_8 ? 1 : 0)
            << " subgroup_size=" << caps.subgroup_size
            << " -> offset-parity repro "
            << (coopmat_device ? "compares coopmat vs staged coopmat"
                               : "is inert without coopmat support")
            << "\n";

  constexpr int group_size = 64;
  constexpr int bits = 4;
  constexpr int m = 64;
  constexpr int k = 896;
  constexpr int n = 896;
  const int words_per_row = k / (32 / bits);
  const int groups_per_row = k / group_size;
  std::mt19937 gen(410u);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> matrix(static_cast<size_t>(n) * k);
  for (auto& value : matrix) {
    value = dist(gen);
  }
  HostQuantizedWeights weights =
      host_affine_quantize(matrix, n, k, group_size, bits);
  weights.scales = round_trip(stream, weights.scales, float16);
  weights.biases = round_trip(stream, weights.biases, float16);
  array w_words(weights.words.begin(), Shape{n, words_per_row}, uint32);
  array scales(weights.scales.begin(), Shape{n, groups_per_row}, float16);
  array biases(weights.biases.begin(), Shape{n, groups_per_row}, float16);

  std::vector<float> x_values(static_cast<size_t>(m) * k);
  for (auto& value : x_values) {
    value = dist(gen);
  }
  x_values = round_trip(stream, x_values, float16);
  const std::vector<float> expected =
      host_quantized_matmul(weights, x_values, m, n, k, group_size, bits);

  // Aligned arm: a whole buffer, element offset 0.
  array x_aligned(x_values.begin(), Shape{m, k}, float16);

  // Unaligned arm: the same values as a row-contiguous view whose first
  // element sits at f16 element offset 1 (byte offset 2) inside a parent
  // buffer - the shape memory pressure produces.
  std::vector<float> parent_values(static_cast<size_t>(m) * k + 1, 123.0f);
  std::copy(x_values.begin(), x_values.end(), parent_values.begin() + 1);
  array parent(parent_values.begin(), Shape{m * k + 1}, float16);
  parent.eval();
  array x_odd = reshape(slice(parent, {1}, {m * k + 1}, stream), {m, k});
  // Slice and reshape are graph nodes; the shared-buffer view with its
  // nonzero offset materializes when the node evaluates.
  x_odd.eval();
  REQUIRE_EQ(x_odd.offset(), size_t{2});
  REQUIRE(x_odd.flags().row_contiguous);

  QmmTileGate gate(true, true);
  array out_aligned = quantized_matmul(
      x_aligned, w_words, scales, biases, true, group_size, bits, "affine",
      stream);
  array out_odd = quantized_matmul(
      x_odd, w_words, scales, biases, true, group_size, bits, "affine",
      stream);

  // Determinism contract: offset parity must not change one bit of the
  // generated activations (exact f32 widenings of the f16 outputs;
  // f16 -> f32 is exact, and finite data rules out signed-zero noise).
  const auto aligned = readback_f32(stream, out_aligned);
  const auto odd = readback_f32(stream, out_odd);
  REQUIRE_EQ(aligned.size(), expected.size());
  REQUIRE_EQ(odd.size(), expected.size());
  size_t mismatched = 0;
  size_t worst = 0;
  for (size_t index = 0; index < aligned.size(); ++index) {
    if (aligned[index] != odd[index]) {
      ++mismatched;
      worst = index;
    }
  }
  INFO("offset-parity mismatches=" << mismatched << " worst=" << worst
       << " aligned=" << aligned[worst] << " odd=" << odd[worst]
       << (coopmat_device ? " (old code reroutes the odd arm to the tile"
                          : ""));
  CHECK_EQ(mismatched, size_t{0});

  // Both arms also hold the coopmat host-reference bound, so the staged
  // arm is not merely self-consistent but correct.
  double max_abs = 1.0;
  for (size_t index = 0; index < expected.size(); ++index) {
    REQUIRE(std::isfinite(aligned[index]));
    REQUIRE(std::isfinite(odd[index]));
    max_abs = std::max({
        max_abs,
        std::fabs(static_cast<double>(aligned[index])),
        std::fabs(static_cast<double>(odd[index])),
        std::fabs(static_cast<double>(expected[index]))});
  }
  const double bound =
      (3.0 * k + 1.0) * (2.0 * max_abs) * std::ldexp(1.0, -23) +
      (2.0 * max_abs) * std::ldexp(1.0, -11);
  double aligned_diff = 0.0;
  double odd_diff = 0.0;
  for (size_t index = 0; index < expected.size(); ++index) {
    aligned_diff = std::max(aligned_diff, std::fabs(
        static_cast<double>(aligned[index]) - expected[index]));
    odd_diff = std::max(odd_diff, std::fabs(
        static_cast<double>(odd[index]) - expected[index]));
  }
  INFO("aligned_diff=" << aligned_diff << " odd_diff=" << odd_diff
       << " bound=" << bound);
  CHECK(aligned_diff <= bound);
  CHECK(odd_diff <= bound);
  std::cout << "[qmm-offset-parity] coopmat_device=" << coopmat_device
            << " m=" << m << " n=" << n << " k=" << k
            << " mismatches=" << mismatched
            << " aligned_diff=" << aligned_diff
            << " odd_diff=" << odd_diff << " bound=" << bound << "\n";
}

// ---------------------------------------------------------------------------
// The bf16 coopmat alpha == 1 bit-identity pin (2026-09-11 bf16-alpha-fix).
//
// The fix made MatmulBF16Coopmat scale its f32 accumulator by alpha at the
// drain. At alpha == 1 that multiply is exact, so every alpha == 1 dispatch
// (the projection and residual matmuls that always rode this kernel) must
// stay bit-identical to the pre-fix kernel, which ignored alpha entirely.
// The digests below are FNV-1a-64 over the raw bf16 output bits of seeded
// workloads, captured in the bf16-alpha-fix D2 window on the fork driver
// (M1-class, subgroup 32), where the pre-fix kernel (fix's relaxed gate,
// pre-fix shader) and the fixed kernel produced byte-identical dumps for
// every alpha == 1 leg. A one-bit move in an alpha == 1 output changes the
// digest and fails the test - bit-identity, not a tolerance. Re-pin only
// with a named, reviewed mechanism change; never widen to a tolerance.

namespace {

uint64_t alpha1_fnv1a64(const void* data, size_t bytes) {
  const uint8_t* p = (const uint8_t*)data;
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

// The D2 instrument's exact input generator (the receipt probe's
// convention); the pinned digests are only meaningful against these bytes.
std::vector<float> alpha1_pattern(size_t count, uint32_t seed) {
  std::vector<float> data(count);
  uint32_t state = seed;
  for (size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    data[i] = ((state >> 8) & 0xFFFF) / 16384.0f - 2.0f;
  }
  return data;
}

}  // namespace

TEST_CASE("MatmulBF16Coopmat alpha==1 stays bit-identical to the pre-fix drain") {
  if (!compute_available()) {
    return;
  }
  const auto& caps = omarchy::device(0).capabilities();
  const bool coopmat_device =
      caps.cooperative_matrix_f32_8 && caps.subgroup_size == 32;
  if (!coopmat_device || std::getenv("MLX_OMARCHY_NO_COOPMAT") != nullptr) {
    skip(
        "alpha==1 bit pin exercises MatmulBF16Coopmat (subgroup 32 +"
        " cooperative_matrix_f32_8); other devices run the untouched"
        " non-coopmat kernel.");
    return;
  }
  Stream stream = gpu_stream();
  struct Pin {
    int m, k, n;
    uint64_t digest;
  };
  const Pin pins[] = {
      {32, 16, 32, 0xb2f727a0b237bd3cull},
      {64, 32, 64, 0x0810652b84898613ull},
      {96, 64, 96, 0xa0fe9153f643412full},
      {128, 8, 64, 0x72c6181aaaa64c3full},
  };
  for (const auto& pin : pins) {
    auto a_data = alpha1_pattern((size_t)pin.m * pin.k, 700 + pin.m);
    auto b_data =
        alpha1_pattern((size_t)pin.k * pin.n, 700 + pin.m + 101);
    array a = astype(
        array(a_data.begin(), Shape{pin.m, pin.k}, float32), bfloat16,
        stream);
    array b = astype(
        array(b_data.begin(), Shape{pin.k, pin.n}, float32), bfloat16,
        stream);
    auto out = matmul(a, b, stream);
    out.eval();
    synchronize(stream);
    const uint16_t* bits = out.data<uint16_t>();
    uint64_t digest = alpha1_fnv1a64(bits, out.size() * 2);
    if (digest != pin.digest) {
      std::cout << "  [alpha1-pin] mat " << pin.m << "x" << pin.k << "x"
                << pin.n << " got=0x" << std::hex << digest << " want=0x"
                << pin.digest << std::dec << "\n";
    }
    CHECK_EQ(digest, pin.digest);
  }
}

// ---------------------------------------------------------------------------
// Prefill-axes twins (receipts/2026-10-03-prefill-axes): the rasterization
// order and issue-quality variants of the X32 FullN prefill route must be
// byte-identical to the shipped route on every shape they can be selected
// for - odd M across 16..2048 and the K,N of the pinned 2B/4B/9B models.
// The Qwen3.8-class snapshots are bf16-hybrid (q4/g64, bf16 scales/biases),
// exactly the route the variants twin. Each arm reruns the same tree with
// one env set; the outputs must be bit-equal (a workgroup-to-tile remap or
// a shared A-tile load never touches a per-output ascending-k chain).
// The landed default (multi-row-tile grids -> G4 order) is itself an arm
// against the shipped mapping. Routing itself (which kernel id each env
// selects) is proven per arm in the on-device GPU_PROFILE dispatch
// census, not here.
TEST_CASE("qmm prefill axes twins are bit-identical to the shipped route") {
  if (!compute_available()) {
    return;
  }
  const auto& caps = omarchy::device(0).capabilities();
  const bool coopmat_device =
      caps.cooperative_matrix_f32_8 && caps.subgroup_size == 32;
  if (!coopmat_device || std::getenv("MLX_OMARCHY_NO_COOPMAT") != nullptr) {
    skip("prefill-axes twins run on the coopmat X32 route only");
    return;
  }
  Stream stream = gpu_stream();

  constexpr int group_size = 64;
  constexpr int bits = 4;
  struct Variant {
    const char* name;
    const char* key;
    const char* value;
    const char* key2;
    const char* value2;
  };
  // The baseline pins the SHIPPED rasterization order via the opt-out
  // env, so the pin keeps holding the shipped mapping even though the
  // landed default routes multi-row-tile grids to the G4 order. The
  // shared B-tile row stride is padded by default (QmmPeak), so the
  // baseline also sets the LDSPAD kill switch to 0: the pin holds the
  // shipped unpadded kernel, and the default (no-env) arm proves the
  // padded default bit-identical to it.
  const Variant variants[] = {
      {"default-g4-pad", nullptr, nullptr, nullptr, nullptr},
      {"raster-swap", "MLX_OMARCHY_QMM_RASTER", "swap", nullptr, nullptr},
      {"raster-g2", "MLX_OMARCHY_QMM_RASTER", "2", nullptr, nullptr},
      {"raster-g8", "MLX_OMARCHY_QMM_RASTER", "8", nullptr, nullptr},
      {"twon", "MLX_OMARCHY_QMM_TWON", "1", nullptr, nullptr},
      {"persist-4", "MLX_OMARCHY_QMM_PERSIST", "4", nullptr, nullptr},
      {"persist-8", "MLX_OMARCHY_QMM_PERSIST", "8", nullptr, nullptr},
      {"ldspad0", "MLX_OMARCHY_QMM_LDSPAD", "0", nullptr, nullptr},
      {"chunk", "MLX_OMARCHY_QMM_CHUNK", "1", "MLX_OMARCHY_QMM_LDSPAD", "0"},
      {"chunk-pad",
       "MLX_OMARCHY_QMM_CHUNK",
       "1",
       "MLX_OMARCHY_QMM_LDSPAD",
       "1"},
  };

  unsigned seed = 407u;
  for (auto [k, n] : {std::pair{2048, 2048}, std::pair{2048, 6144},
           std::pair{6144, 2048}, std::pair{2560, 9728},
           std::pair{9728, 2560}, std::pair{4096, 4096},
           std::pair{4096, 12288}, std::pair{12288, 4096}}) {
    const int words_per_row = k / (32 / bits);
    const int groups_per_row = k / group_size;
    std::mt19937 gen(seed++);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> matrix(static_cast<size_t>(n) * k);
    for (auto& value : matrix) {
      value = dist(gen);
    }
    HostQuantizedWeights weights =
        host_affine_quantize(matrix, n, k, group_size, bits);
    // bf16-hybrid: the pinned models store scales and biases in bf16;
    // x is bf16, so the op routes through CastBF16F32 + X32 FullN.
    weights.scales = round_trip(stream, weights.scales, bfloat16);
    weights.biases = round_trip(stream, weights.biases, bfloat16);
    array w_words(weights.words.begin(), Shape{n, words_per_row}, uint32);
    array scales(
        weights.scales.begin(), Shape{n, groups_per_row}, bfloat16);
    array biases(
        weights.biases.begin(), Shape{n, groups_per_row}, bfloat16);

    for (int m :
         {17, 63, 65, 127, 129, 255, 257, 511, 513, 1023, 1025, 2047}) {
      std::vector<float> x_values(static_cast<size_t>(m) * k);
      for (auto& value : x_values) {
        value = dist(gen);
      }
      array x(x_values.begin(), Shape{m, k}, bfloat16);

      auto run_bits = [&]() {
        array out = quantized_matmul(
            x, w_words, scales, biases, true, group_size, bits, "affine",
            stream);
        out.eval();
        const uint16_t* p = out.data<uint16_t>();
        return std::vector<uint16_t>(p, p + out.size());
      };
      setenv("MLX_OMARCHY_QMM_NO_RASTER", "1", 1);
      setenv("MLX_OMARCHY_QMM_LDSPAD", "0", 1);
      const std::vector<uint16_t> baseline = run_bits();
      unsetenv("MLX_OMARCHY_QMM_NO_RASTER");
      unsetenv("MLX_OMARCHY_QMM_LDSPAD");
      for (const auto& v : variants) {
        if (v.key != nullptr) {
          setenv(v.key, v.value, 1);
        }
        if (v.key2 != nullptr) {
          setenv(v.key2, v.value2, 1);
        }
        std::vector<uint16_t> got = run_bits();
        if (v.key != nullptr) {
          unsetenv(v.key);
        }
        if (v.key2 != nullptr) {
          unsetenv(v.key2);
        }
        REQUIRE_EQ(got.size(), baseline.size());
        size_t mismatches = 0;
        size_t first = 0;
        for (size_t i = 0; i < got.size(); ++i) {
          if (got[i] != baseline[i]) {
            if (mismatches == 0) {
              first = i;
            }
            ++mismatches;
          }
        }
        INFO("variant=", v.name, " m=", m, " k=", k, " n=", n,
             " first_mismatch_index=", first, " got=", got[first],
             " baseline=", baseline[first]);
        CHECK_EQ(mismatches, size_t{0});
      }
    }
    std::cout << "[prefill-axes] k=" << k << " n=" << n
              << " all 10 twin arms bit-identical, m in {17..2047 odd}\n";
  }
}

// The CPU quantized matmul full-row dot accumulates in float32 whatever
// the activation dtype: at K=16384 a bfloat16 accumulator drifts several
// percent from the double-precision dequant reference while every
// 64-wide group dot stays correct, so the row result stops being the sum
// of its group partials (reported at K=16384, bits=4, g=64, bf16, with
// K=8192 and K=12288 still inside the bound). The odd 129-group case
// pins the same contract on a non-power-of-two group count.
TEST_CASE("cpu quantized matmul full-row dot matches host at large K") {
  if (!cpu::is_available()) {
    skip("no CPU backend");
    return;
  }
  set_default_device(Device::cpu);
  Stream cpu_stream = new_stream(Device::cpu);
  std::mt19937 gen(16384);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  auto cpu_round_trip = [&](const std::vector<float>& values, Dtype dtype) {
    array f32(values.begin(),
              Shape{static_cast<int>(values.size())},
              float32);
    array round = astype(astype(f32, dtype, cpu_stream), float32, cpu_stream);
    round.eval();
    synchronize(cpu_stream);
    const float* data = round.data<float>();
    return std::vector<float>(data, data + round.size());
  };

  auto run_case = [&](int k, int n, int group_size, int bits) {
    CAPTURE(k);
    CAPTURE(n);
    CAPTURE(group_size);
    CAPTURE(bits);
    constexpr int m = 1;
    int groups = k / group_size;
    int pack = 32 / bits;
    int words_per_row = k / pack;

    std::vector<float> matrix(static_cast<size_t>(n) * k);
    for (auto& value : matrix) {
      value = dist(gen);
    }
    HostQuantizedWeights host =
        host_affine_quantize(matrix, n, k, group_size, bits);
    std::vector<float> x_values(m * k);
    for (auto& value : x_values) {
      value = dist(gen);
    }
    // bf16 grid: the device sees bfloat16 activations, packed codes, and
    // bfloat16 scales/biases; the host reference uses the same rounded
    // values so the only difference under test is the accumulation dtype.
    std::vector<float> x_bf = cpu_round_trip(x_values, bfloat16);
    std::vector<float> scales_bf = cpu_round_trip(host.scales, bfloat16);
    std::vector<float> biases_bf = cpu_round_trip(host.biases, bfloat16);
    host.scales = scales_bf;
    host.biases = biases_bf;

    array x(x_bf.begin(), Shape{m, k}, bfloat16);
    array w_words(host.words.begin(), Shape{n, words_per_row}, uint32);
    array scales(scales_bf.begin(), Shape{n, groups}, bfloat16);
    array biases(biases_bf.begin(), Shape{n, groups}, bfloat16);
    array out = quantized_matmul(
        x,
        w_words,
        scales,
        biases,
        /*transpose=*/true,
        group_size,
        bits,
        "affine",
        cpu_stream);
    out = astype(out, float32, cpu_stream);
    out.eval();
    synchronize(cpu_stream);
    std::vector<float> device(
        out.data<float>(),
        out.data<float>() + out.size());
    auto expected =
        host_quantized_matmul(host, x_bf, m, n, k, group_size, bits);
    expect_close(device, expected, 5e-2);
  };

  run_case(16384, 8, 64, 4); // the reported failure: 256 groups, bf16
  run_case(8256, 4, 64, 4); // 129-group tail: non-power-of-two group count
}

// Sorted-expert prefill route of gather_qmm (shaders/gather_qmm_tile.comp):
// x is [B, 1, K] with one expert id per row and the ids sorted, the shape
// mlx-lm's SwitchLinear passes at N*k >= 64 rows. The tile kernel must equal
// an f32 reference built on the CPU stream, and the per-row subgroup route
// (MLX_OMARCHY_GATHER_QMM_TILE=0), whatever the expert-run lengths: runs
// shorter than a 32-row tile, runs spanning tiles, an expert with no rows,
// and ids that are not actually sorted.
TEST_CASE("gather_qmm sorted-expert tile route matches the f32 reference and the per-row route") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  Stream cpu = default_stream(Device::cpu);
  auto rel_l2 = [](const std::vector<float>& got, const std::vector<float>& want) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
      double d = double(got[i]) - double(want[i]);
      num += d * d;
      den += double(want[i]) * double(want[i]);
    }
    return std::sqrt(num) / std::max(std::sqrt(den), 1e-12);
  };
  auto run_case = [&](const std::vector<int>& counts,
                      int n,
                      int k,
                      Dtype dtype,
                      bool shuffled,
                      double tolerance) {
    const int experts = static_cast<int>(counts.size());
    std::vector<uint32_t> rhs_v;
    for (int e = 0; e < experts; ++e) {
      rhs_v.insert(rhs_v.end(), counts[e], static_cast<uint32_t>(e));
    }
    if (shuffled) {
      std::mt19937 gen(7);
      std::shuffle(rhs_v.begin(), rhs_v.end(), gen);
    }
    const int rows = static_cast<int>(rhs_v.size());
    std::vector<uint32_t> lhs_v(rows);
    for (int i = 0; i < rows; ++i) {
      lhs_v[i] = static_cast<uint32_t>(i);
    }
    array w = astype(
        random::normal({experts, n, k}, float32, 0.0f, 1.0f, std::nullopt, cpu),
        dtype,
        cpu);
    auto parts = quantize(w, 64, 4, "affine", std::nullopt, cpu);
    array x = astype(
        random::normal({rows, 1, k}, float32, 0.0f, 1.0f, std::nullopt, cpu),
        dtype,
        cpu);
    array lhs(lhs_v.begin(), Shape{rows}, uint32);
    array rhs(rhs_v.begin(), Shape{rows}, uint32);
    array w_deq = dequantize(
        parts[0], parts[1], parts[2], 64, 4, "affine", std::nullopt, float32, cpu);
    array w_sel = take(w_deq, rhs, 0, cpu);
    array reference = matmul(
        astype(x, float32, cpu), swapaxes(w_sel, 1, 2, cpu), cpu);
    reference.eval();
    const float* ref_data = reference.data<float>();
    std::vector<float> expected(ref_data, ref_data + reference.size());

    auto run_gather = [&](const char* tile) {
      setenv("MLX_OMARCHY_GATHER_QMM_TILE", tile, 1);
      array out = gather_qmm(
          x,
          parts[0],
          parts[1],
          parts[2],
          lhs,
          rhs,
          /*transpose=*/true,
          64,
          4,
          "affine",
          std::nullopt,
          /*sorted_indices=*/true,
          stream);
      REQUIRE(evaluation_error(out).empty());
      int64_t kernel = omarchy::trace::counters().last_dispatched_kernel.load();
      return std::make_pair(readback_f32(stream, out), kernel);
    };
    auto tiled = run_gather("1");
    auto per_row = run_gather("0");
    unsetenv("MLX_OMARCHY_GATHER_QMM_TILE");

    const int64_t tile_kernel = static_cast<int64_t>(
        dtype == float16 ? omarchy::ComputeKernel::GatherQmmTileF16
                         : omarchy::ComputeKernel::GatherQmmTileBF16);
    const int64_t other_tile_kernels[2] = {
        static_cast<int64_t>(omarchy::ComputeKernel::GatherQmmNbTileF16),
        static_cast<int64_t>(omarchy::ComputeKernel::GatherQmmNbTileBF16)};
    REQUIRE_EQ(tiled.second, tile_kernel);
    CHECK(per_row.second != tile_kernel);
    CHECK(per_row.second != other_tile_kernels[0]);
    CHECK(per_row.second != other_tile_kernels[1]);

    double vs_ref = rel_l2(tiled.first, expected);
    double vs_row = rel_l2(tiled.first, per_row.first);
    size_t identical = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
      identical += tiled.first[i] == per_row.first[i];
    }
    std::cout << "[gather_qmm_tile] rows=" << rows << " E=" << experts << " N=" << n
              << " K=" << k << (dtype == float16 ? " f16" : " bf16")
              << (shuffled ? " shuffled" : "") << " tile-vs-f32-ref " << vs_ref
              << " tile-vs-per-row " << vs_row << " bit-identical "
              << identical << "/" << expected.size() << "\n";
    std::string what_ref = "tile route vs f32 reference, rel L2 " +
        std::to_string(vs_ref) + " rows=" + std::to_string(rows);
    std::string what_row = "tile route vs per-row route, rel L2 " +
        std::to_string(vs_row) + " rows=" + std::to_string(rows);
    CHECK_MESSAGE(vs_ref <= tolerance, what_ref);
    CHECK_MESSAGE(vs_row <= tolerance, what_row);
  };

  run_case({50, 0, 31, 33, 64, 22}, 128, 128, bfloat16, false, 0.01);
  run_case({1, 95}, 64, 192, bfloat16, false, 0.01);
  run_case({50, 0, 31, 33, 64, 22}, 128, 128, bfloat16, true, 0.01);
  if (float16_available()) {
    run_case({40, 40, 40}, 192, 64, float16, false, 0.005);
  }
}
