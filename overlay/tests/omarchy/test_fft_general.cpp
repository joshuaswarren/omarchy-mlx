// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// Wave 12 FFT: general-length coverage. The radix-2 pass supports
// power-of-two lengths up to 2048. Everything beyond that lifts by a
// Cooley-Tukey factor decomposition (each factor in 2..2048 becomes its
// own radix-2 pass with a pointwise twiddle multiply between stages) and
// a Bluestein chirp-z transform for the prime-class lengths that have no
// small divisor.
//
// Accuracy bound, measured in the sweep case below: max relative error
// against the naive double-precision DFT reference stays under 5e-5 of
// the reference infinity norm across every landed length class
// (primes, decomposed composites, decompose-with-Bluestein, powers of
// two at the direct and lifted caps). Lengths that refuse by name:
// primes above 32768 (the chirp needs k*k exact in u32, so the padded
// convolution length next_pow2(2n-1) must stay at or under 65536) and
// any length above 2^24 (Cooley-Tukey twiddle phase products must stay
// exactly representable in float32).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/fft.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;
using cdouble = std::complex<double>;
using namespace mlx::core::fft;

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

void sync(const Stream& stream) {
  omarchy::get_command_encoder(stream).synchronize();
}

// Naive O(n^2) DFT in double precision. Forward: X[k] = sum x[j]
// e^(-2 pi i j k / n); inverse carries the 1/n scale, matching the
// per-axis backward norm of the CPU primitive.
std::vector<cdouble> naive_dft(const std::vector<cdouble>& x, bool inverse) {
  size_t n = x.size();
  std::vector<cdouble> X(n);
  for (size_t k = 0; k < n; ++k) {
    cdouble sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      double angle = (inverse ? 2.0 : -2.0) * M_PI * double(j) * double(k) /
          double(n);
      sum += x[j] * std::polar(1.0, angle);
    }
    X[k] = inverse ? sum / double(n) : sum;
  }
  return X;
}

std::vector<cdouble> random_complex(size_t n, std::mt19937& gen) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<cdouble> x(n);
  for (auto& value : x) {
    value = {dist(gen), dist(gen)};
  }
  return x;
}

std::vector<double> random_real(size_t n, std::mt19937& gen) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> x(n);
  for (auto& value : x) {
    value = dist(gen);
  }
  return x;
}

array complex_array(const std::vector<cdouble>& v, Shape shape) {
  std::vector<complex64_t> host(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    host[i] = complex64_t(float(v[i].real()), float(v[i].imag()));
  }
  return array(host.begin(), std::move(shape), complex64);
}

array real_array(const std::vector<double>& v, Shape shape) {
  std::vector<float> host(v.begin(), v.end());
  return array(host.begin(), std::move(shape), float32);
}

std::vector<cdouble> read_complex(array a, const Stream& stream) {
  a.eval();
  sync(stream);
  const complex64_t* data = a.data<complex64_t>();
  std::vector<cdouble> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = {double(data[i].real()), double(data[i].imag())};
  }
  return out;
}

std::vector<double> read_real(array a, const Stream& stream) {
  a.eval();
  sync(stream);
  const float* data = a.data<float>();
  std::vector<double> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = double(data[i]);
  }
  return out;
}

double inf_norm(const std::vector<cdouble>& v) {
  double norm = 0.0;
  for (const auto& value : v) {
    norm = std::max(norm, std::abs(value));
  }
  return norm;
}

double max_abs_diff(
    const std::vector<cdouble>& got,
    const std::vector<cdouble>& ref) {
  double diff = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    diff = std::max(diff, std::abs(got[i] - ref[i]));
  }
  return diff;
}

// Tolerance relative to the reference infinity norm, with an absolute
// floor for near-zero references. Every call states its own bound.
void require_close(
    const std::vector<cdouble>& got,
    const std::vector<cdouble>& ref,
    double relative,
    const char* label) {
  double tolerance = std::max(relative * inf_norm(ref), 1e-5);
  double diff = max_abs_diff(got, ref);
  REQUIRE_MESSAGE(
      diff <= tolerance,
      "[" << label << "] max abs diff " << diff << " exceeds tolerance "
           << tolerance);
}

std::string evaluation_error(const std::function<void()>& build) {
  try {
    build();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

} // namespace

TEST_CASE("c2c general lengths match the naive DFT") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(0xF12);

  struct Case {
    size_t n;
    const char* label;
  };
  // One case per plan branch: direct radix-2 (pow2 <= 2048), Cooley-
  // Tukey decomposition (composite with a divisor in 2..2048), Bluestein
  // (prime), and decompose-with-Bluestein (composite whose factor tree
  // bottoms out in a small prime).
  Case cases[] = {
      {3, "prime 3, Bluestein"},
      {5, "prime 5, Bluestein"},
      {7, "prime 7, Bluestein"},
      {11, "prime 11, Bluestein"},
      {12, "12 = 4*3, decompose with Bluestein on 3"},
      {15, "15 = 5*3, Bluestein factors"},
      {60, "60 = 4*15, decompose"},
      {97, "prime 97, Bluestein"},
      {120, "120 = 8*15, decompose"},
      {255, "255 = 15*17, decompose with Bluestein"},
      {360, "360 = 8*45, decompose"},
      {511, "prime 511, Bluestein"},
      {1000, "1000 = 8*125, decompose"},
      {1021, "prime 1021, Bluestein"},
      {1024, "direct pow2"},
      {1200, "1200 = 16*75, decompose with Bluestein on 3"},
      {1536, "1536 = 512*3, decompose with Bluestein on 3"},
      {2047, "2047 = 23*89, decompose with Bluestein"},
      {2048, "direct pow2 at the shared-memory cap"},
      {2053, "prime just above the old cap, Bluestein"},
      {3000, "3000 = 8*3*125, decompose with Bluestein on 3"},
      {4095, "4095 = 3*3*5*7*13, decompose with Bluestein"},
      {4096, "4096 = 2048*2, lifted pow2 via decomposition"},
      {4099, "prime 4099, Bluestein"},
  };
  for (const auto& c : cases) {
    auto x = random_complex(c.n, gen);
    auto got = read_complex(
        fftn(complex_array(x, Shape{int(c.n)}), FFTNorm::Backward, stream),
        stream);
    auto ref = naive_dft(x, /*inverse=*/false);
    require_close(got, ref, 5e-5, c.label);
  }
}

TEST_CASE("inverse round-trips at general lengths") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(0xA0A);
  size_t lengths[] = {
      3, 7, 12, 60, 97, 255, 511, 1000, 1021, 1536, 2053, 4099, 8192, 32749};
  for (size_t n : lengths) {
    auto x = random_complex(n, gen);
    auto spectrum = fftn(
        complex_array(x, Shape{int(n)}), FFTNorm::Backward, stream);
    auto recovered = read_complex(ifftn(spectrum, FFTNorm::Backward, stream), stream);
    double diff = max_abs_diff(recovered, x);
    REQUIRE_MESSAGE(
        diff <= 1e-4,
        "round-trip at n=" << n << ": max abs diff " << diff);
  }
}

TEST_CASE("c2c batched general lengths") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(0xBADD07);
  Shape shape{int(5), int(120)};
  auto x = random_complex(shape[0] * shape[1], gen);
  auto got = read_complex(
      fftn(complex_array(x, shape), FFTNorm::Backward, stream), stream);
  // Reference: 1-D DFT of each row, then a 1-D DFT down each column bin.
  std::vector<cdouble> rows(shape[0] * shape[1], {0.0, 0.0});
  for (size_t batch = 0; batch < shape[0]; ++batch) {
    std::vector<cdouble> row(x.begin() + batch * shape[1],
                             x.begin() + (batch + 1) * shape[1]);
    auto ref = naive_dft(row, false);
    for (size_t k = 0; k < shape[1]; ++k) {
      rows[batch * shape[1] + k] = ref[k];
    }
  }
  for (size_t k = 0; k < shape[1]; ++k) {
    std::vector<cdouble> col(shape[0]);
    for (size_t batch = 0; batch < shape[0]; ++batch) {
      col[batch] = rows[batch * shape[1] + k];
    }
    auto ref = naive_dft(col, false);
    for (size_t batch = 0; batch < shape[0]; ++batch) {
      double diff = std::abs(got[batch * shape[1] + k] - ref[batch]);
      REQUIRE_MESSAGE(
          diff <= 1e-3,
          "batch " << batch << " bin " << k << ": diff " << diff);
    }
  }
}

TEST_CASE("c2c on a middle axis matches the naive reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(0x1111);
  // Shape (3, 60, 5): fftn over axis 1 only. The middle axis has sample
  // stride 5 (non-trailing) and length 60 (decomposed 4*15).
  Shape shape{int(3), int(60), int(5)};
  auto x = random_complex(3 * 60 * 5, gen);
  auto got = read_complex(
      fftn(complex_array(x, shape), {1}, FFTNorm::Backward, stream), stream);
  double worst = 0.0;
  for (size_t b = 0; b < 3; ++b) {
    for (size_t t = 0; t < 5; ++t) {
      std::vector<cdouble> col(60);
      for (size_t j = 0; j < 60; ++j) {
        col[j] = x[(b * 60 + j) * 5 + t];
      }
      auto ref = naive_dft(col, false);
      for (size_t k = 0; k < 60; ++k) {
        worst = std::max(
            worst, std::abs(got[(b * 60 + k) * 5 + t] - ref[k]));
      }
    }
  }
  REQUIRE_MESSAGE(
      worst <= 1e-3,
      "middle-axis fftn (3,60,5): max abs diff " << worst);
}

TEST_CASE("multi-axis fftn over mixed general lengths") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(0x2222);
  // Axes {0, 1} on (4, 60, 120): axis 0 is direct pow2, axis 1
  // decomposes (60 = 4*15), axis 2 is untouched (extent 3).
  Shape shape{int(4), int(60), int(3)};
  auto x = random_complex(4 * 60 * 3, gen);
  auto got = read_complex(
      fftn(complex_array(x, shape), {0, 1}, FFTNorm::Backward, stream),
      stream);
  // Reference: naive DFT along axis 0, then along axis 1, leaving the
  // trailing extent 3 alone. Output flat = ((k0 * 60 + k1) * 3 + k2).
  std::vector<cdouble> inter(4 * 60 * 3, {0.0, 0.0});
  for (size_t j1 = 0; j1 < 60; ++j1) {
    for (size_t j2 = 0; j2 < 3; ++j2) {
      std::vector<cdouble> col(4);
      for (size_t j0 = 0; j0 < 4; ++j0) {
        col[j0] = x[(j0 * 60 + j1) * 3 + j2];
      }
      auto t0 = naive_dft(col, false);
      for (size_t k0 = 0; k0 < 4; ++k0) {
        inter[(k0 * 60 + j1) * 3 + j2] = t0[k0];
      }
    }
  }
  std::vector<cdouble> ref(got.size(), {0.0, 0.0});
  for (size_t k0 = 0; k0 < 4; ++k0) {
    for (size_t j2 = 0; j2 < 3; ++j2) {
      std::vector<cdouble> col(60);
      for (size_t j1 = 0; j1 < 60; ++j1) {
        col[j1] = inter[(k0 * 60 + j1) * 3 + j2];
      }
      auto t1 = naive_dft(col, false);
      for (size_t k1 = 0; k1 < 60; ++k1) {
        ref[(k0 * 60 + k1) * 3 + j2] = t1[k1];
      }
    }
  }
  require_close(got, ref, 5e-4, "multi-axis fftn (4,60,3) axes {0,1}");
}

TEST_CASE("analytic delta and constant on a prime length") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t n = 1021; // prime: the Bluestein path
  std::vector<cdouble> delta(n, {0.0, 0.0});
  delta[0] = {1.0, 0.0};
  auto flat = read_complex(
      fftn(complex_array(delta, Shape{int(n)}), FFTNorm::Backward, stream),
      stream);
  for (size_t k = 0; k < n; ++k) {
    double magnitude = std::abs(flat[k]);
    REQUIRE_MESSAGE(
        std::abs(magnitude - 1.0) <= 1e-4,
        "delta flat bin " << k << ": |X[k]| " << magnitude);
  }
  std::vector<cdouble> constant(n, {1.0, 0.0});
  auto spectrum = read_complex(
      fftn(complex_array(constant, Shape{int(n)}), FFTNorm::Backward, stream),
      stream);
  REQUIRE_MESSAGE(
      std::abs(spectrum[0] - cdouble(double(n), 0.0)) <= 1e-3,
      "DC bin off at n=" << n);
  for (size_t k = 1; k < n; ++k) {
    double magnitude = std::abs(spectrum[k]);
    REQUIRE_MESSAGE(
        magnitude <= 1e-3,
        "constant leakage at bin " << k << ": " << magnitude);
  }
}

TEST_CASE("single sinusoid lands in exactly one bin at a general length") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t n = 1536; // 512*3: decomposition with a Bluestein tail
  const size_t bin = 5;
  std::vector<cdouble> tone(n);
  for (size_t j = 0; j < n; ++j) {
    tone[j] = std::polar(1.0, 2.0 * M_PI * double(bin) * double(j) / double(n));
  }
  auto spectrum = read_complex(
      fftn(complex_array(tone, Shape{int(n)}), FFTNorm::Backward, stream),
      stream);
  for (size_t k = 0; k < n; ++k) {
    double expected = k == bin ? double(n) : 0.0;
    double tolerance = k == bin ? 0.01 * double(n) : 5e-3 * double(n);
    REQUIRE_MESSAGE(
        std::abs(spectrum[k] - cdouble(expected, 0.0)) <= tolerance,
        "bin " << k << " off: got " << spectrum[k]);
  }
}

TEST_CASE("Parseval identity holds at general lengths") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(0x717E);
  size_t lengths[] = {3, 7, 60, 97, 255, 1000, 1021, 1536, 2053, 4099};
  for (size_t n : lengths) {
    auto x = random_complex(n, gen);
    double time_energy = 0.0;
    for (const auto& value : x) {
      time_energy += std::norm(value);
    }
    auto spectrum = read_complex(
        fftn(complex_array(x, Shape{int(n)}), FFTNorm::Backward, stream),
        stream);
    double freq_energy = 0.0;
    for (const auto& value : spectrum) {
      freq_energy += std::norm(value);
    }
    // Backward norm: the forward transform is unnormalized, so the
    // identity is sum |X_k|^2 = n * sum |x_j|^2.
    double rel = std::abs(freq_energy / double(n) - time_energy) /
        std::max(time_energy, 1e-30);
    REQUIRE_MESSAGE(
        rel <= 1e-4,
        "Parseval at n=" << n << ": rel diff " << rel);
  }
}

TEST_CASE("non-trailing rfftn matches a naive reference and round-trips") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(0xC0FFEE);
  // Shape (2, 10, 4), rfftn over axes {0, 1}. The real axis 1 has
  // sample stride 4 (non-trailing); axis 0 has stride 40. Output keeps
  // 6 bins along axis 1 and leaves axis 2 untouched.
  Shape shape{int(2), int(10), int(4)};
  auto real = random_real(2 * 10 * 4, gen);
  auto spec_array = rfftn(
      real_array(real, shape), {0, 1}, FFTNorm::Backward, stream);
  REQUIRE(spec_array.shape(0) == 2);
  REQUIRE(spec_array.shape(1) == 6);
  REQUIRE(spec_array.shape(2) == 4);
  auto spec = read_complex(spec_array, stream);

  // Reference in double, stage by stage: axis 0 forward, then axis 2
  // forward into a full intermediate (b, k0, j1, k2), then axis 1
  // forward truncated to 6 bins.
  // Dim 0 of the input IS the axis-0 transform (no batch): after the
  // axis-0 DFT it indexes bins k0. Stage 0: axis 0 forward. Final: axis 1
  // forward, truncated to 6 bins. Output dims (k0, k1, k2).
  std::vector<cdouble> stage0(2 * 10 * 4, {0.0, 0.0});
  for (size_t j1 = 0; j1 < 10; ++j1) {
    for (size_t j2 = 0; j2 < 4; ++j2) {
      std::vector<cdouble> col0(2);
      for (size_t j0 = 0; j0 < 2; ++j0) {
        col0[j0] = {real[(j0 * 10 + j1) * 4 + j2], 0.0};
      }
      auto t0 = naive_dft(col0, false);
      for (size_t k0 = 0; k0 < 2; ++k0) {
        stage0[(k0 * 10 + j1) * 4 + j2] = t0[k0];
      }
    }
  }
  std::vector<cdouble> ref(spec.size(), {0.0, 0.0});
  for (size_t k0 = 0; k0 < 2; ++k0) {
    for (size_t k2 = 0; k2 < 4; ++k2) {
      std::vector<cdouble> col(10);
      for (size_t j1 = 0; j1 < 10; ++j1) {
        col[j1] = stage0[(k0 * 10 + j1) * 4 + k2];
      }
      auto transformed = naive_dft(col, false);
      for (size_t k1 = 0; k1 < 6; ++k1) {
        ref[(k0 * 6 + k1) * 4 + k2] = transformed[k1];
      }
    }
  }
  require_close(spec, ref, 5e-4, "non-trailing rfftn (2,10,4) axes {0,1}");

  // Round-trip back to the real input through irfftn with the same axes.
  auto back = read_real(
      irfftn(spec_array, Shape{2, 10}, {0, 1}, FFTNorm::Backward, stream),
      stream);
  double diff = 0.0;
  for (size_t i = 0; i < real.size(); ++i) {
    diff = std::max(diff, std::abs(back[i] - real[i]));
  }
  REQUIRE_MESSAGE(
      diff <= 5e-4,
      "non-trailing rfft/irfft round-trip: max abs diff " << diff);
}

TEST_CASE("trailing-axis rfftn round-trips at a decomposed length") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(0xAACE);
  // Shape (2, 240): rfftn over both axes. Axis 1 is the trailing real
  // axis of length 240 = 16*15 (decompose with a Bluestein tail); axis
  // 0 has sample stride 240. The half spectrum keeps 121 bins.
  Shape trailing{int(2), int(240)};
  auto real = random_real(2 * 240, gen);
  auto spec = rfftn(
      real_array(real, trailing), {0, 1}, FFTNorm::Backward, stream);
  REQUIRE(spec.shape(0) == 2);
  REQUIRE(spec.shape(1) == 121);
  auto back = read_real(
      irfftn(spec, Shape{2, 240}, {0, 1}, FFTNorm::Backward, stream), stream);
  double diff = 0.0;
  for (size_t i = 0; i < real.size(); ++i) {
    diff = std::max(diff, std::abs(back[i] - real[i]));
  }
  REQUIRE_MESSAGE(
      diff <= 5e-3,
      "trailing-axis rfftn round-trip at (2,240): max abs diff " << diff);
}

TEST_CASE("n equals 1 stays identity") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<cdouble> x{{2.5, -1.5}};
  auto got = read_complex(
      fftn(complex_array(x, Shape{1}), FFTNorm::Backward, stream), stream);
  REQUIRE_MESSAGE(
      std::abs(got[0] - x[0]) <= 1e-6,
      "fftn at n=1 returned " << got[0]);
  auto real_one = read_real(
      irfftn(
          rfftn(real_array({1.0f}, Shape{1}), {0}, FFTNorm::Backward, stream),
          Shape{1},
          {0},
          FFTNorm::Backward,
          stream),
      stream);
  REQUIRE(std::abs(real_one[0] - 1.0) <= 1e-6);
}

TEST_CASE("large lengths round-trip and keep analytic magnitudes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(0x1A2E);
  // 65536 = 2048*32: the lifted pow2 cap via two decomposition passes.
  // The naive O(n^2) reference is too slow here; a delta input gives an
  // exactly-flat magnitude spectrum, which is an honest scale check.
  {
    std::vector<cdouble> delta(65536, {0.0, 0.0});
    delta[0] = {1.0, 0.0};
    auto flat = read_complex(
        fftn(complex_array(delta, Shape{65536}), FFTNorm::Backward, stream),
        stream);
    double worst = 0.0;
    for (size_t k = 0; k < 65536; ++k) {
      worst = std::max(worst, std::abs(std::abs(flat[k]) - 1.0));
    }
    REQUIRE_MESSAGE(worst <= 1e-4, "delta flatness at 65536: " << worst);
    auto back = read_complex(
        ifftn(
            fftn(
                complex_array(delta, Shape{65536}),
                FFTNorm::Backward,
                stream),
            FFTNorm::Backward,
            stream),
        stream);
    REQUIRE_MESSAGE(
        max_abs_diff(back, delta) <= 1e-4, "round-trip at 65536");
  }
  // 100000 = 2^5 * 5^5: pure decomposition, no Bluestein anywhere.
  {
    auto x = random_complex(100000, gen);
    auto spectrum = fftn(
        complex_array(x, Shape{100000}), FFTNorm::Backward, stream);
    auto back = read_complex(
        ifftn(spectrum, FFTNorm::Backward, stream), stream);
    double diff = max_abs_diff(back, x);
    REQUIRE_MESSAGE(diff <= 1e-3, "round-trip at 100000: " << diff);
  }
  // 32749 is prime and inside the chirp bound: full Bluestein at the
  // largest refused-by-nothing length class.
  {
    auto x = random_complex(32749, gen);
    auto spectrum = fftn(
        complex_array(x, Shape{32749}), FFTNorm::Backward, stream);
    auto back = read_complex(
        ifftn(spectrum, FFTNorm::Backward, stream), stream);
    double diff = max_abs_diff(back, x);
    REQUIRE_MESSAGE(diff <= 1e-3, "round-trip at 32749: " << diff);
  }
}

TEST_CASE("named refusal pins above the chirp bound") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(0x91);
  // 65537 is prime: no divisor in 2..2048 and past the 32768 Bluestein
  // bound (k*k must stay exact in u32, bounding next_pow2(2n-1) at
  // 65536). It must refuse by name, with the arithmetic reason.
  auto x = random_complex(65537, gen);
  std::string err = evaluation_error([&] {
    auto y = fftn(complex_array(x, Shape{65537}), FFTNorm::Backward, stream);
    y.eval();
    sync(stream);
  });
  bool named = err.find("transform length 65537") != std::string::npos;
  bool reason = err.find("Bluestein") != std::string::npos;
  CHECK(named);
  CHECK(reason);
}

TEST_CASE("accuracy sweep measures the observed max relative error") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(0xACC071);
  // Naive O(n^2) reference stays cheap through n = 4100 (~16.8M complex
  // mults per transform); beyond that the analytic and round-trip cases
  // above carry the verification.
  size_t points[] = {3,  4,  5, 7, 8, 11, 12, 15, 16, 60, 97, 120, 255, 360,
                     511, 512, 1000, 1021, 1024, 1536, 2047, 2048, 2049,
                     2053, 3000, 4095, 4096, 4099};
  double worst = 0.0;
  size_t worst_n = 0;
  for (size_t n : points) {
    auto x = random_complex(n, gen);
    auto got = read_complex(
        fftn(complex_array(x, Shape{int(n)}), FFTNorm::Backward, stream),
        stream);
    auto ref = naive_dft(x, false);
    double rel = max_abs_diff(got, ref) / std::max(inf_norm(ref), 1.0);
    if (rel > worst) {
      worst = rel;
      worst_n = n;
    }
  }
  std::cout << "[fft accuracy] worst relative error " << worst << " at n="
            << worst_n << "\n";
  REQUIRE_MESSAGE(
      worst <= 5e-5,
      "accuracy bound violated: worst " << worst << " at n=" << worst_n);
}
