// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Wave 8 FFT coverage: complex-to-complex (fftn/ifftn), real-to-complex
// (rfftn) and complex-to-real (irfftn) over power-of-two lengths 2 to
// 2048, batched and on chosen axes of multi-dimensional arrays, plus the
// 2^24 length that crosses the 65,535 workgroup-per-dimension dispatch
// limit. Every value test compares against a naive O(n^2) DFT computed
// in double precision inside this file (an exact analytic reference for
// the 2^24 delta), plus analytic cases (delta, constant, single
// sinusoid), forward/inverse round-trips, Parseval's identity, and the
// n/2+1 output shape of the real variants.
//
// Tolerance: 2e-4 relative to the reference infinity norm, with a 1e-5
// absolute floor for near-zero references. The kernel computes twiddles on
// the fly in float32 (single turn around the circle per stage), so the
// error grows like eps * log2(n) with observed maxima around 1e-5 at
// n = 2048; 2e-4 leaves an order of magnitude of margin while still
// failing on a wrong bin, a missed 1/n scale, or a transposed axis, each
// of which lands orders of magnitude outside. Zero bins of analytic cases
// scale their tolerance with n for the same reason.

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

// Naive O(n^2) DFT in double precision. Forward: X[k] = sum_j x[j]
// e^(-2*pi*i*j*k/n). Inverse carries the 1/n scale, matching the backward
// norm the CPU primitive applies per axis.
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

// Naive 2-D DFT over both axes of a row-major (rows, cols) array: one
// 1-D pass along rows, then one along columns. Order is irrelevant in
// exact arithmetic, which is the point of an honest reference.
std::vector<cdouble> naive_dft2(
    const std::vector<cdouble>& in,
    size_t rows,
    size_t cols,
    bool inverse) {
  std::vector<cdouble> tmp(in.size());
  for (size_t r = 0; r < rows; ++r) {
    std::vector<cdouble> row(in.begin() + r * cols, in.begin() + (r + 1) * cols);
    auto transformed = naive_dft(row, inverse);
    for (size_t c = 0; c < cols; ++c) {
      tmp[r * cols + c] = transformed[c];
    }
  }
  std::vector<cdouble> out(in.size());
  for (size_t c = 0; c < cols; ++c) {
    std::vector<cdouble> col(rows);
    for (size_t r = 0; r < rows; ++r) {
      col[r] = tmp[r * cols + c];
    }
    auto transformed = naive_dft(col, inverse);
    for (size_t r = 0; r < rows; ++r) {
      out[r * cols + c] = transformed[r];
    }
  }
  return out;
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

double max_real_diff(
    const std::vector<double>& got,
    const std::vector<double>& ref) {
  double diff = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    diff = std::max(diff, std::abs(got[i] - ref[i]));
  }
  return diff;
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

void require_close(
    const std::vector<cdouble>& got,
    const std::vector<cdouble>& ref,
    double relative = 2e-4) {
  double tolerance = std::max(relative * inf_norm(ref), 1e-5);
  double diff = max_abs_diff(got, ref);
  REQUIRE_MESSAGE(
      diff <= tolerance,
      "max abs diff " << diff << " exceeds tolerance " << tolerance);
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

TEST_CASE("fftn delta and constant analytic cases") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t n = 16;

  // A delta transforms to a flat spectrum of ones.
  std::vector<cdouble> delta(n, {0.0, 0.0});
  delta[0] = {1.0, 0.0};
  auto flat = read_complex(fftn(complex_array(delta, Shape{int(n)}), FFTNorm::Backward, stream), stream);
  std::vector<cdouble> ones(n, {1.0, 0.0});
  require_close(flat, ones, 1e-6);

  // A constant transforms to a single DC bin of n.
  std::vector<cdouble> constant(n, {1.0, 0.0});
  auto spectrum = read_complex(fftn(complex_array(constant, Shape{int(n)}), FFTNorm::Backward, stream), stream);
  std::vector<cdouble> dc(n, {0.0, 0.0});
  dc[0] = {double(n), 0.0};
  require_close(spectrum, dc, 1e-6);
}

TEST_CASE("fftn single sinusoid lands in exactly one bin") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t n = 64;
  const size_t bin = 5;

  // exp(2*pi*i*k*j/n) transforms to n at bin k and zero elsewhere.
  std::vector<cdouble> tone(n);
  for (size_t j = 0; j < n; ++j) {
    tone[j] = std::polar(1.0, 2.0 * M_PI * double(bin) * double(j) / double(n));
  }
  auto spectrum = read_complex(fftn(complex_array(tone, Shape{int(n)}), FFTNorm::Backward, stream), stream);
  for (size_t k = 0; k < n; ++k) {
    double expected = k == bin ? double(n) : 0.0;
    double tolerance = k == bin ? 0.01 * double(n) : 5e-3 * double(n);
    REQUIRE_MESSAGE(
        std::abs(spectrum[k] - cdouble(expected, 0.0)) <= tolerance,
        "bin " << k << " off: got " << spectrum[k]);
  }

  // A real cosine splits evenly between bins k and n-k.
  std::vector<cdouble> cosine(n);
  for (size_t j = 0; j < n; ++j) {
    cosine[j] = {std::cos(2.0 * M_PI * double(bin) * double(j) / double(n)), 0.0};
  }
  auto real_spectrum = read_complex(fftn(complex_array(cosine, Shape{int(n)}), FFTNorm::Backward, stream), stream);
  REQUIRE_MESSAGE(std::abs(real_spectrum[bin] - cdouble(n / 2, 0.0)) <= 0.01 * n, "bin k wrong");
  REQUIRE_MESSAGE(
      std::abs(real_spectrum[n - bin] - cdouble(n / 2, 0.0)) <= 0.01 * n,
      "bin n-k wrong");
}

TEST_CASE("fftn matches naive DFT across length classes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(8);
  for (size_t n : {size_t(2), size_t(4), size_t(8), size_t(16), size_t(64),
                   size_t(256), size_t(1024), size_t(2048)}) {
    auto x = random_complex(n, gen);
    auto got = read_complex(fftn(complex_array(x, Shape{int(n)}), FFTNorm::Backward, stream), stream);
    auto ref = naive_dft(x, false);
    require_close(got, ref);
  }
}

TEST_CASE("ifftn matches naive inverse including the 1/n scale") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(19);
  for (size_t n : {size_t(2), size_t(8), size_t(64), size_t(256), size_t(2048)}) {
    auto X = random_complex(n, gen);
    auto got = read_complex(ifftn(complex_array(X, Shape{int(n)}), FFTNorm::Backward, stream), stream);
    auto ref = naive_dft(X, true);
    require_close(got, ref);
  }
}

TEST_CASE("fftn then ifftn round-trips to the input") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(27);
  for (size_t n : {size_t(8), size_t(128), size_t(2048)}) {
    auto x = random_complex(n, gen);
    auto fwd = fftn(complex_array(x, Shape{int(n)}), FFTNorm::Backward, stream);
    auto back = read_complex(ifftn(fwd, FFTNorm::Backward, stream), stream);
    require_close(back, x, 1e-4);
  }
}

TEST_CASE("Parseval identity: sum |x|^2 = (1/n) sum |X|^2") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(33);
  const size_t n = 64;
  auto x = random_complex(n, gen);
  auto X = read_complex(fftn(complex_array(x, Shape{int(n)}), FFTNorm::Backward, stream), stream);
  double time_energy = 0.0;
  for (const auto& value : x) {
    time_energy += std::norm(value);
  }
  double freq_energy = 0.0;
  for (const auto& value : X) {
    freq_energy += std::norm(value);
  }
  double expected = freq_energy / double(n);
  double tolerance = 1e-4 * std::max(1.0, expected);
  REQUIRE_MESSAGE(
      std::abs(time_energy - expected) <= tolerance,
      "Parseval violated: time " << time_energy << " vs freq/n " << expected);
}

TEST_CASE("multi-axis fftn over axes {0,1} matches naive 2-D reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(41);
  const size_t rows = 4;
  const size_t cols = 16;
  auto x = random_complex(rows * cols, gen);

  auto got = read_complex(
      fftn(complex_array(x, Shape{int(rows), int(cols)}), {0, 1}, FFTNorm::Backward, stream), stream);
  auto ref = naive_dft2(x, rows, cols, false);
  require_close(got, ref);

  // Axis order must not change the result.
  auto reordered = read_complex(
      fftn(complex_array(x, Shape{int(rows), int(cols)}), {1, 0}, FFTNorm::Backward, stream), stream);
  double order_diff = max_abs_diff(got, reordered);
  double order_tolerance = 1e-5 * std::max(1.0, inf_norm(ref));
  REQUIRE_MESSAGE(order_diff <= order_tolerance, "axis order changed the result");

  // Round-trip.
  auto back = read_complex(
      ifftn(fftn(complex_array(x, Shape{int(rows), int(cols)}), {0, 1}, FFTNorm::Backward, stream),
            {0, 1},
            FFTNorm::Backward,
            stream),
      stream);
  require_close(back, x, 1e-4);
}

TEST_CASE("fftn over a middle axis uses strided samples") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(47);
  const Shape shape{2, 8, 4};
  auto x = random_complex(2 * 8 * 4, gen);
  auto got = read_complex(fftn(complex_array(x, shape), {1}, FFTNorm::Backward, stream), stream);

  // Reference: for each fixed (i0, i2), naive DFT of the 8 samples down
  // the middle axis.
  for (size_t i0 = 0; i0 < 2; ++i0) {
    for (size_t i2 = 0; i2 < 4; ++i2) {
      std::vector<cdouble> column(8);
      for (size_t k = 0; k < 8; ++k) {
        column[k] = x[(i0 * 8 + k) * 4 + i2];
      }
      auto ref = naive_dft(column, false);
      for (size_t k = 0; k < 8; ++k) {
        double tolerance = std::max(2e-4 * std::abs(ref[k]), 1e-5);
        REQUIRE_MESSAGE(
            std::abs(got[(i0 * 8 + k) * 4 + i2] - ref[k]) <= tolerance,
            "middle-axis bin mismatch at (" << i0 << "," << k << "," << i2 << ")");
      }
    }
  }
}

TEST_CASE("batched last-axis fftn transforms each row independently") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(53);
  const size_t batch = 3;
  const size_t n = 16;
  auto x = random_complex(batch * n, gen);
  auto got = read_complex(fftn(complex_array(x, Shape{int(batch), int(n)}), {-1}, FFTNorm::Backward, stream), stream);
  for (size_t b = 0; b < batch; ++b) {
    std::vector<cdouble> row(x.begin() + b * n, x.begin() + (b + 1) * n);
    auto ref = naive_dft(row, false);
    std::vector<cdouble> got_row(got.begin() + b * n, got.begin() + (b + 1) * n);
    require_close(got_row, ref);
  }
}

TEST_CASE("rfftn returns n/2+1 bins matching the full DFT") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(59);
  for (size_t n : {size_t(8), size_t(64), size_t(256), size_t(2048)}) {
    auto x = random_real(n, gen);
    auto result = rfftn(real_array(x, Shape{int(n)}), FFTNorm::Backward, stream);
    REQUIRE_EQ(result.shape(0), int(n / 2 + 1));
    auto got = read_complex(result, stream);

    std::vector<cdouble> complex_input(n);
    for (size_t i = 0; i < n; ++i) {
      complex_input[i] = {x[i], 0.0};
    }
    auto full = naive_dft(complex_input, false);
    std::vector<cdouble> half(full.begin(), full.begin() + n / 2 + 1);
    require_close(got, half);

    // The dropped bins must equal the conjugates of the kept ones: the
    // half spectrum fully determines the transform of a real signal.
    for (size_t k = 1; k < n / 2; ++k) {
      REQUIRE_MESSAGE(
          std::abs(full[n - k] - std::conj(got[k])) <=
              std::max(2e-4 * std::abs(full[n - k]), 1e-5),
          "conjugate symmetry broken at bin " << k);
    }
  }
}

TEST_CASE("irfftn recovers the real signal from the half spectrum") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(61);
  for (size_t n : {size_t(8), size_t(64), size_t(256), size_t(2048)}) {
    auto x = random_real(n, gen);
    std::vector<cdouble> complex_input(n);
    for (size_t i = 0; i < n; ++i) {
      complex_input[i] = {x[i], 0.0};
    }
    auto full = naive_dft(complex_input, false);
    std::vector<cdouble> half(full.begin(), full.begin() + n / 2 + 1);

    auto got = read_real(irfftn(complex_array(half, Shape{int(n / 2 + 1)}), FFTNorm::Backward, stream), stream);
    REQUIRE_EQ(got.size(), n);
    double diff = max_real_diff(got, x);
    double tolerance = 1e-4 * std::max(1.0, inf_norm(half));
    REQUIRE_MESSAGE(diff <= tolerance, "irfft mismatch: " << diff);
  }
}

TEST_CASE("rfftn then irfftn round-trips a real signal") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(67);
  for (size_t n : {size_t(8), size_t(128), size_t(2048)}) {
    auto x = random_real(n, gen);
    auto fwd = rfftn(real_array(x, Shape{int(n)}), FFTNorm::Backward, stream);
    auto back = read_real(irfftn(fwd, FFTNorm::Backward, stream), stream);
    double diff = max_real_diff(back, x);
    REQUIRE_MESSAGE(
        diff <= 1e-4 * std::max(1.0, *std::max_element(x.begin(), x.end())),
        "round-trip mismatch: " << diff);
  }
}

TEST_CASE("rfftn over axes {0,1} keeps n/2+1 on the real axis") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(71);
  const size_t rows = 4;
  const size_t cols = 8;
  auto x = random_real(rows * cols, gen);

  auto result = rfftn(real_array(x, Shape{int(rows), int(cols)}), {0, 1}, FFTNorm::Backward, stream);
  REQUIRE_EQ(result.shape(0), int(rows));
  REQUIRE_EQ(result.shape(1), int(cols / 2 + 1));
  auto got = read_complex(result, stream);

  std::vector<cdouble> complex_input(rows * cols);
  for (size_t i = 0; i < rows * cols; ++i) {
    complex_input[i] = {x[i], 0.0};
  }
  auto full = naive_dft2(complex_input, rows, cols, false);
  std::vector<cdouble> ref;
  for (size_t r = 0; r < rows; ++r) {
    ref.insert(ref.end(), full.begin() + r * cols, full.begin() + r * cols + cols / 2 + 1);
  }
  require_close(got, ref);

  // And the inverse returns the original signal.
  auto back = read_real(irfftn(result, {0, 1}, FFTNorm::Backward, stream), stream);
  double diff = max_real_diff(back, x);
  REQUIRE_MESSAGE(diff <= 1e-4 * std::max(1.0, inf_norm(ref)), "2-D irfft mismatch");
}

TEST_CASE("fft2 and ifft2 route through the same primitive") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(73);
  const size_t rows = 4;
  const size_t cols = 8;
  auto x = random_complex(rows * cols, gen);
  auto input = complex_array(x, Shape{int(rows), int(cols)});
  auto via_fft2 = read_complex(fft2(input, {0, 1}, FFTNorm::Backward, stream), stream);
  auto via_fftn = read_complex(fftn(input, {0, 1}, FFTNorm::Backward, stream), stream);
  require_close(via_fft2, via_fftn, 1e-6);

  auto spec = fftn(input, {0, 1}, FFTNorm::Backward, stream);
  auto back = read_complex(ifft2(spec, {0, 1}, FFTNorm::Backward, stream), stream);
}

TEST_CASE("lengths beyond the general-length bounds keep the named error") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(79);

  // The wave-12 general-length work lifted the old power-of-two-only
  // gate: composites decompose into radix-2 passes (12 and 4096 now
  // transform correctly) and prime-class lengths embed in a Bluestein
  // chirp-z convolution (see omarchy_fft_general_tests for their value
  // tests). What still refuses, by name:
  //   - a length with no divisor in 2..2048 (a prime, in practice)
  //     above 32768: the Bluestein chirp reduces k*k mod 2n in u32,
  //     which is exact only while the padded convolution length
  //     next_pow2(2n-1) stays at or under 65536;
  //   - any length above 2^24, where Cooley-Tukey twiddle phase
  //     products would stop being exactly representable in float32.
  auto x65537 = random_complex(65537, gen);
  std::string prime_error = evaluation_error([&] {
    auto y = fftn(
        complex_array(x65537, Shape{65537}), FFTNorm::Backward, stream);
    y.eval();
    sync(stream);
  });
  CHECK(prime_error.find("[omarchy] FFT") != std::string::npos);
  CHECK(prime_error.find("transform length 65537") != std::string::npos);
  CHECK(prime_error.find("Bluestein") != std::string::npos);

  // 2053 * 2053 = 4214809: both factors are primes above 2048, so the
  // composite path has no divisor to split on and the length is far
  // past the chirp bound.
  auto x4214809 = random_complex(4214809, gen);
  std::string big_prime_error = evaluation_error([&] {
    auto y = fftn(
        complex_array(x4214809, Shape{4214809}),
        FFTNorm::Backward,
        stream);
    y.eval();
    sync(stream);
  });
  CHECK(
      big_prime_error.find("transform length 4214809") !=
      std::string::npos);
}

TEST_CASE("explicit n pads with zeros or slices the input") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(83);

  // n larger than the input: the op layer zero-pads before the FFT.
  // (The complex pad route goes through SliceUpdate, which has no
  // complex64 strided-copy path; the float32 rfft route exercises the
  // same padded dense input into the FFT primitive.)
  auto x4 = random_real(4, gen);
  auto padded = read_complex(
      rfftn(real_array(x4, Shape{4}), Shape{8}, {0}, FFTNorm::Backward, stream),
      stream);
  REQUIRE_EQ(padded.size(), size_t(5));
  std::vector<cdouble> padded_ref(8, {0.0, 0.0});
  for (size_t i = 0; i < 4; ++i) {
    padded_ref[i] = {x4[i], 0.0};
  }
  auto padded_full = naive_dft(padded_ref, false);
  std::vector<cdouble> padded_half(padded_full.begin(), padded_full.begin() + 5);
  require_close(padded, padded_half);

  // n smaller than the input: the op layer slices, the eval layer
  // materializes the slice, and the FFT sees a dense float32 buffer.
  auto x8 = random_real(8, gen);
  auto truncated = read_complex(
      rfftn(real_array(x8, Shape{8}), Shape{4}, {0}, FFTNorm::Backward, stream),
      stream);
  REQUIRE_EQ(truncated.size(), size_t(3));
  std::vector<cdouble> trunc_input(4);
  for (size_t i = 0; i < 4; ++i) {
    trunc_input[i] = {x8[i], 0.0};
  }
  auto trunc_full = naive_dft(trunc_input, false);
  std::vector<cdouble> trunc_ref(trunc_full.begin(), trunc_full.begin() + 3);
  require_close(truncated, trunc_ref);
}

TEST_CASE("rfft and irfft work on a non-trailing axis") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(89);

  // Wave 8 gated these by name; wave 12 lifted the gate (the half-
  // spectrum load/store paths index packed rows with step 1 while
  // everything else keeps the axis stride). Value-check the forward
  // direction against a two-axis naive reference, then the round-trip.
  auto x = random_real(4 * 8 * 4, gen);
  auto spectrum = read_complex(
      rfftn(real_array(x, Shape{4, 8, 4}), {0, 1}, FFTNorm::Backward, stream),
      stream);
  // Shape (4, 8, 4) with real axes {0, 1}: bins (4, 5, 4). Axis 2 is
  // untouched, so it keeps its full extent of 4.
  REQUIRE_EQ(spectrum.size(), size_t(4 * 5 * 4));

  // Reference: forward DFT along axis 0 (length 4, real promoted), then
  // along axis 1 (length 8), keeping bins 0..4. Output flat index for
  // dims (k0, k1, k2) is (k0 * 5 + k1) * 4 + k2.
  std::vector<cdouble> intermediate(4 * 8 * 4, {0.0, 0.0});
  for (size_t j1 = 0; j1 < 8; ++j1) {
    for (size_t j2 = 0; j2 < 4; ++j2) {
      std::vector<cdouble> col(4);
      for (size_t j0 = 0; j0 < 4; ++j0) {
        col[j0] = {x[(j0 * 8 + j1) * 4 + j2], 0.0};
      }
      auto t0 = naive_dft(col, false);
      for (size_t k0 = 0; k0 < 4; ++k0) {
        intermediate[((j1 * 4 + k0) * 4) + j2] = t0[k0];
      }
    }
  }
  std::vector<cdouble> ref(spectrum.size(), {0.0, 0.0});
  for (size_t k0 = 0; k0 < 4; ++k0) {
    for (size_t j2 = 0; j2 < 4; ++j2) {
      std::vector<cdouble> col(8);
      for (size_t j1 = 0; j1 < 8; ++j1) {
        col[j1] = intermediate[((j1 * 4 + k0) * 4) + j2];
      }
      auto t1 = naive_dft(col, false);
      for (size_t k1 = 0; k1 < 5; ++k1) {
        ref[((k0 * 5 + k1) * 4) + j2] = t1[k1];
      }
    }
  }
  require_close(spectrum, ref, 5e-4);

  // Round-trip: irfftn over the same axes restores the real input.
  auto spec_array = rfftn(
      real_array(x, Shape{4, 8, 4}), {0, 1}, FFTNorm::Backward, stream);
  auto back = read_real(
      irfftn(spec_array, Shape{4, 8}, {0, 1}, FFTNorm::Backward, stream),
      stream);
  double diff = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    diff = std::max(diff, std::abs(back[i] - double(x[i])));
  }
  REQUIRE_MESSAGE(
      diff <= 1e-3,
      "non-trailing rfft/irfft round-trip: max abs diff " << diff);
}

TEST_CASE("fftn at 2^24 exercises the second dispatch dimension") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // One elementwise fft stage covers n elements in 256-thread workgroups;
  // n = 2^24 needs 65,536 of them, one past the guaranteed 65,535
  // x-dimension limit, so the host spills the last group into a second
  // dispatch dimension. A stage that reads only gl_GlobalInvocationID.x
  // repeats the first 65,535 * 256 element addresses and never writes the
  // trailing 256. A delta at element n - 1 (inside that trailing window)
  // transforms to the flat unit-magnitude spectrum X[k] = e^(2*pi*i*k/n),
  // so both the unwritten output tail and the stage temporaries the
  // downstream passes inherit land orders of magnitude outside float32
  // round-off. With the bug, fresh page-zeroed buffers make every bin 0.
  const uint64_t n = 1ull << 24;
  const uint64_t delta_at = n - 1;

  std::vector<complex64_t> host(n, complex64_t(0.0f, 0.0f));
  host[delta_at] = complex64_t(1.0f, 0.0f);
  array input = array(host.begin(), Shape{int(n)}, complex64);
  array output = fftn(input, FFTNorm::Backward, stream);
  output.eval();
  sync(stream);
  const complex64_t* bins = output.data<complex64_t>();

  auto reference = [&](uint64_t k) {
    uint64_t phase = (k * (delta_at % n)) % n;
    double angle = -2.0 * M_PI * double(phase) / double(n);
    return cdouble(std::cos(angle), std::sin(angle));
  };
  // Same 2e-4 relative budget as the value cases: the delta keeps every
  // intermediate magnitude at one, so round-off stays around 1e-5 no
  // matter how large n grows.
  const double tolerance = 2e-4;
  double worst = 0.0;
  uint64_t worst_at = 0;
  auto check = [&](uint64_t k) {
    double diff = std::abs(
        cdouble(bins[k].real(), bins[k].imag()) - reference(k));
    if (diff > worst) {
      worst = diff;
      worst_at = k;
    }
  };
  // Dense over the trailing window a broken stage leaves unwritten.
  for (uint64_t k = n - 2048; k < n; ++k) {
    check(k);
  }
  // Decimated over the rest of the spectrum, which inherits the stage's
  // missing elements through the downstream passes.
  for (uint64_t k = 0; k < n; k += 4096) {
    check(k);
  }
  REQUIRE_MESSAGE(
      worst <= tolerance,
      "2^24 delta: max abs diff " << worst << " at bin " << worst_at
                                  << " exceeds tolerance " << tolerance);
}

TEST_CASE("rfft and irfft at 2^24 real elements cross the extract limit") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The irfft extract kernel takes the .x of the full-length inverse
  // transform one element per thread; an output of n = 2^24 real elements
  // needs 65,536 workgroups of 256, one past the guaranteed 65,535
  // x-dimension limit, so the host refused irfft outputs above
  // 65,535 * 256 by name before any dispatch (the 2026-09-08 core requal's
  // real-fft-extract-output-size-cap cluster). A kernel reading only
  // gl_GlobalInvocationID.x would also leave the trailing 256 outputs
  // unwritten. A real delta at element n - 1 (inside that trailing window)
  // has the exact half spectrum X[k] = e^(-2*pi*i*(n-1)*k/n), k = 0..n/2,
  // and the delta keeps every intermediate magnitude at one, so round-off
  // stays near 1e-5 no matter how large n grows.
  const uint64_t n = 1ull << 24;
  const uint64_t delta_at = n - 1;
  const uint64_t bins = n / 2 + 1;

  auto phase_bin = [&](uint64_t k) {
    uint64_t phase = (k * delta_at) % n;
    double angle = -2.0 * M_PI * double(phase) / double(n);
    return cdouble(std::cos(angle), std::sin(angle));
  };

  // Forward: the real delta's half spectrum is pure phase, checked
  // densely over the top bins and decimated below.
  std::vector<float> host(n, 0.0f);
  host[delta_at] = 1.0f;
  array input = array(host.begin(), Shape{1, int(n)}, float32);
  auto spectrum = read_complex(
      rfftn(input, {1}, FFTNorm::Backward, stream), stream);
  REQUIRE_EQ(spectrum.size(), size_t(bins));
  const double tolerance = 2e-4;
  double worst = 0.0;
  uint64_t worst_at = 0;
  auto check_bin = [&](uint64_t k) {
    double diff = std::abs(spectrum[k] - phase_bin(k));
    if (diff > worst) {
      worst = diff;
      worst_at = k;
    }
  };
  for (uint64_t k = bins - 2048; k < bins; ++k) {
    check_bin(k);
  }
  for (uint64_t k = 0; k < bins; k += 4096) {
    check_bin(k);
  }
  REQUIRE_MESSAGE(
      worst <= tolerance,
      "2^24 rfft delta: max abs diff " << worst << " at bin " << worst_at
                                       << " exceeds tolerance " << tolerance);

  // Inverse: the same half spectrum (built on the host) must synthesize
  // the real delta back. This is the dispatch the extract refusal killed;
  // with the refusal gone, a broken extract still shows as the missing
  // delta (diff 1.0) because page-zeroed buffers read as zero.
  std::vector<complex64_t> spec_host(bins);
  for (uint64_t k = 0; k < bins; ++k) {
    cdouble value = phase_bin(k);
    spec_host[k] = complex64_t(float(value.real()), float(value.imag()));
  }
  array spec = array(spec_host.begin(), Shape{1, int(bins)}, complex64);
  auto back = read_real(
      irfftn(spec, Shape{int(n)}, {1}, FFTNorm::Backward, stream), stream);
  REQUIRE_EQ(back.size(), size_t(n));
  worst = 0.0;
  worst_at = 0;
  auto check_sample = [&](uint64_t j) {
    double ref = j == delta_at ? 1.0 : 0.0;
    double diff = std::abs(back[j] - ref);
    if (diff > worst) {
      worst = diff;
      worst_at = j;
    }
  };
  // Dense over the trailing window past 65,535 * 256 that a broken
  // extract leaves unwritten.
  for (uint64_t j = 65535ull * 256ull; j < n; ++j) {
    check_sample(j);
  }
  // Decimated over the rest.
  for (uint64_t j = 0; j < 65535ull * 256ull; j += 4096) {
    check_sample(j);
  }
  REQUIRE_MESSAGE(
      worst <= tolerance,
      "2^24 irfft delta: max abs diff " << worst << " at element "
                                        << worst_at << " exceeds tolerance "
                                        << tolerance);
}
