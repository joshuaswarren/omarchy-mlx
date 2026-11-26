// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Complex64Transport coverage: complex64 transport (allocation, scalar
// fill, same-dtype strided copies, the zero-copy view primitives),
// dtype casts in both directions against the upstream static_cast
// rules, Conjugate/Real/Imag on complex input, complex add/subtract/
// multiply/divide/negate, and the pad/concatenate/FFT paths that
// previously refused complex64 outright.
//
// Value policy: transport and casts compare bit-exact float32
// components against host references (a wrong value cannot hide, and
// every path here is pure data movement or an exact promotion).
// Multiply and divide compare against std::complex<double> references
// computed in this file at 1e-5 relative tolerance with a 1e-6
// absolute floor: float32 arithmetic on magnitudes <= 4 carries
// ~1e-7 relative error, so 1e-5 fails on a swapped component, a
// dropped conjugate, or a real/imag transposition, each of which
// lands orders of magnitude outside.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <complex>
#include <limits>
#include <cstdint>
#include <iostream>
#include <random>
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

std::vector<cdouble> random_complex(size_t n, std::mt19937& gen) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<cdouble> x(n);
  for (auto& value : x) {
    value = {dist(gen), dist(gen)};
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
  auto dense = contiguous(a);
  dense.eval();
  sync(stream);
  const complex64_t* data = dense.data<complex64_t>();
  std::vector<cdouble> out(dense.size());
  for (size_t i = 0; i < dense.size(); ++i) {
    out[i] = {double(data[i].real()), double(data[i].imag())};
  }
  return out;
}

std::vector<double> read_real(array a, const Stream& stream) {
  auto dense = contiguous(a);
  dense.eval();
  sync(stream);
  const float* data = dense.data<float>();
  return std::vector<double>(data, data + dense.size());
}

template <typename T>
std::vector<T> read_values(array a, const Stream& stream) {
  auto dense = contiguous(a);
  dense.eval();
  sync(stream);
  const T* data = dense.data<T>();
  return std::vector<T>(data, data + dense.size());
}

// Exact component comparison at float32 precision: transport carries
// float32 words unchanged, so the reference quantized through float32
// must equal the readback bit for bit.
void check_exact(
    const std::vector<cdouble>& got,
    const std::vector<cdouble>& ref) {
  REQUIRE_EQ(got.size(), ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    float want_re = float(ref[i].real());
    float want_im = float(ref[i].imag());
    INFO("index ", i, " got (", got[i].real(), ", ", got[i].imag(),
         ") ref (", want_re, ", ", want_im, ")");
    CHECK_EQ(got[i].real(), want_re);
    CHECK_EQ(got[i].imag(), want_im);
  }
}

void check_exact_real(
    const std::vector<double>& got,
    const std::vector<double>& ref) {
  REQUIRE_EQ(got.size(), ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    float want = float(ref[i]);
    INFO("index ", i, " got ", got[i], " ref ", want);
    CHECK_EQ(got[i], want);
  }
}

void check_close(
    const std::vector<cdouble>& got,
    const std::vector<cdouble>& ref,
    double tol) {
  REQUIRE_EQ(got.size(), ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    double dr = std::abs(got[i].real() - ref[i].real());
    double di = std::abs(got[i].imag() - ref[i].imag());
    double scale_r = std::max(1.0, std::abs(ref[i].real()));
    double scale_i = std::max(1.0, std::abs(ref[i].imag()));
    INFO("index ", i, " got (", got[i].real(), ", ", got[i].imag(),
         ") ref (", ref[i].real(), ", ", ref[i].imag(), ")");
    CHECK(dr <= tol * scale_r + 1e-6);
    CHECK(di <= tol * scale_i + 1e-6);
  }
}

} // namespace

TEST_CASE("complex64 host construction and allocation round trip") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(1);
  auto ref = random_complex(37, gen);
  auto out = read_complex(complex_array(ref, Shape{37}), stream);
  check_exact(out, ref);
}

TEST_CASE("astype float32 to complex64 promotes with zero imaginary") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::vector<double> ref{0.5, -1.25, 3.0, 0.0, -7.75};
  auto src = real_array(ref, Shape{5});
  auto dst = astype(src, complex64);
  CHECK_EQ(dst.dtype(), complex64);
  std::vector<cdouble> expect(ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    expect[i] = {ref[i], 0.0};
  }
  check_exact(read_complex(dst, stream), expect);
}

TEST_CASE("astype complex64 to float32 keeps the real part") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(2);
  auto ref = random_complex(19, gen);
  auto src = complex_array(ref, Shape{19});
  auto dst = astype(src, float32);
  CHECK_EQ(dst.dtype(), float32);
  // Upstream complex64_t::operator float() returns real().
  std::vector<double> expect(ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    expect[i] = ref[i].real();
  }
  check_exact_real(read_real(dst, stream), expect);
}

TEST_CASE("astype integer and bool sources promote to (x, 0)") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::vector<int32_t> int_host{3, -4, 0};
  array ints(int_host.begin(), Shape{3});
  std::vector<cdouble> int_expect{{3.0, 0.0}, {-4.0, 0.0}, {0.0, 0.0}};
  check_exact(read_complex(astype(ints, complex64), stream), int_expect);

  std::vector<uint8_t> bool_host{1, 0, 1};
  array bools = array(bool_host.begin(), Shape{3}, bool_);
  std::vector<cdouble> bool_expect{{1.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
  check_exact(read_complex(astype(bools, complex64), stream), bool_expect);
}

TEST_CASE("strided reshape materializes complex64 values") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // Transpose first, then reshape: the reshape cannot be a view, so it
  // routes through the strided copy engine's complex64 path.
  std::mt19937 gen(3);
  auto ref = random_complex(12, gen);
  auto src = complex_array(ref, Shape{3, 4});
  auto transposed = transpose(src);
  auto flat = reshape(transposed, Shape{12});
  std::vector<cdouble> expect(12);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      expect[c * 3 + r] = ref[r * 4 + c];
    }
  }
  check_exact(read_complex(flat, stream), expect);
}

TEST_CASE("transposed view reshaped to dense agrees with the reference") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(4);
  auto ref = random_complex(20, gen);
  auto src = complex_array(ref, Shape{4, 5});
  auto view = transpose(src);
  CHECK_EQ(view.shape()[0], 5);
  CHECK_EQ(view.shape()[1], 4);
  auto dense = reshape(view, Shape{20});
  std::vector<cdouble> expect(20);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 5; ++c) {
      expect[c * 4 + r] = ref[r * 5 + c];
    }
  }
  check_exact(read_complex(dense, stream), expect);
}

TEST_CASE("strided slice of a complex array materializes exactly") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(5);
  auto ref = random_complex(16, gen);
  auto src = complex_array(ref, Shape{16});
  auto sliced = slice(src, Shape{1}, Shape{16}, Shape{3});
  std::vector<cdouble> expect;
  for (int i = 1; i < 16; i += 3) {
    expect.push_back(ref[i]);
  }
  check_exact(read_complex(sliced, stream), expect);
}

TEST_CASE("broadcast view reshaped to dense materializes every element") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(6);
  auto row = random_complex(4, gen);
  auto src = complex_array(row, Shape{1, 4});
  auto wide = broadcast_to(src, Shape{3, 4});
  auto dense = reshape(wide, Shape{12});
  std::vector<cdouble> expect;
  for (int r = 0; r < 3; ++r) {
    expect.insert(expect.end(), row.begin(), row.end());
  }
  check_exact(read_complex(dense, stream), expect);
}

TEST_CASE("full and zeros fill complex64 scalars") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  auto filled = full(Shape{2, 3}, complex64_t(1.5f, -2.5f), complex64);
  std::vector<cdouble> fill_expect(6, {1.5, -2.5});
  check_exact(read_complex(filled, stream), fill_expect);

  auto zeroed = zeros(Shape{4}, complex64);
  std::vector<cdouble> zero_expect(4, {0.0, 0.0});
  check_exact(read_complex(zeroed, stream), zero_expect);
}

TEST_CASE("conjugate negates the imaginary component") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(7);
  auto ref = random_complex(15, gen);
  auto src = complex_array(ref, Shape{15});
  auto conj = conjugate(src);
  std::vector<cdouble> expect(ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    expect[i] = {ref[i].real(), -ref[i].imag()};
  }
  check_exact(read_complex(conj, stream), expect);
}

TEST_CASE("real and imag extract their components") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(8);
  auto ref = random_complex(11, gen);
  auto src = complex_array(ref, Shape{11});
  std::vector<double> real_expect(ref.size());
  std::vector<double> imag_expect(ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    real_expect[i] = ref[i].real();
    imag_expect[i] = ref[i].imag();
  }
  check_exact_real(read_real(real(src), stream), real_expect);
  check_exact_real(read_real(imag(src), stream), imag_expect);
}

TEST_CASE("complex add subtract and negate are componentwise") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(9);
  auto a_ref = random_complex(9, gen);
  auto b_ref = random_complex(9, gen);
  auto a = complex_array(a_ref, Shape{9});
  auto b = complex_array(b_ref, Shape{9});
  std::vector<cdouble> sum_expect(9), diff_expect(9), neg_expect(9);
  for (size_t i = 0; i < 9; ++i) {
    // The device computes in float32 on float32-quantized inputs, so
    // the reference must run the same single-precision arithmetic:
    // rounding a double-precision sum once more into float32 can sit
    // one ulp away from the correctly rounded float32 result.
    float ar = float(a_ref[i].real());
    float ai = float(a_ref[i].imag());
    float br = float(b_ref[i].real());
    float bi = float(b_ref[i].imag());
    sum_expect[i] = {double(ar + br), double(ai + bi)};
    diff_expect[i] = {double(ar - br), double(ai - bi)};
    neg_expect[i] = {double(-ar), double(-ai)};
  }
  check_exact(read_complex(a + b, stream), sum_expect);
  check_exact(read_complex(a - b, stream), diff_expect);
  check_exact(read_complex(negative(a), stream), neg_expect);
}

TEST_CASE("complex multiply matches a double reference") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(10);
  auto a_ref = random_complex(24, gen);
  auto b_ref = random_complex(24, gen);
  auto a = complex_array(a_ref, Shape{24});
  auto b = complex_array(b_ref, Shape{24});
  std::vector<cdouble> expect(24);
  for (size_t i = 0; i < 24; ++i) {
    expect[i] = a_ref[i] * b_ref[i];
  }
  check_close(read_complex(a * b, stream), expect, 1e-5);
}

TEST_CASE("complex divide matches a double reference") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(11);
  auto a_ref = random_complex(24, gen);
  auto b_ref = random_complex(24, gen);
  // Keep the reference magnitudes moderate so the float32 reciprocal
  // is well conditioned.
  for (auto& value : b_ref) {
    value = value * 0.5 + cdouble(0.6, -0.4);
  }
  auto a = complex_array(a_ref, Shape{24});
  auto b = complex_array(b_ref, Shape{24});
  std::vector<cdouble> expect(24);
  for (size_t i = 0; i < 24; ++i) {
    expect[i] = a_ref[i] / b_ref[i];
  }
  check_close(read_complex(a / b, stream), expect, 1e-5);
}

TEST_CASE("complex pad and concatenate transport") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(12);
  auto ref = random_complex(6, gen);
  auto src = complex_array(ref, Shape{6});
  auto padded = pad(src, std::vector<int>{0}, Shape{2}, Shape{1});
  std::vector<cdouble> pad_expect(9, {0.0, 0.0});
  for (size_t i = 0; i < ref.size(); ++i) {
    pad_expect[i + 2] = ref[i];
  }
  check_exact(read_complex(padded, stream), pad_expect);

  auto joined = concatenate({src, src});
  std::vector<cdouble> concat_expect(ref);
  concat_expect.insert(concat_expect.end(), ref.begin(), ref.end());
  check_exact(read_complex(joined, stream), concat_expect);
}

TEST_CASE("fft accepts a non-contiguous complex64 input") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // A (1, 8) row transposed to (8, 1) then reshaped to (8) forces the
  // strided complex64 materialization in front of the FFT pass.
  std::mt19937 gen(13);
  auto ref = random_complex(8, gen);
  auto src = complex_array(ref, Shape{1, 8});
  auto column = transpose(src);
  auto dense = reshape(column, Shape{8});
  auto spectrum = fftn(dense);
  std::vector<cdouble> expect(8, {0.0, 0.0});
  for (size_t k = 0; k < 8; ++k) {
    for (size_t j = 0; j < 8; ++j) {
      double angle = -2.0 * M_PI * double(j) * double(k) / 8.0;
      expect[k] += ref[j] * std::polar(1.0, angle);
    }
  }
  check_close(read_complex(spectrum, stream), expect, 1e-4);
}

TEST_CASE("complex square, logaddexp, and equality match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Square = z*z through the complex multiply kernel.
  array z({complex64_t{1, 2}, complex64_t{-1, 0.5f}, complex64_t{0, -3}}, {3});
  auto sq = read_complex(square(z, stream), stream);
  CHECK(sq[0].real() == doctest::Approx(-3.0));
  CHECK(sq[0].imag() == doctest::Approx(4.0));
  CHECK(sq[1].real() == doctest::Approx(0.75));
  CHECK(sq[1].imag() == doctest::Approx(-1.0));
  CHECK(sq[2].real() == doctest::Approx(-9.0));
  CHECK(sq[2].imag() == doctest::Approx(0.0).epsilon(1e-6));

  // Equality: componentwise exact, including equal_nan=false defaults.
  array e1({complex64_t{1, 2}, complex64_t{0, 0}}, {2});
  array e2({complex64_t{1, 2}, complex64_t{0, 1}}, {2});
  CHECK(all(equal(e1, e2, stream), stream).item<bool>() == false);
  CHECK(any(not_equal(e1, e2, stream), stream).item<bool>() == true);
  CHECK(all(not_equal(e1, e2, stream), stream).item<bool>() == false);
  CHECK(all(equal(e1, e1, stream), stream).item<bool>() == true);

  // LogAddExp on a vector pair against the host formula
  // max + log(1 + exp(min - max)) in complex arithmetic.
  array lv({complex64_t{1, 1}, complex64_t{2, 0}}, {2});
  array rv({complex64_t{1, 1}, complex64_t{1, 1}}, {2});
  auto la = read_complex(logaddexp(lv, rv, stream), stream);
  std::complex<double> expect0 =
      std::complex<double>(1, 1) + std::log(std::complex<double>(2, 0));
  std::complex<double> expect1 = std::complex<double>(2, 0) +
      std::log(1.0 + std::exp(std::complex<double>(-1, 1)));
  CHECK(la[0].real() == doctest::Approx(expect0.real()).epsilon(1e-5));
  CHECK(la[0].imag() == doctest::Approx(expect0.imag()).epsilon(1e-5));
  CHECK(la[1].real() == doctest::Approx(expect1.real()).epsilon(1e-5));
  CHECK(la[1].imag() == doctest::Approx(expect1.imag()).epsilon(1e-5));
}

TEST_CASE("fft of a small real signal is exact at quarter-turn twiddles") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // Upstream's fft tests compare with array_equal, so the radix-2
  // butterflies must produce exact results where the twiddles are
  // exact: cos(pi/2) evaluated directly is -4.4e-8, which once left
  // y[1].real at -1.99999988.
  array x({0.0f, 1.0f, 2.0f, 3.0f});
  array y = fft::fft(x, -1, FFTNorm::Backward, stream);
  std::vector<cdouble> expect{{6, 0}, {-2, 2}, {-2, 0}, {-2, -2}};
  check_exact(read_complex(y, stream), expect);
  array back = fft::ifft(y, -1, FFTNorm::Backward, stream);
  std::vector<cdouble> expect_back{{0, 0}, {1, 0}, {2, 0}, {3, 0}};
  check_exact(read_complex(back, stream), expect_back);
}

TEST_CASE("complex abs matches host reference including overflow-scale") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // Normal range: |3 + 4i| = 5, |0| = 0, pure imaginary, negatives.
  array z({complex64_t{3, 4}, complex64_t{0, 0}, complex64_t{0, -2},
           complex64_t{-1, -1}},
          {4});
  auto got = read_real(abs(z, stream), stream);
  CHECK(got[0] == doctest::Approx(5.0).epsilon(1e-6));
  CHECK(got[1] == 0.0);
  CHECK(got[2] == doctest::Approx(2.0).epsilon(1e-6));
  CHECK(got[3] == doctest::Approx(std::sqrt(2.0)).epsilon(1e-6));
  // Large magnitude: sqrt(re^2 + im^2) overflows float32 for both
  // components above ~1.8e19, so the kernel must scale before squaring.
  // std::abs over double is the reference; the naive form would return
  // inf here.
  std::vector<cdouble> big{{3e19, 4e19}, {1e20, 1e20}, {5e18, -1.2e19}};
  std::vector<complex64_t> host(big.size());
  for (size_t i = 0; i < big.size(); ++i) {
    host[i] = complex64_t(float(big[i].real()), float(big[i].imag()));
  }
  array zb(host.begin(), Shape{3}, complex64);
  auto got_big = read_real(abs(zb, stream), stream);
  for (size_t i = 0; i < big.size(); ++i) {
    double want = std::abs(big[i]);
    INFO("big index ", i, " got ", got_big[i], " want ", want);
    CHECK(got_big[i] == doctest::Approx(want).epsilon(1e-6));
    CHECK(std::isfinite(got_big[i]));
  }
}

TEST_CASE("complex extract and abs agree on broadcast transposed and offset views") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(14);
  auto ref = random_complex(12, gen);

  // Checks Real/Imag/Conjugate bit-exactly and Abs at float32 hypot
  // precision against host references computed from the view's
  // underlying dense values.
  auto check_view = [&](const array& view, const std::vector<cdouble>& dense) {
    std::vector<double> real_expect(dense.size());
    std::vector<double> imag_expect(dense.size());
    std::vector<cdouble> conj_expect(dense.size());
    for (size_t i = 0; i < dense.size(); ++i) {
      real_expect[i] = dense[i].real();
      imag_expect[i] = dense[i].imag();
      conj_expect[i] = {dense[i].real(), -dense[i].imag()};
    }
    check_exact_real(read_real(real(view, stream), stream), real_expect);
    check_exact_real(read_real(imag(view, stream), stream), imag_expect);
    check_exact(read_complex(conjugate(view, stream), stream), conj_expect);
    auto got_abs = read_real(abs(view, stream), stream);
    REQUIRE_EQ(got_abs.size(), dense.size());
    for (size_t i = 0; i < dense.size(); ++i) {
      double want = std::abs(dense[i]);
      INFO("abs index ", i, " got ", got_abs[i], " want ", want);
      CHECK(got_abs[i] == doctest::Approx(want).epsilon(1e-5));
    }
  };

  // (a) Broadcast views: a (1, 4) row fanned to (3, 4) and a scalar
  // fanned to (5). The views keep stride-0 axes over a one-row (or
  // one-item) buffer, so a flat dense read lands outside the row.
  auto row = complex_array({ref[0], ref[1], ref[2], ref[3]}, Shape{1, 4});
  auto wide = broadcast_to(row, Shape{3, 4});
  std::vector<cdouble> wide_expect;
  for (int r = 0; r < 3; ++r) {
    wide_expect.insert(wide_expect.end(), ref.begin(), ref.begin() + 4);
  }
  check_view(wide, wide_expect);

  auto scalar = complex_array({ref[4]}, Shape{});
  auto fan = broadcast_to(scalar, Shape{5});
  check_view(fan, std::vector<cdouble>(5, ref[4]));

  // (b) A transposed 2-D view: storage order is the transpose of the
  // logical order, so a flat dense read permutes the elements.
  auto src = complex_array(
      {ref[0],
       ref[1],
       ref[2],
       ref[3],
       ref[4],
       ref[5],
       ref[6],
       ref[7],
       ref[8],
       ref[9],
       ref[10],
       ref[11]},
      Shape{4, 3});
  auto transposed = transpose(src);
  std::vector<cdouble> transposed_expect(12);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      transposed_expect[c * 4 + r] = ref[r * 3 + c];
    }
  }
  check_view(transposed, transposed_expect);

  // (c) A sliced view with a nonzero offset stays row-contiguous and
  // must keep reading from the offset rather than the buffer base.
  auto sliced = slice(complex_array(ref, Shape{12}), Shape{3}, Shape{11});
  std::vector<cdouble> sliced_expect(ref.begin() + 3, ref.begin() + 11);
  check_view(sliced, sliced_expect);
}

TEST_CASE("complex select copies whole elements by condition") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  array t({complex64_t{1, 2}, complex64_t{3, 4}, complex64_t{5, 6}}, {3});
  array f({complex64_t{-1, -2}, complex64_t{-3, -4}, complex64_t{-5, -6}}, {3});
  // Condition bytes: 1, 0, 1 - exercises both lanes and the odd count.
  array cond({true, false, true});
  auto got = read_complex(where(cond, t, f, stream), stream);
  check_exact(got, {{1, 2}, {-3, -4}, {5, 6}});
  // Scalar false operand rides the broadcast index path.
  auto scalar = read_complex(where(cond, t, array(complex64_t{9, 9}), stream),
                             stream);
  check_exact(scalar, {{1, 2}, {9, 9}, {5, 6}});
  auto flipped =
      read_complex(where(logical_not(cond), t, f, stream), stream);
  check_exact(flipped, {{-1, -2}, {3, 4}, {-5, -6}});
}

TEST_CASE("complex power matches host reference with zero-base cases") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // General bases and exponents against std::pow over double.
  std::vector<cdouble> bases{{2, 1}, {-1, 0.5}, {0.5, -0.5}, {3, -2}};
  std::vector<cdouble> exponents{{2, 0}, {0.5, 0.5}, {-1, 1}, {3, 2}};
  std::vector<complex64_t> hb(bases.size());
  for (size_t i = 0; i < bases.size(); ++i) {
    hb[i] = complex64_t(float(bases[i].real()), float(bases[i].imag()));
  }
  std::vector<complex64_t> he(exponents.size());
  for (size_t i = 0; i < exponents.size(); ++i) {
    he[i] = complex64_t(
        float(exponents[i].real()), float(exponents[i].imag()));
  }
  array b(hb.begin(), Shape{4}, complex64);
  array e(he.begin(), Shape{4}, complex64);
  auto got = read_complex(power(b, e, stream), stream);
  for (size_t i = 0; i < bases.size(); ++i) {
    cdouble want = std::pow(
        cdouble(float(bases[i].real()), float(bases[i].imag())),
        cdouble(float(exponents[i].real()), float(exponents[i].imag())));
    INFO("power index ", i, " got (", got[i].real(), ", ", got[i].imag(),
         ") want (", want.real(), ", ", want.imag(), ")");
    CHECK(got[i].real() == doctest::Approx(want.real()).epsilon(1e-4));
    CHECK(got[i].imag() == doctest::Approx(want.imag()).epsilon(1e-4));
  }
  // Zero-base special cases: 0^0 = 1, 0^positive-real = 0.
  array zero({complex64_t{0, 0}}, {1});
  array e0({complex64_t{0, 0}}, {1});
  array epos({complex64_t{2, 0}}, {1});
  check_exact(read_complex(power(zero, e0, stream), stream), {{1, 0}});
  check_exact(read_complex(power(zero, epos, stream), stream), {{0, 0}});
}

TEST_CASE("complex exp matches host reference") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::vector<cdouble> vals{{0, 0}, {1, 0}, {0, M_PI}, {0.5, -2}, {-1, 3}};
  std::vector<complex64_t> host(vals.size());
  for (size_t i = 0; i < vals.size(); ++i) {
    host[i] = complex64_t(float(vals[i].real()), float(vals[i].imag()));
  }
  array z(host.begin(), Shape{5}, complex64);
  auto got = read_complex(exp(z, stream), stream);
  for (size_t i = 0; i < vals.size(); ++i) {
    cdouble want = std::exp(
        cdouble(float(vals[i].real()), float(vals[i].imag())));
    INFO("exp index ", i, " got (", got[i].real(), ", ", got[i].imag(),
         ") want (", want.real(), ", ", want.imag(), ")");
    CHECK(got[i].real() == doctest::Approx(want.real()).epsilon(1e-5));
    CHECK(got[i].imag() == doctest::Approx(want.imag()).epsilon(1e-5));
  }
}

TEST_CASE("complex exp matches C99 for infinite arguments") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<complex64_t> host = {
      complex64_t{-inf, -inf},
      complex64_t{-inf, 2.0f},
      complex64_t{1.0f, -inf}};
  array z(host.begin(), Shape{3}, complex64);
  auto got = read_complex(exp(z, stream), stream);
  // C99 G.6.3.2: cexp(-inf + i*inf) = +0 + i0; the naive
  // exp(a)*(cos b, sin b) turns 0 * cos(inf) into NaN.
  CHECK_EQ(got[0].real(), 0.0);
  CHECK_EQ(got[0].imag(), 0.0);
  // A -inf real with a finite argument keeps the magnitude-zero
  // result (0 * cos(2) may carry the sign of cos, but compares 0).
  CHECK_EQ(got[1].real(), 0.0);
  CHECK_EQ(got[1].imag(), 0.0);
  // A finite real with an infinite argument stays NaN per C99.
  CHECK(std::isnan(got[2].real()));
  CHECK(std::isnan(got[2].imag()));
}

TEST_CASE("complex sin matches host reference") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::vector<cdouble> vals{{0, 0}, {0.5, 0.25}, {-1, 0}, {2, -1}, {0, 1}};
  std::vector<complex64_t> host(vals.size());
  for (size_t i = 0; i < vals.size(); ++i) {
    host[i] = complex64_t(float(vals[i].real()), float(vals[i].imag()));
  }
  array z(host.begin(), Shape{5}, complex64);
  auto got = read_complex(sin(z, stream), stream);
  for (size_t i = 0; i < vals.size(); ++i) {
    cdouble want = std::sin(
        cdouble(float(vals[i].real()), float(vals[i].imag())));
    INFO("sin index ", i, " got (", got[i].real(), ", ", got[i].imag(),
         ") want (", want.real(), ", ", want.imag(), ")");
    CHECK(got[i].real() == doctest::Approx(want.real()).epsilon(1e-5));
    CHECK(got[i].imag() == doctest::Approx(want.imag()).epsilon(1e-5));
  }
}

TEST_CASE("complex cos matches host reference") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::vector<cdouble> vals{{0, 0}, {0.5, 0.25}, {-1, 0}, {2, -1}, {0, 1}};
  std::vector<complex64_t> host(vals.size());
  for (size_t i = 0; i < vals.size(); ++i) {
    host[i] = complex64_t(float(vals[i].real()), float(vals[i].imag()));
  }
  array z(host.begin(), Shape{5}, complex64);
  auto got = read_complex(cos(z, stream), stream);
  for (size_t i = 0; i < vals.size(); ++i) {
    cdouble want = std::cos(
        cdouble(float(vals[i].real()), float(vals[i].imag())));
    INFO("cos index ", i, " got (", got[i].real(), ", ", got[i].imag(),
         ") want (", want.real(), ", ", want.imag(), ")");
    CHECK(got[i].real() == doctest::Approx(want.real()).epsilon(1e-5));
    CHECK(got[i].imag() == doctest::Approx(want.imag()).epsilon(1e-5));
  }
}

TEST_CASE("complex maximum orders lexicographically") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // Larger real wins regardless of imaginary; equal real compares
  // imaginary; an exact tie keeps the left operand.
  array a({complex64_t{2, -100}, complex64_t{1, 5}, complex64_t{0, 0},
           complex64_t{1, 2}},
          {4});
  array b({complex64_t{1, 99}, complex64_t{1, 6}, complex64_t{0, 0},
           complex64_t{-3, 4}},
          {4});
  auto got = read_complex(maximum(a, b, stream), stream);
  check_exact(got, {{2, -100}, {1, 6}, {0, 0}, {1, 2}});
}

namespace {

// Checks one complex unary against a host double-precision reference
// evaluated on the float32-quantized input, the same comparison the
// sin/cos cases above use.
void check_unary_vs_host(
    const char* label,
    array out,
    const Stream& stream,
    cdouble (*fn)(cdouble),
    const std::vector<cdouble>& vals) {
  auto got = read_complex(out, stream);
  for (size_t i = 0; i < vals.size(); ++i) {
    cdouble want = fn(cdouble(float(vals[i].real()), float(vals[i].imag())));
    INFO(label, " index ", i, " got (", got[i].real(), ", ", got[i].imag(),
         ") want (", want.real(), ", ", want.imag(), ")");
    CHECK(got[i].real() == doctest::Approx(want.real()).epsilon(1e-5));
    CHECK(got[i].imag() == doctest::Approx(want.imag()).epsilon(1e-5));
  }
}

cdouble host_sinh(cdouble z) {
  return std::sinh(z);
}
cdouble host_cosh(cdouble z) {
  return std::cosh(z);
}
cdouble host_tan(cdouble z) {
  return std::tan(z);
}
cdouble host_tanh(cdouble z) {
  return std::tanh(z);
}
cdouble host_arccos(cdouble z) {
  return std::acos(z);
}
cdouble host_arcsin(cdouble z) {
  return std::asin(z);
}
cdouble host_arctan(cdouble z) {
  return std::atan(z);
}
cdouble host_sqrt(cdouble z) {
  return std::sqrt(z);
}
cdouble host_rsqrt(cdouble z) {
  return 1.0 / std::sqrt(z);
}
cdouble host_log(cdouble z) {
  return std::log(z);
}
cdouble host_log2(cdouble z) {
  return std::log(z) / std::log(2.0);
}
cdouble host_log10(cdouble z) {
  return std::log(z) / std::log(10.0);
}
cdouble host_round(cdouble z) {
  return {std::nearbyint(z.real()), std::nearbyint(z.imag())};
}
cdouble host_log1p(cdouble z) {
  // The upstream simd::log1p complex reference: atan2 argument and a
  // small-|z| magnitude branch that keeps log1p(r) exact.
  double x = float(z.real());
  double y = float(z.imag());
  double theta = std::atan2(y, x + 1.0);
  if (std::abs(z) < 0.5) {
    double r = x * (2.0 + x) + y * y;
    if (r == 0.0) {
      return {0.0, theta};
    }
    return {0.5 * std::log1p(r), theta};
  }
  return {std::log(std::hypot(x + 1.0, y)), theta};
}

} // namespace

TEST_CASE("complex sinh/cosh/tan/tanh/log1p match host references") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // {0, 0} pins zero handling, {-0.25, 0.1} exercises the log1p
  // small-magnitude branch, and the rest cross quadrants at moderate
  // magnitude.
  std::vector<cdouble> vals{
      {0, 0}, {0.5, 0.25}, {-0.25, 0.1}, {1, -1}, {0, 1}, {2, -1}};
  std::vector<complex64_t> host(vals.size());
  for (size_t i = 0; i < vals.size(); ++i) {
    host[i] = complex64_t(float(vals[i].real()), float(vals[i].imag()));
  }
  array z(host.begin(), Shape{6}, complex64);

  check_unary_vs_host("sinh", sinh(z, stream), stream, host_sinh, vals);
  check_unary_vs_host("cosh", cosh(z, stream), stream, host_cosh, vals);
  check_unary_vs_host("tan", tan(z, stream), stream, host_tan, vals);
  check_unary_vs_host("tanh", tanh(z, stream), stream, host_tanh, vals);
  check_unary_vs_host("log1p", log1p(z, stream), stream, host_log1p, vals);

  // x = -1 maps to a -inf magnitude with a zero argument, the
  // atan2(0, 0) = 0 convention the logaddexp case pins too.
  array neg1({complex64_t{-1.0f, 0.0f}}, {1});
  auto l1 = read_complex(log1p(neg1, stream), stream);
  CHECK(std::isinf(l1[0].real()));
  CHECK(l1[0].real() < 0.0);
  CHECK_EQ(l1[0].imag(), 0.0);
}

TEST_CASE("integer round preserves identity ties and dtype wrap") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();

  std::vector<int8_t> i8_values{
      -128, -125, -115, -25, -15, -5, 5, 15, 25, 115, 125, 127};
  array i8(i8_values.begin(), Shape{12}, int8);
  CHECK_EQ(read_values<int8_t>(round(i8, stream), stream), i8_values);
  CHECK_EQ(
      read_values<int8_t>(round(i8, -1, stream), stream),
      std::vector<int8_t>{126, -120, -120, -20, -20, 0, 0, 20, 20, 120, 120, -126});
  CHECK_EQ(
      read_values<int8_t>(round(i8, -2, stream), stream),
      std::vector<int8_t>{-100, -100, -100, 0, 0, 0, 0, 0, 0, 100, 100, 100});

  std::vector<uint8_t> u8_values{0, 5, 15, 25, 115, 125, 245, 250, 255};
  array u8(u8_values.begin(), Shape{9}, uint8);
  CHECK_EQ(read_values<uint8_t>(round(u8, stream), stream), u8_values);
  CHECK_EQ(
      read_values<uint8_t>(round(u8, -1, stream), stream),
      std::vector<uint8_t>{0, 0, 20, 20, 120, 120, 240, 250, 4});
  CHECK_EQ(
      read_values<uint8_t>(round(u8, -2, stream), stream),
      std::vector<uint8_t>{0, 0, 0, 0, 100, 100, 200, 200, 44});

  // No INT32_MAX element: its round trip passes through 2147483648.0f,
  // whose int32 conversion is undefined (x86 and llvmpipe wrap to
  // INT32_MIN, aarch64 and the AGX saturate to INT32_MAX). INT32_MIN
  // is exact in float32 and pins the negative extreme.
  std::vector<int32_t> i32_values{
      std::numeric_limits<int32_t>::min(), -250, -150, -50, 50, 150, 250};
  array i32(i32_values.begin(), Shape{7}, int32);
  CHECK_EQ(read_values<int32_t>(round(i32, stream), stream), i32_values);
  CHECK_EQ(
      read_values<int32_t>(round(i32, -1, stream), stream),
      std::vector<int32_t>{
          std::numeric_limits<int32_t>::min(), -250, -150, -50, 50, 150, 250});
  CHECK_EQ(
      read_values<int32_t>(round(i32, -2, stream), stream),
      std::vector<int32_t>{
          std::numeric_limits<int32_t>::min(), -200, -200, 0, 0, 200, 200});
}

TEST_CASE("complex inverse trig roots logs and round match host references") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::vector<cdouble> vals{
      {3, 4}, {-5, 12}, {-8, 0}, {0, 9}, {0.25, -0.5}, {1, 1}};
  auto z = complex_array(vals, Shape{6});

  check_unary_vs_host("arccos", arccos(z, stream), stream, host_arccos, vals);
  check_unary_vs_host("arcsin", arcsin(z, stream), stream, host_arcsin, vals);
  check_unary_vs_host("arctan", arctan(z, stream), stream, host_arctan, vals);
  check_unary_vs_host("sqrt", sqrt(z, stream), stream, host_sqrt, vals);
  check_unary_vs_host("rsqrt", rsqrt(z, stream), stream, host_rsqrt, vals);
  check_unary_vs_host("log", log(z, stream), stream, host_log, vals);
  check_unary_vs_host("log2", log2(z, stream), stream, host_log2, vals);
  check_unary_vs_host("log10", log10(z, stream), stream, host_log10, vals);

  std::vector<cdouble> round_vals{
      {22.2, 3.6}, {18.5, 98.2}, {0.5, -0.5}, {1.5, -1.5}};
  auto round_input = complex_array(round_vals, Shape{4});
  check_unary_vs_host(
      "round", round(round_input, stream), stream, host_round, round_vals);
}

TEST_CASE("complex sqrt and log preserve signed branch cuts") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  array roots(
      {complex64_t{-4.0f, 0.0f}, complex64_t{-4.0f, -0.0f},
       complex64_t{0.0f, 0.0f}, complex64_t{0.0f, -0.0f}},
      {4});
  auto square_roots = read_complex(sqrt(roots, stream), stream);
  CHECK_EQ(square_roots[0].real(), 0.0);
  CHECK_EQ(square_roots[0].imag(), 2.0);
  CHECK_FALSE(std::signbit(square_roots[0].imag()));
  CHECK_EQ(square_roots[1].real(), 0.0);
  CHECK_EQ(square_roots[1].imag(), -2.0);
  CHECK(std::signbit(square_roots[1].imag()));
  CHECK_FALSE(std::signbit(square_roots[2].imag()));
  CHECK(std::signbit(square_roots[3].imag()));

  array cuts(
      {complex64_t{-1.0f, 0.0f}, complex64_t{-1.0f, -0.0f}}, {2});
  auto logs = read_complex(log(cuts, stream), stream);
  CHECK(logs[0].imag() == doctest::Approx(std::acos(-1.0)).epsilon(1e-6));
  CHECK(logs[1].imag() == doctest::Approx(-std::acos(-1.0)).epsilon(1e-6));
}

TEST_CASE("complex inverse functions preserve finite and signed edge semantics") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  auto check_component = [](float got, float want) {
    if (std::isnan(want)) {
      CHECK(std::isnan(got));
    } else if (std::isinf(want)) {
      CHECK(std::isinf(got));
      CHECK_EQ(std::signbit(got), std::signbit(want));
    } else {
      CHECK(got == doctest::Approx(want).epsilon(1e-5).scale(1e-6));
      if (want == 0.0f) {
        CHECK_EQ(std::signbit(got), std::signbit(want));
      }
    }
  };
  auto check_values = [&](
                          const char* label,
                          const std::vector<cdouble>& got,
                          const std::vector<complex64_t>& inputs,
                          std::complex<float> (*fn)(const std::complex<float>&)) {
    REQUIRE_EQ(got.size(), inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      std::complex<float> source(inputs[i].real(), inputs[i].imag());
      std::complex<float> want = fn(source);
      INFO(label, " index ", i);
      check_component(float(got[i].real()), want.real());
      check_component(float(got[i].imag()), want.imag());
    }
  };

  std::vector<complex64_t> signed_zeros{
      {0.0f, 0.0f}, {0.0f, -0.0f}, {-0.0f, 0.0f}, {-0.0f, -0.0f}};
  array zeros(signed_zeros.begin(), Shape{4}, complex64);
  check_values(
      "arccos signed zero", read_complex(arccos(zeros, stream), stream),
      signed_zeros, static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::acos));
  check_values(
      "arcsin signed zero", read_complex(arcsin(zeros, stream), stream),
      signed_zeros, static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::asin));
  check_values(
      "arctan signed zero", read_complex(arctan(zeros, stream), stream),
      signed_zeros, static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::atan));

  std::vector<complex64_t> endpoints{
      {1.0f, 0.0f}, {1.0f, -0.0f}, {-1.0f, 0.0f}, {-1.0f, -0.0f}};
  array endpoint_values(endpoints.begin(), Shape{4}, complex64);
  check_values(
      "arccos endpoints", read_complex(arccos(endpoint_values, stream), stream),
      endpoints, static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::acos));
  check_values(
      "arcsin endpoints", read_complex(arcsin(endpoint_values, stream), stream),
      endpoints, static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::asin));

  std::vector<complex64_t> real_axis{
      {0.25f, 0.0f}, {0.25f, -0.0f}, {-0.25f, 0.0f}, {-0.25f, -0.0f},
      {0.5f, 0.0f}, {0.5f, -0.0f}, {-0.5f, 0.0f}, {-0.5f, -0.0f}};
  array real_axis_values(real_axis.begin(), Shape{8}, complex64);
  auto real_axis_asin = read_complex(arcsin(real_axis_values, stream), stream);
  auto real_axis_acos = read_complex(arccos(real_axis_values, stream), stream);
  check_values(
      "arcsin real axis", real_axis_asin, real_axis,
      static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::asin));
  check_values(
      "arccos real axis", real_axis_acos, real_axis,
      static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::acos));
  for (size_t i = 0; i < real_axis.size(); ++i) {
    INFO("real-axis signed zero index ", i);
    CHECK_EQ(real_axis_asin[i].imag(), 0.0);
    CHECK_EQ(std::signbit(real_axis_asin[i].imag()), std::signbit(real_axis[i].imag()));
    CHECK_EQ(real_axis_acos[i].imag(), 0.0);
    CHECK_EQ(std::signbit(real_axis_acos[i].imag()), !std::signbit(real_axis[i].imag()));
  }

  std::vector<complex64_t> imaginary_axis{
      {0.0f, 0.5f}, {-0.0f, 0.5f}, {0.0f, -0.5f}, {-0.0f, -0.5f},
      {0.0f, 2.0f}, {-0.0f, 2.0f}, {0.0f, -2.0f}, {-0.0f, -2.0f}};
  array imaginary_axis_values(imaginary_axis.begin(), Shape{8}, complex64);
  auto imaginary_axis_asin =
      read_complex(arcsin(imaginary_axis_values, stream), stream);
  check_values(
      "arcsin imaginary axis", imaginary_axis_asin, imaginary_axis,
      static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::asin));
  for (size_t i = 0; i < imaginary_axis.size(); ++i) {
    INFO("imaginary-axis signed zero index ", i);
    CHECK_EQ(imaginary_axis_asin[i].real(), 0.0);
    CHECK_EQ(
        std::signbit(imaginary_axis_asin[i].real()),
        std::signbit(imaginary_axis[i].real()));
  }

  std::vector<float> near_real_x{-0.9f, -0.5f, 0.25f, 0.5f, 0.9f};
  std::vector<float> near_real_y{
      0.0f, -0.0f, 1.0e-10f, -1.0e-10f,
      1.0e-7f, -1.0e-7f, 1.0e-4f, -1.0e-4f};
  std::vector<complex64_t> near_real;
  for (float real : near_real_x) {
    for (float imag : near_real_y) {
      near_real.emplace_back(real, imag);
    }
  }
  array near_real_values(near_real.begin(), Shape{static_cast<int>(near_real.size())}, complex64);
  auto near_real_asin = read_complex(arcsin(near_real_values, stream), stream);
  auto near_real_acos = read_complex(arccos(near_real_values, stream), stream);
  auto check_sensitive_values = [&](
                                    const char* label,
                                    const std::vector<cdouble>& got,
                                    const std::vector<complex64_t>& inputs,
                                    std::complex<float> (*fn)(const std::complex<float>&)) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      std::complex<float> source(inputs[i].real(), inputs[i].imag());
      std::complex<float> want = fn(source);
      INFO(std::string(label), " index ", i);
      CHECK(float(got[i].real()) ==
            doctest::Approx(want.real()).epsilon(3e-5).scale(1e-20));
      if (want.imag() == 0.0f) {
        CHECK_EQ(float(got[i].imag()), 0.0f);
        CHECK_EQ(std::signbit(float(got[i].imag())), std::signbit(want.imag()));
      } else {
        CHECK(float(got[i].imag()) ==
              doctest::Approx(want.imag()).epsilon(3e-5).scale(1e-20));
      }
    }
  };
  check_sensitive_values(
      "arcsin near real", near_real_asin, near_real,
      static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::asin));
  check_sensitive_values(
      "arccos near real", near_real_acos, near_real,
      static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::acos));

  float below_one = std::nextafter(1.0f, 0.0f);
  std::vector<complex64_t> branch_endpoints;
  for (float real : {-1.0f, 1.0f}) {
    for (float imag : {
             1.0e-7f, -1.0e-7f, 1.0e-10f, -1.0e-10f,
             1.0e-20f, -1.0e-20f}) {
      branch_endpoints.emplace_back(real, imag);
    }
  }
  for (float imag : {1.0e-10f, -1.0e-10f, 1.0e-20f, -1.0e-20f}) {
    branch_endpoints.emplace_back(below_one, imag);
  }
  array branch_endpoint_values(
      branch_endpoints.begin(),
      Shape{static_cast<int>(branch_endpoints.size())}, complex64);
  check_sensitive_values(
      "arcsin branch endpoints",
      read_complex(arcsin(branch_endpoint_values, stream), stream),
      branch_endpoints,
      static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::asin));
  check_sensitive_values(
      "arccos branch endpoints",
      read_complex(arccos(branch_endpoint_values, stream), stream),
      branch_endpoints,
      static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::acos));

  std::vector<complex64_t> atan_cuts{{-0.0f, 2.0f}, {-0.0f, -2.0f}};
  array cut_values(atan_cuts.begin(), Shape{2}, complex64);
  check_values(
      "arctan imaginary cut", read_complex(arctan(cut_values, stream), stream),
      atan_cuts, static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::atan));

  std::vector<cdouble> large{
      {1.0e20, 1.0e20}, {-1.0e20, 1.0e20}, {1.0e20, -1.0e20}};
  auto large_values = complex_array(large, Shape{3});
  check_unary_vs_host(
      "arccos large", arccos(large_values, stream), stream, host_arccos, large);
  check_unary_vs_host(
      "arcsin large", arcsin(large_values, stream), stream, host_arcsin, large);
  check_unary_vs_host(
      "arctan large", arctan(large_values, stream), stream, host_arctan, large);

  float max_value = std::numeric_limits<float>::max();
  std::vector<cdouble> finite_logs{
      {max_value, max_value}, {max_value, -max_value}};
  auto finite_log_values = complex_array(finite_logs, Shape{2});
  check_unary_vs_host(
      "log finite max", log(finite_log_values, stream), stream, host_log, finite_logs);
  check_unary_vs_host(
      "log2 finite max", log2(finite_log_values, stream), stream, host_log2, finite_logs);
  check_unary_vs_host(
      "log10 finite max", log10(finite_log_values, stream), stream, host_log10, finite_logs);

  std::vector<complex64_t> finite_roots{
      {max_value, 0.0f}, {-max_value, 0.0f},
      {max_value, -0.0f}, {-max_value, -0.0f}};
  array finite_root_values(finite_roots.begin(), Shape{4}, complex64);
  check_values(
      "sqrt finite extremes", read_complex(sqrt(finite_root_values, stream), stream),
      finite_roots, static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::sqrt));

  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<complex64_t> classified_specials{
      {1.0f, nan}, {nan, 1.0f}, {inf, nan}, {-inf, nan},
      {nan, inf}, {nan, -inf}};
  array classified_special_values(
      classified_specials.begin(), Shape{6}, complex64);
  auto special_logs = read_complex(log(classified_special_values, stream), stream);
  auto special_log2s =
      read_complex(log2(classified_special_values, stream), stream);
  auto special_log10s =
      read_complex(log10(classified_special_values, stream), stream);
  for (size_t i = 0; i < classified_specials.size(); ++i) {
    std::complex<float> source(
        classified_specials[i].real(), classified_specials[i].imag());
    std::complex<float> want_log = std::log(source);
    INFO("log mixed special index ", i);
    check_component(float(special_logs[i].real()), want_log.real());
    check_component(float(special_logs[i].imag()), want_log.imag());
    std::complex<float> want_log2 = want_log / std::log(2.0f);
    INFO("log2 mixed special index ", i);
    check_component(float(special_log2s[i].real()), want_log2.real());
    check_component(float(special_log2s[i].imag()), want_log2.imag());
    std::complex<float> want_log10 = want_log / std::log(10.0f);
    INFO("log10 mixed special index ", i);
    check_component(float(special_log10s[i].real()), want_log10.real());
    check_component(float(special_log10s[i].imag()), want_log10.imag());
  }
  check_values(
      "arcsin mixed specials",
      read_complex(arcsin(classified_special_values, stream), stream),
      classified_specials,
      static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::asin));
  check_values(
      "arccos mixed specials",
      read_complex(arccos(classified_special_values, stream), stream),
      classified_specials,
      static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::acos));

  std::vector<complex64_t> atan_extremes{
      {0.0f, max_value}, {0.0f, -max_value}, {-0.0f, max_value},
      {-0.0f, -max_value}, {nan, inf}, {nan, -inf}, {inf, nan},
      {-inf, nan}, {1.0e-4f, -1.0f}, {1.0e-8f, -1.0f},
      {1.0e-20f, 1.0f}, {0.0f, 1.0f}, {-0.0f, 1.0f},
      {0.0f, -1.0f}, {-0.0f, -1.0f}};
  array atan_extreme_values(atan_extremes.begin(), Shape{15}, complex64);
  check_values(
      "arctan finite and mixed extremes",
      read_complex(arctan(atan_extreme_values, stream), stream), atan_extremes,
      static_cast<std::complex<float> (*)(const std::complex<float>&)>(std::atan));

  array mixed(
      {complex64_t{inf, nan}, complex64_t{-inf, nan}}, {2});
  auto mixed_roots = read_complex(sqrt(mixed, stream), stream);
  CHECK(std::isinf(mixed_roots[0].real()));
  CHECK(std::isnan(mixed_roots[0].imag()));
  CHECK(std::isnan(mixed_roots[1].real()));
  CHECK(std::isinf(mixed_roots[1].imag()));

  std::vector<complex64_t> infinite_roots{
      {inf, 0.0f}, {inf, -0.0f}, {-inf, 0.0f}, {-inf, -0.0f}};
  array infinite_root_values(infinite_roots.begin(), Shape{4}, complex64);
  auto reciprocals = read_complex(rsqrt(infinite_root_values, stream), stream);
  for (size_t i = 0; i < infinite_roots.size(); ++i) {
    std::complex<float> source(
        infinite_roots[i].real(), infinite_roots[i].imag());
    std::complex<float> want = 1.0f / std::sqrt(source);
    INFO("rsqrt infinite index ", i);
    check_component(float(reciprocals[i].real()), want.real());
    check_component(float(reciprocals[i].imag()), want.imag());
  }

  std::vector<complex64_t> zero_roots{
      {0.0f, 0.0f}, {0.0f, -0.0f}, {-0.0f, 0.0f}, {-0.0f, -0.0f}};
  array zero_root_values(zero_roots.begin(), Shape{4}, complex64);
  auto zero_reciprocals = read_complex(rsqrt(zero_root_values, stream), stream);
  for (size_t i = 0; i < zero_roots.size(); ++i) {
    std::complex<float> source(zero_roots[i].real(), zero_roots[i].imag());
    std::complex<float> want = 1.0f / std::sqrt(source);
    INFO("rsqrt zero index ", i);
    check_component(float(zero_reciprocals[i].real()), want.real());
    check_component(float(zero_reciprocals[i].imag()), want.imag());
  }
}

TEST_CASE("complex sign maps zero to itself and z to z/abs(z)") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::vector<cdouble> vals{
      {0, 0}, {0.5, -0.25}, {-1, 0}, {0, 0.75}, {-3, 4}, {2, 2}};
  std::vector<complex64_t> host(vals.size());
  for (size_t i = 0; i < vals.size(); ++i) {
    host[i] = complex64_t(float(vals[i].real()), float(vals[i].imag()));
  }
  array z(host.begin(), Shape{6}, complex64);
  auto got = read_complex(sign(z, stream), stream);
  // The zero element maps to itself; everything else to z/|z| with
  // unit modulus.
  CHECK_EQ(got[0].real(), 0.0);
  CHECK_EQ(got[0].imag(), 0.0);
  for (size_t i = 1; i < vals.size(); ++i) {
    cdouble want = cdouble(float(vals[i].real()), float(vals[i].imag())) /
        std::abs(cdouble(float(vals[i].real()), float(vals[i].imag())));
    INFO("sign index ", i, " got (", got[i].real(), ", ", got[i].imag(),
         ") want (", want.real(), ", ", want.imag(), ")");
    CHECK(got[i].real() == doctest::Approx(want.real()).epsilon(1e-6));
    CHECK(got[i].imag() == doctest::Approx(want.imag()).epsilon(1e-6));
  }
}

TEST_CASE("complex unary operations retain finite large-magnitude results") {
  if (!compute_available()) return;
  auto stream = gpu_stream();
  std::vector<cdouble> hyper{{89.0, 0.25}, {-89.0, -0.25}};
  auto h = complex_array(hyper, Shape{2});
  check_unary_vs_host("sinh large", sinh(h, stream), stream, host_sinh, hyper);
  check_unary_vs_host("cosh large", cosh(h, stream), stream, host_cosh, hyper);
  const float half_pi = 1.5707963267948966f;
  std::vector<cdouble> trig{{0.5, 100.0}, {100.0, -0.5}, {half_pi, 0.0}};
  auto t = complex_array(trig, Shape{3});
  check_unary_vs_host("tanh large", tanh(t, stream), stream, host_tanh, trig);
  // tan(fl(pi/2) + 0i) = -2.2877e7 needs cos(fl(pi/2)) = -4.3711e-8 to
  // full relative accuracy. The stock Honeykrisp builtin returns 0
  // there (inside the Vulkan 2^-11 absolute envelope) and the quotient
  // reads NaN; the hk/precise-math fork trig (979453d) reduces exactly.
  // Probe the builtin and name that driver requirement instead of
  // widening the tolerance.
  std::vector<cdouble> tan_vals(trig.begin(), trig.end() - 1);
  if (read_values<float>(cos(array(half_pi), stream), stream)[0] != 0.0f) {
    tan_vals.push_back(trig.back());
  } else {
    skip("tan(fl(pi/2)): builtin cos(fl(pi/2)) is 0 on this driver; needs"
         " the hk/precise-math Honeykrisp trig");
  }
  auto tan_in = complex_array(tan_vals, Shape{int(tan_vals.size())});
  check_unary_vs_host("tan large", tan(tan_in, stream), stream, host_tan, tan_vals);
  std::vector<cdouble> large{{1e30, 1e30}, {-1e30, 1e30}};
  auto z = complex_array(large, Shape{2});
  check_unary_vs_host("log1p large", log1p(z, stream), stream, host_log1p, large);
  auto signs = read_complex(sign(z, stream), stream);
  for (size_t i = 0; i < large.size(); ++i) {
    cdouble expected = large[i] / std::abs(large[i]);
    CHECK(signs[i].real() == doctest::Approx(expected.real()).epsilon(1e-5));
    CHECK(signs[i].imag() == doctest::Approx(expected.imag()).epsilon(1e-5));
  }
}
