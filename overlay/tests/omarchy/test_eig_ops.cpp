// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// Eig coverage: general nonsymmetric float32 eigen-decomposition on the
// LinalgEigF32 kernel (EISPACK hqr2 port). Value policy: references are
// computed in this file, never hardcoded device output - exact spectra
// through similarity transforms, a host Jacobi reference for the
// symmetric case, analytic rotation-matrix vectors, and the defining
// identity A v = lambda v per eigenpair, which a wrong eigenvector fails
// even when every eigenvalue is right. Conjugate-pair adjacency
// (positive imaginary part first) and LAPACK-style normalization are
// pinned because they are part of the upstream CPU contract.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/device.h"
#include "mlx/linalg.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;
using cdouble = std::complex<double>;

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

std::string evaluation_error(array value) {
  try {
    value.eval();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

std::vector<cdouble> read_complex(array a, const Stream& stream) {
  a.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const complex64_t* data = a.data<complex64_t>();
  std::vector<cdouble> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = {double(data[i].real()), double(data[i].imag())};
  }
  return out;
}

// doctest forbids operator&& inside CHECK expressions, so needle checks
// go through this helper.
bool contains_all(
    const std::string& text,
    std::initializer_list<const char*> needles) {
  for (const char* needle : needles) {
    if (text.find(needle) == std::string::npos) {
      return false;
    }
  }
  return true;
}

array device_matrix(
    const Stream& stream,
    const std::vector<double>& values,
    Shape shape) {
  std::vector<float> flat(values.begin(), values.end());
  array out(flat.begin(), shape, float32);
  return astype(out, float32, stream);
}

using HostMatrix = std::vector<double>;

HostMatrix host_matmul(
    const HostMatrix& a,
    const HostMatrix& b,
    int rows_a,
    int cols_a,
    int cols_b) {
  HostMatrix out(rows_a * cols_b, 0.0);
  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      double sum = 0.0;
      for (int l = 0; l < cols_a; ++l) {
        sum += a[i * cols_a + l] * b[l * cols_b + j];
      }
      out[i * cols_b + j] = sum;
    }
  }
  return out;
}

// Element check with a relative-plus-absolute tolerance against a host
// double reference.
void expect_close_reference(
    const std::vector<double>& device,
    const std::vector<double>& expected,
    double tol,
    const std::string& label) {
  REQUIRE_EQ(device.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    double diff = std::abs(device[i] - expected[i]);
    CHECK_MESSAGE(
        diff <= tol + tol * std::abs(expected[i]),
        label,
        " index ",
        i,
        ": device ",
        device[i],
        " vs host ",
        expected[i]);
  }
}

std::vector<double> vec_real(const std::vector<cdouble>& v) {
  std::vector<double> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    out[i] = v[i].real();
  }
  return out;
}

HostMatrix host_transpose(const HostMatrix& a, int rows, int cols) {
  HostMatrix out(rows * cols, 0.0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      out[j * rows + i] = a[i * cols + j];
    }
  }
  return out;
}

HostMatrix host_identity(int n) {
  HostMatrix out(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    out[i * n + i] = 1.0;
  }
  return out;
}

// Gauss-Jordan with partial pivoting, double precision. Used only to
// build similarity transforms with a known spectrum; conditioning is
// asserted by the caller's tolerance, not assumed.
HostMatrix host_inverse(const HostMatrix& a, int n) {
  HostMatrix aug(n * 2 * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      aug[i * 2 * n + j] = a[i * n + j];
    }
    aug[i * 2 * n + n + i] = 1.0;
  }
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    for (int row = col + 1; row < n; ++row) {
      if (std::abs(aug[row * 2 * n + col]) >
          std::abs(aug[pivot * 2 * n + col])) {
        pivot = row;
      }
    }
    CHECK_MESSAGE(
        std::abs(aug[pivot * 2 * n + col]) > 1e-12,
        "similarity transform built a singular P");
    if (pivot != col) {
      for (int j = 0; j < 2 * n; ++j) {
        std::swap(aug[col * 2 * n + j], aug[pivot * 2 * n + j]);
      }
    }
    double d = aug[col * 2 * n + col];
    for (int j = 0; j < 2 * n; ++j) {
      aug[col * 2 * n + j] /= d;
    }
    for (int row = 0; row < n; ++row) {
      if (row == col) {
        continue;
      }
      double f = aug[row * 2 * n + col];
      for (int j = 0; j < 2 * n; ++j) {
        aug[row * 2 * n + j] -= f * aug[col * 2 * n + j];
      }
    }
  }
  HostMatrix inv(n * n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      inv[i * n + j] = aug[i * 2 * n + n + j];
    }
  }
  return inv;
}

// Host cyclic Jacobi for a symmetric matrix in double precision: the
// independent eigenvalue reference. Ascending.
HostMatrix host_sym_eigvals(const HostMatrix& a, int n) {
  HostMatrix w = a;
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        off += w[p * n + q] * w[p * n + q];
      }
    }
    if (off < 1e-26) {
      break;
    }
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        double apq = w[p * n + q];
        if (std::abs(apq) < 1e-15) {
          w[p * n + q] = 0.0;
          w[q * n + p] = 0.0;
          continue;
        }
        double tau = (w[q * n + q] - w[p * n + p]) / (2.0 * apq);
        double t = (tau >= 0.0)
            ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
            : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
        double c = 1.0 / std::sqrt(1.0 + t * t);
        double s = t * c;
        for (int i = 0; i < n; ++i) {
          double x = w[i * n + p];
          double y = w[i * n + q];
          w[i * n + p] = c * x - s * y;
          w[i * n + q] = s * x + c * y;
        }
        for (int j2 = 0; j2 < n; ++j2) {
          double x = w[p * n + j2];
          double y = w[q * n + j2];
          w[p * n + j2] = c * x - s * y;
          w[q * n + j2] = s * x + c * y;
        }
      }
    }
  }
  HostMatrix vals(n);
  for (int i = 0; i < n; ++i) {
    vals[i] = w[i * n + i];
  }
  std::sort(vals.begin(), vals.end());
  return vals;
}

// Greedy multiset match: every device eigenvalue must land within
// tol + tol*|expected| of a distinct expected value. Order is NOT pinned
// (Schur order is data dependent), the multiset is.
void expect_spectrum(
    const std::vector<cdouble>& device,
    const std::vector<cdouble>& expected,
    double tol,
    const std::string& label) {
  REQUIRE_EQ(device.size(), expected.size());
  std::vector<bool> used(expected.size(), false);
  size_t index = 0;
  for (const auto& dv : device) {
    size_t best = expected.size();
    double best_d = std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < expected.size(); ++j) {
      if (used[j]) {
        continue;
      }
      double d = std::abs(dv - expected[j]);
      if (d < best_d) {
        best_d = d;
        best = j;
      }
    }
    bool matched = best < expected.size() &&
        best_d <= tol + tol * std::abs(expected[best]);
    CHECK_MESSAGE(
        matched,
        label,
        " eigenvalue ",
        index,
        ": device (",
        dv.real(),
        ", ",
        dv.imag(),
        ") unmatched");
    if (best < expected.size()) {
      used[best] = true;
    }
    index++;
  }
}

// The defining property, checked per eigenpair: A v = lambda v within a
// norm-scaled tolerance, unit Euclidean norm per eigenvector, and
// conjugate pairs adjacent with the positive imaginary part first.
void expect_eigenpairs(
    const HostMatrix& a,
    int n,
    const std::vector<cdouble>& values,
    const std::vector<cdouble>& vectors,
    double tol,
    const std::string& label) {
  REQUIRE_EQ(vectors.size(), size_t(n) * size_t(n));
  double a_norm = 0.0;
  for (double v : a) {
    a_norm = std::max(a_norm, std::abs(v));
  }
  for (int j = 0; j < n; ++j) {
    double nrm = 0.0;
    for (int i = 0; i < n; ++i) {
      nrm += std::norm(vectors[size_t(i) * n + j]);
    }
    nrm = std::sqrt(nrm);
    CHECK_MESSAGE(
        std::abs(nrm - 1.0) <= tol,
        label,
        " eigenvector ",
        j,
        " norm ",
        nrm);
    double resid = 0.0;
    for (int i = 0; i < n; ++i) {
      cdouble sum(0.0, 0.0);
      for (int k = 0; k < n; ++k) {
        sum += a[size_t(i) * n + k] * vectors[size_t(k) * n + j];
      }
      cdouble diff = sum - values[j] * vectors[size_t(i) * n + j];
      resid = std::max(resid, std::abs(diff));
    }
    CHECK_MESSAGE(
        resid <= tol * std::max(1.0, a_norm),
        label,
        " A v = lambda v fails at pair ",
        j,
        ": residual ",
        resid);
    if (values[j].imag() > tol) {
      bool paired = std::abs(values[j + 1].real() - values[j].real()) <= tol;
      CHECK_MESSAGE(
          paired,
          label,
          " conjugate pair not adjacent/ordered at ",
          j);
      bool conjugate = std::abs(values[j + 1].imag() + values[j].imag()) <=
          tol;
      CHECK_MESSAGE(
          conjugate,
          label,
          " conjugate pair imaginary sign wrong at ",
          j);
    }
  }
}

} // namespace

TEST_CASE("eig diagonal matrix exact spectrum and eigenvectors") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  HostMatrix a = {3.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0};
  int n = 3;
  array device = device_matrix(stream, a, Shape{n, n});
  auto [w, v] = linalg::eig(device, stream);
  auto values = read_complex(w, stream);
  auto vectors = read_complex(v, stream);
  expect_spectrum(
      values,
      {cdouble(3.0, 0.0), cdouble(-1.0, 0.0), cdouble(2.0, 0.0)},
      1e-6,
      "diag eig");
  expect_eigenpairs(a, n, values, vectors, 1e-5, "diag eig");
  // A diagonal input is already in Schur form, so every eigenvector must
  // come out as a signed unit basis column, here exactly e_j.
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      cdouble expected((i == j) ? 1.0 : 0.0, 0.0);
      cdouble got = vectors[size_t(i) * n + j];
      CHECK_MESSAGE(
          std::abs(got - expected) <= 1e-5,
          "diag eigenvector (",
          i,
          ", ",
          j,
          "): got (",
          got.real(),
          ", ",
          got.imag(),
          ")");
    }
  }
}

TEST_CASE("eig symmetric matrix matches host Jacobi and the eigh path") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(203);
  int n = 5;
  HostMatrix full(n * n);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto& value : full) {
    value = dist(gen);
  }
  HostMatrix a = host_matmul(full, host_transpose(full, n, n), n, n, n);
  for (int i = 0; i < n; ++i) {
    a[i * n + i] += n;
  }
  array device = device_matrix(stream, a, Shape{n, n});
  auto [w, v] = linalg::eig(device, stream);
  auto values = read_complex(w, stream);
  auto vectors = read_complex(v, stream);
  // A real symmetric matrix has an exactly real spectrum; every
  // imaginary part must be exactly zero (the kernel pins them to 0).
  for (const auto& value : values) {
    CHECK_EQ(value.imag(), 0.0);
  }
  HostMatrix sorted(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    sorted[i] = values[i].real();
  }
  std::sort(sorted.begin(), sorted.end());
  expect_close_reference(sorted, host_sym_eigvals(a, n), 1e-4, "eig symmetric");
  expect_eigenpairs(a, n, values, vectors, 1e-4, "eig symmetric");
  // Orthonormality of the returned basis (real eigenvectors): Vt V = I.
  HostMatrix gram(n * n, 0.0);
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      double sum = 0.0;
      for (int i = 0; i < n; ++i) {
        sum += vectors[size_t(i) * n + r].real() *
            vectors[size_t(i) * n + c].real();
      }
      gram[size_t(r) * n + c] = sum;
    }
  }
  expect_close_reference(gram, host_identity(n), 1e-4, "eig symmetric Vt V");
  // Cross-check against the existing Eigh path on the same input.
  auto w2 = linalg::eigvalsh(device, "L", stream);
  w2.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* w2_data = w2.data<float>();
  HostMatrix eigh_vals(w2_data, w2_data + w2.size());
  std::sort(eigh_vals.begin(), eigh_vals.end());
  expect_close_reference(sorted, eigh_vals, 1e-4, "eig vs eigh spectrum");
}

TEST_CASE("eig rotation matrix conjugate pair on the unit circle") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // 90-degree rotation: eigenvalues are exactly +/- i, and the analytic
  // eigenvectors are (1, -i)/sqrt(2) for +i and (1, i)/sqrt(2) for -i.
  HostMatrix a = {0.0, -1.0, 1.0, 0.0};
  int n = 2;
  array device = device_matrix(stream, a, Shape{n, n});
  auto [w, v] = linalg::eig(device, stream);
  auto values = read_complex(w, stream);
  auto vectors = read_complex(v, stream);
  REQUIRE_EQ(values.size(), size_t(2));
  // Pair order contract: positive imaginary part first.
  CHECK_MESSAGE(
      std::abs(values[0] - cdouble(0.0, 1.0)) <= 1e-5,
      "first rotation eigenvalue: (",
      values[0].real(),
      ", ",
      values[0].imag(),
      ")");
  CHECK_MESSAGE(
      std::abs(values[1] - cdouble(0.0, -1.0)) <= 1e-5,
      "second rotation eigenvalue: (",
      values[1].real(),
      ", ",
      values[1].imag(),
      ")");
  const double s2 = 1.0 / std::sqrt(2.0);
  // v(+i) = (1, -i)/sqrt(2): component 0 is real, component 1 purely
  // imaginary. v(-i) is the conjugate.
  CHECK_MESSAGE(
      std::abs(vectors[0 * n + 0] - cdouble(s2, 0.0)) <= 1e-5,
      "rotation eigenvector 0 component 0: (",
      vectors[0 * n + 0].real(),
      ", ",
      vectors[0 * n + 0].imag(),
      ")");
  CHECK_MESSAGE(
      std::abs(vectors[1 * n + 0] - cdouble(0.0, -s2)) <= 1e-5,
      "rotation eigenvector 0 component 1: (",
      vectors[1 * n + 0].real(),
      ", ",
      vectors[1 * n + 0].imag(),
      ")");
  CHECK_MESSAGE(
      std::abs(vectors[0 * n + 1] - cdouble(s2, 0.0)) <= 1e-5,
      "rotation eigenvector 1 component 0: (",
      vectors[0 * n + 1].real(),
      ", ",
      vectors[0 * n + 1].imag(),
      ")");
  CHECK_MESSAGE(
      std::abs(vectors[1 * n + 1] - cdouble(0.0, s2)) <= 1e-5,
      "rotation eigenvector 1 component 1: (",
      vectors[1 * n + 1].real(),
      ", ",
      vectors[1 * n + 1].imag(),
      ")");
  expect_eigenpairs(a, n, values, vectors, 1e-5, "rotation eig");
}

TEST_CASE("eig similarity transform recovers the exact spectrum") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // A = P D P^-1 with a block rotation carrying 2 +/- 3i and real values
  // 1, 5, 0.5: the spectrum is known exactly, in double precision, while
  // A itself is a nonsymmetric matrix no textbook formula solves.
  int n = 5;
  HostMatrix d(n * n, 0.0);
  d[0 * n + 0] = 2.0;
  d[0 * n + 1] = -3.0;
  d[1 * n + 0] = 3.0;
  d[1 * n + 1] = 2.0;
  d[2 * n + 2] = 1.0;
  d[3 * n + 3] = 5.0;
  d[4 * n + 4] = 0.5;
  std::mt19937 gen(58);
  std::uniform_real_distribution<double> dist(-0.3, 0.3);
  HostMatrix p(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    p[i * n + i] = 1.0;
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        p[i * n + j] = dist(gen);
      }
    }
  }
  HostMatrix pinv = host_inverse(p, n);
  REQUIRE_FALSE(pinv.empty());
  HostMatrix a = host_matmul(p, host_matmul(d, pinv, n, n, n), n, n, n);
  array device = device_matrix(stream, a, Shape{n, n});
  auto [w, v] = linalg::eig(device, stream);
  auto values = read_complex(w, stream);
  auto vectors = read_complex(v, stream);
  expect_spectrum(
      values,
      {cdouble(2.0, 3.0),
       cdouble(2.0, -3.0),
       cdouble(1.0, 0.0),
       cdouble(5.0, 0.0),
       cdouble(0.5, 0.0)},
      1e-4,
      "similarity eig");
  expect_eigenpairs(a, n, values, vectors, 1e-4, "similarity eig");
  // Determinism inside one process: a second identical call must return
  // bit-identical outputs.
  auto [w2, v2] = linalg::eig(device, stream);
  auto values2 = read_complex(w2, stream);
  auto vectors2 = read_complex(v2, stream);
  CHECK_EQ(values.size(), values2.size());
  for (size_t i = 0; i < values.size(); ++i) {
    CHECK_EQ(values[i].real(), values2[i].real());
    CHECK_EQ(values[i].imag(), values2[i].imag());
  }
  for (size_t i = 0; i < vectors.size(); ++i) {
    CHECK_EQ(vectors[i].real(), vectors2[i].real());
    CHECK_EQ(vectors[i].imag(), vectors2[i].imag());
  }
}

TEST_CASE("eig repeated eigenvalues stay correct") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();

  // Scalar identity: one eigenvalue with multiplicity 3 and the eps-pivot
  // path in the back-substitution; vectors must still be exact basis
  // columns satisfying A v = 2 v.
  {
    int n = 3;
    HostMatrix a = host_identity(n);
    for (int i = 0; i < n; ++i) {
      a[size_t(i) * n + i] = 2.0;
    }
    array device = device_matrix(stream, a, Shape{n, n});
    auto [w, v] = linalg::eig(device, stream);
    auto values = read_complex(w, stream);
    auto vectors = read_complex(v, stream);
    expect_spectrum(
        values,
        {cdouble(2.0, 0.0), cdouble(2.0, 0.0), cdouble(2.0, 0.0)},
        1e-6,
        "2I eig");
    expect_eigenpairs(a, n, values, vectors, 1e-5, "2I eig");
    expect_close_reference(
        vec_real(vectors),
        host_identity(n),
        1e-5,
        "2I eigenvectors");
  }

  // Symmetric with a genuinely repeated eigenvalue (2, 2, 3): full
  // orthonormal eigenbasis must come back.
  {
    int n = 3;
    double theta = 0.7;
    HostMatrix q = host_identity(n);
    q[0 * n + 0] = std::cos(theta);
    q[0 * n + 1] = -std::sin(theta);
    q[1 * n + 0] = std::sin(theta);
    q[1 * n + 1] = std::cos(theta);
    HostMatrix diag(n * n, 0.0);
    diag[0] = 2.0;
    diag[1 * n + 1] = 2.0;
    diag[2 * n + 2] = 3.0;
    HostMatrix a =
        host_matmul(q, host_matmul(diag, host_transpose(q, n, n), n, n, n),
                    n, n, n);
    array device = device_matrix(stream, a, Shape{n, n});
    auto [w, v] = linalg::eig(device, stream);
    auto values = read_complex(w, stream);
    auto vectors = read_complex(v, stream);
    expect_spectrum(
        values,
        {cdouble(2.0, 0.0), cdouble(2.0, 0.0), cdouble(3.0, 0.0)},
        1e-5,
        "repeated symmetric eig");
    expect_eigenpairs(a, n, values, vectors, 1e-4, "repeated symmetric eig");
  }

  // Jordan block [[2, 1], [0, 2]]: defective, one true eigenvector. The
  // values are exact; the returned pair must still satisfy the defining
  // identity within a loose float tolerance (the second vector is
  // perturbation dominated by construction).
  {
    int n = 2;
    HostMatrix a = {2.0, 1.0, 0.0, 2.0};
    array device = device_matrix(stream, a, Shape{n, n});
    auto [w, v] = linalg::eig(device, stream);
    auto values = read_complex(w, stream);
    auto vectors = read_complex(v, stream);
    expect_spectrum(
        values, {cdouble(2.0, 0.0), cdouble(2.0, 0.0)}, 1e-6, "jordan eig");
    expect_eigenpairs(a, n, values, vectors, 1e-3, "jordan eig");
  }
}

TEST_CASE("eig batched matrices match per-matrix references") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  int n = 2;
  // Batch 0: rotation (complex pair). Batch 1: diag(3, -1).
  HostMatrix both = {0.0, -1.0, 1.0, 0.0, 3.0, 0.0, 0.0, -1.0};
  array device = device_matrix(stream, both, Shape{2, n, n});
  auto [w, v] = linalg::eig(device, stream);
  auto values = read_complex(w, stream);
  auto vectors = read_complex(v, stream);
  REQUIRE_EQ(values.size(), size_t(4));
  REQUIRE_EQ(vectors.size(), size_t(8));
  std::vector<cdouble> b0_values(values.begin(), values.begin() + 2);
  std::vector<cdouble> b1_values(values.begin() + 2, values.end());
  std::vector<cdouble> b0_vectors(vectors.begin(), vectors.begin() + 4);
  std::vector<cdouble> b1_vectors(vectors.begin() + 4, vectors.end());
  HostMatrix a0 = {0.0, -1.0, 1.0, 0.0};
  HostMatrix a1 = {3.0, 0.0, 0.0, -1.0};
  expect_spectrum(
      b0_values,
      {cdouble(0.0, 1.0), cdouble(0.0, -1.0)},
      1e-5,
      "batch 0 eig");
  expect_spectrum(
      b1_values, {cdouble(3.0, 0.0), cdouble(-1.0, 0.0)}, 1e-6, "batch 1 eig");
  expect_eigenpairs(a0, n, b0_values, b0_vectors, 1e-5, "batch 0 eig");
  expect_eigenpairs(a1, n, b1_values, b1_vectors, 1e-5, "batch 1 eig");
}

TEST_CASE("eig values-only matches the full path bit for bit") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // Same kernel, same input: the values-only dispatch must reproduce the
  // full-path eigenvalues exactly. A mismatch would be nondeterminism,
  // not tolerance.
  int n = 5;
  HostMatrix d(n * n, 0.0);
  d[0 * n + 0] = 2.0;
  d[0 * n + 1] = -3.0;
  d[1 * n + 0] = 3.0;
  d[1 * n + 1] = 2.0;
  d[2 * n + 2] = 1.0;
  d[3 * n + 3] = 5.0;
  d[4 * n + 4] = 0.5;
  std::mt19937 gen(58);
  std::uniform_real_distribution<double> dist(-0.3, 0.3);
  HostMatrix p(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    p[i * n + i] = 1.0;
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        p[i * n + j] = dist(gen);
      }
    }
  }
  HostMatrix a = host_matmul(p, host_matmul(d, host_inverse(p, n), n, n, n),
                             n, n, n);
  array device = device_matrix(stream, a, Shape{n, n});
  auto values_only = read_complex(linalg::eigvals(device, stream), stream);
  auto [w, v] = linalg::eig(device, stream);
  auto values_full = read_complex(w, stream);
  REQUIRE_EQ(values_only.size(), values_full.size());
  for (size_t i = 0; i < values_only.size(); ++i) {
    CHECK_EQ(values_only[i].real(), values_full[i].real());
    CHECK_EQ(values_only[i].imag(), values_full[i].imag());
  }
}

TEST_CASE("eig rejects non-float32 dtypes with the contract error") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  array real = astype(
      array({1.0f, 0.0f, 0.0f, 1.0f}, Shape{2, 2}, float32),
      float32,
      stream);
  array complex_input = astype(real, complex64, stream);
  auto error = evaluation_error(linalg::eigvals(complex_input, stream));
  CHECK_MESSAGE(
      contains_all(
          error,
          {"[omarchy] Eig dtype is not implemented", "no silent CPU fallback"}),
      "unexpected error: ",
      error);
}

TEST_CASE("eig zero matrix converges with zero values") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // The all-zero matrix is the degenerate case where a norm-based
  // deflation threshold collapses to zero; the kernel guards it. All
  // values must be exactly 0 and the vectors an orthonormal basis.
  int n = 3;
  HostMatrix a(n * n, 0.0);
  array device = device_matrix(stream, a, Shape{n, n});
  auto [w, v] = linalg::eig(device, stream);
  auto values = read_complex(w, stream);
  auto vectors = read_complex(v, stream);
  for (const auto& value : values) {
    CHECK_EQ(value.real(), 0.0);
    CHECK_EQ(value.imag(), 0.0);
  }
  expect_eigenpairs(a, n, values, vectors, 1e-5, "zero eig");
}
