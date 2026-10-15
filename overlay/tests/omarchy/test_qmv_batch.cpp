// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Batch-shared bf16 decode GEMV (QmmVecQ4WordSubgroupBatch4BF16, on by default;
// MLX_OMARCHY_QMV_BATCH=0 disables it). A [B, 1, K] decode batch (B <= 4) against one 2D
// 4-bit weight reads each weight word once per block and runs every row's
// single-row chain on it, so each row must equal the per-row kernel's output
// bit for bit. The batch-shared token kernel (QMM_VEC_TOKENS) passed fixed-seed
// checks and then failed per-row identity on AGX in 38% of 5,400 random
// datasets (receipts/2026-10-05-midm-forward), so this compares many random
// datasets per cell, on both shader loops (K % 512 == 0 with N % 8 == 0 takes
// the two-word walk; the others the one-word walk), and proves through the
// dispatch trace that the batch kernel really ran.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

constexpr int kDatasets = 24;

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool have_gpu() {
  if (!gpu::is_available()) {
    std::cout << "Skipping: no qualifying Vulkan device\n";
    return false;
  }
  return true;
}

struct Weight {
  array w;
  array scales;
  array biases;
};

Weight make_weight(int n, int k, Stream s) {
  array w = multiply(
      random::normal(Shape{n, k}, float32, std::nullopt, s), array(0.05f), s);
  auto parts = quantize(w, 64, 4, "affine", std::nullopt, s);
  Weight q{parts[0], astype(parts[1], bfloat16, s), astype(parts[2], bfloat16, s)};
  eval({q.w, q.scales, q.biases});
  return q;
}

array gemv(const array& x, const Weight& q, bool batched, Stream s) {
  setenv("MLX_OMARCHY_QMV_BATCH", batched ? "1" : "0", 1);
  array out = quantized_matmul(
      x, q.w, q.scales, q.biases, true, 64, 4, "affine", s);
  out.eval();
  omarchy::get_command_encoder(s).synchronize();
  unsetenv("MLX_OMARCHY_QMV_BATCH");
  return out;
}

// Kernel ids dispatched while `run` evaluates, from the encoder's
// MLX_OMARCHY_TRACE_DISPATCH lines on stderr.
std::vector<int> traced_kernels(const std::function<void()>& run) {
  std::fflush(stderr);
  int saved = dup(2);
  std::FILE* sink = std::tmpfile();
  dup2(fileno(sink), 2);
  setenv("MLX_OMARCHY_TRACE_DISPATCH", "1", 1);
  run();
  unsetenv("MLX_OMARCHY_TRACE_DISPATCH");
  std::fflush(stderr);
  dup2(saved, 2);
  close(saved);
  std::rewind(sink);
  std::vector<int> kernels;
  char line[512];
  while (std::fgets(line, sizeof(line), sink)) {
    int id = -1;
    if (std::sscanf(line, "[rtmod] DISPATCH kernel=%d", &id) == 1) {
      kernels.push_back(id);
    }
  }
  std::fclose(sink);
  return kernels;
}

bool ran(const std::vector<int>& kernels, omarchy::ComputeKernel kernel) {
  for (int id : kernels) {
    if (id == static_cast<int>(kernel)) {
      return true;
    }
  }
  return false;
}

} // namespace

TEST_CASE("batch-shared bf16 GEMV rows match the per-row kernel bit for bit") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  struct Cell {
    int k;
    int n;
  };
  for (Cell cell : {Cell{2048, 6144}, Cell{6144, 2048}, Cell{2048, 520},
                    Cell{448, 130}}) {
    for (int b : {2, 3, 4}) {
      CAPTURE(cell.k);
      CAPTURE(cell.n);
      CAPTURE(b);
      int failing = 0;
      for (int seed = 0; seed < kDatasets; ++seed) {
        random::seed(1000003u * cell.k + 101u * cell.n + 17u * b + seed);
        Weight q = make_weight(cell.n, cell.k, s);
        array x = astype(
            random::normal(Shape{b, 1, cell.k}, float32, std::nullopt, s),
            bfloat16, s);
        x.eval();
        array want = gemv(x, q, false, s);
        array got = gemv(x, q, true, s);
        if (!array_equal(got, want, s).item<bool>()) {
          ++failing;
          int diff = sum(astype(not_equal(got, want, s), int32, s), s).item<int>();
          std::cout << "[qmv_batch] mismatch k=" << cell.k << " n=" << cell.n
                    << " b=" << b << " seed=" << seed << " elements=" << diff
                    << "\n";
        }
      }
      CHECK_MESSAGE(failing == 0, failing, " of ", kDatasets,
                    " datasets differ from the per-row kernel");
    }
  }
}

TEST_CASE("the batch-shared GEMV engages only where it is enabled and B <= 4") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  random::seed(7);
  Weight q = make_weight(2048, 2048, s);
  auto x_rows = [&](int b) {
    array x = astype(
        random::normal(Shape{b, 1, 2048}, float32, std::nullopt, s), bfloat16, s);
    x.eval();
    return x;
  };
  array x4 = x_rows(4);
  array x5 = x_rows(5);
  const auto batch4 = omarchy::ComputeKernel::QmmVecQ4WordSubgroupBatch4BF16;
  if (!ran(traced_kernels([&] { gemv(x4, q, false, s); }),
           omarchy::ComputeKernel::QmmVecQ4WordSubgroupBF16)) {
    std::cout << "Skipping: the bf16 subgroup GEMV route does not engage here\n";
    return;
  }
  CHECK(ran(traced_kernels([&] { gemv(x4, q, true, s); }), batch4));
  CHECK_FALSE(ran(traced_kernels([&] { gemv(x4, q, false, s); }), batch4));
  CHECK_FALSE(ran(traced_kernels([&] { gemv(x5, q, true, s); }), batch4));
}

// ---------------------------------------------------------------------------
// T-row GEMV tests (design A, speculative verify): x of shape [1, T, K], 2 <= T <= 8, one 2D 4-bit weight.
//
// What they decide, and what they do not (docs/numerics-gate.md): the gate judges a verify route by its error against an independent
// reference, not by bit identity with the M = 1 kernel. R1 (row independence, finite, repeatable) is a necessary interface. R2 prints
// the per-op error of three routes against a host fp64 reference built from the SAME packed words, scales and biases the kernels read:
//   (a) the M = 1 kernel, one row at a time;
//   (b) the deployed composed route for T >= 2: the same [1, T, K] call with MLX_OMARCHY_QMV_BATCH=0, the kill switch of any T-row route;
//   (c) the [1, T, K] call with default routing (today identical to (b); with a T-row kernel it is that kernel).
// The gate comparator is (b), the deployed op; (a) is the plain-decode control. Bit equality of (c) with (a) is printed as information only.
// MLX_OMARCHY_ROWERR_ENFORCE=1 turns "(c) not worse than (b)" into CHECKs; by default it is printed, because the pass rule is not decided here
// and no tolerance is added. The test does NOT cover token choice against an independent reference (OPEN: none exists for the 9B or 27B).
// The engagement test needs the T-row kernel and is compiled only with -DOMARCHY_QMV_BATCH8 (no kernel exists yet).

namespace {

constexpr int kRowDatasets = 6;
constexpr int kRowDatasetsLarge = 2;

struct RowCell {
  int k;
  int n;
  bool large;
};

// K x N = input x output features. The first four are the existing odd and even cells; the rest are the 9B and 27B linear shapes
// (Qwen3.5-9B hidden 4096, intermediate 12288; Qwen3.8-27B hidden 5120, intermediate 17408, qkv 10240, z 6144).
const RowCell kRowCells[] = {
    {2048, 6144, false}, {6144, 2048, false}, {2048, 520, false}, {448, 130, false},
    {4096, 12288, true}, {12288, 4096, true}, {5120, 17408, true}, {17408, 5120, true},
    {5120, 10240, true}, {5120, 6144, true}};

std::vector<float> to_f32(array a, Stream s) {
  a = astype(a, float32, s);
  a.eval();
  omarchy::get_command_encoder(s).synchronize();
  const float* d = a.data<float>();
  return std::vector<float>(d, d + a.size());
}

array qmm_rows(const array& x, const Weight& q, bool kill_batch, Stream s) {
  if (kill_batch) {
    setenv("MLX_OMARCHY_QMV_BATCH", "0", 1);
  } else {
    unsetenv("MLX_OMARCHY_QMV_BATCH");
  }
  array out = quantized_matmul(x, q.w, q.scales, q.biases, true, 64, 4, "affine", s);
  out.eval();
  omarchy::get_command_encoder(s).synchronize();
  unsetenv("MLX_OMARCHY_QMV_BATCH");
  return out;
}

// y[t, n] = sum_k x[t, k] * dequant(W)[n, k] in double, from the packed words (little-endian 4-bit codes, one scale and one bias per
// group of 64 along K), the same layout host_affine_dequantize in test_primitives.cpp checks against mx.dequantize.
std::vector<double> host_reference(const Weight& q, const array& x_rows, int n, int k, Stream s) {
  array w = q.w;
  w.eval();
  omarchy::get_command_encoder(s).synchronize();
  const uint32_t* words = w.data<uint32_t>();
  std::vector<float> scales = to_f32(q.scales, s);
  std::vector<float> biases = to_f32(q.biases, s);
  std::vector<float> xs = to_f32(x_rows, s);
  const int t = x_rows.shape(-2);
  const int groups = k / 64;
  const size_t words_per_row = static_cast<size_t>(k) * 4 / 32;
  std::vector<double> y(static_cast<size_t>(t) * n, 0.0);
  std::vector<double> dq(k);
  for (int col = 0; col < n; ++col) {
    for (int e = 0; e < k; ++e) {
      size_t bit = static_cast<size_t>(col) * words_per_row * 32 + static_cast<size_t>(e) * 4;
      uint32_t code = (words[bit / 32] >> (bit % 32)) & 0xFu;
      size_t g = static_cast<size_t>(col) * groups + e / 64;
      dq[e] = static_cast<double>(code) * scales[g] + biases[g];
    }
    for (int r = 0; r < t; ++r) {
      const float* xr = xs.data() + static_cast<size_t>(r) * k;
      double acc = 0.0;
      for (int e = 0; e < k; ++e) {
        acc += static_cast<double>(xr[e]) * dq[e];
      }
      y[static_cast<size_t>(r) * n + col] = acc;
    }
  }
  return y;
}

// bf16 spacing at the magnitude of v: 2^(floor(log2|v|) - 7), the unit the gate uses for near ties.
double bf16_step(double v) {
  v = std::fabs(v);
  return v == 0.0 ? 0.0 : std::ldexp(1.0, static_cast<int>(std::floor(std::log2(v))) - 7);
}

struct RowErr {
  double mean_abs = 0.0;
  double max_abs = 0.0;
  double max_steps = 0.0;
  bool finite = true;
};

RowErr row_error(const std::vector<float>& out, const std::vector<double>& ref) {
  RowErr e;
  double sum = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (!std::isfinite(out[i])) {
      e.finite = false;
      continue;
    }
    double d = std::fabs(static_cast<double>(out[i]) - ref[i]);
    sum += d;
    e.max_abs = std::max(e.max_abs, d);
    double step = bf16_step(ref[i]);
    if (step > 0.0) {
      e.max_steps = std::max(e.max_steps, d / step);
    }
  }
  e.mean_abs = sum / static_cast<double>(ref.size());
  return e;
}

bool rows_equal(const array& a, const array& b, int row, int n, Stream s) {
  return array_equal(slice(a, {0, row, 0}, {1, row + 1, n}, s), slice(b, {0, row, 0}, {1, row + 1, n}, s), s).item<bool>();
}

int rows_datasets(const RowCell& cell) {
  const char* env = std::getenv("MLX_OMARCHY_ROWTEST_DATASETS");
  if (env != nullptr && std::atoi(env) > 0) {
    return std::atoi(env);
  }
  return cell.large ? kRowDatasetsLarge : kRowDatasets;
}

// Default: the four small cells only, so the standing test battery stays fast. MLX_OMARCHY_ROWTEST_LARGE=1 adds the 9B and 27B shapes
// (the host reference is minutes of host time there); MLX_OMARCHY_ROWTEST_CELLS="5120x17408 4096x12288" selects cells by name.
bool cell_selected(const RowCell& cell) {
  const char* env = std::getenv("MLX_OMARCHY_ROWTEST_CELLS");
  if (env != nullptr && env[0] != '\0') {
    return std::string(env).find(std::to_string(cell.k) + "x" + std::to_string(cell.n)) != std::string::npos;
  }
  if (cell.large) {
    const char* big = std::getenv("MLX_OMARCHY_ROWTEST_LARGE");
    return big != nullptr && big[0] == '1';
  }
  return true;
}

} // namespace

TEST_CASE("T-row GEMV: row i depends only on input row i, outputs are finite and repeatable (R1)") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  for (const RowCell& cell : kRowCells) {
    if (!cell_selected(cell)) continue;
    for (int t = 2; t <= 8; ++t) {
      CAPTURE(cell.k);
      CAPTURE(cell.n);
      CAPTURE(t);
      int failing = 0;
      const int datasets = rows_datasets(cell);
      for (int seed = 0; seed < datasets; ++seed) {
        random::seed(2000003u * cell.k + 211u * cell.n + 31u * t + seed);
        Weight q = make_weight(cell.n, cell.k, s);
        array x = astype(random::normal(Shape{1, t, cell.k}, float32, std::nullopt, s), bfloat16, s);
        x.eval();
        array base = qmm_rows(x, q, false, s);
        array again = qmm_rows(x, q, false, s);
        bool ok = array_equal(base, again, s).item<bool>();
        ok = ok && all(isfinite(base, s), s).item<bool>();
        array others[2] = {
            astype(random::normal(Shape{1, t, cell.k}, float32, std::nullopt, s), bfloat16, s),
            astype(zeros(Shape{1, t, cell.k}, float32, s), bfloat16, s)};
        for (int i = 0; i < t && ok; ++i) {
          array pick = reshape(equal(arange(t, int32, s), array(i), s), Shape{1, t, 1}, s);
          for (const array& other : others) {
            array mixed = where(pick, x, other, s);
            mixed.eval();
            array out = qmm_rows(mixed, q, false, s);
            if (!rows_equal(out, base, i, cell.n, s)) {
              ok = false;
              std::cout << "[qmv_rows] row " << i << " changed with the other rows: k=" << cell.k << " n=" << cell.n
                        << " t=" << t << " seed=" << seed << "\n";
              break;
            }
          }
        }
        failing += ok ? 0 : 1;
      }
      CHECK_MESSAGE(failing == 0, failing, " of ", datasets, " datasets broke row independence, finiteness or repeatability");
    }
  }
}

TEST_CASE("T-row GEMV: per-op error against the fp64 host reference for the M=1, composed and default routes (R2)") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  const bool enforce = std::getenv("MLX_OMARCHY_ROWERR_ENFORCE") != nullptr;
  for (const RowCell& cell : kRowCells) {
    if (!cell_selected(cell)) continue;
    for (int t : {2, 5, 8}) {
      CAPTURE(cell.k);
      CAPTURE(cell.n);
      CAPTURE(t);
      const int datasets = rows_datasets(cell);
      double mean_a = 0.0, mean_b = 0.0, mean_c = 0.0, max_steps_c = 0.0;
      int c_le_b = 0, c_eq_a = 0, c_eq_b = 0;
      bool all_finite = true;
      for (int seed = 0; seed < datasets; ++seed) {
        random::seed(3000017u * cell.k + 223u * cell.n + 37u * t + seed);
        Weight q = make_weight(cell.n, cell.k, s);
        array x = astype(random::normal(Shape{1, t, cell.k}, float32, std::nullopt, s), bfloat16, s);
        x.eval();
        std::vector<double> ref = host_reference(q, reshape(x, Shape{t, cell.k}, s), cell.n, cell.k, s);
        // (a) one row at a time through the M = 1 kernel
        std::vector<float> out_a(static_cast<size_t>(t) * cell.n);
        std::vector<array> rows_a;
        for (int r = 0; r < t; ++r) {
          array xr = contiguous(slice(x, {0, r, 0}, {1, r + 1, cell.k}, s), false, s);
          xr.eval();
          array o = qmm_rows(xr, q, true, s);
          rows_a.push_back(o);
          std::vector<float> h = to_f32(o, s);
          std::copy(h.begin(), h.end(), out_a.begin() + static_cast<size_t>(r) * cell.n);
        }
        array b = qmm_rows(x, q, true, s);   // (b) composed, kill switch on
        array c = qmm_rows(x, q, false, s);  // (c) default routing
        std::vector<float> out_b = to_f32(b, s);
        std::vector<float> out_c = to_f32(c, s);
        RowErr ea = row_error(out_a, ref), eb = row_error(out_b, ref), ec = row_error(out_c, ref);
        all_finite = all_finite && ea.finite && eb.finite && ec.finite;
        mean_a += ea.mean_abs / datasets;
        mean_b += eb.mean_abs / datasets;
        mean_c += ec.mean_abs / datasets;
        max_steps_c = std::max(max_steps_c, ec.max_steps);
        c_le_b += ec.mean_abs <= eb.mean_abs ? 1 : 0;
        c_eq_b += array_equal(b, c, s).item<bool>() ? 1 : 0;
        bool eq_a = true;
        for (int r = 0; r < t; ++r) {
          eq_a = eq_a && array_equal(slice(c, {0, r, 0}, {1, r + 1, cell.n}, s), rows_a[r], s).item<bool>();
        }
        c_eq_a += eq_a ? 1 : 0;
      }
      std::printf(
          "[qmv_rows] R2 k=%d n=%d t=%d datasets=%d | mean abs err a(M=1)=%.4e b(composed)=%.4e c(default)=%.4e | c max err %.2f bf16 steps | "
          "c<=b %d/%d | %s | INFO c==b bits %d/%d (equal means the T-row route did not engage), c==a bits %d/%d\n",
          cell.k, cell.n, t, datasets, mean_a, mean_b, mean_c, max_steps_c, c_le_b, datasets,
          c_le_b == datasets ? "c NOT WORSE than b" : "c WORSE than b on at least one dataset", c_eq_b, datasets, c_eq_a, datasets);
      CHECK_MESSAGE(all_finite, "a route produced a non-finite output");
      if (enforce) {
        CHECK_MESSAGE(c_le_b == datasets, "default route has a larger mean abs error than the composed route on some dataset");
      }
    }
  }
}

#ifdef OMARCHY_QMV_BATCH8
TEST_CASE("the T-row GEMV engages only for 2 <= T <= 8 on one sequence, and only when enabled") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  random::seed(11);
  Weight q = make_weight(2048, 2048, s);
  const auto batch8 = omarchy::ComputeKernel::QmmVecQ4WordSubgroupBatch8BF16;
  auto x_rows = [&](int t) {
    array x = astype(random::normal(Shape{1, t, 2048}, float32, std::nullopt, s), bfloat16, s);
    x.eval();
    return x;
  };
  for (int t : {2, 5, 8}) {
    array x = x_rows(t);
    CHECK(ran(traced_kernels([&] { qmm_rows(x, q, false, s); }), batch8));
    CHECK_FALSE(ran(traced_kernels([&] { qmm_rows(x, q, true, s); }), batch8));  // kill switch
  }
  array x9 = x_rows(9);
  CHECK_FALSE(ran(traced_kernels([&] { qmm_rows(x9, q, false, s); }), batch8));  // T = 9 is outside the route
  array x1 = x_rows(1);
  CHECK_FALSE(ran(traced_kernels([&] { qmm_rows(x1, q, false, s); }), batch8));  // T = 1 is the M = 1 kernel
}
#endif
