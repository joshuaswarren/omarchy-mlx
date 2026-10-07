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
