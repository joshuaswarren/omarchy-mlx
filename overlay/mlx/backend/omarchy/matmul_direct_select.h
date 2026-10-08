// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Shape- and chip-keyed choice among builds of
// shaders/matmul_coopmat_direct.comp (K_UNROLL, SWIZZLE_ROWS, WIDE_N).
// A row holds a measured winner: the chip (a device_name substring such
// as "G13G"), dtype, orientation, the smallest m and n it won at, and
// the pipeline plus grid width to run. The first matching row wins;
// anything unmatched keeps the shipped route. A route also needs
// n >= tile_n / 2: a subgroup's edge tile shifts back to n - tile_n / 2
// (64 for WIDE_N), so a narrower n would underflow. Add a row only with
// the receipt that measured it.

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

#include "mlx/backend/omarchy/compute.h"
#include "mlx/dtype.h"

namespace mlx::core::omarchy {

struct DirectMatmulRoute {
  ComputeKernel kernel;
  uint32_t tile_n;  // output columns per workgroup (64; WIDE_N builds 128)
};

struct DirectMatmulRow {
  std::string_view chip;
  Dtype dtype;
  bool a_transposed;
  bool b_transposed;
  uint32_t min_m;
  uint32_t min_n;
  DirectMatmulRoute route;
};

// Measured winners: receipts/2026-10-07-direct-gemm-variants (MatmulGap H8,
// f16), receipts/2026-10-07-bf16-direct-gemm (MatmulGap H13, H14: bf16 and
// f32), receipts/2026-10-08-direct-gemm-floors (M2 Max rows, lower m floors)
// and receipts/2026-10-08-g13g-f32-matmul-row-removal (the G13G f32 a @ b row
// removed). A bf16 a @ b.T takes the direct route only through a row here.
inline constexpr std::array<DirectMatmulRow, 14> kDirectMatmulRows{{
    {"G13C", float16, false, true, 512u, 4096u,
     {ComputeKernel::MatmulDirectF16NtK4S8, 64u}},
    {"G13C", float16, true, false, 4096u, 4096u,
     {ComputeKernel::MatmulDirectF16TnWS8, 128u}},
    {"G13G", float16, false, true, 512u, 4096u,
     {ComputeKernel::MatmulDirectF16NtK4S8, 64u}},
    {"G13G", float16, true, false, 1024u, 4096u,
     {ComputeKernel::MatmulDirectF16TnWS8, 128u}},
    {"G13C", bfloat16, true, false, 4096u, 4096u,
     {ComputeKernel::MatmulDirectBF16TnWS8, 128u}},
    {"G13G", bfloat16, true, false, 512u, 4096u,
     {ComputeKernel::MatmulDirectBF16TnWS8, 128u}},
    {"G13C", float32, false, true, 4096u, 4096u,
     {ComputeKernel::MatmulDirectF32NtK4S8, 64u}},
    {"G13G", float32, false, true, 512u, 4096u,
     {ComputeKernel::MatmulDirectF32NtK4S8, 64u}},
    // Candidate C2 (MatmulGap H15, H16): per-chip rows.
    {"G13C", bfloat16, false, true, 512u, 4096u,
     {ComputeKernel::MatmulDirectBF16NtK4S8, 64u}},
    {"G13G", bfloat16, false, true, 4096u, 4096u,
     {ComputeKernel::MatmulDirectBF16Nt, 64u}},
    {"G13G", float32, true, false, 512u, 4096u,
     {ComputeKernel::MatmulDirectF32TnWS8, 128u}},
    // M2 Max (MatmulGap H20): bf16 and f16 rows with the pipelines above.
    {"G14C", bfloat16, true, false, 4096u, 4096u,
     {ComputeKernel::MatmulDirectBF16TnWS8, 128u}},
    {"G14C", bfloat16, false, true, 512u, 4096u,
     {ComputeKernel::MatmulDirectBF16NtK4S8, 64u}},
    {"G14C", float16, false, true, 1024u, 4096u,
     {ComputeKernel::MatmulDirectF16NtK2S8, 64u}},
}};

inline DirectMatmulRoute select_direct_matmul_route(
    std::span<const DirectMatmulRow> rows,
    std::string_view device_name,
    Dtype dtype,
    bool a_transposed,
    bool b_transposed,
    uint32_t m,
    uint32_t n,
    DirectMatmulRoute shipped) {
  for (const auto& row : rows) {
    if (device_name.find(row.chip) != std::string_view::npos &&
        row.dtype == dtype && row.a_transposed == a_transposed &&
        row.b_transposed == b_transposed && m >= row.min_m &&
        n >= row.min_n && n >= row.route.tile_n / 2u) {
      return row.route;
    }
  }
  return shipped;
}

} // namespace mlx::core::omarchy
