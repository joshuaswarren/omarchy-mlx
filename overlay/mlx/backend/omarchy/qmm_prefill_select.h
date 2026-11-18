// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Pure row-tile selection for the cooperative-matrix qmm prefill
// (shaders/qmm_prefill_coopmat.comp and its 16/32/64-row builds).
// Moved here from primitives.cpp so tests can pin the 64-row gate: the
// 64-row default is a 4-bit group-64 transposed FullN route only
// (review of #508); every other format and non-FullN shape keeps the
// 32-row tile, and parts outside G16S never take 64 rows.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace mlx::core::omarchy {

// GPU core count for the parts this backend has measured. Vulkan
// reports no core count, Device::capabilities() carries none and
// mx.device_info() exposes none, so the width has to come from the
// device name the driver reports. Only names this project has run a
// per-shape occupancy census on are listed; anything else returns 0,
// which keeps every dispatch on the shape it has today. Match is
// exact, because the name does not distinguish a 24-core M1 Max from a
// 32-core one and guessing high would step tiles down on a part that
// never showed it needs them.
// The M2 Max name ("Apple M2 Max (G14C B1)") does not distinguish the
// 30-core from the 38-core part either; our fleet M2 is 38-core
// (device_info.cpp maps apple,t6021 -> 38 from the device tree). If a
// 30-core G14C appears it will also match this entry: the occupancy
// step-down is conservative on a smaller part (fewer cores means the
// grid clears the per-core floor sooner, so the wider tile wins), which
// is the safe direction.
inline uint32_t apple_gpu_cores(const std::string& device_name) {
  if (device_name == "Apple M1 (G13G B1)") {
    return 8u;
  }
  if (device_name == "Apple M1 Max (G13C C0)") {
    return 32u;
  }
  if (device_name == "Apple M2 Max (G14C B1)") {
    return 38u;
  }
  return 0u;
}

// Occupancy floor for the coopmat prefill tile, in workgroups per GPU
// core: below it the 16-row twin is dispatched instead of the shipped
// 32-row one. Both sides of this number are measured on the M1 Max at
// m=1053, k=896, 32-rows against 16-rows per shape
// (receipts/2026-09-14-qmm-occupancy-tilem/):
//
//   n    workgroups  wg/core   32 rows   16 rows
//   128         132      4.1   213.6us   165.7us   -22.4%
//   256         264      8.2   225.7us   300.7us   +33.3%
//   384         396     12.4   249.7us   434.8us   +74.1%
//   896         924     28.9   509.2us   759.5us   +49.2%
//
// So the trade turns over between 4.1 and 8.2, and 6 is the midpoint
// that separates the two measured grids (6 x 32 = 192, strictly between
// 132 and 264). It is deliberately not larger: halving the rows halves
// how many outputs share one staged 16x32 weight tile, and past the
// crossover that reuse loss dominates by tens of percent. On the base
// M1 the same floor is 48 workgroups, which every real prefill grid
// clears, so the base M1 keeps the shipped tile untouched.
// MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE overrides it for A/B; 0 disables
// the step-down and restores the shipped pick exactly.
constexpr uint32_t kCoopmatWorkgroupsPerCore = 6u;

inline uint32_t coopmat_workgroups_per_core() {
  static const uint32_t value = [] {
    const char* env = std::getenv("MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE");
    if (env == nullptr) {
      return kCoopmatWorkgroupsPerCore;
    }
    char* end = nullptr;
    unsigned long parsed = std::strtoul(env, &end, 10);
    // A malformed value keeps the compiled-in floor; a huge one is
    // clamped rather than quietly reinterpreted as the default, because
    // "as small a tile as possible" is a thing an A/B run asks for.
    if (end == env || *end != '\0') {
      return kCoopmatWorkgroupsPerCore;
    }
    return static_cast<uint32_t>(
        std::min<unsigned long>(parsed, 0xfffffffful));
  }();
  return value;
}

// Output rows per coopmat workgroup: the shipped 32, or the 16-row twin
// when the shape-derived grid cannot fill the part. Halving the rows
// doubles the grid and halves weight-tile reuse; 16 is the floor
// because an 8-row, 32-lane variant was built and measured and lost
// even at 4.1 workgroups per core (58% slower than 32 rows), so there
// is nothing below 16 worth dispatching. Both row counts keep one
// output's k chain identical, so the pick does not move generated ids.
//
// A prompt of at most 16 rows fills one 16-row tile: the 32-row tile
// would pad it to 32 (a 13-token prompt runs 19 dead rows through
// every mma and A load) while the grid stays one m-group either way,
// so nothing is traded for the halved per-step work. Contract prompts
// are 11-18 tokens; on T8103 the 13-token prefill GEMMs at 32 rows
// cost 400-950 us each, 133 of them per first token.
inline bool qmm_coopmat_rows64_requested(const std::string& device_name) {
  const char* env = std::getenv("MLX_OMARCHY_QMM_COOPMAT_ROWS");
  if (env != nullptr && env[0] != '\0') {
    const std::string v(env);
    if (v == "32" || v == "0" || v == "16") {
      return false;
    }
    if (v == "64" || v == "64sg4") {
      return true;
    }
  }
  // Default on G16S: 64-row G4+LdsPad, measured +6.6% pf2060
  // prefill with bit-identical tokens (kill switch: =32).
  return device_name.find("G16S") != std::string::npos;
}

inline bool qmm_coopmat_sg4_env() {
  const char* env = std::getenv("MLX_OMARCHY_QMM_COOPMAT_ROWS");
  return env != nullptr && std::string(env) == "64sg4";
}

inline uint32_t qmm_coopmat_stepk_env() {
  const char* env = std::getenv("MLX_OMARCHY_QMM_COOPMAT_STEPK");
  if (env != nullptr && std::string(env) == "32") {
    return 32u;
  }
  return 16u;
}

inline bool qmm_cheap_deq_env() {
  const char* env = std::getenv("MLX_OMARCHY_QMM_CHEAP_DEQ");
  return env != nullptr && env[0] == '1';
}

// Landed G4+LdsPad default, or an env-gated dequant-ratio twin of it.
// The per-output k chain is unchanged; only staging granularity, the
// number of rows sharing one W tile, or the nibble->f32 conversion
// changes.
inline omarchy::ComputeKernel qmm_g4_pad_ratio_kernel(
    uint32_t rows, bool sg4, uint32_t step_k, bool cheap) {
  using K = omarchy::ComputeKernel;
  if (rows == 64u && sg4) {
    if (step_k == 32u) {
      return cheap ? K::QmmPrefillCoopmatSG4BF16X32FullNLdsPadK32Cheap
                   : K::QmmPrefillCoopmatSG4BF16X32FullNLdsPadK32;
    }
    return cheap ? K::QmmPrefillCoopmatSG4BF16X32FullNLdsPadCheap
                 : K::QmmPrefillCoopmatSG4BF16X32FullNLdsPad;
  }
  if (rows == 64u) {
    if (step_k == 32u) {
      return cheap ? K::QmmPrefillCoopmatM64BF16X32FullNLdsPadK32Cheap
                   : K::QmmPrefillCoopmatM64BF16X32FullNLdsPadK32;
    }
    return cheap ? K::QmmPrefillCoopmatM64BF16X32FullNLdsPadCheap
                 : K::QmmPrefillCoopmatM64BF16X32FullNLdsPad;
  }
  if (step_k == 32u) {
    return cheap ? K::QmmPrefillCoopmatBF16X32FullNLdsPadK32Cheap
                 : K::QmmPrefillCoopmatBF16X32FullNLdsPadK32;
  }
  return cheap ? K::QmmPrefillCoopmatBF16X32FullNLdsPadCheap
               : K::QmmPrefillCoopmatBF16X32FullNLdsPad;
}

inline uint32_t coopmat_tile_rows(
    uint32_t matrix_m,
    uint32_t n_groups,
    const std::string& device_name,
    bool allow_rows64 = false) {
  if (matrix_m <= 16u) {
    return 16u;
  }
  uint32_t cores = apple_gpu_cores(device_name);
  uint32_t per_core = coopmat_workgroups_per_core();
  // 64-row experiment: take it only when M is large enough that the
  // 64-row grid still fills the part. G16S is not in apple_gpu_cores
  // (returns 0) and would otherwise be stuck at 32; the env is still
  // honored there for M >= 128.
  if (allow_rows64 && qmm_coopmat_rows64_requested(device_name) &&
      matrix_m >= 128u) {
    uint32_t m_groups_64 = (matrix_m + 63u) / 64u;
    if (cores == 0u || per_core == 0u ||
        static_cast<uint64_t>(m_groups_64) * n_groups >=
            static_cast<uint64_t>(cores) * per_core) {
      return 64u;
    }
  }
  if (cores == 0u || per_core == 0u) {
    return 32u;
  }
  uint64_t target = static_cast<uint64_t>(cores) * per_core;
  uint32_t m_groups_32 = (matrix_m + 31u) / 32u;
  return static_cast<uint64_t>(m_groups_32) * n_groups < target ? 16u : 32u;
}

// The 64-row tile needs both the format that measured it (4-bit
// group-64 transposed) and a FullN shape; anything else reaching the
// route table with 64 rows would fall through to the 32-row X32FullN
// build with half the m-groups.
inline bool qmm_prefill_allow_rows64(bool q4_g64_transpose, bool full_n) {
  return q4_g64_transpose && full_n;
}

} // namespace mlx::core::omarchy
