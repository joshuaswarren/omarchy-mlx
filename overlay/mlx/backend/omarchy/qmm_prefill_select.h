// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Pure row-tile and kernel-variant selection for the cooperative-matrix qmm
// prefill (shaders/qmm_prefill_coopmat.comp and its 16/32/64-row builds).
// Moved here from primitives.cpp so tests can pin the 64-row gate: the
// 64-row default is a 4-bit group-64 transposed FullN route only
// (review of #508); every other format and non-FullN shape keeps the
// 32-row tile. G16S defaults to 64-row G4+LdsPad; G13C, G14C and G13G
// default to 128-row SG8 K=64 where the grid fills the part, stepping
// down to 64-row SG4 (G13G at STEP_K=32); other parts keep 32 rows.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include "mlx/backend/omarchy/device.h"

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
  if (const char* env = omarchy::env_str("MLX_OMARCHY_GPU_CORES")) {
    return static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
  }
  if (device_name == "Apple M1 (G13G B1)") {
    return 8u;
  }
  if (device_name == "Apple M1 Max (G13C C0)") {
    return 32u;
  }
  if (device_name == "Apple M2 Max (G14C B1)") {
    return 38u;
  }
  if (device_name == "Apple M4 Pro (G16S B1)") {
    // M4 Pro J616s/T6040, 20-core GPU (docs/m4-linux.md; the G401 audit
    // names its small-M prefill arm). The name does not distinguish the
    // binned 16-core part; a smaller real part clears the floor sooner
    // than a 20-core guess assumes, so the step-down fires less often
    // than ideal, never more - the conservative direction this table
    // already accepts for G14C. Without a count here the G16S grid
    // never compares against the floor at all: the small-M arm of the
    // route was unreachable and the 64-row fill gate was vacuous.
    return 20u;
  }
  if (device_name.find("G17C") != std::string::npos) {
    // M5 Max: initial value matching G14C (same Max class) pending calibration on M5
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
    const char* env = omarchy::env_str("MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE");
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
inline bool qmm_coopmat_rows128_requested(const std::string& device_name) {
  const char* env = omarchy::env_str("MLX_OMARCHY_QMM_COOPMAT_ROWS");
  if (env != nullptr && env[0] != '\0') {
    const std::string v(env);
    if (v == "128sg8" || v == "128") {
      return true;
    }
    if (v == "64" || v == "64sg4" || v == "32" || v == "0" || v == "16") {
      return false;
    }
  }
  // G13C (M1 Max), G14C (M2 Max) and G13G (13-inch M1) default to 128-row
  // SG8 coalesced-W K=64 where the grid fills the part, stepping down to
  // 64-row SG4 K=64 for smaller M/N. G13G measured: prefill512 2.05-2.15x
  // vs the 64-row SG4 default (9B 101.6 vs 49.5 tok/s, 4B 210.3 vs 98.0,
  // ids identical, 5-pair ABBA on main 1f854b9b85).
  return device_name.find("G13G") != std::string::npos ||
      device_name.find("G13C") != std::string::npos ||
      device_name.find("G14C") != std::string::npos;
}

inline bool qmm_coopmat_rows64_requested(const std::string& device_name) {
  const char* env = omarchy::env_str("MLX_OMARCHY_QMM_COOPMAT_ROWS");
  if (env != nullptr && env[0] != '\0') {
    const std::string v(env);
    if (v == "32" || v == "0" || v == "16") {
      return false;
    }
    if (v == "64" || v == "64sg4" || v == "128sg8" || v == "128") {
      return true;
    }
  }
  // G16S defaults on (64-row G4+LdsPad); G13G defaults on (64-row
  // SG4+LdsPad+STEPK=32); G13C and G14C use 64-row SG4+K=64 as their
  // step-down from 128-row SG8. G17C keeps 32-row pending calibration.
  return device_name.find("G16S") != std::string::npos ||
      device_name.find("G13G") != std::string::npos ||
      device_name.find("G13C") != std::string::npos ||
      device_name.find("G14C") != std::string::npos;
}

inline bool qmm_coopmat_sg4_requested(const std::string& device_name) {
  const char* env = omarchy::env_str("MLX_OMARCHY_QMM_COOPMAT_ROWS");
  if (env != nullptr && env[0] != '\0') {
    const std::string v(env);
    if (v == "64sg4" || v == "128sg8" || v == "128") {
      return true;
    }
    if (v == "64" || v == "32" || v == "0" || v == "16") {
      return false;
    }
  }
  return device_name.find("G13G") != std::string::npos ||
      device_name.find("G13C") != std::string::npos ||
      device_name.find("G14C") != std::string::npos;
}

inline uint32_t qmm_coopmat_stepk_requested(
    const std::string& device_name, uint32_t rows, bool sg4) {
  if (rows == 128u) {
    return 64u;
  }
  const char* env = omarchy::env_str("MLX_OMARCHY_QMM_COOPMAT_STEPK");
  if (env != nullptr && env[0] != '\0') {
    const std::string v(env);
    if (v == "64" && rows == 64u && sg4) {
      return 64u;
    }
    if (v == "32") {
      return 32u;
    }
    if (v == "16") {
      return 16u;
    }
  }
  if (rows == 64u && sg4 &&
      (device_name.find("G13C") != std::string::npos ||
       device_name.find("G14C") != std::string::npos)) {
    return 64u;
  }
  // Only 64-row SG4 on G13G defaults to STEP_K=32; standalone 32-row
  // STEP_K=32 regressed on G13G, so fallback 32-row grids keep 16.
  if (rows == 64u && sg4 &&
      device_name.find("G13G") != std::string::npos) {
    return 32u;
  }
  return 16u;
}

inline bool qmm_cheap_deq_env() {
  const char* env = omarchy::env_str("MLX_OMARCHY_QMM_CHEAP_DEQ");
  return env != nullptr && env[0] == '1';
}

// Landed G4+LdsPad default, or an env-gated dequant-ratio twin of it.
// The per-output k chain is unchanged; only staging granularity, the
// number of rows sharing one W tile, or the nibble->f32 conversion
// changes. full_m selects the compile-time tail-free build when
// matrix_m % rows == 0 (keeps GPRs at 103 / 1024 threads per core).
inline omarchy::ComputeKernel qmm_g4_pad_ratio_kernel(
    uint32_t rows,
    bool sg4,
    uint32_t step_k,
    bool cheap,
    bool full_m = false) {
  using K = omarchy::ComputeKernel;
  if (rows == 128u) {
    return full_m ? K::QmmPrefillCoopmatSG8BF16X32FullNLdsPadK64FullM
                  : K::QmmPrefillCoopmatSG8BF16X32FullNLdsPadK64;
  }
  if (rows == 64u && sg4) {
    if (step_k == 64u) {
      return full_m ? K::QmmPrefillCoopmatSG4BF16X32FullNLdsPadK64FullM
                    : K::QmmPrefillCoopmatSG4BF16X32FullNLdsPadK64;
    }
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
    bool allow_rows64 = false,
    bool allow_rows128 = true) {
  if (matrix_m <= 16u) {
    return 16u;
  }
  uint32_t cores = apple_gpu_cores(device_name);
  uint32_t per_core = coopmat_workgroups_per_core();
  // 128-row SG8 K=64 (256 threads/WG): take it when M >= 256 and the
  // 128-row grid provides at least 4 workgroups per core (1024 threads/core).
  if (allow_rows64 && allow_rows128 &&
      qmm_coopmat_rows128_requested(device_name) && matrix_m >= 256u) {
    uint32_t m_groups_128 = (matrix_m + 127u) / 128u;
    uint32_t sg8_per_core = per_core == 0u ? 0u : std::min(per_core, 4u);
    if (cores == 0u || sg8_per_core == 0u ||
        static_cast<uint64_t>(m_groups_128) * n_groups >=
            static_cast<uint64_t>(cores) * sg8_per_core) {
      return 128u;
    }
  }
  // 64-row tile: take it only when M is large enough that the 64-row grid
  // still fills the part (m_groups_64 * n_groups against the part's core
  // count; G16S carries 20). On G13C/G14C (SG4 128-lane), 4 WG/core (512
  // threads/core) suffices.
  if (allow_rows64 && qmm_coopmat_rows64_requested(device_name) &&
      matrix_m >= 128u) {
    uint32_t m_groups_64 = (matrix_m + 63u) / 64u;
    const bool max_class_sg4 =
        device_name.find("G13C") != std::string::npos ||
        device_name.find("G14C") != std::string::npos;
    uint32_t r64_per_core = (per_core != 0u && max_class_sg4)
        ? std::min(per_core, 4u)
        : per_core;
    if (cores == 0u || r64_per_core == 0u ||
        static_cast<uint64_t>(m_groups_64) * n_groups >=
            static_cast<uint64_t>(cores) * r64_per_core) {
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

// CMAT16 (16x16x16 fp16-operand) staging variants, DEC-20. Opt-in
// staging arms of the shipped 32-row K16 kernel on the NVIDIA-class
// route: MLX_OMARCHY_QMM_CMAT16_ROWS=64|64sg4|128|128sg8 selects the
// 64-row SG4 (128-lane) or 128-row SG8 (256-lane) tile, and
// MLX_OMARCHY_QMM_CMAT16_STEPK=32|64 the deeper K dequant staging.
// Every build keeps the same per-output ascending-k f32 chain and the
// same fp16 staged values, so the pick never moves output bits; unset
// or any other value keeps the shipped kernel exactly. Unlike the fp32
// twins there is no per-part default: stock behavior is the 32-row K16
// kernel on every device until a variant measures out.
inline uint32_t qmm_cmat16_rows_requested() {
  const char* env = omarchy::env_str("MLX_OMARCHY_QMM_CMAT16_ROWS");
  if (env != nullptr && env[0] != '\0') {
    const std::string v(env);
    if (v == "64" || v == "64sg4") {
      return 64u;
    }
    if (v == "128" || v == "128sg8") {
      return 128u;
    }
  }
  return 32u;
}

inline uint32_t qmm_cmat16_stepk_requested() {
  const char* env = omarchy::env_str("MLX_OMARCHY_QMM_CMAT16_STEPK");
  if (env != nullptr && env[0] != '\0') {
    const std::string v(env);
    if (v == "32") {
      return 32u;
    }
    if (v == "64") {
      return 64u;
    }
  }
  return 16u;
}

// Workgroup lanes the CMAT16 row tile needs (SG4 = 128, SG8 = 256).
inline uint32_t qmm_cmat16_lanes(uint32_t rows) {
  return rows == 128u ? 256u : rows == 64u ? 128u : 64u;
}

// Shared-memory bytes a CMAT16 build stages: x_s rows*step_k fp16, w_s
// step_k*32 fp16, and the 2 x 16 x 16 fp32 drain tile. The shipped
// 32-row K16 build is 4096 bytes (kQmmCoopmat16SharedBytes).
inline uint32_t qmm_cmat16_shared_bytes(uint32_t rows, uint32_t step_k) {
  return (rows * step_k + step_k * 32u) * 2u + 2u * 16u * 16u * 4u;
}

// The CMAT16 build a (rows, step_k) request maps to; the shader guards
// 64 rows to SG4 and 128 to SG8, so the lane count is implied.
inline omarchy::ComputeKernel qmm_cmat16_kernel(
    uint32_t rows,
    uint32_t step_k) {
  using K = omarchy::ComputeKernel;
  if (rows == 128u) {
    return step_k == 64u ? K::QmmPrefillCoopmat16BF16M128K64
        : step_k == 32u  ? K::QmmPrefillCoopmat16BF16M128K32
                         : K::QmmPrefillCoopmat16BF16M128;
  }
  if (rows == 64u) {
    return step_k == 64u ? K::QmmPrefillCoopmat16BF16M64K64
        : step_k == 32u  ? K::QmmPrefillCoopmat16BF16M64K32
                         : K::QmmPrefillCoopmat16BF16M64;
  }
  return step_k == 64u ? K::QmmPrefillCoopmat16BF16K64
      : step_k == 32u  ? K::QmmPrefillCoopmat16BF16K32
                       : K::QmmPrefillCoopmat16BF16;
}

} // namespace mlx::core::omarchy
