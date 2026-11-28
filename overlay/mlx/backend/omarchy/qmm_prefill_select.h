// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Pure row-tile and kernel-variant selection for the cooperative-matrix qmm
// prefill (shaders/qmm_prefill_coopmat.comp and its 16/32/64-row builds).
// Moved here from primitives.cpp so tests can pin the 64-row gate: the
// 64-row default is a 4-bit group-64 transposed FullN route only
// (review of #508); every other format and non-FullN shape keeps the
// 32-row tile. G16S defaults to 64-row G4+LdsPad; G13C, G14C and G13G
// default to 128-row SG8 K=64 where the grid fills the part, stepping
// down to 64-row SG4 (G13G at STEP_K=32). Unlisted parts follow the
// capability probe (DEC-17): the measured-good tile defaults apply
// where the fp32 8x8x8 shape, 32-lane subgroup, workgroup size and
// shared-memory floor all admit them, so a part that can legally
// dispatch the family takes the same picks the measured parts do.

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
// Cooperative-matrix capability probe for the qmm tile family (DEC-17):
// every kernel the row/variant selectors below pick emits the fp32 8x8x8
// subgroup shape, so a part has to advertise that shape and report the
// 32-lane subgroup the shaders are built for before any tile default can
// apply to it. A part that meets the probe takes the measured-good
// defaults of the parts the family was measured on (no-gating: a
// measured loss is a named exception, not another entry condition).
// RADV-class parts advertise subgroup_size 64 and stay off until a
// wave64 build exists; ANV (bf16 XMX only) and NVIDIA (f16 16x16x16
// only) parts advertise no fp32 8x8x8 shape and take their own routes.
inline bool qmm_coopmat_tile_family_capable(const CapabilityReport& caps) {
  return caps.cooperative_matrix_f32_8 && caps.subgroup_size == 32u;
}

// Staged-tile shared-memory floors, matching need_ratio at the dispatch
// site: STEP_K=64 doubles the B tile to 64x36 f32 and adds the 64-word
// coalesced-W run (kSg8K64SharedBytes); STEP_K=32 halves it.
constexpr uint32_t kQmmCoopmatTileSharedK64 = (64u * 36u + 64u) * 4u;
constexpr uint32_t kQmmCoopmatTileSharedK32 = 32u * 36u * 4u;

inline bool qmm_coopmat_rows128_requested(const CapabilityReport& caps) {
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
  // Measured table: G13C (M1 Max), G14C (M2 Max) and G13G (13-inch M1)
  // default to 128-row SG8 coalesced-W K=64 where the grid fills the
  // part, stepping down to 64-row SG4 K=64 for smaller M/N. G13G
  // measured: prefill512 2.05-2.15x vs the 64-row SG4 default (9B 101.6
  // vs 49.5 tok/s, 4B 210.3 vs 98.0, ids identical, 5-pair ABBA on main
  // 1f854b9b85).
  const std::string& device_name = caps.device_name;
  if (device_name.find("G13G") != std::string::npos ||
      device_name.find("G13C") != std::string::npos ||
      device_name.find("G14C") != std::string::npos) {
    return true;
  }
  // Capability default for unlisted parts: the 128-row SG8 build needs a
  // 256-lane workgroup and its K=64 staged tile in shared memory.
  return qmm_coopmat_tile_family_capable(caps) &&
      caps.max_compute_work_group_invocations >= 256u &&
      caps.max_compute_work_group_size[0] >= 256u &&
      kQmmCoopmatTileSharedK64 <= caps.max_compute_shared_memory_size;
}

inline bool qmm_coopmat_rows64_requested(const CapabilityReport& caps) {
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
  // Measured table: G16S defaults on (64-row G4+LdsPad); G13G defaults
  // on (64-row SG4+LdsPad+STEPK=32); G13C and G14C use 64-row SG4+K=64
  // as their step-down from 128-row SG8.
  const std::string& device_name = caps.device_name;
  if (device_name.find("G16S") != std::string::npos ||
      device_name.find("G13G") != std::string::npos ||
      device_name.find("G13C") != std::string::npos ||
      device_name.find("G14C") != std::string::npos) {
    return true;
  }
  // Capability default for unlisted parts: the 64-row variants are
  // 128-lane workgroups; the K=64 staged tile is the largest they ask
  // shared memory for.
  return qmm_coopmat_tile_family_capable(caps) &&
      caps.max_compute_work_group_invocations >= 128u &&
      caps.max_compute_work_group_size[0] >= 128u &&
      kQmmCoopmatTileSharedK64 <= caps.max_compute_shared_memory_size;
}

inline bool qmm_coopmat_sg4_requested(const CapabilityReport& caps) {
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
  const std::string& device_name = caps.device_name;
  if (device_name.find("G13G") != std::string::npos ||
      device_name.find("G13C") != std::string::npos ||
      device_name.find("G14C") != std::string::npos) {
    return true;
  }
  // Measured exception: G16S defaults to the M64 non-SG4 kernel, so the
  // SG4 flag stays off for it even though the capability probe admits
  // the family (the probe below would otherwise reroute its measured
  // pick to the SG4 build).
  if (device_name.find("G16S") != std::string::npos) {
    return false;
  }
  // Capability default: SG4 variants are 128-lane workgroups.
  return qmm_coopmat_tile_family_capable(caps) &&
      caps.max_compute_work_group_invocations >= 128u &&
      caps.max_compute_work_group_size[0] >= 128u;
}

inline uint32_t qmm_coopmat_stepk_requested(
    const CapabilityReport& caps, uint32_t rows, bool sg4) {
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
  const std::string& device_name = caps.device_name;
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
  // Capability ladder for unlisted parts on 64-row SG4: the largest
  // staged tile the shared-memory budget admits, matching the Max-class
  // measured pick (K=64) down to the base-part pick (K=32).
  if (rows == 64u && sg4 && qmm_coopmat_tile_family_capable(caps)) {
    if (kQmmCoopmatTileSharedK64 <= caps.max_compute_shared_memory_size) {
      return 64u;
    }
    if (kQmmCoopmatTileSharedK32 <= caps.max_compute_shared_memory_size) {
      return 32u;
    }
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
    const CapabilityReport& caps,
    bool allow_rows64 = false,
    bool allow_rows128 = true) {
  if (matrix_m <= 16u) {
    return 16u;
  }
  const std::string& device_name = caps.device_name;
  uint32_t cores = apple_gpu_cores(device_name);
  uint32_t per_core = coopmat_workgroups_per_core();
  // 128-row SG8 K=64 (256 threads/WG): take it when M >= 256 and the
  // 128-row grid provides at least 4 workgroups per core (1024 threads/core).
  if (allow_rows64 && allow_rows128 &&
      qmm_coopmat_rows128_requested(caps) && matrix_m >= 256u) {
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
  if (allow_rows64 && qmm_coopmat_rows64_requested(caps) &&
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

// CMAT16 (fp16-operand 16x16x16) variant ladder for the qmm prefill
// (DEC-20): the shipped 32-row K=16 build stays the default and
// MLX_OMARCHY_QMM_COOPMAT_ROWS / MLX_OMARCHY_QMM_COOPMAT_STEPK opt into
// the 64/128-row twins where the device's workgroup-size and
// shared-memory caps admit them. Every variant stages the same values
// and keeps each output's ascending-k chain, so the pick does not move
// generated ids. ROWS=32 (or an unset/unrecognized value) is the
// kill-switch restore of the shipped kernel; STEPK has no 32-row twin
// and only applies at 64/128 rows.
inline omarchy::ComputeKernel qmm_coopmat16_kernel(
    const CapabilityReport& caps) {
  using K = omarchy::ComputeKernel;
  uint32_t rows = 32u;
  bool sg4 = false;
  bool sg8 = false;
  if (const char* env = omarchy::env_str("MLX_OMARCHY_QMM_COOPMAT_ROWS");
      env != nullptr && env[0] != '\0') {
    const std::string v(env);
    if (v == "64") {
      rows = 64u;
    } else if (v == "64sg4") {
      rows = 64u;
      sg4 = true;
    } else if (v == "128" || v == "128sg8") {
      rows = 128u;
      sg8 = true;
    }
  }
  uint32_t step_k = 16u;
  if (const char* env = omarchy::env_str("MLX_OMARCHY_QMM_COOPMAT_STEPK");
      env != nullptr && env[0] != '\0') {
    const std::string v(env);
    if (v == "32") {
      step_k = 32u;
    } else if (v == "64") {
      step_k = 64u;
    }
  }
  if (rows == 32u) {
    return K::QmmPrefillCoopmat16BF16;
  }
  // SG8 stages at least K=32: the 8-part dequant has no 16-K layout, so
  // a 128-row request at K=16 takes the full-chunk K=64 build.
  if (sg8 && step_k == 16u) {
    step_k = 64u;
  }
  const uint32_t lanes = sg8 ? 256u : (sg4 ? 128u : 64u);
  // fp16 x_s (rows x step_k) + fp16 w_s (step_k x 32) + f32 drain
  // (2 x 16x16): the widest tile the candidate asks shared memory for.
  const uint32_t shared_bytes =
      (rows * step_k + step_k * 32u) * 2u + 2u * 16u * 16u * 4u;
  if (caps.max_compute_work_group_invocations < lanes ||
      caps.max_compute_work_group_size[0] < lanes ||
      shared_bytes > caps.max_compute_shared_memory_size) {
    return K::QmmPrefillCoopmat16BF16;
  }
  if (rows == 128u) {
    return step_k == 32u ? K::QmmPrefillCoopmat16BF16SG8K32
                         : K::QmmPrefillCoopmat16BF16SG8K64;
  }
  if (sg4) {
    return step_k == 64u ? K::QmmPrefillCoopmat16BF16SG4K64
         : step_k == 32u ? K::QmmPrefillCoopmat16BF16SG4K32
                         : K::QmmPrefillCoopmat16BF16SG4;
  }
  return step_k == 64u ? K::QmmPrefillCoopmat16BF16M64K64
       : step_k == 32u ? K::QmmPrefillCoopmat16BF16M64K32
                       : K::QmmPrefillCoopmat16BF16M64;
}

inline uint32_t qmm_coopmat16_tile_rows(omarchy::ComputeKernel kernel) {
  switch (kernel) {
    case omarchy::ComputeKernel::QmmPrefillCoopmat16BF16M64:
    case omarchy::ComputeKernel::QmmPrefillCoopmat16BF16M64K32:
    case omarchy::ComputeKernel::QmmPrefillCoopmat16BF16M64K64:
    case omarchy::ComputeKernel::QmmPrefillCoopmat16BF16SG4:
    case omarchy::ComputeKernel::QmmPrefillCoopmat16BF16SG4K32:
    case omarchy::ComputeKernel::QmmPrefillCoopmat16BF16SG4K64:
      return 64u;
    case omarchy::ComputeKernel::QmmPrefillCoopmat16BF16SG8K32:
    case omarchy::ComputeKernel::QmmPrefillCoopmat16BF16SG8K64:
      return 128u;
    default:
      return 32u;
  }
}

} // namespace mlx::core::omarchy
