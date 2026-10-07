// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/capability_sim.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace mlx::core::omarchy::capsim {
namespace {

constexpr uint32_t kAllSubgroupOps = 0xffffffffu;

// The registry. Each entry is a delta over the discovered hardware
// report; sentinels (-1 / 0 / 0xffffffff) inherit the hardware value so
// a profile pins exactly the axes it represents.
//
// The first two profiles are the two M1 driver builds of record:
//   - honeykrisp_fork, the installed fork with the matrix unit
//     (receipts/hk/2026-09-08-honeykrisp-omarchy-integration.json);
//   - stock_mesa_honeykrisp, Omarchy's Mesa 26.1.7 without
//     VK_KHR_cooperative_matrix.
// The remaining profiles are capability axes on generic silicon: the
// M5/M6 class is unknown until it arrives, so nothing here names a
// future chip; a bring-up adds a row per docs/new-chip-bringup.md.
const SimulationProfile kProfiles[] = {
    {
        "m1-honeykrisp-fork",
        "honeykrisp_fork",
        "Apple M1 reference: the installed Honeykrisp fork with the "
        "8x8x8 fp32 cooperative matrix, subgroup size 32 with "
        "ARITHMETIC/SHUFFLE/SHUFFLE_RELATIVE",
        32,
        VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
            VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
            VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT,
        kAllSubgroupOps,
        1,
        0,
        0,
        0,
        -1,
    },
    {
        "m1-stock-no-coopmat",
        "stock_mesa_honeykrisp",
        "Apple M1 on stock Mesa Honeykrisp: subgroup size 32 with the "
        "full op mask, no cooperative matrix",
        32,
        VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
            VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
            VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT,
        kAllSubgroupOps,
        0,
        0,
        0,
        0,
        -1,
    },
    {
        "subgroup-size-64",
        "other",
        "Wide-subgroup device: subgroup size 64 with the full op mask, "
        "no cooperative matrix; exercises every == 32 gate falling back",
        64,
        VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
            VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
            VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT,
        kAllSubgroupOps,
        0,
        0,
        0,
        0,
        -1,
    },
    {
        "small-shared-memory",
        "honeykrisp_fork",
        "M1-fork capability set with shared_memory_limit_bytes cut to "
        "16384: only the 21504-byte SdpaDecodeNative staging gate "
        "flips; the 4096-byte coopmat staging gates stay satisfied",
        32,
        VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
            VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
            VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT,
        kAllSubgroupOps,
        1,
        16384,
        0,
        0,
        -1,
    },
    {
        "no-cooperative-matrix",
        "other",
        "Minimal generic device: no cooperative matrix, subgroup size "
        "64, op mask reduced to BASIC|ARITHMETIC; the dense BF16 decode "
        "GEMV SHUFFLE_RELATIVE gate and the subgroup qmm gate both fall "
        "back on op bits alone",
        64,
        VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT,
        0xffffffffu ^
            (VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
                VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT),
        0,
        0,
        0,
        0,
        -1,
    },
    {
        "m1-g13-legacy",
        "honeykrisp_fork",
        "Base-M1 (G13G) device name over the runtime capability set: "
        "every G13-legacy kernel selection (GDN perrow decode, walk "
        "prefetch off) takes its legacy arm; no capability axis changes",
        -1,
        0,
        kAllSubgroupOps,
        -1,
        0,
        0,
        0,
        -1,
        "Apple M1 (G13G B1)",
    },
};

std::string lowered(const char* v) {
  std::string s = v ? v : "";
  for (auto& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  // Profile names are lowercase; compare case-insensitively so
  // MLX_OMARCHY_CAPS_SIM=M1-Honeykrisp-Fork still resolves explicitly.
  return s;
}

} // namespace

const std::vector<const SimulationProfile*>& profiles() {
  static std::vector<const SimulationProfile*> table = [] {
    std::vector<const SimulationProfile*> v;
    for (const auto& p : kProfiles) {
      v.push_back(&p);
    }
    return v;
  }();
  return table;
}

const SimulationProfile* find(const std::string& name) {
  const std::string key = lowered(name.c_str());
  for (const auto* p : profiles()) {
    if (key == p->name) {
      return p;
    }
  }
  return nullptr;
}

std::string requested_profile() {
  const char* v = std::getenv("MLX_OMARCHY_CAPS_SIM");
  if (!v || !*v) {
    return {};
  }
  const SimulationProfile* profile = find(v);
  if (!profile) {
    std::string known;
    for (const auto* p : profiles()) {
      known += known.empty() ? p->name : std::string(", ") + p->name;
    }
    throw std::runtime_error(
        "[omarchy] MLX_OMARCHY_CAPS_SIM='" + std::string(v) +
        "' names no registered capability-simulation profile."
        " Known profiles: " +
        known +
        ". Refusing to continue on the real capability set:"
        " unset the variable or fix the name.");
  }
  return profile->name;
}

CapabilityReport apply(const CapabilityReport& hw, const SimulationProfile& profile) {
  CapabilityReport caps = hw;
  if (profile.subgroup_size >= 0) {
    caps.subgroup_size = static_cast<uint32_t>(profile.subgroup_size);
  }
  caps.subgroup_operations =
      (caps.subgroup_operations | profile.subgroup_ops_mask_or) &
      profile.subgroup_ops_mask_and;
  // One axis covers the extension: a simulated cooperative-matrix driver
  // lists the fp32 and fp16 shapes, as Honeykrisp does. The bf16 shape
  // needs a second extension that device creation enables only from
  // hardware truth, so the axis can clear it but never grant it.
  if (profile.cooperative_matrix >= 0) {
    caps.cooperative_matrix_f32_8 = profile.cooperative_matrix != 0;
    caps.cooperative_matrix_f16_8 = profile.cooperative_matrix != 0;
    caps.cooperative_matrix_bf16_8 =
        caps.cooperative_matrix_bf16_8 && profile.cooperative_matrix != 0;
  }
  if (profile.shared_memory_limit_bytes != 0) {
    caps.max_compute_shared_memory_size = profile.shared_memory_limit_bytes;
  }
  if (profile.workgroup_invocations != 0) {
    caps.max_compute_work_group_invocations = profile.workgroup_invocations;
  }
  if (profile.workgroup_size_x != 0) {
    caps.max_compute_work_group_size[0] = profile.workgroup_size_x;
  }
  if (profile.atomic_float_add >= 0) {
    caps.shader_atomic_float_add = profile.atomic_float_add != 0;
  }
  if (profile.device_name != nullptr) {
    caps.device_name = profile.device_name;
  }
  caps.simulated = true;
  caps.simulation_profile = profile.name;
  caps.simulated_driver_variant = profile.represents_driver_variant;
  return caps;
}

void require_backed(
    const Device& device,
    const CapabilityReport& caps,
    bool claimed,
    const char* kernel,
    const char* axis,
    bool hardware_provides) {
  if (!caps.simulated || !claimed || hardware_provides) {
    return;
  }
  throw std::runtime_error(
      "[omarchy] capability simulation '" + caps.simulation_profile +
      "' (driver_variant " + caps.simulated_driver_variant +
      ") dispatches " + kernel + ", which requires axis " + axis +
      " on the runtime device, but the hardware ('" + device.hardware_capabilities().device_name +
      "', " + device.hardware_capabilities().driver_name +
      ") does not provide it. Refusing: run this profile on hardware"
      " that provides the axis, or use a profile without it. No"
      " simulated result was produced.");
}

} // namespace mlx::core::omarchy::capsim
