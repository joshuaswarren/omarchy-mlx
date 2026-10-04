// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

// Capability simulation: named profiles that override the discovered
// device capability set so non-M1 capability axes can be developed and
// regression-tested on whatever device the box actually has (llvmpipe,
// lavapipe, or the one M1).
//
// Axis vocabulary (shared with docs/compatibility-matrix.md and the
// chip-portability audit; profile fields use these names):
//
//   driver_variant                  which driver BUILD serves the chip.
//                                   Never simulated: the runtime reports the
//                                   real driver. A profile records which
//                                   driver_variant it stands in for.
//   cooperative_matrix_fp32_8x8x8   CapabilityReport::cooperative_matrix_f32_8
//   subgroup_size                   CapabilityReport::subgroup_size
//   subgroup_ops_mask               CapabilityReport::subgroup_operations
//   shared_memory_limit_bytes       CapabilityReport::max_compute_shared_memory_size
//   workgroup_limits                CapabilityReport::max_compute_work_group_invocations
//                                   + max_compute_work_group_size[]
//   atomic_float_add                CapabilityReport::shader_atomic_float_add
//   memory_model                    CapabilityReport::unified_memory +
//                                   host_visible_coherent
//   digest_policy                   policy, not a device property: simulated
//                                   runs never produce digest-gated or
//                                   benchmark evidence (the bench scripts
//                                   refuse; docs/new-chip-bringup.md).
//
// Gating contract, all enforced in code:
//   1. Explicit: MLX_OMARCHY_CAPS_SIM=<profile>. Unset means real
//      discovery, byte-for-byte today's behavior. An unknown name is a
//      loud init failure, never a silent fall-through to real caps.
//   2. Stamped: a simulated CapabilityReport carries simulated=true and
//      the profile name; device_info() and mlx-omarchy-info emit
//      simulated/simulation_profile keys only when a profile is active.
//   3. Backed: simulation overrides the DISPATCH-visible report only.
//      VkDevice creation and extension enabling always use hardware
//      truth. When a simulated bit selects a kernel the runtime device
//      cannot execute (cooperative matrix on llvmpipe), the dispatch
//      site refuses with a named error instead of crashing in pipeline
//      creation.
//   4. Evidence-free: benchmark and generated-id-digest tooling refuses
//      to run under MLX_OMARCHY_CAPS_SIM, so simulated numbers can never
//      be mistaken for hardware results.

#include <string>
#include <vector>

#include "mlx/backend/omarchy/device.h"

namespace mlx::core::omarchy::capsim {

// One named capability profile: a DELTA over the discovered hardware
// report. Fields with an "inherit" sentinel keep the hardware value, so
// a profile pins only the axes it represents and stays meaningful on any
// host device.
struct SimulationProfile {
  const char* name;
  // Axis driver_variant this profile stands in for (metadata for
  // provenance and the audit table; never written into the runtime
  // driver identity).
  const char* represents_driver_variant;
  const char* description;
  // Axis subgroup_size. -1 = inherit hardware value.
  int subgroup_size;
  // Axis subgroup_ops_mask: bits OR-ed in, then bits AND-ed out
  // (0xffffffff keeps everything).
  uint32_t subgroup_ops_mask_or;
  uint32_t subgroup_ops_mask_and;
  // Axis cooperative_matrix_fp32_8x8x8. -1 = inherit, 0/1 = set.
  int cooperative_matrix;
  // Axis shared_memory_limit_bytes. 0 = inherit.
  size_t shared_memory_limit_bytes;
  // Axis workgroup_limits (invocations / size_x). 0 = inherit.
  uint32_t workgroup_invocations;
  uint32_t workgroup_size_x;
  // Axis atomic_float_add. -1 = inherit, 0/1 = set.
  int atomic_float_add;
  // Axis device_name: the reported name, which the G13-legacy kernel
  // selections key on. nullptr = inherit.
  const char* device_name = nullptr;
};

// The registered profiles, in registry order. docs/new-chip-bringup.md
// describes how to add one for a new chip.
const std::vector<const SimulationProfile*>& profiles();

// The named profile, or nullptr when unknown.
const SimulationProfile* find(const std::string& name);

// Resolve MLX_OMARCHY_CAPS_SIM. Empty when unset or empty. Throws
// std::runtime_error naming the value and every known profile otherwise,
// so a typo refuses loudly instead of running on real caps.
std::string requested_profile();

// Apply |profile| as a delta onto hardware truth |hw|. The result is
// stamped simulated=true with the profile name and represented
// driver_variant. Pure; unit-testable without a Vulkan loader.
CapabilityReport apply(const CapabilityReport& hw, const SimulationProfile& profile);

// Refusal at the point of use. When |caps| is simulated and |claimed|,
// but the runtime device's hardware truth does not provide
// |hardware_bit|, throw std::runtime_error naming the profile, the
// kernel, and the axis. With no simulation active this returns without
// reading anything: the default path is untouched.
void require_backed(
    const Device& device,
    const CapabilityReport& caps,
    bool claimed,
    const char* kernel,
    const char* axis,
    bool hardware_provides);

} // namespace mlx::core::omarchy::capsim
