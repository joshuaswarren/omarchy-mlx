// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Device extension-name selection, deliberately free of GPU, Vulkan and
// mlx linkage so its doctest binary runs on any build host (the
// omarchy_m2v_route_tests / omarchy_coopmat_select_tests pattern).
// Extension names are string literals: the Vulkan header is not needed
// to decide which names to enable.
#pragma once

#include <vector>

namespace mlx::core::omarchy {

// Capability bits and request flags that decide the device extension
// list. Mirrors the CapabilityReport fields of the same names; Device
// fills this from hardware truth, so a simulated report can never
// enable an extension the driver does not advertise.
struct DeviceExtInput {
  bool shader_atomic_float_add{false};
  bool cooperative_matrix_f32_8{false};
  bool cooperative_matrix_bf16_8{false};
  // VK_EXT_memory_priority and VK_EXT_pageable_device_local_memory:
  // enabled only as a pair, matching lily's device init on
  // Honeykrisp-M5 - pageable memory is what the priority hint acts on.
  bool memory_priority{false};
  bool pageable_device_local_memory{false};
  // Probed and reported only; enabling it is deferred to the change
  // that imports host weight memory. Never appears in the name list.
  bool external_memory_host{false};
  bool calibrated_timestamps{false};
  bool queue_global_priority{false};
};

// The exact enabledExtensionNames list for one device. |low_queue| is
// the caller's intent to request a non-default queue priority;
// |trace_timestamps| gates the profiling extension to trace runs, as
// lily gates it on trace::on().
std::vector<const char*> device_extension_names(
    const DeviceExtInput& in,
    bool low_queue,
    bool trace_timestamps);

}  // namespace mlx::core::omarchy
