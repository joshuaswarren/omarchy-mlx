// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace mlx::core::omarchy {

class Device;

// CPU deep-idle hold for the Apple M1 (T8103, G13G) GPU performance state.
//
// Measured on jwm1 (receipts/2026-10-06-cpu-pd-hold): while every CPU core
// can sit in the deep idle state `CPU PD` (cpuidle state1, exit latency
// 10 us) the GPU firmware settles in a lower performance state: a 4096^2
// fp16 matmul loop runs 0.506 TFLOPS instead of 0.598 (+18 %), and the real
// cells gain TTFT -2.7 % (2B) / -6.5 % (4B) / -7.5 % (9B) and decode
// +0.6..0.8 %. Any CPU activity that keeps the cores out of CPU PD has the
// same effect. The M1 Max (G13C, jw16) shows no effect.
//
// While a stream has GPU work in flight the backend therefore holds a
// PM-QoS CPU latency request of 0 us (an open /dev/cpu_dma_latency fd
// with the int32 value 0, the kernel's documented interface). The
// request is dropped once the completion timeline has drained and no
// submission arrived for kIdleGraceMs, so idle power returns to the
// baseline within about half a second of idle.
//
// MLX_OMARCHY_CPU_PD_HOLD: unset = on for G13 parts other than G13C,
// off elsewhere; 0 = off; any other value = on for every part. If
// /dev/cpu_dma_latency cannot be opened (the packaged udev rule grants
// the seat user and group video access) the hold disables itself for
// the process; an explicit =1 prints one line to stderr.
//
// Called once per submission from CommandEncoder::submit(), after the
// completion value is reserved. The fast path is one relaxed atomic
// load plus two relaxed stores.
void cpu_pd_hold_note_submit(Device& device, uint64_t completion_value);

// Device teardown: stops the watchdog thread and drops the request before
// the device (and its completion dispatcher) goes away.
void cpu_pd_hold_device_destroyed(Device& device);

// Pure decision function (unit-tested): env is the value of
// MLX_OMARCHY_CPU_PD_HOLD or nullptr; returns true when the hold is
// enabled for a device of this name.
bool cpu_pd_hold_enabled_for(const char* env, const std::string& device_name);

} // namespace mlx::core::omarchy
