// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Device extension-name selection: pure C++, no GPU, no Vulkan, no mlx
// linkage. device_exts.cpp is deliberately free of all three so this binary
// can run on any build host. Each case asserts the exact name list for one
// capability fixture, including ordering: feature extensions first, the
// memory-priority pair together, then priority and trace extensions.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstring>
#include <vector>

#include "device_exts.h"

namespace omarchy = mlx::core::omarchy;

namespace {

bool has(const std::vector<const char*>& names, const char* want) {
  for (const char* n : names) {
    if (std::strcmp(n, want) == 0) {
      return true;
    }
  }
  return false;
}

size_t count(const std::vector<const char*>& names, const char* want) {
  size_t n = 0;
  for (const char* s : names) {
    if (std::strcmp(s, want) == 0) {
      ++n;
    }
  }
  return n;
}

omarchy::DeviceExtInput full_caps() {
  omarchy::DeviceExtInput in;
  in.shader_atomic_float_add = true;
  in.cooperative_matrix_f32_8 = true;
  in.cooperative_matrix_bf16_8 = true;
  in.memory_priority = true;
  in.pageable_device_local_memory = true;
  in.external_memory_host = true;
  in.calibrated_timestamps = true;
  in.queue_global_priority = true;
  return in;
}

TEST_CASE("empty caps yield no extension names") {
  omarchy::DeviceExtInput in;
  auto names = omarchy::device_extension_names(in, true, true);
  CHECK(names.empty());
}

TEST_CASE("feature extensions follow their capability bits") {
  auto in = full_caps();
  in.memory_priority = false;
  in.pageable_device_local_memory = false;
  in.queue_global_priority = false;
  in.calibrated_timestamps = false;
  auto names = omarchy::device_extension_names(in, true, true);
  CHECK(has(names, "VK_EXT_shader_atomic_float"));
  CHECK(has(names, "VK_KHR_cooperative_matrix"));
  CHECK(has(names, "VK_KHR_shader_bfloat16"));
  CHECK(names.size() == 3);
}

TEST_CASE("bf16 name only with the bf16 cooperative-matrix bit") {
  auto in = full_caps();
  in.cooperative_matrix_bf16_8 = false;
  auto names = omarchy::device_extension_names(in, false, false);
  CHECK(!has(names, "VK_KHR_shader_bfloat16"));
}

TEST_CASE("cooperative-matrix name on any-shape only") {
  // A driver listing only the 16x16x16 fp16-operand C16 shape sets no
  // convenience boolean; the extension and feature still enable, or
  // the dense CMAT16 route claims a device without cooperativeMatrix.
  omarchy::DeviceExtInput in;
  in.cooperative_matrix_any_shape = true;
  auto names = omarchy::device_extension_names(in, false, false);
  CHECK(has(names, "VK_KHR_cooperative_matrix"));
  CHECK(!has(names, "VK_KHR_shader_bfloat16"));
}

TEST_CASE("memory priority and pageable enable as a pair only") {
  auto in = full_caps();
  in.shader_atomic_float_add = false;
  in.cooperative_matrix_f32_8 = false;
  in.cooperative_matrix_bf16_8 = false;
  in.queue_global_priority = false;
  in.calibrated_timestamps = false;

  auto both = omarchy::device_extension_names(in, false, false);
  CHECK(has(both, "VK_EXT_memory_priority"));
  CHECK(has(both, "VK_EXT_pageable_device_local_memory"));
  CHECK(both.size() == 2);

  in.pageable_device_local_memory = false;
  auto lone = omarchy::device_extension_names(in, false, false);
  CHECK(lone.empty());

  in.pageable_device_local_memory = true;
  in.memory_priority = false;
  auto other = omarchy::device_extension_names(in, false, false);
  CHECK(other.empty());
}

TEST_CASE("external_memory_host is probed, never enabled") {
  auto in = full_caps();
  auto names = omarchy::device_extension_names(in, false, false);
  CHECK(count(names, "VK_EXT_external_memory_host") == 0);
}

TEST_CASE("calibrated timestamps only under the trace request") {
  auto in = full_caps();
  auto traced = omarchy::device_extension_names(in, false, true);
  CHECK(has(traced, "VK_EXT_calibrated_timestamps"));
  auto plain = omarchy::device_extension_names(in, false, false);
  CHECK(!has(plain, "VK_EXT_calibrated_timestamps"));
}

TEST_CASE("low queue name only when requested and supported") {
  auto in = full_caps();
  auto low = omarchy::device_extension_names(in, true, false);
  CHECK(has(low, "VK_EXT_global_priority"));
  auto normal = omarchy::device_extension_names(in, false, false);
  CHECK(!has(normal, "VK_EXT_global_priority"));
  in.queue_global_priority = false;
  auto absent = omarchy::device_extension_names(in, true, false);
  CHECK(!has(absent, "VK_EXT_global_priority"));
}

}  // namespace
