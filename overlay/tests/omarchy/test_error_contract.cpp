// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Error-contract regression tests for the Omarchy backend:
//   1. an unsupported primitive raises the named std::runtime_error contract
//      through eval ("[omarchy] ... is not implemented"),
//   2. a zero-size array saves and loads round-trip instead of segfaulting
//      (upstream test_load empty-save crash: the copy path skipped set_data
//      for zero-size outputs, so Contiguous::buffer_size dereferenced a null
//      array::Data),
//   3. evaluating a stream created on another thread raises the upstream
//      std::runtime_error contract. The old encoder lookup threw
//      std::out_of_range (unordered_map::at), which escaped
//      catch(std::runtime_error) sites and terminated the process.
//
// Needs MLX_BUILD_OMARCHY=ON and a usable Vulkan device; on non-Omarchy
// machines set MLX_OMARCHY_ALLOW_NON_APPLE=1.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "mlx/transforms.h"
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/device.h"
#include "mlx/io.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

TEST_CASE("unsupported primitive raises a catchable named error") {
  if (!gpu::is_available()) {
    MESSAGE("Skipping: no Omarchy Vulkan device available\n");
    return;
  }
  set_default_device(Device::gpu);

  // Length three has no supported Hadamard factorization.
  array a = hadamard_transform(
      array({1.0f, 2.0f, 3.0f}, {3}, float32), 1.0f);
  bool caught = false;
  std::string message;
  try {
    a.eval();
  } catch (const std::runtime_error& e) {
    caught = true;
    message = e.what();
  }
  REQUIRE(caught);
  CHECK(message.find("[omarchy]") != std::string::npos);
  CHECK(message.find("is not implemented") != std::string::npos);
  CHECK(message.find("no silent CPU fallback") != std::string::npos);
}

TEST_CASE("zero-size array saves and loads round-trip") {
  if (!gpu::is_available()) {
    MESSAGE("Skipping: no Omarchy Vulkan device available\n");
    return;
  }
  set_default_device(Device::gpu);

  auto path = (std::filesystem::temp_directory_path() /
               "omarchy_error_contract_zeros.npy")
                  .string();
  array empty = zeros({0});
  CHECK_NOTHROW(empty.eval());
  CHECK_NOTHROW(save(path, empty));

  array loaded = load(path);
  CHECK_EQ(loaded.ndim(), 1);
  CHECK_EQ(loaded.shape(0), 0);
  CHECK_EQ(loaded.dtype(), float32);
  // The eval'd loaded array must expose valid (empty) storage too.
  CHECK_NOTHROW(loaded.eval());
  CHECK_EQ(loaded.buffer_size(), 0);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("access stream in other thread raises runtime_error") {
  if (!gpu::is_available()) {
    MESSAGE("Skipping: no Omarchy Vulkan device available\n");
    return;
  }
  set_default_device(Device::gpu);

  auto main_thread_stream = new_stream(Device::gpu);
  eval(arange(10, main_thread_stream));

  bool caught_runtime_error = false;
  bool caught_other = false;
  std::string message;
  std::thread t([&] {
    try {
      eval(arange(10, main_thread_stream));
    } catch (const std::runtime_error& e) {
      caught_runtime_error = true;
      message = e.what();
    } catch (...) {
      // std::out_of_range or anything else: not the contract.
      caught_other = true;
    }
  });
  t.join();

  CHECK(caught_runtime_error);
  CHECK_FALSE(caught_other);
  CHECK(message.find("Stream(gpu,") != std::string::npos);
}
