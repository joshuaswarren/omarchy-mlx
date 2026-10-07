// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// Unit tests for the custom-kernel MSL->GLSL translator's helper-function
// support: threadgroup-pointer parameter specialization (mlx-serve's fused
// residual+RMSNorm reduction stage), body shared-memory declarations, and
// the named refusals.

#define MLX_OMARCHY_TEST_TRANSLATE
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "mlx/fast.h"

#include <stdexcept>
#include <string>

namespace mlx::core::fast {
std::string mlx_omarchy_translate_msl_for_test(
    const std::string& source,
    int grid_x,
    int grid_y,
    int grid_z,
    int threads_x,
    int threads_y,
    int threads_z,
    std::size_t output_count);
}

namespace {

std::string translate(const std::string& source, std::size_t output_count = 3) {
  return mlx::core::fast::mlx_omarchy_translate_msl_for_test(
      source, 1, 1, 1, 640, 1, 1, output_count);
}

// The mlx-serve fused residual+RMSNorm kernel: a header helper with two
// `threadgroup float*` parameters, called from the body with the same-named
// shared arrays. Shaped like the real source (transformer.zig
// RESIDUAL_NORM_HEADER / RESIDUAL_NORM_SOURCE), scaled to one row read.
const char* residual_norm_style =
    "// One row reduction stage.\n"
    "METAL_FUNC float msv_row_inv_rms(float acc, float eps, int axis, "
    "threadgroup float* local_sums, threadgroup float* local_inv, "
    "uint simd_gid, uint simd_lid) {\n"
    "  acc = simd_sum(acc);\n"
    "  if (simd_lid == 0) local_sums[simd_gid] = acc;\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (simd_gid == 0) {\n"
    "    acc = simd_sum(local_sums[simd_lid]);\n"
    "    if (simd_lid == 0) local_inv[0] = metal::precise::rsqrt(acc / axis + eps);\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  return local_inv[0];\n"
    "}\n"
    "[[kernel]] void mlxserve_residual_norm(\n"
    "    const device bfloat16_t* b1 [[buffer(0)]],\n"
    "    const device bfloat16_t* b2 [[buffer(1)]],\n"
    "    device float* res [[buffer(2)]],\n"
    "    uint thread_position_in_grid [[thread_position_in_grid]],\n"
    "    uint thread_position_in_threadgroup [[thread_position_in_threadgroup]],\n"
    "    uint threads_per_threadgroup [[threads_per_threadgroup]]) {\n"
    "  threadgroup float local_inv[1];\n"
    "  threadgroup float local_sums[32];\n"
    "  uint lid = thread_position_in_threadgroup;\n"
    "  if (lid < 32) local_sums[lid] = 0;\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  float acc1 = 0;\n"
    "  float inv1 = msv_row_inv_rms(acc1, 1e-6f, 2560, local_sums, local_inv, 0u, lid);\n"
    "  res[lid] = b1[lid] + b2[lid] * inv1;\n"
    "}\n";

} // namespace

TEST_CASE("threadgroup helper parameters specialize to same-named globals") {
  auto glsl = translate(residual_norm_style, 1);
  // The pointer parameters are gone from the helper signature.
  CHECK(glsl.find("threadgroup float*") == std::string::npos);
  CHECK(glsl.find("float msv_row_inv_rms(float acc, float eps, int axis, uint") !=
        std::string::npos);
  // Body shared-memory declarations translated.
  CHECK(glsl.find("shared float local_inv[1]") != std::string::npos);
  CHECK(glsl.find("shared float local_sums[32]") != std::string::npos);
  // MSL spellings rewritten everywhere (header helpers included).
  CHECK(glsl.find("METAL_FUNC") == std::string::npos);
  CHECK(glsl.find("threadgroup") == std::string::npos);
  CHECK(glsl.find("simd_sum") == std::string::npos);
  CHECK(glsl.find("threadgroup_barrier") == std::string::npos);
  CHECK(glsl.find("subgroupAdd") != std::string::npos);
  CHECK(glsl.find("barrier()") != std::string::npos);
  CHECK(glsl.find("inversesqrt") != std::string::npos);
  // The helper body still references the (now global) arrays.
  CHECK(glsl.find("local_sums[") != std::string::npos);
  CHECK(glsl.find("local_inv[0]") != std::string::npos);
}

TEST_CASE("aliased threadgroup argument is refused by name") {
  std::string source(residual_norm_style);
  const auto at = source.find("msv_row_inv_rms(acc1, 1e-6f, 2560, ");
  REQUIRE(at != std::string::npos);
  // Pass an alias instead of the shared array's own name.
  source.replace(
      at, std::string("msv_row_inv_rms(acc1, 1e-6f, 2560, local_sums").size(),
      "msv_row_inv_rms(acc1, 1e-6f, 2560, scratch");
  try {
    translate(source);
    FAIL("aliased argument must be refused");
  } catch (const std::exception& error) {
    INFO("actual error: " << error.what());
    CHECK(std::string(error.what()).find("passes `scratch`") !=
          std::string::npos);
  }
}

TEST_CASE("helper with threadgroup parameter that is never called is refused") {
  const char* source =
      "METAL_FUNC float stage(float acc, threadgroup float* sums, uint gid) {\n"
      "  sums[gid] = acc;\n"
      "  return acc;\n"
      "}\n"
      "[[kernel]] void mlxserve_unused(\n"
      "    const device float* b0 [[buffer(0)]],\n"
      "    device float* res [[buffer(1)]],\n"
      "    uint thread_position_in_threadgroup "
      "[[thread_position_in_threadgroup]]) {\n"
      "  res[thread_position_in_threadgroup] = b0[thread_position_in_threadgroup];\n"
      "}\n";
  try {
    translate(source, 1);
    FAIL("uncalled helper must be refused");
  } catch (const std::exception& error) {
    INFO("actual error: " << error.what());
    CHECK(std::string(error.what()).find("never calls it") != std::string::npos);
  }
}

TEST_CASE("body threadgroup declarations become shared without helpers") {
  const char* source =
      "[[kernel]] void mlxserve_scratch(\n"
      "    const device float* b0 [[buffer(0)]],\n"
      "    device float* res [[buffer(1)]],\n"
      "    uint thread_position_in_threadgroup "
      "[[thread_position_in_threadgroup]]) {\n"
      "  threadgroup float acc[8];\n"
      "  acc[thread_position_in_threadgroup % 8] = b0[thread_position_in_threadgroup];\n"
      "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  res[thread_position_in_threadgroup] = acc[0];\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("shared float acc[8]") != std::string::npos);
  CHECK(glsl.find("threadgroup") == std::string::npos);
  CHECK(glsl.find("barrier()") != std::string::npos);
}