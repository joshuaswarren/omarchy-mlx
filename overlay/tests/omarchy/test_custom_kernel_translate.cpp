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
  // Both parameters were dropped from every call site and the neighbouring
  // arguments survived byte-for-byte (regression: an off-by-two in the
  // call-site erase corrupted the third argument's trailing digit).
  CHECK(glsl.find("msv_row_inv_rms(acc1, 1e-6f, 2560, 0u, lid)") !=
        std::string::npos);
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

TEST_CASE("as_type<ushort> on a bf16 value indexes the pattern table") {
  // The real mlx-serve form: a templated kernel whose bf16 local feeds the
  // table lookup (`T g = gate[i]; sigtab[as_type<ushort>(g)]; T act = ...`).
  const char* source =
      "template <typename T>\n"
      "[[kernel]] void mlxserve_table(\n"
      "    const device bfloat16_t* gate [[buffer(0)]],\n"
      "    const device bfloat16_t* sigtab [[buffer(1)]],\n"
      "    const device bfloat16_t* up [[buffer(2)]],\n"
      "    device float* y [[buffer(3)]],\n"
      "    uint thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint i = thread_position_in_grid;\n"
      "  T g = gate[i];\n"
      "  T sig = sigtab[as_type<ushort>(g)];\n"
      "  T act = g * sig;\n"
      "  y[i] = act * up[i];\n"
      "}\n"
      "template [[kernel]] decltype(mlxserve_table<bfloat16_t>) "
      "mlxserve_table<bfloat16_t>;\n";
  auto glsl = translate(source, 1);
  // The table index is the bf16 pattern of the value, not the widened
  // float's 32-bit bits (floatBitsToUint would index out of the 65536-row
  // table and read garbage activations).
  CHECK(glsl.find("_mlx_float_to_bf16(") != std::string::npos);
  // The bad composite (old translator): the pattern index computed from the
  // widened float's 32-bit bits.
  CHECK(glsl.find("_b2.data[floatBitsToUint(") == std::string::npos);
  CHECK(glsl.find("_b1.data[_mlx_float_to_bf16(") != std::string::npos);
  // The bf16 locals round like Metal's bfloat16_t operators (compute fp32,
  // round on store) so the shader stays bit-identical to the composed bf16
  // op chain.
  CHECK(glsl.find("float act = _mlx_bf16_round_trip(g * sig);") !=
        std::string::npos);
  CHECK(glsl.find("float g = _mlx_bf16_round_trip(") != std::string::npos);
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
      "  bool in = thread_position_in_threadgroup < 4;\n"
      "  res[thread_position_in_threadgroup] = acc[0] + (in ? 1.0f : 0.0f);\n"
      "}\n";
  auto glsl = translate(source, 1);
  // Hoisted to file scope as GLSL shared, before the kernel body.
  CHECK(glsl.find("shared float acc[8]") != std::string::npos);
  CHECK(glsl.find("threadgroup") == std::string::npos);
  CHECK(glsl.find("barrier()") != std::string::npos);
  // MSL identifier `in` is a GLSL reserved word; renamed in the body.
  CHECK(glsl.find("_mlx_in") != std::string::npos);
}
// ---------------------------------------------------------------------------
// KernelRecheck 2026-10-08: the section-(b) inventory kernels on current main.
// Each case mirrors a construct the M2 recheck run showed dying at GLSL
// compile time or at the pointer guard.

TEST_CASE("numeric_limits<T>::infinity() maps to INFINITY, ordinary max survives") {
  const char* source =
      "[[kernel]] void llguidance_mask(\n"
      "    const device float* logits [[buffer(0)]],\n"
      "    const device uint* mask [[buffer(1)]],\n"
      "    device float* out [[buffer(2)]],\n"
      "    uint i [[thread_position_in_grid]]) {\n"
      "  bool allowed = ((as_type<uint>(mask[i]) >> (i & 31u)) & 1u) != 0u;\n"
      "  out[i] = allowed ? logits[i]\n"
      "                  : -metal::numeric_limits<float>::infinity();\n"
      "  out[i] = metal::max(out[i], 0.0f);\n"
      "}\n";
  auto glsl = translate(source, 1);
  // The bitcast is exact and the mask constant is the IEEE infinity pattern.
  CHECK(glsl.find("floatBitsToUint(") != std::string::npos);
  CHECK(glsl.find("-INFINITY") != std::string::npos);
  CHECK(glsl.find("#define INFINITY uintBitsToFloat(0x7F800000u)") !=
        std::string::npos);
  CHECK(glsl.find("numeric_limits") == std::string::npos);
  // The ordinary metal::max call must survive: a bare-word mapping of
  // max/min would corrupt every clamped kernel in the corpus.
  CHECK(glsl.find("max(") != std::string::npos);
  CHECK(glsl.find("3.402823") == std::string::npos);
}

TEST_CASE("constant-space body pointer aliases rewrite like device aliases") {
  const char* source =
      "[[kernel]] void glue_post_style(\n"
      "    const device bfloat16_t* inp [[buffer(0)]],\n"
      "    const device float* cons [[buffer(1)]],\n"
      "    device bfloat16_t* y [[buffer(2)]],\n"
      "    uint c [[thread_position_in_grid]]) {\n"
      "  constant float* cs = cons;\n"
      "  float x = (float)inp[c];\n"
      "  y[c] = (bfloat16_t)(x * rsqrt(cs[0]));\n"
      "}\n";
  auto glsl = translate(source, 1);
  // The alias is gone; uses go through the buffer-1 alias; the guard
  // never fires.
  CHECK(glsl.find("constant") == std::string::npos);
  CHECK(glsl.find("float* cs") == std::string::npos);
  CHECK(glsl.find("inversesqrt(_mlx_arg1[") != std::string::npos);
  CHECK(glsl.find("_mlx_float_to_bf16(") != std::string::npos);

  // An alias whose base is not a buffer parameter is still refused by name.
  const char* bad =
      "[[kernel]] void glue_post_bad(\n"
      "    const device float* cons [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint c [[thread_position_in_grid]]) {\n"
      "  float x = cons[0];\n"
      "  constant float* cs = x + 1.0f;\n"
      "  y[c] = cs[c];\n"
      "}\n";
  CHECK_THROWS_AS(translate(bad, 1), std::runtime_error);
}

TEST_CASE("inline header helpers lose the MSL-only qualifier") {
  const char* header =
      "inline float twice(float v) { return v + v; }\n";
  std::string source = std::string(header) +
      "[[kernel]] void kda_glue_style(\n"
      "    const device float* values [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint c [[thread_position_in_grid]]) {\n"
      "  y[c] = twice(values[c]);\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("inline") == std::string::npos);
  CHECK(glsl.find("float twice(float v)") != std::string::npos);
  CHECK(glsl.find("twice(_mlx_arg0[") != std::string::npos);
}

TEST_CASE("c-style cast of a bf16 buffer read keeps the subscript attached") {
  // (float)inp[c] casts the ELEMENT (C: cast binds to the whole postfix
  // expression). The scanner used to rewrite it to float(inp)[c], which
  // separated the index from the buffer name: the bf16 read rewrite then
  // could not match, and the bare-use check refused the kernel
  // ("unsupported bfloat16 buffer expression") — the dominant failure of the
  // 2026-10-08 KernelRecheck corpus (mask, sconv, K3 glue, situ).
  const char* source =
      "[[kernel]] void cast_read(\n"
      "    const device bfloat16_t* inp [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint c [[thread_position_in_grid]]) {\n"
      "  float x = (float)inp[c];\n"
      "  y[c] = x;\n"
      "}\n";
  auto glsl = translate(source, 1);
  // The read is widened from the bf16 pattern storage.
  CHECK(glsl.find("_mlx_bf16_to_float(_b0.data[c])") != std::string::npos);
  // No bare buffer token survives the widening.
  CHECK(glsl.find("inp") == std::string::npos);
  CHECK(glsl.find("float(inp)[c]") == std::string::npos);
}
