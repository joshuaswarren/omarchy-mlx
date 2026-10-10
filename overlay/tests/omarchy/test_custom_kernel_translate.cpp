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
std::string translation_cache_material_for_test(
    const std::string& identity,
    const std::string& source_sha,
    const std::string& library_hash);
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
      "  bool POST = i < 4u;\n"
      "  if (POST || i > 100u) y[i] = act;\n"
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
  // Word-bounded substitution: an identifier ending in the template name
  // (POST) must survive the `T ` -> `float ` rewrite.
  CHECK(glsl.find("POSfloat") == std::string::npos);
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
  // as_type<uint> on a uint buffer read is the identity conversion (420271614),
  // not a float-bits reinterpret; the mask constant is the IEEE infinity pattern.
  CHECK(glsl.find("floatBitsToUint(") == std::string::npos);
  CHECK(glsl.find("uint(_mlx_arg1[i])") != std::string::npos);
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

TEST_CASE("bf16 buffer named x coexists with the threadgrid swizzle") {
  // The inkling sconv decode kernel names its bf16 buffer `x`; the bare-use
  // check used to count the `.x` swizzle of thread_position_in_grid as a
  // bare buffer use and refuse the kernel.
  const char* source =
      "[[kernel]] void sconv_style(\n"
      "    const device bfloat16_t* x [[buffer(0)]],\n"
      "    device float* nstate [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint c = thread_position_in_grid.x;\n"
      "  float v = (float)x[c];\n"
      "  nstate[c] = v;\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("_mlx_bf16_to_float(_b0.data[c])") != std::string::npos);
  CHECK(glsl.find("thread_position_in_grid.x") == std::string::npos);
  CHECK(glsl.find(".x;") == std::string::npos);
}

TEST_CASE("nested c-style casts rewrite inside constructor arguments") {
  // moe_route_fused / sconv: `uint((r + (int)(4 - 1)))` leaves a surviving
  // C-style cast inside a constructor argument; glslc rejects that with
  // GL_NV_explicit_typecast. The scanner must rescan its own replacements.
  const char* source =
      "[[kernel]] void k(\n"
      "    const device bfloat16_t* x [[buffer(0)]],\n"
      "    device float* nstate [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint c = thread_position_in_grid.x;\n"
      "  int r = int((c + (int)(4 - 1)));\n"
      "  nstate[c] = float(x[uint((r + (int)(4 - 1))) * c]);\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("(int)(") == std::string::npos);
  CHECK(glsl.find("(4 - 1)") != std::string::npos);
}

TEST_CASE("const multi-declarator lines keep comma declarators") {
  // moe_route_fused: `const int K = mp[0], KHI = mp[1];` must not fold into
  // a single int(A, B) constructor call.
  const char* source =
      "[[kernel]] void k(\n"
      "    const device int* mp [[buffer(0)]],\n"
      "    device int* y [[buffer(1)]],\n"
      "    uint e [[thread_position_in_grid]]) {\n"
      "  constant int* mp_alias = mp;\n"
      "  const int K = mp[0], KHI = mp[1];\n"
      "  y[e] = K + KHI;\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("int(K,") == std::string::npos);
  CHECK(glsl.find(", KHI") != std::string::npos);
}

TEST_CASE("metal fabs maps to GLSL abs and header casts rewrite") {
  const char* header =
      "inline float sigmoidish(float x) {\n"
      "  return (float)exp(abs((float)x)) - fabs(x);\n"
      "}\n";
  std::string source = std::string(header) +
      "[[kernel]] void k(\n"
      "    const device float* values [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint c [[thread_position_in_grid]]) {\n"
      "  y[c] = sigmoidish(values[c]) * metal::fabs(values[c]);\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("fabs") == std::string::npos);
  CHECK(glsl.find("abs(float(x))") != std::string::npos);
  CHECK(glsl.find("abs(") != std::string::npos);
}

TEST_CASE("as_type<uint> on an int buffer read is the identity conversion") {
  // The llguidance mask: mask is int32; as_type<uint>(int) is an identity
  // bitcast. floatBitsToUint would reinterpret the FLOAT bits of the
  // converted value and mask ~half the vocabulary at random.
  const char* source =
      "[[kernel]] void k(\n"
      "    const device float* logits [[buffer(0)]],\n"
      "    const device int* mask [[buffer(1)]],\n"
      "    device float* out [[buffer(2)]],\n"
      "    uint i [[thread_position_in_grid]]) {\n"
      "  bool allowed = ((as_type<uint>(mask[i]) >> (i & 31u)) & 1u) != 0u;\n"
      "  out[i] = allowed ? logits[i] : -INFINITY;\n"
      "}\n";
  auto glsl = translate(source, 1);
  // doctest rejects `||` inside CHECK; hoist the disjunction.
  const bool either_bad_form_absent =
      glsl.find("uint(mask[") == std::string::npos ||
      glsl.find("floatBitsToUint(mask") == std::string::npos;
  CHECK(either_bad_form_absent);
  CHECK(glsl.find("floatBitsToUint(_mlx_arg1[") == std::string::npos);
  // The identity form: an integer operand converts by value.
  CHECK(glsl.find("uint(_mlx_arg1[i])") != std::string::npos);
}

TEST_CASE("triple-nested casts and float literal cast arguments") {
  // kda_glue_pre: `(float)((float)cq * (float)sq_)` — the fixpoint pass must
  // rewrite the inner casts after the outer one, and never re-match its own
  // output into `float((float))(float(cq) * ...)`. moe_route_fused:
  // `(bfloat16_t)0.0f` must cast the whole literal, not stop at the digit.
  const char* source =
      "[[kernel]] void k(\n"
      "    const device float* v [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint c [[thread_position_in_grid]]) {\n"
      "  float a = (float)((float)v[c] * (float)v[c]);\n"
      "  float b = (float)0.0f;\n"
      "  float w = (float)(3 + (int)(7 - 1));\n"
      "  y[c] = a + b + w;\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("float(") != std::string::npos);
  CHECK(glsl.find("(float)") == std::string::npos);
  CHECK(glsl.find("float((float))") == std::string::npos);
  CHECK(glsl.find("float(0).0f") == std::string::npos);
  CHECK(glsl.find("0.0f") != std::string::npos);
  CHECK(glsl.find("(int)(") == std::string::npos);
}

TEST_CASE("chained scalar casts collapse before the scan") {
  // kda_glue_pre: `(float)((float)cq * (float)sq_)` and
  // moe_route_fused: `(bfloat16_t)0.0f` — after the collapse no C-style
  // cast survives inside a constructor argument.
  const char* source =
      "[[kernel]] void k(\n"
      "    const device float* v [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint c [[thread_position_in_grid]]) {\n"
      "  float a = (float)((float)v[c] * (float)v[c]);\n"
      "  float b = (float)0.0f;\n"
      "  float w = (float)(3 + (int)(7 - 1));\n"
      "  y[c] = a + b + w;\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("(float)") == std::string::npos);
  CHECK(glsl.find("float((float))") == std::string::npos);
  CHECK(glsl.find("float(0).0f") == std::string::npos);
  CHECK(glsl.find("(int)(") == std::string::npos);
}



TEST_CASE("translation cache material changes with translator identity") {
  // Different source_sha or library_hash must produce different cache
  // materials so two translator builds never share a .tr entry.
  auto m1 = mlx::core::fast::translation_cache_material_for_test(
      "kernel_id", "sha_aaa", "lib_111");
  auto m2 = mlx::core::fast::translation_cache_material_for_test(
      "kernel_id", "sha_bbb", "lib_111");
  auto m3 = mlx::core::fast::translation_cache_material_for_test(
      "kernel_id", "sha_aaa", "lib_222");
  auto m1b = mlx::core::fast::translation_cache_material_for_test(
      "kernel_id", "sha_aaa", "lib_111");
  CHECK(m1 != m2);   // different source sha
  CHECK(m1 != m3);   // different library hash
  CHECK(m1 == m1b);  // deterministic
}

TEST_CASE("translation cache material differs per library identity") {
  // Two libmlx builds with the same kernel source but different library
  // hashes must produce different .tr cache entries. The 'unknown'
  // fallback for the build-time SHA is acceptable ONLY because the
  // runtime library hash is always unique per build.
  auto m1 = mlx::core::fast::translation_cache_material_for_test(
      "kernel_identity_string", "source_sha_aaa", "lib_hash_111");
  auto m2 = mlx::core::fast::translation_cache_material_for_test(
      "kernel_identity_string", "source_sha_aaa", "lib_hash_222");
  auto m3 = mlx::core::fast::translation_cache_material_for_test(
      "kernel_identity_string", "source_sha_bbb", "lib_hash_111");
  auto m1b = mlx::core::fast::translation_cache_material_for_test(
      "kernel_identity_string", "source_sha_aaa", "lib_hash_111");
  // Different library hash → different cache material
  CHECK(m1 != m2);
  // Different source sha → different cache material
  CHECK(m1 != m3);
  // Same inputs → same material (deterministic)
  CHECK(m1 == m1b);
}

TEST_CASE("the 'unknown' fallback with different library hashes yields different keys") {
  // This is the pip-built wheel case: the build-time SHA is "unknown"
  // but the runtime library hash still distinguishes builds.
  auto m1 = mlx::core::fast::translation_cache_material_for_test(
      "kid", "unknown", "lib_hash_aaa");
  auto m2 = mlx::core::fast::translation_cache_material_for_test(
      "kid", "unknown", "lib_hash_bbb");
  CHECK(m1 != m2);
}

// ---------------------------------------------------------------------------
// Qwen3.5 GDN serve failure 2026-10-09 (Qwen3.5-9B-MLX-4bit, first
// decode token, bf16 q_out [1,1,16,128]): the omlx_qwen35_gdn_prework kernel
// body declares `const auto sy = 1 / (1 + exp(abs(conv)));`. MSL `auto` has
// no GLSL counterpart — `auto` is a reserved word — and glslang died with
// `.comp:73: syntax error, unexpected IDENTIFIER, expecting COMMA or
// SEMICOLON`. v0.7.31 (9b5c938) has no auto handling either: the construct
// was never supported, this is its first serving use.

TEST_CASE("MSL auto value declarations become float locals") {
  const char* source =
      "[[kernel]] void gdn_siluscale(\n"
      "    const device bfloat16_t* x [[buffer(0)]],\n"
      "    device bfloat16_t* y [[buffer(1)]],\n"
      "    uint i [[thread_position_in_grid]]) {\n"
      "  float v = (float)x[i];\n"
      "  const auto sy = 1 / (1 + metal::precise::exp(metal::abs(v)));\n"
      "  float sig = v < 0.0f ? sy : 1 - sy;\n"
      "  y[i] = (bfloat16_t)(v * sig);\n"
      "}\n";
  auto glsl = translate(source, 1);
  // The reserved word must not survive anywhere.
  CHECK(glsl.find("auto ") == std::string::npos);
  // The declaration keeps its initializer, now concretely typed.
  CHECK(glsl.find("const float sy = 1 / (1 + exp(abs(v)));") !=
        std::string::npos);
}

// ---------------------------------------------------------------------------
// Qwen3.6-35B-A3B prefill (no expert offload, custom-kernel gate off — these
// are mlx_vlm kernels): the qwen3_5 ragged-SDPA and gated-delta kernels walk
// buffer rows with `kptr += BN * D_SIZE;` — the alias pass refused the bare
// advance as `device pointer arithmetic` (serve refusal, 2026-10-09).

TEST_CASE("pointer advance on an alias becomes a location variable") {
  const char* source =
      "[[kernel]] void row_walk(\n"
      "    const device float* keys [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint3 tg [[threadgroup_position_in_grid]]) {\n"
      "  const device float* kptr = keys + tg.x * 8u;\n"
      "  float acc = 0.0f;\n"
      "  for (int i = 0; i < 4; ++i) {\n"
      "    acc += kptr[0] + kptr[1];\n"
      "    kptr += 8u;\n"
      "  }\n"
      "  y[tg.x] = acc;\n"
      "}\n";
  auto glsl = translate(source, 1);
  // The pointer itself is gone (the location variable legitimately keeps
  // the name as a suffix); no declaration, no indexed pointer use, no
  // advance statement survives.
  CHECK(glsl.find("float* kptr") == std::string::npos);
  CHECK(glsl.find("kptr[") == std::string::npos);
  CHECK(glsl.find("kptr +=") == std::string::npos);
  // The location starts at the row offset and advances in the loop; uses
  // read through the base buffer via its macro.
  CHECK(glsl.find("uint _mlx_kptr_loc = uint(") != std::string::npos);
  CHECK(glsl.find("_mlx_kptr_loc += 8u;") != std::string::npos);
  CHECK(glsl.find("_mlx_arg0[(_mlx_kptr_loc) + (0)]") != std::string::npos);
}

TEST_CASE("pointer advance on a buffer parameter becomes a location") {
  const char* source =
      "[[kernel]] void step_walk(\n"
      "    const device float* q [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint dv [[thread_position_in_grid.y]]) {\n"
      "  y += dv * 4u;\n"
      "  y[0] = q[0];\n"
      "  y += 2u;\n"
      "  y[1] = q[1];\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("y +=") == std::string::npos);
  // The location declaration survives later constructor wrapping.
  CHECK(glsl.find("uint _mlx_y_loc = ") != std::string::npos);
  CHECK(glsl.find("_mlx_y_loc += dv * 4u;") != std::string::npos);
  CHECK(glsl.find("_mlx_arg1[(_mlx_y_loc) + (0)]") != std::string::npos);
  CHECK(glsl.find("_mlx_arg1[(_mlx_y_loc) + (1)]") != std::string::npos);
}

TEST_CASE("single-token typedefs expand to their type") {
  // `typedef float U;` inside the kernel body (mlx_vlm qwen3_5 ragged SDPA)
  // has no GLSL meaning.
  const char* source =
      "[[kernel]] void widen(\n"
      "    const device float* x [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint i [[thread_position_in_grid]]) {\n"
      "  typedef float U;\n"
      "  U v = x[i];\n"
      "  y[i] = v * 2.0f;\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("typedef") == std::string::npos);
  CHECK(glsl.find("U ") == std::string::npos);
  CHECK(glsl.find("float v = _mlx_arg0[i];") != std::string::npos);
}

TEST_CASE("auto pointer declarations alias like explicit device pointers") {
  // The Qwen3.5 MoE router/decode row-walk idiom: row alias off a buffer
  // parameter, then a scoped rebind of the buffer's own name to the row.
  const char* source =
      "[[kernel]] void row_walk(\n"
      "    const device float* logits [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint3 tg [[threadgroup_position_in_grid]]) {\n"
      "  const auto logits_row = logits + tg.x * 4u;\n"
      "  {\n"
      "    const auto logits = logits_row;\n"
      "    y[tg.x] = logits[0] + logits[1];\n"
      "  }\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("auto ") == std::string::npos);
  // Both the row alias and the rebind are gone; the use carries the
  // parent offset through the buffer-0 macro.
  CHECK(glsl.find("logits_row") == std::string::npos);
  CHECK(glsl.find("_mlx_arg1[tg.x] = float(_mlx_arg0[((tg.x * 4u + 0)) + (0)]") !=
        std::string::npos);
}

TEST_CASE("header helper numeric_limits constants map like the body") {
  // The omlx_log1p header helper of the Qwen3.5/Qwen4 GDN kernels tests
  // against metal::numeric_limits<float>::max(); the body-only limits pass
  // left `numeric_limits` in the emitted helper (glslang: undeclared
  // identifier at the helper line).
  const char* source =
      "inline float omlx_cap(float x) {\n"
      "    if (x == metal::numeric_limits<float>::max()) {\n"
      "        return metal::numeric_limits<float>::max();\n"
      "    }\n"
      "    return x;\n"
      "}\n"
      "[[kernel]] void capped(\n"
      "    const device float* x [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint i [[thread_position_in_grid]]) {\n"
      "  y[i] = omlx_cap(x[i]);\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("numeric_limits") == std::string::npos);
  CHECK(glsl.find("3.4028234663852886e+38f") != std::string::npos);
  CHECK(glsl.find("omlx_cap(_mlx_arg0[i])") != std::string::npos);
}

TEST_CASE("float16_t buffer reads stay half in a plain assignment and a call argument") {
  // Parakeet decoder-step chains (v0.7.32 freeze, G13C): `sh_a[i] =
  // embedding[j];` and `exact_fma16(bc, sh_a[k], weights[j])` need float16_t.
  // The widened read gave glslc `cannot convert from temp float to temp
  // float16_t` and `exact_fma16: no matching overloaded function`.
  const char* source =
      "float16_t exact_fma16(float16_t acc, float16_t x, float16_t y) {\n"
      "    precise float p = float(x) * float(y);\n"
      "    return float16_t(float(acc) + p);\n"
      "}\n"
      "[[kernel]] void chains(\n"
      "    const device float16_t* embedding [[buffer(0)]],\n"
      "    const device float16_t* weights [[buffer(1)]],\n"
      "    device float16_t* bsum [[buffer(2)]],\n"
      "    uint t [[thread_position_in_threadgroup]]) {\n"
      "  threadgroup float16_t sh_a[64];\n"
      "  sh_a[t] = embedding[t];\n"
      "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  float16_t bc = float16_t(0.0f);\n"
      "  for (uint k = 0u; k < 8u; ++k) {\n"
      "    bc = exact_fma16(bc, sh_a[k], weights[k * 64u + t]);\n"
      "  }\n"
      "  bsum[t] = bc;\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("sh_a[t] = float16_t(_b0.data[t]);") != std::string::npos);
  CHECK(glsl.find("= float(_b0.data[t])") == std::string::npos);
  CHECK(glsl.find("exact_fma16(bc, sh_a[k], _b1.data[") != std::string::npos);
  CHECK(glsl.find("float(_b1.data[") == std::string::npos);
}

TEST_CASE("bitlinear_matmul: float16_t reads widen in int-times-half arithmetic, stay half into a float array") {
  // The reason 123469dec widens half reads: `1 / weight_scale[0]` is int / half,
  // and GLSL has no such operation (glslang: "wrong operand types: no
  // operation '/' exists that takes a left-hand operand of type const int and
  // a right operand of type readonly temp float16_t"). The same kernel's
  // `v[j] = x[...]` stores a half read into a float array, which GLSL widens
  // implicitly, and `float sum[4] = {0.0}` is the under-supplied initializer
  // the same commit zero-fills. Source: mlx-lm bitlinear_matmul, T = half.
  const char* source =
      "[[kernel]] void bitlinear_matmul(\n"
      "    const device float16_t* x [[buffer(0)]],\n"
      "    const device uint8_t* packed_weights [[buffer(1)]],\n"
      "    const device float16_t* weight_scale [[buffer(2)]],\n"
      "    const device int* invert [[buffer(3)]],\n"
      "    device float* out [[buffer(4)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint tid = thread_position_in_grid.x;\n"
      "  float v[4];\n"
      "  for (int j = 0; j < 4; j++) {\n"
      "    v[j] = x[tid * 4 + j];\n"
      "  }\n"
      "  float sum[4] = {0.0};\n"
      "  for (int j = 0; j < 4; j++) {\n"
      "    uint8_t w = packed_weights[tid * 4 + j];\n"
      "    sum[0] += v[j] * ((w & 3) - 1);\n"
      "  }\n"
      "  float scale = invert[0] != 0 ? 1 / weight_scale[0] : weight_scale[0];\n"
      "  out[tid] = sum[0] * scale;\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("1 / float(_b2.data[0]) : float(_b2.data[0])") != std::string::npos);
  CHECK(glsl.find("v[j] = _b0.data[tid * 4 + j];") != std::string::npos);
  CHECK(glsl.find("float sum[4] = {0.0, 0.0, 0.0, 0.0};") != std::string::npos);
}

TEST_CASE("compound and self-referencing assignment into a float16_t local stores through float16_t") {
  // `acc += w[i]` and `acc = acc + w[i]` translated and compiled before the
  // half-read widening (353da309a) and failed glslang after it ("cannot
  // convert from temp float to temp float16_t"). The assignment into a half
  // lvalue now stores through float16_t(); a float accumulator is untouched.
  const char* source =
      "[[kernel]] void accum(\n"
      "    const device float16_t* w [[buffer(0)]],\n"
      "    device float16_t* y [[buffer(1)]],\n"
      "    device float* z [[buffer(2)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint t = thread_position_in_grid.x;\n"
      "  float16_t acc = float16_t(0.0f);\n"
      "  float16_t acc2 = float16_t(0.0f);\n"
      "  float facc = 0.0f;\n"
      "  for (uint i = 0; i < 8; ++i) {\n"
      "    acc += w[t * 8 + i];\n"
      "    acc2 = acc2 + w[t * 8 + i];\n"
      "    facc += w[t * 8 + i];\n"
      "  }\n"
      "  y[t] = acc + acc2;\n"
      "  z[t] = facc;\n"
      "}\n";
  auto glsl = translate(source, 2);
  CHECK(glsl.find("acc = float16_t(acc + (float(_b0.data[t * 8 + i])));") != std::string::npos);
  CHECK(glsl.find("acc2 = float16_t(acc2 + float(_b0.data[t * 8 + i]));") != std::string::npos);
  CHECK(glsl.find("facc += float(_b0.data[t * 8 + i]);") != std::string::npos);
  CHECK(glsl.find(" acc += ") == std::string::npos);
}

TEST_CASE("float16_t lvalue wrapping never evaluates a side-effecting index twice") {
  // `h[i++] += x` expands to `h[i++] = float16_t(h[i++] + (x))` if the left
  // side is repeated: i increments twice. A compound statement whose index
  // has ++, --, a call or an assignment stays exactly as written; a pure index
  // still expands; a plain `=` names the left side once and still wraps.
  const char* source =
      "int bump(int k) { return k + 1; }\n"
      "[[kernel]] void idx(\n"
      "    const device float16_t* w [[buffer(0)]],\n"
      "    device float* out [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint t = thread_position_in_grid.x;\n"
      "  float16_t h[8];\n"
      "  int i = 0;\n"
      "  int j = 0;\n"
      "  float facc = 0.0f;\n"
      "  h[i++] += float16_t(1.0);\n"
      "  h[i + 1] += w[t];\n"
      "  h[bump(i)] += w[t];\n"
      "  h[j++] = w[t];\n"
      "  facc += w[t];\n"
      "  out[t] = float(i) + facc;\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("h[i++] += ") != std::string::npos);
  CHECK(glsl.find("float16_t(h[i++]") == std::string::npos);
  CHECK(glsl.find("h[bump(i)] += ") != std::string::npos);
  CHECK(glsl.find("float16_t(h[bump(i)]") == std::string::npos);
  CHECK(glsl.find("h[i + 1] = float16_t(h[i + 1] + (float(_b0.data[t])));") != std::string::npos);
  CHECK(glsl.find("h[j++] = float16_t(_b0.data[t]);") != std::string::npos);
  CHECK(glsl.find("facc += float(_b0.data[t]);") != std::string::npos);
}

TEST_CASE("float16_t buffer reads in a brace initializer keep their half type") {
  // `float16_t arr[2] = {e[0], e[1]};` compiled before the half-read widening
  // and failed glslang after it ("constructor: cannot convert parameter 1 from
  // temp float to temp float16_t"): the array constructor takes the element
  // type. The same initializer into a float array also compiles with half
  // elements, so every element of the list stays unwidened.
  const char* source =
      "[[kernel]] void init(\n"
      "    const device float16_t* e [[buffer(0)]],\n"
      "    device float16_t* out [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint t = thread_position_in_grid.x;\n"
      "  float16_t arr[2] = {e[0], e[1]};\n"
      "  float f[2] = {e[2], e[3]};\n"
      "  out[t] = arr[1] + float16_t(f[0]);\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("{_b0.data[0], _b0.data[1]}") != std::string::npos);
  CHECK(glsl.find("{_b0.data[2], _b0.data[3]}") != std::string::npos);
  CHECK(glsl.find("float(_b0.data[0])") == std::string::npos);
}
