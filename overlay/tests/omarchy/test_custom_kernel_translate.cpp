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

#include <algorithm>
#include <cstdlib>
#include <fstream>
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

namespace {
std::size_t paren_balance(const std::string& s) {
  return std::count(s.begin(), s.end(), '(') -
      std::count(s.begin(), s.end(), ')');
}
}

TEST_CASE("chained scalar cast collapse stays balanced") {
  // kda_glue_pre: `float yq = (float)(bfloat16_t)((float)cq * (float)sq_);`
  // — the 2026-10-08 collapse replaced the cast pair with a bare "(",
  // deleting one ')' per match, so the emitted line held an unclosed paren
  // and glslc died on "unexpected SEMICOLON". Replacement "" (the pair is
  // a no-op under the literal-bfloat16 promotion) keeps the parens
  // balanced; the bf16 rounding of the TEMPLATE path is covered by the
  // _bf16_cast mapping, which never enters this collapse.
  const char* source =
      "[[kernel]] void glue_style(\n"
      "    const device bfloat16_t* hp [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint c = thread_position_in_grid.x;\n"
      "  bfloat16_t cq = (bfloat16_t)hp[c];\n"
      "  float sq_ = cq * 0.5f;\n"
      "  float yq = (float)(bfloat16_t)((float)cq * (float)sq_);\n"
      "  y[c] = yq;\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(paren_balance(glsl) == 0);
  CHECK(glsl.find("(float)(") == std::string::npos);
}

TEST_CASE("alias functional cast matches only the callee position") {
  // kda_glue_pre: `float(nq * cs[2])` with `constant float* cs = cons;` —
  // the alias pass walked back from `cs(` and accepted `nq` as the cast
  // type token, replacing the WHOLE call (including its closing paren)
  // with a first-element read: the kernel lost the expression and its
  // balance in one rewrite. The form is `T(cs)` only: the identifier must
  // start immediately after the call's own '('.
  const char* source =
      "[[kernel]] void glue_style(\n"
      "    const constant float* cons [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint c [[thread_position_in_grid]]) {\n"
      "  constant float* cs = cons;\n"
      "  float nq = 2.0f;\n"
      "  y[c] = float(nq * cs[2]);\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(paren_balance(glsl) == 0);
  CHECK(glsl.find("nq") != std::string::npos);
  CHECK(glsl.find("nq(_mlx_arg0[") == std::string::npos);
  CHECK(glsl.find("(0) + (2)]") != std::string::npos);
}

TEST_CASE("alias cast-strip erases only casts adjacent to the alias") {
  // kda_glue_pre: `float(nq) * cs[2]` — the (b) walk-back ate `nq` as the
  // cast type and erased `(nq)` wholesale, leaving `float * cs[2]` (the
  // variable became the bare word "float"). The cast span must end
  // immediately before the alias.
  const char* source =
      "[[kernel]] void glue_style(\n"
      "    const constant float* cons [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint c [[thread_position_in_grid]]) {\n"
      "  constant float* cs = cons;\n"
      "  float nq = 2.0f;\n"
      "  y[c] = (float)(nq) * cs[2];\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(paren_balance(glsl) == 0);
  CHECK(glsl.find("nq") != std::string::npos);
  CHECK(glsl.find("float * ") == std::string::npos);
  CHECK(glsl.find("(0) + (2)]") != std::string::npos);
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

TEST_CASE("float16_t lvalue wrapping evaluates a side-effecting index once") {
  // `h[i++] += x` must not expand to `h[i++] = float16_t(h[i++] + (x))`: i
  // would increment twice. A compound statement whose index has ++, --, a
  // call or an assignment hoists that index into a temporary first; a pure
  // index expands in place; a plain `=` names the left side once and wraps.
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
  // Side-effecting and call indices are evaluated once into a block-local
  // temporary used on both sides: i increments once, the sink stays half.
  CHECK(glsl.find("{ int _mlx_ix0 = int(i++); h[_mlx_ix0] = float16_t(h[_mlx_ix0] + (float16_t(1.0))); }") != std::string::npos);
  CHECK(glsl.find("i++") == glsl.rfind("i++"));
  CHECK(glsl.find("{ int _mlx_ix1 = int(bump(i)); h[_mlx_ix1] = float16_t(h[_mlx_ix1] + (float(_b0.data[t]))); }") != std::string::npos);
  CHECK(glsl.find("bump(i)") == glsl.rfind("bump(i)"));
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

TEST_CASE("compound assignment into a float16_t output element stores through float16_t") {
  // Tracked as issue 63 (p6): `out[t] += w[j];` with a half output compiled
  // before the half-read widening and failed glslang after it. A side-effecting
  // index is left as written (never evaluated twice); a plain `=` keeps the
  // existing output-store wrap; a float output is untouched.
  const char* source =
      "[[kernel]] void k(\n"
      "    const device float16_t* w [[buffer(0)]],\n"
      "    device float16_t* out [[buffer(1)]],\n"
      "    device float* fout [[buffer(2)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint t = thread_position_in_grid.x;\n"
      "  uint j = t + 1;\n"
      "  int i = 0;\n"
      "  out[t] = float16_t(0.0f);\n"
      "  out[t] += w[j];\n"
      "  out[i++] += w[j];\n"
      "  fout[t] += w[j];\n"
      "}\n";
  auto glsl = translate(source, 2);
  CHECK(glsl.find("float16_t(_mlx_arg1[t] + (float(_b0.data[j])))") != std::string::npos);
  CHECK(glsl.find("{ int _mlx_ix0 = int(i++); _mlx_arg1[_mlx_ix0] = float16_t(_mlx_arg1[_mlx_ix0] + (float(_b0.data[j]))); }") != std::string::npos);
  CHECK(glsl.find("i++") == glsl.rfind("i++"));
  CHECK(glsl.find("_mlx_arg2[t] += float(_b0.data[j]);") != std::string::npos);
}

TEST_CASE("an operator expression passed to a float16_t helper parameter converts through float16_t") {
  // Tracked as issue 63 (p1): `fma16(bc, w[t], w[j] + w[j + 1])` needs a half
  // third argument. Only the helper's declared signature decides: a float
  // parameter is never narrowed, and plain values keep their text.
  const char* source =
      "float16_t fma16(float16_t acc, float16_t x, float16_t y) {\n"
      "  return float16_t(float(acc) + float(x) * float(y));\n"
      "}\n"
      "float mix(float a, float b) {\n"
      "  return a + b;\n"
      "}\n"
      "[[kernel]] void k(\n"
      "    const device float16_t* w [[buffer(0)]],\n"
      "    device float16_t* out [[buffer(1)]],\n"
      "    device float* fout [[buffer(2)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint t = thread_position_in_grid.x;\n"
      "  uint j = t + 1;\n"
      "  float16_t bc = float16_t(0.0f);\n"
      "  bc = fma16(bc, w[t], w[j] + w[j + 1]);\n"
      "  fout[t] = mix(float(bc), w[j] + w[j + 1]);\n"
      "  out[t] = bc;\n"
      "}\n";
  auto glsl = translate(source, 2);
  CHECK(glsl.find("fma16(bc, _b0.data[t], float16_t(float(_b0.data[j]) + float(_b0.data[j + 1])))") != std::string::npos);
  CHECK(glsl.find("mix(float(bc), float(_b0.data[j]) + float(_b0.data[j + 1]))") != std::string::npos);
}

TEST_CASE("helper overloads: a float overload keeps its float argument, an all-half set still converts") {
  // `pick(float16_t)` and `pick(float)` in the same header: the unwrapped float
  // argument picks the float overload, as before the helper-argument pass.
  // Wrapping it in float16_t() would silently pick the half overload (both
  // compile, so only the translated text shows it). With only half overloads
  // of that arity the argument still converts. A different arity is a
  // different signature.
  const char* source =
      "float pick(float16_t x) { return 1.0; }\n"
      "float pick(float x) { return 2.0; }\n"
      "float both(float16_t a, float16_t b) { return 3.0; }\n"
      "float both(float16_t a) { return 4.0; }\n"
      "[[kernel]] void k(\n"
      "    const device float16_t* w [[buffer(0)]],\n"
      "    device float* out [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  out[0] = pick(w[0] + w[1]);\n"
      "  out[1] = both(w[0], w[0] + w[1]);\n"
      "  out[2] = both(w[0] + w[1]);\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("pick(float(_b0.data[0]) + float(_b0.data[1]))") != std::string::npos);
  CHECK(glsl.find("pick(float16_t(") == std::string::npos);
  CHECK(glsl.find("both(_b0.data[0], float16_t(float(_b0.data[0]) + float(_b0.data[1])))") != std::string::npos);
  CHECK(glsl.find("both(float16_t(float(_b0.data[0]) + float(_b0.data[1])))") != std::string::npos);
}

TEST_CASE("helper arguments: enclosing parentheses and nested helper calls over widened reads convert") {
  // `fma16(bc, w[0], (w[1] + w[2]))`: the operator sits inside parentheses, so
  // a top-level operator scan misses it, but the argument is still a float
  // sum. Any argument containing a widened read is float-typed. A nested helper
  // call is scanned after its parent is rewritten. A float helper still gets
  // its float argument unwrapped.
  const char* source =
      "float16_t fma16(float16_t acc, float16_t x, float16_t y) {\n"
      "  return float16_t(float(acc) + float(x) * float(y));\n"
      "}\n"
      "float sum32(float x, float y) { return x + y; }\n"
      "[[kernel]] void k(\n"
      "    const device float16_t* w [[buffer(0)]],\n"
      "    device float* out [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  float16_t bc = float16_t(0.0f);\n"
      "  bc = fma16(bc, w[0], (w[1] + w[2]));\n"
      "  bc = fma16(bc, w[0], fma16(bc, w[1], w[2] + w[3]));\n"
      "  out[0] = sum32(float(bc), (w[1] + w[2]));\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("fma16(bc, _b0.data[0], float16_t((float(_b0.data[1]) + float(_b0.data[2]))))") != std::string::npos);
  CHECK(glsl.find("fma16(bc, _b0.data[1], float16_t(float(_b0.data[2]) + float(_b0.data[3])))") != std::string::npos);
  CHECK(glsl.find("sum32(float(bc), (float(_b0.data[1]) + float(_b0.data[2])))") != std::string::npos);
}

TEST_CASE("helper arguments: a leading float16_t() cast is skipped only when it spans the whole argument") {
  // `float16_t(w[1]) + w[2]` starts with a constructor but is a float sum and
  // must be wrapped; `float16_t(w[1] + w[2])` is already one whole constructor
  // and keeps its text.
  const char* source =
      "float16_t fma16(float16_t acc, float16_t x, float16_t y) {\n"
      "  return float16_t(float(acc) + float(x) * float(y));\n"
      "}\n"
      "[[kernel]] void k(\n"
      "    const device float16_t* w [[buffer(0)]],\n"
      "    device float* out [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  float16_t bc = float16_t(0.0f);\n"
      "  bc = fma16(bc, w[0], float16_t(w[1]) + w[2]);\n"
      "  bc = fma16(bc, w[0], float16_t(w[1] + w[2]));\n"
      "  out[0] = float(bc);\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("float16_t(float16_t(_b0.data[1]) + float(_b0.data[2]))") != std::string::npos);
  CHECK(glsl.find("float16_t(float16_t(float(_b0.data[1]) + float(_b0.data[2])))") == std::string::npos);
  CHECK(glsl.find("fma16(bc, _b0.data[0], float16_t(float(_b0.data[1]) + float(_b0.data[2])))") != std::string::npos);
}

TEST_CASE("hoisted compound index: multidimensional sinks hoist only the effectful groups") {
  // `h[i++][2] += x` hoists the first group; `h[i + 1][j--] += x` hoists the
  // second; a pure `h[1][2] += x` expands in place. Each effect happens once.
  const char* source =
      "[[kernel]] void k(\n"
      "    const device float16_t* w [[buffer(0)]],\n"
      "    device float* out [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint t = thread_position_in_grid.x;\n"
      "  float16_t h[4][4];\n"
      "  int i = 0;\n"
      "  int j = 3;\n"
      "  h[i++][2] += w[t];\n"
      "  h[i + 1][j--] += w[t];\n"
      "  h[1][2] += w[t];\n"
      "  out[t] = float(i) + float(j);\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("{ int _mlx_ix0 = int(i++); h[_mlx_ix0][2] = float16_t(h[_mlx_ix0][2] + (float(_b0.data[t]))); }") != std::string::npos);
  CHECK(glsl.find("{ int _mlx_ix1 = int(j--); h[i + 1][_mlx_ix1] = float16_t(h[i + 1][_mlx_ix1] + (float(_b0.data[t]))); }") != std::string::npos);
  CHECK(glsl.find("h[1][2] = float16_t(h[1][2] + (float(_b0.data[t])));") != std::string::npos);
  CHECK(glsl.find("i++") == glsl.rfind("i++"));
  CHECK(glsl.find("j--") == glsl.rfind("j--"));
}

namespace {
// True when every ( and [ in `text` closes in order.
bool brackets_balanced(const std::string& text) {
  std::string stack;
  for (char c : text) {
    if (c == '(' || c == '[') {
      stack.push_back(c);
    } else if (c == ')' || c == ']') {
      if (stack.empty() || stack.back() != (c == ')' ? '(' : '[')) {
        return false;
      }
      stack.pop_back();
    }
  }
  return stack.empty();
}
}  // namespace

TEST_CASE("bf16 input read with a nested subscript keeps its brackets balanced") {
  // inkling_banded_mask_v2: proj[(size_t)d * st[0] + (size_t)dist * st[1]].
  // The read rewrite stopped at the FIRST ']' (the one closing st[0]) and
  // produced `_mlx_bf16_to_float(_b0.data[... st[0])] ...` — glslc then
  // failed with "scalar integer expression required" (2026-11 recheck).
  const char* source =
      "[[kernel]] void nested_read(\n"
      "    const device bfloat16_t* proj [[buffer(0)]],\n"
      "    const constant int64_t* st [[buffer(1)]],\n"
      "    device float* y [[buffer(2)]],\n"
      "    uint d [[thread_position_in_grid]]) {\n"
      "  y[d] = (float)proj[(size_t)d * st[0] + (size_t)d * st[1]];\n"
      "}\n";
  auto glsl = translate(source, 1);
  const auto at = glsl.find("_mlx_bf16_to_float(_b0.data[");
  REQUIRE(at != std::string::npos);
  // The whole statement (not a prefix of it) must close in order.
  const auto begin = glsl.rfind('\n', at) + 1;
  const auto statement = glsl.substr(begin, glsl.find(";", at) - begin);
  CHECK(brackets_balanced(statement));
  CHECK(statement.find("_mlx_arg1[0])]") == std::string::npos);
}

TEST_CASE("bf16 output store with a nested subscript keeps its brackets balanced") {
  const char* source =
      "[[kernel]] void nested_store(\n"
      "    const device float* v [[buffer(0)]],\n"
      "    const constant int* ix [[buffer(1)]],\n"
      "    device bfloat16_t* out [[buffer(2)]],\n"
      "    uint d [[thread_position_in_grid]]) {\n"
      "  out[ix[d] + d] = v[d];\n"
      "}\n";
  auto glsl = translate(source, 1);
  const auto at = glsl.find("_b2.data[");
  REQUIRE(at != std::string::npos);
  const auto begin = glsl.rfind('\n', at) + 1;
  const auto statement = glsl.substr(begin, glsl.find(";", at) - begin);
  CHECK(brackets_balanced(statement));
  CHECK(statement.find("] = _mlx_float_to_bf16(") != std::string::npos);
}

TEST_CASE("int64 stride reads narrow to int so they can index a GLSL array") {
  // inkling_banded_mask_v2: `x[d * x_strides[0]]` made a 64-bit array index
  // (`_b3.data[uint(d) * _mlx_arg4[0] ...]`); glslc rejects it with "scalar
  // integer expression required". GLSL indexes with 32-bit ints only.
  const char* source =
      "[[kernel]] void stride_index(\n"
      "    const device float* x [[buffer(0)]],\n"
      "    const constant int64_t* x_strides [[buffer(1)]],\n"
      "    device float* y [[buffer(2)]],\n"
      "    uint d [[thread_position_in_grid]]) {\n"
      "  y[d] = x[(size_t)d * x_strides[0] + x_strides[1]];\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("int(_mlx_arg1[0])") != std::string::npos);
  CHECK(glsl.find("int(_mlx_arg1[1])") != std::string::npos);
  // No stride read is left at 64 bits.
  const auto body = glsl.substr(glsl.find("void main()"));
  for (auto at = body.find("_mlx_arg1["); at != std::string::npos;
       at = body.find("_mlx_arg1[", at + 1)) {
    CHECK(body.substr(at >= 4 ? at - 4 : 0, 4) == "int(");
  }
}

TEST_CASE("in-place store into a bf16 input buffer keeps its read and narrows") {
  // Qwen3.5 GDN state update: `x[i] = x[i] + 1;` writes into a bf16 INPUT
  // buffer. The old code rewrote the read first and then narrowed the store
  // with a dedicated regex; the helper must do both in one pass, and the
  // right-hand read must survive as a bf16 read (not a bare `x[` token).
  const char* source =
      "[[kernel]] void inplace(\n"
      "    device bfloat16_t* state [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint i [[thread_position_in_grid]]) {\n"
      "  state[i] = state[i] + 1;\n"
      "  y[i] = 0.0f;\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("_b0.data[i] = _mlx_float_to_bf16(") != std::string::npos);
  CHECK(glsl.find("_mlx_bf16_to_float(_b0.data[i])") != std::string::npos);
  CHECK(glsl.find("state[") == std::string::npos);
}

TEST_CASE("C-style cast to the bf16 template parameter rounds like Metal") {
  // sconv_decode: `float conv_r = (float)((T)acc);` with T=bfloat16_t means
  // CONSTRUCT a bfloat16_t from acc - a round to the bf16 grid. The template
  // resolver substituted the type name first, leaving `(float)acc`, a no-op:
  // the conv output skipped Metal's bf16 rounding and the kernel diverged
  // from the reference (2026-11-07 recheck, 423/4096 outputs bad).
  const char* source =
      "template <typename T>\n"
      "[[kernel]] void k(\n"
      "    const device float* a [[buffer(0)]],\n"
      "    device bfloat16_t* y [[buffer(1)]],\n"
      "    uint i [[thread_position_in_grid]]) {\n"
      "  float conv_r = (float)((T)a[i]);\n"
      "  y[i] = (T)conv_r;\n"
      "}\n"
      "template [[host_name(\"k__bfloat16_t\")]] [[kernel]] decltype(k)<bfloat16_t>(\n"
      "    const device float*, device bfloat16_t*, uint);\n";
  auto glsl = translate(source, 1);
  const auto body = glsl.substr(glsl.find("void main()"));
  CHECK(body.find("_mlx_bf16_round_trip(") != std::string::npos);
  CHECK(body.find("_mlx_float_to_bf16(") != std::string::npos);
}

TEST_CASE("literal bfloat16_t locals and casts round like Metal") {
  // kda_glue_pre/post (2026-11-13 device dump): the literal bfloat16_t type
  // was a plain float, so every cast, local initializer and reassignment
  // skipped the bf16 rounding Metal applies, and only the output store
  // rounded (6755 of 12288 q elements differed from the Metal-faithful
  // reference). Mirrors the typename path: cast, declaration, reassignment
  // and a comma declarator list all round.
  const char* source =
      "[[kernel]] void glue_literal(\n"
      "    const device float* in [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint c = thread_position_in_grid.x;\n"
      "  float a = in[c];\n"
      "  bfloat16_t cq = (bfloat16_t)a, ck = (bfloat16_t)(a * 2.0f);\n"
      "  bfloat16_t sq_ = cq * ck;\n"
      "  float yq = (float)(bfloat16_t)((float)cq * (float)sq_);\n"
      "  sq_ = yq + 1.0f;\n"
      "  y[c] = yq + sq_;\n"
      "}\n";
  auto glsl = translate(source, 1);
  const auto body = glsl.substr(glsl.find("void main()"));
  CHECK(paren_balance(glsl) == 0);
  CHECK(body.find("float cq = _mlx_bf16_round_trip(") != std::string::npos);
  CHECK(body.find("float ck = _mlx_bf16_round_trip(") != std::string::npos);
  CHECK(body.find("float sq_ = _mlx_bf16_round_trip(cq * ck)") != std::string::npos);
  CHECK(body.find("float yq = _mlx_bf16_round_trip(") != std::string::npos);
  CHECK(body.find("sq_ = _mlx_bf16_round_trip(yq + 1.0") != std::string::npos);
  CHECK(body.find("bfloat16_t") == std::string::npos);
}

TEST_CASE("literal bfloat16_t comma declarators without a space stay valid") {
  // Review finding: `bfloat16_t a = x,b = y;` must not become `bfloat16_tb`.
  const char* source =
      "[[kernel]] void glue_nospace(\n"
      "    const device float* in [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint c = thread_position_in_grid.x;\n"
      "  float x = in[c];\n"
      "  bfloat16_t a = x,b = x * 2.0f;\n"
      "  y[c] = a + b;\n"
      "}\n";
  auto glsl = translate(source, 1);
  const auto body = glsl.substr(glsl.find("void main()"));
  CHECK(body.find("bfloat16_tb") == std::string::npos);
  CHECK(body.find("float a = _mlx_bf16_round_trip(x)") != std::string::npos);
  CHECK(body.find("float b = _mlx_bf16_round_trip(x * 2.0f)") != std::string::npos);
}

TEST_CASE("literal bfloat16_t rounding does not leak into a float local of the same name") {
  // Review finding: `{ bfloat16_t v ... } { float v ... }` must not round the
  // float block (rounding is scoped to the bf16 declaration's block).
  const char* source =
      "[[kernel]] void glue_scope(\n"
      "    const device float* in [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint c = thread_position_in_grid.x;\n"
      "  {\n"
      "    bfloat16_t v = in[c];\n"
      "    y[c] = v;\n"
      "  }\n"
      "  {\n"
      "    float v = in[c + 1];\n"
      "    v = v * 3.0f;\n"
      "    y[c + 1] = v;\n"
      "  }\n"
      "}\n";
  auto glsl = translate(source, 1);
  const auto body = glsl.substr(glsl.find("void main()"));
  CHECK(body.find("float v = _mlx_bf16_round_trip(") != std::string::npos);
  CHECK(body.find("v = v * 3.0f;") != std::string::npos);
  CHECK(body.find("_mlx_bf16_round_trip(v * 3.0f)") == std::string::npos);
}

TEST_CASE("bf16 local reassignment rounds in its own scope only") {
  // Review finding: with a same-name float elsewhere, `v = e;` in the bf16
  // scope must still round, and the float scope must stay untouched.
  const char* source =
      "[[kernel]] void glue_scope2(\n"
      "    const device float* in [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint c = thread_position_in_grid.x;\n"
      "  {\n"
      "    bfloat16_t v = in[c];\n"
      "    v = v * 2.0f;\n"
      "    y[c] = v;\n"
      "  }\n"
      "  {\n"
      "    float v = in[c + 1];\n"
      "    v = v * 3.0f;\n"
      "    y[c + 1] = v;\n"
      "  }\n"
      "}\n";
  auto glsl = translate(source, 1);
  const auto body = glsl.substr(glsl.find("void main()"));
  CHECK(body.find("v = _mlx_bf16_round_trip(v * 2.0f);") != std::string::npos);
  CHECK(body.find("v = v * 3.0f;") != std::string::npos);
  CHECK(body.find("_mlx_bf16_round_trip(v * 3.0f)") == std::string::npos);
}

TEST_CASE("bf16 scope end ignores braces inside comments") {
  const char* source =
      "[[kernel]] void glue_scope3(\n"
      "    const device float* in [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint c = thread_position_in_grid.x;\n"
      "  {\n"
      "    bfloat16_t v = in[c];\n"
      "    // closing brace in a line comment }\n"
      "    v = v * 2.0f;\n"
      "    y[c] = v;\n"
      "  }\n"
      "}\n";
  auto glsl = translate(source, 1);
  const auto body = glsl.substr(glsl.find("void main()"));
  CHECK(body.find("v = _mlx_bf16_round_trip(v * 2.0f);") != std::string::npos);
}

TEST_CASE("a brace inside a block comment does not truncate the kernel body") {
  const char* source =
      "[[kernel]] void glue_block_comment(\n"
      "    const device float* in [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint c = thread_position_in_grid.x;\n"
      "  {\n"
      "    bfloat16_t v = in[c];\n"
      "    // a closing brace } in a line comment\n"
      "    /* and one } in a block comment */\n"
      "    v = v * 2.0f;\n"
      "    y[c] = v;\n"
      "  }\n"
      "}\n";
  auto glsl = translate(source, 1);
  const auto body = glsl.substr(glsl.find("void main()"));
  CHECK(body.find("v = _mlx_bf16_round_trip(v * 2.0f);") != std::string::npos);
  CHECK(body.find("_mlx_arg1[c] = float(v);") != std::string::npos);
}

TEST_CASE("a comma inside a comment in a bf16 initializer is not a declarator separator") {
  // Review finding: `bfloat16_t v = in[c] /* comma, here */;` lost its ')'.
  const char* source =
      "[[kernel]] void glue_comment_comma(\n"
      "    const device float* in [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint c = thread_position_in_grid.x;\n"
      "  bfloat16_t v = in[c] /* comma, here */;\n"
      "  y[c] = v;\n"
      "}\n";
  auto glsl = translate(source, 1);
  const auto body = glsl.substr(glsl.find("void main()"));
  CHECK(body.find("float v = _mlx_bf16_round_trip(_mlx_arg0[c] );") != std::string::npos);
}

TEST_CASE("a parenthesis in a kernel-signature comment is ignored") {
  const char* source =
      "[[kernel]] void /* ignored unmatched ( */ // ignored line ( too\n"
      "    glue_signature_comment(\n"
      "    device float* y [[buffer(0)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  y[thread_position_in_grid.x] = 1.0f;\n"
      "}\n";
  const auto glsl = translate(source, 1);
  CHECK(glsl.find("void main()") != std::string::npos);
}

namespace {
// Real compiler check: skipped (stated in the log) only where glslc is absent.
bool glslc_present() {
  return std::system("command -v glslc >/dev/null 2>&1") == 0;
}
int glslc_status(const std::string& glsl, const std::string& tag) {
  const std::string base = "/tmp/omarchy-translate-test-" + tag;
  {
    std::ofstream out(base + ".comp");
    out << glsl;
  }
  const std::string cmd =
      "glslc -fshader-stage=compute --target-env=vulkan1.2 " + base +
      ".comp -o /dev/null 2> " + base + ".err";
  return std::system(cmd.c_str());
}
} // namespace

TEST_CASE("literal bfloat16_t locals on float-only buffers compile with glslc") {
  // Review finding: the body calls _mlx_bf16_round_trip but nothing declared
  // the helper or the int16 extension because no buffer is bfloat16.
  if (!glslc_present()) {
    MESSAGE("glslc not found: skipping the compile check");
    return;
  }
  const char* source =
      "[[kernel]] void glue_glslc(\n"
      "    const device float* in [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint c = thread_position_in_grid.x;\n"
      "  bfloat16_t v = in[c] * 2.0f;\n"
      "  v = v + in[c + 1];\n"
      "  y[c] = v;\n"
      "}\n";
  CHECK(glslc_status(translate(source, 1), "literal-locals") == 0);
}

namespace {
// Run the emitted GLSL helpers and kernel body as C++ (GLSL scalar subset)
// and print `mismatches=N`. `declarations` defines the buffers the body names;
// `driver` is the body of main(), calling kernel_main() per invocation.
int run_emitted_as_cpp(
    const std::string& glsl,
    const std::string& driver,
    const std::string& tag) {
  std::string helpers;
  std::size_t at = 0;
  while ((at = glsl.find("_mlx_", at)) != std::string::npos) {
    const auto begin = glsl.rfind('\n', at) + 1;
    const auto end = glsl.find('\n', at);
    const auto line = glsl.substr(begin, end - begin);
    if ((line.rfind("float _mlx_", 0) == 0 || line.rfind("uint16_t _mlx_", 0) == 0)) {
      helpers += line + "\n";
    }
    at = end;
  }
  const auto main_at = glsl.find("void main() {\n");
  REQUIRE(main_at != std::string::npos);
  auto body = glsl.substr(main_at + 14, glsl.rfind('}') - (main_at + 14));
  // The grid guard (`if (gl_GlobalInvocationID.x >= 640u ...) return;`) would
  // skip most of the 65536 invocations; the driver sizes its own grid.
  const auto guard = body.find("if (gl_GlobalInvocationID.x >=");
  REQUIRE(guard != std::string::npos);
  body.erase(guard, body.find('\n', guard) - guard);
  const std::string base = "/tmp/omarchy-translate-test-" + tag;
  {
    std::ofstream out(base + ".cpp");
    out << "#include <bit>\n#include <cstdint>\n#include <cstdio>\n"
           "using uint = uint32_t;\n"
           "static struct { uint x, y, z; } gl_GlobalInvocationID;\n"
           "static float uintBitsToFloat(uint v) { return std::bit_cast<float>(v); }\n"
           "static uint floatBitsToUint(float v) { return std::bit_cast<uint>(v); }\n"
           "static struct { uint16_t data[1 << 19]; } _b0, _b1;\n"
           "static float _mlx_arg0[1 << 19];\n"
           "static uint _mlx_arg1[1 << 19];\n"
        << helpers << "static void kernel_main() {\n" << body << "\n}\n"
        << "int main() { uint mismatches = 0;\n" << driver
        << "\nstd::printf(\"mismatches=%u\\n\", mismatches); return mismatches != 0; }\n";
  }
  const std::string compile = "c++ -std=c++20 -O1 " + base + ".cpp -o " + base +
      ".bin 2> " + base + ".err";
  REQUIRE(std::system(compile.c_str()) == 0);
  return std::system((base + ".bin > " + base + ".out").c_str());
}
} // namespace

TEST_CASE("bf16 storage bits survive copy, bitcast and local round trips") {
  // Every one of the 65536 bf16 patterns (NaN payloads and signs included)
  // must come out bit-identical from a bf16-to-bf16 copy, from as_type<ushort>
  // of a bf16 local, and from a bf16 local stored back.
  const char* copy_source =
      "[[kernel]] void glue_copy(\n"
      "    const device bfloat16_t* in [[buffer(0)]],\n"
      "    device bfloat16_t* y [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint c = thread_position_in_grid.x;\n"
      "  y[c] = in[c];\n"
      "}\n";
  const char* local_source =
      "[[kernel]] void glue_local(\n"
      "    const device bfloat16_t* in [[buffer(0)]],\n"
      "    device bfloat16_t* y [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint c = thread_position_in_grid.x;\n"
      "  bfloat16_t v = in[c];\n"
      "  y[c] = v;\n"
      "}\n";
  const char* bitcast_source =
      "[[kernel]] void glue_bitcast(\n"
      "    const device bfloat16_t* in [[buffer(0)]],\n"
      "    device uint* y [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint c = thread_position_in_grid.x;\n"
      "  bfloat16_t g = in[c];\n"
      "  y[c] = as_type<ushort>(g);\n"
      "}\n";
  const std::string fill =
      "for (uint p = 0; p < 65536; ++p) _b0.data[p] = uint16_t(p);\n"
      "for (uint p = 0; p < 65536; ++p) { gl_GlobalInvocationID.x = p; kernel_main();\n";
  CHECK(run_emitted_as_cpp(translate(copy_source, 1),
      fill + "if (_b1.data[p] != p) ++mismatches; }", "bits-copy") == 0);
  CHECK(run_emitted_as_cpp(translate(local_source, 1),
      fill + "if (_b1.data[p] != p) ++mismatches; }", "bits-local") == 0);
  CHECK(run_emitted_as_cpp(translate(bitcast_source, 1),
      fill + "if (_mlx_arg1[p] != p) ++mismatches; }", "bits-bitcast") == 0);
}

TEST_CASE("f32 to bf16 narrowing keeps NaN a NaN and rounds everything else") {
  // The bias-add rounding carries into the exponent for NaN payloads:
  // 0x7f800001 became +inf and 0x7fffffff became -0. For every float with a
  // 16-bit high half p and low half in {0, 1, 0x7fff, 0x8000, 0x8001, 0xffff}:
  // NaN in gives NaN out; anything else equals round-to-nearest-even.
  const char* source =
      "[[kernel]] void glue_narrow(\n"
      "    const device float* in [[buffer(0)]],\n"
      "    device bfloat16_t* y [[buffer(1)]],\n"
      "    uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
      "  uint c = thread_position_in_grid.x;\n"
      "  y[c] = in[c];\n"
      "}\n";
  const std::string driver = R"(
const uint lows[6] = {0u, 1u, 0x7fffu, 0x8000u, 0x8001u, 0xffffu};
for (uint p = 0; p < 65536; ++p) for (uint k = 0; k < 6; ++k) {
  const uint bits = (p << 16) | lows[k];
  const uint idx = p * 6 + k;
  _mlx_arg0[idx] = std::bit_cast<float>(bits);
  gl_GlobalInvocationID.x = idx; kernel_main();
  uint hi = bits >> 16; const uint lo = bits & 0xffffu;
  const bool nan_in = (bits & 0x7fffffffu) > 0x7f800000u;
  const uint got = _b1.data[idx];
  bool ok;
  if (nan_in) ok = (got & 0x7fffu) > 0x7f80u;
  else { if (lo > 0x8000u || (lo == 0x8000u && (hi & 1u))) hi += 1u; ok = got == hi; }
  if (!ok) ++mismatches;
})";
  CHECK(run_emitted_as_cpp(translate(source, 1), driver, "narrow") == 0);
}

TEST_CASE("a header comment with an unbalanced paren does not break the helper scan") {
  // The helper-signature scan is comment-blind: it matches plain word pairs
  // inside // comments and its find('(') can land on a paren inside comment
  // prose (the fma-emulation header ships comments like these). When the
  // delimiter scan starts mid-comment, the balancing close paren that lives
  // in a LATER comment must still be reachable: the scan must not begin
  // inside a comment at all. Kernel text itself is balanced.
  const char* source =
      "// fast path (v2). blend_bits/round_bits\n"
      "// they differ from emulation only on odd (the GPU\n"
      "// flushes denormals). fallback keeps the slow path\n"
      "[[kernel]] void comment_scan1(\n"
      "    const device float* in [[buffer(0)]],\n"
      "    device float* y [[buffer(1)]]) {\n"
      "  y[0] = in[0];\n"
      "}\n";
  auto glsl = translate(source, 1);
  CHECK(glsl.find("void main()") != std::string::npos);
}

TEST_CASE("a line comment naming a stage does not steal the helper signature") {
  // Review reproduction (PR #427): in the residual_norm_style fixture,
  // changing only the first comment to name a stage with a paren made the
  // helper-signature regex match the word pair inside the comment; the
  // comment paren was skipped and the name 'stage' was paired with the
  // next real function's parameters, refusing the kernel that base
  // accepted. A regex match inside a comment must be skipped entirely.
  std::string source(residual_norm_style);
  source.replace(0, std::strlen("// One row reduction stage.\n"),
                 "// One row reduction stage (shared scratch).\n");
  auto glsl = translate(source, 1);
  CHECK(glsl.find("threadgroup") == std::string::npos);
  CHECK(glsl.find("float msv_row_inv_rms(float acc, float eps, int axis, uint") !=
        std::string::npos);
  CHECK(glsl.find("msv_row_inv_rms(acc1, 1e-6f, 2560, 0u, lid)") !=
        std::string::npos);
}

TEST_CASE("a block comment naming a stage does not steal the helper signature") {
  // Same reproduction through a block comment: the word pair inside
  // comment prose must not become a helper name paired with the next
  // real function's parameters.
  std::string source(residual_norm_style);
  source.replace(0, std::strlen("// One row reduction stage.\n"),
                 "/* One row reduction stage (shared scratch). */\n");
  auto glsl = translate(source, 1);
  CHECK(glsl.find("threadgroup") == std::string::npos);
  CHECK(glsl.find("float msv_row_inv_rms(float acc, float eps, int axis, uint") !=
        std::string::npos);
  CHECK(glsl.find("msv_row_inv_rms(acc1, 1e-6f, 2560, 0u, lid)") !=
        std::string::npos);
}

TEST_CASE("a quoted include containing a comment marker stays a string") {
  // Review reproduction (PR #427): prepending this quoted include to the
  // unchanged residual fixture is accepted by base. Lexing the /* inside
  // the string as a comment opener swallowed the following helper, its
  // threadgroup parameters stayed unspecialized, and translation threw.
  // String and character literals (escapes included) must never be
  // scanned for comment markers; the slash-slash control accepts too.
  std::string source = "#include \"dir/*marker.h\"\n";
  source += residual_norm_style;
  auto glsl = translate(source, 1);
  CHECK(glsl.find("threadgroup") == std::string::npos);
  CHECK(glsl.find("msv_row_inv_rms(acc1, 1e-6f, 2560, 0u, lid)") !=
        std::string::npos);
  std::string control = "#include \"dir//marker.h\"\n";
  control += residual_norm_style;
  auto glsl_control = translate(control, 1);
  CHECK(glsl_control.find("threadgroup") == std::string::npos);
}

TEST_CASE("a digit separator in a define does not open a char literal") {
  // Review reproduction (PR #427): prepending a C++14 numeric separator
  // to the unchanged residual fixture is accepted by base. Lexing the
  // separator apostrophe as a character-literal opener masked every
  // following helper signature, its threadgroup parameters stayed
  // unspecialized, and translation threw. The odd separator count
  // 1'000 is the failing shape; 1'000'000 already passed because the
  // second apostrophe closed the phantom literal. A real character
  // literal must keep working.
  std::string source = "#define COUNT 1'000\n";
  source += residual_norm_style;
  auto glsl = translate(source, 1);
  CHECK(glsl.find("threadgroup") == std::string::npos);
  CHECK(glsl.find("msv_row_inv_rms(acc1, 1e-6f, 2560, 0u, lid)") !=
        std::string::npos);
  std::string even = "#define COUNT 1'000'000\n";
  even += residual_norm_style;
  CHECK(translate(even, 1).find("threadgroup") == std::string::npos);
  std::string literal =
      "[[kernel]] void char_literal_probe(\n"
      "    const device float* in [[buffer(0)]],\n"
      "    device float* out [[buffer(1)]],\n"
      "    uint lid [[thread_position_in_threadgroup]]) {\n"
      "  out[0] = (in[lid] == 'x') ? in[0] : in[1];\n"
      "}\n";
  auto glsl_literal = translate(literal, 1);
  CHECK(glsl_literal.find("'x'") != std::string::npos);
}

TEST_CASE("a raw string with an embedded quote stays one token") {
  // Reviewer-noted gap: the lexer has no raw-string state, so the quote
  // inside R"tag(")tag" toggles a plain string and the unterminated
  // remainder masks every following helper. The raw token must be
  // scanned to its closing delimiter instead.
  std::string source = "#define DOC R\"tag(\")tag\"\n";
  source += residual_norm_style;
  auto glsl = translate(source, 1);
  CHECK(glsl.find("threadgroup") == std::string::npos);
  CHECK(glsl.find("msv_row_inv_rms(acc1, 1e-6f, 2560, 0u, lid)") !=
        std::string::npos);
}

TEST_CASE("hex separators, suffixes and digit-adjacent literals keep their roles") {
  // Separators after hex digits inside a number token stay separators:
  // the helpers that follow each define stay visible to the scan.
  std::string hex = "#define HEXMASK 0x1'a'f\n";
  hex += residual_norm_style;
  auto glsl_hex = translate(hex, 1);
  CHECK(glsl_hex.find("threadgroup") == std::string::npos);
  CHECK(glsl_hex.find("msv_row_inv_rms(acc1, 1e-6f, 2560, 0u, lid)") !=
        std::string::npos);
  std::string hex_pairs = "#define ROUTE 0xDE'AD'BEEF\n";
  hex_pairs += residual_norm_style;
  CHECK(translate(hex_pairs, 1).find("threadgroup") == std::string::npos);
  std::string suffixed = "#define COUNTU 1'000u\n";
  suffixed += residual_norm_style;
  CHECK(translate(suffixed, 1).find("threadgroup") == std::string::npos);
  // A digit-adjacent character literal must stay a literal: the paren
  // or the operator ends the number before the quote opens.
  std::string neighbors =
      "[[kernel]] void separator_neighbors(\n"
      "    const device float* in [[buffer(0)]],\n"
      "    device float* out [[buffer(1)]],\n"
      "    uint lid [[thread_position_in_threadgroup]]) {\n"
      "  out[0] = (in[lid] == '1' ? 1.0f : 0.0f) + ('2' == '2' ? 1.0f : 0.0f);\n"
      "  out[1] = 1+'\\'' ? in[0] : in[1];\n"
      "}\n";
  auto glsl_neighbors = translate(neighbors, 1);
  CHECK(glsl_neighbors.find("'1'") != std::string::npos);
  CHECK(glsl_neighbors.find("'2'") != std::string::npos);
  CHECK(glsl_neighbors.find("'\\''") != std::string::npos);
}

TEST_CASE("a digit ending an identifier does not start a number token") {
  // Review reproduction (PR #427): prepending these defines to the
  // unchanged residual fixture is accepted by base. EMPTY1 is an
  // identifier, not a preprocessing number: treating its final digit
  // as a number start made the next apostrophe a phantom separator,
  // the paren reset the state, and the character literal in the
  // second define then ran unterminated across the rest of the text,
  // masking every helper. An identifier with a trailing digit followed
  // by a normal helper stays visible.
  std::string source = "#define EMPTY1\n#define DOC EMPTY1'('\n";
  source += residual_norm_style;
  auto glsl = translate(source, 1);
  CHECK(glsl.find("threadgroup") == std::string::npos);
  CHECK(glsl.find("msv_row_inv_rms(acc1, 1e-6f, 2560, 0u, lid)") !=
        std::string::npos);
  std::string plain = "#define EMPTY1\n";
  plain += residual_norm_style;
  auto glsl_plain = translate(plain, 1);
  CHECK(glsl_plain.find("threadgroup") == std::string::npos);
  CHECK(glsl_plain.find("msv_row_inv_rms(acc1, 1e-6f, 2560, 0u, lid)") !=
        std::string::npos);
}

TEST_CASE("an identifier ending in R before a quote stays a plain string") {
  // Review reproduction (PR #427): prepending these defines to the
  // unchanged residual fixture is accepted by base. STR is an
  // identifier, so STR"hello" is an ordinary string; the lexer took
  // the identifier's final R as a raw-string prefix, searched past
  // the closing quote for a delimiter, found the next helper's open
  // paren, and masked the rest of the text when the bogus terminator
  // never came. A raw prefix must start a token: the real raw forms
  // with R and u8R keep working, the quoted-paren control accepts,
  // and the separator define stays a number.
  std::string source = "#define STR\n#define DOC STR\"hello\"\n";
  source += residual_norm_style;
  auto glsl = translate(source, 1);
  CHECK(glsl.find("threadgroup") == std::string::npos);
  CHECK(glsl.find("msv_row_inv_rms(acc1, 1e-6f, 2560, 0u, lid)") !=
        std::string::npos);
  std::string quoted = "#define DOC STR\"(hello)\"\n";
  quoted += residual_norm_style;
  CHECK(translate(quoted, 1).find("threadgroup") == std::string::npos);
  std::string real_raw =
      "#define DOC2 R\"x(has \" quote ( and ) inside)x\"\n";
  real_raw += residual_norm_style;
  auto glsl_raw = translate(real_raw, 1);
  CHECK(glsl_raw.find("threadgroup") == std::string::npos);
  CHECK(glsl_raw.find("msv_row_inv_rms(acc1, 1e-6f, 2560, 0u, lid)") !=
        std::string::npos);
  std::string u8_raw = "#define DOC3 u8R\"(quote \" here)\"\n";
  u8_raw += residual_norm_style;
  CHECK(translate(u8_raw, 1).find("threadgroup") == std::string::npos);
  std::string separator = "#define COUNT 1'000\n";
  separator += residual_norm_style;
  CHECK(translate(separator, 1).find("threadgroup") == std::string::npos);
}

TEST_CASE("a line splice inside an identifier keeps the identifier whole") {
  // Review reproduction (PR #427): a backslash-newline joins tokens,
  // so EMPTY spliced as EMPTY\<newline>1 is one identifier and
  // EMPTY\<newline>1'(' carries a character literal. Lexing the digit
  // across the splice as a number start made the next apostrophe a
  // phantom separator and the paren left an unterminated literal that
  // masked every helper. Both prepended shapes are accepted by the
  // parent and must translate here.
  std::string source = "#define EMPTY\\\n1\n#define DOC EMPTY\\\n1'('\n";
  source += residual_norm_style;
  auto glsl = translate(source, 1);
  CHECK(glsl.find("threadgroup") == std::string::npos);
  CHECK(glsl.find("msv_row_inv_rms(acc1, 1e-6f, 2560, 0u, lid)") !=
        std::string::npos);
}

TEST_CASE("a line splice inside a raw-looking identifier stays a plain string") {
  // Review reproduction (PR #427): ST spliced as ST\<newline>R ends in
  // R on the far side of the splice, so ST\<newline>R"x(hello" is an
  // identifier followed by an ordinary string. Reading the spliced R
  // as a raw-string prefix sent the delimiter search past the string
  // into the next helper and masked the rest of the text. The parent
  // accepts the prepended shape and it must translate here.
  std::string source = "#define DOC ST\\\nR\"x(hello\"\n";
  source += residual_norm_style;
  auto glsl = translate(source, 1);
  CHECK(glsl.find("threadgroup") == std::string::npos);
  CHECK(glsl.find("msv_row_inv_rms(acc1, 1e-6f, 2560, 0u, lid)") !=
        std::string::npos);
}

TEST_CASE("splices inside strings, chars, comments and plain defines keep working") {
  // Controls: a spliced identifier without any quote, a string and a
  // character literal continued across a splice by the existing
  // escape handling, and a comment whose spliced newline ends it
  // today. Each leaves the residual fixture helpers visible.
  std::string spliced_id = "#define EMPTY\\\n1\n";
  spliced_id += residual_norm_style;
  CHECK(translate(spliced_id, 1).find("threadgroup") == std::string::npos);
  std::string spliced_str = "#define S \"a\\\nz\"\n";
  spliced_str += residual_norm_style;
  CHECK(translate(spliced_str, 1).find("threadgroup") == std::string::npos);
  std::string spliced_char = "#define Q 'a\\\n'\n";
  spliced_char += residual_norm_style;
  CHECK(translate(spliced_char, 1).find("threadgroup") == std::string::npos);
  std::string spliced_comment = "// note\\\nmore\n";
  spliced_comment += residual_norm_style;
  CHECK(translate(spliced_comment, 1).find("threadgroup") == std::string::npos);
}

TEST_CASE("a punctuation raw-string delimiter is a valid delimiter") {
  // Reviewer finding on PR #427: the C++ raw-string delimiter may be
  // any characters except space, parentheses and backslash, so the
  // punctuation delimiter ++ is standards-valid and the parent
  // accepts this prepended shape. Requiring identifier characters in
  // the delimiter fell the raw string through to the plain-string
  // path, the embedded quote opened string state, and translation
  // died on the later helper.
  std::string source = "#define DOC R\"++(has \" quote)++\"\n";
  source += residual_norm_style;
  auto glsl = translate(source, 1);
  CHECK(glsl.find("threadgroup") == std::string::npos);
  CHECK(glsl.find("msv_row_inv_rms(acc1, 1e-6f, 2560, 0u, lid)") !=
        std::string::npos);
}
