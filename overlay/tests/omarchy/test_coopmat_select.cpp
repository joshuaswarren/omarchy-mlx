// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Host tests for the cooperative-matrix shape selector. No GPU, no Vulkan,
// no mlx linkage: coopmat_select.cpp is deliberately free of all three, so
// these run in CI on any machine. They pin the contract primitives.cpp relies
// on when it decides whether the quantized-matmul prefill coopmat route is
// legal on the running device.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "coopmat_select.h"

#include <vector>

namespace omarchy = mlx::core::omarchy;
using omarchy::CoopmatScope;
using omarchy::CoopmatShape;
using omarchy::CoopmatType;

namespace {

// An 8x8x8 subgroup shape with matching A/B operands and an fp32
// accumulator/result, non-saturating: the shape the booleans summarize.
CoopmatShape same_operand(CoopmatType t) {
  return CoopmatShape{8u, 8u, 8u, t, t, CoopmatType::Float32,
      CoopmatType::Float32, CoopmatScope::Subgroup, false};
}

// What Apple AGX3B (mesa #299) advertises: fp32xfp32 and fp16xfp16 8x8x8,
// both fp32 accumulate, subgroup, non-saturating. No mixed-operand shape.
std::vector<CoopmatShape> agx3b_shapes() {
  return {same_operand(CoopmatType::Float32),
      same_operand(CoopmatType::Float16)};
}

} // namespace

TEST_CASE("qmm_prefill_coopmat_shape names the shape both output dtypes emit") {
  // Both builds widen the activations to an fp32 A matrix: fp32xfp32 8x8x8,
  // fp32 accumulate, subgroup, non-saturating. One shape covers f16 and
  // bf16 output, so the selector takes no dtype argument any more.
  const CoopmatShape f16 = omarchy::qmm_prefill_coopmat_shape();
  CHECK(f16.m == 8u);
  CHECK(f16.n == 8u);
  CHECK(f16.k == 8u);
  CHECK(f16.a == CoopmatType::Float32);
  CHECK(f16.b == CoopmatType::Float32);
  CHECK(f16.c == CoopmatType::Float32);
  CHECK(f16.result == CoopmatType::Float32);
  CHECK(f16.scope == CoopmatScope::Subgroup);
  CHECK(f16.saturating == false);
  CHECK(f16 == same_operand(CoopmatType::Float32));

  const CoopmatShape bf16 = omarchy::qmm_prefill_coopmat_shape();
  CHECK(bf16 == f16);
  CHECK(bf16 == same_operand(CoopmatType::Float32));
}

TEST_CASE("AGX3B advertises the shape of both qmm prefill routes") {
  const std::vector<CoopmatShape> adv = agx3b_shapes();
  CHECK(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape()));
  CHECK(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape()));
}

TEST_CASE("a device without the fp32xfp32 shape declines both routes") {
  const std::vector<CoopmatShape> adv = {same_operand(CoopmatType::Float16)};
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape()));
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape()));
}

TEST_CASE("qmm_prefill_coopmat_shape_f16x16 names the CMAT16 shape") {
  // The -DCMAT16 build emits fp16 A and B operands with the fp32
  // accumulator at 16x16x16 subgroup scope - the entry NVIDIA's coopmat
  // list carries and no fp32-operand shape.
  const CoopmatShape c16 = omarchy::qmm_prefill_coopmat_shape_f16x16();
  CHECK(c16.m == 16u);
  CHECK(c16.n == 16u);
  CHECK(c16.k == 16u);
  CHECK(c16.a == CoopmatType::Float16);
  CHECK(c16.b == CoopmatType::Float16);
  CHECK(c16.c == CoopmatType::Float32);
  CHECK(c16.result == CoopmatType::Float32);
  CHECK(c16.scope == CoopmatScope::Subgroup);
  CHECK(c16.saturating == false);
}

TEST_CASE("an NVIDIA-style list advertises the CMAT16 route only") {
  // NVIDIA 615.78.08: the same-operand 16x16x16 entries (fp16 and bf16)
  // and no fp32-operand shape of any size. The fp32 8x8x8 route declines
  // and the CMAT16 route is selected.
  auto same_operand_16 = [](CoopmatType t) {
    return CoopmatShape{16u, 16u, 16u, t, t, CoopmatType::Float32,
        CoopmatType::Float32, CoopmatScope::Subgroup, false};
  };
  const std::vector<CoopmatShape> adv = {
      same_operand_16(CoopmatType::Float16),
      same_operand_16(CoopmatType::BFloat16)};
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape()));
  CHECK(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape_f16x16()));
  // The bf16 twin is a different shape: the fp16-operand selector must
  // not alias onto it.
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      {same_operand_16(CoopmatType::BFloat16)},
      omarchy::qmm_prefill_coopmat_shape_f16x16()));
}

TEST_CASE("no coopmat device advertises nothing") {
  const std::vector<CoopmatShape> adv;
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape()));
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape_f16x16()));
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape()));
}

TEST_CASE("matmul_direct_coopmat_shape_c16 names the dense CMAT16 shapes") {
  // The -DCMAT16 dense direct build emits same-operand 16x16x16
  // matrices of the storage dtype with the fp32 accumulator and result.
  const CoopmatShape f16 =
      omarchy::matmul_direct_coopmat_shape_c16(CoopmatType::Float16);
  CHECK(f16.m == 16u);
  CHECK(f16.n == 16u);
  CHECK(f16.k == 16u);
  CHECK(f16.a == CoopmatType::Float16);
  CHECK(f16.b == CoopmatType::Float16);
  CHECK(f16.c == CoopmatType::Float32);
  CHECK(f16.result == CoopmatType::Float32);
  CHECK(f16.scope == CoopmatScope::Subgroup);
  CHECK(f16.saturating == false);
  const CoopmatShape bf16 =
      omarchy::matmul_direct_coopmat_shape_c16(CoopmatType::BFloat16);
  CHECK(bf16.a == CoopmatType::BFloat16);
  CHECK(bf16.b == CoopmatType::BFloat16);
  CHECK(bf16.c == CoopmatType::Float32);
  CHECK(bf16.result == CoopmatType::Float32);
}

TEST_CASE("an NVIDIA-style list advertises the dense CMAT16 routes") {
  // NVIDIA 615.78.08 lists the same-operand 16x16x16 entries (fp16 and
  // bf16): both dense CMAT16 routes are selected. A driver with only
  // the fp16 entry takes only the fp16 route.
  auto same_operand_16 = [](CoopmatType t) {
    return CoopmatShape{16u, 16u, 16u, t, t, CoopmatType::Float32,
        CoopmatType::Float32, CoopmatScope::Subgroup, false};
  };
  const std::vector<CoopmatShape> adv = {
      same_operand_16(CoopmatType::Float16),
      same_operand_16(CoopmatType::BFloat16)};
  CHECK(omarchy::cooperative_matrix_advertises(
      adv,
      omarchy::matmul_direct_coopmat_shape_c16(CoopmatType::Float16)));
  CHECK(omarchy::cooperative_matrix_advertises(
      adv,
      omarchy::matmul_direct_coopmat_shape_c16(CoopmatType::BFloat16)));
  const std::vector<CoopmatShape> f16_only = {
      same_operand_16(CoopmatType::Float16)};
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      f16_only,
      omarchy::matmul_direct_coopmat_shape_c16(CoopmatType::BFloat16)));
  // An 8x8x8-operand list never aliases onto the 16x16x16 route.
  const std::vector<CoopmatShape> eight = {
      CoopmatShape{8u, 8u, 8u, CoopmatType::Float16, CoopmatType::Float16,
          CoopmatType::Float32, CoopmatType::Float32,
          CoopmatScope::Subgroup, false}};
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      eight,
      omarchy::matmul_direct_coopmat_shape_c16(CoopmatType::Float16)));
}

TEST_CASE("RADV GFX1201 advertises the 16x16x16 bf16 route, not the 8x8x8 f32 one") {
  // Shapes as measured by
  // vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR on RADV GFX1201
  // (Mesa 26.3.1): 16x16x16 subgroup shapes only, bf16 and fp16 A/B
  // feeding fp32 accumulators, plus fp8/int8 entries that record as
  // Other. No 8x8x8 and no fp32 A/B operands.
  const std::vector<CoopmatShape> adv = {
      {16u, 16u, 16u, CoopmatType::Float16, CoopmatType::Float16,
       CoopmatType::Float16, CoopmatType::Float16, CoopmatScope::Subgroup,
       false},
      {16u, 16u, 16u, CoopmatType::Float16, CoopmatType::Float16,
       CoopmatType::Float32, CoopmatType::Float32, CoopmatScope::Subgroup,
       false},
      {16u, 16u, 16u, CoopmatType::BFloat16, CoopmatType::BFloat16,
       CoopmatType::BFloat16, CoopmatType::BFloat16, CoopmatScope::Subgroup,
       false},
      {16u, 16u, 16u, CoopmatType::BFloat16, CoopmatType::BFloat16,
       CoopmatType::Float32, CoopmatType::Float32, CoopmatScope::Subgroup,
       false},
      {16u, 16u, 16u, CoopmatType::Other, CoopmatType::Other,
       CoopmatType::Other, CoopmatType::Other, CoopmatScope::Subgroup,
       false},
  };
  CHECK(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat16_bf16_shape()));
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape()));
}

TEST_CASE("the 8x8x8 devices do not match the 16x16x16 route") {
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      agx3b_shapes(), omarchy::qmm_prefill_coopmat16_bf16_shape()));
}

TEST_CASE("the sdpa 16x16x16 shape matches the qmm 16x16x16 shape") {
  CHECK(
      omarchy::sdpa_causal_flash_coopmat16_bf16_shape() ==
      omarchy::qmm_prefill_coopmat16_bf16_shape());
  CHECK(omarchy::cooperative_matrix_advertises(
      {omarchy::sdpa_causal_flash_coopmat16_bf16_shape()},
      omarchy::sdpa_causal_flash_coopmat16_bf16_shape()));
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      agx3b_shapes(), omarchy::sdpa_causal_flash_coopmat16_bf16_shape()));
}

TEST_CASE("the 16x16x16 causal flash gate keeps capability mandatory under force-on") {
  // Unset defaults on where the shape is advertised and the q_len floor
  // and full-tile alignment hold.
  CHECK(omarchy::sdpa_causal_flash_coopmat16_enabled(
      nullptr, true, 512, true));
  // Kill switch wins over everything.
  CHECK_FALSE(omarchy::sdpa_causal_flash_coopmat16_enabled(
      "0", true, 512, true));
  // Force-on cannot conjure the hardware requirement: no advertised
  // shape, no dispatch, env=1 or not.
  CHECK_FALSE(omarchy::sdpa_causal_flash_coopmat16_enabled(
      "1", false, 512, true));
  CHECK_FALSE(omarchy::sdpa_causal_flash_coopmat16_enabled(
      nullptr, false, 512, true));
  // Ragged query tails overlap the last two workgroups' stores; the
  // route requires full 16-row tiles even when forced.
  CHECK_FALSE(omarchy::sdpa_causal_flash_coopmat16_enabled(
      nullptr, true, 512, false));
  CHECK_FALSE(omarchy::sdpa_causal_flash_coopmat16_enabled(
      "1", true, 512, false));
  // Below the q_len floor the route stays off unless explicitly forced.
  CHECK_FALSE(omarchy::sdpa_causal_flash_coopmat16_enabled(
      nullptr, true, 32, true));
  CHECK(omarchy::sdpa_causal_flash_coopmat16_enabled(
      "1", true, 32, true));
  CHECK(omarchy::sdpa_causal_flash_coopmat16_enabled(
      nullptr, true, 64, true));
}

TEST_CASE("the match is exact in every field") {
  const CoopmatShape want = omarchy::qmm_prefill_coopmat_shape();

  // A shape that differs in exactly one field must not match.
  auto differs = [&](CoopmatShape s) {
    return !omarchy::cooperative_matrix_advertises({s}, want);
  };

  CoopmatShape wrong_k = want;
  wrong_k.k = 16u;
  CHECK(differs(wrong_k));

  CoopmatShape wrong_scope = want;
  wrong_scope.scope = CoopmatScope::Workgroup;
  CHECK(differs(wrong_scope));

  CoopmatShape wrong_sat = want;
  wrong_sat.saturating = true;
  CHECK(differs(wrong_sat));

  CoopmatShape wrong_b = want;
  wrong_b.b = CoopmatType::Float16;
  CHECK(differs(wrong_b));

  CoopmatShape wrong_accum = want;
  wrong_accum.c = CoopmatType::BFloat16;
  CHECK(differs(wrong_accum));

  // The identical shape matches.
  CHECK(omarchy::cooperative_matrix_advertises({want}, want));
}

TEST_CASE("matmul_xe16_bf16_shape names the Intel/ANV 8x16x16 bf16 shape") {
  // ANV on Xe2 advertises only M8xN16 shapes: K16 bf16 operands with an
  // fp32 accumulator, subgroup scope, non-saturating.
  const CoopmatShape s = omarchy::matmul_xe16_bf16_shape();
  CHECK(s.m == 8u);
  CHECK(s.n == 16u);
  CHECK(s.k == 16u);
  CHECK(s.a == CoopmatType::BFloat16);
  CHECK(s.b == CoopmatType::BFloat16);
  CHECK(s.c == CoopmatType::Float32);
  CHECK(s.result == CoopmatType::Float32);
  CHECK(s.scope == CoopmatScope::Subgroup);
  CHECK(s.saturating == false);

  const std::vector<CoopmatShape> adv = {s};
  CHECK(omarchy::cooperative_matrix_advertises(
      adv, omarchy::matmul_xe16_bf16_shape()));
  // The QMM prefill Xe16 variant emits the same bf16 shape, so the
  // advertised list authorizes both routes.
  CHECK(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat16_xe_shape()));
  // An fp32 8x8x8 shape alone does not authorize the Xe16 kernel.
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      {same_operand(CoopmatType::Float32)},
      omarchy::matmul_xe16_bf16_shape()));
}

TEST_CASE("qmm_prefill_coopmat16_xe_shape names the bf16-operand Xe shape") {
  const CoopmatShape s = omarchy::qmm_prefill_coopmat16_xe_shape();
  CHECK(s.m == 8u);
  CHECK(s.n == 16u);
  CHECK(s.k == 16u);
  CHECK(s.a == CoopmatType::BFloat16);
  CHECK(s.b == CoopmatType::BFloat16);
  CHECK(s.c == CoopmatType::Float32);
  CHECK(s.result == CoopmatType::Float32);
  CHECK(s.scope == CoopmatScope::Subgroup);
  CHECK(s.saturating == false);

  const std::vector<CoopmatShape> adv = {s};
  CHECK(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat16_xe_shape()));
  // NVIDIA-style M16 fp16 shapes do not authorize the M8 kernel.
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      {CoopmatShape{16u, 16u, 16u, CoopmatType::Float16,
          CoopmatType::Float16, CoopmatType::Float32,
          CoopmatType::Float32, CoopmatScope::Subgroup, false}},
      omarchy::qmm_prefill_coopmat16_xe_shape()));
  // And the M8 bf16 shape does not alias onto the M16 CMAT16 route.
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape_f16x16()));
}
