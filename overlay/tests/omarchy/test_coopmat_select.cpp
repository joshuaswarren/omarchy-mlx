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

TEST_CASE("no coopmat device advertises nothing") {
  const std::vector<CoopmatShape> adv;
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape()));
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape()));
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
