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

TEST_CASE("qmm_prefill_coopmat_shape names the shape each output dtype emits") {
  const CoopmatShape f16 = omarchy::qmm_prefill_coopmat_shape(true);
  CHECK(f16.m == 8u);
  CHECK(f16.n == 8u);
  CHECK(f16.k == 8u);
  // fp16 output: activations stay fp16 (A), weights dequantize to fp32 (B) -
  // a mixed matmad.
  CHECK(f16.a == CoopmatType::Float16);
  CHECK(f16.b == CoopmatType::Float32);
  CHECK(f16.c == CoopmatType::Float32);
  CHECK(f16.result == CoopmatType::Float32);
  CHECK(f16.scope == CoopmatScope::Subgroup);
  CHECK(f16.saturating == false);

  const CoopmatShape bf16 = omarchy::qmm_prefill_coopmat_shape(false);
  // bf16 output: activations are widened to fp32 before the matmad, so A is
  // fp32 and the shape is same-operand fp32xfp32.
  CHECK(bf16.a == CoopmatType::Float32);
  CHECK(bf16.b == CoopmatType::Float32);
  CHECK(bf16.c == CoopmatType::Float32);
  CHECK(bf16.result == CoopmatType::Float32);
  CHECK(bf16 == same_operand(CoopmatType::Float32));
}

TEST_CASE("AGX3B advertises the bf16 route's shape but not the fp16 route's") {
  const std::vector<CoopmatShape> adv = agx3b_shapes();
  // bf16 output needs fp32xfp32: advertised -> coopmat stays (the 2.19x
  // prefill win is kept).
  CHECK(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape(false)));
  // fp16 output needs the mixed fp16-A/fp32-B shape: not advertised -> the
  // route must fall back to the non-coopmat tile kernel.
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape(true)));
}

TEST_CASE("a device that lists the mixed shape keeps the fp16 coopmat route") {
  std::vector<CoopmatShape> adv = agx3b_shapes();
  adv.push_back(omarchy::qmm_prefill_coopmat_shape(true)); // mixed fp16/fp32
  CHECK(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape(true)));
  CHECK(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape(false)));
}

TEST_CASE("no coopmat device advertises nothing") {
  const std::vector<CoopmatShape> adv;
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape(true)));
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      adv, omarchy::qmm_prefill_coopmat_shape(false)));
}

TEST_CASE("the match is exact in every field") {
  const CoopmatShape want = omarchy::qmm_prefill_coopmat_shape(false);

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

TEST_CASE("the mixed fp16 shape is distinct from same-operand fp16") {
  // fp16xfp16 (what cooperative_matrix_f16_8 summarizes) is not the fp16 qmm
  // route's shape: that route is fp16-A/fp32-B. A device offering only
  // fp16xfp16 must not satisfy the fp16 qmm route.
  const std::vector<CoopmatShape> f16f16 = {same_operand(CoopmatType::Float16)};
  CHECK_FALSE(omarchy::cooperative_matrix_advertises(
      f16f16, omarchy::qmm_prefill_coopmat_shape(true)));
}
