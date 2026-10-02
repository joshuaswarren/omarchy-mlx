// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Barrier scope masks: the omarchy backend records only compute dispatches
// and transfer-style fills/copies, so the default dependency barrier uses
// the narrow COMPUTE|TRANSFER stage set. The previous ALL_COMMANDS mask
// forced Mesa's heaviest VM/cache maintenance on every barrier; the cost
// showed up as 41-196 ms stalls on barrier-flagged Convolutions in the
// Kokoro per-dispatch profile (15,412 dispatches over 78 submissions,
// ~9,253 barriers emitted, 4.5 s of GPU-domain span inside the 5.0 s wall
// — see receipts/2026-10-01-opcost-microbench).
//
// The mask logic is a constexpr free function in the header, so this test
// needs no GPU device and no backend link.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "mlx/backend/omarchy/encoder.h"

using namespace mlx::core;

namespace {

constexpr auto kCompute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
constexpr auto kTransfer = VK_PIPELINE_STAGE_TRANSFER_BIT;
constexpr auto kBoth = kCompute | kTransfer;
constexpr auto kMem =
    VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

} // namespace

TEST_CASE("narrow scope covers exactly compute+transfer, both directions") {
  auto masks = omarchy::barrier_masks_for_scope(
      omarchy::CommandEncoder::BarrierScope::ComputeTransfer);
  CHECK(masks.srcStages == kBoth);
  CHECK(masks.dstStages == kBoth);
  CHECK(masks.srcAccess == kMem);
  CHECK(masks.dstAccess == kMem);
}

TEST_CASE("all scope is the ALL_COMMANDS rollback with memory masks") {
  auto masks = omarchy::barrier_masks_for_scope(
      omarchy::CommandEncoder::BarrierScope::AllCommands);
  CHECK(masks.srcStages == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  CHECK(masks.dstStages == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  CHECK(masks.srcAccess == kMem);
  CHECK(masks.dstAccess == kMem);
}

TEST_CASE("narrow scope is a strict subset of all scope") {
  auto narrow = omarchy::barrier_masks_for_scope(
      omarchy::CommandEncoder::BarrierScope::ComputeTransfer);
  auto all = omarchy::barrier_masks_for_scope(
      omarchy::CommandEncoder::BarrierScope::AllCommands);
  CHECK((narrow.srcStages & all.srcStages) == narrow.srcStages);
  CHECK((narrow.dstStages & all.dstStages) == narrow.dstStages);
  CHECK((narrow.srcStages & ~all.srcStages) == 0u);
  CHECK((narrow.dstStages & ~all.dstStages) == 0u);
}