// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Selection counter for the fused EXL3 MoE arm (M3d). Its own translation
// unit at the global namespace so the definition is unambiguous: the
// counter lives in an anonymous namespace here and the accessor is the
// only symbol that crosses the boundary. The doctest reads the accessor
// to prove the fused arm was actually selected.

#include "mlx/backend/omarchy/compute.h"

#include <atomic>

namespace mlx::core::omarchy {
namespace {
std::atomic<uint64_t> exl3_moe_fused_ops{0};
// Composed-arm decoded f32 bank materializations. fast::exl3_moe bumps
// this once per call the composed graph serves; the tests assert it stays
// zero on the fused arm (no full expert bank may materialize there).
std::atomic<uint64_t> decoded_bank_allocs{0};
} // namespace

uint64_t exl3_moe_fused_dispatch_count() {
  return exl3_moe_fused_ops.load();
}

void exl3_moe_fused_count_fetch_add() {
  exl3_moe_fused_ops.fetch_add(1);
}

uint64_t exl3_moe_decoded_bank_allocations() {
  return decoded_bank_allocs.load();
}

void exl3_moe_decoded_bank_allocations_fetch_add() {
  decoded_bank_allocs.fetch_add(1);
}

void exl3_moe_decoded_bank_allocations_reset() {
  decoded_bank_allocs.store(0);
}
} // namespace mlx::core::omarchy
