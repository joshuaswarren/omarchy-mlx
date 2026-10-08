// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Batched GDN decode (B > 1, T = 1, no mask) runs the fused decode kernel
// over B * Hv heads. Each batch row must match the same row run alone at
// B = 1 bit for bit (output and state): the kernel does the same per-head
// arithmetic, and A_log / dt_bias are per model head (read at head % Hv),
// so a batch-offset bug in either shows up as a row mismatch. Every head
// gets distinct A_log / dt_bias for that reason. A third case checks the
// batched call stays on the fused dispatch count: before this route, B > 1
// sent every GDN layer to the composed per-token chain (oMLX c4 aggregate
// 58.1 -> 71.8 tok/s on an M1 Max once fused).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;
using mlx::core::omarchy::trace::counters;

namespace {

constexpr int B = 4, H = 16, D = 128;

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

array pattern(Shape shape, uint32_t seed, float scale, float shift, Stream s) {
  size_t n = 1;
  for (auto d : shape) {
    n *= static_cast<size_t>(d);
  }
  std::vector<float> v;
  v.reserve(n);
  uint32_t x = seed;
  for (size_t i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    v.push_back(
        shift +
        scale *
            (static_cast<float>(static_cast<double>(x % 20000u) / 10000.0) -
             1.0f));
  }
  return array(v.data(), shape, float32);
}

array bf16(Shape shape, uint32_t seed, float scale, Stream s,
           float shift = 0.0f) {
  return astype(pattern(shape, seed, scale, shift, s), bfloat16, s);
}

array row(const array& x, int i, Stream s) {
  Shape start(x.ndim(), 0), stop = x.shape();
  start[0] = i;
  stop[0] = i + 1;
  return slice(x, start, stop, s);
}

// Batched call vs the same rows run one at a time; `call` maps the
// per-sequence inputs (all [B or 1, ...]) to {out, state}.
void check_rows(const char* what, const std::vector<array>& per_seq,
                const std::function<std::vector<array>(
                    const std::vector<array>&)>& call,
                Stream s) {
  auto batched = call(per_seq);
  std::vector<array> outs, states;
  for (int i = 0; i < B; ++i) {
    std::vector<array> one;
    for (const auto& x : per_seq) {
      one.push_back(row(x, i, s));
    }
    auto r = call(one);
    outs.push_back(r[0]);
    states.push_back(r[1]);
  }
  auto out1 = concatenate(outs, 0, s);
  auto state1 = concatenate(states, 0, s);
  eval({batched[0], batched[1], out1, state1});
  CHECK_MESSAGE(array_equal(batched[0], out1, s).item<bool>(), what,
                ": batched output differs from the per-row B=1 output");
  CHECK_MESSAGE(array_equal(batched[1], state1, s).item<bool>(), what,
                ": batched state differs from the per-row B=1 state");
}

std::vector<array> raw_inputs(Stream s) {
  return {bf16({B, 1, H, D}, 1, 0.1f, s),  bf16({B, 1, H, D}, 2, 0.1f, s),
          bf16({B, 1, H, D}, 3, 1.0f, s),  bf16({B, 1, H}, 4, 1.0f, s),
          bf16({B, 1, H}, 5, 1.0f, s),
          pattern({B, H, D, D}, 6, 0.05f, 0.0f, s)};
}

bool have_gpu() {
  if (!gpu::is_available()) {
    std::cout << "Skipping: no qualifying Vulkan device\n";
    return false;
  }
  return true;
}

} // namespace

TEST_CASE("batched raw-gate decode matches per-row decode bit for bit") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  array a_log = bf16({H}, 7, 1.0f, s, 1.0f);
  array dt = bf16({H}, 8, 1.0f, s);
  check_rows("raw decode B=4", raw_inputs(s),
             [&](const std::vector<array>& x) {
               return fast::gated_delta_update_raw(
                   x[0], x[1], x[2], x[3], x[4], a_log, dt, x[5]);
             },
             s);
}

TEST_CASE("batched gated decode matches per-row decode bit for bit") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  std::vector<array> x = {
      bf16({B, 1, H, D}, 11, 0.1f, s), bf16({B, 1, H, D}, 12, 0.1f, s),
      bf16({B, 1, H, D}, 13, 1.0f, s), bf16({B, 1, H}, 14, 0.05f, s, 0.9f),
      bf16({B, 1, H}, 15, 0.25f, s, 0.5f),
      pattern({B, H, D, D}, 16, 0.05f, 0.0f, s)};
  check_rows("gated decode B=4", x,
             [&](const std::vector<array>& y) {
               return fast::gated_delta_update(y[0], y[1], y[2], y[3], y[4],
                                               y[5]);
             },
             s);
}

TEST_CASE("batched raw-gate decode stays fused") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  auto x = raw_inputs(s);
  array a_log = bf16({H}, 7, 1.0f, s, 1.0f);
  array dt = bf16({H}, 8, 1.0f, s);
  // A padding mask keeps the composed chain: the reference count.
  array mask = full({B, 1}, true, s);
  auto count = [&](bool masked) {
    eval(x);
    eval({a_log, dt, mask});
    auto& enc = omarchy::get_command_encoder(s);
    enc.synchronize("gdn_batch_inputs");
    uint64_t before = counters().vk_compute_dispatches.load();
    auto r = masked
        ? fast::gated_delta_update_raw(x[0], x[1], x[2], x[3], x[4], a_log,
                                       dt, x[5], mask)
        : fast::gated_delta_update_raw(x[0], x[1], x[2], x[3], x[4], a_log,
                                       dt, x[5]);
    eval(r);
    enc.synchronize("gdn_batch_outputs");
    return counters().vk_compute_dispatches.load() - before;
  };
  uint64_t fused = count(false), composed = count(true);
  std::cout << "[gdn_decode_batch] B=4 raw decode " << fused
            << " dispatches, masked (composed) " << composed << "\n";
  CHECK_MESSAGE(fused <= 4, "B=4 raw decode took ", fused,
                " dispatches; the fused decode kernel needs <= 4");
  CHECK_MESSAGE(composed > 4 * fused, "the composed reference (", composed,
                ") is not far above the batched count (", fused, ")");
}

// B > 1 prefill (T > 1) runs the batch row by row through the B = 1 fused
// route (patches/mlx-gated-delta-prefill-rows.patch). Before it, every GDN
// layer of a batched prefill took the per-token composed fallback: a [4, T]
// prefill was 5.6-7.6x slower than four [1, T] ones (jw16 G13C, T 128-1024).
std::vector<array> prefill_inputs(int t, Stream s) {
  return {bf16({B, t, H, D}, 21, 0.1f, s), bf16({B, t, H, D}, 22, 0.1f, s),
          bf16({B, t, H, D}, 23, 1.0f, s),
          pattern({B, t, H}, 24, 0.05f, 0.9f, s),
          bf16({B, t, H}, 25, 0.25f, s, 0.5f),
          pattern({B, H, D, D}, 26, 0.05f, 0.0f, s)};
}

TEST_CASE("batched prefill matches per-row prefill bit for bit") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  // T = 128 takes the chunked coopmat route, T = 32 the short-prompt route.
  for (int t : {32, 128}) {
    auto x = prefill_inputs(t, s);
    check_rows(t == 32 ? "prefill B=4 T=32" : "prefill B=4 T=128", x,
               [&](const std::vector<array>& y) {
                 return fast::gated_delta_update(y[0], y[1], y[2], y[3], y[4],
                                                 y[5]);
               },
               s);
  }
}

TEST_CASE("batched masked prefill matches per-row prefill bit for bit") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  constexpr int t = 64;
  auto x = prefill_inputs(t, s);
  // Left padding of b * 5 tokens per row: a real, row-dependent mask.
  std::vector<bool> bits;
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < t; ++i) {
      bits.push_back(i >= b * 5);
    }
  }
  x.push_back(array(bits.begin(), {B, t}, bool_));
  check_rows("masked prefill B=4 T=64", x,
             [&](const std::vector<array>& y) {
               return fast::gated_delta_update(y[0], y[1], y[2], y[3], y[4],
                                               y[5], y[6]);
             },
             s);
}

TEST_CASE("batched prefill stays on the fused dispatches") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  constexpr int t = 128;
  auto x = prefill_inputs(t, s);
  eval(x);
  auto& enc = omarchy::get_command_encoder(s);
  auto count = [&](const std::vector<array>& y) {
    enc.synchronize("gdn_prefill_inputs");
    uint64_t before = counters().vk_compute_dispatches.load();
    auto r = fast::gated_delta_update(y[0], y[1], y[2], y[3], y[4], y[5]);
    eval(r);
    enc.synchronize("gdn_prefill_outputs");
    return counters().vk_compute_dispatches.load() - before;
  };
  std::vector<array> one;
  for (const auto& v : x) {
    one.push_back(row(v, 0, s));
  }
  eval(one);
  uint64_t single = count(one), batched = count(x);
  std::cout << "[gdn_prefill_rows] B=1 T=128 " << single << " dispatches, B=4 "
            << batched << "\n";
  CHECK_MESSAGE(single >= 1, "the B=1 prefill dispatched nothing");
  // Four row calls plus the dense copies and the two concatenations; the
  // composed per-token chain is thousands of dispatches at T = 128.
  CHECK_MESSAGE(batched <= 4 * single + 16, "B=4 prefill took ", batched,
                " dispatches against ", single, " for one row: it fell back");
}
