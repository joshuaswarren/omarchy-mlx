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
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"
#include "mlx/version.h"

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
// per-sequence inputs (all [rows or 1, ...]) to {out, state}.
void check_rows(const char* what, const std::vector<array>& per_seq,
                const std::function<std::vector<array>(
                    const std::vector<array>&)>& call,
                Stream s, int rows = B) {
  auto batched = call(per_seq);
  std::vector<array> outs, states;
  for (int i = 0; i < rows; ++i) {
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

// Provenance beside every measurement (AGENTS.md test rules): the loaded
// libmlx version stamp (the wheel's source commit) and the device
// capabilities that choose the GDN prefill route.
void provenance(const char* tag) {
  const auto& caps = omarchy::device(0).capabilities();
  std::cout << "[provenance] " << tag << " mlx=" << mlx::core::version()
            << " cooperative_matrix_f32_8="
            << (caps.cooperative_matrix_f32_8 ? 1 : 0)
            << " subgroup_size=" << caps.subgroup_size << "\n";
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
  provenance("gdn_decode_batch");
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
// prefill was 5.6-7.6x slower than four [1, T] ones (M1 Max, T 128-1024,
// receipts/2026-10-08-gdn-prefill-rows). The row split applies only to the
// shapes and dtypes the fused route accepts (bf16, Hk == Hv, Dk == Dv == 128).
std::vector<array> prefill_inputs(int rows, int t, Stream s,
                                  Dtype dt = bfloat16) {
  auto cast = [&](array x) { return astype(x, dt, s); };
  return {cast(pattern({rows, t, H, D}, 21, 0.1f, 0.0f, s)),
          cast(pattern({rows, t, H, D}, 22, 0.1f, 0.0f, s)),
          cast(pattern({rows, t, H, D}, 23, 1.0f, 0.0f, s)),
          pattern({rows, t, H}, 24, 0.05f, 0.9f, s),
          cast(pattern({rows, t, H}, 25, 0.25f, 0.5f, s)),
          pattern({rows, H, D, D}, 26, 0.05f, 0.0f, s)};
}

TEST_CASE("batched prefill matches per-row prefill bit for bit") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  // T < 64 takes the short-prompt scan, T >= 64 the chunked coopmat route
  // (kGdnCoopmatMinTokens); 33/63/64/65/129 straddle the threshold and the
  // 8-token chunk grid.
  for (int rows : {2, 3, 4}) {
    for (int t : {2, 7, 33, 63, 64, 65, 128, 129}) {
      auto x = prefill_inputs(rows, t, s);
      std::string label =
          "prefill B=" + std::to_string(rows) + " T=" + std::to_string(t);
      check_rows(label.c_str(), x,
                 [&](const std::vector<array>& y) {
                   return fast::gated_delta_update(y[0], y[1], y[2], y[3],
                                                   y[4], y[5]);
                 },
                 s, rows);
    }
  }
}

TEST_CASE("batched masked prefill matches per-row prefill bit for bit") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  for (int rows : {2, 3, 4}) {
    for (int t : {33, 64}) {
      auto x = prefill_inputs(rows, t, s);
      // Left padding of b * 5 tokens per row: a real, row-dependent mask.
      std::vector<bool> bits;
      for (int b = 0; b < rows; ++b) {
        for (int i = 0; i < t; ++i) {
          bits.push_back(i >= b * 5);
        }
      }
      x.push_back(array(bits.begin(), {rows, t}, bool_));
      std::string label = "masked prefill B=" + std::to_string(rows) +
          " T=" + std::to_string(t);
      check_rows(label.c_str(), x,
                 [&](const std::vector<array>& y) {
                   return fast::gated_delta_update(y[0], y[1], y[2], y[3],
                                                   y[4], y[5], y[6]);
                 },
                 s, rows);
    }
  }
}

// Dispatches of one gated_delta_update call (inputs already evaluated).
uint64_t dispatches(const std::vector<array>& y, Stream s) {
  auto& enc = omarchy::get_command_encoder(s);
  enc.synchronize("gdn_prefill_inputs");
  uint64_t before = counters().vk_compute_dispatches.load();
  auto r = y.size() > 6
      ? fast::gated_delta_update(y[0], y[1], y[2], y[3], y[4], y[5], y[6])
      : fast::gated_delta_update(y[0], y[1], y[2], y[3], y[4], y[5]);
  eval(r);
  enc.synchronize("gdn_prefill_outputs");
  return counters().vk_compute_dispatches.load() - before;
}

TEST_CASE("batched prefill stays on the fused dispatches") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  constexpr int t = 128;
  auto x = prefill_inputs(B, t, s);
  eval(x);
  std::vector<array> one;
  for (const auto& v : x) {
    one.push_back(row(v, 0, s));
  }
  eval(one);
  uint64_t single = dispatches(one, s), batched = dispatches(x, s);
  provenance("gdn_prefill_rows");
  std::cout << "[gdn_prefill_rows] B=1 T=128 " << single << " dispatches, B=4 "
            << batched << "\n";
  CHECK_MESSAGE(single >= 1, "the B=1 prefill dispatched nothing");
  // Four row calls plus the dense copies and the two concatenations; the
  // composed per-token chain is thousands of dispatches at T = 128.
  CHECK_MESSAGE(batched <= 4 * single + 16, "B=4 prefill took ", batched,
                " dispatches against ", single, " for one row: it fell back");
}

// A maskless prefill must not take the recur32 kernel by default on any part:
// on G13G it did until 2026-10-09, and that changed the greedy tokens of
// Qwen3.5-9B against macOS (H392). The env value 2 still selects it.
TEST_CASE("a maskless prefill takes the recur32 kernel only when asked") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  constexpr int t = 128;
  auto x = prefill_inputs(B, t, s);
  eval(x);
  std::vector<array> one;
  for (const auto& v : x) {
    one.push_back(row(v, 0, s));
  }
  eval(one);
  const auto& caps = omarchy::device(0).capabilities();
  const bool env_free = std::getenv("MLX_OMARCHY_GDN_RECUR32") == nullptr &&
      std::getenv("MLX_OMARCHY_NO_COOPMAT_GDN") == nullptr;
  if (caps.subgroup_size != 32 || !env_free) return;
  const int64_t recur32 =
      static_cast<int64_t>(omarchy::ComputeKernel::GatedDeltaPrefillRecur32BF16);
  auto last_kernel = [&]() {
    dispatches(one, s);
    return counters().last_dispatched_kernel.load();
  };
  int64_t by_default = last_kernel();
  setenv("MLX_OMARCHY_GDN_RECUR32", "2", 1);
  int64_t forced_two = last_kernel();
  unsetenv("MLX_OMARCHY_GDN_RECUR32");
  std::cout << "[gdn_maskless_route] " << caps.device_name << " default kernel "
            << by_default << ", forced 2 kernel " << forced_two
            << ", recur32 kernel id " << recur32 << "\n";
  CHECK_MESSAGE(forced_two == recur32, "RECUR32=2 did not select the recur32"
                " kernel (last kernel ", forced_two, ")");
  CHECK_MESSAGE(by_default != recur32, "a maskless prefill took the recur32"
                " kernel by default on ", caps.device_name);
}

// A masked row takes the single-pass per-token route (recur32, one dispatch per
// row at T = 128): by default on G13 parts (measured: G13G and G13C), forced
// by MLX_OMARCHY_GDN_RECUR32=2 on any part with 32-lane subgroups. The padded
// rows of a real batched prefill carry a mask, and a masked row used to fall
// to the two-pass snapshot scan (two dispatches; a padded batch ran 1.6x of
// sequential on G13C). A maskless row keeps its own default route.
TEST_CASE("masked prefill takes the single-pass route") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  constexpr int t = 128;
  auto x = prefill_inputs(B, t, s);
  std::vector<bool> bits;
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < t; ++i) {
      bits.push_back(i >= b * 5 + 7);
    }
  }
  auto masked = x;
  masked.push_back(array(bits.begin(), {B, t}, bool_));
  eval(x);
  eval(masked);
  std::vector<array> one, one_masked;
  for (const auto& v : x) one.push_back(row(v, 0, s));
  for (const auto& v : masked) one_masked.push_back(row(v, 0, s));
  eval(one);
  eval(one_masked);
  uint64_t plain = dispatches(one, s), padded = dispatches(one_masked, s),
           padded_batch = dispatches(masked, s);
  provenance("gdn_masked_prefill");
  std::cout << "[gdn_masked_prefill] B=1 T=128 maskless " << plain
            << " dispatches, masked " << padded << ", B=4 masked "
            << padded_batch << "\n";
  const bool env_free = std::getenv("MLX_OMARCHY_GDN_RECUR32") == nullptr &&
      std::getenv("MLX_OMARCHY_NO_COOPMAT_GDN") == nullptr;
  const auto& caps = omarchy::device(0).capabilities();
  if (caps.device_name.find("G13") != std::string::npos && env_free) {
    CHECK_MESSAGE(padded == 1, "a masked row took ", padded,
                  " dispatches on the default route (one expected on G13)");
    CHECK_MESSAGE(padded_batch <= 4 * padded + 16, "B=4 masked prefill took ",
                  padded_batch, " dispatches against ", padded, " for one row");
  }
  if (caps.subgroup_size == 32 && env_free) {
    setenv("MLX_OMARCHY_GDN_RECUR32", "2", 1);
    uint64_t forced = dispatches(one_masked, s);
    unsetenv("MLX_OMARCHY_GDN_RECUR32");
    std::cout << "[gdn_masked_prefill] forced recur32: masked " << forced
              << " dispatches\n";
    CHECK_MESSAGE(forced == 1, "a masked row took ", forced,
                  " dispatches on the forced single-pass route (one expected)");
  }
}

// A dtype the fused route does not take (fp16) keeps ONE batched composed
// fallback: splitting it would run the composed chain once per row (B x the
// dispatches, one encoder synchronize each). Same result as per-row, and the
// batched call must not cost more than about one row's dispatches.
TEST_CASE("batched fp16 prefill keeps the single batched fallback") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  constexpr int t = 8;
  auto x = prefill_inputs(B, t, s, float16);
  eval(x);
  check_rows("fp16 prefill B=4 T=8", x,
             [&](const std::vector<array>& y) {
               return fast::gated_delta_update(y[0], y[1], y[2], y[3], y[4],
                                               y[5]);
             },
             s);
  std::vector<array> one;
  for (const auto& v : x) {
    one.push_back(row(v, 0, s));
  }
  eval(one);
  uint64_t single = dispatches(one, s), batched = dispatches(x, s);
  provenance("gdn_fp16_prefill");
  std::cout << "[gdn_fp16_prefill] B=1 T=8 " << single << " dispatches, B=4 "
            << batched << "\n";
  CHECK_MESSAGE(batched <= 2 * single + 8, "fp16 B=4 prefill took ", batched,
                " dispatches against ", single,
                " for one row: the batch was split into per-row fallbacks");
}
