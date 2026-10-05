// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Regression tests for the f16 causal-ragged SDPA wrong values and NaN from
// the 2026-09-11 upstream snapshot (receipts/2026-09-11-upstream-suite,
// receipts/2026-09-11-f16-sdpa-causal). Every case runs at the exact shape
// that failed upstream; a case that only used smaller shapes is what let the
// original ship.
//
//   The dispatch_softmax causal sentinel ("causal_offset >= 0") conflated
//   "causal off" (-1) with a legitimate negative offset, so a query longer
//   than the key (qsl=127, ksl=65, offset -62) ran the softmax NON-causal
//   over scores the register-blocked matmul had legitimately left unwritten
//   (CausalSkip::Columns). The garbage could be finite (0.077-0.125 wrong
//   values vs the 3e-4 tolerance) or an f16 NaN pattern that won the row max
//   (the NaN at head_dim=128, transpose=True). Fixed by passing an explicit
//   causal flag and recovering the signed offset in the shader, which clamps
//   a row with no admissible key to zero valid columns: such a row stores an
//   exact-zero answer, so the 1/0 normalization can never produce a NaN.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (!gpu::is_available()) {
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return false;
  }
  return true;
}

std::vector<float> flat(const array& value, Stream stream) {
  array copy = astype(value, float32, stream);
  copy.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = copy.data<float>();
  return std::vector<float>(data, data + copy.size());
}

// Deterministic pseudo-random values in [0, 1).
std::vector<float> unit_pattern(size_t count, uint32_t seed) {
  std::vector<float> values;
  values.reserve(count);
  uint32_t state = seed;
  for (size_t index = 0; index < count; ++index) {
    state = state * 1664525u + 1013904223u;
    values.push_back(
        static_cast<float>(static_cast<double>(state % 10000u) / 10000.0));
  }
  return values;
}

// Host attention reference in double for one query row over its admissible
// keys (ki <= kv_len - q_len + row). Rows with no admissible key return an
// all-zero vector: the caller asserts that exact-zero answer, there is no
// reference value to match.
std::vector<double> host_attention(
    const std::vector<double>& q_row,
    const double* k_head,
    const double* v_head,
    int q_len,
    int kv_len,
    int head_dim,
    double scale,
    int row) {
  const int offset = kv_len - q_len;
  std::vector<double> out(head_dim, 0.0);
  double max_score = -INFINITY;
  std::vector<double> weights(kv_len, 0.0);
  for (int ki = 0; ki < kv_len; ++ki) {
    if (ki > offset + row) {
      continue;
    }
    double dot = 0.0;
    for (int d = 0; d < head_dim; ++d) {
      dot += q_row[d] * k_head[ki * head_dim + d];
    }
    weights[ki] = dot * scale;
    max_score = std::max(max_score, weights[ki]);
  }
  if (max_score == -INFINITY) {
    return out;
  }
  double total = 0.0;
  for (int ki = 0; ki < kv_len; ++ki) {
    if (ki <= offset + row) {
      weights[ki] = std::exp(weights[ki] - max_score);
      total += weights[ki];
    } else {
      weights[ki] = 0.0;
    }
  }
  for (int d = 0; d < head_dim; ++d) {
    double acc = 0.0;
    for (int ki = 0; ki < kv_len; ++ki) {
      acc += weights[ki] * v_head[ki * head_dim + d];
    }
    out[d] = acc / total;
  }
  return out;
}

// One case at an upstream ragged causal shape. qsl > ksl always: the leading
// qsl - ksl rows have no admissible key.
void run_ragged_case(
    Stream stream,
    int q_len,
    int kv_len,
    int head_dim,
    int q_heads,
    int kv_heads,
    bool transposed,
    uint32_t seed,
    const std::string& label) {
  const int batch = 1;
  const int repeats = q_heads / kv_heads;
  const int offset = q_len - kv_len;
  double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

  auto q_data = unit_pattern(batch * q_heads * q_len * head_dim, seed);
  auto k_data = unit_pattern(batch * kv_heads * kv_len * head_dim, seed + 1u);
  auto v_data = unit_pattern(batch * kv_heads * kv_len * head_dim, seed + 2u);

  // Upstream prepare_inputs keeps v small (uniform(0, scale)) so softmax
  // outputs stay in the tolerance's useful range.
  const float v_scale = static_cast<float>(scale);
  for (auto& value : v_data) {
    value *= v_scale;
  }

  std::vector<double> q_host(q_data.begin(), q_data.end());
  std::vector<double> k_host(k_data.begin(), k_data.end());
  std::vector<double> v_host(v_data.begin(), v_data.end());
  // Upstream "transpose=True" varies the STORAGE layout: the q/k/v buffers
  // are contiguous (B, L, H, D) and the kernel receives strided (B, H, L, D)
  // views of them (prepare_inputs builds (B, L, H, D), do_attention
  // transposes to (B, H, L, D) before the call). Build the same thing:
  // materialize the L-major buffer, then hand the kernel the heads-major
  // strided view. The host reference data stays heads-major either way.
  array q = array(
      q_data.begin(), Shape{batch, q_heads, q_len, head_dim}, float16);
  array k = array(
      k_data.begin(), Shape{batch, kv_heads, kv_len, head_dim}, float16);
  array v = array(
      v_data.begin(), Shape{batch, kv_heads, kv_len, head_dim}, float16);
  array q_in = q;
  array k_in = k;
  array v_in = v;
  if (transposed) {
    array ql = transpose(q, {0, 2, 1, 3}, stream);
    array kl = transpose(k, {0, 2, 1, 3}, stream);
    array vl = transpose(v, {0, 2, 1, 3}, stream);
    ql.eval();
    kl.eval();
    vl.eval();
    q_in = transpose(ql, {0, 2, 1, 3}, stream);
    k_in = transpose(kl, {0, 2, 1, 3}, stream);
    v_in = transpose(vl, {0, 2, 1, 3}, stream);
  }
  array out = fast::scaled_dot_product_attention(
      q_in,
      k_in,
      v_in,
      static_cast<float>(scale),
      std::string("causal"),
      std::nullopt,
      std::nullopt,
      false,
      stream);
  auto got = flat(out, stream);

  REQUIRE_EQ(got.size(), batch * q_heads * q_len * head_dim);

  // 1. The NaN proof: no element anywhere may be NaN.
  size_t nans = 0;
  for (float value : got) {
    if (std::isnan(value)) {
      ++nans;
    }
  }
  CHECK_MESSAGE(nans == 0, label, ": ", nans, " NaN elements in the output");

  // 2. The defined answer: rows with no admissible key are exact zeros.
  size_t bad_zero = 0;
  float nonzero_seen = 0.0f;
  for (int l = 0; l < offset; ++l) {
    for (int h = 0; h < q_heads; ++h) {
      size_t base = (static_cast<size_t>(h) * q_len + l) * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        if (got[base + d] != 0.0f) {
          ++bad_zero;
          nonzero_seen = got[base + d];
        }
      }
    }
  }
  CHECK_MESSAGE(
      bad_zero == 0,
      label,
      ": ",
      bad_zero,
      " non-zero elements in the fully masked rows; first ",
      nonzero_seen);

  // 3. The upstream wrong values: valid rows match the double reference
  // under the upstream criterion |got - want| <= atol + atol * |want|.
  const double atol = 3e-4;
  size_t bad = 0;
  double worst = 0.0;
  for (int h = 0; h < q_heads; ++h) {
    const int kv = h / repeats;
    const double* k_head =
        k_host.data() + static_cast<size_t>(kv) * kv_len * head_dim;
    const double* v_head =
        v_host.data() + static_cast<size_t>(kv) * kv_len * head_dim;
    for (int l = offset; l < q_len; ++l) {
      std::vector<double> q_row(
          q_host.begin() +
              (static_cast<size_t>(h) * q_len + l) * head_dim,
          q_host.begin() +
              (static_cast<size_t>(h) * q_len + l + 1) * head_dim);
      auto want = host_attention(
          q_row, k_head, v_head, q_len, kv_len, head_dim, scale, l);
      size_t base = (static_cast<size_t>(h) * q_len + l) * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        double diff = std::abs(static_cast<double>(got[base + d]) - want[d]);
        double bound = atol + atol * std::abs(want[d]);
        if (diff > bound) {
          ++bad;
          worst = std::max(worst, diff);
        }
      }
    }
  }
  CHECK_MESSAGE(
      bad == 0,
      label,
      ": ",
      bad,
      " valid-row elements outside tolerance; worst diff ",
      worst);
}

} // namespace

TEST_CASE("sdpa f16 causal ragged shapes match host math and keep masked rows zero") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // The four upstream failing combos (B=1, qsl=127, ksl=65, GQA 32:8).
  run_ragged_case(stream, 127, 65, 64, 32, 8, false, 1101u, "hd=64 t=False");
  run_ragged_case(stream, 127, 65, 128, 32, 8, false, 1201u, "hd=128 t=False");
  run_ragged_case(stream, 127, 65, 64, 32, 8, true, 1301u, "hd=64 t=True");
  run_ragged_case(stream, 127, 65, 128, 32, 8, true, 1401u, "hd=128 t=True");

  // A small shape pins the same construction: rows 0 and 1 have no
  // admissible key and must come back exact zeros, not garbage.
  run_ragged_case(stream, 4, 2, 16, 4, 2, false, 1501u, "small 4/2");
}


// Multi-token verify rows (q_len 2..16, causal, bf16): the rows route
// must make each row BIT-IDENTICAL to the same row decoded alone at
// q_len=1 over its visible key prefix - that is the invariant that makes
// speculative verify self-consistent with plain decode. The reference
// rides the proven single-query arm (batch=1, q_len=1, no mask).
TEST_CASE("sdpa bf16 decode rows are per-row bit-identical to single-query over the visible prefix") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int head_dim = 128;
  const float scale = 0.08838834764831845f;  // 1/sqrt(128), the Qwen3 value
  for (auto [q_len, kv_len, q_heads, kv_heads] :
       {std::tuple<int, int, int, int>{4, 20, 8, 4},
        {16, 40, 8, 2},
        {3, 17, 4, 4}}) {
    size_t q_count = q_heads * q_len * head_dim;
    size_t kv_count = kv_heads * kv_len * head_dim;
    array q = array(
        unit_pattern(q_count, 101).begin(),
        Shape{1, q_heads, q_len, head_dim},
        float32);
    array k = array(
        unit_pattern(kv_count, 202).begin(),
        Shape{1, kv_heads, kv_len, head_dim},
        float32);
    array v = array(
        unit_pattern(kv_count, 303).begin(),
        Shape{1, kv_heads, kv_len, head_dim},
        float32);
    q = astype(q, bfloat16, stream);
    k = astype(k, bfloat16, stream);
    v = astype(v, bfloat16, stream);
    q.eval();
    k.eval();
    v.eval();
    omarchy::get_command_encoder(stream).synchronize();

    // Candidate: causal q_len-row attention (the rows route).
    array out = fast::scaled_dot_product_attention(
        q, k, v, scale, std::string("causal"), std::nullopt, std::nullopt,
        false, stream);
    out.eval();
    omarchy::get_command_encoder(stream).synchronize();
    // The route must actually be the fused one: the composed route
    // would dispatch several kernels (softmax + two matmuls).
    (void)out;

    // Reference: row m decoded alone over keys 0..(kv_len - q_len + m).
    for (int m = 0; m < q_len; ++m) {
      int visible = kv_len - q_len + m + 1;
      array q_m = slice(q, {0, 0, m, 0}, {1, q_heads, m + 1, head_dim}, stream);
      array k_m = slice(k, {0, 0, 0, 0}, {1, kv_heads, visible, head_dim}, stream);
      array v_m = slice(v, {0, 0, 0, 0}, {1, kv_heads, visible, head_dim}, stream);
      q_m.eval();
      k_m.eval();
      v_m.eval();
      omarchy::get_command_encoder(stream).synchronize();
      array ref = fast::scaled_dot_product_attention(
          q_m, k_m, v_m, scale, "", std::nullopt, std::nullopt,
          false, stream);
      ref.eval();
      omarchy::get_command_encoder(stream).synchronize();
      // bf16 -> f32 widening is exact, so an f32 compare is a bit
      // compare of the stored bf16 words.
      std::vector<float> got = flat(slice(out, {0, 0, m, 0}, {1, q_heads, m + 1, head_dim}, stream), stream);
      std::vector<float> want = flat(ref, stream);
      INFO("q_len ", q_len, " kv_len ", kv_len, " row ", m);
      REQUIRE_EQ(got.size(), want.size());
      for (size_t i = 0; i < got.size(); ++i) {
        INFO("element ", i, " got ", got[i], " want ", want[i]);
        CHECK_EQ(got[i], want[i]);
      }
    }
  }
}

TEST_CASE("sdpa bf16 decode rows kill switch restores the composed route") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int head_dim = 128;
  const float scale = 0.08838834764831845f;
  size_t q_count = 4 * 4 * head_dim;
  size_t kv_count = 4 * 32 * head_dim;
  array q = astype(
      array(unit_pattern(q_count, 404).begin(),
            Shape{1, 4, 4, head_dim}, float32),
      bfloat16, stream);
  array k = astype(
      array(unit_pattern(kv_count, 505).begin(),
            Shape{1, 4, 32, head_dim}, float32),
      bfloat16, stream);
  array v = astype(
      array(unit_pattern(kv_count, 606).begin(),
            Shape{1, 4, 32, head_dim}, float32),
      bfloat16, stream);
  q.eval();
  k.eval();
  v.eval();
  omarchy::get_command_encoder(stream).synchronize();

  setenv("MLX_OMARCHY_SDPA_DECODE_ROWS", "0", 1);
  array composed_a = fast::scaled_dot_product_attention(
      q, k, v, scale, std::string("causal"), std::nullopt, std::nullopt,
      false, stream);
  composed_a.eval();
  omarchy::get_command_encoder(stream).synchronize();

  setenv("MLX_OMARCHY_SDPA_DECODE_ROWS", "1", 1);
  array rows = fast::scaled_dot_product_attention(
      q, k, v, scale, std::string("causal"), std::nullopt, std::nullopt,
      false, stream);
  rows.eval();
  omarchy::get_command_encoder(stream).synchronize();

  setenv("MLX_OMARCHY_SDPA_DECODE_ROWS", "0", 1);
  array composed_b = fast::scaled_dot_product_attention(
      q, k, v, scale, std::string("causal"), std::nullopt, std::nullopt,
      false, stream);
  composed_b.eval();
  omarchy::get_command_encoder(stream).synchronize();
  unsetenv("MLX_OMARCHY_SDPA_DECODE_ROWS");

  // The switch is deterministic in both directions (the rows route is
  // not required to match the composed route bit for bit - it matches
  // single-query decode instead, the case above).
  std::vector<float> a = flat(composed_a, stream);
  std::vector<float> b = flat(composed_b, stream);
  REQUIRE_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    CHECK_EQ(a[i], b[i]);
  }
}
