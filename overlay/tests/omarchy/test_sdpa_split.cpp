// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Split-K bf16 single-query decode SDPA (sdpa_decode_split_p1.comp + sdpa_decode_split_p2.comp, routed by MLX_OMARCHY_SDPA_DECODE_SPLIT=1 and by default on the
// parts of omarchy::sdpa_split_min_keys). The route
// computes in float32 in a different order from the fused kernel, so it is gated against a float64 reference, not against the fused kernel's bits:
// the maximum error of the split route against float64 must be no larger than the fused route's maximum error plus two half steps of the largest
// reference value (one bf16 rounding step, ref * 2^-8), the mean error no larger than the fused mean plus a quarter of that half step pair, and
// the maximum error at most three steps; it must be exactly two dispatches; and shapes it does not cover must keep their previous dispatch count
// and bits. A one-hot value pattern with uniform scores makes a dropped or doubled key visible. The route assertions run only when the device has
// 32-lane subgroup arithmetic (the guard of the route); otherwise the cases print Skipping.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

bool compute_available() {
  if (!gpu::is_available()) {
    printf("Skipping: no qualifying Vulkan device\n");
    return false;
  }
  return true;
}

bool split_device_ready() {
  const auto& caps = omarchy::device(0).capabilities();
  constexpr VkSubgroupFeatureFlags need = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
  return caps.subgroup_size == 32u && caps.max_compute_work_group_invocations >= 128u &&
      caps.max_compute_work_group_size[0] >= 128u && (caps.subgroup_operations & need) == need;
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

uint64_t dispatches() {
  return omarchy::trace::counters().vk_compute_dispatches.load();
}

std::vector<float> pattern(size_t count, uint32_t seed, float amplitude) {
  std::vector<float> values;
  values.reserve(count);
  uint32_t state = seed;
  for (size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    values.push_back(amplitude * (static_cast<float>(state % 20000u) / 10000.0f - 1.0f));
  }
  return values;
}

// A bf16 device array from float values, and the stored (rounded) values.
struct Stored {
  array x;
  std::vector<float> values;
};

Stored make_bf16(const std::vector<float>& host, Shape shape, const Stream& stream) {
  array f(host.data(), shape, float32);
  array b = astype(f, bfloat16, stream);
  b.eval();
  array back = astype(b, float32, stream);
  back.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* p = back.data<float>();
  return {b, std::vector<float>(p, p + back.size())};
}

std::vector<float> flat(const array& value, const Stream& stream) {
  array copy = astype(value, float32, stream);
  copy.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* p = copy.data<float>();
  return std::vector<float>(p, p + copy.size());
}

// float64 reference of single-query attention over the stored values.
std::vector<double> reference(
    const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
    int heads, int kv_heads, int keys, int capacity, int key_start, int width, double scale) {
  std::vector<double> out(static_cast<size_t>(heads) * width);
  int group = heads / kv_heads;
  for (int h = 0; h < heads; ++h) {
    int kvh = h / group;
    std::vector<double> s(keys);
    double mx = -1e300;
    for (int j = 0; j < keys; ++j) {
      double dot = 0.0;
      const float* kr = &k[(static_cast<size_t>(kvh) * capacity + key_start + j) * width];
      for (int d = 0; d < width; ++d) {
        dot += static_cast<double>(q[static_cast<size_t>(h) * width + d]) * kr[d];
      }
      s[j] = dot * scale;
      mx = std::max(mx, s[j]);
    }
    double sum = 0.0;
    for (int j = 0; j < keys; ++j) {
      s[j] = std::exp(s[j] - mx);
      sum += s[j];
    }
    for (int d = 0; d < width; ++d) {
      double acc = 0.0;
      for (int j = 0; j < keys; ++j) {
        acc += s[j] * v[(static_cast<size_t>(kvh) * capacity + key_start + j) * width + d];
      }
      out[static_cast<size_t>(h) * width + d] = acc / sum;
    }
  }
  return out;
}

struct Run {
  std::vector<float> out;
  uint64_t launched;
};

Run run_attention(
    const array& q, const array& k, const array& v, float scale, const char* split, const Stream& stream) {
  setenv("MLX_OMARCHY_SDPA_DECODE_SPLIT", split, 1);
  uint64_t before = dispatches();
  array r = fast::scaled_dot_product_attention(q, k, v, scale, "", {}, std::nullopt, false, stream);
  r.eval();
  omarchy::get_command_encoder(stream).synchronize();
  uint64_t launched = dispatches() - before;
  return {flat(r, stream), launched};
}

struct Errors {
  double mean;
  double max;
};

Errors error_against(const std::vector<float>& out, const std::vector<double>& ref) {
  double sum = 0.0;
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double e = std::fabs(static_cast<double>(out[i]) - ref[i]);
    sum += e;
    worst = std::max(worst, e);
  }
  return {sum / static_cast<double>(ref.size()), worst};
}

// One decode shape: heads, kv heads, cache keys, capacity of the cache buffer, first key of the view. With one_hot the scores are uniform (K = 0)
// and the values are one-hot rows: output d is the share of keys whose index modulo the width is d, so a dropped or doubled key moves it.
void check_shape(int heads, int kv_heads, int keys, int capacity, int key_start, float amplitude, bool one_hot = false) {
  const int width = 128;
  Stream stream = gpu_stream();
  Stored q = make_bf16(pattern(static_cast<size_t>(heads) * width, 11, amplitude), Shape{1, heads, 1, width}, stream);
  std::vector<float> k_host = pattern(static_cast<size_t>(kv_heads) * capacity * width, 23, amplitude);
  std::vector<float> v_host = pattern(static_cast<size_t>(kv_heads) * capacity * width, 37, 1.0f);
  if (one_hot) {
    std::fill(k_host.begin(), k_host.end(), 0.0f);
    for (size_t i = 0; i != v_host.size(); ++i) {
      size_t key = (i / width) % capacity;
      v_host[i] = (i % width) == (key % width) ? 1.0f : 0.0f;
    }
  }
  Stored kfull = make_bf16(k_host, Shape{1, kv_heads, capacity, width}, stream);
  Stored vfull = make_bf16(v_host, Shape{1, kv_heads, capacity, width}, stream);
  array k = slice(kfull.x, Shape{0, 0, key_start, 0}, Shape{1, kv_heads, key_start + keys, width}, stream);
  array v = slice(vfull.x, Shape{0, 0, key_start, 0}, Shape{1, kv_heads, key_start + keys, width}, stream);
  const float scale = 1.0f / std::sqrt(static_cast<float>(width));
  std::vector<double> ref = reference(q.values, kfull.values, vfull.values, heads, kv_heads, keys, capacity, key_start, width, scale);
  Run fused = run_attention(q.x, k, v, scale, "0", stream);
  setenv("MLX_OMARCHY_SDPA_SPLIT_PV", "0", 1);
  Run split = run_attention(q.x, k, v, scale, "1", stream);
  setenv("MLX_OMARCHY_SDPA_SPLIT_PV", "1", 1);
  Run split_pv = run_attention(q.x, k, v, scale, "1", stream);
  unsetenv("MLX_OMARCHY_SDPA_SPLIT_PV");
  Errors e_fused = error_against(fused.out, ref);
  Errors e_split = error_against(split.out, ref);
  Errors e_pv = error_against(split_pv.out, ref);
  INFO("heads " << heads << " kv " << kv_heads << " keys " << keys << " start " << key_start << " split launches " << split.launched);
  if (split_device_ready()) {
    CHECK(split.launched == 2);
    CHECK(split_pv.launched == 2);
  } else {
    printf("Skipping: split route not available on this device\n");
  }
  // The output is bf16: one rounding step of the largest reference magnitude bounds the representable difference.
  double rounding = 0.0;
  for (double r : ref) {
    rounding = std::max(rounding, std::fabs(r) * std::ldexp(1.0, -8));
  }
  CHECK(e_split.max <= e_fused.max + 2.0 * rounding + 1e-6);
  CHECK(e_split.mean <= e_fused.mean + 0.25 * rounding + 1e-7);
  CHECK(e_split.max <= 3.0 * rounding + 1e-5);
  // The value-phase variant of pass 1 (8-byte value loads, four key partitions per dimension group) meets the same gates.
  CHECK(e_pv.max <= e_fused.max + 2.0 * rounding + 1e-6);
  CHECK(e_pv.mean <= e_fused.mean + 0.25 * rounding + 1e-7);
  CHECK(e_pv.max <= 3.0 * rounding + 1e-5);
}

} // namespace

TEST_CASE("split-K decode SDPA: error against float64 and two dispatches, group 4") {
  if (!compute_available()) {
    return;
  }
  for (int keys : {1, 2, 17, 127, 128, 129, 255, 256, 500, 1023, 2048, 4097}) {
    check_shape(32, 8, keys, keys, 0, 1.0f);
  }
}

TEST_CASE("split-K decode SDPA: group sizes 1, 2 and 8, long and large-magnitude inputs") {
  if (!compute_available()) {
    return;
  }
  check_shape(16, 16, 500, 500, 0, 1.0f);
  check_shape(16, 8, 777, 777, 0, 1.0f);
  check_shape(64, 8, 300, 300, 0, 1.0f);
  check_shape(32, 8, 9000, 9000, 0, 1.0f);
  check_shape(32, 8, 500, 500, 0, 8.0f);
}

TEST_CASE("split-K decode SDPA: strided cache views") {
  if (!compute_available()) {
    return;
  }
  check_shape(32, 8, 500, 640, 0, 1.0f);
  check_shape(32, 8, 333, 640, 24, 1.0f);
  check_shape(32, 8, 130, 640, 5, 1.0f);
}

TEST_CASE("split-K decode SDPA: one-hot values and uniform scores expose a dropped or doubled key") {
  if (!compute_available()) {
    return;
  }
  for (int keys : {127, 128, 129, 255, 256, 257, 500}) {
    for (int heads : {8, 16, 32, 64}) {
      check_shape(heads, 8, keys, keys, 0, 1.0f, true);
    }
  }
}

TEST_CASE("split-K decode SDPA: misaligned K views keep the fused route and the unset flag follows the part rule") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int heads = 32;
  const int kv_heads = 8;
  const int keys = 300;
  const int stored_width = 130;
  Stored q = make_bf16(pattern(static_cast<size_t>(heads) * 128, 3, 1.0f), Shape{1, heads, 1, 128}, stream);
  Stored kw = make_bf16(pattern(static_cast<size_t>(kv_heads) * keys * stored_width, 4, 1.0f), Shape{1, kv_heads, keys, stored_width}, stream);
  Stored v = make_bf16(pattern(static_cast<size_t>(kv_heads) * keys * 128, 5, 1.0f), Shape{1, kv_heads, keys, 128}, stream);
  const float scale = 1.0f / std::sqrt(128.0f);
  // Key stride 130 (not a multiple of 8), then the same with a one-element start offset: both fall back to the previous route, bits and dispatches.
  for (int start : {0, 1}) {
    array k = slice(kw.x, Shape{0, 0, 0, start}, Shape{1, kv_heads, keys, start + 128}, stream);
    Run off = run_attention(q.x, k, v.x, scale, "0", stream);
    Run on = run_attention(q.x, k, v.x, scale, "1", stream);
    CHECK(on.launched == off.launched);
    CHECK(on.out == off.out);
  }
  // Flag unset: the route follows the part rule of this device at 300 keys and at 1000 keys (one shape, 32 query heads and 8 key-value heads), with the bits and
  // dispatch count of the route it names.
  const std::string part = omarchy::device(0).capabilities().device_name;
  for (int n : {300, 1000}) {
    Stored kn = make_bf16(pattern(static_cast<size_t>(kv_heads) * n * 128, 6, 1.0f), Shape{1, kv_heads, n, 128}, stream);
    Stored vn = make_bf16(pattern(static_cast<size_t>(kv_heads) * n * 128, 7, 1.0f), Shape{1, kv_heads, n, 128}, stream);
    Run fused = run_attention(q.x, kn.x, vn.x, scale, "0", stream);
    Run split = run_attention(q.x, kn.x, vn.x, scale, "1", stream);
    const bool want_split = split_device_ready() && omarchy::sdpa_split_default(part, static_cast<uint32_t>(n));
    unsetenv("MLX_OMARCHY_SDPA_DECODE_SPLIT");
    uint64_t before = dispatches();
    array r = fast::scaled_dot_product_attention(q.x, kn.x, vn.x, scale, "", {}, std::nullopt, false, stream);
    r.eval();
    omarchy::get_command_encoder(stream).synchronize();
    CHECK(dispatches() - before == (want_split ? split.launched : fused.launched));
    CHECK(flat(r, stream) == (want_split ? split.out : fused.out));
  }
  unsetenv("MLX_OMARCHY_SDPA_DECODE_SPLIT");
}

TEST_CASE("split-K decode SDPA default: the part rule and its key thresholds") {
  CHECK(omarchy::sdpa_split_min_keys("Apple M1 (G13G B1)") == 128);
  CHECK(omarchy::sdpa_split_min_keys("Apple M1 Max (G13C C0)") == 320);
  CHECK(omarchy::sdpa_split_min_keys("Apple M2 Max (G14C B1)") == 448);
  CHECK(omarchy::sdpa_split_min_keys("") == 0);
  CHECK(omarchy::sdpa_split_min_keys("Apple M2 Pro (G14S B1)") == 0);
  CHECK_FALSE(omarchy::sdpa_split_default("Apple M2 Max (G14C B1)", 447));
  CHECK(omarchy::sdpa_split_default("Apple M2 Max (G14C B1)", 448));
  CHECK_FALSE(omarchy::sdpa_split_default("Apple M1 Max (G13C C0)", 319));
  CHECK(omarchy::sdpa_split_default("Apple M1 Max (G13C C0)", 320));
  CHECK_FALSE(omarchy::sdpa_split_default("Apple M1 (G13G B1)", 127));
  CHECK(omarchy::sdpa_split_default("Apple M1 (G13G B1)", 128));
  CHECK(omarchy::sdpa_split_default("Apple M1 (G13G B1)", 1048576));
  CHECK_FALSE(omarchy::sdpa_split_default("", 4096));
}

TEST_CASE("split-K decode SDPA: the value-phase variant default is on up to a key limit on G14C and G13C and off on G13G") {
  CHECK(omarchy::sdpa_split_pv_max_keys("Apple M2 Max (G14C B1)") == 2048);
  CHECK(omarchy::sdpa_split_pv_max_keys("Apple M1 Max (G13C C0)") == 2048);
  CHECK(omarchy::sdpa_split_pv_max_keys("Apple M1 (G13G B1)") == 0);
  CHECK(omarchy::sdpa_split_pv_max_keys("") == 0);
  CHECK(omarchy::sdpa_split_pv_max_keys("Apple M2 Pro (G14S B1)") == 0);
  CHECK(omarchy::sdpa_split_pv_default("Apple M2 Max (G14C B1)", 128));
  CHECK(omarchy::sdpa_split_pv_default("Apple M2 Max (G14C B1)", 2048));
  CHECK_FALSE(omarchy::sdpa_split_pv_default("Apple M2 Max (G14C B1)", 2049));
  CHECK(omarchy::sdpa_split_pv_default("Apple M1 Max (G13C C0)", 2048));
  CHECK_FALSE(omarchy::sdpa_split_pv_default("Apple M1 Max (G13C C0)", 4096));
  CHECK_FALSE(omarchy::sdpa_split_pv_default("Apple M1 (G13G B1)", 128));
  CHECK_FALSE(omarchy::sdpa_split_pv_default("", 128));
}

TEST_CASE("split-K decode SDPA: the value-phase variant falls back to the old pass 1 when V is not 4-aligned") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int heads = 32;
  const int kv_heads = 8;
  const int keys = 300;
  const int stored_width = 130;
  Stored q = make_bf16(pattern(static_cast<size_t>(heads) * 128, 3, 1.0f), Shape{1, heads, 1, 128}, stream);
  Stored k = make_bf16(pattern(static_cast<size_t>(kv_heads) * keys * 128, 4, 1.0f), Shape{1, kv_heads, keys, 128}, stream);
  Stored vw = make_bf16(pattern(static_cast<size_t>(kv_heads) * keys * stored_width, 5, 1.0f), Shape{1, kv_heads, keys, stored_width}, stream);
  // Element offset 2 and key stride 130: even (the old pass 1 accepts it) but not a multiple of 4 (the variant needs that).
  array v = slice(vw.x, Shape{0, 0, 0, 2}, Shape{1, kv_heads, keys, 130}, stream);
  const float scale = 1.0f / std::sqrt(128.0f);
  setenv("MLX_OMARCHY_SDPA_SPLIT_PV", "0", 1);
  Run off = run_attention(q.x, k.x, v, scale, "1", stream);
  setenv("MLX_OMARCHY_SDPA_SPLIT_PV", "1", 1);
  Run on = run_attention(q.x, k.x, v, scale, "1", stream);
  unsetenv("MLX_OMARCHY_SDPA_SPLIT_PV");
  CHECK(on.launched == off.launched);
  CHECK(on.out == off.out);
}

TEST_CASE("split-K decode SDPA: uncovered shapes keep their dispatch count") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Head width 64 and batch 2 are not covered: the flag must not change the route.
  for (int width : {64}) {
    Stored q = make_bf16(pattern(14 * width, 5, 1.0f), Shape{1, 14, 1, width}, stream);
    Stored k = make_bf16(pattern(2 * 300 * width, 6, 1.0f), Shape{1, 2, 300, width}, stream);
    Stored v = make_bf16(pattern(2 * 300 * width, 7, 1.0f), Shape{1, 2, 300, width}, stream);
    float scale = 1.0f / std::sqrt(static_cast<float>(width));
    Run off = run_attention(q.x, k.x, v.x, scale, "0", stream);
    Run on = run_attention(q.x, k.x, v.x, scale, "1", stream);
    CHECK(on.launched == off.launched);
    CHECK(on.out == off.out);
  }
  {
    Stored q = make_bf16(pattern(2 * 32 * 128, 8, 1.0f), Shape{2, 32, 1, 128}, stream);
    Stored k = make_bf16(pattern(2 * 8 * 300 * 128, 9, 1.0f), Shape{2, 8, 300, 128}, stream);
    Stored v = make_bf16(pattern(2 * 8 * 300 * 128, 10, 1.0f), Shape{2, 8, 300, 128}, stream);
    float scale = 1.0f / std::sqrt(128.0f);
    Run off = run_attention(q.x, k.x, v.x, scale, "0", stream);
    Run on = run_attention(q.x, k.x, v.x, scale, "1", stream);
    CHECK(on.launched == off.launched);
    CHECK(on.out == off.out);
  }
  unsetenv("MLX_OMARCHY_SDPA_DECODE_SPLIT");
}
