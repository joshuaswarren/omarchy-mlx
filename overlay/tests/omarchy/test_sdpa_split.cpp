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

uint32_t split_binding_limit(const Stream& stream) {
  return omarchy::get_command_encoder(stream).device().compute().binding_limit();
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
void check_shape(int heads, int kv_heads, int keys, int capacity, int key_start, float amplitude, bool one_hot = false, int width = 128) {
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
  // The 64-key block variant (MLX_OMARCHY_SDPA_SPLIT_BK64) covers group 4 at head width 128 with the value phase on; other shapes keep the 128-key route.
  const bool bk64_shape = heads == 4 * kv_heads && width == 128;
  setenv("MLX_OMARCHY_SDPA_SPLIT_PV", "1", 1);
  setenv("MLX_OMARCHY_SDPA_SPLIT_BK64", "1", 1);
  Run split_bk64 = run_attention(q.x, k, v, scale, "1", stream);
  unsetenv("MLX_OMARCHY_SDPA_SPLIT_BK64");
  unsetenv("MLX_OMARCHY_SDPA_SPLIT_PV");
  Errors e_bk64 = error_against(split_bk64.out, ref);
  if (split_device_ready()) {
    CHECK(split_bk64.launched == 2);
  }
  CHECK(e_bk64.max <= e_fused.max + 2.0 * rounding + 1e-6);
  CHECK(e_bk64.mean <= e_fused.mean + 0.25 * rounding + 1e-7);
  CHECK(e_bk64.max <= 3.0 * rounding + 1e-5);
  if (!bk64_shape) {
    CHECK(split_bk64.out == split_pv.out);
  }
  // The single-dispatch merge variant (MLX_OMARCHY_SDPA_SPLIT_MERGE) covers the same shape as the plain value-phase
  // route at group 4 and head width 128: the value-phase pass 1 with the merge folded into the last block's workgroup is
  // one dispatch, and its merge folds blocks in the same order as the serial pass 2. Other shapes keep the two-dispatch
  // route and its bits. A second call on the same stream exercises the counter pool's self-cleaning.
  const bool merge_shape = heads == 4 * kv_heads && width == 128;
  setenv("MLX_OMARCHY_SDPA_SPLIT_PV", "1", 1);
  setenv("MLX_OMARCHY_SDPA_SPLIT_MERGE", "1", 1);
  Run split_merge = run_attention(q.x, k, v, scale, "1", stream);
  Run split_merge_again = run_attention(q.x, k, v, scale, "1", stream);
  unsetenv("MLX_OMARCHY_SDPA_SPLIT_MERGE");
  unsetenv("MLX_OMARCHY_SDPA_SPLIT_PV");
  Errors e_merge = error_against(split_merge.out, ref);
  if (split_device_ready()) {
    CHECK(split_merge.launched == (merge_shape && split_binding_limit(stream) >= 6 ? 1 : 2));
    CHECK(split_merge_again.launched == split_merge.launched);
    CHECK(split_merge_again.out == split_merge.out);
  }
  CHECK(e_merge.max <= e_fused.max + 2.0 * rounding + 1e-6);
  CHECK(e_merge.mean <= e_fused.mean + 0.25 * rounding + 1e-7);
  CHECK(e_merge.max <= 3.0 * rounding + 1e-5);
  if (!merge_shape) {
    CHECK(split_merge.out == split_pv.out);
  }
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

TEST_CASE("split-K decode SDPA: head width 256 (the Qwen3.5 and Qwen3.8 full-attention layers), groups 1, 2, 4 and 8") {
  if (!compute_available()) {
    return;
  }
  for (int keys : {1, 17, 127, 128, 129, 255, 257, 500, 2048, 4097}) {
    check_shape(16, 4, keys, keys, 0, 1.0f, false, 256);
  }
  check_shape(8, 2, 777, 777, 0, 1.0f, false, 256);
  check_shape(8, 8, 300, 300, 0, 1.0f, false, 256);
  check_shape(8, 4, 513, 513, 0, 1.0f, false, 256);
  check_shape(32, 4, 400, 400, 0, 1.0f, false, 256);
  check_shape(16, 4, 9000, 9000, 0, 1.0f, false, 256);
  check_shape(16, 4, 500, 500, 0, 8.0f, false, 256);
  check_shape(16, 4, 500, 640, 0, 1.0f, false, 256);
  check_shape(16, 4, 333, 640, 24, 1.0f, false, 256);
}

TEST_CASE("split-K decode SDPA: head width 256 one-hot values and uniform scores expose a dropped or doubled key") {
  if (!compute_available()) {
    return;
  }
  for (int keys : {127, 128, 129, 255, 256, 257, 500}) {
    for (int heads : {4, 8, 16, 32}) {
      check_shape(heads, 4, keys, keys, 0, 1.0f, true, 256);
    }
  }
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
  CHECK(omarchy::sdpa_split_min_keys("Apple M1 Max (G13C C0)") == 256);
  CHECK(omarchy::sdpa_split_min_keys("Apple M2 Max (G14C B1)") == 128);
  CHECK(omarchy::sdpa_split_min_keys("") == 320);
  CHECK(omarchy::sdpa_split_min_keys("Apple M1 Pro (G13S C0)") == 192);
  CHECK(omarchy::sdpa_split_min_keys("Apple M2 (G14G B0)") == 192);
  CHECK_FALSE(omarchy::sdpa_split_default("Apple M1 Pro (G13S C0)", 191));
  CHECK(omarchy::sdpa_split_default("Apple M1 Pro (G13S C0)", 192));
  CHECK_FALSE(omarchy::sdpa_split_default("Apple M2 (G14G B0)", 191));
  CHECK(omarchy::sdpa_split_default("Apple M2 (G14G B0)", 192));
  CHECK(omarchy::sdpa_split_min_keys("Apple M2 Pro (G14S B1)") == 320);
  // BMG G21 (Intel Arc Pro B60 eGPU): EG-1 showed the fused kernel at ~2 GB/s
  // on a fixed 32-group grid; EG-2 measured the split route winning at 512.
  CHECK(omarchy::sdpa_split_min_keys("Intel(R) Arc(tm) Pro B60 Graphics (BMG G21)") == 512);
  CHECK_FALSE(omarchy::sdpa_split_default("Intel(R) Arc(tm) Pro B60 Graphics (BMG G21)", 511));
  CHECK(omarchy::sdpa_split_default("Intel(R) Arc(tm) Pro B60 Graphics (BMG G21)", 512));
  // Ultra exclusions (#418): neither Ultra part is qualified for a nonzero
  // default, so both keep the fused kernel (min_keys 0) and the default is
  // off at any key length.
  CHECK(omarchy::sdpa_split_min_keys("Apple M1 Ultra (G13D B1)") == 320);
  CHECK(omarchy::sdpa_split_min_keys("Apple M2 Ultra (G14D B1)") == 320);
  CHECK(omarchy::sdpa_split_default("Apple M1 Ultra (G13D B1)", 4096));
  CHECK(omarchy::sdpa_split_default("Apple M2 Ultra (G14D B1)", 4096));
  CHECK_FALSE(omarchy::sdpa_split_default("Apple M2 Max (G14C B1)", 127));
  CHECK(omarchy::sdpa_split_default("Apple M2 Max (G14C B1)", 128));
  CHECK_FALSE(omarchy::sdpa_split_default("Apple M1 Max (G13C C0)", 255));
  CHECK(omarchy::sdpa_split_default("Apple M1 Max (G13C C0)", 256));
  CHECK_FALSE(omarchy::sdpa_split_default("Apple M1 (G13G B1)", 127));
  CHECK(omarchy::sdpa_split_default("Apple M1 (G13G B1)", 128));
  CHECK(omarchy::sdpa_split_default("Apple M1 (G13G B1)", 1048576));
  CHECK(omarchy::sdpa_split_default("", 4096));
  CHECK(omarchy::sdpa_split_min_keys("Apple M4 Pro (G16S B1)") == 320);
  CHECK_FALSE(omarchy::sdpa_split_default("Apple M4 Pro (G16S B1)", 319));
  CHECK(omarchy::sdpa_split_default("Apple M4 Pro (G16S B1)", 320));
  CHECK(omarchy::sdpa_split_min_keys("Apple M4 (G16G B0)") == 320);
  CHECK(omarchy::sdpa_split_min_keys("Apple M5 Max (G17C B1)") == 320);
  CHECK_FALSE(omarchy::sdpa_split_default("Apple M5 Max (G17C B1)", 319));
  CHECK(omarchy::sdpa_split_default("Apple M5 Max (G17C B1)", 320));
  // Head width 256: G16S starts at 768 keys, G14C at 256, G15S at 1408; every other part keeps its width-128 threshold.
  CHECK(omarchy::sdpa_split_wide_min_keys("Apple M4 Pro (G16S B1)") == 768);
  CHECK_FALSE(omarchy::sdpa_split_default("Apple M4 Pro (G16S B1)", 767, 256));
  CHECK(omarchy::sdpa_split_default("Apple M4 Pro (G16S B1)", 768, 256));
  CHECK(omarchy::sdpa_split_wide_min_keys("Apple M2 Max (G14C B1)") == 256);
  CHECK_FALSE(omarchy::sdpa_split_default("Apple M2 Max (G14C B1)", 255, 256));
  CHECK(omarchy::sdpa_split_default("Apple M2 Max (G14C B1)", 256, 256));
  CHECK(omarchy::sdpa_split_wide_min_keys("Apple M5 Max (G17C B1)") == 320);
  CHECK_FALSE(omarchy::sdpa_split_default("Apple M5 Max (G17C B1)", 319, 256));
  CHECK(omarchy::sdpa_split_default("Apple M5 Max (G17C B1)", 320, 256));
  CHECK(omarchy::sdpa_split_default("Apple M4 Pro (G16S B1)", 320, 128));
  CHECK(omarchy::sdpa_split_wide_min_keys("Apple M1 Max (G13C C0)") == 320);
  CHECK_FALSE(omarchy::sdpa_split_default("Apple M1 Max (G13C C0)", 319, 256));
  CHECK(omarchy::sdpa_split_default("Apple M1 Max (G13C C0)", 320, 256));
  CHECK(omarchy::sdpa_split_wide_min_keys("Apple M1 (G13G B1)") == 128);
  CHECK(omarchy::sdpa_split_wide_min_keys("") == 320);
  CHECK(omarchy::sdpa_split_default("", 4096, 256));
}

TEST_CASE("split-K decode SDPA: the value-phase variant default is on up to a key limit on G14C, G13C and G16S and off on G13G") {
  CHECK(omarchy::sdpa_split_pv_max_keys("Apple M2 Max (G14C B1)") == 2176);
  CHECK(omarchy::sdpa_split_pv_max_keys("Apple M1 Max (G13C C0)") == 1152);
  CHECK(omarchy::sdpa_split_pv_max_keys("Apple M1 (G13G B1)") == 0);
  CHECK(omarchy::sdpa_split_pv_max_keys("") == 2048);
  CHECK(omarchy::sdpa_split_pv_max_keys("Apple M2 Pro (G14S B1)") == 2048);
  CHECK(omarchy::sdpa_split_pv_default("Apple M2 Max (G14C B1)", 128));
  CHECK(omarchy::sdpa_split_pv_default("Apple M2 Max (G14C B1)", 2176));
  CHECK_FALSE(omarchy::sdpa_split_pv_default("Apple M2 Max (G14C B1)", 2177));
  CHECK(omarchy::sdpa_split_pv_default("Apple M1 Max (G13C C0)", 1152));
  CHECK_FALSE(omarchy::sdpa_split_pv_default("Apple M1 Max (G13C C0)", 1153));
  CHECK_FALSE(omarchy::sdpa_split_pv_default("Apple M1 (G13G B1)", 128));
  CHECK(omarchy::sdpa_split_pv_default("", 128));
  CHECK(omarchy::sdpa_split_pv_max_keys("Apple M4 Pro (G16S B1)") == 16384);
  CHECK(omarchy::sdpa_split_pv_default("Apple M4 Pro (G16S B1)", 320));
  CHECK(omarchy::sdpa_split_pv_default("Apple M4 Pro (G16S B1)", 16384));
  CHECK_FALSE(omarchy::sdpa_split_pv_default("Apple M4 Pro (G16S B1)", 16384 + 1));
  CHECK(omarchy::sdpa_split_pv_max_keys("Apple M5 Max (G17C B1)") == 2048);
  CHECK(omarchy::sdpa_split_pv_default("Apple M5 Max (G17C B1)", 320));
  CHECK(omarchy::sdpa_split_pv_default("Apple M5 Max (G17C B1)", 2048));
  CHECK_FALSE(omarchy::sdpa_split_pv_default("Apple M5 Max (G17C B1)", 2048 + 1));
}

TEST_CASE("split-K decode SDPA: the 64-key block variant default is on up to per-part key limits; unlisted parts take the 512-key newest-class ceiling") {
  CHECK(omarchy::sdpa_split_bk64_max_keys("Apple M2 Max (G14C B1)") == 1088);
  CHECK(omarchy::sdpa_split_bk64_max_keys("Apple M1 Max (G13C C0)") == 640);
  CHECK(omarchy::sdpa_split_bk64_max_keys("Apple M1 (G13G B1)") == 512);
  CHECK(omarchy::sdpa_split_bk64_max_keys("") == 512);
  CHECK(omarchy::sdpa_split_bk64_default("Apple M2 Max (G14C B1)", 128));
  CHECK(omarchy::sdpa_split_bk64_default("Apple M2 Max (G14C B1)", 1088));
  CHECK_FALSE(omarchy::sdpa_split_bk64_default("Apple M2 Max (G14C B1)", 1089));
  CHECK(omarchy::sdpa_split_bk64_default("Apple M1 Max (G13C C0)", 512));
  CHECK(omarchy::sdpa_split_bk64_default("Apple M1 Max (G13C C0)", 640));
  CHECK_FALSE(omarchy::sdpa_split_bk64_default("Apple M1 Max (G13C C0)", 641));
  CHECK(omarchy::sdpa_split_bk64_default("", 512));
  CHECK(omarchy::sdpa_split_bk64_max_keys("Apple M4 Pro (G16S B1)") == 512);
  CHECK(omarchy::sdpa_split_bk64_default("Apple M4 Pro (G16S B1)", 320));
  CHECK_FALSE(omarchy::sdpa_split_bk64_default("Apple M4 Pro (G16S B1)", 513));
  CHECK(omarchy::sdpa_split_bk64_max_keys("Apple M5 Max (G17C B1)") == 512);
  CHECK(omarchy::sdpa_split_bk64_default("Apple M5 Max (G17C B1)", 320));
  CHECK_FALSE(omarchy::sdpa_split_bk64_default("Apple M5 Max (G17C B1)", 513));
}

TEST_CASE("split-K decode SDPA: the width-256 value-phase variant needs group 4, V alignment, and follows its per-chip workgroup limit when unset") {
  const std::string m1 = "Apple M1 (G13G B1)";
  const std::string m1max = "Apple M1 Max (G13C C0)";
  const std::string m2max = "Apple M2 Max (G14C B1)";
  // Explicit flag value: selected at any part, any length and head count the split route covers.
  CHECK(omarchy::sdpa_split_pv_wide(256, 4, 2, 1, 9000, m1, 0, 8192, 256));
  CHECK(omarchy::sdpa_split_pv_wide(256, 4, 2, 1, 500, "", 0, 8192, 256));
  CHECK(omarchy::sdpa_split_pv_wide(256, 4, 8, 1, 9000, m1max, 0, 8192, 256));
  // Unset: pass 1 launches one workgroup per key-value head and 128-key block; the variant is on while that count is at most 16 on G13G, 64 on G13C and 68 on G14C.
  // The limit is one block wide: 128 keys are one block, 129 are two.
  CHECK(omarchy::sdpa_split_pv_wide_max_workgroups(m1) == 16);
  CHECK(omarchy::sdpa_split_pv_wide_max_workgroups(m1max) == 64);
  CHECK(omarchy::sdpa_split_pv_wide_max_workgroups(m2max) == 68);
  CHECK(omarchy::sdpa_split_pv_wide_max_workgroups("Apple M5 Max (G17C B1)") == 68);
  CHECK(omarchy::sdpa_split_pv_wide_max_workgroups("") == 68);
  struct Edge { uint32_t kv_heads; uint32_t last_on; };
  // G13G (16 workgroups): 1 head 2048 keys, 2 heads 1024 (the Qwen3.8-2B), 3 heads 640, 4 heads 512 (the Qwen3.8-9B), 8 heads 256.
  for (Edge e : {Edge{1, 2048}, Edge{2, 1024}, Edge{3, 640}, Edge{4, 512}, Edge{8, 256}}) {
    CHECK(omarchy::sdpa_split_pv_wide(256, 4, e.kv_heads, -1, e.last_on, m1, 0, 8192, 256));
    CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, e.kv_heads, -1, e.last_on + 1, m1, 0, 8192, 256));
  }
  // G13C (64 workgroups): 1 head 8192 keys, 2 heads 4096 (the 2B), 3 heads 2688, 4 heads 2048 (the 9B), 8 heads 1024.
  for (Edge e : {Edge{1, 8192}, Edge{2, 4096}, Edge{3, 2688}, Edge{4, 2048}, Edge{8, 1024}}) {
    CHECK(omarchy::sdpa_split_pv_wide(256, 4, e.kv_heads, -1, e.last_on, m1max, 0, 8192, 256));
    CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, e.kv_heads, -1, e.last_on + 1, m1max, 0, 8192, 256));
  }
  // G14C (68 workgroups): 1 head 8704 keys, 2 heads 4352 (the 2B), 3 heads 2816, 4 heads 2176 (the 9B at 2048 context), 8 heads 1024.
  for (Edge e : {Edge{1, 8704}, Edge{2, 4352}, Edge{3, 2816}, Edge{4, 2176}, Edge{8, 1024}}) {
    CHECK(omarchy::sdpa_split_pv_wide(256, 4, e.kv_heads, -1, e.last_on, m2max, 0, 8192, 256));
    CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, e.kv_heads, -1, e.last_on + 1, m2max, 0, 8192, 256));
  }
  CHECK(omarchy::sdpa_split_pv_wide(256, 4, 2, -1, 320, m1max, 0, 8192, 256));
  CHECK(omarchy::sdpa_split_pv_wide(256, 4, 2, 1, 4097, m1max, 0, 8192, 256));  // the explicit flag still selects it past the limit
  // Unlisted parts take the newest-class workgroup limit; no heads or no keys must not pass on a count of zero.
  CHECK(omarchy::sdpa_split_pv_wide(256, 4, 2, -1, 500, "", 0, 8192, 256));
  CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, 2, -1, 0, m1max, 0, 8192, 256));
  CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, 0, -1, 100, m1max, 0, 8192, 256));
  CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, 0, -1, 0, m1, 0, 8192, 256));
  // A 32-bit product of key-value heads and blocks would wrap to 4 here (0x40000001 heads times 4 blocks is 2^32 + 4); the count is 64-bit.
  CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, 0x40000001u, -1, 512, m1max, 0, 8192, 256));
  CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, 0x40000001u, -1, 512, m1, 0, 8192, 256));
  // "0" keeps the old pass 1 everywhere.
  CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, 2, 0, 500, m1, 0, 8192, 256));
  // Other widths and groups keep their own routes (group 8 would not fit its partial-sum tile in shared memory).
  CHECK_FALSE(omarchy::sdpa_split_pv_wide(128, 4, 2, 1, 500, m1, 0, 8192, 256));
  for (uint32_t group : {1u, 2u, 8u}) {
    CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, group, 2, 1, 500, m1, 0, 8192, 256));
  }
  // V alignment: each of offset, head stride and key stride alone breaks it; multiples of 4 (enough for the width-128 variant) are not enough here.
  CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, 2, 1, 500, m1, 4, 8192, 256));
  CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, 2, 1, 500, m1, 0, 8196, 256));
  CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, 2, 1, 500, m1, 0, 8192, 252));
  CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, 2, -1, 500, m1, 0, 8192, 252));
}

TEST_CASE("split-K decode SDPA: the M3 Pro (G15S) defaults and their variable overrides; the unlisted M3 (G15G) takes the newest-class defaults") {
  const std::string m3pro = "Apple M3 Pro (G15S B1)";
  const std::string m3 = "Apple M3 (G15G B1)";
  CHECK(omarchy::sdpa_split_min_keys(m3pro) == 192);
  CHECK_FALSE(omarchy::sdpa_split_default(m3pro, 191));
  CHECK(omarchy::sdpa_split_default(m3pro, 192));
  CHECK(omarchy::sdpa_split_wide_min_keys(m3pro) == 1408);
  CHECK_FALSE(omarchy::sdpa_split_default(m3pro, 1407, 256));
  CHECK(omarchy::sdpa_split_default(m3pro, 1408, 256));
  CHECK(omarchy::sdpa_split_pv_default(m3pro, 1280));
  CHECK_FALSE(omarchy::sdpa_split_pv_default(m3pro, 1281));
  CHECK(omarchy::sdpa_split_bk64_default(m3pro, 640));
  CHECK_FALSE(omarchy::sdpa_split_bk64_default(m3pro, 641));
  // Width 256, 36 workgroups: 2 key-value heads (the Qwen3.8-2B) through 2304 keys, 4 heads through 1152.
  CHECK(omarchy::sdpa_split_pv_wide_max_workgroups(m3pro) == 36);
  CHECK(omarchy::sdpa_split_pv_wide(256, 4, 2, -1, 2304, m3pro, 0, 8192, 256));
  CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, 2, -1, 2305, m3pro, 0, 8192, 256));
  CHECK(omarchy::sdpa_split_pv_wide(256, 4, 4, -1, 1152, m3pro, 0, 8192, 256));
  CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, 4, -1, 1153, m3pro, 0, 8192, 256));
  // The base M3 is unlisted: every rule gives it the newest-class default.
  CHECK(omarchy::sdpa_split_min_keys(m3) == 320);
  CHECK(omarchy::sdpa_split_wide_min_keys(m3) == 320);
  CHECK(omarchy::sdpa_split_pv_max_keys(m3) == 2048);
  CHECK(omarchy::sdpa_split_bk64_max_keys(m3) == 512);
  CHECK(omarchy::sdpa_split_pv_wide_max_workgroups(m3) == 68);
  CHECK(omarchy::sdpa_split_default(m3, 4096));
  // Each variable replaces the part value; the wide threshold falls back to the width-128 variable on a part without its own row.
  for (const char* name : {"MLX_OMARCHY_SDPA_SPLIT_MIN_KEYS", "MLX_OMARCHY_SDPA_SPLIT_WIDE_MIN_KEYS", "MLX_OMARCHY_SDPA_SPLIT_PV_MAX_KEYS",
                           "MLX_OMARCHY_SDPA_SPLIT_BK64_MAX_KEYS", "MLX_OMARCHY_SDPA_HD256_PV_LIMIT"}) {
    REQUIRE(std::getenv(name) == nullptr);
  }
  setenv("MLX_OMARCHY_SDPA_SPLIT_MIN_KEYS", "4096", 1);
  CHECK(omarchy::sdpa_split_min_keys(m3pro) == 4096);
  CHECK(omarchy::sdpa_split_min_keys(m3) == 4096);
  CHECK(omarchy::sdpa_split_wide_min_keys(m3) == 4096);
  CHECK(omarchy::sdpa_split_wide_min_keys(m3pro) == 1408);
  unsetenv("MLX_OMARCHY_SDPA_SPLIT_MIN_KEYS");
  setenv("MLX_OMARCHY_SDPA_SPLIT_WIDE_MIN_KEYS", "0", 1);
  CHECK_FALSE(omarchy::sdpa_split_default(m3pro, 4096, 256));
  CHECK(omarchy::sdpa_split_default(m3pro, 4096, 128));
  unsetenv("MLX_OMARCHY_SDPA_SPLIT_WIDE_MIN_KEYS");
  setenv("MLX_OMARCHY_SDPA_SPLIT_PV_MAX_KEYS", "0", 1);
  CHECK_FALSE(omarchy::sdpa_split_pv_default(m3pro, 320));
  unsetenv("MLX_OMARCHY_SDPA_SPLIT_PV_MAX_KEYS");
  setenv("MLX_OMARCHY_SDPA_SPLIT_BK64_MAX_KEYS", "0", 1);
  CHECK_FALSE(omarchy::sdpa_split_bk64_default(m3pro, 320));
  unsetenv("MLX_OMARCHY_SDPA_SPLIT_BK64_MAX_KEYS");
  setenv("MLX_OMARCHY_SDPA_HD256_PV_LIMIT", "0", 1);
  CHECK_FALSE(omarchy::sdpa_split_pv_wide(256, 4, 2, -1, 500, m3pro, 0, 8192, 256));
  unsetenv("MLX_OMARCHY_SDPA_HD256_PV_LIMIT");
  CHECK(omarchy::sdpa_split_pv_wide(256, 4, 2, -1, 500, m3pro, 0, 8192, 256));
}

TEST_CASE("split-K decode SDPA: the width-256 value-phase variant meets the float64 gate at group 4, and the flag-off arm keeps the old pass 1") {
  if (!compute_available()) {
    return;
  }
  // check_shape runs three arms on every shape: the fused kernel, the split route with the flag off, and the split route with the flag on.
  // At these head-width-256 group-4 shapes the flag-on arm is the wide value-phase kernel (omarchy::sdpa_split_pv_wide selects it:
  // the selector test above pins that rule for exactly this width, group and alignment), the flag-off arm stays on the old pass 1,
  // and both are gated against the float64 reference with the same bounds the width-128 value-phase case uses.
  check_shape(8, 2, 500, 500, 0, 1.0f, false, 256);
  check_shape(8, 2, 1536, 1536, 0, 1.0f, false, 256);  // 2 key-value heads at the middle of the G13C range
  check_shape(8, 2, 3072, 3072, 0, 1.0f, false, 256);  // 2 key-value heads at the G13C table limit
  check_shape(16, 4, 1536, 1536, 0, 1.0f, false, 256); // 4 key-value heads at the G13C table limit
  check_shape(32, 8, 768, 768, 0, 1.0f, false, 256);   // 8 key-value heads (group 4 over 32 query heads)
  check_shape(16, 4, 777, 777, 0, 1.0f, false, 256);
  check_shape(16, 4, 129, 129, 0, 1.0f, false, 256);  // a single partial block
  check_shape(16, 4, 500, 640, 0, 1.0f, false, 256);  // strided view
  check_shape(16, 4, 500, 500, 0, 8.0f, false, 256);  // large magnitude
  check_shape(16, 4, 333, 640, 24, 1.0f, false, 256); // offset strided view
  check_shape(16, 4, 257, 257, 0, 1.0f, true, 256);   // one-hot values: a dropped or doubled key moves the output
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

TEST_CASE("split-K verify rows: each causal row equals the same token decoded alone, two dispatches per row") {
  if (!compute_available()) {
    return;
  }
  struct Case {
    int heads, kv_heads, width, keys, rows;
  };
  for (const Case& c : {Case{32, 8, 128, 500, 2}, Case{32, 8, 128, 2048, 4}, Case{32, 8, 128, 3000, 16}, Case{16, 4, 256, 700, 3}, Case{16, 16, 128, 130, 8}}) {
    Stream stream = gpu_stream();
    Stored q = make_bf16(pattern(static_cast<size_t>(c.heads) * c.rows * c.width, 5, 1.0f), Shape{1, c.heads, c.rows, c.width}, stream);
    Stored k = make_bf16(pattern(static_cast<size_t>(c.kv_heads) * c.keys * c.width, 6, 1.0f), Shape{1, c.kv_heads, c.keys, c.width}, stream);
    Stored v = make_bf16(pattern(static_cast<size_t>(c.kv_heads) * c.keys * c.width, 7, 1.0f), Shape{1, c.kv_heads, c.keys, c.width}, stream);
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.width));
    setenv("MLX_OMARCHY_SDPA_DECODE_SPLIT", "1", 1);
    // Pins the default: the route is on both with the flag unset and set
    // to "1"; "0" is the opt-out back to the composed route.
    for (const char* rows_env : {static_cast<const char*>(nullptr), "1"}) {
      if (rows_env) {
        setenv("MLX_OMARCHY_SDPA_SPLIT_ROWS", rows_env, 1);
      } else {
        unsetenv("MLX_OMARCHY_SDPA_SPLIT_ROWS");
      }
      uint64_t before = dispatches();
      array rows = fast::scaled_dot_product_attention(q.x, k.x, v.x, scale, "causal", {}, std::nullopt, false, stream);
      rows.eval();
      omarchy::get_command_encoder(stream).synchronize();
      uint64_t launched = dispatches() - before;
      std::vector<float> got = flat(rows, stream);
      INFO("heads " << c.heads << " width " << c.width << " keys " << c.keys << " rows " << c.rows
           << " rows_env " << (rows_env ? rows_env : "unset"));
      if (!split_device_ready()) {
        printf("Skipping: split route not available on this device\n");
        break;
      }
      CHECK(launched == 2u * static_cast<uint64_t>(c.rows));
      for (int r = 0; r < c.rows; ++r) {
        const int visible = c.keys - c.rows + 1 + r;
        array qr = slice(q.x, Shape{0, 0, r, 0}, Shape{1, c.heads, r + 1, c.width}, stream);
        array kr = slice(k.x, Shape{0, 0, 0, 0}, Shape{1, c.kv_heads, visible, c.width}, stream);
        array vr = slice(v.x, Shape{0, 0, 0, 0}, Shape{1, c.kv_heads, visible, c.width}, stream);
        Run one = run_attention(qr, kr, vr, scale, "1", stream);
        for (int h = 0; h < c.heads; ++h) {
          for (int d = 0; d < c.width; ++d) {
            REQUIRE(got[(static_cast<size_t>(h) * c.rows + r) * c.width + d] == one.out[static_cast<size_t>(h) * c.width + d]);
          }
        }
      }
    }
    unsetenv("MLX_OMARCHY_SDPA_SPLIT_ROWS");
  }
  unsetenv("MLX_OMARCHY_SDPA_DECODE_SPLIT");
}

TEST_CASE("split-K P1KV: float64 accuracy gate and two-dispatch count across HD=128/256 and BK=32/64/128") {
  if (!compute_available()) {
    return;
  }
  const auto& caps = omarchy::device(0).capabilities();
  const bool shuffle_ready =
      (caps.subgroup_operations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0u;
  struct ShapeCase {
    int heads, kv_heads, width, keys;
    bool one_hot;
  };
  for (const ShapeCase& sc : {
           ShapeCase{32, 8, 128, 1, false},
           ShapeCase{32, 8, 128, 17, false},
           ShapeCase{32, 8, 128, 31, false},
           ShapeCase{32, 8, 128, 33, false},
           ShapeCase{32, 8, 128, 65, true},
           ShapeCase{32, 8, 128, 513, false},
           ShapeCase{16, 4, 256, 17, false},
           ShapeCase{16, 4, 256, 65, true},
           ShapeCase{16, 4, 256, 513, false},
       }) {
    Stream stream = gpu_stream();
    Stored q = make_bf16(
        pattern(static_cast<size_t>(sc.heads) * sc.width, 11, 1.0f),
        Shape{1, sc.heads, 1, sc.width},
        stream);
    std::vector<float> k_host =
        pattern(static_cast<size_t>(sc.kv_heads) * sc.keys * sc.width, 23, 1.0f);
    std::vector<float> v_host =
        pattern(static_cast<size_t>(sc.kv_heads) * sc.keys * sc.width, 37, 1.0f);
    if (sc.one_hot) {
      std::fill(k_host.begin(), k_host.end(), 0.0f);
      for (size_t i = 0; i != v_host.size(); ++i) {
        size_t key = (i / sc.width) % sc.keys;
        v_host[i] = (i % sc.width) == (key % sc.width) ? 1.0f : 0.0f;
      }
    }
    Stored k = make_bf16(k_host, Shape{1, sc.kv_heads, sc.keys, sc.width}, stream);
    Stored v = make_bf16(v_host, Shape{1, sc.kv_heads, sc.keys, sc.width}, stream);
    const float scale = 1.0f / std::sqrt(static_cast<float>(sc.width));
    std::vector<double> ref = reference(
        q.values, k.values, v.values, sc.heads, sc.kv_heads, sc.keys, sc.keys, 0, sc.width, scale);
    Run baseline = run_attention(q.x, k.x, v.x, scale, "0", stream);
    Errors e_base = error_against(baseline.out, ref);
    double rounding = 0.0;
    for (double r : ref) {
      rounding = std::max(rounding, std::fabs(r) * std::ldexp(1.0, -8));
    }
    for (const char* bk : {"32", "64", "128"}) {
      setenv("MLX_OMARCHY_SDPA_SPLIT_P1KV", bk, 1);
      Run p1kv = run_attention(q.x, k.x, v.x, scale, "1", stream);
      unsetenv("MLX_OMARCHY_SDPA_SPLIT_P1KV");
      Errors e_p1kv = error_against(p1kv.out, ref);
      INFO("width=" << sc.width << " keys=" << sc.keys << " bk=" << bk);
      CHECK(e_p1kv.max <= e_base.max + 2.0 * rounding + 1e-6);
      CHECK(e_p1kv.mean <= e_base.mean + 0.25 * rounding + 1e-7);
      CHECK(e_p1kv.max <= 3.0 * rounding + 1e-5);
      if (split_device_ready() && (sc.width == 256 || shuffle_ready)) {
        CHECK(p1kv.launched == 2);
      }
    }
  }
  unsetenv("MLX_OMARCHY_SDPA_DECODE_SPLIT");
}
