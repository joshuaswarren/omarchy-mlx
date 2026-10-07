// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Multi-row greedy head (fast::greedy_quantized_argmax over [B, K] decode
// rows, one QMM_VEC_GREEDY dispatch y slice per row; the batch greedy head of
// scripts/patch-mlx-lm-batch-greedy-head.py). Every row must return the token
// and the three stats words its own 1-row call returns, and that token must be
// the composed argmax(logits - logsumexp(logits)) of the row. The CPU case
// pins the composed fallback's per-row contract (token [B], stats [3 B]).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <unistd.h>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

constexpr int kK = 2048;

struct Head {
  array w;
  array scales;
  array biases;
  array high;
};

// q >> 1 of every code, code e of a 64-code group at bit 3e of its six words.
array high_plane(const array& w) {
  int n = w.shape(0);
  int words = w.shape(1);
  int k = words * 8;
  int plane_words = k * 3 / 32;
  std::vector<uint32_t> plane(static_cast<size_t>(n) * plane_words, 0u);
  const uint32_t* src = w.data<uint32_t>();
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < k; ++c) {
      uint32_t a = ((src[static_cast<size_t>(r) * words + c / 8] >> (4 * (c % 8))) & 15u) >> 1;
      size_t bit = static_cast<size_t>(c / 64) * 192 + 3 * (c % 64);
      for (int j = 0; j < 3; ++j, ++bit) {
        if ((a >> j) & 1u) {
          plane[static_cast<size_t>(r) * plane_words + bit / 32] |= 1u << (bit % 32);
        }
      }
    }
  }
  return array(plane.begin(), Shape{n, plane_words}, uint32);
}

Head make_head(int n, Stream s) {
  array w = multiply(random::normal(Shape{n, kK}, float32, std::nullopt, s), array(0.05f), s);
  auto parts = quantize(w, 64, 4, "affine", std::nullopt, s);
  eval(parts);
  Head h{parts[0], astype(parts[1], bfloat16, s), astype(parts[2], bfloat16, s), high_plane(parts[0])};
  eval({h.w, h.scales, h.biases, h.high});
  return h;
}

std::vector<array> greedy(const array& x, const Head& h, Stream s) {
  auto out = fast::greedy_quantized_argmax(x, h.w, h.scales, h.biases, h.high, 64, 4, s);
  eval(out);
  return out;
}

std::vector<uint32_t> host(const array& a) {
  const uint32_t* p = a.data<uint32_t>();
  return std::vector<uint32_t>(p, p + a.size());
}

// QmmVecGreedyBF16 dispatches while `run` evaluates, from the encoder's
// MLX_OMARCHY_TRACE_DISPATCH lines on stderr.
int greedy_dispatches(const std::function<void()>& run) {
  std::fflush(stderr);
  int saved = dup(2);
  std::FILE* sink = std::tmpfile();
  dup2(fileno(sink), 2);
  setenv("MLX_OMARCHY_TRACE_DISPATCH", "1", 1);
  run();
  unsetenv("MLX_OMARCHY_TRACE_DISPATCH");
  std::fflush(stderr);
  dup2(saved, 2);
  close(saved);
  std::rewind(sink);
  int count = 0;
  char line[512];
  while (std::fgets(line, sizeof(line), sink)) {
    int id = -1;
    if (std::sscanf(line, "[rtmod] DISPATCH kernel=%d", &id) == 1 &&
        id == static_cast<int>(omarchy::ComputeKernel::QmmVecGreedyBF16)) {
      ++count;
    }
  }
  std::fclose(sink);
  return count;
}

} // namespace

TEST_CASE("the composed fallback answers every row with its own stats triple") {
  // A release build refuses every tensor when the GPU backend is unavailable.
  if (!gpu::is_available()) {
    std::cout << "Skipping: no qualifying Vulkan device\n";
    return;
  }
  Stream s = new_stream(Device::cpu);
  random::seed(11);
  Head h = make_head(300, s);
  array x = astype(random::normal(Shape{3, kK}, float32, std::nullopt, s), bfloat16, s);
  auto out = greedy(x, h, s);
  REQUIRE(out[0].shape() == Shape{3});
  REQUIRE(out[1].shape() == Shape{9});
  array logits = quantized_matmul(x, h.w, h.scales, h.biases, true, 64, 4, "affine", s);
  array want = argmax(subtract(logits, logsumexp(logits, -1, true, s), s), -1, false, s);
  eval(want);
  CHECK(array_equal(astype(out[0], uint32, s), astype(want, uint32, s), s).item<bool>());
  CHECK(host(out[1]) == std::vector<uint32_t>{300, 1, 0, 300, 1, 0, 300, 1, 0});
}

TEST_CASE("multi-row greedy rows equal their 1-row calls bit for bit") {
  if (!gpu::is_available()) {
    std::cout << "Skipping: no qualifying Vulkan device\n";
    return;
  }
  set_default_device(Device::gpu);
  Stream s = new_stream(Device::gpu);
  for (int n : {4096, 20000}) {
    random::seed(1000003u + n);
    Head h = make_head(n, s);
    array probe = astype(random::normal(Shape{1, kK}, float32, std::nullopt, s), bfloat16, s);
    probe.eval();
    if (greedy_dispatches([&] { greedy(probe, h, s); }) == 0) {
      std::cout << "Skipping: the greedy head route does not engage here\n";
      return;
    }
    for (int b : {2, 3, 4, 5, 8}) {
      CAPTURE(n);
      CAPTURE(b);
      array x = astype(random::normal(Shape{b, 1, kK}, float32, std::nullopt, s), bfloat16, s);
      x.eval();
      // All rows share each stage's dispatch: 8 stages, not 8 per row.
      CHECK(greedy_dispatches([&] { greedy(x, h, s); }) == 8);
      auto got = greedy(x, h, s);
      REQUIRE(got[0].shape() == Shape{b, 1});
      REQUIRE(got[1].shape() == Shape{3 * b});
      std::vector<uint32_t> tokens = host(got[0]);
      std::vector<uint32_t> stats = host(got[1]);
      for (int r = 0; r < b; ++r) {
        CAPTURE(r);
        array row = slice(x, {r, 0, 0}, {r + 1, 1, kK}, s);
        auto one = greedy(reshape(row, {1, kK}, s), h, s);
        std::vector<uint32_t> one_stats = host(one[1]);
        CHECK(tokens[r] == host(one[0])[0]);
        CHECK(std::vector<uint32_t>(stats.begin() + 3 * r, stats.begin() + 3 * r + 3) == one_stats);
        array logits = quantized_matmul(row, h.w, h.scales, h.biases, true, 64, 4, "affine", s);
        array want = argmax(subtract(logits, logsumexp(logits, -1, true, s), s), -1, false, s);
        CHECK(tokens[r] == static_cast<uint32_t>(want.item<uint32_t>()));
      }
    }
  }
}
