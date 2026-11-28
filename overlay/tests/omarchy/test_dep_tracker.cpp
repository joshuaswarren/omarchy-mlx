// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// CPU tests for the per-buffer dependency tracker: the hazard answers
// (RAW, WAW, WAR, disjoint, same buffer without overlap, touching,
// empty ranges) and a randomized check that the indexed tracker answers
// exactly like a linear scan of every tracked range, the semantics it
// replaced. No GPU: only Vulkan handle types are used.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <random>
#include <vector>

#include "dep_tracker.h"

using mlx::core::omarchy::DependencyTracker;
using mlx::core::omarchy::TrackedRange;
using mlx::core::omarchy::view_hazard_window;

namespace {

VkBuffer buf(uintptr_t id) {
  return reinterpret_cast<VkBuffer>(id);
}

TrackedRange range(uintptr_t b, VkDeviceSize off, VkDeviceSize end) {
  return TrackedRange{buf(b), off, end};
}

bool needs(
    const DependencyTracker& t,
    std::vector<TrackedRange> reads,
    std::vector<TrackedRange> writes) {
  return t.needs_barrier(reads, writes);
}

// The linear scan the tracker replaced, kept as the oracle.
struct LinearTracker {
  std::vector<TrackedRange> reads, writes;

  static bool overlaps(const TrackedRange& a, const TrackedRange& b) {
    return a.buffer == b.buffer && a.offset < b.end && b.offset < a.end;
  }

  bool needs_barrier(
      const std::vector<TrackedRange>& rd,
      const std::vector<TrackedRange>& wr) const {
    for (const auto& r : rd) {
      for (const auto& w : writes) {
        if (overlaps(r, w)) {
          return true;
        }
      }
    }
    for (const auto& w : wr) {
      for (const auto& tw : writes) {
        if (overlaps(w, tw)) {
          return true;
        }
      }
      for (const auto& tr : reads) {
        if (overlaps(w, tr)) {
          return true;
        }
      }
    }
    return false;
  }
};

} // namespace

TEST_CASE("an empty tracker needs no barrier") {
  DependencyTracker t;
  CHECK_FALSE(needs(t, {range(1, 0, 64)}, {range(2, 0, 64)}));
}

TEST_CASE("read after write on overlapping bytes needs a barrier") {
  DependencyTracker t;
  t.add_write(range(1, 0, 64));
  CHECK(needs(t, {range(1, 32, 48)}, {}));
  CHECK(needs(t, {range(1, 63, 100)}, {}));
}

TEST_CASE("write after write on overlapping bytes needs a barrier") {
  DependencyTracker t;
  t.add_write(range(1, 0, 64));
  CHECK(needs(t, {}, {range(1, 0, 8)}));
  CHECK(needs(t, {}, {range(1, 60, 70)}));
}

TEST_CASE("write after read on overlapping bytes needs a barrier") {
  DependencyTracker t;
  t.add_read(range(1, 0, 64));
  CHECK(needs(t, {}, {range(1, 16, 32)}));
}

TEST_CASE("read after read never needs a barrier") {
  DependencyTracker t;
  t.add_read(range(1, 0, 64));
  CHECK_FALSE(needs(t, {range(1, 0, 64)}, {}));
}

TEST_CASE("another buffer never conflicts") {
  DependencyTracker t;
  t.add_write(range(1, 0, 64));
  t.add_read(range(1, 0, 64));
  CHECK_FALSE(needs(t, {range(2, 0, 64)}, {range(3, 0, 64)}));
}

TEST_CASE("same buffer without overlap and touching ranges are disjoint") {
  DependencyTracker t;
  t.add_write(range(1, 64, 128));
  t.add_read(range(1, 256, 320));
  CHECK_FALSE(needs(t, {range(1, 0, 64)}, {}));
  CHECK_FALSE(needs(t, {range(1, 128, 192)}, {}));
  CHECK_FALSE(needs(t, {}, {range(1, 0, 64), range(1, 128, 256)}));
  CHECK_FALSE(needs(t, {}, {range(1, 320, 400)}));
}

TEST_CASE("a gap between merged ranges stays free") {
  DependencyTracker t;
  t.add_write(range(1, 0, 32));
  t.add_write(range(1, 16, 48));
  t.add_write(range(1, 96, 128));
  CHECK(needs(t, {range(1, 40, 44)}, {}));
  CHECK_FALSE(needs(t, {range(1, 48, 96)}, {}));
  CHECK(needs(t, {range(1, 47, 97)}, {}));
}

TEST_CASE("one wide range swallows the narrower ones it overlaps") {
  DependencyTracker t;
  t.add_write(range(1, 10, 20));
  t.add_write(range(1, 30, 40));
  t.add_write(range(1, 50, 60));
  t.add_write(range(1, 0, 100));
  CHECK(needs(t, {range(1, 70, 71)}, {}));
  CHECK_FALSE(needs(t, {range(1, 100, 120)}, {}));
}

TEST_CASE("clear forgets every range") {
  DependencyTracker t;
  t.add_write(range(1, 0, 64));
  t.add_read(range(2, 0, 64));
  t.clear();
  CHECK_FALSE(needs(t, {range(1, 0, 64)}, {range(2, 0, 64)}));
}

TEST_CASE("clear after more than 64 buffers releases the bucket table and the tracker still answers") {
  DependencyTracker t;
  for (uintptr_t b = 1; b <= 5000; ++b) {
    t.add_write(range(b, 0, 64));
  }
  REQUIRE(t.bucket_count() > 64);
  t.clear();
  CHECK(t.bucket_count() <= 64);
  CHECK_FALSE(needs(t, {range(1, 0, 64)}, {range(2, 0, 64)}));
  t.add_write(range(7, 8, 16));
  CHECK(needs(t, {range(7, 0, 64)}, {}));
  CHECK_FALSE(needs(t, {range(8, 0, 64)}, {}));
}

TEST_CASE("indexed answers match the linear scan across 200 buffers and saturated ranges") {
  std::mt19937_64 rng(20261113);
  constexpr VkDeviceSize kMax = UINT64_MAX;
  for (uint64_t nbuf : {3u, 200u}) {
    for (int round = 0; round < 300; ++round) {
      DependencyTracker t;
      LinearTracker oracle;
      auto pick = [&]() {
        auto b = static_cast<uintptr_t>(1 + rng() % nbuf);
        if (rng() % 16 == 0) {
          VkDeviceSize off = (rng() % 4 == 0) ? kMax : rng() % 200;
          return range(b, off, kMax);
        }
        VkDeviceSize off = rng() % 200;
        VkDeviceSize len = (rng() % 5 == 0) ? 0 : 1 + rng() % 40;
        return range(b, off, off + len);
      };
      int steps = 1 + static_cast<int>(rng() % 60);
      for (int s = 0; s < steps; ++s) {
        std::vector<TrackedRange> rd, wr;
        for (uint64_t i = rng() % 3; i > 0; --i) {
          rd.push_back(pick());
        }
        for (uint64_t i = rng() % 3; i > 0; --i) {
          wr.push_back(pick());
        }
        REQUIRE(t.needs_barrier(rd, wr) == oracle.needs_barrier(rd, wr));
        for (const auto& r : rd) {
          t.add_read(r);
          oracle.reads.push_back(r);
        }
        for (const auto& w : wr) {
          t.add_write(w);
          oracle.writes.push_back(w);
        }
        if (rng() % 20 == 0) {
          t.clear();
          oracle.reads.clear();
          oracle.writes.clear();
        }
      }
    }
  }
}

// M2D-6: the hazard window a view contributes to the tracker. Whole-
// allocation registration collided every disjoint view of one buffer
// (the ResNorm -> Word stall edge barriered on every occurrence); the
// window derives from shape*strides because data_size understates a
// strided footprint (the concat column slice).

TEST_CASE("view hazard window covers a contiguous view exactly") {
  int32_t shape[]{4, 8};
  int64_t strides[]{8, 1};
  auto [lo, hi] = view_hazard_window(0, 2, shape, strides, 2, 64);
  CHECK(lo == 0);
  CHECK(hi == 64);
  // A tail slice: elements 24..32 of a 32-element f16 buffer.
  int32_t s1[]{8};
  int64_t st1[]{1};
  auto [lo2, hi2] = view_hazard_window(48, 1, s1, st1, 2, 64);
  CHECK(lo2 == 48);
  CHECK(hi2 == 64);
}

TEST_CASE("view hazard window spans a strided footprint data_size understates") {
  // The concat column slice: a 4x2 view with parent strides (8,1) at
  // element offset 2. Logical size is 8 items (16 B) but the footprint
  // reaches element 2 + 3*8 + 1 = 27, so the window must end at 56.
  int32_t shape[]{4, 2};
  int64_t strides[]{8, 1};
  auto [lo, hi] = view_hazard_window(4, 2, shape, strides, 2, 64);
  CHECK(lo == 4);
  CHECK(hi == 56);
}

TEST_CASE("view hazard window follows negative strides below the view offset") {
  int32_t shape[]{4};
  int64_t strides[]{-1};
  auto [lo, hi] = view_hazard_window(6, 1, shape, strides, 1, 64);
  CHECK(lo == 3);
  CHECK(hi == 7);
}

TEST_CASE("view hazard window is one element for stride-0 broadcasts and scalars") {
  int32_t shape[]{8};
  int64_t strides[]{0};
  auto [lo, hi] = view_hazard_window(0, 1, shape, strides, 4, 64);
  CHECK(lo == 0);
  CHECK(hi == 4);
  auto [slo, shi] = view_hazard_window(8, 0, shape, strides, 4, 64);
  CHECK(slo == 8);
  CHECK(shi == 12);
}

TEST_CASE("view hazard window degenerates on empty axes and clamps to the allocation") {
  int32_t empty_shape[]{0, 4};
  int64_t strides[]{4, 1};
  auto [lo, hi] = view_hazard_window(0, 2, empty_shape, strides, 2, 64);
  CHECK(lo == 0);
  CHECK(hi == 0);
  // A reach past the allocation end clamps (the caller's fallback keeps
  // the bound range only when the window is fully degenerate).
  int32_t shape[]{8, 8};
  int64_t st[]{8, 1};
  auto [lo2, hi2] = view_hazard_window(0, 2, shape, st, 2, 100);
  CHECK(lo2 == 0);
  CHECK(hi2 == 100);
}

TEST_CASE("disjoint views of one buffer no longer hazard each other") {
  // Two views of one 256 B buffer: a res-norm output slice at [0,64)
  // and a staged read slice at [192,256). Whole-allocation ranges
  // collided on every batch head; the view windows are disjoint.
  DependencyTracker t;
  t.add_write(range(1, 0, 64));
  CHECK_FALSE(needs(t, {range(1, 192, 256)}, {}));
  // A view that does overlap the written slice still hazards.
  CHECK(needs(t, {range(1, 32, 96)}, {}));
}
