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
