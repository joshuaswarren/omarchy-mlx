// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Dependency tracker of one open command-buffer batch: the buffer byte
// ranges the batch has read and written since the last full barrier.
// CommandEncoder asks it whether a new node must wait on earlier work.
//
// The question is "does any tracked range overlap this one", answered per
// buffer in O(log n) from a sorted vector of merged intervals, so a batch
// of N independent dispatches costs O(N log n) and not the O(N^2) of
// scanning every tracked range per dispatch.
//
// Buffers index by SLOT, not by hash node: an open-addressed table assigns
// each VkBuffer a dense slot on first sight and lookups probe one flat
// array, so neither the probe nor the per-buffer entry touches the heap
// (the previous unordered_map did both per range). The tracked structures
// keep their storage across clear() - the per-batch arena - so steady-state
// decode records hazards with no allocation at all.
#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <span>
#include <utility>
#include <vector>

namespace mlx::core::omarchy {

// One tracked buffer byte-range (start saturates at the address space; see
// tracked_range_end). Two ranges overlap when a.offset < b.end and
// b.offset < a.end, so touching ranges do not and an empty range overlaps
// only a range that strictly contains its offset.
struct TrackedRange {
  VkBuffer buffer;
  VkDeviceSize offset;
  VkDeviceSize end;
};

// The byte window [lo, hi) of one allocation that a view can name: the
// view's storage offset plus the element span its shape and strides
// describe. data_size is the logical item count and understates a strided
// footprint (a concat column slice over a wider parent), so the span
// derives from strides: first = sum(min(0, (shape-1)*stride)), last =
// sum(max(0, (shape-1)*stride)) items. Returns {0, 0} for a degenerate or
// empty view; the caller keeps the bound descriptor range as the
// conservative tracked window then.
inline std::pair<VkDeviceSize, VkDeviceSize> view_hazard_window(
    int64_t offset_bytes,
    int ndim,
    const int32_t* shape,
    const int64_t* strides,
    int64_t itemsize,
    VkDeviceSize allocation_bytes) {
  int64_t first = 0;
  int64_t last = 0;
  for (int ax = 0; ax < ndim; ++ax) {
    if (shape[ax] <= 0) {
      return {0, 0};
    }
    const int64_t span = static_cast<int64_t>(shape[ax] - 1) * strides[ax];
    first += std::min<int64_t>(0, span);
    last += std::max<int64_t>(0, span);
  }
  const int64_t lo = offset_bytes + first * itemsize;
  const int64_t hi = offset_bytes + (last + 1) * itemsize;
  const VkDeviceSize begin =
      static_cast<VkDeviceSize>(std::max<int64_t>(0, lo));
  const VkDeviceSize end = static_cast<VkDeviceSize>(
      std::min<int64_t>(static_cast<int64_t>(allocation_bytes), hi));
  return end > begin ? std::pair{begin, end}
                     : std::pair<VkDeviceSize, VkDeviceSize>{0, 0};
}

// Open-addressed buffer -> dense slot index. Insert-only, linear probe,
// bulk clear: the table is a flat array of entries (no per-entry node, no
// bucket walk through pointers), so find() costs one cache line walk. Slot
// numbers count up 0..count-1 so a parallel vector holds the per-buffer
// value. VK_NULL_HANDLE is the empty key; a null buffer therefore cannot
// be tracked (a dispatch always binds real buffers).
class BufferSlots {
 public:
  // The slot for |buffer|, assigning the next free index on first sight.
  uint32_t slot_of(VkBuffer buffer) {
    if (count_ * 4 >= table_.size() * 3) {
      grow();
    }
    size_t i = mix(buffer) & (table_.size() - 1);
    while (table_[i].buffer != VK_NULL_HANDLE && table_[i].buffer != buffer) {
      i = (i + 1) & (table_.size() - 1);
    }
    if (table_[i].buffer == buffer) {
      return table_[i].slot;
    }
    table_[i] = {buffer, count_};
    return count_++;
  }

  // The slot for |buffer|, or UINT32_MAX when the buffer is not tracked.
  uint32_t find(VkBuffer buffer) const {
    size_t i = mix(buffer) & (table_.size() - 1);
    while (table_[i].buffer != VK_NULL_HANDLE) {
      if (table_[i].buffer == buffer) {
        return table_[i].slot;
      }
      i = (i + 1) & (table_.size() - 1);
    }
    return UINT32_MAX;
  }

  // Forget every key. The table keeps its capacity; callers clear the
  // parallel value array themselves (its storage also persists).
  void clear() {
    for (auto& e : table_) {
      e.buffer = VK_NULL_HANDLE;
    }
    count_ = 0;
  }

  // Table slot count (capacity), for tests of the release rule.
  size_t capacity() const {
    return table_.size();
  }

  uint32_t count() const {
    return count_;
  }

 private:
  struct Entry {
    VkBuffer buffer{VK_NULL_HANDLE};
    uint32_t slot{0};
  };

  static size_t mix(VkBuffer b) {
    uintptr_t v = reinterpret_cast<uintptr_t>(b);
    v ^= v >> 33;
    v *= static_cast<uintptr_t>(0xff51afd7ed558ccdULL);
    v ^= v >> 33;
    return static_cast<size_t>(v);
  }

  void grow() {
    std::vector<Entry> next(table_.size() * 2);
    const size_t mask = next.size() - 1;
    for (auto& e : table_) {
      if (e.buffer == VK_NULL_HANDLE) {
        continue;
      }
      size_t i = mix(e.buffer) & mask;
      while (next[i].buffer != VK_NULL_HANDLE) {
        i = (i + 1) & mask;
      }
      next[i] = e;
    }
    table_.swap(next);
  }

  std::vector<Entry> table_{kInitialSlots};
  uint32_t count_{0};
  static constexpr size_t kInitialSlots = 64;
};

// Union of byte ranges of one buffer for the overlap predicate above.
// Non-empty ranges that overlap merge into one interval, so the stored
// intervals are pairwise non-overlapping and sorted by both start and
// end; the last interval that starts before a query ends has the largest
// end of all of them and alone decides the query. Empty ranges are kept
// apart as points. Both lists are flat sorted vectors (the previous
// std::map/std::set paid a tree-node allocation per inserted range).
class RangeSet {
 public:
  bool overlaps(VkDeviceSize offset, VkDeviceSize end) const {
    auto next = std::lower_bound(
        intervals_.begin(),
        intervals_.end(),
        end,
        [](const Interval& iv, VkDeviceSize e) { return iv.first < e; });
    if (next != intervals_.begin() && std::prev(next)->second > offset) {
      return true;
    }
    if (offset < end) {
      auto p = std::upper_bound(points_.begin(), points_.end(), offset);
      return p != points_.end() && *p < end;
    }
    return false;
  }

  void add(VkDeviceSize offset, VkDeviceSize end) {
    if (offset >= end) {
      auto p = std::lower_bound(points_.begin(), points_.end(), offset);
      if (p == points_.end() || *p != offset) {
        points_.insert(p, offset);
      }
      return;
    }
    auto it = std::lower_bound(
        intervals_.begin(),
        intervals_.end(),
        offset,
        [](const Interval& iv, VkDeviceSize o) { return iv.first < o; });
    size_t pos = static_cast<size_t>(it - intervals_.begin());
    if (pos > 0 && intervals_[pos - 1].second > offset) {
      --pos;
    }
    size_t erase_end = pos;
    while (erase_end < intervals_.size() &&
           intervals_[erase_end].first < end) {
      if (intervals_[erase_end].first < offset) {
        offset = intervals_[erase_end].first;
      }
      if (intervals_[erase_end].second > end) {
        end = intervals_[erase_end].second;
      }
      ++erase_end;
    }
    intervals_.erase(
        intervals_.begin() + pos, intervals_.begin() + erase_end);
    intervals_.insert(intervals_.begin() + pos, {offset, end});
  }

  // Forget the ranges. The vectors keep their storage.
  void clear() {
    intervals_.clear();
    points_.clear();
  }

 private:
  using Interval = std::pair<VkDeviceSize, VkDeviceSize>;
  std::vector<Interval> intervals_;
  std::vector<VkDeviceSize> points_;
};

class DependencyTracker {
 public:
  // True when a node that reads `reads` and writes `writes` must wait on
  // tracked work: read after write, write after write, write after read.
  // Read after read never does.
  bool needs_barrier(
      std::span<const TrackedRange> reads,
      std::span<const TrackedRange> writes) const {
    for (const auto& r : reads) {
      if (hit(r, &Sets::writes)) {
        return true;
      }
    }
    for (const auto& w : writes) {
      if (hit(w, &Sets::writes) || hit(w, &Sets::reads)) {
        return true;
      }
    }
    return false;
  }

  void add_read(const TrackedRange& r) {
    uint32_t s = slots_.slot_of(r.buffer);
    if (s == sets_.size()) {
      sets_.emplace_back();
    }
    sets_[s].reads.add(r.offset, r.end);
    ++tracked_accesses_;
  }

  void add_write(const TrackedRange& w) {
    uint32_t s = slots_.slot_of(w.buffer);
    if (s == sets_.size()) {
      sets_.emplace_back();
    }
    sets_[s].writes.add(w.offset, w.end);
    ++tracked_accesses_;
  }

  // Forget everything: a full barrier orders all earlier work. A table
  // that a long batch grew is released, so a later short batch does not
  // pay to reset it; below the bound the table and per-buffer lists keep
  // their storage and the next segment allocates nothing.
  void clear() {
    if (slots_.capacity() > kKeepBuckets) {
      slots_ = decltype(slots_){};
      sets_ = decltype(sets_){};
    } else {
      slots_.clear();
      for (auto& s : sets_) {
        s.reads.clear();
        s.writes.clear();
      }
    }
    tracked_accesses_ = 0;
  }

  // Slot-table capacity, for tests of the release rule.
  size_t bucket_count() const {
    return slots_.capacity();
  }

  // Range accesses recorded since the last clear(). The submission
  // boundary carries the table forward, so this bounds how much carried
  // state the next batch head scans before the fallback barrier fires.
  uint32_t tracked_accesses() const {
    return tracked_accesses_;
  }

 private:
  static constexpr size_t kKeepBuckets = 64;

  struct Sets {
    RangeSet reads;
    RangeSet writes;
  };

  bool hit(const TrackedRange& q, RangeSet Sets::* which) const {
    uint32_t s = slots_.find(q.buffer);
    return s != UINT32_MAX && (sets_[s].*which).overlaps(q.offset, q.end);
  }

  BufferSlots slots_;
  std::vector<Sets> sets_;
  uint32_t tracked_accesses_{0};
};

// When the carried tracker recorded this many accesses, the next batch
// head emits a real boundary barrier and the table clears: the amortized
// fallback bounds carried-state memory and scan cost, correctness
// unchanged. Phase 2 of docs/research/m2d6-boundary-barrier-design.md
// (the M2D-6 submission-boundary carry).
inline constexpr uint32_t kBoundaryTrackerAccessBudget = 4096;

enum class BoundaryDecision : uint8_t { Skip, Barrier };

// The submission-boundary decision on the carried tracker: an overlap
// with surviving ranges, or the access-budget fallback, emits a real
// barrier; provably disjoint head ranges skip it and ride the
// submission's in-order wait. budget_forced is the encoder's flag set
// when tracked_accesses() crossed kBoundaryTrackerAccessBudget at the
// boundary.
inline BoundaryDecision boundary_decision(
    const DependencyTracker& carried,
    std::span<const TrackedRange> reads,
    std::span<const TrackedRange> writes,
    bool budget_forced) {
  return (budget_forced || carried.needs_barrier(reads, writes))
             ? BoundaryDecision::Barrier
             : BoundaryDecision::Skip;
}

// One buffered node's access sets for wave scheduling: the byte ranges a
// node reads and writes (SPIR-V reflected readonly/writeonly split).
// Moved here from encoder.h: the schedule indexes these ranges with the
// same slot machinery as the barrier tracker.
struct WaveNode {
  std::span<const TrackedRange> reads;
  std::span<const TrackedRange> writes;
};

// Per-buffer exact-range -> deepest wave index, for wave_levels: earlier
// accesses of one buffer keyed by their exact byte range so a node can
// find the deepest hazard source in O(log n + overlaps). Flat sorted
// vector for the same reason as RangeSet.
struct WaveRangeLevels {
  int64_t max_overlap(const TrackedRange& q) const {
    int64_t best = -1;
    auto it = std::lower_bound(
        ranges.begin(),
        ranges.end(),
        std::pair{q.end, VkDeviceSize{0}},
        [](const Entry& e, const Key& k) { return e.first < k; });
    for (; it != ranges.begin();) {
      --it;
      const auto [offset, end] = it->first;
      if (offset <= q.offset && q.offset - offset >= max_len) {
        break;
      }
      if (q.offset < end) {
        best = std::max<int64_t>(best, it->second);
      }
    }
    return best;
  }

  void add(const TrackedRange& r, uint32_t level) {
    Key key{r.offset, r.end};
    auto it = std::lower_bound(
        ranges.begin(),
        ranges.end(),
        key,
        [](const Entry& e, const Key& k) { return e.first < k; });
    if (it != ranges.end() && it->first == key) {
      it->second = std::max(it->second, level);
    } else {
      ranges.insert(it, {key, level});
    }
    max_len = std::max(max_len, r.end - r.offset);
  }

  void clear() {
    ranges.clear();
    max_len = 0;
  }

  using Key = std::pair<VkDeviceSize, VkDeviceSize>;
  using Entry = std::pair<Key, uint32_t>;
  std::vector<Entry> ranges;
  VkDeviceSize max_len = 0;
};

// Reusable wave-index state: the buffer->slot table plus the writer and
// reader indexes, one WaveRangeLevels per slot. Held by the encoder so a
// submission's scheduling pass allocates nothing after warmup.
struct WaveIndex {
  BufferSlots slots;
  std::vector<WaveRangeLevels> writers;
  std::vector<WaveRangeLevels> readers;

  void reset() {
    slots.clear();
    for (auto& w : writers) {
      w.clear();
    }
    for (auto& r : readers) {
      r.clear();
    }
  }
};

// Greedy earliest-wave schedule into |levels| (resized to nodes.size()):
// level[i] is the smallest wave index such that every earlier tape node
// that node i hazards sits in a strictly smaller wave. Reuses |index|
// storage; callers that cannot share state use the allocating wave_levels
// wrapper in encoder.h.
void wave_levels_into(
    std::span<const WaveNode> nodes,
    WaveIndex& index,
    std::vector<uint32_t>& levels);

} // namespace mlx::core::omarchy
