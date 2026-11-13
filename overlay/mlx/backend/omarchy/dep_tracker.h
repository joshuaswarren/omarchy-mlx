// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Dependency tracker of one open command-buffer batch: the buffer byte
// ranges the batch has read and written since the last full barrier.
// CommandEncoder asks it whether a new node must wait on earlier work.
//
// The question is "does any tracked range overlap this one", answered per
// buffer in O(log n) from a set of merged intervals, so a batch of N
// independent dispatches costs O(N log N) and not the O(N^2) of scanning
// every tracked range per dispatch.
#pragma once

#include <vulkan/vulkan.h>

#include <iterator>
#include <map>
#include <set>
#include <span>
#include <unordered_map>

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

// Union of byte ranges of one buffer for the overlap predicate above.
// Non-empty ranges that overlap merge into one interval, so the stored
// intervals are pairwise non-overlapping and sorted by both start and
// end; the last interval that starts before a query ends has the
// largest end of all of them and alone decides the query. Empty ranges
// are kept apart as points.
class RangeSet {
 public:
  bool overlaps(VkDeviceSize offset, VkDeviceSize end) const {
    auto next = intervals_.lower_bound(end);
    if (next != intervals_.begin() && std::prev(next)->second > offset) {
      return true;
    }
    if (offset < end) {
      auto p = points_.upper_bound(offset);
      return p != points_.end() && *p < end;
    }
    return false;
  }

  void add(VkDeviceSize offset, VkDeviceSize end) {
    if (offset >= end) {
      points_.insert(offset);
      return;
    }
    auto it = intervals_.lower_bound(offset);
    if (it != intervals_.begin() && std::prev(it)->second > offset) {
      --it;
    }
    while (it != intervals_.end() && it->first < end) {
      if (it->first < offset) {
        offset = it->first;
      }
      if (it->second > end) {
        end = it->second;
      }
      it = intervals_.erase(it);
    }
    intervals_.emplace_hint(it, offset, end);
  }

 private:
  std::map<VkDeviceSize, VkDeviceSize> intervals_;
  std::set<VkDeviceSize> points_;
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
    sets_[r.buffer].reads.add(r.offset, r.end);
  }

  void add_write(const TrackedRange& w) {
    sets_[w.buffer].writes.add(w.offset, w.end);
  }

  // Forget everything: a full barrier orders all earlier work. A table that
  // a long batch grew is released, so a later short batch does not pay to
  // reset its buckets.
  void clear() {
    if (sets_.bucket_count() > kKeepBuckets) {
      sets_ = decltype(sets_){};
    } else {
      sets_.clear();
    }
  }

  // Bucket count of the buffer table, for tests of the release rule.
  size_t bucket_count() const {
    return sets_.bucket_count();
  }

 private:
  static constexpr size_t kKeepBuckets = 64;

  struct Sets {
    RangeSet reads;
    RangeSet writes;
  };

  bool hit(const TrackedRange& q, RangeSet Sets::* which) const {
    auto it = sets_.find(q.buffer);
    return it != sets_.end() && (it->second.*which).overlaps(q.offset, q.end);
  }

  std::unordered_map<VkBuffer, Sets> sets_;
};

} // namespace mlx::core::omarchy
