// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlx::core::omarchy {

// Flat open-addressed scratch map for one scope pass. eval_impl builds a
// scope per pass, and unordered_map pays a node allocation per insert plus
// bucket rehashing each time; here entries are POD slots and "clear" is one
// generation bump, so a reused table allocates nothing per pass. prepare()
// sizes for 2x the expected distinct keys and shrinks back when the table
// is more than 4x oversized, so a rare large tape does not pin peak memory.
// A pass that outgrows the estimate rehashes at 75% load rather than
// walking a full table. GenT is the generation counter width: the default
// never wraps in a process; a narrow GenT makes the wrap path reachable
// in a test.
template <typename V, typename GenT = std::uint32_t>
class TapeIndex {
 public:
  void prepare(std::size_t expected) {
    std::size_t want = 16;
    while (want < expected * 2) {
      want <<= 1;
    }
    if (slots_.size() != want &&
        (slots_.size() < want || slots_.size() > want * 4)) {
      std::vector<Slot>(want).swap(slots_);
      gen_ = 0;
    }
    if (++gen_ == 0) {
      for (auto& s : slots_) {
        s.gen = 0;
      }
      gen_ = 1;
    }
    size_ = 0;
  }

  V* find(std::uintptr_t key) {
    return const_cast<V*>(const_cast<const TapeIndex*>(this)->find(key));
  }

  const V* find(std::uintptr_t key) const {
    if (slots_.empty()) {
      return nullptr;
    }
    std::size_t i = hash(key) & mask();
    while (slots_[i].gen == gen_ && slots_[i].key != key) {
      i = (i + 1) & mask();
    }
    return slots_[i].gen == gen_ ? &slots_[i].value : nullptr;
  }

  bool put_if_absent(std::uintptr_t key, V v) {
    maybe_grow();
    std::size_t i = hash(key) & mask();
    while (slots_[i].gen == gen_ && slots_[i].key != key) {
      i = (i + 1) & mask();
    }
    Slot& s = slots_[i];
    if (s.gen == gen_) {
      return false;
    }
    s.gen = gen_;
    s.key = key;
    s.value = v;
    ++size_;
    return true;
  }

  V& operator[](std::uintptr_t key) {
    maybe_grow();
    std::size_t i = hash(key) & mask();
    while (slots_[i].gen == gen_ && slots_[i].key != key) {
      i = (i + 1) & mask();
    }
    Slot& s = slots_[i];
    if (s.gen != gen_) {
      s.gen = gen_;
      s.key = key;
      s.value = V{};
      ++size_;
    }
    return s.value;
  }

  std::size_t count(std::uintptr_t key) const {
    return find(key) != nullptr ? 1 : 0;
  }

  std::size_t size() const {
    return size_;
  }

  std::size_t table_size() const {
    return slots_.size();
  }

 private:
  struct Slot {
    std::uintptr_t key = 0;
    GenT gen = 0;
    V value{};
  };

  std::size_t mask() const {
    return slots_.size() - 1;
  }

  static std::size_t hash(std::uintptr_t key) {
    std::size_t h = key * 0x9E3779B97F4A7C15ULL;
    return h ^ (h >> 33);
  }

  // The probe chain is a walk over live slots only, so it must stop before
  // the table fills: double and reinsert once a new key would push the
  // load past 75%. Entries keep their generation, so reinsertion is a
  // slot-for-slot copy.
  void maybe_grow() {
    if (slots_.empty()) {
      slots_.resize(16);
      gen_ = 1;
      return;
    }
    if ((size_ + 1) * 4 > slots_.size() * 3) {
      std::vector<Slot> bigger(slots_.size() * 2);
      const std::size_t m = bigger.size() - 1;
      for (const Slot& s : slots_) {
        if (s.gen != gen_) {
          continue;
        }
        std::size_t i = hash(s.key) & m;
        while (bigger[i].gen == gen_) {
          i = (i + 1) & m;
        }
        bigger[i].gen = gen_;
        bigger[i].key = s.key;
        bigger[i].value = s.value;
      }
      slots_.swap(bigger);
    }
  }

  std::vector<Slot> slots_;
  GenT gen_ = 0;
  std::size_t size_ = 0;
};

} // namespace mlx::core::omarchy
