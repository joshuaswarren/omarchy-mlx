// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

// Opt-in, low-overhead whole-step host timeline for the Omarchy Vulkan backend.
//
// Always compiled into release builds so any wheel can attribute host-side step
// latency without rebuilding or inserting Vulkan timestamp/barrier commands.
//
// Environment variables:
//   MLX_OMARCHY_STEP_TIMELINE=<path>  Enable recording and dump NDJSON at exit.
//                                     Unset or empty: disabled (one cached bool
//                                     check at each hook; zero allocation, zero
//                                     clock reads, zero atexit registration).
//   MLX_OMARCHY_STEP_TIMELINE_CAP=<n> Ring capacity in events (rounded up to a
//                                     power of 2, clamped to [1024, 4194304];
//                                     default 262144).

#include <atomic>
#include <array>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#if defined(__linux__)
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define MLX_OMARCHY_TIMELINE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define MLX_OMARCHY_TIMELINE_UNLIKELY(x) (x)
#endif

namespace mlx::core::omarchy::timeline {

inline constexpr uint64_t kDefaultCapacity = 1ull << 18; // 262,144 events (~10 MiB)
inline constexpr uint64_t kMinCapacity = 1ull << 10;     // 1,024 events
inline constexpr uint64_t kMaxCapacity = 1ull << 22;     // 4,194,304 events

enum class Kind : uint8_t {
  StepBegin = 0,
  StepEnd = 1,
  CmdBegin = 2,
  CmdRecord = 3,
  DescUpdate = 4,
  PipelineLookup = 5,
  QueueSubmit = 6,
  FenceOp = 7,
  Wait = 8,
  Readback = 9,
  Count = 10,
};

enum class SubKind : uint8_t {
  BatchOpen = 0,
  StepBeginMarker = 1,
  Finalize = 2,
  Synchronize = 3,
  StepEndMarker = 4,
  Begin = 5,
  SlotWait = 6,
  Dispatch = 7,
  WaveSchedule = 8,
  WaveEmit = 9,
  CmdEnd = 10,
  DescAlloc = 11,
  DescPoolCreate = 12,
  DescWrite = 13,
  PipelineHit = 14,
  PipelineMiss = 15,
  BindingAccess = 16,
  QueueLockWait = 17,
  StampBatch = 18,
  VkQueueSubmit = 19,
  HostSignalTimeline = 20,
  FenceAcquire = 21,
  FencePoll = 22,
  TimelineWait = 23,
  DrainThrough = 24,
  DispatcherCvWait = 25,
  EventWait = 26,
  SyncJoin = 27,
  FlushNoncoherent = 28,
  InvalidateNoncoherent = 29,
  Count = 30,
};

inline constexpr const char* kind_name(Kind kind) noexcept {
  switch (kind) {
    case Kind::StepBegin: return "step_begin";
    case Kind::StepEnd: return "step_end";
    case Kind::CmdBegin: return "cmd_begin";
    case Kind::CmdRecord: return "cmd_record";
    case Kind::DescUpdate: return "desc_update";
    case Kind::PipelineLookup: return "pipeline_lookup";
    case Kind::QueueSubmit: return "queue_submit";
    case Kind::FenceOp: return "fence_op";
    case Kind::Wait: return "wait";
    case Kind::Readback: return "readback";
    case Kind::Count: break;
  }
  return "unknown";
}

inline constexpr const char* subkind_name(SubKind sub) noexcept {
  switch (sub) {
    case SubKind::BatchOpen: return "batch_open";
    case SubKind::StepBeginMarker: return "marker";
    case SubKind::Finalize: return "finalize";
    case SubKind::Synchronize: return "synchronize";
    case SubKind::StepEndMarker: return "marker";
    case SubKind::Begin: return "begin";
    case SubKind::SlotWait: return "slot_wait";
    case SubKind::Dispatch: return "dispatch";
    case SubKind::WaveSchedule: return "wave_schedule";
    case SubKind::WaveEmit: return "wave_emit";
    case SubKind::CmdEnd: return "cmd_end";
    case SubKind::DescAlloc: return "desc_alloc";
    case SubKind::DescPoolCreate: return "desc_pool_create";
    case SubKind::DescWrite: return "desc_write";
    case SubKind::PipelineHit: return "pipeline_hit";
    case SubKind::PipelineMiss: return "pipeline_miss";
    case SubKind::BindingAccess: return "binding_access";
    case SubKind::QueueLockWait: return "queue_lock_wait";
    case SubKind::StampBatch: return "stamp_batch";
    case SubKind::VkQueueSubmit: return "vk_queue_submit";
    case SubKind::HostSignalTimeline: return "host_signal_timeline";
    case SubKind::FenceAcquire: return "fence_acquire";
    case SubKind::FencePoll: return "fence_poll";
    case SubKind::TimelineWait: return "timeline_wait";
    case SubKind::DrainThrough: return "drain_through";
    case SubKind::DispatcherCvWait: return "dispatcher_cv_wait";
    case SubKind::EventWait: return "event_wait";
    case SubKind::SyncJoin: return "sync_join";
    case SubKind::FlushNoncoherent: return "flush_noncoherent";
    case SubKind::InvalidateNoncoherent: return "invalidate_noncoherent";
    case SubKind::Count: break;
  }
  return "unknown";
}

inline constexpr Kind subkind_kind(SubKind sub) noexcept {
  switch (sub) {
    case SubKind::BatchOpen:
    case SubKind::StepBeginMarker:
      return Kind::StepBegin;
    case SubKind::Finalize:
    case SubKind::Synchronize:
    case SubKind::StepEndMarker:
      return Kind::StepEnd;
    case SubKind::Begin:
    case SubKind::SlotWait:
      return Kind::CmdBegin;
    case SubKind::Dispatch:
    case SubKind::WaveSchedule:
    case SubKind::WaveEmit:
    case SubKind::CmdEnd:
      return Kind::CmdRecord;
    case SubKind::DescAlloc:
    case SubKind::DescPoolCreate:
    case SubKind::DescWrite:
      return Kind::DescUpdate;
    case SubKind::PipelineHit:
    case SubKind::PipelineMiss:
    case SubKind::BindingAccess:
      return Kind::PipelineLookup;
    case SubKind::QueueLockWait:
    case SubKind::StampBatch:
    case SubKind::VkQueueSubmit:
    case SubKind::HostSignalTimeline:
      return Kind::QueueSubmit;
    case SubKind::FenceAcquire:
    case SubKind::FencePoll:
      return Kind::FenceOp;
    case SubKind::TimelineWait:
    case SubKind::DrainThrough:
    case SubKind::DispatcherCvWait:
    case SubKind::EventWait:
      return Kind::Wait;
    case SubKind::SyncJoin:
    case SubKind::FlushNoncoherent:
    case SubKind::InvalidateNoncoherent:
      return Kind::Readback;
    case SubKind::Count:
      break;
  }
  return Kind::StepBegin;
}

inline uint64_t monotonic_ns() noexcept {
#if defined(__linux__)
  struct timespec ts{};
  ::clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
      static_cast<uint64_t>(ts.tv_nsec);
#else
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
#endif
}

inline uint32_t current_tid() noexcept {
  thread_local uint32_t tid = []() -> uint32_t {
#if defined(__linux__)
    return static_cast<uint32_t>(::syscall(SYS_gettid));
#else
    return static_cast<uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id()));
#endif
  }();
  return tid;
}

inline uint64_t next_power_of_two(uint64_t n) noexcept {
  if (n <= 1) {
    return 1;
  }
  uint64_t p = 1;
  while (p < n && p < (uint64_t{1} << 62)) {
    p <<= 1;
  }
  return p;
}

inline uint64_t parse_capacity(const char* env_value) noexcept {
  if (env_value == nullptr || env_value[0] == '\0') {
    return kDefaultCapacity;
  }
  char* end = nullptr;
  long long parsed = std::strtoll(env_value, &end, 10);
  if (end == env_value || *end != '\0' || parsed <= 0) {
    return kDefaultCapacity;
  }
  uint64_t clamped = static_cast<uint64_t>(parsed);
  if (clamped < kMinCapacity) {
    clamped = kMinCapacity;
  } else if (clamped > kMaxCapacity) {
    clamped = kMaxCapacity;
  }
  return next_power_of_two(clamped);
}

struct Event {
  uint64_t t0_ns{0};
  uint64_t dur_ns{0};
  uint64_t detail{0};
  uint32_t tid{0};
  uint16_t stream_id{0};
  uint8_t kind{0};
  uint8_t subkind{0};
};
static_assert(sizeof(Event) == 32, "Timeline Event must be 32 bytes");

class StepTimelineRing {
 public:
  explicit StepTimelineRing(uint64_t requested_capacity) {
    if (requested_capacity == 0) {
      return;
    }
    capacity_ = next_power_of_two(requested_capacity);
    mask_ = capacity_ - 1;
    slots_ = std::make_unique<Slot[]>(static_cast<size_t>(capacity_));
    t0_origin_ns_ = monotonic_ns();
    gate_.store(0, std::memory_order_release);
  }

  StepTimelineRing(const StepTimelineRing&) = delete;
  StepTimelineRing& operator=(const StepTimelineRing&) = delete;

  bool enabled() const noexcept {
    return (gate_.load(std::memory_order_relaxed) & kGateStoppedBit) == 0;
  }

  // Stop admitting new records and wait for every in-flight producer inside
  // record() to finish before any dump reads slots_ or stats_.
  void stop() const noexcept {
    gate_.fetch_or(kGateStoppedBit, std::memory_order_acq_rel);
    while ((gate_.load(std::memory_order_acquire) & ~kGateStoppedBit) != 0) {
      std::this_thread::yield();
    }
  }

  uint64_t capacity() const noexcept {
    return capacity_;
  }

  uint64_t written() const noexcept {
    return write_index_.load(std::memory_order_acquire);
  }

  uint64_t dropped() const noexcept {
    const uint64_t w = written();
    return w > capacity_ ? w - capacity_ : 0;
  }

  uint64_t busy_dropped() const noexcept {
    return busy_dropped_.load(std::memory_order_acquire);
  }

  void record(
      Kind kind,
      SubKind subkind,
      uint64_t t0_ns,
      uint64_t dur_ns,
      uint16_t stream_id,
      uint64_t detail) noexcept {
    uint64_t g = gate_.load(std::memory_order_relaxed);
    for (;;) {
      if ((g & kGateStoppedBit) != 0) {
        return;
      }
      if (gate_.compare_exchange_weak(
              g, g + 1, std::memory_order_acquire, std::memory_order_relaxed)) {
        break;
      }
    }
    struct GateGuard {
      std::atomic<uint64_t>& gate;
      ~GateGuard() noexcept {
        gate.fetch_sub(1, std::memory_order_release);
      }
    } guard{gate_};

    const uint64_t idx = write_index_.fetch_add(1, std::memory_order_relaxed);
    const uint64_t target_seq = idx + 1;
    Slot& slot = slots_[static_cast<size_t>(idx & mask_)];
    uint64_t cur = slot.state.load(std::memory_order_acquire);
    if ((cur & kSlotBusyBit) != 0 || cur > target_seq ||
        !slot.state.compare_exchange_strong(
            cur,
            cur | kSlotBusyBit,
            std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      // Another writer owns this wrapped slot or a newer sequence already
      // published here. Drop the ring entry without touching slot.ev; whole-run
      // summary totals below still account for the event.
      busy_dropped_.fetch_add(1, std::memory_order_relaxed);
    } else {
      slot.ev.t0_ns = t0_ns;
      slot.ev.dur_ns = dur_ns;
      slot.ev.detail = detail;
      slot.ev.tid = current_tid();
      slot.ev.stream_id = stream_id;
      slot.ev.kind = static_cast<uint8_t>(kind);
      slot.ev.subkind = static_cast<uint8_t>(subkind);
      slot.state.store(target_seq, std::memory_order_release);
    }

    const size_t sub_idx = static_cast<size_t>(subkind);
    if (sub_idx < stats_.size()) {
      SubStats& st = stats_[sub_idx];
      st.count.fetch_add(1, std::memory_order_relaxed);
      st.total_ns.fetch_add(dur_ns, std::memory_order_relaxed);
      uint64_t prev_max = st.max_ns.load(std::memory_order_relaxed);
      while (dur_ns > prev_max &&
             !st.max_ns.compare_exchange_weak(
                 prev_max, dur_ns, std::memory_order_relaxed)) {
      }
    }
  }

  bool dump(std::FILE* out) const noexcept {
    if (out == nullptr || capacity_ == 0 || !slots_) {
      return false;
    }
    // Quiesce all in-flight producers so slots_, write_index_, busy_dropped_,
    // and stats_ are immutable for the entire snapshot.
    stop();
    const uint64_t total = write_index_.load(std::memory_order_acquire);
    const uint64_t start = total > capacity_ ? total - capacity_ : 0;
    const uint64_t drop = total > capacity_ ? total - capacity_ : 0;
    const uint64_t busy_drop = busy_dropped_.load(std::memory_order_acquire);
    bool ok = std::fprintf(
                  out,
                  "{\"k\":\"meta\",\"clock\":\"%s\",\"t0_ns\":%" PRIu64
                  ",\"capacity\":%" PRIu64 ",\"written\":%" PRIu64
                  ",\"dropped\":%" PRIu64 ",\"busy_dropped\":%" PRIu64 "}\n",
#if defined(__linux__)
                  "CLOCK_MONOTONIC",
#else
                  "steady_clock",
#endif
                  t0_origin_ns_,
                  capacity_,
                  total,
                  drop,
                  busy_drop) >= 0;
    for (uint64_t i = start; ok && i < total; ++i) {
      const Slot& slot = slots_[static_cast<size_t>(i & mask_)];
      if (slot.state.load(std::memory_order_acquire) != i + 1) {
        continue;
      }
      const Event& ev = slot.ev;
      if (std::fprintf(
              out,
              "{\"k\":\"e\",\"seq\":%" PRIu64 ",\"kind\":\"%s\",\"sub\":\"%s\""
              ",\"t0\":%" PRIu64 ",\"dur\":%" PRIu64 ",\"tid\":%" PRIu32
              ",\"st\":%" PRIu16 ",\"d\":%" PRIu64 "}\n",
              i,
              kind_name(static_cast<Kind>(ev.kind)),
              subkind_name(static_cast<SubKind>(ev.subkind)),
              ev.t0_ns,
              ev.dur_ns,
              ev.tid,
              ev.stream_id,
              ev.detail) < 0) {
        ok = false;
      }
    }
    if (ok && std::fprintf(out, "{\"k\":\"summary\",\"totals\":[") < 0) {
      ok = false;
    }
    bool first = true;
    for (size_t i = 0; ok && i < stats_.size(); ++i) {
      const uint64_t cnt = stats_[i].count.load(std::memory_order_relaxed);
      if (cnt == 0) {
        continue;
      }
      const auto sub = static_cast<SubKind>(i);
      if (std::fprintf(
              out,
              "%s{\"kind\":\"%s\",\"sub\":\"%s\",\"count\":%" PRIu64
              ",\"total_ns\":%" PRIu64 ",\"max_ns\":%" PRIu64 "}",
              first ? "" : ",",
              kind_name(subkind_kind(sub)),
              subkind_name(sub),
              cnt,
              stats_[i].total_ns.load(std::memory_order_relaxed),
              stats_[i].max_ns.load(std::memory_order_relaxed)) < 0) {
        ok = false;
      }
      first = false;
    }
    if (ok && std::fprintf(out, "]}\n") < 0) {
      ok = false;
    }
    if (std::fflush(out) != 0) {
      ok = false;
    }
    if (std::ferror(out) != 0) {
      ok = false;
    }
    return ok;
  }

  bool dump_to_file(const char* path) const noexcept {
    if (path == nullptr || path[0] == '\0' || capacity_ == 0 || !slots_) {
      return false;
    }
    std::FILE* f = std::fopen(path, "w");
    if (f == nullptr) {
      const int err = errno;
      std::fprintf(
          stderr,
          "[omarchy] cannot open MLX_OMARCHY_STEP_TIMELINE path %s: %s\n",
          path,
          err != 0 ? std::strerror(err) : "open failed");
      std::fflush(stderr);
      return false;
    }
    const bool dump_ok = dump(f);
    const int dump_err = dump_ok ? 0 : errno;
    const bool close_ok = (std::fclose(f) == 0);
    const int close_err = close_ok ? 0 : errno;
    if (!dump_ok || !close_ok) {
      const int err = !dump_ok ? dump_err : close_err;
      std::fprintf(
          stderr,
          "[omarchy] failed to write MLX_OMARCHY_STEP_TIMELINE path %s: %s\n",
          path,
          err != 0 ? std::strerror(err) : "I/O error");
      std::fflush(stderr);
      return false;
    }
    return true;
  }

 private:
  static constexpr uint64_t kGateStoppedBit = uint64_t{1} << 63;
  static constexpr uint64_t kSlotBusyBit = uint64_t{1} << 63;

  struct Slot {
    std::atomic<uint64_t> state{0};
    Event ev{};
  };

  struct SubStats {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> total_ns{0};
    std::atomic<uint64_t> max_ns{0};
  };

  uint64_t capacity_{0};
  uint64_t mask_{0};
  uint64_t t0_origin_ns_{0};
  std::unique_ptr<Slot[]> slots_;
  std::atomic<uint64_t> write_index_{0};
  std::atomic<uint64_t> busy_dropped_{0};
  mutable std::atomic<uint64_t> gate_{kGateStoppedBit};
  std::array<SubStats, static_cast<size_t>(SubKind::Count)> stats_{};
};

template <class ClockFn>
class BasicScopedEvent {
 public:
  BasicScopedEvent(
      StepTimelineRing& ring,
      Kind kind,
      SubKind subkind,
      uint16_t stream_id,
      uint64_t detail,
      ClockFn clock) noexcept
      : ring_(ring),
        kind_(kind),
        subkind_(subkind),
        stream_id_(stream_id),
        detail_(detail),
        clock_(clock),
        active_(ring.enabled()) {
    if (MLX_OMARCHY_TIMELINE_UNLIKELY(active_)) {
      t0_ns_ = clock_();
    }
  }

  ~BasicScopedEvent() noexcept {
    if (MLX_OMARCHY_TIMELINE_UNLIKELY(active_)) {
      const uint64_t t1_ns = clock_();
      ring_.record(
          kind_,
          subkind_,
          t0_ns_,
          t1_ns >= t0_ns_ ? t1_ns - t0_ns_ : 0,
          stream_id_,
          detail_);
    }
  }

  BasicScopedEvent(const BasicScopedEvent&) = delete;
  BasicScopedEvent& operator=(const BasicScopedEvent&) = delete;

  void set_subkind(SubKind subkind) noexcept {
    subkind_ = subkind;
  }

  void set_detail(uint64_t detail) noexcept {
    detail_ = detail;
  }

  void cancel() noexcept {
    active_ = false;
  }

 private:
  StepTimelineRing& ring_;
  Kind kind_;
  SubKind subkind_;
  uint16_t stream_id_;
  uint64_t detail_;
  ClockFn clock_;
  uint64_t t0_ns_{0};
  bool active_{false};
};

namespace detail {

struct GlobalState {
  std::string path;
  StepTimelineRing ring;
  std::atomic<bool> dumped{false};

  GlobalState(std::string p, uint64_t cap) : path(std::move(p)), ring(cap) {}
};

// Intentionally leaked on process exit so late background threads during
// static teardown never race with destructor ordering; dump runs in atexit.
inline GlobalState*& global_state_slot() noexcept {
  static GlobalState* ptr = nullptr;
  return ptr;
}

inline void atexit_dump() {
  GlobalState* state = global_state_slot();
  if (state == nullptr) {
    return;
  }
  if (state->dumped.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  state->ring.stop();
  state->ring.dump_to_file(state->path.c_str());
}

inline GlobalState* init_global_state() noexcept {
  const char* env_path = std::getenv("MLX_OMARCHY_STEP_TIMELINE");
  if (env_path == nullptr || env_path[0] == '\0') {
    return nullptr;
  }
  const uint64_t cap =
      parse_capacity(std::getenv("MLX_OMARCHY_STEP_TIMELINE_CAP"));
  auto* state = new GlobalState(env_path, cap);
  global_state_slot() = state;
  std::atexit(&atexit_dump);
  return state;
}

inline GlobalState* global_state() noexcept {
  static GlobalState* const state = init_global_state();
  return state;
}

} // namespace detail

inline bool enabled() noexcept {
  const detail::GlobalState* state = detail::global_state();
  return state != nullptr && state->ring.enabled();
}

// Startup probes (Device::probe_float_power in new_stream) dispatch and
// synchronize outside the step model, before any batch_open. Recording them
// would break the step-timeline ordering contract, so a probe wraps its body
// in SuppressTimeline and the hooks below drop their events.
inline thread_local int suppressed_depth = 0;

struct SuppressTimeline {
  SuppressTimeline() noexcept {
    ++suppressed_depth;
  }
  ~SuppressTimeline() noexcept {
    --suppressed_depth;
  }
  SuppressTimeline(const SuppressTimeline&) = delete;
  SuppressTimeline& operator=(const SuppressTimeline&) = delete;
};

inline void record(
    Kind kind,
    SubKind subkind,
    uint16_t stream_id = 0,
    uint64_t detail_val = 0,
    uint64_t dur_ns = 0) noexcept {
  detail::GlobalState* state = detail::global_state();
  if (MLX_OMARCHY_TIMELINE_UNLIKELY(
          state != nullptr && suppressed_depth == 0)) {
    state->ring.record(
        kind, subkind, monotonic_ns(), dur_ns, stream_id, detail_val);
  }
}

inline void record_span(
    Kind kind,
    SubKind subkind,
    uint64_t t0_ns,
    uint64_t t1_ns,
    uint16_t stream_id = 0,
    uint64_t detail_val = 0) noexcept {
  detail::GlobalState* state = detail::global_state();
  if (MLX_OMARCHY_TIMELINE_UNLIKELY(
          state != nullptr && suppressed_depth == 0)) {
    state->ring.record(
        kind,
        subkind,
        t0_ns,
        t1_ns >= t0_ns ? t1_ns - t0_ns : 0,
        stream_id,
        detail_val);
  }
}

class ScopedEvent {
 public:
  ScopedEvent(
      Kind kind,
      SubKind subkind,
      uint16_t stream_id = 0,
      uint64_t detail_val = 0) noexcept
      : kind_(kind),
        subkind_(subkind),
        stream_id_(stream_id),
        detail_(detail_val),
        state_(detail::global_state()) {
    if (MLX_OMARCHY_TIMELINE_UNLIKELY(
            state_ != nullptr && state_->ring.enabled() &&
            suppressed_depth == 0)) {
      t0_ns_ = monotonic_ns();
    } else {
      state_ = nullptr;
    }
  }

  ~ScopedEvent() noexcept {
    if (MLX_OMARCHY_TIMELINE_UNLIKELY(state_ != nullptr)) {
      const uint64_t t1_ns = monotonic_ns();
      state_->ring.record(
          kind_,
          subkind_,
          t0_ns_,
          t1_ns >= t0_ns_ ? t1_ns - t0_ns_ : 0,
          stream_id_,
          detail_);
    }
  }

  ScopedEvent(const ScopedEvent&) = delete;
  ScopedEvent& operator=(const ScopedEvent&) = delete;

  void set_subkind(SubKind subkind) noexcept {
    subkind_ = subkind;
  }

  void set_detail(uint64_t detail_val) noexcept {
    detail_ = detail_val;
  }

  void cancel() noexcept {
    state_ = nullptr;
  }

 private:
  Kind kind_;
  SubKind subkind_;
  uint16_t stream_id_;
  uint64_t detail_;
  detail::GlobalState* state_{nullptr};
  uint64_t t0_ns_{0};
};

} // namespace mlx::core::omarchy::timeline
