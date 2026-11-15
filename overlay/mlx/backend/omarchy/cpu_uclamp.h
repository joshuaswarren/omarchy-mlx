// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstdlib>

#if defined(__linux__) && defined(__aarch64__)
#include <unistd.h>
#endif

namespace mlx::core::omarchy {

// CPU frequency clamp for the threads that submit GPU work.
//
// Each GPU submit is a short CPU burst in a kernel worker that runs next to
// the submitting thread. Under schedutil the cluster idles at its lowest
// clock between bursts, so the burst runs slow and decode is CPU-latency
// bound. A utilization floor (uclamp.min = 1024) on the submitting thread
// lifts the cluster. Measured on an M6 (G18G), Qwen3.8-2B 4-bit decode with
// HK_G18_BATCH=32: 159 -> 121 ms/token, same token hash. sched_setattr with
// SCHED_FLAG_UTIL_CLAMP_MIN needs no privilege and applies to the calling
// thread; threads it creates inherit it. A floor or cap the caller already
// set on the thread is kept (read with sched_getattr first).
//
// MLX_OMARCHY_CPU_UCLAMP: unset or empty = on, "0" = off. Failure is ignored.
// Linux aarch64 only; every other build is a no-op.

// struct sched_attr (VER1, 56 bytes), spelled out so no kernel header is
// needed.
struct SchedAttr {
  uint32_t size;
  uint32_t sched_policy;
  uint64_t sched_flags;
  int32_t sched_nice;
  uint32_t sched_priority;
  uint64_t sched_runtime;
  uint64_t sched_deadline;
  uint64_t sched_period;
  uint32_t sched_util_min;
  uint32_t sched_util_max;
};
static_assert(sizeof(SchedAttr) == 56, "sched_attr VER1 is 56 bytes");

inline constexpr uint32_t kUclampMin = 1024;
inline constexpr uint64_t kSchedKeepPolicy = 0x08;
inline constexpr uint64_t kSchedKeepParams = 0x10;
inline constexpr uint64_t kSchedUtilClampMin = 0x20;
inline constexpr long kSchedSetattrAarch64 = 274;
inline constexpr long kSchedGetattrAarch64 = 275;

inline bool cpu_uclamp_enabled_for(const char* env) {
  return !(env != nullptr && env[0] == '0' && env[1] == '\0');
}

// Keeps the policy and the scheduling parameters, sets only the floor.
inline SchedAttr uclamp_min_attr(uint32_t util_min) {
  SchedAttr a{};
  a.size = sizeof(SchedAttr);
  a.sched_flags = kSchedKeepPolicy | kSchedKeepParams | kSchedUtilClampMin;
  a.sched_util_min = util_min;
  a.sched_util_max = 1024;
  return a;
}

// A floor or a cap the caller set on this thread is theirs: a nonzero
// util_min, or a util_max below the full range, is left alone.
inline bool caller_clamp_present(const SchedAttr& cur) {
  return cur.sched_util_min != 0 || cur.sched_util_max < 1024;
}

// |done| is the caller's once-per-thread flag; a failed attempt is not
// retried. |get| is sched_getattr(0, attr, size, 0) and |set| is
// sched_setattr(0, attr, 0); each returns 0 on success. A failed read does
// not block the write, which refuses harmlessly on a kernel without uclamp.
template <class Get, class Set>
bool apply_cpu_uclamp_once(Get&& get, Set&& set, const char* env, bool& done) {
  if (done) {
    return false;
  }
  done = true;
  if (!cpu_uclamp_enabled_for(env)) {
    return false;
  }
  SchedAttr cur{};
  cur.size = sizeof(SchedAttr);
  if (get(cur) == 0 && caller_clamp_present(cur)) {
    return false;
  }
  SchedAttr a = uclamp_min_attr(kUclampMin);
  return set(a) == 0;
}

// The calling thread, once. Cheap after the first call (a thread-local load).
inline void raise_cpu_clamp_this_thread() {
#if defined(__linux__) && defined(__aarch64__)
  thread_local bool done = false;
  if (done) {
    return;
  }
  apply_cpu_uclamp_once(
      [](SchedAttr& a) {
        return ::syscall(kSchedGetattrAarch64, 0, &a, sizeof(a), 0u);
      },
      [](SchedAttr& a) {
        return ::syscall(kSchedSetattrAarch64, 0, &a, 0u);
      },
      std::getenv("MLX_OMARCHY_CPU_UCLAMP"),
      done);
#endif
}

} // namespace mlx::core::omarchy
