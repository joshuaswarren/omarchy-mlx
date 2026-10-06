// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/cpu_pd_hold.h"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

#include "mlx/backend/omarchy/device.h"

namespace mlx::core::omarchy {

namespace {

constexpr int64_t kIdleGraceMs = 300;
constexpr int64_t kTickMs = 100;
constexpr const char* kPmQosPath = "/dev/cpu_dma_latency";

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct State {
  std::atomic<int> mode{-1}; // -1 undecided, 0 off, 1 on
  std::atomic<uint64_t> reserved{0};
  std::atomic<int64_t> last_submit_ns{0};
  std::mutex mu;
  std::condition_variable cv;
  int fd{-1};
  Device* device{nullptr};
  std::thread watchdog;
  bool stop{false};
  bool forced{false};
};

// Leaked on purpose: the watchdog and the device teardown order must not
// depend on static destruction order.
State& state() {
  static State* s = new State();
  return *s;
}

// Caller holds s.mu.
bool open_request_locked(State& s) {
  if (s.fd >= 0) {
    return true;
  }
  int fd = ::open(kPmQosPath, O_WRONLY | O_CLOEXEC);
  if (fd < 0) {
    if (s.forced) {
      std::fprintf(
          stderr,
          "[omarchy] CPU PD hold: cannot open %s (%s); hold disabled\n",
          kPmQosPath,
          std::strerror(errno));
    }
    return false;
  }
  int32_t zero = 0;
  if (::write(fd, &zero, sizeof(zero)) != static_cast<ssize_t>(sizeof(zero))) {
    ::close(fd);
    if (s.forced) {
      std::fprintf(
          stderr,
          "[omarchy] CPU PD hold: write to %s failed (%s); hold disabled\n",
          kPmQosPath,
          std::strerror(errno));
    }
    return false;
  }
  s.fd = fd;
  return true;
}

// Caller holds s.mu.
void close_request_locked(State& s) {
  if (s.fd >= 0) {
    ::close(s.fd);
    s.fd = -1;
  }
}

void watchdog_main(State* sp) {
  State& s = *sp;
  std::unique_lock<std::mutex> lk(s.mu);
  while (!s.stop) {
    s.cv.wait_for(lk, std::chrono::milliseconds(kTickMs));
    if (s.stop) {
      break;
    }
    if (s.fd < 0 || s.device == nullptr) {
      continue;
    }
    int64_t idle_ms = (now_ns() - s.last_submit_ns.load(std::memory_order_relaxed)) / 1000000;
    if (idle_ms < kIdleGraceMs) {
      continue;
    }
    // Everything reserved so far must have drained before the hold drops:
    // a long kernel keeps the GPU busy with no new submission.
    Device* dev = s.device;
    uint64_t reserved = s.reserved.load(std::memory_order_relaxed);
    lk.unlock();
    uint64_t drained = dev->completions().drained_value();
    lk.lock();
    if (s.stop || s.device != dev) {
      continue;
    }
    int64_t idle_now_ms = (now_ns() - s.last_submit_ns.load(std::memory_order_relaxed)) / 1000000;
    if (drained >= reserved && idle_now_ms >= kIdleGraceMs &&
        s.reserved.load(std::memory_order_relaxed) == reserved) {
      close_request_locked(s);
    }
  }
  close_request_locked(s);
}

} // namespace

bool cpu_pd_hold_enabled_for(const char* env, const std::string& device_name) {
  if (env != nullptr && env[0] != '\0') {
    return !(env[0] == '0' && env[1] == '\0');
  }
  return device_name.find("G13") != std::string::npos &&
      device_name.find("G13C") == std::string::npos;
}

void cpu_pd_hold_note_submit(Device& device, uint64_t completion_value) {
  State& s = state();
  int mode = s.mode.load(std::memory_order_relaxed);
  if (mode == 0) {
    return;
  }
  if (mode < 0) {
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.mode.load(std::memory_order_relaxed) < 0) {
      const char* env = std::getenv("MLX_OMARCHY_CPU_PD_HOLD");
      s.forced = env != nullptr && env[0] != '\0' && !(env[0] == '0' && env[1] == '\0');
      bool on = cpu_pd_hold_enabled_for(env, device.hardware_capabilities().device_name);
      s.mode.store(on ? 1 : 0, std::memory_order_relaxed);
      if (on) {
        s.device = &device;
        s.watchdog = std::thread(watchdog_main, &s);
      }
    }
    if (s.mode.load(std::memory_order_relaxed) == 0) {
      return;
    }
  }
  s.reserved.store(completion_value, std::memory_order_relaxed);
  s.last_submit_ns.store(now_ns(), std::memory_order_relaxed);
  std::lock_guard<std::mutex> lk(s.mu);
  if (s.fd < 0 && !open_request_locked(s)) {
    // No access (udev rule missing) or write refused: stop trying.
    s.mode.store(0, std::memory_order_relaxed);
    s.stop = true;
    s.cv.notify_all();
  }
}

void cpu_pd_hold_device_destroyed(Device& device) {
  State& s = state();
  std::thread t;
  {
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.device != &device) {
      return;
    }
    s.stop = true;
    s.device = nullptr;
    s.mode.store(0, std::memory_order_relaxed);
    s.cv.notify_all();
    t = std::move(s.watchdog);
  }
  if (t.joinable()) {
    t.join();
  }
  std::lock_guard<std::mutex> lk(s.mu);
  close_request_locked(s);
}

} // namespace mlx::core::omarchy
