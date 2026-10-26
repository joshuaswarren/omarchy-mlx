// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

// NDJSON output sink of the diagnostic GPU profiler (gpu_profiler.h), kept
// free of Vulkan so CPU tests drive the real code. The encoder thread
// stages one record at a time and commits it only when the record ends
// with a newline and fits the process-lifetime byte cap, so reaching the
// cap never leaves a partial line. A short write or flush error (ENOSPC)
// stops output for good and may leave one. control() is the one entry
// point another thread may call; it and every FILE write share one mutex.

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

namespace mlx::core::omarchy::prof {

class ProfileSink {
 public:
  // Cap used when control() is never called.
  static constexpr uint64_t kDefaultCap = 1ull << 30;
  // Room kept for the {"k":"cap",...} marker; ordinary records stop short.
  static constexpr uint64_t kMarkerReserve = 96;

  bool is_open() const {
    return out_ != nullptr;
  }

  // Hooks capture new work only while enabled, below the cap, and while
  // the output has not failed.
  bool active() const {
    return enabled_.load(std::memory_order_relaxed) && !stopped();
  }

  // Nothing more will be written: cap reached or output failed.
  bool stopped() const {
    return exhausted_.load(std::memory_order_relaxed) ||
        failed_.load(std::memory_order_relaxed);
  }

  bool exhausted() const {
    return exhausted_.load(std::memory_order_relaxed);
  }

  bool failed() const {
    return failed_.load(std::memory_order_relaxed);
  }

  uint64_t bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return bytes_;
  }

  // Encoder thread only.
  void open(FILE* out) {
    std::lock_guard<std::mutex> lock(mu_);
    out_ = out;
  }

  // Encoder thread only; the destructor of the owner calls it at exit.
  void close() {
    std::lock_guard<std::mutex> lock(mu_);
    if (out_ != nullptr) {
      io_ok_locked(std::fflush(out_) == 0);
      std::fclose(out_);
      out_ = nullptr;
    }
  }

  // 1 accepted, 0 output failed (a short write or flush error stopped
  // capture for good), -1 invalid (enabled not 0/1, max_bytes 0, or
  // max_bytes below bytes already written; nothing changes), -2 cap
  // exhausted (the enable flag still updates, capture stays off). Pausing
  // flushes committed records; a staged partial record never reaches the
  // file through the cap path (a failed write may leave a partial line).
  int control(int enabled, uint64_t max_bytes) {
    if ((enabled != 0 && enabled != 1) || max_bytes == 0) {
      return -1;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (failed_.load(std::memory_order_relaxed)) {
      return 0;
    }
    if (!exhausted_.load(std::memory_order_relaxed)) {
      if (max_bytes < bytes_) {
        return -1;
      }
      cap_ = max_bytes;
    }
    enabled_.store(enabled == 1, std::memory_order_relaxed);
    if (enabled == 0 && out_ != nullptr &&
        !io_ok_locked(std::fflush(out_) == 0)) {
      return 0;
    }
    return exhausted_.load(std::memory_order_relaxed) ? -2 : 1;
  }

  // Encoder thread only. Appends to the staged record and commits it when
  // fmt ends with a newline.
  void emitf(const char* fmt, ...) {
    if (out_ == nullptr || stopped()) {
      return;
    }
    if (!overflow_) {
      va_list args;
      va_start(args, fmt);
      va_list sizing;
      va_copy(sizing, args);
      int n = std::vsnprintf(nullptr, 0, fmt, sizing);
      va_end(sizing);
      // Staging never grows past what the cap could still accept.
      if (n < 0 || rec_.size() + static_cast<uint64_t>(n) > room()) {
        overflow_ = true;
        rec_.clear();
      } else {
        size_t at = rec_.size();
        rec_.resize(at + n + 1);
        std::vsnprintf(rec_.data() + at, n + 1, fmt, args);
        rec_.resize(at + n);
      }
      va_end(args);
    }
    size_t len = std::strlen(fmt);
    if (len > 0 && fmt[len - 1] == '\n') {
      commit();
    }
  }

 private:
  uint64_t room() {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t used = bytes_ + kMarkerReserve;
    return used < cap_ ? cap_ - used : 0;
  }

  // A short write or stream error leaves the file unreliable: latch
  // failed_ and stop for good, never throwing into the workload.
  bool io_ok_locked(bool ok) {
    if (ok && !std::ferror(out_)) {
      return true;
    }
    int err = errno;
    if (!failed_.exchange(true, std::memory_order_relaxed)) {
      std::fprintf(
          stderr,
          "[prof] MLX_OMARCHY_GPU_PROFILE write failed after %" PRIu64
          " bytes (%s); capture stopped\n",
          bytes_,
          std::strerror(err));
    }
    return false;
  }

  void commit() {
    std::lock_guard<std::mutex> lock(mu_);
    // control() may latch an output failure while this record is staged.
    bool failed = failed_.load(std::memory_order_relaxed);
    if (!failed && !overflow_ &&
        bytes_ + rec_.size() + kMarkerReserve <= cap_) {
      size_t written = std::fwrite(rec_.data(), 1, rec_.size(), out_);
      bytes_ += written;
      io_ok_locked(written == rec_.size());
    } else if (!failed) {
      exhausted_.store(true, std::memory_order_relaxed);
      char marker[kMarkerReserve];
      int n = std::snprintf(
          marker,
          sizeof(marker),
          "{\"k\":\"cap\",\"bytes\":%" PRIu64 ",\"cap\":%" PRIu64 "}\n",
          bytes_,
          cap_);
      if (n > 0 && bytes_ + static_cast<uint64_t>(n) <= cap_) {
        size_t written = std::fwrite(marker, 1, n, out_);
        bytes_ += written;
        io_ok_locked(written == static_cast<size_t>(n));
      }
      io_ok_locked(std::fflush(out_) == 0);
      std::fprintf(
          stderr,
          "[prof] MLX_OMARCHY_GPU_PROFILE byte cap %" PRIu64
          " reached at %" PRIu64 " bytes; capture stopped\n",
          cap_,
          bytes_);
    }
    rec_.clear();
    overflow_ = false;
  }

  mutable std::mutex mu_;
  FILE* out_{nullptr};
  uint64_t cap_{kDefaultCap};
  uint64_t bytes_{0};
  std::atomic<bool> enabled_{true};
  std::atomic<bool> exhausted_{false};
  std::atomic<bool> failed_{false};
  // Encoder-thread staging for the record being built.
  std::string rec_;
  bool overflow_{false};
};

} // namespace mlx::core::omarchy::prof
