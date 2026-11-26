// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
#pragma once

// Bounded ANE worker boundary (plan sections 22-27).
//
// A worker run is a supervised child process: the child loads the
// validated bundle programs onto a device, executes the dispatch plan,
// reads results, releases the programs, and reports through a pipe. The
// parent enforces a wall-clock deadline; a deadline expiry or an
// abnormal child death leaves the device completion state uncertain, so
// the worker enters quarantine and refuses every later submission.
//
// This is not an inference server (section 25): one run() is one bounded
// execution of one bundle, nothing persists between runs except the
// quarantine state, and no listening socket or external request surface
// exists.

#include "mlx/backend/omarchy/ane/bundle.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx::core::omarchy::ane {

// Device seam. Bindings and validated programs are passed through so a
// backend can map tensors onto device channels; the generic worker never
// interprets channel indices itself.
struct AneDevice {
  virtual ~AneDevice() = default;
  virtual std::string describe() const = 0;
  virtual void load(const AneValidatedProgram& program) = 0;
  virtual void send(
      size_t manifest_index,
      uint32_t channel,
      const AneProgramBinding& binding,
      const uint8_t* data,
      size_t size) = 0;
  virtual void exec(size_t manifest_index) = 0;
  virtual void read(
      size_t manifest_index,
      uint32_t channel,
      const AneProgramBinding& binding,
      uint8_t* out,
      size_t size) = 0;
  virtual void release() = 0;
};

struct AneDeviceError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class AneWorkerStatus {
  Completed,
  DeviceFailed,      // clean child failure with a named reason
  DeadlineExceeded,  // killed at the deadline; device state uncertain
  WorkerDied,        // signal death or mid-protocol exit; uncertain
  QuarantinedRefused // a prior run quarantined this worker
};

struct AneWorkerReport {
  AneWorkerStatus status{AneWorkerStatus::Completed};
  int iterations{0};
  int released_programs{0};
  std::chrono::milliseconds elapsed{0};
  std::string detail;
  // Internal: raw child report bytes carried out of supervise() so run()
  // can extract output payloads. Empty in every report handed to callers.
  std::string detail_extra;
};

struct AneWorkerOptions {
  std::chrono::milliseconds deadline{std::chrono::milliseconds(2000)};
  int iterations{1};
};

// MLX_API: the standalone fd-protocol worker exe links this class out
// of the shared libmlx; without the annotation the hidden-visibility
// preset keeps its symbols private and the exe fails to link.
class MLX_API AneWorker {
 public:
  using DeviceFactory = std::function<std::unique_ptr<AneDevice>()>;
  using Buffer = std::vector<uint8_t>;

  // Throws std::invalid_argument on a non-positive deadline or
  // iteration count: an unbounded worker is a contract violation, not a
  // configuration.
  AneWorker(DeviceFactory factory, AneWorkerOptions options);

  // A resident child never outlives its worker: an unclosed session is
  // killed and reaped here rather than left holding the device.
  ~AneWorker();

  // Executes one bounded run. `inputs` maps manifest input tensor names
  // to payload bytes; on Completed, `outputs` receives the manifest
  // output tensors of the final iteration.
  AneWorkerReport run(
      const AneBundle& bundle,
      const std::map<std::string, Buffer>& inputs,
      std::map<std::string, Buffer>* outputs = nullptr);

  // Resident session (section 24, "reuse model resources"). One
  // supervised child owns the device for the whole session: it loads
  // every named bundle's programs once and keeps them resident, so N
  // submits cost one process, one bundle load, and one device load
  // instead of N of each. It is still not an inference server (section
  // 25): no socket, no listener, no user-managed daemon -- the child is
  // a private helper whose lifetime the caller owns, and every submit
  // is bounded by the same deadline and refused by the same quarantine
  // rule as a one-shot run().
  //
  // Throws std::invalid_argument when a session is opened twice, when
  // no bundle is named, when submit()/close() is called without an
  // open session, or when session_names does not name every bundle
  // exactly once.
  //
  // session_names fixes the WIRE-PROTOCOL bundle names (what travels in
  // "submit <name>" and what the resident child resolves). They are the
  // caller's session keys, not the bundles' manifest names: the relay
  // keys bundles by its CLI name and the relay-bypass runner speaks
  // those keys directly. Empty means the manifest names (the historical
  // behavior, still correct whenever every caller resolves bundles by
  // index).
  AneWorkerReport open(
      const std::vector<AneBundle>& bundles,
      const std::vector<std::string>& session_names = {});

  // One bounded submit against an already-resident bundle, addressed by
  // its index in the open() vector. A deadline expiry or an abnormal
  // child death quarantines the worker exactly as in run(): the session
  // is torn down and every later call is refused.
  //
  // Inside an open batch scope (open_batch) the submit is bounded by the
  // batch's absolute deadline instead of a fresh per-submit window: the
  // batch is the deadline-bounded unit, and its rounds (this call, N
  // times) are the encoder's per-island-per-layer work.
  AneWorkerReport submit(
      size_t bundle,
      const std::map<std::string, Buffer>& inputs,
      std::map<std::string, Buffer>* outputs = nullptr);

  // Opens a batch scope: one absolute deadline, named here, bounds every
  // submit() until close_batch(). The safety model is unchanged -- one
  // private child, one bounded unit, failure ends the session, no retry
  // -- the bounded unit just spans the caller's whole island pass, which
  // is what lets 72 per-layer-per-island submits become one.
  AneWorkerReport open_batch(std::chrono::milliseconds deadline);

  // Closes the batch scope. The session stays open; later submits are
  // bounded per submit again.
  AneWorkerReport close_batch();

  // Rounds served inside the current batch scope.
  int batch_rounds() const {
    return batch_rounds_;
  }
  bool batching() const {
    return batch_until_.count() != 0;
  }

  // Releases the resident programs and reaps the child.
  AneWorkerReport close();

  bool resident() const {
    return child_ > 0;
  }

  // The resident child's pid, or -1 when no session is open. Callers
  // that record worker liveness (docs/ane-worker-liveness.md) report
  // this rather than pattern-matching process names.
  pid_t resident_pid() const {
    return child_;
  }

  // The parent-side end of the socketpair that talks to the resident
  // child. Exposed only for the relay-bypass pump: when the worker is
  // invoked as a pure byte bridge, it must splice bytes between its
  // own stdin/stdout and this fd with no parsing. Returns -1 when no
  // resident session is open.
  int channel_fd() const {
    return channel_;
  }

  bool quarantined() const {
    return !quarantine_reason_.empty();
  }
  const std::string& quarantine_reason() const {
    return quarantine_reason_;
  }

private:
  AneWorkerReport supervise(pid_t child, int report_fd);

  struct Wait {
    bool ok{false};
    AneWorkerStatus status{AneWorkerStatus::WorkerDied};
    std::string detail;
  };
  Wait send_request(
      const void* data,
      size_t size,
      std::chrono::milliseconds until);
  Wait recv_line(std::string& line, std::chrono::milliseconds until);
  Wait recv_bytes(size_t count, Buffer& out, std::chrono::milliseconds until);
  Wait recv_more(std::chrono::milliseconds until);
  Wait recv_raw(
      void* destination,
      size_t size,
      size_t& got,
      std::chrono::milliseconds until);
  Wait await_channel(
      short events,
      std::chrono::milliseconds until,
      const char* what);
  AneWorkerReport end_session(const Wait& failure);
  void teardown();

  DeviceFactory factory_;
  AneWorkerOptions options_;
  std::string quarantine_reason_;

  pid_t child_{-1};
  int channel_{-1};
  std::string inbox_;
  size_t inbox_cursor_{0};
  size_t resident_programs_{0};
  // Mirror of the bundles passed to open(); the wire protocol uses
  // bundle names (so the runner can speak the resident's protocol
  // directly without a relay translation step). Cleared by close().
  std::vector<std::string> resident_bundle_names_;

  // Batch scope: absolute deadline (0 = no scope open) and the number
  // of submits served inside it.
  std::chrono::milliseconds batch_until_{0};
  int batch_rounds_{0};
};

} // namespace mlx::core::omarchy::ane
