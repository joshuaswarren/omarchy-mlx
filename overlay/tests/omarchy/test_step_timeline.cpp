// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "step_timeline.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace mlx::core::omarchy::timeline;

namespace {

int fail(int code, const char* msg) {
  std::fprintf(stderr, "test_step_timeline FAIL (%d): %s\n", code, msg);
  return code;
}

std::string read_all(const std::string& path) {
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

int test_disabled_zero_cost() {
  StepTimelineRing ring(0);
  if (ring.enabled() || ring.capacity() != 0 || ring.written() != 0 ||
      ring.dropped() != 0) {
    return fail(1, "disabled ring must have enabled=false and capacity=0");
  }
  uint64_t clock_reads = 0;
  auto counting_clock = [&]() -> uint64_t {
    ++clock_reads;
    return clock_reads * 100ull;
  };
  {
    BasicScopedEvent scope(
        ring, Kind::DescUpdate, SubKind::DescWrite, 1, 4, counting_clock);
  }
  if (clock_reads != 0) {
    return fail(2, "disabled scoped event must not read the clock");
  }
  ring.record(Kind::QueueSubmit, SubKind::VkQueueSubmit, 100, 50, 1, 7);
  if (ring.written() != 0) {
    return fail(3, "disabled ring must not record events");
  }
  std::FILE* tmp = std::tmpfile();
  if (!tmp) {
    return fail(4, "tmpfile failed");
  }
  const bool dumped = ring.dump(tmp);
  std::fclose(tmp);
  if (dumped) {
    return fail(5, "disabled ring dump must return false");
  }
  return 0;
}

int test_ring_records_wraps_and_summarizes() {
  StepTimelineRing ring(5); // rounds up to 8
  if (!ring.enabled() || ring.capacity() != 8) {
    return fail(10, "non-power-of-2 capacity must round up to 8");
  }

  uint64_t fake_now = 1000;
  auto fake_clock = [&]() -> uint64_t {
    uint64_t v = fake_now;
    fake_now += 25;
    return v;
  };
  {
    BasicScopedEvent scope(
        ring, Kind::CmdRecord, SubKind::Dispatch, 2, 42, fake_clock);
  }
  // 11 more events -> 12 total in an 8-slot ring (4 dropped).
  for (uint64_t i = 1; i < 12; ++i) {
    ring.record(
        Kind::QueueSubmit,
        SubKind::VkQueueSubmit,
        2000 + i * 100,
        10 * i,
        1,
        i);
  }
  if (ring.written() != 12 || ring.dropped() != 4) {
    return fail(11, "written/dropped counts mismatch after wrap");
  }

  char tmpl[] = "/tmp/omarchy-tl-wrap-XXXXXX";
  int fd = ::mkstemp(tmpl);
  if (fd < 0) {
    return fail(12, "mkstemp failed");
  }
  ::close(fd);
  const std::string path(tmpl);
  if (!ring.dump_to_file(path.c_str())) {
    ::unlink(path.c_str());
    return fail(13, "dump_to_file failed");
  }
  const std::string out = read_all(path);
  ::unlink(path.c_str());

  if (out.find("\"k\":\"meta\"") == std::string::npos ||
      out.find("\"capacity\":8") == std::string::npos ||
      out.find("\"written\":12") == std::string::npos ||
      out.find("\"dropped\":4") == std::string::npos) {
    return fail(14, "meta header missing expected capacity/written/dropped");
  }
  // Oldest retained event must be seq=4, newest seq=11, and seq=0..3 absent.
  if (out.find("\"seq\":3,") != std::string::npos ||
      out.find("\"seq\":4,") == std::string::npos ||
      out.find("\"seq\":11,") == std::string::npos) {
    return fail(15, "ring wrap retained wrong sequence window");
  }
  // Summary must include ALL 12 events (1 CmdRecord/Dispatch dur=25 + 11 QueueSubmit/VkQueueSubmit dur=sum(10..110)=660, max=110).
  if (out.find("\"k\":\"summary\"") == std::string::npos ||
      out.find("{\"kind\":\"cmd_record\",\"sub\":\"dispatch\",\"count\":1,\"total_ns\":25,\"max_ns\":25}") ==
          std::string::npos ||
      out.find("{\"kind\":\"queue_submit\",\"sub\":\"vk_queue_submit\",\"count\":11,\"total_ns\":660,\"max_ns\":110}") ==
          std::string::npos) {
    return fail(16, "summary missing exact whole-run totals across wrapped events");
  }
  return 0;
}

int test_scoped_cancel_and_mutations() {
  StepTimelineRing ring(16);
  uint64_t fake_now = 5000;
  auto fake_clock = [&]() -> uint64_t {
    uint64_t v = fake_now;
    fake_now += 40;
    return v;
  };
  {
    BasicScopedEvent cancelled(
        ring, Kind::Wait, SubKind::DrainThrough, 1, 9, fake_clock);
    cancelled.cancel();
  }
  if (ring.written() != 0) {
    return fail(17, "cancelled scope must not record an event");
  }
  {
    BasicScopedEvent mutated(
        ring, Kind::PipelineLookup, SubKind::PipelineHit, 3, 1, fake_clock);
    mutated.set_subkind(SubKind::PipelineMiss);
    mutated.set_detail(99);
  }
  if (ring.written() != 1) {
    return fail(18, "mutated scope must record exactly one event");
  }
  char tmpl[] = "/tmp/omarchy-tl-mut-XXXXXX";
  int fd = ::mkstemp(tmpl);
  if (fd < 0) {
    return fail(19, "mkstemp failed");
  }
  ::close(fd);
  const std::string path(tmpl);
  if (!ring.dump_to_file(path.c_str())) {
    ::unlink(path.c_str());
    return fail(19, "dump_to_file failed");
  }
  const std::string out = read_all(path);
  ::unlink(path.c_str());
  if (out.find("\"kind\":\"pipeline_lookup\",\"sub\":\"pipeline_miss\"") ==
          std::string::npos ||
      out.find("\"st\":3,\"d\":99") == std::string::npos ||
      out.find("\"dur\":40,") == std::string::npos) {
    return fail(19, "mutated subkind/detail/dur missing from dump");
  }
  return 0;
}

int test_concurrent_recording() {
  constexpr int kThreads = 4;
  constexpr int kPerThread = 500;
  StepTimelineRing ring(2048);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < kPerThread; ++i) {
        const uint64_t t0 = monotonic_ns();
        ring.record(
            Kind::FenceOp,
            SubKind::FencePoll,
            t0,
            static_cast<uint64_t>(i),
            static_cast<uint16_t>(t + 1),
            static_cast<uint64_t>(i));
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  if (ring.written() != static_cast<uint64_t>(kThreads * kPerThread) ||
      ring.dropped() != 0) {
    return fail(20, "concurrent recording lost events");
  }
  char tmpl[] = "/tmp/omarchy-tl-mt-XXXXXX";
  int fd = ::mkstemp(tmpl);
  if (fd < 0) {
    return fail(21, "mkstemp failed");
  }
  ::close(fd);
  const std::string path(tmpl);
  if (!ring.dump_to_file(path.c_str())) {
    ::unlink(path.c_str());
    return fail(22, "concurrent dump failed");
  }
  const std::string out = read_all(path);
  ::unlink(path.c_str());

  size_t event_lines = 0;
  std::istringstream iss(out);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.rfind("{\"k\":\"e\",", 0) == 0) {
      ++event_lines;
    }
  }
  if (event_lines != static_cast<size_t>(kThreads * kPerThread)) {
    return fail(23, "concurrent dump missed committed slots");
  }
  if (out.find("{\"kind\":\"fence_op\",\"sub\":\"fence_poll\",\"count\":2000,") ==
      std::string::npos) {
    return fail(24, "concurrent summary count mismatch");
  }
  return 0;
}

bool extract_u64_field(
    const std::string& line,
    const char* key,
    uint64_t& out) {
  const auto pos = line.find(key);
  if (pos == std::string::npos) {
    return false;
  }
  char* end = nullptr;
  const char* start = line.c_str() + pos + std::strlen(key);
  out = std::strtoull(start, &end, 10);
  return end != start;
}

int test_concurrent_wrap_ownership() {
  constexpr int kThreads = 4;
  constexpr uint64_t kPerThread = 4000;
  constexpr uint64_t kCap = 8;
  StepTimelineRing ring(kCap);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      const uint16_t st = static_cast<uint16_t>(t + 1);
      for (uint64_t i = 0; i < kPerThread; ++i) {
        const uint64_t t0 = 1000000ull + i * 10ull;
        const uint64_t detail = (static_cast<uint64_t>(st) << 32) | i;
        ring.record(Kind::CmdRecord, SubKind::Dispatch, t0, i, st, detail);
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  const uint64_t expected_total = static_cast<uint64_t>(kThreads) * kPerThread;
  if (ring.written() != expected_total) {
    return fail(25, "wrapping concurrency test lost admitted count");
  }

  char tmpl[] = "/tmp/omarchy-tl-wrapmt-XXXXXX";
  int fd = ::mkstemp(tmpl);
  if (fd < 0) {
    return fail(26, "mkstemp failed");
  }
  ::close(fd);
  const std::string path(tmpl);
  if (!ring.dump_to_file(path.c_str())) {
    ::unlink(path.c_str());
    return fail(26, "wrapping concurrent dump failed");
  }
  const std::string out = read_all(path);
  ::unlink(path.c_str());

  if (out.find("\"written\":16000") == std::string::npos ||
      out.find("\"busy_dropped\":") == std::string::npos ||
      out.find("{\"kind\":\"cmd_record\",\"sub\":\"dispatch\",\"count\":16000,") ==
          std::string::npos) {
    return fail(27, "wrapping concurrent dump missing exact meta/summary totals");
  }

  std::istringstream iss(out);
  std::string line;
  uint64_t prev_seq = 0;
  bool have_seq = false;
  uint64_t retained = 0;
  while (std::getline(iss, line)) {
    if (line.rfind("{\"k\":\"e\",", 0) != 0) {
      continue;
    }
    uint64_t seq = 0, t0 = 0, dur = 0, st = 0, detail = 0;
    if (!extract_u64_field(line, "\"seq\":", seq) ||
        !extract_u64_field(line, "\"t0\":", t0) ||
        !extract_u64_field(line, "\"dur\":", dur) ||
        !extract_u64_field(line, "\"st\":", st) ||
        !extract_u64_field(line, "\"d\":", detail)) {
      return fail(28, "malformed event line in wrapping dump");
    }
    if (seq < expected_total - kCap || seq >= expected_total) {
      return fail(28, "retained seq outside tail window");
    }
    if (have_seq && seq <= prev_seq) {
      return fail(28, "retained seq not strictly increasing");
    }
    // Verify fields were never mixed across threads or iterations.
    if (st < 1 || st > static_cast<uint64_t>(kThreads) ||
        t0 != 1000000ull + dur * 10ull ||
        detail != ((st << 32) | dur)) {
      return fail(29, "torn or mixed event fields detected under wrap");
    }
    prev_seq = seq;
    have_seq = true;
    ++retained;
  }
  if (retained > kCap) {
    return fail(29, "retained more events than ring capacity");
  }
  return 0;
}

int test_concurrent_stop_and_dump() {
  constexpr int kThreads = 4;
  constexpr uint64_t kCap = 16;
  StepTimelineRing ring(kCap);
  std::atomic<bool> keep_running{true};
  std::atomic<int> started{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      const uint16_t st = static_cast<uint16_t>(t + 1);
      started.fetch_add(1, std::memory_order_release);
      uint64_t i = 0;
      while (keep_running.load(std::memory_order_relaxed)) {
        const uint64_t t0 = 2000000ull + i * 7ull;
        const uint64_t detail = (static_cast<uint64_t>(st) << 32) | i;
        ring.record(Kind::Wait, SubKind::DrainThrough, t0, i, st, detail);
        ++i;
      }
    });
  }
  while (started.load(std::memory_order_acquire) < kThreads ||
         ring.written() < 2000) {
    std::this_thread::yield();
  }

  // Dump while producers are actively hammering record(); stop() inside dump()
  // must quiesce all in-flight producers before reading slots or stats.
  char tmpl[] = "/tmp/omarchy-tl-stop-XXXXXX";
  int fd = ::mkstemp(tmpl);
  if (fd < 0) {
    keep_running.store(false, std::memory_order_relaxed);
    for (auto& th : threads) {
      th.join();
    }
    return fail(35, "mkstemp failed");
  }
  ::close(fd);
  const std::string path(tmpl);
  const bool ok = ring.dump_to_file(path.c_str());
  const uint64_t written_after_dump = ring.written();
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  const uint64_t written_later = ring.written();
  keep_running.store(false, std::memory_order_relaxed);
  for (auto& th : threads) {
    th.join();
  }
  const std::string out = read_all(path);
  ::unlink(path.c_str());

  if (!ok || ring.enabled() || written_after_dump != written_later) {
    return fail(36, "stop/dump did not quiesce producers");
  }

  uint64_t meta_written = 0;
  uint64_t summary_count = 0;
  std::istringstream iss(out);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.rfind("{\"k\":\"meta\",", 0) == 0) {
      if (!extract_u64_field(line, "\"written\":", meta_written)) {
        return fail(37, "meta missing written");
      }
    } else if (line.rfind("{\"k\":\"summary\",", 0) == 0) {
      if (!extract_u64_field(line, "\"count\":", summary_count)) {
        return fail(37, "summary missing count");
      }
    } else if (line.rfind("{\"k\":\"e\",", 0) == 0) {
      uint64_t t0 = 0, dur = 0, st = 0, detail = 0;
      if (!extract_u64_field(line, "\"t0\":", t0) ||
          !extract_u64_field(line, "\"dur\":", dur) ||
          !extract_u64_field(line, "\"st\":", st) ||
          !extract_u64_field(line, "\"d\":", detail)) {
        return fail(38, "malformed event during concurrent stop/dump");
      }
      if (t0 != 2000000ull + dur * 7ull || detail != ((st << 32) | dur)) {
        return fail(38, "torn event during concurrent stop/dump");
      }
    }
  }
  if (meta_written == 0 || meta_written != written_after_dump ||
      summary_count != meta_written) {
    return fail(39, "meta written and summary count diverged across stop/dump");
  }
  return 0;
}

int test_pre_op_step_begin_ordering() {
  StepTimelineRing ring(32);
  uint64_t fake_now = 10000;
  auto fake_clock = [&]() -> uint64_t {
    uint64_t v = fake_now;
    fake_now += 50;
    return v;
  };
  // Mirrors eval.cpp: capture pre-op start timestamp before the first
  // primitive runs, record the op, then publish batch_open with the captured t0.
  const uint64_t pre_op_t0 = fake_clock();
  {
    BasicScopedEvent op(
        ring, Kind::CmdRecord, SubKind::Dispatch, 1, 7, fake_clock);
  }
  ring.record(Kind::StepBegin, SubKind::BatchOpen, pre_op_t0, 0, 1, 0);

  char tmpl[] = "/tmp/omarchy-tl-ord-XXXXXX";
  int fd = ::mkstemp(tmpl);
  if (fd < 0) {
    return fail(50, "mkstemp failed");
  }
  ::close(fd);
  const std::string path(tmpl);
  if (!ring.dump_to_file(path.c_str())) {
    ::unlink(path.c_str());
    return fail(50, "dump_to_file failed");
  }
  const std::string out = read_all(path);
  ::unlink(path.c_str());

  uint64_t batch_open_t0 = UINT64_MAX;
  uint64_t op_t0 = UINT64_MAX;
  std::istringstream iss(out);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.find("\"kind\":\"step_begin\",\"sub\":\"batch_open\"") !=
        std::string::npos) {
      extract_u64_field(line, "\"t0\":", batch_open_t0);
    } else if (line.find("\"kind\":\"cmd_record\",\"sub\":\"dispatch\"") !=
               std::string::npos) {
      extract_u64_field(line, "\"t0\":", op_t0);
    }
  }
  if (batch_open_t0 == UINT64_MAX || op_t0 == UINT64_MAX ||
      batch_open_t0 > op_t0) {
    return fail(51, "pre-op batch_open timestamp must precede first op");
  }
  return 0;
}

int test_dump_write_and_close_errors() {
  // 1. Read-only FILE* must make dump() return false.
  StepTimelineRing ring1(16);
  ring1.record(Kind::StepBegin, SubKind::BatchOpen, 100, 0, 1, 0);
  std::FILE* ro = std::fopen("/dev/null", "r");
  if (ro == nullptr) {
    return fail(52, "fopen /dev/null r failed");
  }
  const bool ro_ok = ring1.dump(ro);
  std::fclose(ro);
  if (ro_ok) {
    return fail(53, "dump to read-only FILE* must return false");
  }

  // 2. Non-existent directory must make dump_to_file() return false.
  StepTimelineRing ring2(16);
  ring2.record(Kind::StepBegin, SubKind::BatchOpen, 100, 0, 1, 0);
  if (ring2.dump_to_file("/nonexistent-omarchy-tl-dir/out.ndjson")) {
    return fail(54, "dump_to_file to nonexistent directory must return false");
  }

  // 3. On Linux, /dev/full accepts fopen("w") and fails with ENOSPC on
  // fflush/fclose. Verify dump_to_file() returns false AND atexit_dump()
  // surfaces the failure to stderr.
  if (::access("/dev/full", W_OK) == 0) {
    StepTimelineRing ring3(16);
    ring3.record(Kind::CmdRecord, SubKind::Dispatch, 100, 20, 1, 3);
    if (ring3.dump_to_file("/dev/full")) {
      return fail(55, "dump_to_file(/dev/full) must return false on ENOSPC");
    }

    int pipefd[2];
    if (::pipe(pipefd) != 0) {
      return fail(56, "pipe failed");
    }
    pid_t pid = ::fork();
    if (pid < 0) {
      ::close(pipefd[0]);
      ::close(pipefd[1]);
      return fail(56, "fork failed");
    }
    if (pid == 0) {
      ::close(pipefd[0]);
      ::dup2(pipefd[1], STDERR_FILENO);
      ::close(pipefd[1]);
      ::setenv("MLX_OMARCHY_STEP_TIMELINE", "/dev/full", 1);
      ::setenv("MLX_OMARCHY_STEP_TIMELINE_CAP", "1024", 1);
      record(Kind::StepBegin, SubKind::BatchOpen, 1, 0);
      std::exit(0);
    }
    ::close(pipefd[1]);
    std::string err_out;
    char buf[256];
    ssize_t n = 0;
    while ((n = ::read(pipefd[0], buf, sizeof(buf))) > 0) {
      err_out.append(buf, static_cast<size_t>(n));
    }
    ::close(pipefd[0]);
    int status = 0;
    ::waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      return fail(57, "dev/full child exited abnormally");
    }
    if (err_out.find("failed to write MLX_OMARCHY_STEP_TIMELINE") ==
        std::string::npos) {
      std::fprintf(stderr, "unexpected stderr: %s\n", err_out.c_str());
      return fail(58, "atexit_dump did not surface /dev/full write failure");
    }
  }
  return 0;
}

int test_capacity_parsing() {
  if (parse_capacity(nullptr) != kDefaultCapacity ||
      parse_capacity("") != kDefaultCapacity ||
      parse_capacity("abc") != kDefaultCapacity ||
      parse_capacity("0") != kDefaultCapacity ||
      parse_capacity("-5") != kDefaultCapacity) {
    return fail(30, "invalid capacity env must fall back to default");
  }
  if (parse_capacity("16") != kMinCapacity) {
    return fail(31, "small capacity must clamp to kMinCapacity");
  }
  if (parse_capacity("999999999") != kMaxCapacity) {
    return fail(32, "huge capacity must clamp to kMaxCapacity");
  }
  if (parse_capacity("1500") != 2048) {
    return fail(33, "capacity must round up to power of 2");
  }
  return 0;
}

int test_env_gate_and_atexit_dump() {
  char tmpl[] = "/tmp/omarchy-tl-exit-XXXXXX";
  int fd = ::mkstemp(tmpl);
  if (fd < 0) {
    return fail(40, "mkstemp failed");
  }
  ::close(fd);
  const std::string path(tmpl);
  ::unlink(path.c_str());

  // Child 1: env unset -> disabled, no file written.
  pid_t p1 = ::fork();
  if (p1 < 0) {
    return fail(41, "fork 1 failed");
  }
  if (p1 == 0) {
    ::unsetenv("MLX_OMARCHY_STEP_TIMELINE");
    ::unsetenv("MLX_OMARCHY_STEP_TIMELINE_CAP");
    if (enabled()) {
      std::exit(101);
    }
    record(Kind::StepBegin, SubKind::BatchOpen, 1, 0);
    std::exit(0);
  }
  int status = 0;
  ::waitpid(p1, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    return fail(42, "disabled child failed");
  }
  if (::access(path.c_str(), F_OK) == 0) {
    ::unlink(path.c_str());
    return fail(43, "disabled child must not create timeline file");
  }

  // Child 2: env set -> enabled, records all required kinds, dumps at exit.
  pid_t p2 = ::fork();
  if (p2 < 0) {
    return fail(44, "fork 2 failed");
  }
  if (p2 == 0) {
    ::setenv("MLX_OMARCHY_STEP_TIMELINE", path.c_str(), 1);
    ::setenv("MLX_OMARCHY_STEP_TIMELINE_CAP", "1024", 1);
    if (!enabled()) {
      std::exit(102);
    }
    record(Kind::StepBegin, SubKind::BatchOpen, 1, 1);
    { ScopedEvent e(Kind::PipelineLookup, SubKind::PipelineHit, 1, 7); }
    { ScopedEvent e(Kind::DescUpdate, SubKind::DescWrite, 1, 4); }
    { ScopedEvent e(Kind::CmdBegin, SubKind::Begin, 1, 0); }
    { ScopedEvent e(Kind::CmdRecord, SubKind::Dispatch, 1, 7); }
    { ScopedEvent e(Kind::QueueSubmit, SubKind::VkQueueSubmit, 1, 1); }
    { ScopedEvent e(Kind::FenceOp, SubKind::FenceAcquire, 1, 1); }
    { ScopedEvent e(Kind::Wait, SubKind::TimelineWait, 1, 1); }
    { ScopedEvent e(Kind::Readback, SubKind::SyncJoin, 1, 1); }
    record(Kind::StepEnd, SubKind::Finalize, 1, 1);
    std::exit(0);
  }
  ::waitpid(p2, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    ::unlink(path.c_str());
    return fail(45, "enabled child failed");
  }
  const std::string out = read_all(path);
  ::unlink(path.c_str());
  if (out.empty()) {
    return fail(46, "atexit dump produced empty or missing file");
  }
  for (const char* expected : {
           "\"k\":\"meta\"",
           "\"clock\":\"CLOCK_MONOTONIC\"",
           "\"capacity\":1024",
           "\"written\":10",
           "\"dropped\":0",
           "\"kind\":\"step_begin\",\"sub\":\"batch_open\"",
           "\"kind\":\"pipeline_lookup\",\"sub\":\"pipeline_hit\"",
           "\"kind\":\"desc_update\",\"sub\":\"desc_write\"",
           "\"kind\":\"cmd_begin\",\"sub\":\"begin\"",
           "\"kind\":\"cmd_record\",\"sub\":\"dispatch\"",
           "\"kind\":\"queue_submit\",\"sub\":\"vk_queue_submit\"",
           "\"kind\":\"fence_op\",\"sub\":\"fence_acquire\"",
           "\"kind\":\"wait\",\"sub\":\"timeline_wait\"",
           "\"kind\":\"readback\",\"sub\":\"sync_join\"",
           "\"kind\":\"step_end\",\"sub\":\"finalize\"",
           "\"k\":\"summary\"",
       }) {
    if (out.find(expected) == std::string::npos) {
      std::fprintf(stderr, "missing token in atexit dump: %s\n", expected);
      return fail(47, "atexit dump missing required event or metadata");
    }
  }
  // Verify timestamps in the dumped event lines are monotonic non-decreasing.
  uint64_t prev_t0 = 0;
  std::istringstream iss(out);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.rfind("{\"k\":\"e\",", 0) != 0) {
      continue;
    }
    const auto pos = line.find("\"t0\":");
    if (pos == std::string::npos) {
      return fail(48, "event line missing t0");
    }
    const uint64_t t0 = std::strtoull(line.c_str() + pos + 5, nullptr, 10);
    if (t0 == 0 || t0 < prev_t0) {
      return fail(49, "event timestamps must be non-zero and monotonic");
    }
    prev_t0 = t0;
  }
  return 0;
}

} // namespace

int main() {
  if (int rc = test_disabled_zero_cost()) {
    return rc;
  }
  if (int rc = test_ring_records_wraps_and_summarizes()) {
    return rc;
  }
  if (int rc = test_scoped_cancel_and_mutations()) {
    return rc;
  }
  if (int rc = test_concurrent_recording()) {
    return rc;
  }
  if (int rc = test_concurrent_wrap_ownership()) {
    return rc;
  }
  if (int rc = test_concurrent_stop_and_dump()) {
    return rc;
  }
  if (int rc = test_pre_op_step_begin_ordering()) {
    return rc;
  }
  if (int rc = test_dump_write_and_close_errors()) {
    return rc;
  }
  if (int rc = test_capacity_parsing()) {
    return rc;
  }
  if (int rc = test_env_gate_and_atexit_dump()) {
    return rc;
  }
  std::printf("test_step_timeline: 10/10 tests passed\n");
  return 0;
}
