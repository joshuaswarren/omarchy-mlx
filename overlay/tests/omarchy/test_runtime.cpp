// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Focused runtime tests for the Omarchy Vulkan backend slice:
//   1. discovery classification (supported / unsupported hardware),
//   2. device selection through the MLX device model,
//   3. a real Vulkan buffer round trip through the encoder,
//   4. event lifetime, cross-stream ordering, handler-only commits, and
//      the completion-wait drain join when the dispatcher thread wins the
//      pending_ race,
//   5. temporary ownership across asynchronous (handler-free) completion,
//   6. device_info reference stability under concurrent access,
//   7. measured concurrency of independent streams on a hardware GPU,
//   8. fresh-process device reopen and bounded failed-submit errors,
//      proven in process-isolated child runs,
// 10. safe command-buffer reuse across many asynchronous commits.
// 11. in-order-stream contract: a small eager output crosses deep
//     submit boundaries into a later consumer dispatch.
// 12. a no-progress watchdog throw unwinds through FULL process teardown
//     (encoder and device destruction) without crashing, in a
//     process-isolated child run.
// 13. dependency-gated barriers (MLX_OMARCHY_GATED_BARRIERS): RAW, WAW,
//     and WAR chains on one buffer hold their values with the gate on,
//     hazard pairs emit, disjoint pairs skip.
//
// The suite needs MLX_BUILD_OMARCHY=ON and compiles against Vulkan 1.3
// headers (the Honeykrisp driver id is pinned in device.h). Round-trip and
// device-selection tests need a qualifying Vulkan device; on non-Omarchy
// machines set MLX_OMARCHY_ALLOW_NON_APPLE=1 to exercise them against a
// development driver (reported as dev-only by mlx-omarchy-info).

#define DOCTEST_CONFIG_IMPLEMENT
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "doctest/doctest.h"
#include "mlx/backend/cpu/device_info.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/cpu_pd_hold.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/backend/omarchy/vulkan.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/scheduler.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

omarchy::DeviceSupport classify(
    const char* name,
    uint32_t vendor_id,
    uint32_t api_version,
    int32_t driver_id,
    bool allow_non_apple) {
  VkPhysicalDeviceProperties props{};
  std::strncpy(props.deviceName, name, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);
  props.vendorID = vendor_id;
  props.deviceID = 0x6001;
  props.apiVersion = api_version;
  return omarchy::classify_physical_device(props, driver_id, allow_non_apple);
}

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

// --- Process-isolated failure-mode scenarios ------------------------------
//
// The parent test re-execs this same binary with MLX_OMARCHY_TEST_CHILD set.
// The child runs one scenario from a fresh process (fresh Vulkan discovery
// and a fresh VkDevice) and exits with a code the parent asserts on. This
// isolates wedged queues and device teardown from the test process, per the
// hardware-safety rules in AGENTS.md.

int run_child_scenario(const std::string& mode) {
  if (!gpu::is_available()) {
    return 77;
  }
  if (mode == "reopen") {
    // Fresh process: discovery, VkDevice creation, work, and full teardown.
    auto& dev = omarchy::device(0);
    if (dev.handle() == VK_NULL_HANDLE || dev.queue() == VK_NULL_HANDLE) {
      return 1;
    }
    Stream s = new_stream(Device::gpu);
    auto& enc = omarchy::get_command_encoder(s);
    auto buf = omarchy::allocator().malloc(4096);
    auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());
    enc.fill_buffer(p->buffer, 0x2a, 4096);
    enc.commit();
    enc.synchronize();
    auto* words = static_cast<uint32_t*>(p->data);
    for (size_t i = 0; i < 4096 / sizeof(uint32_t); ++i) {
      if (words[i] != 0x2a) {
        omarchy::allocator().free(buf);
        return 2;
      }
    }
    omarchy::allocator().free(buf);
    std::cout << "[child/reopen] fresh process device reopen ok\n";
    return 0;
  }
  if (mode == "bounded_submit") {
    // Queue a wait that can never be satisfied, then block on completion.
    // The submit must return control with a typed Omarchy error instead of
    // hanging forever (plan R16). No CPU rescue is attempted for a hung GPU.
    Stream a = new_stream(Device::gpu);
    Stream b = new_stream(Device::gpu);
    Event e{a};
    e.set_value(42);
    auto& enc = omarchy::get_command_encoder(b);
    e.wait(b);
    auto buf = omarchy::allocator().malloc(4096);
    auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());
    enc.fill_buffer(p->buffer, 1, 4096);
    enc.commit();
    auto t0 = std::chrono::steady_clock::now();
    try {
      enc.synchronize();
      std::cout << "[child/bounded_submit] ERROR: hung submit never threw\n";
      omarchy::allocator().free(buf);
      return 3;
    } catch (const std::exception& ex) {
      auto elapsed = std::chrono::steady_clock::now() - t0;
      std::string msg = ex.what();
      std::cout << "[child/bounded_submit] typed error: " << msg << "\n";
      if (msg.find("[omarchy]") == std::string::npos) {
        omarchy::allocator().free(buf);
        return 4;
      }
      if (msg.find("no CPU fallback") == std::string::npos) {
        omarchy::allocator().free(buf);
        return 5;
      }
      // The recovery ladder spends up to two full no-progress windows
      // (kick+resubmit rounds) before the budget-exhausted throw, so the
      // designed worst case at the default 10 s window is ~30 s; the
      // bounded-error contract this child tests is that the error EXISTS
      // and is typed, not that it races the old single-window latency.
      if (elapsed > std::chrono::seconds(60)) {
        omarchy::allocator().free(buf);
        return 6;
      }
      // _Exit skips destruction on purpose: the queue keeps the wedged
      // submission, so teardown would block. The parent bounds this child.
      std::_Exit(0);
    }
  }
  if (mode == "progressing_long") {
    // Run a submission long enough to exceed the no-progress interval
    // while keeping the timeline counter advancing. Old behavior with
    // the 10 s wall cap failed by name after exactly 10 s regardless
    // of counter motion; new behavior completes successfully because
    // the watchdog only fires when no progress is observed. To force a
    // multi-second-but-progressing submission on a software renderer
    // we issue a sequence of batches whose combined wall time crosses
    // the watchdog interval; the per-batch signal keeps the counter
    // moving, so the watchdog never trips.
    Stream s = new_stream(Device::gpu);
    auto& enc = omarchy::get_command_encoder(s);
    setenv("MLX_OMARCHY_HANG_NO_PROGRESS_NS", "500000000", 1); // 500 ms
    auto buf = omarchy::allocator().malloc(64u << 20); // 64 MiB
    auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());
    auto t0 = std::chrono::steady_clock::now();
    try {
      // Issue several batches so the per-batch signal keeps the
      // counter advancing while the total wall crosses 2 s. Each
      // fill is large enough to keep the GPU busy for a measurable
      // interval even on llvmpipe.
      // 36 batches at 64 MiB each cross the 2 s no-progress interval
      // on llvmpipe while keeping the per-batch signal moving the
      // counter; the watchdog therefore must not fire even though the
      // total wall time exceeds the interval.
      for (int i = 0; i < 36; ++i) {
        enc.fill_buffer(p->buffer, static_cast<uint32_t>(i + 1), 64u << 20);
        enc.commit();
      }
      enc.synchronize();
      auto elapsed = std::chrono::steady_clock::now() - t0;
      std::cout << "[child/progressing_long] completed in "
                << std::chrono::duration<double>(elapsed).count()
                << " s with no hang\n";
      omarchy::allocator().free(buf);
      std::_Exit(0);
    } catch (const std::exception& ex) {
      std::cout << "[child/progressing_long] FAILED: " << ex.what() << "\n";
      omarchy::allocator().free(buf);
      std::_Exit(7);
    }
  }
  if (mode == "watchdog_teardown") {
    // Fire the no-progress watchdog on a genuinely slow dispatch, catch
    // the typed error, and then exit through FULL process teardown. The
    // setenv lands before any completion wait in this fresh child, so
    // the 1 ns interval is what wait_for_timeline_progress caches. One
    // batch of 64 MiB fills takes well over the 100 ms poll interval on
    // llvmpipe, so the watchdog fires on real in-flight work (the same
    // misclassification a jumbo prefill hits); on fast hardware the
    // fills complete inside the first poll and the child still asserts
    // clean teardown after a successful join. Per docs/known-defects.md
    // a process that hit a wedge must be restarted before its output is
    // trusted, so the contract under test is exit 0 through teardown,
    // not same-device recovery. The allocation is leaked on purpose:
    // the fill is still executing, and a cache-free would hand that
    // memory back while the queue is writing it.
    setenv("MLX_OMARCHY_HANG_NO_PROGRESS_NS", "1", 1);
    Stream s = new_stream(Device::gpu);
    auto& enc = omarchy::get_command_encoder(s);
    auto buf = omarchy::allocator().malloc(64u << 20);
    auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());
    for (int i = 0; i < 16; ++i) {
      enc.fill_buffer(p->buffer, static_cast<uint32_t>(i + 1), 64u << 20);
    }
    enc.commit();
    try {
      enc.synchronize();
      std::cout << "[child/watchdog_teardown] submit completed before the"
                   " watchdog fired; teardown is the assertion\n";
      return 0;
    } catch (const std::exception& ex) {
      std::string msg = ex.what();
      std::cout << "[child/watchdog_teardown] typed error: " << msg << "\n";
      if (msg.find("[omarchy]") == std::string::npos) {
        return 8;
      }
      if (msg.find("no CPU fallback") == std::string::npos) {
        return 9;
      }
      return 0;
    }
  }
  if (mode == "cpu_event_signal_first" || mode == "cpu_event_wait_first") {
    if (!cpu::is_available()) {
      return 77;
    }
    const bool wait_first = mode == "cpu_event_wait_first";
    const uint32_t expected = wait_first ? 0xa5a5a5a5u : 0x5a5a5a5au;
    Stream producer = new_stream(Device::cpu);
    Stream consumer = new_stream(Device::gpu);
    auto src = omarchy::allocator().malloc(4096);
    auto dst = omarchy::allocator().malloc(4096);
    auto* src_buf = static_cast<omarchy::VulkanBuffer*>(src.ptr());
    auto* dst_buf = static_cast<omarchy::VulkanBuffer*>(dst.ptr());
    std::memset(src_buf->data, 0, 4096);
    std::memset(dst_buf->data, 0, 4096);

    std::promise<void> release_producer;
    std::shared_future<void> released = release_producer.get_future().share();
    scheduler::enqueue(producer, [src_buf, released, expected]() {
      released.wait();
      std::fill_n(static_cast<uint32_t*>(src_buf->data), 1024, expected);
    });

    auto& encoder = omarchy::get_command_encoder(consumer);
    {
      Event produced{producer};
      produced.set_value(1);
      if (!wait_first) {
        produced.signal(producer);
      }
      produced.wait(consumer);
      if (wait_first) {
        produced.signal(producer);
      }
      encoder.copy_buffer(src_buf->buffer, dst_buf->buffer, 4096);
    }
    encoder.commit();

    std::atomic<bool> consumer_returned{false};
    std::thread waiter([&]() {
      encoder.synchronize();
      consumer_returned.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const bool returned_before_release = consumer_returned.load();
    release_producer.set_value();
    waiter.join();
    synchronize(producer);

    auto* words = static_cast<uint32_t*>(dst_buf->data);
    const bool correct = words[0] == expected && words[1023] == expected;
    omarchy::allocator().free(src);
    omarchy::allocator().free(dst);
    if (returned_before_release) {
      return 20;
    }
    if (!consumer_returned.load()) {
      return 21;
    }
    return correct ? 0 : 22;
  }
  if (mode == "reused_slot_blocked_dependency") {
    setenv("MLX_OMARCHY_HANG_NO_PROGRESS_NS", "200000000", 1);
    setenv("MLX_OMARCHY_MAX_WALL_NS", "5000000000", 1);
    Stream s = new_stream(Device::gpu);
    auto& enc = omarchy::get_command_encoder(s);
    auto buffer = omarchy::allocator().malloc(4096);
    auto* p = static_cast<omarchy::VulkanBuffer*>(buffer.ptr());
    for (int i = 0; i < 8; ++i) {
      enc.fill_buffer(p->buffer, static_cast<uint32_t>(i), 4096);
      enc.commit();
    }
    enc.synchronize();

    Event unsignaled{new_stream(Device::gpu)};
    unsignaled.set_value(42);
    unsignaled.wait(s);
    enc.fill_buffer(p->buffer, 99, 4096);
    try {
      enc.synchronize();
      std::_Exit(33);
    } catch (const std::exception& ex) {
      std::string message = ex.what();
      std::cout << "[child/reused_slot_blocked_dependency] " << message
                << std::endl;
      std::_Exit(
          message.find("failed to advance") != std::string::npos ? 0 : 34);
    }
  }
  if (mode == "late_reused_event_publication") {
    setenv("MLX_OMARCHY_HANG_NO_PROGRESS_NS", "1", 1);
    setenv("MLX_OMARCHY_MAX_WALL_NS", "30000000000", 1);
    Stream s = new_stream(Device::gpu);
    auto& enc = omarchy::get_command_encoder(s);
    auto buffer = omarchy::allocator().malloc(4096);
    auto* p = static_cast<omarchy::VulkanBuffer*>(buffer.ptr());
    Event event{s};
    event.set_value(1);
    enc.fill_buffer(p->buffer, 1, 4096);
    event.signal(s);
    event.wait();

    constexpr int kSize = 1024;
    auto x = ones({kSize, kSize}, float32, s);
    auto y = matmul(x, x, s);
    y.eval();

    event.set_value(2);
    std::atomic<bool> waiter_started{false};
    std::string wait_error;
    std::thread waiter([&]() {
      waiter_started.store(true, std::memory_order_release);
      try {
        event.wait();
      } catch (const std::exception& ex) {
        wait_error = ex.what();
      }
    });
    while (!waiter_started.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    event.signal(s);
    waiter.join();
    enc.synchronize();
    const auto* values = y.data<float>();
    const bool correct =
        values[0] == kSize && values[y.size() - 1] == kSize;
    omarchy::allocator().free(buffer);
    if (!wait_error.empty()) {
      std::cout << "[child/late_reused_event_publication] " << wait_error
                << std::endl;
      std::_Exit(35);
    }
    std::_Exit(correct ? 0 : 36);
  }
  if (mode == "unrelated_work_event_wait") {
    setenv("MLX_OMARCHY_HANG_NO_PROGRESS_NS", "100000000", 1);
    setenv("MLX_OMARCHY_MAX_WALL_NS", "30000000000", 1);
    Stream busy = new_stream(Device::gpu);
    constexpr int kSize = 1024;
    auto x = ones({kSize, kSize}, float32, busy);
    auto y = matmul(x, x, busy);
    y.eval();

    Event unsignaled{new_stream(Device::gpu)};
    unsignaled.set_value(1);
    const auto start = std::chrono::steady_clock::now();
    std::string message;
    try {
      unsignaled.wait();
      std::_Exit(37);
    } catch (const std::exception& ex) {
      message = ex.what();
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    omarchy::get_command_encoder(busy).synchronize();
    const auto* values = y.data<float>();
    const bool correct =
        values[0] == kSize && values[y.size() - 1] == kSize;
    const bool bounded = elapsed < std::chrono::milliseconds(600);
    std::cout << "[child/unrelated_work_event_wait] " << message << std::endl;
    std::_Exit(
        correct && bounded &&
                message.find("failed to advance") != std::string::npos
            ? 0
            : 38);
  }
  if (mode == "completed_marker_behind_handler") {
    setenv("MLX_OMARCHY_HANG_NO_PROGRESS_NS", "100000000", 1);
    setenv("MLX_OMARCHY_MAX_WALL_NS", "1000000000", 1);
    Stream s = new_stream(Device::gpu);
    auto& enc = omarchy::get_command_encoder(s);
    auto buffer = omarchy::allocator().malloc(4096);
    auto* p = static_cast<omarchy::VulkanBuffer*>(buffer.ptr());
    std::promise<void> handler_started;
    auto started = handler_started.get_future();
    std::promise<void> release_handler;
    auto release = release_handler.get_future().share();
    enc.add_completed_handler([&handler_started, release]() {
      handler_started.set_value();
      release.wait();
    });
    enc.fill_buffer(p->buffer, 1, 4096);
    enc.commit();
    if (started.wait_for(std::chrono::seconds(2)) !=
        std::future_status::ready) {
      std::_Exit(39);
    }

    enc.fill_buffer(p->buffer, 2, 4096);
    enc.commit();
    const uint64_t completed_value = enc.last_submitted_completion();
    auto& device = omarchy::device(s.device.index);
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    uint64_t reached = 0;
    while (reached < completed_value &&
           std::chrono::steady_clock::now() < deadline) {
      VKX_CHECK(omarchy::vk::device_table().GetSemaphoreCounterValue(
          device.handle(), device.completions().semaphore(), &reached));
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (reached < completed_value) {
      release_handler.set_value();
      std::_Exit(40);
    }

    Event unsignaled{new_stream(Device::gpu)};
    unsignaled.set_value(42);
    unsignaled.wait(s);
    enc.fill_buffer(p->buffer, 3, 4096);
    const auto start = std::chrono::steady_clock::now();
    std::string message;
    try {
      enc.synchronize();
      release_handler.set_value();
      std::_Exit(41);
    } catch (const std::exception& ex) {
      message = ex.what();
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    release_handler.set_value();
    std::cout << "[child/completed_marker_behind_handler] " << message
              << std::endl;
    std::_Exit(
        elapsed < std::chrono::milliseconds(700) &&
                message.find("failed to advance") != std::string::npos
            ? 0
            : 42);
  }
  if (mode == "gpu_event_cpu_signal_progress") {
    setenv("MLX_OMARCHY_HANG_NO_PROGRESS_NS", "1", 1);
    setenv("MLX_OMARCHY_MAX_WALL_NS", "30000000000", 1);
    if (!cpu::is_available()) {
      std::_Exit(77);
    }
    Stream gpu_stream = new_stream(Device::gpu);
    Stream cpu_stream = new_stream(Device::cpu);
    Event event{gpu_stream};
    event.wait();

    constexpr int kSize = 1024;
    auto x = ones({kSize, kSize}, float32, gpu_stream);
    auto y = x;
    for (int i = 0; i < 3; ++i) {
      y = matmul(y, x, gpu_stream) / static_cast<float>(kSize);
    }
    y.eval();
    event.set_value(1);
    event.signal(cpu_stream);
    std::string wait_error;
    try {
      event.wait();
    } catch (const std::exception& ex) {
      wait_error = ex.what();
    }
    omarchy::get_command_encoder(gpu_stream).synchronize();
    const auto* values = y.data<float>();
    const bool correct =
        values[0] == 1.0f && values[y.size() - 1] == 1.0f;
    if (!wait_error.empty()) {
      std::cout << "[child/gpu_event_cpu_signal_progress] " << wait_error
                << std::endl;
      std::_Exit(43);
    }
    std::_Exit(correct ? 0 : 44);
  }
  if (mode == "long_matmul_progress") {
    setenv("MLX_OMARCHY_HANG_NO_PROGRESS_NS", "1", 1);
    Stream s = new_stream(Device::gpu);
    set_default_device(Device::gpu);
    set_default_stream(s);
    try {
      constexpr int kSize = 1024;
      auto x = ones({kSize, kSize}, float32, s);
      auto y = matmul(x, x, s);
      y.eval();
      omarchy::get_command_encoder(s).synchronize();
      const auto* values = y.data<float>();
      if (values[0] != kSize || values[y.size() - 1] != kSize) {
        std::_Exit(31);
      }
      std::cout << "[child/long_matmul_progress] valid matmul completed"
                << std::endl;
      std::_Exit(0);
    } catch (const std::exception& ex) {
      std::cout << "[child/long_matmul_progress] " << ex.what()
                << std::endl;
      std::_Exit(32);
    }
  }
  if (mode == "stuck_warning") {
    // Fresh process: the two env knobs are read lazily on this child's
    // first wait, so the 1 s stuck interval and the 3 s wall deadline are
    // what wait_for_timeline_progress caches. The waited value sits above
    // the newest reservation - nothing will ever signal it - which is the
    // silent shape the 2026-10-08 sushi first-forward park hid inside
    // 30-minute tickets: no counter motion, no watchdog STALL line, only
    // a waiter holding the timeline. The stuck-submit diagnostic must
    // fire its named warning (submit_stuck_warnings) BEFORE the wall
    // typed error arrives.
    ::setenv("MLX_OMARCHY_SUBMIT_STUCK_S", "1", 1);
    ::setenv("MLX_OMARCHY_MAX_WALL_NS", "3000000000", 1); // 3 s
    auto& dev = omarchy::device(0);
    Stream s = new_stream(Device::gpu);
    auto& enc = omarchy::get_command_encoder(s);
    auto buf = omarchy::allocator().malloc(4096);
    auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());
    enc.fill_buffer(p->buffer, 0x2a, 4096);
    enc.commit();
    enc.synchronize(); // healthy wait; must not warn (default interval)
    const uint64_t healthy =
        omarchy::trace::counters().submit_stuck_warnings.load();
    const uint64_t target = dev.completions().last_reserved() + 1000;
    bool threw = false;
    try {
      omarchy::wait_for_timeline_progress(
          dev.handle(),
          dev.completions().semaphore(),
          target,
          &dev.completions(),
          {},
          nullptr);
    } catch (const std::exception& ex) {
      threw = std::string(ex.what()).find("[omarchy]") !=
          std::string::npos;
    }
    omarchy::allocator().free(buf);
    if (!threw) {
      std::cout << "[child/stuck_warning] wall typed error missing\n";
      return 31;
    }
    const uint64_t warned =
        omarchy::trace::counters().submit_stuck_warnings.load();
    if (healthy != 0 || warned < 1) {
      std::cout << "[child/stuck_warning] warned=" << warned
                << " healthy=" << healthy << "\n";
      return 32;
    }
    std::cout << "[child/stuck_warning] warning fired before the wall"
                 " typed error\n";
    return 0;
  }
  return 126;
}

struct ChildRun {
  int code;
  bool timed_out;
  bool signaled;
};

ChildRun run_child(const char* mode, int timeout_s) {
  char self[4096];
  ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
  if (n <= 0) {
    return {127, false, false};
  }
  self[n] = '\0';
  pid_t pid = ::fork();
  if (pid == 0) {
    ::setenv("MLX_OMARCHY_TEST_CHILD", mode, 1);
    ::execl(self, self, static_cast<char*>(nullptr));
    ::_exit(127);
  }
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
  int status = 0;
  for (;;) {
    pid_t done = ::waitpid(pid, &status, WNOHANG);
    if (done == pid) {
      break;
    }
    if (std::chrono::steady_clock::now() > deadline) {
      ::kill(pid, SIGKILL);
      ::waitpid(pid, &status, 0);
      return {-1, true, false};
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (WIFEXITED(status)) {
    return {WEXITSTATUS(status), false, false};
  }
  return {-2, false, true};
}

} // namespace

int main(int argc, char** argv) {
  if (const char* mode = std::getenv("MLX_OMARCHY_TEST_CHILD")) {
    return run_child_scenario(mode);
  }
  doctest::Context ctx;
  ctx.applyCommandLine(argc, argv);
  return ctx.run();
}

TEST_CASE("discovery classifies supported and unsupported hardware") {
  // kMesaHoneykrispDriverId is pinned in device.h so the suite compiles
  // against Vulkan 1.3 headers that predate the enum entry.
  constexpr auto honeykrisp_id = omarchy::kMesaHoneykrispDriverId;

  SUBCASE("Apple GPU with Honeykrisp driver is supported") {
    auto s = classify(
        "Apple M1 (G13G B1)", 0x106b, VK_API_VERSION_1_3, honeykrisp_id, false);
    CHECK(s.supported);
    CHECK_FALSE(s.non_apple_dev);
    CHECK(s.reason.empty());
  }

  SUBCASE("M1 target is supported by driver identity alone") {
    // Real M1 receipt: vendor 0x10005, name "Apple M1",
    // driverID MESA_HONEYKRISP. Driver id is the authoritative signal.
    auto s =
        classify("Apple M1", 0x10005, VK_API_VERSION_1_3, honeykrisp_id, false);
    CHECK(s.supported);
    CHECK_FALSE(s.non_apple_dev);
  }

  SUBCASE("Apple vendor remains an alternate signal") {
    auto s = classify(
        "Apple M1 (G13G B1)",
        0x106b,
        VK_MAKE_API_VERSION(0, 1, 4, 0),
        -1,
        false);
    CHECK(s.supported);
  }
  SUBCASE("llvmpipe is refused without the development override") {
    auto s = classify(
        "llvmpipe (LLVM 19.1.0, 256 bits)",
        0x10005,
        VK_API_VERSION_1_3,
        VK_DRIVER_ID_MESA_LLVMPIPE,
        false);
    CHECK_FALSE(s.supported);
    CHECK(s.reason.find("MLX_OMARCHY_ALLOW_NON_APPLE") != std::string::npos);
  }

  SUBCASE("llvmpipe is accepted only as a development device") {
    auto s = classify(
        "llvmpipe (LLVM 19.1.0, 256 bits)",
        0x10005,
        VK_API_VERSION_1_3,
        VK_DRIVER_ID_MESA_LLVMPIPE,
        true);
    CHECK(s.supported);
    CHECK(s.non_apple_dev);
  }

  SUBCASE("devices below Vulkan 1.3 are refused") {
    auto s = classify(
        "Apple M1 (G13G B1)", 0x106b, VK_API_VERSION_1_2, honeykrisp_id, false);
    CHECK_FALSE(s.supported);
    CHECK(s.reason.find("Vulkan 1.3") != std::string::npos);
  }

  SUBCASE("non-Apple 1.3 device without override is refused") {
    auto s = classify(
        "NVIDIA GeForce RTX 4090", 0x10de, VK_API_VERSION_1_3, -1, false);
    CHECK_FALSE(s.supported);
  }
}

TEST_CASE("gpu is unavailable reports a reason and refuses selection") {
  if (gpu::is_available()) {
    skip(
        "a qualifying GPU is present; the refusal path is exercised by"
        " the classification tests and this test is meaningless there.");
    return;
  }
  CHECK(omarchy::device_count() == 0);
  CHECK_FALSE(omarchy::init_error().empty());
  CHECK_THROWS(omarchy::device(0));
  CHECK_THROWS(set_default_device(Device::gpu));
  CHECK_THROWS(new_stream(Device::gpu));
}

TEST_CASE("mx.gpu selects the Omarchy device") {
  if (!gpu::is_available()) {
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return;
  }

  CHECK(gpu::device_count() >= 1);
  set_default_device(Device::gpu);
  CHECK(default_device() == Device::gpu);
  CHECK(is_available(Device::gpu));

  const auto& info = gpu::device_info(0);
  auto name_it = info.find("device_name");
  REQUIRE(name_it != info.end());
  auto name = std::get<std::string>(name_it->second);
  CHECK_FALSE(name.empty());

  auto& dev = omarchy::device(0);
  CHECK(dev.handle() != VK_NULL_HANDLE);
  CHECK(dev.queue() != VK_NULL_HANDLE);
  std::cout << "[receipt] compute queue families expose "
            << dev.capabilities().queue_count << " queue(s)\n";

  // Stream creation routes through gpu::new_stream and allocates an encoder.
  Stream s = new_stream(Device::gpu);
  auto& encoder = omarchy::get_command_encoder(s);
  CHECK_FALSE(encoder.needs_commit());
  encoder.commit(); // no-op with nothing pending
}

TEST_CASE("allocator tracks buffers and reuses the cache") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto& alloc = omarchy::allocator();
  size_t before = alloc.get_active_memory();

  auto* a = static_cast<omarchy::VulkanBuffer*>(alloc.malloc(4096).ptr());
  auto* b = static_cast<omarchy::VulkanBuffer*>(alloc.malloc(4096).ptr());
  REQUIRE(a->data != nullptr);
  REQUIRE(b->data != nullptr);
  CHECK(alloc.get_active_memory() == before + 2 * 4096);
  alloc.free(allocator::Buffer{a});
  alloc.free(allocator::Buffer{b});
  CHECK(alloc.get_active_memory() == before);

  // A same-size malloc after frees must succeed and keep accounting exact
  // (the page came from cache or the device).
  auto* c = static_cast<omarchy::VulkanBuffer*>(alloc.malloc(4096).ptr());
  REQUIRE(c->data != nullptr);
  CHECK(alloc.get_active_memory() == before + 4096);
  alloc.free(allocator::Buffer{c});
  CHECK(alloc.get_active_memory() == before);
}

TEST_CASE("memory and wired limits drive cache release, never fake failures") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto& alloc = omarchy::allocator();

  // Wired limit round-trips through the allocator (dflash acquire/restore
  // contract).
  CHECK(alloc.set_wired_limit(0) == alloc.get_wired_limit());
  CHECK(alloc.set_wired_limit(1u << 30) == 0);
  CHECK(alloc.set_wired_limit(0) == (1u << 30));

  const size_t saved_limit = alloc.get_memory_limit();
  alloc.set_cache_limit(64u << 20);
  alloc.clear_cache();
  const size_t before = alloc.get_active_memory();

  // Cache one 4 MiB block, then park the limit just above active memory so
  // any fresh allocation exceeds it.
  auto* blk = static_cast<omarchy::VulkanBuffer*>(alloc.malloc(4u << 20).ptr());
  REQUIRE(blk != nullptr);
  alloc.free(allocator::Buffer{blk});
  REQUIRE(alloc.get_cache_memory() == (4u << 20));
  alloc.set_memory_limit(before + 1);

  // Exceeding the limit releases the reuse cache (Metal semantics); the
  // fresh allocation itself still succeeds — the limit never fabricates a
  // driver failure the driver did not report.
  auto* fresh = static_cast<omarchy::VulkanBuffer*>(alloc.malloc(8u << 20).ptr());
  REQUIRE(fresh != nullptr);
  CHECK(alloc.get_cache_memory() == 0);
  alloc.free(allocator::Buffer{fresh});
  alloc.clear_cache();

  // A raised wired limit lifts the effective ceiling: the same fresh
  // allocation no longer forces a cache release.
  auto* blk2 = static_cast<omarchy::VulkanBuffer*>(alloc.malloc(4u << 20).ptr());
  REQUIRE(blk2 != nullptr);
  alloc.free(allocator::Buffer{blk2});
  REQUIRE(alloc.get_cache_memory() == (4u << 20));
  alloc.set_memory_limit(before + 1);
  CHECK(alloc.set_wired_limit(64ull << 30) == 0);
  auto* fresh2 =
      static_cast<omarchy::VulkanBuffer*>(alloc.malloc(8u << 20).ptr());
  REQUIRE(fresh2 != nullptr);
  CHECK(alloc.get_cache_memory() == (4u << 20));
  alloc.free(allocator::Buffer{fresh2});

  alloc.clear_cache();
  alloc.set_memory_limit(saved_limit);
  CHECK(alloc.set_wired_limit(0) == (64ull << 30));
  CHECK(alloc.get_active_memory() == before);
}

TEST_CASE("get/set cache_limit round-trips and exposes the default") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto& alloc = omarchy::allocator();
  // Default cache limit is non-zero (the constructor wires it to the
  // working-set ceiling, 0.95 * total_memory on Honeykrisp).
  CHECK(alloc.get_cache_limit() > 0);
  const size_t prev = alloc.get_cache_limit();
  CHECK(alloc.set_cache_limit(64u << 20) == prev);
  CHECK(alloc.get_cache_limit() == (64u << 20));
  // Restore the prior value.
  alloc.set_cache_limit(prev);
  CHECK(alloc.get_cache_limit() == prev);
}

TEST_CASE("malloc releases cache before reaching the working-set ceiling") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto& alloc = omarchy::allocator();

  // Bring the working-set ceiling down to a known value so the gate
  // fires on a predictable size. Save and restore the prior value.
  const size_t saved_gc = alloc.get_gc_limit();
  const size_t saved_cache = alloc.get_cache_limit();
  const size_t saved_memory = alloc.get_memory_limit();
  // Pick a small ceiling so we can drive active + cache over it
  // without carving through the device's real heap.
  const size_t ceiling = 8u << 20;
  alloc.set_gc_limit(ceiling);
  alloc.set_cache_limit(ceiling);
  alloc.clear_cache();

  const size_t before = alloc.get_active_memory();
  // Fill the cache with one block.
  auto* blk = static_cast<omarchy::VulkanBuffer*>(
      alloc.malloc(4u << 20).ptr());
  REQUIRE(blk != nullptr);
  alloc.free(allocator::Buffer{blk});
  REQUIRE(alloc.get_cache_memory() >= (4u << 20));

  // A fresh allocation that would push active + cache + size past
  // the ceiling must release the cache before vkAllocateMemory; the
  // malloc itself still succeeds (the gate only releases, it never
  // fabricates a driver failure).
  auto* fresh = static_cast<omarchy::VulkanBuffer*>(
      alloc.malloc(8u << 20).ptr());
  REQUIRE(fresh != nullptr);
  CHECK(alloc.get_cache_memory() < (4u << 20));
  alloc.free(allocator::Buffer{fresh});

  alloc.clear_cache();
  alloc.set_cache_limit(saved_cache);
  alloc.set_gc_limit(saved_gc);
  alloc.set_memory_limit(saved_memory);
  CHECK(alloc.get_active_memory() == before);
}

TEST_CASE(
    "malloc retries once with a cleared cache on VK_ERROR_OUT_OF_DEVICE_MEMORY") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto& alloc = omarchy::allocator();
  alloc.clear_cache();
  auto* cached =
      static_cast<omarchy::VulkanBuffer*>(alloc.malloc(2u << 20).ptr());
  REQUIRE(cached != nullptr);
  alloc.free(allocator::Buffer{cached});
  REQUIRE(alloc.get_cache_memory() >= (2u << 20));

  setenv("MLX_OMARCHY_TEST_OOM_REMAINING", "1", 1);
  const size_t before_active = alloc.get_active_memory();
  auto* buf =
      static_cast<omarchy::VulkanBuffer*>(alloc.malloc(4u << 20).ptr());
  REQUIRE(buf != nullptr);
  CHECK(buf->memory != VK_NULL_HANDLE);
  CHECK(alloc.get_active_memory() == before_active + (4u << 20));
  CHECK(alloc.get_cache_memory() == 0);
  alloc.free(allocator::Buffer{buf});
  alloc.clear_cache();
  unsetenv("MLX_OMARCHY_TEST_OOM_REMAINING");
}

TEST_CASE(
    "cache stays bounded under a shape-changing alloc/free loop") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto& alloc = omarchy::allocator();
  const size_t saved_gc = alloc.get_gc_limit();
  const size_t saved_cache = alloc.get_cache_limit();
  const size_t saved_memory = alloc.get_memory_limit();
  const size_t ceiling = 32u << 20;
  alloc.set_gc_limit(ceiling);
  alloc.set_cache_limit(ceiling);
  alloc.clear_cache();

  // Requests grow by 64 KiB per iteration; power-of-two bins reuse
  // storage across neighboring shapes while the GC ceiling bounds the pool.
  std::vector<omarchy::VulkanBuffer*> live;
  for (int i = 0; i < 64; ++i) {
    size_t sz = (1u << 20) + static_cast<size_t>(i) * (64u << 10);
    auto* b = static_cast<omarchy::VulkanBuffer*>(alloc.malloc(sz).ptr());
    REQUIRE(b != nullptr);
    size_t bin = 1u << 20;
    while (bin < sz) bin *= 2;
    CHECK(b->size == bin);
    if (!live.empty()) {
      alloc.free(allocator::Buffer{live.back()});
      live.pop_back();
    }
    live.push_back(b);
  }
  // Cache must be at or below the cap.
  CHECK(alloc.get_cache_memory() <= ceiling);
  for (auto* b : live) {
    alloc.free(allocator::Buffer{b});
  }
  alloc.clear_cache();
  alloc.set_cache_limit(saved_cache);
  alloc.set_gc_limit(saved_gc);
  alloc.set_memory_limit(saved_memory);
  CHECK(alloc.get_gc_limit() == saved_gc);
}

TEST_CASE(
    "freeing an in-flight buffer drops accounting but quarantines reuse") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto& alloc = omarchy::allocator();
  Stream s = new_stream(Device::gpu);
  auto& encoder = omarchy::get_command_encoder(s);

  constexpr size_t kBytes = 1 << 16;
  auto src = alloc.malloc(kBytes);
  auto dst = alloc.malloc(kBytes);
  auto* src_buf = static_cast<omarchy::VulkanBuffer*>(src.ptr());
  auto* dst_buf = static_cast<omarchy::VulkanBuffer*>(dst.ptr());
  REQUIRE(src_buf->data != nullptr);
  REQUIRE(dst_buf->data != nullptr);

  // Record (do not submit) a dispatch reading src and register the
  // buffer with the open batch the way primitive eval does: a
  // non-owning array view feeds encoder.add_temporary, which stamps the
  // buffer kPendingCompletion and queues it for the submit-time stamp.
  array src_view(
      Shape{static_cast<int>(kBytes / sizeof(float))},
      float32,
      nullptr,
      {});
  src_view.set_data(
      allocator::Buffer{src_buf},
      src_view.size(),
      src_view.strides(),
      src_view.flags(),
      0,
      [](allocator::Buffer) {});
  encoder.add_temporary(src_view);
  CHECK(src_buf->completion == omarchy::kPendingCompletion);
  encoder.copy_buffer(src_buf->buffer, dst_buf->buffer, kBytes);

  size_t active_with_src = alloc.get_active_memory();
  size_t cache_before = alloc.get_cache_memory();
  alloc.free(src);
  // Accounting drops at free() (upstream Metal semantics) even though
  // the recorded dispatch still references the buffer.
  CHECK(alloc.get_active_memory() == active_with_src - kBytes);
  // The buffer is quarantined, not recycled: the reuse cache must not
  // have grown, so the next malloc cannot alias the recorded dispatch.
  CHECK(alloc.get_cache_memory() == cache_before);
  auto* replacement = static_cast<omarchy::VulkanBuffer*>(
      alloc.malloc(kBytes).ptr());
  CHECK(replacement != src_buf);

  // Submit, synchronize (drain V), then push one more submission so the
  // drain runs through V+1: cleanup for the buffer's generation has run
  // and release_quarantine recycles it into the reuse cache.
  encoder.commit();
  encoder.synchronize();
  encoder.fill_buffer(dst_buf->buffer, 0, 4);
  encoder.commit();
  encoder.synchronize();
  CHECK(alloc.get_cache_memory() >= cache_before + kBytes);
  alloc.free(allocator::Buffer{replacement});
  alloc.free(dst);
}

TEST_CASE("re-recorded buffers stay quarantined until completion") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto& alloc = omarchy::allocator();
  Stream s = new_stream(Device::gpu);
  auto& encoder = omarchy::get_command_encoder(s);
  constexpr size_t kBytes = 1 << 16;
  constexpr uint32_t kPattern = 0x1234abcd;
  auto src = alloc.malloc(kBytes);
  auto dst = alloc.malloc(kBytes);
  auto* src_buf = static_cast<omarchy::VulkanBuffer*>(src.ptr());
  auto* dst_buf = static_cast<omarchy::VulkanBuffer*>(dst.ptr());
  array src_view(
      Shape{static_cast<int>(kBytes / sizeof(float))}, float32, nullptr, {});
  src_view.set_data(
      src, src_view.size(), src_view.strides(), src_view.flags(), 0,
      [](allocator::Buffer) {});
  encoder.add_temporary(src_view);
  encoder.fill_buffer(src_buf->buffer, kPattern, kBytes);
  encoder.copy_buffer(src_buf->buffer, dst_buf->buffer, kBytes);
  encoder.commit();
  encoder.synchronize();

  encoder.add_temporary(src_view);
  encoder.copy_buffer(src_buf->buffer, dst_buf->buffer, kBytes);
  const size_t cache_before = alloc.get_cache_memory();
  alloc.free(src);
  CHECK(alloc.get_cache_memory() == cache_before);
  auto replacement = alloc.malloc(kBytes);
  CHECK(replacement.ptr() != src.ptr());
  const size_t cache_before_completion = alloc.get_cache_memory();
  encoder.commit();
  encoder.synchronize();
  encoder.fill_buffer(dst_buf->buffer, kPattern, 4);
  encoder.commit();
  encoder.synchronize();
  CHECK(alloc.get_cache_memory() == cache_before_completion + kBytes);
  const auto* words = static_cast<const uint32_t*>(dst_buf->data);
  CHECK(std::all_of(words, words + kBytes / sizeof(uint32_t),
                    [](uint32_t word) { return word == kPattern; }));
  alloc.free(replacement);
  alloc.free(dst);
}

TEST_CASE("dispatch bindings stamp their owners against the open batch") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto& alloc = omarchy::allocator();
  Stream s = new_stream(Device::gpu);
  auto& encoder = omarchy::get_command_encoder(s);

  constexpr size_t kBytes = 1 << 16;
  constexpr uint32_t kFloats = kBytes / sizeof(float);
  auto src = alloc.malloc(kBytes);
  auto dst = alloc.malloc(kBytes);
  auto* src_buf = static_cast<omarchy::VulkanBuffer*>(src.ptr());
  auto* dst_buf = static_cast<omarchy::VulkanBuffer*>(dst.ptr());

  // A dispatch whose binding is the buffer's only reference: no array
  // feeds add_temporary here. dispatch_compute must stamp the binding
  // owner, or a free before submit hands the live VkBuffer handle back
  // to the cache or the destroyer while the batch is still open.
  omarchy::ComputeParams params;
  params.count = kFloats;
  params.output_size = kFloats;
  std::array<omarchy::ComputeBinding, 1> bindings{
      omarchy::ComputeBinding{src_buf->buffer, 0, kBytes, src_buf}};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::FillF32,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(kFloats));
  CHECK(src_buf->completion == omarchy::kPendingCompletion);

  size_t cache_before = alloc.get_cache_memory();
  alloc.free(src);
  CHECK(alloc.get_cache_memory() == cache_before);

  // Submit (drain V), then push one more submission so the drain runs
  // through V+1 and release_quarantine recycles the buffer.
  encoder.commit();
  encoder.synchronize();
  encoder.fill_buffer(dst_buf->buffer, 0, 4);
  encoder.commit();
  encoder.synchronize();
  CHECK(alloc.get_cache_memory() >= cache_before + kBytes);
  alloc.free(dst);
}

TEST_CASE("wave levels schedule hazards into strictly ordered waves") {
  using omarchy::TrackedRange;
  using omarchy::WaveNode;
  // The scheduler treats VkBuffer as an opaque key: fake handles suffice.
  auto buf = [](uintptr_t v) {
    return reinterpret_cast<VkBuffer>(v);
  };
  auto rng = [](VkBuffer b, uint64_t off, uint64_t end) {
    return TrackedRange{b, static_cast<VkDeviceSize>(off),
                        static_cast<VkDeviceSize>(end)};
  };
  // Owning node builder: WaveNode holds non-owning spans, so the range
  // vectors must outlive the wave_levels call (same contract as
  // emit_pending, whose spans point into the buffered PendingNodes).
  struct Owned {
    std::vector<TrackedRange> reads;
    std::vector<TrackedRange> writes;
  };
  auto to_nodes = [](std::vector<Owned>& owned) {
    std::vector<WaveNode> nodes;
    nodes.reserve(owned.size());
    for (auto& o : owned) {
      nodes.push_back({o.reads, o.writes});
    }
    return nodes;
  };
  // Property checker: (a) no hazard pair shares a wave, (b) every hazard
  // edge crosses waves (later level > earlier level), (c) each level is
  // the earliest legal wave: 0 without hazard sources, else
  // 1 + max(level of hazard sources).
  auto check = [](std::span<const WaveNode> nodes,
                  std::span<const uint32_t> level) {
    auto overlaps = [](const TrackedRange& a, const TrackedRange& b) {
      return a.buffer == b.buffer && a.offset < b.end && b.offset < a.end;
    };
    for (size_t i = 0; i < nodes.size(); ++i) {
      uint32_t expect = 0;
      for (size_t j = 0; j < i; ++j) {
        bool raw = false, war = false, waw = false;
        for (const auto& r : nodes[i].reads) {
          for (const auto& w : nodes[j].writes) {
            raw = raw || overlaps(r, w);
          }
        }
        for (const auto& w : nodes[i].writes) {
          for (const auto& tw : nodes[j].writes) {
            waw = waw || overlaps(w, tw);
          }
          for (const auto& tr : nodes[j].reads) {
            war = war || overlaps(w, tr);
          }
        }
        bool hazard = raw || waw || war;
        if (hazard) {
          REQUIRE(level[i] > level[j]);
        }
        if (hazard) {
          expect = std::max(expect, level[j] + 1);
        }
      }
      CHECK(level[i] == expect);
      for (size_t k = i + 1; k < nodes.size(); ++k) {
        if (level[i] == level[k]) {
          // Mirror the hazard test on the ordered pair (i earlier).
          bool hazard = false;
          for (const auto& r : nodes[k].reads) {
            for (const auto& w : nodes[i].writes) {
              hazard = hazard || overlaps(r, w);
            }
          }
          for (const auto& w : nodes[k].writes) {
            for (const auto& tw : nodes[i].writes) {
              hazard = hazard || overlaps(w, tw);
            }
            for (const auto& tr : nodes[i].reads) {
              hazard = hazard || overlaps(w, tr);
            }
          }
          CHECK_FALSE(hazard);
        }
      }
    }
  };

  // Read-read sharing stays in one wave.
  {
    VkBuffer b = buf(0x1000);
    std::vector<Owned> owned = {
        {{}, {rng(b, 0, 64)}},   // producer writes b
        {{rng(b, 0, 64)}, {}},   // consumer 1 reads b
        {{rng(b, 0, 64)}, {}},   // consumer 2 reads b
    };
    std::vector<WaveNode> nodes = to_nodes(owned);
    auto level = omarchy::wave_levels(nodes);
    const std::vector<uint32_t> want = {0, 1, 1};
    CHECK(level == want);
    check(nodes, level);
  }
  // RAW, WAR and WAW each force one wave of separation.
  {
    VkBuffer b = buf(0x2000);
    std::vector<Owned> owned = {
        {{rng(b, 0, 64)}, {}},   // reader
        {{}, {rng(b, 0, 64)}},   // WAR writer
        {{}, {rng(b, 0, 64)}},   // WAW writer
        {{rng(b, 0, 64)}, {}},   // RAW reader
    };
    std::vector<WaveNode> nodes = to_nodes(owned);
    auto level = omarchy::wave_levels(nodes);
    const std::vector<uint32_t> want = {0, 1, 2, 3};
    CHECK(level == want);
    check(nodes, level);
  }
  // Disjoint ranges of one buffer do not hazard; independent work hoists
  // ahead of the dependent tail (the wave win: 4 nodes, 2 waves).
  {
    VkBuffer b = buf(0x3000);
    VkBuffer c = buf(0x4000);
    std::vector<Owned> owned = {
        {{}, {rng(b, 0, 100)}},        // A writes b[0,100)
        {{}, {rng(c, 0, 100)}},        // B writes c (independent)
        {{rng(b, 0, 100)}, {}},        // C reads b (needs A)
        {{}, {rng(b, 200, 300)}},      // D writes b[200,300) (independent of A)
    };
    std::vector<WaveNode> nodes = to_nodes(owned);
    auto level = omarchy::wave_levels(nodes);
    const std::vector<uint32_t> want = {0, 0, 1, 0};
    CHECK(level == want);
    check(nodes, level);
  }
  // Diamond: one producer, two consumers, one join writer closes at the
  // wave after both consumers (WAW on the producer, WAR on the reads).
  {
    VkBuffer b = buf(0x5000);
    std::vector<Owned> owned = {
        {{}, {rng(b, 0, 64)}},
        {{rng(b, 0, 64)}, {}},
        {{rng(b, 0, 64)}, {}},
        {{}, {rng(b, 0, 64)}},
    };
    std::vector<WaveNode> nodes = to_nodes(owned);
    auto level = omarchy::wave_levels(nodes);
    const std::vector<uint32_t> want = {0, 1, 1, 2};
    CHECK(level == want);
    check(nodes, level);
  }
}

TEST_CASE("buffer round trip through the Vulkan encoder") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = new_stream(Device::gpu);
  auto& encoder = omarchy::get_command_encoder(s);

  constexpr size_t kBytes = 1 << 16;
  auto src = omarchy::allocator().malloc(kBytes);
  auto dst = omarchy::allocator().malloc(kBytes);
  REQUIRE(src.raw_ptr() != nullptr);
  REQUIRE(dst.raw_ptr() != nullptr);

  auto* src_buf = static_cast<omarchy::VulkanBuffer*>(src.ptr());
  auto* dst_buf = static_cast<omarchy::VulkanBuffer*>(dst.ptr());

  auto* src_bytes = static_cast<uint8_t*>(src_buf->data);
  auto* dst_bytes = static_cast<uint8_t*>(dst_buf->data);
  for (size_t i = 0; i < kBytes; ++i) {
    src_bytes[i] = static_cast<uint8_t>(i % 251);
  }
  std::memset(dst_bytes, 0, kBytes);

  uint64_t submissions_before =
      omarchy::trace::counters().vk_submissions.load();
  uint64_t copies_before = omarchy::trace::counters().vk_buffer_copies.load();

  encoder.copy_buffer(src_buf->buffer, dst_buf->buffer, kBytes);
  encoder.commit();
  encoder.synchronize();

  CHECK(
      omarchy::trace::counters().vk_submissions.load() ==
      submissions_before + 1);
  CHECK(
      omarchy::trace::counters().vk_buffer_copies.load() == copies_before + 1);

  size_t mismatches = 0;
  for (size_t i = 0; i < kBytes; ++i) {
    if (src_bytes[i] != dst_bytes[i]) {
      mismatches++;
    }
  }
  CHECK(mismatches == 0);
}

TEST_CASE("events synchronize between streams") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream a = new_stream(Device::gpu);
  Stream b = new_stream(Device::gpu);

  Event e{a};
  e.set_value(1);
  CHECK_FALSE(e.is_signaled());

  e.signal(a); // asynchronous signal submission
  e.wait(); // host wait on the device timeline semaphore
  CHECK(e.is_signaled());

  // A wait recorded on another stream must not hang and must be honored by
  // that stream's submissions.
  auto& enc_b = omarchy::get_command_encoder(b);
  e.wait(b);
  auto buf = omarchy::allocator().malloc(4096);
  auto* buf_ptr = static_cast<omarchy::VulkanBuffer*>(buf.ptr());
  enc_b.fill_buffer(buf_ptr->buffer, 0, 4096);
  enc_b.commit();
  enc_b.synchronize();
  CHECK(e.is_signaled());
  omarchy::allocator().free(buf);
}

TEST_CASE("queued event wait survives destruction of the Event") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream a = new_stream(Device::gpu);
  Stream b = new_stream(Device::gpu);
  {
    Event e{a};
    e.set_value(1);
    e.signal(a);
    e.wait(b); // queued on encoder b as a raw wait before the fix
  } // e destroyed here: the encoder must keep the semaphore alive
  auto& enc_b = omarchy::get_command_encoder(b);
  auto buf = omarchy::allocator().malloc(4096);
  auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());
  enc_b.fill_buffer(p->buffer, 7, 4096);
  enc_b.commit(); // must never submit a destroyed VkSemaphore
  enc_b.synchronize();
  CHECK(*static_cast<uint32_t*>(p->data) == 7u);
  omarchy::allocator().free(buf);
}

TEST_CASE("cross-stream event ordering is preserved") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream a = new_stream(Device::gpu);
  Stream b = new_stream(Device::gpu);

  constexpr size_t kBytes = 1 << 16;
  auto src = omarchy::allocator().malloc(kBytes);
  auto mid = omarchy::allocator().malloc(kBytes);
  auto dst = omarchy::allocator().malloc(kBytes);
  auto* src_bytes = static_cast<uint8_t*>(
      static_cast<omarchy::VulkanBuffer*>(src.ptr())->data);
  auto* mid_bytes = static_cast<uint8_t*>(
      static_cast<omarchy::VulkanBuffer*>(mid.ptr())->data);
  auto* dst_bytes = static_cast<uint8_t*>(
      static_cast<omarchy::VulkanBuffer*>(dst.ptr())->data);
  for (size_t i = 0; i < kBytes; ++i) {
    src_bytes[i] = static_cast<uint8_t>(i % 251);
  }
  std::memset(mid_bytes, 0, kBytes);
  std::memset(dst_bytes, 0, kBytes);

  // MLX order: the consumer records its wait before the producer signals.
  Event e{a};
  e.set_value(1);
  e.wait(b);

  auto& enc_a = omarchy::get_command_encoder(a);
  auto& enc_b = omarchy::get_command_encoder(b);
  auto* src_buf = static_cast<omarchy::VulkanBuffer*>(src.ptr());
  auto* mid_buf = static_cast<omarchy::VulkanBuffer*>(mid.ptr());
  auto* dst_buf = static_cast<omarchy::VulkanBuffer*>(dst.ptr());

  enc_a.copy_buffer(src_buf->buffer, mid_buf->buffer, kBytes);
  e.signal(a); // signal submission follows the copy on stream a
  enc_b.copy_buffer(mid_buf->buffer, dst_buf->buffer, kBytes);
  enc_b.commit(); // gated on the event by the queued timeline wait
  enc_b.synchronize();

  size_t mismatches = 0;
  for (size_t i = 0; i < kBytes; ++i) {
    if (src_bytes[i] != dst_bytes[i]) {
      mismatches++;
    }
  }
  CHECK(mismatches == 0);

  omarchy::allocator().free(src);
  omarchy::allocator().free(mid);
  omarchy::allocator().free(dst);
}

TEST_CASE("handler-only commit runs handlers after prior queue work") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = new_stream(Device::gpu);
  auto& enc = omarchy::get_command_encoder(s);
  auto buf = omarchy::allocator().malloc(4096);
  auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());

  // Prior GPU work, submitted asynchronously.
  enc.fill_buffer(p->buffer, 0x5a, 4096);

  // A submission with only a completion handler (the fill_zero edge: no
  // aligned 4-byte word to record). It must still be submitted and ordered
  // after the fill.
  std::atomic<bool> ran{false};
  enc.add_completed_handler([&ran]() { ran.store(true); });
  enc.commit();
  enc.synchronize();
  // synchronize() joins the dispatcher drain: the handler must have run
  // before it returns (pins the drain-join contract).
  CHECK(ran.load());
  CHECK(*static_cast<uint32_t*>(p->data) == 0x5au);
  omarchy::allocator().free(buf);
}

TEST_CASE("wait joins a handler the background drain is still running") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = new_stream(Device::gpu);
  auto& enc = omarchy::get_command_encoder(s);

  // Race under test: the background dispatcher thread takes a submission
  // out of pending_ and starts its handler while another thread calls
  // wait() for the same value. The caller's inline drain then finds an
  // empty pending_, and wait() must still not return until the handler
  // finishes (the drained_value_ join). Each round blocks the handler
  // until this test releases it, so a wait() that returns first is
  // detected by state, not by timing.
  constexpr int kRaces = 20;
  for (int i = 0; i < kRaces; ++i) {
    std::promise<void> started_promise;
    std::future<void> started = started_promise.get_future();
    std::promise<void> release_promise;
    std::future<void> release = release_promise.get_future();
    std::atomic<bool> handler_finished{false};
    std::atomic<bool> sync_returned{false};
    std::atomic<bool> finished_at_return{false};
    std::string sync_error;

    enc.add_completed_handler([&]() {
      started_promise.set_value();
      release.wait();
      handler_finished.store(true);
    });
    enc.commit();

    // The handler signals only after the dispatcher moved the entry out
    // of pending_, so every synchronize() below is guaranteed to hit the
    // raced path: the caller-side drain finds nothing to run.
    auto started_status = started.wait_for(std::chrono::seconds(10));
    CHECK(started_status == std::future_status::ready);
    if (started_status != std::future_status::ready) {
      release_promise.set_value();
      enc.synchronize();
      continue;
    }

    std::thread waiter([&]() {
      try {
        enc.synchronize();
      } catch (const std::exception& ex) {
        sync_error = ex.what();
      }
      // Handler state at the instant synchronize returned.
      finished_at_return.store(handler_finished.load());
      sync_returned.store(true);
    });

    // Bounded detection window, not the proof: a broken wait() returns
    // immediately once pending_ is empty, so sync_returned lands here
    // within microseconds of the thread starting. The proof is the state
    // check below: synchronize returned while the blocked handler
    // provably had not finished (this thread has not released it yet).
    auto window =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (!sync_returned.load() && std::chrono::steady_clock::now() < window) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Unblock the handler and join. On correct code the waiter is still
    // inside the drained_value_ join here; the release is what lets it
    // return.
    release_promise.set_value();
    waiter.join();

    CHECK(sync_error.empty());
    CHECK(finished_at_return.load());
    CHECK(handler_finished.load());
  }
}

TEST_CASE(
    "wait serializes against the background drain to preserve handler order") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = new_stream(Device::gpu);
  auto& enc = omarchy::get_command_encoder(s);
  auto buf = omarchy::allocator().malloc(64);
  auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());

  // V1's handler blocks until the test releases it. V2 is enqueued and
  // waited on from a second thread while V1 is still executing on the
  // background dispatcher. drain_through is serialized end to end
  // through drain_mutex_, so the waiter's drain cannot run V2's handler
  // before V1's drain finishes.
  std::promise<void> v1_started_promise;
  std::future<void> v1_started = v1_started_promise.get_future();
  std::promise<void> v1_release_promise;
  std::future<void> v1_release = v1_release_promise.get_future();
  std::atomic<bool> v1_done{false};
  std::atomic<bool> wait_returned{false};
  std::atomic<bool> wait_returned_after_v1{false};
  std::string wait_error;

  enc.fill_buffer(p->buffer, 0xa5, 64);
  enc.add_completed_handler([&]() {
    v1_started_promise.set_value();
    v1_release.wait();
    v1_done.store(true);
  });
  // This test targets the CompletionDispatcher's drain serialization, so
  // the submission must be on the queue immediately: commit_now() (plain
  // commit() defers into the open batch by design).
  enc.commit();

  auto v1_status = v1_started.wait_for(std::chrono::seconds(10));
  CHECK(v1_status == std::future_status::ready);
  if (v1_status != std::future_status::ready) {
    v1_release_promise.set_value();
    enc.synchronize();
    omarchy::allocator().free(buf);
    return;
  }

  // Submit V2 (handler-only) on this thread, then synchronize it from a
  // second thread. The waiter enters drain_through(V2) and must block
  // until V1's drain releases drain_mutex_.
  enc.add_completed_handler([]() {});
  enc.commit();

  std::thread waiter([&]() {
    try {
      enc.synchronize();
    } catch (const std::exception& ex) {
      wait_error = ex.what();
    }
    wait_returned_after_v1.store(v1_done.load());
    wait_returned.store(true);
  });

  // Bounded window for the waiter to reach drain_through. A broken
  // serializer returns here within microseconds: V2 was enqueued, the
  // GPU signaled it, and the waiter's drain_through would find nothing
  // blocking it.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK_FALSE(wait_returned.load());

  // Release V1; the waiter now makes progress and synchronize returns.
  v1_release_promise.set_value();
  waiter.join();

  CHECK(wait_error.empty());
  CHECK(wait_returned.load());
  CHECK(wait_returned_after_v1.load());
  omarchy::allocator().free(buf);
}

TEST_CASE(
    "freed arrays release promptly while buffers stay quarantined in"
    " flight") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = new_stream(Device::gpu);
  auto& enc = omarchy::get_command_encoder(s);
  auto& alloc = omarchy::allocator();

  auto arr = zeros({1024}, float32);
  arr.eval();
  std::weak_ptr<void> observed = arr.data_shared_ptr();
  auto* arr_buf = static_cast<omarchy::VulkanBuffer*>(arr.buffer().ptr());

  // A large fill stretches execution so the submission is still running
  // while the checks below run. (Gating via an unsatisfied timeline wait
  // would stall the queue: the satisfying signal itself travels on the
  // same single queue.)
  constexpr VkDeviceSize kBigBytes = 256ull << 20;
  auto scratch = alloc.malloc(kBigBytes);
  auto* scratch_buf = static_cast<omarchy::VulkanBuffer*>(scratch.ptr());
  enc.fill_buffer(scratch_buf->buffer, 0x11, kBigBytes);
  enc.fill_buffer(arr_buf->buffer, 0x33, 4096);
  enc.add_temporary(arr); // register the buffer with the open batch
  enc.commit(); // in flight

  size_t cache_before = alloc.get_cache_memory();
  // Drop the caller's reference while work is queued (arrays have no
  // default ctor: reassign to a fresh array). Array lifetime is
  // independent of in-flight work: the Data releases immediately, while
  // the raw buffer is quarantined out of the reuse cache until a
  // generation after its submission drains.
  arr = zeros({1}, float32);
  CHECK(observed.expired());
  CHECK(alloc.get_cache_memory() == cache_before);

  enc.synchronize(); // bounded completion wait; joins handler execution
  // The submission's execution fence gates the release now: the block
  // recycles on the first drain after the fence signals (the timeline
  // semaphore alone does not prove the batch's writes are done). Poll
  // briefly for that release instead of demanding it synchronously.
  bool released = false;
  for (int i = 0; i < 200 && !released; ++i) {
    released = alloc.get_cache_memory() >= cache_before + 4096;
    if (!released) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  CHECK(released);

  alloc.free(scratch);
}

TEST_CASE(
    "eager per-node commits reuse ring slots without host joins") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = new_stream(Device::gpu);
  auto& enc = omarchy::get_command_encoder(s);
  constexpr size_t kBytes = 16u << 20;
  auto buf = omarchy::allocator().malloc(kBytes);
  auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());

  // Explicit-commit contract: every commit submits. The in-flight ring does
  // not host-join the previous submission as the pre-ring encoder did
  // (that join was the dominant decode cost in the 2026-09-02 profile).
  // Ordering still holds end to end (each node carries its full barrier
  // pair), so the final bytes carry the last iteration's value.
  constexpr int kIters = 8;
  uint64_t submissions_before =
      omarchy::trace::counters().vk_submissions.load();
  for (int i = 0; i < kIters; ++i) {
    enc.fill_buffer(p->buffer, (i % 2) ? 0x5a5a5a5au : 0xa5a5a5a5u, kBytes);
    enc.commit();
  }
  CHECK(
      omarchy::trace::counters().vk_submissions.load() ==
      submissions_before + kIters);
  enc.synchronize();
  CHECK(enc.synchronized());
  const uint32_t expected = ((kIters - 1) % 2) ? 0x5a5a5a5au : 0xa5a5a5a5u;
  const auto* words = static_cast<const uint32_t*>(p->data);
  size_t mismatches = 0;
  for (size_t i = 0; i < kBytes / sizeof(uint32_t); ++i) {
    if (words[i] != expected) {
      mismatches++;
    }
  }
  CHECK(mismatches == 0);
  omarchy::allocator().free(buf);
}

TEST_CASE("Event::wait joins completion handlers without polling") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = new_stream(Device::gpu);
  auto buf = omarchy::allocator().malloc(8);
  auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());
  std::memset(p->data, 0, 8);
  auto& enc = omarchy::get_command_encoder(s);
  enc.fill_buffer(p->buffer, 0x2a, 8);
  uint8_t* bytes = static_cast<uint8_t*>(p->data);
  enc.add_completed_handler([&bytes]() { bytes[0] = 0x5a; });
  enc.commit();
  Event e{s};
  e.set_value(1);
  e.signal(s);
  e.wait(); // direct host wait; must join the handler boundary
  CHECK(bytes[0] == 0x5a);
  omarchy::allocator().free(buf);
}

TEST_CASE("sub-word zero fill then copy stays zero without syncs") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  set_default_device(Device::gpu);
  // 3 bytes: sub-word, so fill_zero writes host-side edges; the follow-up
  // copy must never observe stale bytes despite no intermediate sync.
  auto a = zeros({3}, uint8);
  a.eval();
  auto b = copy(a);
  b.eval();
  synchronize(default_stream(default_device()));
  const auto* d = b.data<uint8_t>();
  for (int i = 0; i < 3; ++i) {
    CHECK_EQ(d[i], uint8_t(0));
  }
}

TEST_CASE("zero-scalar fast path refuses GPU-produced scalars") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  set_default_device(Device::gpu);
  Stream s = new_stream(Device::gpu);

  auto lit = array(0.0f);
  auto copied = copy(lit, s);
  copied.eval();
  synchronize(s);
  CHECK_EQ(copied.data<float>()[0], 0.0f);

  auto produced = zeros({1}, float32, s);
  array out({1.0f, 1.0f, 1.0f, 1.0f});
  CHECK_THROWS_AS(fill_gpu(produced, out, s), std::runtime_error);
}

TEST_CASE("host event bridges a GPU-stream signal") {
  if (!cpu::is_available()) {
    skip("requires a development build with the CPU backend enabled.");
    return;
  }
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream cs = new_stream(Device::cpu);
  Stream gs = new_stream(Device::gpu);
  auto buf = omarchy::allocator().malloc(16);
  auto* vkbuf = static_cast<omarchy::VulkanBuffer*>(buf.ptr());
  auto& encoder = omarchy::get_command_encoder(gs);
  Event e{cs};
  e.set_value(1);
  e.signal(cs);
  e.wait();
  CHECK(e.is_signaled());

  e.wait(gs);
  encoder.fill_buffer(vkbuf->buffer, 0x11, 16);
  encoder.commit();
  encoder.synchronize();
  CHECK_EQ(static_cast<uint32_t*>(vkbuf->data)[0], 0x11u);

  e.set_value(2);
  encoder.fill_buffer(vkbuf->buffer, 0x22, 16);
  e.signal(gs);
  e.wait(gs);
  encoder.fill_buffer(vkbuf->buffer, 0x33, 16);
  encoder.commit();
  encoder.synchronize();
  e.wait();
  CHECK(e.is_signaled());
  CHECK_EQ(static_cast<uint32_t*>(vkbuf->data)[0], 0x33u);
  omarchy::allocator().free(buf);
}

TEST_CASE("CPU-produced event gates a GPU consumer without blocking eval") {
  if (!cpu::is_available()) {
    skip("requires a development build with the CPU backend enabled.");
    return;
  }
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }

  auto result = run_child("cpu_event_signal_first", 10);
  CHECK_FALSE(result.timed_out);
  CHECK_EQ(result.code, 0);
}

TEST_CASE("GPU wait before CPU signal keeps producer stream order") {
  if (!cpu::is_available()) {
    skip("requires a development build with the CPU backend enabled.");
    return;
  }
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }

  auto result = run_child("cpu_event_wait_first", 10);
  CHECK_FALSE(result.timed_out);
  CHECK_EQ(result.code, 0);
}

TEST_CASE("CPU-only event keeps scheduler errors and needs no GPU") {
  if (!cpu::is_available()) {
    skip("requires a development build with the CPU backend enabled.");
    return;
  }

  Stream producer = new_stream(Device::cpu);
  Event produced{producer};
  produced.set_value(1);
  scheduler::enqueue(producer, []() {
    throw std::runtime_error("cpu producer failed");
  });
  produced.signal(producer);
  CHECK_THROWS_WITH(produced.wait(), "cpu producer failed");
}

TEST_CASE("device_info returns stable references under concurrent access") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  const auto& info0 = gpu::device_info(0);
  auto name_it = info0.find("device_name");
  REQUIRE(name_it != info0.end());
  const auto* name0 = std::get_if<std::string>(&name_it->second);
  REQUIRE(name0 != nullptr);

  std::atomic<bool> stop{false};
  std::vector<std::thread> hammers;
  for (int t = 0; t < 4; ++t) {
    hammers.emplace_back([&stop]() {
      while (!stop.load()) {
        (void)gpu::device_info(0);
      }
    });
  }
  // The same map object and the same entries must come back every time.
  for (int i = 0; i < 2000; ++i) {
    const auto& info = gpu::device_info(0);
    CHECK(&info == &info0);
    auto it = info.find("device_name");
    REQUIRE(it != info.end());
    CHECK(&it->second == &name_it->second);
  }
  stop.store(true);
  for (auto& t : hammers) {
    t.join();
  }
  CHECK_FALSE(name0->empty());
}

TEST_CASE("independent streams measure against the serialized sum") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto& dev = omarchy::device(0);
  if (dev.capabilities().driver_id == VK_DRIVER_ID_MESA_LLVMPIPE) {
    skip("software Vulkan has no independent GPU execution.");
    return;
  }
  std::cout << "[receipt] device exposes " << dev.capabilities().queue_count
            << " queue(s); measuring overlap on one VkQueue\n";

  constexpr size_t kBytes = 8 << 20;
  Stream s1 = new_stream(Device::gpu);
  Stream s2 = new_stream(Device::gpu);
  auto a1 = omarchy::allocator().malloc(kBytes);
  auto b1 = omarchy::allocator().malloc(kBytes);
  auto a2 = omarchy::allocator().malloc(kBytes);
  auto b2 = omarchy::allocator().malloc(kBytes);
  auto* x1 = static_cast<omarchy::VulkanBuffer*>(a1.ptr());
  auto* y1 = static_cast<omarchy::VulkanBuffer*>(b1.ptr());
  auto* x2 = static_cast<omarchy::VulkanBuffer*>(a2.ptr());
  auto* y2 = static_cast<omarchy::VulkanBuffer*>(b2.ptr());
  std::memset(x1->data, 0x11, kBytes);
  std::memset(x2->data, 0x22, kBytes);

  auto& enc1 = omarchy::get_command_encoder(s1);
  auto& enc2 = omarchy::get_command_encoder(s2);
  auto clock = [](std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
  };

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 2; ++i) {
    enc1.copy_buffer(x1->buffer, y1->buffer, kBytes);
    enc1.commit();
  }
  enc1.synchronize();
  double per_iter = clock(t0) / 2.0;
  int iters = static_cast<int>(0.35 / per_iter);
  iters = std::max(2, std::min(iters, 400));

  auto run_serialized = [&]() {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      enc1.copy_buffer(x1->buffer, y1->buffer, kBytes);
      enc1.commit();
    }
    enc1.synchronize();
    for (int i = 0; i < iters; ++i) {
      enc2.copy_buffer(x2->buffer, y2->buffer, kBytes);
      enc2.commit();
    }
    enc2.synchronize();
    return clock(start);
  };
  auto run_concurrent = [&]() {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      enc1.copy_buffer(x1->buffer, y1->buffer, kBytes);
      enc1.commit();
      enc2.copy_buffer(x2->buffer, y2->buffer, kBytes);
      enc2.commit();
    }
    enc1.synchronize();
    enc2.synchronize();
    return clock(start);
  };

  std::array<double, 3> serialized{};
  std::array<double, 3> concurrent{};
  for (size_t trial = 0; trial < serialized.size(); ++trial) {
    if ((trial & 1) == 0) {
      serialized[trial] = run_serialized();
      concurrent[trial] = run_concurrent();
    } else {
      concurrent[trial] = run_concurrent();
      serialized[trial] = run_serialized();
    }
  }
  auto median = [](std::array<double, 3> values) {
    std::sort(values.begin(), values.end());
    return values[1];
  };
  double serialized_median = median(serialized);
  double concurrent_median = median(concurrent);
  double ratio = serialized_median / concurrent_median;
  std::cout << "[receipt] iters=" << iters
            << " serialized_median=" << serialized_median
            << "s concurrent_median=" << concurrent_median
            << "s ratio=" << ratio << "\n";
  // Stream independence: interleaving two streams' batches must not cost
  // more than running them alone; a global serialization lock would push
  // the ratio well below 1, and that is what the lower bound guards. On
  // real async hardware the interleaved run can legitimately FINISH
  // FASTER than serialized: the serialized pattern exposes a host-GPU
  // sync round trip between its two loops that interleaving hides
  // (measured ratio 1.21 on the M1/Honeykrisp device; llvmpipe's
  // synchronous QueueSubmit lands at ratio ~1, which is why this upper
  // direction passed there). The old `ratio < 1.10` upper bound asserted
  // that the overlap win was structurally subsumed, and that assumption
  // is false on the hardware we ship for, so it is not asserted here.
  // Interleaved being faster is only legitimate if every copy actually
  // ran, so the destination buffers are verified byte-for-byte instead.
  CHECK(ratio > 0.95);
  CHECK(std::memcmp(y1->data, x1->data, kBytes) == 0);
  CHECK(std::memcmp(y2->data, x2->data, kBytes) == 0);

  omarchy::allocator().free(a1);
  omarchy::allocator().free(b1);
  omarchy::allocator().free(a2);
  omarchy::allocator().free(b2);
}

TEST_CASE("a fresh process reopens the Vulkan device") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto r = run_child("reopen", 60);
  REQUIRE_FALSE(r.timed_out);
  CHECK_MESSAGE(
      r.code == 0, "child reopen scenario failed with code " << r.code);
}

TEST_CASE("a hung submit returns a bounded Omarchy error") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  // The child queues an unsatisfiable timeline wait. The bounded wait must
  // throw a typed Omarchy error (the 10 s bound is the behavior under
  // test); process isolation keeps the wedged queue away from this process.
  auto r = run_child("bounded_submit", 90);
  REQUIRE_FALSE(r.timed_out);
  CHECK_MESSAGE(
      r.code == 0, "child bounded_submit scenario failed with code " << r.code);
}

TEST_CASE("a long-but-progressing submission completes without a hang") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  // The child lowers the no-progress interval to 2 s and submits a
  // sequence of large fill batches whose combined wall time crosses
  // the interval while the per-batch signals keep the timeline
  // counter advancing. Old behavior (10 s wall cap) would fail by
  // name regardless of progress; new behavior must complete because
  // the watchdog only fires on stalled counter advance.
  auto r = run_child("progressing_long", 60);
  REQUIRE_FALSE(r.timed_out);
  CHECK_MESSAGE(
      r.code == 0,
      "child progressing_long scenario failed with code " << r.code);
}

TEST_CASE("slot reuse cannot lend stale progress to a blocked dependency") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto r = run_child("reused_slot_blocked_dependency", 10);
  REQUIRE_FALSE(r.timed_out);
  CHECK_MESSAGE(
      r.code == 0,
      "child reused_slot_blocked_dependency failed with code " << r.code);
}

TEST_CASE("a reused Event accepts a later published producer generation") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto r = run_child("late_reused_event_publication", 60);
  REQUIRE_FALSE(r.timed_out);
  CHECK_MESSAGE(
      r.code == 0,
      "child late_reused_event_publication failed with code " << r.code);
}

TEST_CASE("unrelated active work does not extend an unsignaled Event wait") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto r = run_child("unrelated_work_event_wait", 60);
  REQUIRE_FALSE(r.timed_out);
  CHECK_MESSAGE(
      r.code == 0,
      "child unrelated_work_event_wait failed with code " << r.code);
}

TEST_CASE("completed markers behind a blocked handler do not report progress") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto r = run_child("completed_marker_behind_handler", 10);
  REQUIRE_FALSE(r.timed_out);
  CHECK_MESSAGE(
      r.code == 0,
      "child completed_marker_behind_handler failed with code " << r.code);
}

TEST_CASE("a CPU signal of a GPU Event tracks earlier GPU work") {
  if (!gpu::is_available() || !cpu::is_available()) {
    skip("GPU and CPU devices are required.");
    return;
  }
  auto r = run_child("gpu_event_cpu_signal_progress", 60);
  REQUIRE_FALSE(r.timed_out);
  CHECK_MESSAGE(
      r.code == 0,
      "child gpu_event_cpu_signal_progress failed with code " << r.code);
}

TEST_CASE("an executing long matmul is not mistaken for a stalled queue") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto r = run_child("long_matmul_progress", 60);
  REQUIRE_FALSE(r.timed_out);
  CHECK_MESSAGE(
      r.code == 0,
      "child long_matmul_progress scenario failed with code " << r.code);
}

TEST_CASE("a watchdog throw unwinds cleanly through process teardown") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  // The child trips the no-progress watchdog on a dispatch that is still
  // executing, catches the typed error, and returns through FULL teardown
  // (encoder destruction, device destruction). A pre-fix process died
  // here with SIGSEGV (exit 139) or "pure virtual method called"
  // (exit 134): the main-thread encoder destroyed its command and
  // descriptor pools while the last submission was still pending.
  auto r = run_child("watchdog_teardown", 90);
  REQUIRE_FALSE(r.timed_out);
  CHECK_MESSAGE(
      r.code == 0,
      "child watchdog_teardown scenario failed with code " << r.code);
}

TEST_CASE("tensor ops dispatch on Vulkan and never silently on CPU") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }

  set_default_device(Device::gpu);

  // Default-device work must land on the Vulkan stream. The GPU counters
  // prove the kernel ran there; a silent CPU substitution would leave
  // them flat.
  auto x = array({1.0f, 2.0f}, float32);
  auto y = array({3.0f, 4.0f}, float32);
  auto z = add(x, y);
  uint64_t gpu_dispatches_before =
      omarchy::trace::counters().gpu_primitive_dispatches.load();
  uint64_t compute_dispatches_before =
      omarchy::trace::counters().vk_compute_dispatches.load();
  z.eval();
  synchronize(default_stream(default_device()));
  CHECK_EQ(z.data<float>()[0], 4.0f);
  CHECK_EQ(z.data<float>()[1], 6.0f);
  CHECK(
      omarchy::trace::counters().gpu_primitive_dispatches.load() >
      gpu_dispatches_before);
  CHECK(
      omarchy::trace::counters().vk_compute_dispatches.load() >
      compute_dispatches_before);

  if (cpu::is_available()) {
    // This build carries a CPU tensor backend. The stream boundary is the
    // contract: an op created on an explicit CPU stream runs there and
    // leaves the GPU counters flat. Nothing else may move work onto CPU.
    uint64_t gpu_dispatches_before_cpu =
        omarchy::trace::counters().gpu_primitive_dispatches.load();
    uint64_t compute_dispatches_before_cpu =
        omarchy::trace::counters().vk_compute_dispatches.load();
    Stream cpu_stream = new_stream(Device::cpu);
    auto c = add(x, y, cpu_stream);
    c.eval();
    synchronize(cpu_stream);
    CHECK_EQ(c.data<float>()[0], 4.0f);
    CHECK_EQ(c.data<float>()[1], 6.0f);
    CHECK_EQ(
        omarchy::trace::counters().gpu_primitive_dispatches.load(),
        gpu_dispatches_before_cpu);
    CHECK_EQ(
        omarchy::trace::counters().vk_compute_dispatches.load(),
        compute_dispatches_before_cpu);
  } else {
    // No CPU backend in this build: the CPU device does not exist at all.
    CHECK_FALSE(is_available(Device::cpu));
    CHECK_EQ(device_count(Device::cpu), 0);
  }

  auto zero_array = mlx::core::zeros({2, 3}, float32);
  zero_array.eval();
  synchronize(default_stream(default_device()));
  const auto* data = zero_array.data<float>();
  for (int i = 0; i < 6; ++i) {
    CHECK_EQ(data[i], 0.0f);
  }
}

TEST_CASE("small eager output stays ordered across deep submit boundaries") {
  // Pins the in-order-stream contract: an eager one-element f32 output
  // committed as its own submission must be visible to a consumer
  // dispatch in a LATER submission, even when long work fills the queue
  // in between. This is the shape of the compiled 4-bit decode abort:
  // the RoPE positions chain lost its device write on Honeykrisp when a
  // consumer submission ran without a dependency on the producer
  // submission (Vulkan defines no cross-submission ordering without a
  // wait; encoder submit() now waits on the stream's previous completion
  // value). llvmpipe executes submissions synchronously in order, so
  // this test pins the contract there and guards regressions of the
  // wait itself; the pre-fix failure needs an out-of-order queue and is
  // validated on Honeykrisp hardware.
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto s = default_stream(default_device());
  for (int iter = 0; iter < 8; ++iter) {
    // Tiny eager producer: one element, its own submission.
    auto positions = arange(iter, iter + 1, float32, s);
    positions.eval();
    // Long work fills the queue so the consumer below is submitted
    // while earlier submissions are still executing.
    auto heavy = astype(ones({256, 256}, float32, s), float32);
    for (int i = 0; i < 3; ++i) {
      heavy = matmul(heavy, heavy, s);
      heavy.eval();
    }
    // Consumer submission reads the producer's buffer across the queue.
    auto theta = positions * ones({1}, float32, s);
    auto c = cos(theta);
    c.eval();
    synchronize(s);
    CHECK_EQ(theta.item<float>(), static_cast<float>(iter));
    CHECK(std::fabs(c.item<float>() - std::cos(static_cast<float>(iter))) <
          1e-5f);
  }
}

TEST_CASE("workless evals submit nothing and pin nothing") {
  // Pins the temporaries flush contract for evals whose primitive
  // records no GPU work (view rearrangements, host-materialized values):
  // no submission may be issued for them, and their buffers must return
  // to the allocator once dropped, because nothing in flight references
  // them. Before the batch flush contract was explicit, every workless
  // eval boundary issued a signal-only QueueSubmit (measured ~1,645 per
  // decode token) that existed mostly to flush accumulated temporaries.
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto s = default_stream(default_device());
  synchronize(s);
  auto& counters = omarchy::trace::counters();
  uint64_t subs0 = counters.vk_submissions.load();
  uint64_t disp0 = counters.vk_compute_dispatches.load();
  size_t active0 = omarchy::allocator().get_active_memory();

  array x = ones({64}, float32, s);
  x.eval();
  synchronize(s);
  subs0 = counters.vk_submissions.load();
  disp0 = counters.vk_compute_dispatches.load();
  active0 = omarchy::allocator().get_active_memory();

  for (int i = 0; i < 200; ++i) {
    array v = transpose(reshape(x, {8, 8}));
    v.eval();
  }
  synchronize(s);
  CHECK_EQ(
      counters.vk_compute_dispatches.load() - disp0,
      static_cast<unsigned long long>(0));
  CHECK(counters.vk_submissions.load() - subs0 <= 4);
  CHECK(omarchy::allocator().get_active_memory() <= active0 + 4096);
}

TEST_CASE("one graph evaluation batches into bounded submissions") {
  // A single eval whose graph records far more nodes than the batch
  // budget must flush at the budget (kBatchNodeBudget) instead of
  // submitting per op, and the batched result must match elementwise.
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto s = default_stream(default_device());
  array w = ones({1}, float32, s);
  array y = zeros({8, 8}, float32, s);
  y.eval();
  synchronize(s);
  auto& counters = omarchy::trace::counters();
  uint64_t subs0 = counters.vk_submissions.load();

  constexpr int kAdds = 585;
  for (int i = 0; i < kAdds; ++i) {
    y = y + w;
  }
  y.eval();
  synchronize(s);
  uint64_t subs = counters.vk_submissions.load() - subs0;
  CHECK(subs >= 1);
  CHECK(subs <= 4);
  synchronize(s);
  const auto* data = y.data<float>();
  for (int i = 0; i < 64; ++i) {
    CHECK_EQ(data[i], static_cast<float>(kAdds));
  }
}

TEST_CASE("one graph evaluation flushes when freed intermediates reach the byte budget") {
  // The 2048-token forward of 2026-09-08: intermediates freed under an
  // open batch wait in the allocator quarantine until that batch
  // submits, so a 257-node batch held 8.11 GB against a 7.56 GiB heap
  // and vkAllocateMemory failed. The evaluator must flush the batch
  // once pending quarantine bytes reach memory_limit /
  // kBatchByteBudgetDivisor, well before the node budget, and the
  // batched result must still match elementwise. Shrinking the limit
  // makes the budget reachable with small tensors: 16 MiB / 16 = 1 MiB,
  // and each add frees one 64 KiB intermediate. The transposed input
  // is not donatable, so every add allocates fresh (a plain y + w chain
  // reuses one buffer in place and frees nothing).
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto s = default_stream(default_device());
  array w = ones({1}, float32, s);
  array y = zeros({128, 128}, float32, s);
  y.eval();
  synchronize(s);
  auto& counters = omarchy::trace::counters();
  auto& alloc = omarchy::allocator();
  size_t limit0 = alloc.set_memory_limit(16u << 20);
  uint64_t subs0 = counters.vk_submissions.load();

  constexpr int kAdds = 585;
  for (int i = 0; i < kAdds; ++i) {
    y = transpose(y) + w;
  }
  y.eval();
  synchronize(s);
  uint64_t subs = counters.vk_submissions.load() - subs0;
  alloc.set_memory_limit(limit0);
  // Node budget alone gives at most ceil(585 / 256) + 1 submissions; the
  // byte budget (16 frees per MiB) forces roughly one every 16 adds.
  CHECK(subs > 4);
  CHECK(subs <= static_cast<uint64_t>(kAdds));
  CHECK_EQ(alloc.pending_quarantine_bytes(), static_cast<size_t>(0));
  const auto* data = y.data<float>();
  for (int i = 0; i < 128 * 128; ++i) {
    CHECK_EQ(data[i], static_cast<float>(kAdds));
  }
}

TEST_CASE("dependency-gated barriers keep hazard chains correct") {
  // Pins the MLX_OMARCHY_GATED_BARRIERS contract: RAW, WAW, and WAR
  // chains on ONE buffer across consecutive nodes (fills, copies, and
  // compute dispatches) must hold their values with the gate on, the
  // tracker must emit a barrier for every hazard pair and never skip
  // one, and a disjoint pair must actually be skipped. Value asserts
  // run in both gate modes; the counter asserts are the tracker's
  // observable behavior and are asserted per mode (the default
  // unconditional path records no copy/fill barriers and skips
  // nothing). llvmpipe executes commands in order, so a missed hazard
  // would not always corrupt these values in software - that half of
  // the proof is the counter asserts here plus the Honeykrisp hardware
  // runs in the M1 protocol.
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto& alloc = omarchy::allocator();
  auto& counters = omarchy::trace::counters();
  Stream s = new_stream(Device::gpu);
  auto& encoder = omarchy::get_command_encoder(s);
  const bool gated = []() {
    const char* v = std::getenv("MLX_OMARCHY_GATED_BARRIERS");
    if (v == nullptr) {
      return true;  // default on
    }
    std::string t = v;
    return t == "1" || t == "on" || t == "true" || t == "yes";
  }();

  constexpr size_t kBytes = 1 << 16;
  constexpr uint32_t kFloats = kBytes / sizeof(float);
  auto handle = [](auto& buf) {
    return static_cast<omarchy::VulkanBuffer*>(buf.ptr())->buffer;
  };
  auto bytes_of = [](auto& buf) {
    return static_cast<uint8_t*>(
        static_cast<omarchy::VulkanBuffer*>(buf.ptr())->data);
  };
  auto all_bytes = [&](auto& buf, uint8_t v) {
    auto* p = bytes_of(buf);
    for (size_t i = 0; i < kBytes; ++i) {
      if (p[i] != v) {
        return false;
      }
    }
    return true;
  };
  auto all_floats = [&](auto& buf, float v) {
    auto* p = reinterpret_cast<float*>(bytes_of(buf));
    for (size_t i = 0; i < kFloats; ++i) {
      if (p[i] != v) {
        return false;
      }
    }
    return true;
  };
  auto fill_dispatch = [&](auto& buf, float value) {
    omarchy::ComputeParams params;
    params.count = kFloats;
    params.output_size = kFloats;
    params.alpha = value;
    std::array<omarchy::ComputeBinding, 1> bindings{
        omarchy::ComputeBinding{
            handle(buf), 0, kBytes, buf.ptr()}};
    encoder.dispatch_compute(
        omarchy::ComputeKernel::FillF32,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(kFloats));
  };
  auto snapshot = [&]() {
    return std::pair<uint64_t, uint64_t>(
        counters.barriers_emitted.load(), counters.barriers_skipped.load());
  };

  // RAW across nodes: a fill writes buf, a following copy reads it.
  auto raw_src = alloc.malloc(kBytes);
  auto raw_dst = alloc.malloc(kBytes);
  auto raw0 = snapshot();
  encoder.fill_buffer(handle(raw_src), 0x2a2a2a2au, kBytes);
  encoder.copy_buffer(handle(raw_src), handle(raw_dst), kBytes);
  encoder.commit();
  encoder.synchronize();
  CHECK(all_bytes(raw_src, 0x2a));
  CHECK(all_bytes(raw_dst, 0x2a));
  CHECK(
      counters.barriers_emitted.load() - raw0.first >=
      (gated ? 1u : 0u));
  CHECK(counters.barriers_skipped.load() - raw0.second == 0);

  // WAW: two fills to the same buffer, the later must win.
  auto waw = alloc.malloc(kBytes);
  auto waw0 = snapshot();
  encoder.fill_buffer(handle(waw), 0x01010101u, kBytes);
  encoder.fill_buffer(handle(waw), 0x02020202u, kBytes);
  encoder.commit();
  encoder.synchronize();
  CHECK(all_bytes(waw, 0x02));
  CHECK(
      counters.barriers_emitted.load() - waw0.first >=
      (gated ? 1u : 0u));
  CHECK(counters.barriers_skipped.load() - waw0.second == 0);

  // WAR: a copy reads buf, then a fill overwrites it. The copy must
  // keep the pre-fill bytes and the fill must land.
  auto war_src = alloc.malloc(kBytes);
  auto war_dst = alloc.malloc(kBytes);
  encoder.fill_buffer(handle(war_src), 0x03030303u, kBytes);
  encoder.commit();
  encoder.synchronize();
  auto war0 = snapshot();
  encoder.copy_buffer(handle(war_src), handle(war_dst), kBytes);
  encoder.fill_buffer(handle(war_src), 0x5e5e5e5eu, kBytes);
  encoder.commit();
  encoder.synchronize();
  CHECK(all_bytes(war_dst, 0x03));
  CHECK(all_bytes(war_src, 0x5e));
  CHECK(
      counters.barriers_emitted.load() - war0.first >=
      (gated ? 1u : 0u));
  CHECK(counters.barriers_skipped.load() - war0.second == 0);

  // Disjoint pair: writes to two separate buffers may skip. Only the
  // gate makes the skip observable; with the gate off nothing records
  // or skips a barrier for fills.
  auto dis_a = alloc.malloc(kBytes);
  auto dis_b = alloc.malloc(kBytes);
  auto dis0 = snapshot();
  encoder.fill_buffer(handle(dis_a), 0x11111111u, kBytes);
  encoder.fill_buffer(handle(dis_b), 0x22222222u, kBytes);
  encoder.commit();
  encoder.synchronize();
  CHECK(all_bytes(dis_a, 0x11));
  CHECK(all_bytes(dis_b, 0x22));
  if (gated) {
    CHECK(counters.barriers_skipped.load() - dis0.second >= 1);
  } else {
    CHECK(counters.barriers_skipped.load() - dis0.second == 0);
  }

  // Dispatch RAW: a compute dispatch writes, a copy reads. The copy
  // must see the dispatched floats, not stale bytes.
  auto dsp_src = alloc.malloc(kBytes);
  auto dsp_dst = alloc.malloc(kBytes);
  auto dsp0 = snapshot();
  fill_dispatch(dsp_src, 6.5f);
  encoder.copy_buffer(handle(dsp_src), handle(dsp_dst), kBytes);
  encoder.commit();
  encoder.synchronize();
  CHECK(all_floats(dsp_src, 6.5f));
  CHECK(all_floats(dsp_dst, 6.5f));
  CHECK(counters.barriers_emitted.load() - dsp0.first >= 1);
  CHECK(counters.barriers_skipped.load() - dsp0.second == 0);

  // Dispatch WAR: a copy reads, then a compute dispatch overwrites the
  // same buffer. The copy keeps the old bytes and the dispatch lands.
  auto wr_src = alloc.malloc(kBytes);
  auto wr_dst = alloc.malloc(kBytes);
  encoder.fill_buffer(handle(wr_src), 0x04040404u, kBytes);
  encoder.commit();
  encoder.synchronize();
  auto wr0 = snapshot();
  encoder.copy_buffer(handle(wr_src), handle(wr_dst), kBytes);
  fill_dispatch(wr_src, 2.25f);
  encoder.commit();
  encoder.synchronize();
  CHECK(all_bytes(wr_dst, 0x04));
  CHECK(all_floats(wr_src, 2.25f));
  CHECK(counters.barriers_emitted.load() - wr0.first >= 1);
  CHECK(counters.barriers_skipped.load() - wr0.second == 0);

  alloc.free(raw_src);
  alloc.free(raw_dst);
  alloc.free(waw);
  alloc.free(war_src);
  alloc.free(war_dst);
  alloc.free(dis_a);
  alloc.free(dis_b);
  alloc.free(dsp_src);
  alloc.free(dsp_dst);
  alloc.free(wr_src);
  alloc.free(wr_dst);
}

TEST_CASE("CPU PD hold: default on for G13 parts other than G13C, env overrides") {
  using mlx::core::omarchy::cpu_pd_hold_enabled_for;
  // jwm1 / M1 (T8103): the measured part.
  CHECK(cpu_pd_hold_enabled_for(nullptr, "Apple M1 (G13G B1)"));
  CHECK(cpu_pd_hold_enabled_for("", "Apple M1 (G13G B1)"));
  // M1 Max (jw16) showed no effect: off by default.
  CHECK_FALSE(cpu_pd_hold_enabled_for(nullptr, "Apple M1 Max (G13C C0)"));
  // Other generations stay on the previous behavior.
  CHECK_FALSE(cpu_pd_hold_enabled_for(nullptr, "Apple M2 (G14G B1)"));
  CHECK_FALSE(cpu_pd_hold_enabled_for(nullptr, "Apple M2 Max (G14C C0)"));
  // Off switch and force-on.
  CHECK_FALSE(cpu_pd_hold_enabled_for("0", "Apple M1 (G13G B1)"));
  CHECK(cpu_pd_hold_enabled_for("1", "Apple M1 Max (G13C C0)"));
  CHECK(cpu_pd_hold_enabled_for("on", "Apple M2 (G14G B1)"));
}

TEST_CASE("recycled block is not rewritten by the previous owner in flight") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto& alloc = omarchy::allocator();
  Stream s1 = new_stream(Device::gpu);
  Stream s2 = new_stream(Device::gpu);
  auto& e1 = omarchy::get_command_encoder(s1);
  auto& e2 = omarchy::get_command_encoder(s2);
  constexpr size_t kBytes = 1 << 26;
  constexpr uint32_t kP1 = 0x11111111u;
  constexpr uint32_t kP2 = 0x22222222u;
  auto a = alloc.malloc(kBytes);
  auto* a_buf = static_cast<omarchy::VulkanBuffer*>(a.ptr());
  array a_view(
      Shape{static_cast<int>(kBytes / sizeof(float))}, float32, nullptr, {});
  a_view.set_data(
      a, a_view.size(), a_view.strides(), a_view.flags(), 0,
      [](allocator::Buffer) {});
  for (int i = 0; i < 96; ++i) {
    e1.fill_buffer(a_buf->buffer, kP1, kBytes);
  }
  e1.commit();
  alloc.free(a);
  auto b = alloc.malloc(kBytes);
  CHECK(b.ptr() != a.ptr());
  auto* b_buf = static_cast<omarchy::VulkanBuffer*>(b.ptr());
  array b_view(
      Shape{static_cast<int>(kBytes / sizeof(float))}, float32, nullptr, {});
  b_view.set_data(
      b, b_view.size(), b_view.strides(), b_view.flags(), 0,
      [](allocator::Buffer) {});
  e2.fill_buffer(b_buf->buffer, kP2, kBytes);
  e2.commit();
  e2.synchronize();
  e1.synchronize();
  const auto* words = static_cast<const uint32_t*>(b_buf->data);
  CHECK(std::all_of(words, words + kBytes / sizeof(uint32_t),
                    [](uint32_t word) { return word == kP2; }));
  alloc.free(b);
}

TEST_CASE("stuck-submit warning fires before the wall typed error") {
  // 2026-10-08 sushi first-forward park: a submission that starts but
  // never retires keeps the no-progress watchdog refreshed forever
  // (has_active_submission), so the only bound was the 30-minute wall
  // deadline - invisible inside 30-minute gpu-turn tickets. The
  // stuck-submit diagnostic must emit its warning (and bump the counter)
  // within its interval, before the wall error; a healthy wait must not
  // warn. The child is process-isolated because both env knobs are
  // cached lazily on the first wait of a process.
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  ChildRun run = run_child("stuck_warning", 60);
  CHECK_FALSE(run.timed_out);
  CHECK_FALSE(run.signaled);
  CHECK(run.code == 0);
}

TEST_CASE("dispatch time estimator: fixed floor plus bandwidth term") {
  using namespace mlx::core::omarchy;
  // A zero-byte dispatch still costs the measured per-dispatch floor.
  CHECK(dispatch_est_ns(0) == kDispatchFixedNs);
  // Linear above the floor at the reference bandwidth (kBytesPerNs).
  CHECK(dispatch_est_ns(kBytesPerNs * 1'000'000) ==
        kDispatchFixedNs + 1'000'000);
  // Monotone.
  CHECK(dispatch_est_ns(1'000) >= dispatch_est_ns(0));
}

TEST_CASE("submission budget resolution: time default, legacy groups, off") {
  using namespace mlx::core::omarchy;
  // Default (neither variable set): time mode at kBatchTimeBudgetNs.
  BatchBudget b = resolve_batch_budget(nullptr, nullptr);
  CHECK_FALSE(b.off);
  CHECK_FALSE(b.groups_mode);
  CHECK(b.ns == kBatchTimeBudgetNs);
  // Legacy MLX_OMARCHY_BATCH_WORK keeps its shipped semantics.
  b = resolve_batch_budget("40000", nullptr);
  CHECK_FALSE(b.off);
  CHECK(b.groups_mode);
  CHECK(b.groups == 40000);
  b = resolve_batch_budget("0", nullptr); // 0 = cap off (shipped switch)
  CHECK(b.off);
  // New time knob overrides the default when no legacy variable is set.
  b = resolve_batch_budget(nullptr, "2.5");
  CHECK_FALSE(b.off);
  CHECK_FALSE(b.groups_mode);
  CHECK(b.ns == 2'500'000);
  b = resolve_batch_budget(nullptr, "0"); // 0 ms = cap off
  CHECK(b.off);
  // A set legacy variable wins over a set time variable (documented
  // precedence: the shipped knob keeps meaning what it meant).
  b = resolve_batch_budget("1000", "8");
  CHECK(b.groups_mode);
  CHECK(b.groups == 1000);
  // Unparseable legacy values map to the shipped atoll path: 0 = off.
  // A set-but-garbage legacy knob must not silently become a time cap.
  b = resolve_batch_budget("banana", "");
  CHECK(b.off);
}

TEST_CASE("batch flush predicate: boundaries across budget modes") {
  using namespace mlx::core::omarchy;
  BatchBudget off{true, false, 0, 0};
  BatchBudget groups_mode{false, true, 1000, 0};
  BatchBudget time_mode{false, false, 0, 1'000'000};
  // Off mode: only the node budget flushes.
  CHECK_FALSE(batch_over_budget(1, 999'999, 999, 4096, off));
  CHECK(batch_over_budget(4096, 0, 0, 4096, off));
  // Groups mode: summed groups flush, estimated ns does not.
  CHECK(batch_over_budget(1, 0, 1000, 4096, groups_mode));
  CHECK_FALSE(batch_over_budget(1, 2'000'000, 999, 4096, groups_mode));
  // Time mode: estimated ns flush, summed groups do not. Boundary is
  // inclusive: reaching the budget flushes.
  CHECK(batch_over_budget(1, 1'000'000, 10, 4096, time_mode));
  CHECK_FALSE(batch_over_budget(1, 999'999, 100'000, 4096, time_mode));
  // Node budget fires in every mode.
  CHECK(batch_over_budget(4096, 0, 0, 4096, groups_mode));
  CHECK(batch_over_budget(4096, 0, 0, 4096, time_mode));
}

TEST_CASE(
    "frame pacing: a 7B-class 8-bit decode token splits under the time "
    "budget") {
  // Synthetic dispatch plan for one Qwen2-7B-family 8-bit decode token
  // (28 layers, hidden 3584, MLP 18944, vocab 152064, GQA 28 Q / 4 KV,
  // head_dim 128), the shape of the issue #19 reporter's workload. Each
  // quantized matvec reads its weight bytes (8-bit, group 64, fp16
  // scales: out*in + out*(in/64)*2); norms, rope, sdpa, and glue record
  // as zero-byte dispatches that cost the fixed floor.
  using namespace mlx::core::omarchy;
  auto q8_bytes = [](uint64_t out, uint64_t in) {
    return out * in + out * (in / 64) * 2;
  };
  const uint64_t hidden = 3584, mlp = 18944, vocab = 152064;
  const uint64_t qkv_out = (28 + 2 * 4) * 128;
  std::vector<std::pair<uint64_t, uint64_t>> plan; // (binding bytes, groups)
  for (int l = 0; l < 28; ++l) {
    plan.emplace_back(q8_bytes(qkv_out, hidden), (qkv_out + 31) / 32);
    plan.emplace_back(0, 26); // input layernorm
    plan.emplace_back(0, 12); // q/k norm
    plan.emplace_back(0, 6);  // rope
    plan.emplace_back(0, 28); // sdpa
    plan.emplace_back(q8_bytes(hidden, hidden), (hidden + 31) / 32);
    plan.emplace_back(q8_bytes(mlp, hidden), (mlp + 31) / 32); // gate
    plan.emplace_back(q8_bytes(mlp, hidden), (mlp + 31) / 32); // up
    plan.emplace_back(0, 12);                                  // silu-mul
    plan.emplace_back(q8_bytes(hidden, mlp), (hidden + 31) / 32); // down
    plan.emplace_back(0, 13); // post layernorm
  }
  plan.emplace_back(q8_bytes(vocab, hidden), (vocab + 31) / 32); // lm head

  uint64_t token_est = 0;
  for (const auto& d : plan) {
    token_est += dispatch_est_ns(d.first);
  }

  // Walk the evaluator's flush loop under the default time budget: no
  // submission may carry an estimate above the budget (a single dispatch
  // bigger than the budget still runs whole, but the lm head here is
  // under it), and the token must split into the budget-quota of
  // submissions or more.
  BatchBudget budget = resolve_batch_budget(nullptr, nullptr);
  uint64_t run_est = 0, run_groups = 0, run_nodes = 0;
  uint64_t max_sub_est = 0, subs = 0;
  for (const auto& d : plan) {
    // eval() records the eval's dispatches first, then flushes the open
    // batch when the predicate fires on the accumulated totals.
    run_est += dispatch_est_ns(d.first);
    run_groups += d.second;
    ++run_nodes;
    if (batch_over_budget(
            static_cast<int>(run_nodes),
            run_est,
            run_groups,
            kBatchNodeBudget,
            budget)) {
      max_sub_est = std::max(max_sub_est, run_est);
      ++subs;
      run_est = 0;
      run_groups = 0;
      run_nodes = 0;
    }
  }
  if (run_nodes > 0) {
    max_sub_est = std::max(max_sub_est, run_est);
    ++subs;
  }
  // Every submission stays under the budget plus at most one dispatch
  // (the dispatch that crosses the line rides the flushed batch).
  uint64_t max_disp_est = 0;
  for (const auto& d : plan) {
    max_disp_est = std::max(max_disp_est, dispatch_est_ns(d.first));
  }
  CHECK(max_sub_est <= budget.ns + max_disp_est);
  CHECK(subs * budget.ns >= token_est);
  // The whole token must not fit in one submission.
  CHECK(subs >= 2);

  // The regression this replaces: under the shipped group-count budget
  // the same plan allows submissions the reporter measured at about
  // 2.1 microseconds per group (40000-group submissions, frame p50
  // 83.7 ms at 120 Hz). Ten-ms-plus submissions stutter a compositor;
  // the time budget exists so that cannot recur on a slower chip or a
  // bytes-heavier model class.
  uint64_t token_groups = 0;
  for (const auto& d : plan) {
    token_groups += d.second;
  }
  BatchBudget groups_budget = resolve_batch_budget("40000", nullptr);
  uint64_t max_sub_groups = 0;
  uint64_t g_run = 0;
  for (const auto& d : plan) {
    if (batch_over_budget(
            1, 0, g_run + d.second, kBatchNodeBudget, groups_budget)) {
      max_sub_groups = std::max(max_sub_groups, g_run);
      g_run = 0;
    }
    g_run += d.second;
  }
  max_sub_groups = std::max(max_sub_groups, g_run);
  const uint64_t reporter_ns_per_group = 2100;
  CHECK(max_sub_groups * reporter_ns_per_group >= 80'000'000);
}
