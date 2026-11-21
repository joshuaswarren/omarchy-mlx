// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Issue 537 (test_export_unique on the M4 Pro, M4 Air and M5 Max rows):
// an imported function's constants come back through Load. The evaluator
// fences cross-stream consumers as soon as Load::eval_cpu returns, so a
// size-1 load must hold its bytes by then. A scalar copy host-reads that
// word and broadcasts it; a deferred I/O write gave it the stale word.
//
// The reader below sleeps before it writes, so a deferred read cannot have
// finished when eval_cpu returns. The check is CPU only; start-up still
// opens a Vulkan device, and software Vulkan is enough.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "mlx/array.h"
#include "mlx/io/load.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

class SlowReader : public io::Reader {
 public:
  explicit SlowReader(std::vector<int32_t> values)
      : values_(std::move(values)) {}

  bool is_open() const override {
    return true;
  }
  bool good() const override {
    return true;
  }
  size_t tell() override {
    return 0;
  }
  void seek(int64_t, std::ios_base::seekdir) override {}
  void read(char*, size_t) override {
    FAIL("Load must use the positional read");
  }
  void read(char* data, size_t n, size_t offset) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::memcpy(data, reinterpret_cast<const char*>(values_.data()) + offset, n);
    reader_thread_ = std::this_thread::get_id();
    done_.store(true, std::memory_order_release);
  }
  std::string label() const override {
    return "slow-reader";
  }

  bool done() const {
    return done_.load(std::memory_order_acquire);
  }
  std::thread::id reader_thread() const {
    return reader_thread_;
  }

 private:
  std::vector<int32_t> values_;
  std::atomic<bool> done_{false};
  std::thread::id reader_thread_{};
};

// Run Load::eval_cpu directly, the way the evaluator does, and report what
// was visible the moment it returned.
struct LoadResult {
  bool done_on_return;
  std::vector<int32_t> values_on_return;
  std::thread::id reader_thread;
};

LoadResult eval_load(const std::vector<int32_t>& values) {
  auto s = default_stream(Device::cpu);
  auto reader = std::make_shared<SlowReader>(values);
  array out(
      {static_cast<int>(values.size())},
      int32,
      std::make_shared<Load>(s, reader, 0),
      {});
  static_cast<Load&>(out.primitive()).eval_cpu({}, out);

  LoadResult result;
  result.done_on_return = reader->done();
  result.values_on_return.assign(
      out.data<int32_t>(), out.data<int32_t>() + out.size());
  // Let a deferred read finish before the reader and buffer go away.
  synchronize(s);
  while (!reader->done()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  result.reader_thread = reader->reader_thread();
  return result;
}

} // namespace

TEST_CASE("a size-1 load publishes its bytes before eval_cpu returns") {
  auto result = eval_load({257});
  CHECK(result.done_on_return);
  CHECK(result.reader_thread == std::this_thread::get_id());
  REQUIRE(result.values_on_return.size() == 1);
  CHECK(result.values_on_return[0] == 257);
}

TEST_CASE("a multi-element load keeps the pipelined I/O read") {
  auto result = eval_load({1, 4, 3, 2});
  CHECK_FALSE(result.done_on_return);
  CHECK(result.reader_thread != std::this_thread::get_id());
}
