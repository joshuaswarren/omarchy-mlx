// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// CPU-only checks of the GPU profiler output sink and its runtime control
// (gpu_profile_sink.h) against real files: a temporary file and /dev/full.
// Pure C++: no Vulkan device and no mlx linkage.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "mlx/backend/omarchy/gpu_profile_sink.h"

namespace {

using mlx::core::omarchy::prof::ProfileSink;

struct TempFile {
  std::string path;
  FILE* f{nullptr};
  TempFile() {
    char name[] = "/tmp/omarchy_prof_sink_XXXXXX";
    int fd = mkstemp(name);
    REQUIRE(fd >= 0);
    path = name;
    f = fdopen(fd, "w");
    REQUIRE(f != nullptr);
  }
  ~TempFile() {
    unlink(path.c_str());
  }
  // Reads through a separate handle; the sink keeps its FILE open.
  std::string contents() const {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
  }
};

std::vector<std::string> lines_of(const std::string& text) {
  // Every byte belongs to a newline-terminated JSON object line.
  REQUIRE((text.empty() || text.back() == '\n'));
  std::vector<std::string> out;
  size_t at = 0;
  while (at < text.size()) {
    size_t nl = text.find('\n', at);
    std::string line = text.substr(at, nl - at);
    CHECK(line.size() >= 2);
    CHECK(line.front() == '{');
    CHECK(line.back() == '}');
    out.push_back(line);
    at = nl + 1;
  }
  return out;
}

void emit_meta(ProfileSink& sink) {
  sink.emitf("{\"k\":\"meta\",\"device\":\"%s\",\"pool\":%u}\n", "cpu", 65536u);
}

// A d-like record built from several emitf calls, as flush_slot does.
void emit_multipart(ProfileSink& sink, int i, int bindings) {
  sink.emitf("{\"k\":\"d\",\"s\":%d", i);
  sink.emitf(",\"b\":[");
  for (int j = 0; j < bindings; ++j) {
    sink.emitf("%s[%d,0,64]", j > 0 ? "," : "", j);
  }
  sink.emitf("]}\n");
}

} // namespace

TEST_CASE("control before the file opens, meta still lands while paused") {
  TempFile tmp;
  ProfileSink sink;
  CHECK(sink.control(0, 4096) == 1);
  CHECK_FALSE(sink.active());
  sink.open(tmp.f);
  emit_meta(sink);
  // Committed records reach the file by close (or a pause flush).
  sink.close();
  auto lines = lines_of(tmp.contents());
  REQUIRE(lines.size() == 1);
  CHECK(lines[0].find("\"k\":\"meta\"") != std::string::npos);
  CHECK(sink.bytes() == tmp.contents().size());
}

TEST_CASE("pause flushes committed records and keeps the file open") {
  TempFile tmp;
  ProfileSink sink;
  sink.open(tmp.f);
  CHECK(sink.active());
  emit_meta(sink);
  emit_multipart(sink, 1, 3);
  // A staged partial record must never be flushed by a pause.
  sink.emitf("{\"k\":\"d\",\"s\":2");
  CHECK(sink.control(0, ProfileSink::kDefaultCap) == 1);
  CHECK_FALSE(sink.active());
  auto lines = lines_of(tmp.contents());
  CHECK(lines.size() == 2);
  CHECK(sink.bytes() == tmp.contents().size());
  // Work captured before the pause still drains and still counts.
  sink.emitf(",\"b\":[]}\n");
  CHECK(sink.control(1, ProfileSink::kDefaultCap) == 1);
  CHECK(sink.active());
  CHECK(sink.control(0, ProfileSink::kDefaultCap) == 1);
  lines = lines_of(tmp.contents());
  CHECK(lines.size() == 3);
  CHECK(lines[2] == "{\"k\":\"d\",\"s\":2,\"b\":[]}");
  CHECK(sink.bytes() == tmp.contents().size());
  sink.close();
}

TEST_CASE("invalid arguments change nothing") {
  TempFile tmp;
  ProfileSink sink;
  sink.open(tmp.f);
  emit_meta(sink);
  uint64_t written = sink.bytes();
  REQUIRE(written > 1);
  CHECK(sink.control(2, 4096) == -1);
  CHECK(sink.control(-1, 4096) == -1);
  CHECK(sink.control(0, 0) == -1);
  CHECK(sink.control(0, written - 1) == -1);
  CHECK(sink.active());
  CHECK(sink.bytes() == written);
  CHECK(sink.control(1, written) == 1);
  sink.close();
}

TEST_CASE("lifetime cap counts every byte, keeps whole lines, and latches") {
  TempFile tmp;
  ProfileSink sink;
  const uint64_t cap = 400;
  CHECK(sink.control(1, cap) == 1);
  sink.open(tmp.f);
  emit_meta(sink);
  int i = 0;
  while (!sink.exhausted() && i < 1000) {
    emit_multipart(sink, i, 2);
    // Toggling with the same cap never resets the lifetime count.
    uint64_t before = sink.bytes();
    CHECK(sink.control(i & 1, cap) == (sink.exhausted() ? -2 : 1));
    CHECK(sink.bytes() == before);
    ++i;
  }
  REQUIRE(sink.exhausted());
  CHECK_FALSE(sink.active());
  CHECK(sink.control(1, 1ull << 40) == -2);
  CHECK_FALSE(sink.active());
  CHECK(sink.control(0, 1ull << 40) == -2);
  uint64_t final_bytes = sink.bytes();
  emit_multipart(sink, 9999, 1);
  emit_meta(sink);
  CHECK(sink.bytes() == final_bytes);
  sink.close();
  std::string text = tmp.contents();
  CHECK(text.size() == final_bytes);
  CHECK(text.size() <= cap);
  auto lines = lines_of(text);
  REQUIRE(lines.size() >= 3);
  CHECK(lines.front().find("\"k\":\"meta\"") != std::string::npos);
  CHECK(lines.back().find("{\"k\":\"cap\",") == 0);
  CHECK(lines.back().find("\"cap\":400") != std::string::npos);
  for (size_t n = 1; n + 1 < lines.size(); ++n) {
    CHECK(lines[n].find("{\"k\":\"d\",") == 0);
  }
}

TEST_CASE("a multi-part record that crosses the cap is dropped whole") {
  TempFile tmp;
  ProfileSink sink;
  const uint64_t cap = 200;
  CHECK(sink.control(1, cap) == 1);
  sink.open(tmp.f);
  emit_meta(sink);
  uint64_t after_meta = sink.bytes();
  // Far larger than the remaining budget: staging stops growing and the
  // record never reaches the file.
  emit_multipart(sink, 1, 500);
  CHECK(sink.exhausted());
  sink.close();
  std::string text = tmp.contents();
  auto lines = lines_of(text);
  REQUIRE(lines.size() == 2);
  CHECK(lines[0].find("\"k\":\"meta\"") != std::string::npos);
  CHECK(lines[1].find("{\"k\":\"cap\",\"bytes\":") == 0);
  CHECK(text.size() <= cap);
  CHECK(text.size() > after_meta);
}

TEST_CASE("a cap too small for the marker still writes nothing partial") {
  TempFile tmp;
  ProfileSink sink;
  CHECK(sink.control(1, 10) == 1);
  sink.open(tmp.f);
  emit_meta(sink);
  CHECK(sink.exhausted());
  sink.close();
  CHECK(tmp.contents().empty());
  CHECK(sink.bytes() == 0);
}

TEST_CASE("an observer thread toggling during emission keeps lines whole") {
  TempFile tmp;
  ProfileSink sink;
  const uint64_t cap = 1ull << 20;
  sink.open(tmp.f);
  emit_meta(sink);
  std::atomic<int> bad{0};
  std::thread observer([&] {
    for (int k = 0; k < 2000; ++k) {
      if (sink.control(k & 1, cap) != 1) {
        bad++;
      }
    }
  });
  for (int i = 0; i < 4000; ++i) {
    emit_multipart(sink, i, 4);
  }
  observer.join();
  CHECK(bad.load() == 0);
  uint64_t written = sink.bytes();
  sink.close();
  std::string text = tmp.contents();
  CHECK(text.size() == written);
  CHECK(text.size() <= cap);
  lines_of(text);
}

TEST_CASE("a failed pause flush reports unavailable and stops capture") {
  FILE* full = std::fopen("/dev/full", "w");
  REQUIRE(full != nullptr);
  ProfileSink sink;
  sink.open(full);
  // stdio buffers the meta record; ENOSPC surfaces at the pause flush.
  emit_meta(sink);
  CHECK(sink.control(0, ProfileSink::kDefaultCap) == 0);
  CHECK(sink.failed());
  CHECK_FALSE(sink.exhausted());
  CHECK_FALSE(sink.active());
  CHECK(sink.control(1, ProfileSink::kDefaultCap) == 0);
  CHECK_FALSE(sink.active());
  uint64_t before = sink.bytes();
  emit_multipart(sink, 1, 2);
  CHECK(sink.bytes() == before);
  // Argument validation still comes first.
  CHECK(sink.control(2, ProfileSink::kDefaultCap) == -1);
  sink.close();
}

TEST_CASE("a short write while emitting stops capture without a pause") {
  FILE* full = std::fopen("/dev/full", "w");
  REQUIRE(full != nullptr);
  ProfileSink sink;
  sink.open(full);
  CHECK(sink.active());
  // Enough records to overflow the stdio buffer and force a real write.
  int i = 0;
  for (; i < 100000 && !sink.failed(); ++i) {
    emit_multipart(sink, i, 4);
  }
  CHECK(sink.failed());
  CHECK_FALSE(sink.exhausted());
  CHECK_FALSE(sink.active());
  CHECK(sink.control(1, ProfileSink::kDefaultCap) == 0);
  uint64_t before = sink.bytes();
  emit_multipart(sink, i, 4);
  CHECK(sink.bytes() == before);
  sink.close();
}
