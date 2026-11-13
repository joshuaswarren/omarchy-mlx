// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Host tests for the resident worker's decode-step features: state tensors
// that persist between submits (and their reset), and a step that runs
// several programs of a bundle and several bundles in order under one
// deadline. The device is a deterministic arithmetic fake; no ANE hardware,
// no libane, no kernel node is touched. The features live in the generic
// worker, so they hold for every lane.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "mlx/backend/omarchy/ane/worker.h"

#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace mlx::core::omarchy::ane;

namespace {

constexpr size_t kBytes = 16;
constexpr uint8_t kHangMarker = 0xFF;

using Buffer = AneWorker::Buffer;
using Tensors = std::map<std::string, Buffer>;

// Output byte k of a program = sum of its input bytes at k, plus a bias read
// from the program's payload name ("bias-N"). A program that is sent a
// first input byte of 0xFF sleeps past any test deadline.
struct ArithDevice : AneDevice {
  std::string describe() const override {
    return "arith-device";
  }
  void load(const AneValidatedProgram& program) override {
    const std::string name = program.anec.filename().string();
    bias_[program.manifest_index] = std::atoi(name.c_str() + 5);
  }
  void send(
      size_t index,
      uint32_t,
      const AneProgramBinding& binding,
      const uint8_t* data,
      size_t size) override {
    inputs_[{index, binding.tensor}] = Buffer(data, data + size);
  }
  void exec(size_t index) override {
    Buffer sum(kBytes, static_cast<uint8_t>(bias_.at(index)));
    for (const auto& entry : inputs_) {
      if (entry.first.first != index) continue;
      if (entry.second[0] == kHangMarker) {
        ::sleep(30);
      }
      for (size_t k = 0; k < kBytes; ++k) sum[k] += entry.second[k];
    }
    result_[index] = sum;
  }
  void read(
      size_t index,
      uint32_t,
      const AneProgramBinding&,
      uint8_t* out,
      size_t size) override {
    const Buffer& sum = result_.at(index);
    for (size_t k = 0; k < size; ++k) out[k] = sum[k % kBytes];
  }
  void release() override {}

  std::map<size_t, int> bias_;
  std::map<std::pair<size_t, std::string>, Buffer> inputs_;
  std::map<size_t, Buffer> result_;
};

AneTensor tensor(const std::string& name) {
  AneTensor t;
  t.name = name;
  t.dtype = "fp16";
  t.shape = {1, kBytes / 2, 1, 1};
  t.byte_size = kBytes;
  return t;
}

AneProgramBinding binding(const std::string& name) {
  AneProgramBinding b;
  b.tensor = name;
  b.logical_bytes = kBytes;
  b.allocation_bytes = kBytes;
  return b;
}

AneProgram program(
    int bias,
    const std::vector<std::string>& reads,
    const std::vector<std::string>& writes) {
  AneProgram p;
  p.payload = "bias-" + std::to_string(bias);
  for (const auto& name : reads) p.inputs.push_back(binding(name));
  for (const auto& name : writes) p.outputs.push_back(binding(name));
  return p;
}

AneValidatedProgram validated(size_t index, int bias) {
  AneValidatedProgram v;
  v.manifest_index = index;
  v.anec = "/nonexistent/bias-" + std::to_string(bias);
  return v;
}

// One program: y = in + state + bias, state' = y (a running sum).
AneBundle accumulator(const std::string& name, int bias) {
  AneBundle bundle;
  bundle.manifest.manifest_version = 4;
  bundle.manifest.name = name;
  bundle.manifest.inputs = {tensor("x")};
  bundle.manifest.outputs = {tensor("y")};
  bundle.manifest.state = {tensor("s")};
  bundle.manifest.programs = {program(bias, {"x", "s"}, {"y", "s"})};
  bundle.manifest.dispatch_plan = {0};
  bundle.programs = {validated(0, bias)};
  return bundle;
}

// Two programs in one plan: t = x + 1, then y = t + state + 2, state' = y.
AneBundle two_program_accumulator() {
  AneBundle bundle;
  bundle.manifest.manifest_version = 4;
  bundle.manifest.name = "two";
  bundle.manifest.inputs = {tensor("x")};
  bundle.manifest.outputs = {tensor("y")};
  bundle.manifest.state = {tensor("s")};
  bundle.manifest.intermediates = {tensor("t")};
  bundle.manifest.programs = {
      program(1, {"x"}, {"t"}), program(2, {"t", "s"}, {"y", "s"})};
  bundle.manifest.dispatch_plan = {0, 1};
  bundle.programs = {validated(0, 1), validated(1, 2)};
  return bundle;
}

// A stateless stage: out = in + bias.
AneBundle stage(
    const std::string& name,
    int bias,
    const std::string& in,
    const std::string& out) {
  AneBundle bundle;
  bundle.manifest.manifest_version = 4;
  bundle.manifest.name = name;
  bundle.manifest.inputs = {tensor(in)};
  bundle.manifest.outputs = {tensor(out)};
  bundle.manifest.programs = {program(bias, {in}, {out})};
  bundle.manifest.dispatch_plan = {0};
  bundle.programs = {validated(0, bias)};
  return bundle;
}

Buffer fill(uint8_t value) {
  return Buffer(kBytes, value);
}

AneWorker make_worker(int deadline_ms = 4000, int iterations = 1) {
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(deadline_ms);
  options.iterations = iterations;
  return AneWorker(
      [] { return std::unique_ptr<AneDevice>(new ArithDevice); }, options);
}

uint8_t first(const Tensors& outputs, const std::string& name) {
  auto found = outputs.find(name);
  REQUIRE(found != outputs.end());
  REQUIRE(found->second.size() == kBytes);
  return found->second[0];
}

} // namespace

TEST_CASE("state tensors carry from one resident submit to the next") {
  AneWorker worker = make_worker();
  REQUIRE(
      worker.open({accumulator("acc", 0)}, {"acc"}).status ==
      AneWorkerStatus::Completed);
  Tensors outputs;
  // y = x + s: 3, then 3 + 4, then 7 + 5.
  REQUIRE(worker.submit(0, {{"x", fill(3)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 3);
  REQUIRE(worker.submit(0, {{"x", fill(4)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 7);
  REQUIRE(worker.submit(0, {{"x", fill(5)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 12);
  CHECK(outputs.count("s") == 0); // state is not a manifest output
  CHECK(worker.close().status == AneWorkerStatus::Completed);
}

TEST_CASE("reset zeroes the state of one bundle, or of all, and nothing else") {
  AneWorker worker = make_worker();
  REQUIRE(
      worker.open({accumulator("a", 0), accumulator("b", 0)}, {"a", "b"})
          .status == AneWorkerStatus::Completed);
  Tensors outputs;
  REQUIRE(worker.submit(0, {{"x", fill(10)}}).status ==
          AneWorkerStatus::Completed);
  REQUIRE(worker.submit(1, {{"x", fill(20)}}).status ==
          AneWorkerStatus::Completed);
  // Bundles keep separate state: a is at 10, b is at 20.
  REQUIRE(worker.submit(0, {{"x", fill(1)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 11);
  REQUIRE(worker.submit(1, {{"x", fill(1)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 21);

  auto reset = worker.reset_state(0);
  REQUIRE(reset.status == AneWorkerStatus::Completed);
  CHECK(reset.detail.find("bundle 'a'") != std::string::npos);
  REQUIRE(worker.submit(0, {{"x", fill(1)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 1);   // a restarted
  REQUIRE(worker.submit(1, {{"x", fill(1)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 22);  // b untouched

  REQUIRE(worker.reset_all_state().status == AneWorkerStatus::Completed);
  REQUIRE(worker.submit(1, {{"x", fill(2)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 2);
  // The session survives a reset: same child, same loaded programs.
  CHECK(worker.resident());
  CHECK(!worker.quarantined());
  CHECK(worker.close().released_programs == 2);
}

TEST_CASE("a caller input of the state's name seeds it for that submit") {
  AneWorker worker = make_worker();
  REQUIRE(
      worker.open({accumulator("acc", 0)}, {"acc"}).status ==
      AneWorkerStatus::Completed);
  Tensors outputs;
  REQUIRE(worker.submit(0, {{"x", fill(1)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 1);
  // Seeding replaces the stored value (1) with 100, and what the plan
  // leaves behind (101) is the new state.
  REQUIRE(worker.submit(0, {{"x", fill(1)}, {"s", fill(100)}}, &outputs)
              .status == AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 101);
  REQUIRE(worker.submit(0, {{"x", fill(1)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 102);
  CHECK(worker.close().status == AneWorkerStatus::Completed);
}

TEST_CASE("state advances once per iteration inside one submit") {
  AneWorker worker = make_worker(4000, 3);
  REQUIRE(
      worker.open({accumulator("acc", 0)}, {"acc"}).status ==
      AneWorkerStatus::Completed);
  Tensors outputs;
  auto report = worker.submit(0, {{"x", fill(2)}}, &outputs);
  REQUIRE(report.status == AneWorkerStatus::Completed);
  CHECK(report.iterations == 3);
  CHECK(first(outputs, "y") == 6); // 2, 4, 6: the last iteration's output
  CHECK(worker.close().status == AneWorkerStatus::Completed);
}

TEST_CASE("a one-shot run carries state across its iterations only") {
  AneWorker first_run = make_worker(4000, 2);
  Tensors outputs;
  AneBundle bundle = accumulator("acc", 0);
  REQUIRE(first_run.run(bundle, {{"x", fill(5)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 10); // 5, then 5 + 5
  // A second run starts from zero state again.
  AneWorker second_run = make_worker(4000, 1);
  REQUIRE(second_run.run(bundle, {{"x", fill(5)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 5);
}

TEST_CASE("a plan of several programs keeps its state and intermediates") {
  AneWorker worker = make_worker();
  REQUIRE(
      worker.open({two_program_accumulator()}, {"two"}).status ==
      AneWorkerStatus::Completed);
  Tensors outputs;
  // y = (x + 1) + s + 2, s' = y.
  REQUIRE(worker.submit(0, {{"x", fill(10)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 13);
  REQUIRE(worker.submit(0, {{"x", fill(10)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 26);
  REQUIRE(worker.reset_state(0).status == AneWorkerStatus::Completed);
  REQUIRE(worker.submit(0, {{"x", fill(10)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 13);
  CHECK(worker.close().released_programs == 2);
}

TEST_CASE("a step runs bundles in order and routes tensors by name") {
  AneWorker worker = make_worker();
  // a: t = x + 10; b: u = t + 20; c: y = u + t + 1 (reads two earlier tensors).
  AneBundle c;
  c.manifest.manifest_version = 4;
  c.manifest.name = "c";
  c.manifest.inputs = {tensor("u"), tensor("t")};
  c.manifest.outputs = {tensor("y")};
  c.manifest.programs = {program(1, {"u", "t"}, {"y"})};
  c.manifest.dispatch_plan = {0};
  c.programs = {validated(0, 1)};
  REQUIRE(
      worker
          .open(
              {stage("a", 10, "x", "t"), stage("b", 20, "t", "u"), c},
              {"a", "b", "c"})
          .status == AneWorkerStatus::Completed);

  Tensors outputs;
  // Default result: the last bundle's manifest outputs.
  // t = 5 + 10 = 15, u = 35, y = 35 + 15 + 1 = 51.
  auto report = worker.step({0, 1, 2}, {{"x", fill(5)}}, &outputs);
  REQUIRE(report.status == AneWorkerStatus::Completed);
  CHECK(report.detail.find("3-bundle step") != std::string::npos);
  CHECK(outputs.size() == 1);
  CHECK(first(outputs, "y") == 51);

  // Emit chooses what comes back, intermediates included.
  REQUIRE(worker.step({0, 1}, {{"x", fill(5)}}, &outputs, {"t", "u"}).status ==
          AneWorkerStatus::Completed);
  CHECK(outputs.size() == 2);
  CHECK(first(outputs, "t") == 15);
  CHECK(first(outputs, "u") == 35);

  // A bundle may appear twice: a's output feeds a again as nothing, so use
  // b twice (u = t + 20 each time, same input -> same output).
  REQUIRE(worker.step({1, 1}, {{"t", fill(1)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "u") == 21);
  CHECK(worker.close().status == AneWorkerStatus::Completed);
}

TEST_CASE("a step names the bundle that lacks an input or an emit") {
  AneWorker worker = make_worker();
  REQUIRE(
      worker.open({stage("a", 1, "x", "t"), stage("b", 1, "q", "u")},
                  {"a", "b"})
          .status == AneWorkerStatus::Completed);
  Tensors outputs;
  // b needs q, which neither the caller nor a supplies.
  auto missing = worker.step({0, 1}, {{"x", fill(1)}}, &outputs);
  CHECK(missing.status == AneWorkerStatus::DeviceFailed);
  CHECK(missing.detail.find("step bundle 'b' (2 of 2)") != std::string::npos);
  CHECK(missing.detail.find("input tensor 'q' has no staged value") !=
        std::string::npos);
  CHECK(!worker.resident()); // a failed step ends the session...
  CHECK(!worker.quarantined()); // ...but the device state is not in doubt

  AneWorker second = make_worker();
  REQUIRE(
      second.open({stage("a", 1, "x", "t")}, {"a"}).status ==
      AneWorkerStatus::Completed);
  auto unknown = second.step({0}, {{"x", fill(1)}}, &outputs, {"nope"});
  CHECK(unknown.status == AneWorkerStatus::DeviceFailed);
  CHECK(unknown.detail.find("step emit 'nope' was never produced") !=
        std::string::npos);
}

TEST_CASE("state of the bundles in a step advances with the step") {
  AneWorker worker = make_worker();
  // acc runs as the second stage of the step: y = t + s, s' = y.
  AneBundle acc;
  acc.manifest.manifest_version = 4;
  acc.manifest.name = "acc";
  acc.manifest.inputs = {tensor("t")};
  acc.manifest.outputs = {tensor("y")};
  acc.manifest.state = {tensor("s")};
  acc.manifest.programs = {program(0, {"t", "s"}, {"y", "s"})};
  acc.manifest.dispatch_plan = {0};
  acc.programs = {validated(0, 0)};
  REQUIRE(
      worker.open({stage("pre", 1, "x", "t"), acc}, {"pre", "acc"}).status ==
      AneWorkerStatus::Completed);
  Tensors outputs;
  REQUIRE(worker.step({0, 1}, {{"x", fill(1)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 2);
  REQUIRE(worker.step({0, 1}, {{"x", fill(1)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 4);
  // A plain submit on the second bundle sees the same state.
  REQUIRE(worker.submit(1, {{"t", fill(1)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 5);
  REQUIRE(worker.reset_all_state().status == AneWorkerStatus::Completed);
  REQUIRE(worker.step({0, 1}, {{"x", fill(1)}}, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(first(outputs, "y") == 2);
  CHECK(worker.close().status == AneWorkerStatus::Completed);
}

TEST_CASE("a step is one deadline unit and a miss quarantines the worker") {
  using namespace std::chrono;
  // The per-submit window is long; the step's own deadline is short.
  AneWorker worker = make_worker(8000);
  REQUIRE(
      worker.open({stage("a", 1, "x", "t"), stage("b", 1, "t", "u")},
                  {"a", "b"})
          .status == AneWorkerStatus::Completed);
  Tensors outputs;
  // Within the step deadline: fine.
  REQUIRE(worker.step({0, 1}, {{"x", fill(1)}}, &outputs, {},
                      milliseconds(3000))
              .status == AneWorkerStatus::Completed);
  // The first stage hangs: the 300 ms step deadline, not the 8 s window,
  // decides.
  const auto started = steady_clock::now();
  auto report = worker.step({0, 1}, {{"x", fill(kHangMarker)}}, &outputs, {},
                            milliseconds(300));
  const auto elapsed = duration_cast<milliseconds>(steady_clock::now() - started);
  CHECK(report.status == AneWorkerStatus::DeadlineExceeded);
  CHECK(elapsed < milliseconds(3000));
  CHECK(report.detail.find("deadline of 300 ms") != std::string::npos);
  CHECK(worker.quarantined());
  CHECK(!worker.resident());
  CHECK(worker.step({0}, {{"x", fill(1)}}, &outputs).status ==
        AneWorkerStatus::QuarantinedRefused);
  CHECK(worker.reset_all_state().status ==
        AneWorkerStatus::QuarantinedRefused);
}

TEST_CASE("inside a batch the earlier of batch and step deadline wins") {
  using namespace std::chrono;
  AneWorker worker = make_worker(8000);
  REQUIRE(
      worker.open({stage("a", 1, "x", "t")}, {"a"}).status ==
      AneWorkerStatus::Completed);
  REQUIRE(worker.open_batch(milliseconds(250)).status ==
          AneWorkerStatus::Completed);
  Tensors outputs;
  // The step asks for 5 s, the batch only has 250 ms left.
  const auto started = steady_clock::now();
  auto report = worker.step({0}, {{"x", fill(kHangMarker)}}, &outputs, {},
                            milliseconds(5000));
  CHECK(report.status == AneWorkerStatus::DeadlineExceeded);
  CHECK(duration_cast<milliseconds>(steady_clock::now() - started) <
        milliseconds(3000));
  CHECK(worker.quarantined());
}

TEST_CASE("step and reset misuse is rejected, not guessed at") {
  AneWorker worker = make_worker();
  Tensors outputs;
  CHECK_THROWS_AS(worker.step({0}, {}, &outputs), std::invalid_argument);
  CHECK_THROWS_AS(worker.reset_state(0), std::invalid_argument);
  CHECK_THROWS_AS(worker.reset_all_state(), std::invalid_argument);
  REQUIRE(
      worker.open({stage("a", 1, "x", "t")}, {"a"}).status ==
      AneWorkerStatus::Completed);
  CHECK_THROWS_AS(worker.step({}, {}, &outputs), std::invalid_argument);
  CHECK_THROWS_AS(worker.step({3}, {}, &outputs), std::invalid_argument);
  CHECK_THROWS_AS(worker.reset_state(1), std::invalid_argument);
  CHECK_THROWS_AS(
      worker.step({0}, {}, &outputs, {}, std::chrono::milliseconds(-1)),
      std::invalid_argument);
  CHECK(worker.resident()); // none of that touched the session
  CHECK(worker.close().status == AneWorkerStatus::Completed);
}
