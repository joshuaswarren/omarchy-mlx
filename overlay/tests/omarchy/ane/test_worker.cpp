// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Host lifecycle tests for the bounded ANE worker. The device is a
// scripted fake; no ANE hardware, no libane, no GPU lock is touched.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "mlx/backend/omarchy/ane/tile_layout.h"
#include "mlx/backend/omarchy/ane/worker.h"

#include <csignal>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace mlx::core::omarchy::ane;

namespace {

// Behavior is selected through statics before fork(); the child inherits
// the selection, so no cross-process coordination is needed.
struct FakeDevice : AneDevice {
  enum class Behavior {
    Succeed,
    HangPastDeadline,
    CrashInExec,
    RefuseLoad,
    FailExecNamed,
    // Resident mode: the device reports its own state through the
    // output payload (byte 0 = programs loaded so far, byte 1 = the
    // last input byte it was sent) and takes per-submit instructions
    // from the input payload, so the parent can prove across submits
    // that one load served all of them and that each submit's inputs
    // really crossed the channel.
    ReportState,
  };
  static Behavior behavior;
  static inline const char* kFailReason = "fake device refused exec";
  // Input payload markers honoured in ReportState.
  static constexpr uint8_t kHangMarker = 0xFF;
  static constexpr uint8_t kAbortMarker = 0xFE;

  std::string describe() const override {
    return "fake-device";
  }

  void load(const AneValidatedProgram& program) override {
    if (behavior == Behavior::RefuseLoad) {
      throw AneDeviceError(
          "fake device refused to load " + program.anec.string());
    }
    // libane keys loaded networks by manifest index and refuses a
    // duplicate; the fake device holds the same contract so a resident
    // session that loads several bundles has to key them apart.
    if (!loaded_.insert(program.manifest_index).second) {
      throw AneDeviceError("program already loaded");
    }
  }

  void send(
      size_t /*manifest_index*/,
      uint32_t /*channel*/,
      const AneProgramBinding& binding,
      const uint8_t* data,
      size_t size) override {
    sent_.push_back({binding.tensor, size});
    if (size > 0) {
      last_input_ = data[0];
    }
  }

  void exec(size_t manifest_index) override {
    if (behavior == Behavior::HangPastDeadline ||
        (behavior == Behavior::ReportState && last_input_ == kHangMarker)) {
      // Sleep far past any test deadline; the supervisor must kill us.
      ::sleep(30);
      return;
    }
    if (behavior == Behavior::CrashInExec ||
        (behavior == Behavior::ReportState && last_input_ == kAbortMarker)) {
      ::abort(); // a real signal death (SIGABRT), not a clean _exit
    }
    if (behavior == Behavior::FailExecNamed) {
      throw AneDeviceError(kFailReason);
    }
    execs_.push_back(manifest_index);
  }

  void read(
      size_t /*manifest_index*/,
      uint32_t /*channel*/,
      const AneProgramBinding& binding,
      uint8_t* out,
      size_t size) override {
    // Deterministic payload: fill with the tensor name's bytes so the
    // parent can prove output bytes really crossed the pipe.
    for (size_t i = 0; i < size; ++i) {
      out[i] = static_cast<uint8_t>(binding.tensor.empty()
                                        ? '?'
                                        : binding.tensor[i % binding.tensor.size()]);
    }
    if (behavior == Behavior::ReportState && size >= 2) {
      out[0] = static_cast<uint8_t>(loaded_.size());
      out[1] = last_input_;
    }
  }

  void release() override {
    released_ = true;
  }

  struct Sent {
    std::string tensor;
    size_t size;
  };
  std::vector<Sent> sent_;
  std::vector<size_t> execs_;
  std::set<size_t> loaded_;
  uint8_t last_input_{0};
  bool released_{false};
};

FakeDevice::Behavior FakeDevice::behavior = FakeDevice::Behavior::Succeed;

AneBundle toy_bundle() {
  AneBundle bundle;
  bundle.manifest.manifest_version = 4;
  bundle.manifest.name = "toy-add";
  bundle.manifest.graph_hash = "deadbeef";
  bundle.manifest.driver_abi_major = 1;

  AneTensor a;
  a.name = "a";
  a.dtype = "fp16";
  a.shape = {1, 64, 1, 1};
  a.byte_size = 128;
  bundle.manifest.inputs = {a};
  AneTensor b = a;
  b.name = "b";
  bundle.manifest.inputs.push_back(b);

  AneTensor y;
  y.name = "y";
  y.dtype = "fp16";
  y.shape = {1, 64, 1, 1};
  y.byte_size = 128;
  bundle.manifest.logical_results = {};
  bundle.manifest.outputs = {y};

  AneProgram program;
  program.payload = "program-0.anec";
  program.operation = "add";
  AneProgramBinding in;
  in.tensor = "a";
  in.logical_bytes = 128;
  in.allocation_bytes = 128;
  program.inputs = {in};
  AneProgramBinding in_b = in;
  in_b.tensor = "b";
  program.inputs.push_back(in_b);
  AneProgramBinding out = in;
  out.tensor = "y";
  program.outputs = {out};
  bundle.manifest.programs = {program};
  bundle.manifest.dispatch_plan = {0};

  AneValidatedProgram validated;
  validated.manifest_index = 0;
  validated.anec = "/dev/null";
  bundle.programs = {validated};
  return bundle;
}

AneWorker::Buffer input_bytes(char fill) {
  return AneWorker::Buffer(128, static_cast<uint8_t>(fill));
}

// A second bundle whose program also carries manifest index 0, exactly
// like two independently adapted island bundles: a resident session has
// to key them apart before the device sees them.
AneBundle second_bundle() {
  AneBundle bundle = toy_bundle();
  bundle.manifest.name = "toy-add-2";
  return bundle;
}

AneWorker::Buffer marked_input(uint8_t marker) {
  AneWorker::Buffer payload(128, marker);
  return payload;
}

} // namespace

TEST_CASE("a completed run reports iterations, outputs, and release") {
  FakeDevice::behavior = FakeDevice::Behavior::Succeed;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  options.iterations = 2;
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  std::map<std::string, AneWorker::Buffer> inputs;
  inputs["a"] = input_bytes(0x11);
  inputs["b"] = input_bytes(0x22);
  std::map<std::string, AneWorker::Buffer> outputs;
  auto report = worker.run(toy_bundle(), inputs, &outputs);

  REQUIRE(report.status == AneWorkerStatus::Completed);
  CHECK(report.iterations == 2);
  CHECK(report.released_programs == 1);
  CHECK(!worker.quarantined());
  CHECK(report.elapsed < options.deadline);
  REQUIRE(outputs.count("y") == 1);
  CHECK(outputs["y"].size() == 128);
  std::string expected(128, 'y');
  CHECK(std::memcmp(outputs["y"].data(), expected.data(), 128) == 0);
  // detail_extra never leaks to callers.
  CHECK(report.detail_extra.empty());
}

TEST_CASE("a hanging device is killed at the deadline and quarantined") {
  FakeDevice::behavior = FakeDevice::Behavior::HangPastDeadline;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(250);
  options.iterations = 1;
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  auto report = worker.run(toy_bundle(), {{"a", input_bytes(1)}, {"b", input_bytes(2)}});
  REQUIRE(report.status == AneWorkerStatus::DeadlineExceeded);
  CHECK(report.elapsed >= std::chrono::milliseconds(200));
  CHECK(worker.quarantined());
  CHECK(worker.quarantine_reason().find("deadline") != std::string::npos);
  CHECK(
      worker.quarantine_reason().find("uncertain") != std::string::npos);

  // The quarantined worker refuses without spawning anything.
  auto refused = worker.run(toy_bundle(), {});
  REQUIRE(refused.status == AneWorkerStatus::QuarantinedRefused);
  CHECK(refused.iterations == 0);
  CHECK(refused.detail.find("quarantined") == 0);
}

TEST_CASE("a worker that dies in exec is quarantined with the signal") {
  FakeDevice::behavior = FakeDevice::Behavior::CrashInExec;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  auto report = worker.run(toy_bundle(), {{"a", input_bytes(1)}, {"b", input_bytes(2)}});
  REQUIRE(report.status == AneWorkerStatus::WorkerDied);
  CHECK(worker.quarantined());
  CHECK(worker.quarantine_reason().find("signal 6") != std::string::npos);
}

TEST_CASE("a named device failure is clean and does not quarantine") {
  std::map<std::string, AneWorker::Buffer> inputs = {
      {"a", input_bytes(1)}, {"b", input_bytes(2)}};
  FakeDevice::behavior = FakeDevice::Behavior::RefuseLoad;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  auto report = worker.run(toy_bundle(), inputs);
  REQUIRE(report.status == AneWorkerStatus::DeviceFailed);
  CHECK(!worker.quarantined());
  CHECK(report.detail.find("refused to load") != std::string::npos);

  FakeDevice::behavior = FakeDevice::Behavior::FailExecNamed;
  report = worker.run(toy_bundle(), inputs);
  REQUIRE(report.status == AneWorkerStatus::DeviceFailed);
  CHECK(!worker.quarantined());
  CHECK(report.detail == FakeDevice::kFailReason);

  // A clean-failed worker stays usable.
  FakeDevice::behavior = FakeDevice::Behavior::Succeed;
  report = worker.run(toy_bundle(), inputs);
  REQUIRE(report.status == AneWorkerStatus::Completed);
}

TEST_CASE("missing staging input is a named failure, not a crash") {
  FakeDevice::behavior = FakeDevice::Behavior::Succeed;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  std::map<std::string, AneWorker::Buffer> inputs;
  inputs["a"] = input_bytes(1);
  // "b" intentionally missing.
  auto report = worker.run(toy_bundle(), inputs);
  REQUIRE(report.status == AneWorkerStatus::DeviceFailed);
  CHECK(report.detail.find("'b' has no staged value") != std::string::npos);
  CHECK(!worker.quarantined());
}

TEST_CASE("worker construction rejects unbounded configurations") {
  auto factory = [] {
    return std::unique_ptr<AneDevice>(new FakeDevice);
  };
  AneWorkerOptions bad_deadline;
  bad_deadline.deadline = std::chrono::milliseconds(0);
  CHECK_THROWS_AS(AneWorker(factory, bad_deadline), std::invalid_argument);

  AneWorkerOptions bad_iterations;
  bad_iterations.iterations = 0;
  CHECK_THROWS_AS(AneWorker(factory, bad_iterations), std::invalid_argument);
}

TEST_CASE("a resident session loads once and serves every submit") {
  FakeDevice::behavior = FakeDevice::Behavior::ReportState;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  options.iterations = 1;
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  std::vector<AneBundle> bundles = {toy_bundle(), second_bundle()};
  auto opened = worker.open(bundles);
  REQUIRE(opened.status == AneWorkerStatus::Completed);
  CHECK(worker.resident());
  CHECK(opened.detail == "resident worker loaded 2 program(s) from 2 bundle(s)");

  // Six submits alternating between the two resident bundles. Output
  // byte 0 is the device's own count of loaded programs and byte 1 is
  // the last input byte it received, so a constant 2 across every
  // submit is the load happening once, and a changing byte 1 is each
  // submit's payload genuinely reaching the device.
  for (int round = 0; round < 6; ++round) {
    const uint8_t marker = static_cast<uint8_t>(0x40 + round);
    std::map<std::string, AneWorker::Buffer> inputs;
    inputs["a"] = marked_input(marker);
    inputs["b"] = marked_input(marker);
    std::map<std::string, AneWorker::Buffer> outputs;
    auto report = worker.submit(
        static_cast<size_t>(round % 2), inputs, &outputs);
    REQUIRE(report.status == AneWorkerStatus::Completed);
    CHECK(report.iterations == 1);
    CHECK(report.released_programs == 0); // nothing is released mid-session
    REQUIRE(outputs.count("y") == 1);
    REQUIRE(outputs["y"].size() == 128);
    CHECK(outputs["y"][0] == 2);
    CHECK(outputs["y"][1] == marker);
    CHECK(!worker.quarantined());
  }

  auto closed = worker.close();
  REQUIRE(closed.status == AneWorkerStatus::Completed);
  CHECK(closed.released_programs == 2);
  CHECK(!worker.resident());
  CHECK(!worker.quarantined());
}

TEST_CASE("a resident submit past its deadline quarantines the worker") {
  FakeDevice::behavior = FakeDevice::Behavior::ReportState;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(300);
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  std::vector<AneBundle> bundles = {toy_bundle()};
  REQUIRE(worker.open(bundles).status == AneWorkerStatus::Completed);

  std::map<std::string, AneWorker::Buffer> good;
  good["a"] = marked_input(0x11);
  good["b"] = marked_input(0x11);
  REQUIRE(worker.submit(0, good).status == AneWorkerStatus::Completed);

  // The deadline is per submit, not per session: the first submit
  // succeeded inside it, and this one hangs in exec.
  std::map<std::string, AneWorker::Buffer> hang;
  hang["a"] = marked_input(FakeDevice::kHangMarker);
  hang["b"] = marked_input(FakeDevice::kHangMarker);
  auto report = worker.submit(0, hang);
  REQUIRE(report.status == AneWorkerStatus::DeadlineExceeded);
  CHECK(report.elapsed >= std::chrono::milliseconds(250));
  CHECK(worker.quarantined());
  CHECK(worker.quarantine_reason().find("deadline") != std::string::npos);
  CHECK(worker.quarantine_reason().find("uncertain") != std::string::npos);
  CHECK(!worker.resident());

  // Same refusal contract as the one-shot path, for both entry points.
  auto refused = worker.submit(0, good);
  CHECK(refused.status == AneWorkerStatus::QuarantinedRefused);
  CHECK(worker.open(bundles).status == AneWorkerStatus::QuarantinedRefused);
  CHECK(worker.run(toy_bundle(), good).status ==
        AneWorkerStatus::QuarantinedRefused);
}

TEST_CASE("a resident child that dies mid-submit is quarantined") {
  FakeDevice::behavior = FakeDevice::Behavior::ReportState;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  std::vector<AneBundle> bundles = {toy_bundle()};
  REQUIRE(worker.open(bundles).status == AneWorkerStatus::Completed);

  std::map<std::string, AneWorker::Buffer> crash;
  crash["a"] = marked_input(FakeDevice::kAbortMarker);
  crash["b"] = marked_input(FakeDevice::kAbortMarker);
  auto report = worker.submit(0, crash);
  REQUIRE(report.status == AneWorkerStatus::WorkerDied);
  CHECK(worker.quarantined());
  CHECK(worker.quarantine_reason().find("signal 6") != std::string::npos);
  CHECK(worker.quarantine_reason().find("uncertain") != std::string::npos);
  CHECK(!worker.resident());
}

TEST_CASE("a named resident failure ends the session without quarantine") {
  FakeDevice::behavior = FakeDevice::Behavior::ReportState;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  std::vector<AneBundle> bundles = {toy_bundle()};
  REQUIRE(worker.open(bundles).status == AneWorkerStatus::Completed);

  std::map<std::string, AneWorker::Buffer> incomplete;
  incomplete["a"] = marked_input(0x22); // "b" intentionally missing
  auto report = worker.submit(0, incomplete);
  REQUIRE(report.status == AneWorkerStatus::DeviceFailed);
  CHECK(report.detail.find("'b' has no staged value") != std::string::npos);
  CHECK(!worker.quarantined());
  CHECK(!worker.resident());

  // A clean failure leaves the worker usable: a new session works.
  REQUIRE(worker.open(bundles).status == AneWorkerStatus::Completed);
  std::map<std::string, AneWorker::Buffer> good;
  good["a"] = marked_input(0x33);
  good["b"] = marked_input(0x33);
  std::map<std::string, AneWorker::Buffer> outputs;
  REQUIRE(worker.submit(0, good, &outputs).status ==
          AneWorkerStatus::Completed);
  CHECK(outputs["y"][1] == 0x33);
  REQUIRE(worker.close().status == AneWorkerStatus::Completed);
}

TEST_CASE("resident session misuse is rejected, not guessed at") {
  FakeDevice::behavior = FakeDevice::Behavior::ReportState;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  std::map<std::string, AneWorker::Buffer> inputs;
  inputs["a"] = marked_input(0x44);
  inputs["b"] = marked_input(0x44);
  CHECK_THROWS_AS(worker.submit(0, inputs), std::invalid_argument);
  CHECK_THROWS_AS(worker.close(), std::invalid_argument);
  CHECK_THROWS_AS(worker.open({}), std::invalid_argument);

  std::vector<AneBundle> bundles = {toy_bundle()};
  REQUIRE(worker.open(bundles).status == AneWorkerStatus::Completed);
  CHECK_THROWS_AS(worker.open(bundles), std::invalid_argument);
  // An out-of-range bundle index is a named API failure, not a silent
  // dispatch to the wrong programs. The bounds check happens here, in
  // the parent, before any wire bytes go out -- the resident would
  // also reject it as "unknown request", but the parent catches the
  // shape error first and refuses the submit outright.
  CHECK_THROWS_AS(worker.submit(7, inputs), std::invalid_argument);
  CHECK(worker.resident());
}

TEST_CASE("an unclosed resident session does not outlive its worker") {
  FakeDevice::behavior = FakeDevice::Behavior::ReportState;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  pid_t child = -1;
  {
    AneWorker worker(
        [] { return std::unique_ptr<AneDevice>(new FakeDevice); }, options);
    std::vector<AneBundle> bundles = {toy_bundle()};
    REQUIRE(worker.open(bundles).status == AneWorkerStatus::Completed);
    std::map<std::string, AneWorker::Buffer> inputs;
    inputs["a"] = marked_input(0x55);
    inputs["b"] = marked_input(0x55);
    REQUIRE(worker.submit(0, inputs).status == AneWorkerStatus::Completed);
    child = worker.resident_pid();
    REQUIRE(child > 0);
  }
  // The destructor killed and reaped it, so the pid is gone for good.
  CHECK(::kill(child, 0) != 0);
}

TEST_CASE("row pack and unpack match the per-element reference placement") {
  // The reference placement is ane_packed_offset; the row engine must
  // produce byte-identical tiles for every stride shape a bundle can
  // carry: contiguous, row-padded, plane-padded, bool, and a truncated
  // element count.
  auto reference_pack = [](const AneProgramBinding& binding,
                           const std::vector<uint8_t>& dense,
                           size_t elements) {
    std::vector<uint8_t> tile(binding.allocation_bytes, 0);
    const size_t es = ane_element_size(binding);
    for (size_t element = 0; element < elements; ++element) {
      const size_t at = ane_packed_offset(binding, element);
      std::memcpy(
          tile.data() + at, dense.data() + element * es, es);
    }
    return tile;
  };

  auto make_binding = [](const std::string& dtype,
                         const std::array<uint64_t, 6>& nchw) {
    AneProgramBinding binding;
    binding.tensor = "t";
    binding.dtype = dtype;
    binding.nchw = {nchw[0], nchw[1], nchw[2], nchw[3], nchw[4], nchw[5]};
    const size_t es = dtype == "bool" ? 1 : 2;
    const uint64_t rows = nchw[0] * nchw[1] * nchw[2];
    binding.logical_bytes = static_cast<size_t>(rows * nchw[3]) * es;
    binding.allocation_bytes =
        static_cast<size_t>((rows - 1) * nchw[5] + nchw[3] * es +
                            (nchw[0] * nchw[1] - 1) * nchw[4]);
    return binding;
  };

  struct Shape {
    std::array<uint64_t, 6> nchw;
    const char* dtype;
  };
  const std::vector<Shape> shapes = {
      // contiguous fp16 [2, 3, 4, 5]
      {{2, 3, 4, 5, 4 * 5 * 2, 5 * 2}, "fp16"},
      // row-padded fp16
      {{2, 3, 4, 5, 4 * 10 * 2, 10 * 2}, "fp16"},
      // plane-padded fp16 (rows still dense)
      {{1, 2, 3, 4, 3 * 4 * 2 + 64, 4 * 2}, "fp16"},
      // plane-padded fp16 (rows padded too)
      {{2, 1, 3, 4, 3 * 9 * 2 + 32, 9 * 2}, "fp16"},
      // contiguous bool
      {{1, 2, 2, 3, 2 * 3, 3}, "bool"},
  };

  for (const auto& shape : shapes) {
    AneProgramBinding binding = make_binding(shape.dtype, shape.nchw);
    CAPTURE(binding.tensor);
    CAPTURE(binding.allocation_bytes);

    std::vector<uint8_t> dense(binding.logical_bytes);
    for (size_t i = 0; i < dense.size(); ++i) {
      dense[i] = static_cast<uint8_t>(i * 7 + 1);
    }

    const size_t es = ane_element_size(binding);
    const size_t elements = binding.logical_bytes / es;

    std::vector<uint8_t> tile(binding.allocation_bytes, 0);
    ane_pack_rows(binding, dense.data(), tile.data(), elements);
    auto expected = reference_pack(binding, dense, elements);
    REQUIRE(tile == expected);

    std::vector<uint8_t> round_trip(elements * es, 0xEE);
    ane_unpack_rows(binding, tile.data(), round_trip.data(), elements);
    CHECK(round_trip == dense);

    // A truncated element count keeps the reference placement for the
    // elements it does transfer.
    if (elements > 5) {
      std::vector<uint8_t> partial_tile(binding.allocation_bytes, 0);
      ane_pack_rows(binding, dense.data(), partial_tile.data(), 5);
      CHECK(partial_tile == reference_pack(binding, dense, 5));
    }
  }
}

TEST_CASE("a batch scope bounds N submits under one deadline") {
  FakeDevice::behavior = FakeDevice::Behavior::ReportState;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  std::vector<AneBundle> bundles = {toy_bundle(), second_bundle()};
  REQUIRE(worker.open(bundles).status == AneWorkerStatus::Completed);
  CHECK(!worker.batching());

  auto opened = worker.open_batch(std::chrono::milliseconds(4000));
  REQUIRE(opened.status == AneWorkerStatus::Completed);
  CHECK(worker.batching());
  CHECK(worker.batch_rounds() == 0);

  // The whole pass is one bounded unit: rounds alternate bundles and
  // each payload still crosses, exactly like per-submit residency.
  for (int round = 0; round < 6; ++round) {
    const uint8_t marker = static_cast<uint8_t>(0x50 + round);
    std::map<std::string, AneWorker::Buffer> inputs;
    inputs["a"] = marked_input(marker);
    inputs["b"] = marked_input(marker);
    std::map<std::string, AneWorker::Buffer> outputs;
    auto report = worker.submit(
        static_cast<size_t>(round % 2), inputs, &outputs);
    REQUIRE(report.status == AneWorkerStatus::Completed);
    REQUIRE(outputs.count("y") == 1);
    CHECK(outputs["y"][0] == 2);
    CHECK(outputs["y"][1] == marker);
    CHECK(worker.batch_rounds() == round + 1);
  }

  auto closed = worker.close_batch();
  REQUIRE(closed.status == AneWorkerStatus::Completed);
  CHECK(closed.iterations == 6);
  CHECK(!worker.batching());

  // Outside the scope the per-submit window applies again, and the
  // session is still alive and releasable.
  std::map<std::string, AneWorker::Buffer> inputs;
  inputs["a"] = marked_input(0x60);
  inputs["b"] = marked_input(0x60);
  REQUIRE(worker.submit(0, inputs).status == AneWorkerStatus::Completed);
  auto release = worker.close();
  CHECK(release.status == AneWorkerStatus::Completed);
  CHECK(release.released_programs == 2);
}

TEST_CASE("a batch deadline miss quarantines the worker") {
  FakeDevice::behavior = FakeDevice::Behavior::ReportState;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  REQUIRE(worker.open({toy_bundle()}).status == AneWorkerStatus::Completed);
  REQUIRE(worker.open_batch(std::chrono::milliseconds(300)).status ==
          AneWorkerStatus::Completed);

  std::map<std::string, AneWorker::Buffer> good;
  good["a"] = marked_input(0x11);
  good["b"] = marked_input(0x11);
  // A quick successful round first: the deadline is the batch's, not the
  // first round's, so this passing does not discharge it.
  REQUIRE(worker.submit(0, good).status == AneWorkerStatus::Completed);

  std::map<std::string, AneWorker::Buffer> hang;
  hang["a"] = marked_input(FakeDevice::kHangMarker);
  hang["b"] = marked_input(FakeDevice::kHangMarker);
  auto report = worker.submit(0, hang);
  REQUIRE(report.status == AneWorkerStatus::DeadlineExceeded);
  CHECK(worker.quarantined());
  CHECK(worker.quarantine_reason().find("deadline") != std::string::npos);
  CHECK(!worker.resident());

  auto refused = worker.submit(0, good);
  CHECK(refused.status == AneWorkerStatus::QuarantinedRefused);
  CHECK(!worker.batching());
}

TEST_CASE("batch scope misuse is rejected, not guessed at") {
  FakeDevice::behavior = FakeDevice::Behavior::ReportState;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  CHECK_THROWS_AS(worker.open_batch(std::chrono::milliseconds(4000)),
                  std::invalid_argument);
  CHECK_THROWS_AS(worker.close_batch(), std::invalid_argument);
  CHECK_THROWS_AS(worker.open_batch(std::chrono::milliseconds(0)),
                  std::invalid_argument);

  REQUIRE(worker.open({toy_bundle()}).status == AneWorkerStatus::Completed);
  REQUIRE(worker.open_batch(std::chrono::milliseconds(4000)).status ==
          AneWorkerStatus::Completed);
  CHECK_THROWS_AS(worker.open_batch(std::chrono::milliseconds(4000)),
                  std::invalid_argument);
  auto closed = worker.close_batch();
  CHECK(closed.status == AneWorkerStatus::Completed);
  CHECK_THROWS_AS(worker.close_batch(), std::invalid_argument);
  auto release = worker.close();
  CHECK(release.status == AneWorkerStatus::Completed);
}

TEST_CASE("session names drive the wire, not manifest names") {
  // The relay-bypass runner speaks the caller's session keys (the
  // --bundle CLI names), which need not match the bundles' internal
  // manifest names. The first submit of the EncWallRelayBypass A/B
  // died exactly here: the child resolved the wire name against the
  // manifest namespace, rejected the request, and exited while the
  // runner was still writing the payload. The wire namespace is the
  // session namespace end to end or this test fails.
  FakeDevice::behavior = FakeDevice::Behavior::ReportState;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  options.iterations = 1;
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);
  AneBundle first = toy_bundle();
  first.manifest.name = "manifest-name-a";
  AneBundle second = second_bundle();
  second.manifest.name = "manifest-name-b";
  CHECK_THROWS_AS(
      worker.open({first, second}, {"only-one"}),
      std::invalid_argument);
  CHECK_THROWS_AS(
      worker.open({first, second}, {"", "x"}),
      std::invalid_argument);
  CHECK_THROWS_AS(
      worker.open({first, second}, {"dup", "dup"}),
      std::invalid_argument);
  auto opened = worker.open({first, second}, {"cli-a", "cli-b"});
  REQUIRE(opened.status == AneWorkerStatus::Completed);
  for (int round = 0; round < 4; ++round) {
    const uint8_t marker = static_cast<uint8_t>(0x50 + round);
    std::map<std::string, AneWorker::Buffer> inputs;
    inputs["a"] = marked_input(marker);
    inputs["b"] = marked_input(marker);
    std::map<std::string, AneWorker::Buffer> outputs;
    auto report = worker.submit(
        static_cast<size_t>(round % 2), inputs, &outputs);
    REQUIRE(report.status == AneWorkerStatus::Completed);
    REQUIRE(outputs.count("y") == 1);
    CHECK(outputs["y"][1] == marker);
    CHECK(!worker.quarantined());
  }
  auto closed = worker.close();
  REQUIRE(closed.status == AneWorkerStatus::Completed);
}
