// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Bounded ANE worker CLI (plan sections 24-27). The one-shot form is one
// supervised execution of one bundle: load programs, run the dispatch
// plan for N iterations inside a wall-clock deadline, verify expected
// outputs byte-exactly when asked, release, exit. --serve and
// --relay-bypass keep one session for their parent process; --daemon
// keeps one session for successive clients on an owner-only unix socket.

#include "mlx/backend/omarchy/ane/bundle.h"
#include "mlx/backend/omarchy/ane/worker.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <poll.h>
#include <sys/stat.h>
#include <sstream>
#include <utility>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#endif

using namespace mlx::core::omarchy::ane;

#ifdef MLX_OMARCHY_ANE_DEVICE
namespace mlx::core::omarchy::ane {
std::unique_ptr<AneDevice> make_libane_device(const std::string& library);
}
#endif

namespace {

int usage() {
  std::fprintf(
      stderr,
      "usage: mlx-omarchy-ane-worker --bundle DIR --libane PATH\n"
      "       [--deadline-ms N] [--iterations N]\n"
      "       [--input NAME=FILE]... [--expect NAME=FILE]...\n"
      "       [--save NAME=FILE]...\n"
      "\n"
      "resident form (one process, one bundle load, many bounded\n"
      "submits; jobs are read from stdin, one per line):\n"
      "  mlx-omarchy-ane-worker --serve --libane PATH\n"
      "       --bundle NAME=DIR [--bundle NAME=DIR]...\n"
      "       [--deadline-ms N] [--iterations N]\n"
      "       [--seal-expect BUNDLE:FILE=SHA256]...\n"
      "       [--seal-expect-libane FILE=SHA256]\n"
      "  stdin: batch DEADLINE_MS | batch-end | quit\n"
      "         submit NAME [--input NAME=FILE]... [--save NAME=FILE]...\n"
      "         [--expect NAME=FILE]...\n"
      "         submit NAME [--inline NAME=BYTES]... [--emit NAME]...\n"
      "           followed by each inline payload's raw bytes, in order;\n"
      "           each emitted output comes back as 'out NAME BYTES'\n"
      "           plus its raw bytes, before the job status line\n"
      "         submits between 'batch DEADLINE_MS' and 'batch-end' are\n"
      "           one deadline-bounded unit; a miss ends the session\n"
      "\n"
      "daemon form (one process, one bundle load, one unix-socket at\n"
      "MLX_OMARCHY_ANE_SOCK serving one client at a time, idle\n"
      "shutdown when no client connects for --idle-time-ms; the\n"
      "device fd and resident buffer objects belong to the daemon\n"
      "for its whole lifetime):\n"
      "  mlx-omarchy-ane-worker --daemon --libane PATH\n"
      "       --socket PATH --bundle NAME=DIR [--bundle NAME=DIR]...\n"
      "       [--deadline-ms N] [--iterations N] [--idle-time-ms N]\n"
      "       [--seal-expect BUNDLE:FILE=SHA256]...\n"
      "       [--seal-expect-libane FILE=SHA256]\n"
      "  client side:\n"
      "  mlx-omarchy-ane-worker --stop --socket PATH\n"
      "\n"
      "every mode except the one-shot form requires --libane and at\n"
      "least one --bundle; --serve, --relay-bypass, --daemon and\n"
      "--stop are mutually exclusive\n");
  return 64;
}

// fp16 value equality: identical bits, except +0.0 and -0.0 compare
// equal. The H13 compiler models zero products as unsigned (aa688df
// "model unsigned zero products"), so a -0.0 expectation against a +0.0
// device result is a value match, not a mismatch.
bool fp16_values_equal(const uint8_t* lhs, const uint8_t* rhs, size_t size) {
  for (size_t i = 0; i + 1 < size; i += 2) {
    uint16_t left = static_cast<uint16_t>(lhs[i]) |
                    (static_cast<uint16_t>(lhs[i + 1]) << 8);
    uint16_t right = static_cast<uint16_t>(rhs[i]) |
                     (static_cast<uint16_t>(rhs[i + 1]) << 8);
    if (left == right) continue;
    if ((left & 0x7FFF) == 0 && (right & 0x7FFF) == 0) continue; // +/-0
    return false;
  }
  return true;
}

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open " + path);
  }
  return std::vector<uint8_t>(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
}

void write_file(const std::string& path, const std::vector<uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(
      reinterpret_cast<const char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
  if (!stream) {
    throw std::runtime_error("cannot write " + path);
  }
}

// NAME=FILE, the same assignment form the one-shot flags take.
bool split_assignment(
    const std::string& text,
    std::string& name,
    std::string& value) {
  auto separator = text.find('=');
  if (separator == std::string::npos || separator == 0) return false;
  name = text.substr(0, separator);
  value = text.substr(separator + 1);
  return !value.empty();
}

// A pinned digest is a lowercase hex sha256; anything else is a caller
// typo and must fail here, not silently disable the pin binding.
bool is_sha256_hex(const std::string& text) {
  return text.size() == 64 &&
      std::all_of(text.begin(), text.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      });
}

struct OwnedFd {
  int fd;
  explicit OwnedFd(int fd = -1) : fd(fd) {}
  ~OwnedFd() {
    if (fd >= 0) {
      ::close(fd);
    }
  }
  OwnedFd(const OwnedFd&) = delete;
  OwnedFd& operator=(const OwnedFd&) = delete;
};

void log_sealed_image(
    const std::string& session_name,
    const std::string& name,
    const AneSealedFile& image) {
  struct stat status {};
  size_t bytes =
      ::fstat(image.fd, &status) == 0 ? size_t(status.st_size) : 0;
  int seals = ::fcntl(image.fd, F_GET_SEALS);
  std::fprintf(
      stderr,
      "[omarchy-ane] sealed %s/%s bytes=%zu sha256=%s seals=0x%x\n",
      session_name.c_str(), name.c_str(), bytes, image.sha256.c_str(),
      seals > 0 ? unsigned(seals) : 0u);
}

// One resident bundle plus, when the caller pinned this session name,
// the sealed images its load consumed. `sealed` holds the descriptors
// backing the bundle's program and weights paths; it must outlive the
// bundle, the worker session, and the device that loaded its programs.
struct SealedLoad {
  AneBundle bundle;
  std::map<std::string, AneSealedFile> sealed;
};

// The load boundary for resident sessions: when the session name is
// pinned, every byte the bundle load consumes (manifest, every payload)
// is snapshotted into a sealed memfd, hashed, and bound to the pin
// before anything parses it, and all later consumption reads the sealed
// images — a flip or replacement of the on-disk bundle after this point
// cannot reach execution. Without a pin the bundle loads exactly as the
// one-shot path always has.
SealedLoad load_resident_bundle(
    const std::string& session_name,
    const std::string& dir,
    const std::map<std::string, std::map<std::string, std::string>>&
        seal_expects) {
  auto pinned = seal_expects.find(session_name);
  if (pinned == seal_expects.end() || pinned->second.empty()) {
    SealedLoad plain;
    plain.bundle = load_bundle(dir);
    return plain;
  }
  std::vector<AneSealedFile> sealed;
  AneBundle bundle = load_bundle_sealed(dir, pinned->second, sealed);
  SealedLoad load;
  load.bundle = std::move(bundle);
  // load_bundle_sealed appends the manifest first, then the payloads in
  // manifest order; pair that order back to file names for the log.
  size_t slot = 0;
  load.sealed.emplace("manifest.json", std::move(sealed.at(slot++)));
  for (const auto& payload : load.bundle.manifest.payloads) {
    load.sealed.emplace(payload.path, std::move(sealed.at(slot++)));
  }
  for (const auto& [name, image] : load.sealed) {
    log_sealed_image(session_name, name, image);
  }
  std::fflush(stderr);
  return load;
}

// The device library: when the caller pinned it, the dlopen performed by
// the resident child consumes a sealed image of the pinned bytes instead
// of the mutable installed path. The pin must name exactly the library
// that will be opened — an unpinned or differently-named library is a
// refusal, not a fallback.
std::string sealed_device_library(
    const std::string& libane_path,
    const std::map<std::string, std::string>& seal_libane_expect,
    std::vector<AneSealedFile>& keep) {
  if (seal_libane_expect.empty()) {
    return libane_path;
  }
  const std::filesystem::path path(libane_path);
  const std::string name = path.filename().string();
  auto pinned = seal_libane_expect.find(name);
  if (pinned == seal_libane_expect.end()) {
    throw std::runtime_error(
        "[omarchy-ane] no pinned digest for device library " + name +
        "; refusing to load unpinned ANE userspace");
  }
  if (seal_libane_expect.size() != 1) {
    throw std::runtime_error(
        "[omarchy-ane] --seal-expect-libane must name only " + name);
  }
  OwnedFd directory(
      ::open(path.parent_path().c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY));
  if (directory.fd < 0) {
    throw std::runtime_error(
        "[omarchy-ane] cannot open " + path.parent_path().string() + ": " +
        std::strerror(errno));
  }
  AneSealedFile image = sealed_file_at(directory.fd, name);
  if (image.sha256 != pinned->second) {
    throw std::runtime_error(
        "[omarchy-ane] sealed " + name + " sha256 " + image.sha256 +
        " does not match the pin " + pinned->second +
        "; refusing to load unverified ANE userspace");
  }
  log_sealed_image("libane", name, image);
  std::fflush(stderr);
  keep.push_back(std::move(image));
  return "/proc/self/fd/" + std::to_string(keep.back().fd);
}

struct ResidentJob {
  std::string bundle;
  std::map<std::string, std::string> inputs;
  std::map<std::string, std::string> saves;
  std::map<std::string, std::string> expects;
  // Payloads carried on stdin/stdout instead of through files, in the
  // order the job named them. Staging a megabyte island input through
  // a file costs a write in the caller and a read here; measured on
  // m1-test-host that read alone was 14-19 ms per submit, more than the
  // process launch a resident worker saves.
  std::vector<std::pair<std::string, size_t>> inline_inputs;
  std::vector<std::string> emits;
};

// "submit NAME [--input n=FILE]... [--inline n=BYTES]... [--save n=FILE]...
//  [--emit n]... [--expect n=FILE]..." or "quit". A malformed line is a
// named refusal, never a guess.
bool parse_job(const std::string& line, ResidentJob& job, std::string& error) {
  std::istringstream stream(line);
  std::string token;
  if (!(stream >> token) || token != "submit") {
    error = "expected 'submit BUNDLE' or 'quit'";
    return false;
  }
  if (!(stream >> job.bundle)) {
    error = "submit is missing its bundle name";
    return false;
  }
  while (stream >> token) {
    std::string argument;
    if (!(stream >> argument)) {
      error = token + " is missing its value";
      return false;
    }
    if (token == "--emit") {
      job.emits.push_back(argument);
      continue;
    }
    std::string name;
    std::string value;
    if (!split_assignment(argument, name, value)) {
      error = token + " expects NAME=VALUE, got '" + argument + "'";
      return false;
    }
    if (token == "--input") {
      job.inputs[name] = value;
    } else if (token == "--save") {
      job.saves[name] = value;
    } else if (token == "--expect") {
      job.expects[name] = value;
    } else if (token == "--inline") {
      size_t consumed = 0;
      size_t length = 0;
      try {
        length = static_cast<size_t>(std::stoull(value, &consumed));
      } catch (...) {
        consumed = 0;
      }
      if (consumed == 0 || consumed != value.size()) {
        error = "--inline expects NAME=BYTES, got '" + argument + "'";
        return false;
      }
      job.inline_inputs.emplace_back(name, length);
    } else {
      error = "unknown job flag '" + token + "'";
      return false;
    }
  }
  return true;
}

#ifdef MLX_OMARCHY_ANE_DEVICE

// The prologue both resident transports share: load every named bundle
// (sealed and pin-bound when that session name is pinned), seal the
// device library when it is pinned, and construct the worker whose
// child dlopens the sealed library and loads the sealed bundles.
struct ResidentSession {
  std::vector<SealedLoad> loads;                // sealed bundle images
  std::vector<AneSealedFile> library_images;    // sealed device library
  std::vector<AneBundle> bundles;
  std::map<std::string, size_t> index_of;
  std::vector<std::string> session_names;
  std::unique_ptr<AneWorker> worker;
};

ResidentSession open_resident_session(
    const std::vector<std::pair<std::string, std::string>>& bundle_args,
    const std::string& libane_path,
    long deadline_ms,
    long iterations,
    const std::map<std::string, std::map<std::string, std::string>>&
        seal_expects,
    const std::map<std::string, std::string>& seal_libane_expect) {
  // Optional open-phase timing (MLX_OMARCHY_OPEN_TIMING=1): parent-side
  // seal/load cost per bundle and for the device library; the child's
  // per-program load and the fork-to-loaded handshake are logged by
  // AneWorker under the same gate.
  const bool timing = [] {
    const char* v = ::getenv("MLX_OMARCHY_OPEN_TIMING");
    return v != nullptr && v[0] != '\0';
  }();
  auto ms_since = [](std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
  };
  ResidentSession session;
  session.loads.reserve(bundle_args.size());
  session.bundles.reserve(bundle_args.size());
  session.session_names.reserve(bundle_args.size());
  for (const auto& entry : bundle_args) {
    if (session.index_of.count(entry.first)) {
      std::fprintf(
          stderr, "duplicate resident bundle name '%s'\n",
          entry.first.c_str());
      std::exit(64);
    }
    const auto load_started = std::chrono::steady_clock::now();
    session.loads.push_back(
        load_resident_bundle(entry.first, entry.second, seal_expects));
    if (timing) {
      std::fprintf(
          stderr, "[omarchy-ane] timing bundle %s load %.1f ms\n",
          entry.first.c_str(), ms_since(load_started));
    }
    const AneBundle& bundle = session.loads.back().bundle;
    std::printf(
        "resident bundle=%s index=%zu name=%s programs=%zu driver_abi=%llu "
        "graph=%s\n",
        entry.first.c_str(), session.bundles.size(),
        bundle.manifest.name.c_str(), bundle.programs.size(),
        static_cast<unsigned long long>(bundle.manifest.driver_abi_major),
        bundle.manifest.graph_hash.c_str());
    session.index_of[entry.first] = session.bundles.size();
    session.session_names.push_back(entry.first);
    session.bundles.push_back(std::move(session.loads.back().bundle));
  }
  std::fflush(stdout);

  const auto library_started = std::chrono::steady_clock::now();
  const std::string device_library = sealed_device_library(
      libane_path, seal_libane_expect, session.library_images);
  if (timing) {
    std::fprintf(
        stderr, "[omarchy-ane] timing libane seal %.1f ms\n",
        ms_since(library_started));
  }
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(deadline_ms);
  options.iterations = static_cast<int>(iterations);
  session.worker = std::make_unique<AneWorker>(
      [device_library] { return make_libane_device(device_library); },
      options);
  return session;
}

#endif // MLX_OMARCHY_ANE_DEVICE

// Resident service: load every named bundle once, keep the programs on
// the device, and serve bounded submits from stdin until the caller
// quits or one submit fails. Not an inference server (plan section 25):
// no socket, no listener, and the process is a private child of the
// caller that owns it.
int serve_resident(
    const std::vector<std::pair<std::string, std::string>>& bundle_args,
    const std::string& libane_path,
    long deadline_ms,
    long iterations,
    const std::map<std::string, std::map<std::string, std::string>>&
        seal_expects,
    const std::map<std::string, std::string>& seal_libane_expect) {
#ifndef MLX_OMARCHY_ANE_DEVICE
  (void)bundle_args;
  (void)libane_path;
  (void)deadline_ms;
  (void)iterations;
  (void)seal_expects;
  (void)seal_libane_expect;
  std::fprintf(
      stderr,
      "this binary was built without MLX_OMARCHY_ANE_DEVICE; no device "
      "backend is linked\n");
  return 70;
#else
  ResidentSession session = open_resident_session(
      bundle_args, libane_path, deadline_ms, iterations, seal_expects,
      seal_libane_expect);
  AneWorker& worker = *session.worker;
  const std::map<std::string, size_t>& index_of = session.index_of;

  AneWorkerReport opened = worker.open(session.bundles, session.session_names);
  if (opened.status != AneWorkerStatus::Completed) {
    std::fprintf(
        stderr, "resident open failed: %s\n", opened.detail.c_str());
    return 1;
  }
  std::printf(
      "resident loaded pid=%lld deadline_ms=%ld iterations=%ld detail=%s\n",
      static_cast<long long>(worker.resident_pid()), deadline_ms, iterations,
      opened.detail.c_str());
  std::fflush(stdout);

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    if (line == "quit") {
      AneWorkerReport closed = worker.close();
      if (closed.status != AneWorkerStatus::Completed) {
        std::fprintf(
            stderr, "resident close failed: %s\n", closed.detail.c_str());
        return 1;
      }
      std::printf(
          "resident released programs=%d\n", closed.released_programs);
      std::fflush(stdout);
      return 0;
    }
    if (line == "batch-end" || line.compare(0, 6, "batch ") == 0) {
      if (line == "batch-end") {
        AneWorkerReport closed;
        try {
          closed = worker.close_batch();
        } catch (const std::invalid_argument& error) {
          std::fprintf(stderr, "batch refused: %s\n", error.what());
          return 64;
        }
        if (closed.status != AneWorkerStatus::Completed) {
          std::fprintf(
              stderr, "batch close failed: %s\n", closed.detail.c_str());
          return 1;
        }
        std::printf("batch closed rounds=%d\n", closed.iterations);
        std::fflush(stdout);
        continue;
      }
      const std::string value = line.substr(6);
      size_t consumed = 0;
      long batch_deadline = 0;
      try {
        batch_deadline = std::stol(value, &consumed);
      } catch (...) {
        consumed = 0;
      }
      if (consumed == 0 || consumed != value.size() || batch_deadline <= 0) {
        std::fprintf(
            stderr,
            "malformed batch: expected 'batch DEADLINE_MS', got '%s'\n",
            line.c_str());
        return 64;
      }
      AneWorkerReport opened;
      try {
        opened = worker.open_batch(std::chrono::milliseconds(batch_deadline));
      } catch (const std::invalid_argument& error) {
        std::fprintf(stderr, "batch refused: %s\n", error.what());
        return 64;
      }
      if (opened.status != AneWorkerStatus::Completed) {
        std::fprintf(stderr, "batch open failed: %s\n", opened.detail.c_str());
        return 1;
      }
      std::printf("%s\n", opened.detail.c_str());
      std::fflush(stdout);
      continue;
    }


    ResidentJob job;
    std::string error;
    if (!parse_job(line, job, error)) {
      std::fprintf(stderr, "malformed job: %s\n", error.c_str());
      return 64;
    }
    auto found = index_of.find(job.bundle);
    if (found == index_of.end()) {
      std::fprintf(
          stderr, "job names unknown resident bundle '%s'\n",
          job.bundle.c_str());
      return 64;
    }
    const AneBundle& bundle = session.bundles[found->second];

    // Three phases, reported per job: an operator who sees a slow
    // submit needs to know whether the time went to host staging or to
    // the device.
    const auto stage_started = std::chrono::steady_clock::now();
    std::map<std::string, AneWorker::Buffer> inputs;
    size_t input_bytes = 0;
    for (const auto& entry : job.inputs) {
      inputs[entry.first] = read_file(entry.second);
      input_bytes += inputs[entry.first].size();
    }
    for (const auto& entry : job.inline_inputs) {
      AneWorker::Buffer payload(entry.second);
      if (entry.second > 0 &&
          !std::cin.read(
              reinterpret_cast<char*>(payload.data()),
              static_cast<std::streamsize>(entry.second))) {
        std::fprintf(
            stderr, "stdin ended inside the inline payload for '%s'\n",
            entry.first.c_str());
        return 65;
      }
      input_bytes += payload.size();
      inputs[entry.first] = std::move(payload);
    }
    const auto stage_ended = std::chrono::steady_clock::now();
    for (const auto& tensor : bundle.manifest.inputs) {
      if (!inputs.count(tensor.name)) {
        std::fprintf(
            stderr, "missing --input for manifest input '%s'\n",
            tensor.name.c_str());
        return 65;
      }
    }

    std::map<std::string, AneWorker::Buffer> outputs;
    AneWorkerReport report = worker.submit(found->second, inputs, &outputs);
    if (report.status != AneWorkerStatus::Completed) {
      std::printf(
          "job status=1 bundle=%s elapsed_ms=%lld detail=%s\n",
          job.bundle.c_str(),
          static_cast<long long>(report.elapsed.count()),
          report.detail.c_str());
      std::fflush(stdout);
      if (worker.quarantined()) {
        std::fprintf(
            stderr, "worker quarantined: %s\n",
            worker.quarantine_reason().c_str());
      }
      return 1;
    }

    size_t output_bytes = 0;
    for (const auto& entry : job.saves) {
      auto produced = outputs.find(entry.first);
      if (produced == outputs.end()) {
        std::fprintf(
            stderr, "cannot save missing output '%s'\n",
            entry.first.c_str());
        return 1;
      }
      write_file(entry.second, produced->second);
      output_bytes += produced->second.size();
    }
    for (const auto& entry : job.expects) {
      auto produced = outputs.find(entry.first);
      if (produced == outputs.end()) {
        std::fprintf(
            stderr, "expected output '%s' was not produced\n",
            entry.first.c_str());
        return 1;
      }
      auto expected = read_file(entry.second);
      if (produced->second.size() != expected.size() ||
          !fp16_values_equal(
              produced->second.data(), expected.data(), expected.size())) {
        std::fprintf(
            stderr, "output '%s' does not match the expected fp16 values\n",
            entry.first.c_str());
        return 1;
      }
    }
    for (const auto& name : job.emits) {
      auto produced = outputs.find(name);
      if (produced == outputs.end()) {
        std::fprintf(
            stderr, "cannot emit missing output '%s'\n", name.c_str());
        return 1;
      }
      std::printf("out %s %zu\n", name.c_str(), produced->second.size());
      std::fflush(stdout);
      if (std::fwrite(
              produced->second.data(), 1, produced->second.size(), stdout) !=
          produced->second.size()) {
        std::fprintf(stderr, "cannot emit output '%s'\n", name.c_str());
        return 1;
      }
      std::fflush(stdout);
      output_bytes += produced->second.size();
    }
    const auto save_ended = std::chrono::steady_clock::now();
    const auto milliseconds = [](auto from, auto to) {
      return static_cast<long long>(
          std::chrono::duration_cast<std::chrono::milliseconds>(to - from)
              .count());
    };
    std::printf(
        "job status=0 bundle=%s elapsed_ms=%lld iterations=%d "
        "input_bytes=%zu output_bytes=%zu stage_ms=%lld save_ms=%lld\n",
        job.bundle.c_str(),
        static_cast<long long>(report.elapsed.count()), report.iterations,
        input_bytes, output_bytes,
        milliseconds(stage_started, stage_ended),
        milliseconds(stage_ended, save_ended) - report.elapsed.count());
    std::fflush(stdout);
  }

  // stdin closed without a quit: release rather than leave the child
  // holding the device.
  AneWorkerReport closed = worker.close();
  if (closed.status != AneWorkerStatus::Completed) {
    std::fprintf(
        stderr, "resident close failed: %s\n", closed.detail.c_str());
    return 1;
  }
  std::printf("resident released programs=%d\n", closed.released_programs);
  std::fflush(stdout);
  return 0;
#endif
}

// Relay-bypass pump: after spawning the resident, splice(2) bytes
// bidirectionally between the relay's own stdin/stdout and the
// resident's socketpair. No parsing, no copy: the wire protocol is
// whatever the runner and resident agreed on, and the relay is a
// kernel pipe between them. One direction per thread; either EOF ends
// the pump and tears the resident down.
int serve_resident_bypass(
    const std::vector<std::pair<std::string, std::string>>& bundle_args,
    const std::string& libane_path,
    long deadline_ms,
    long iterations,
    const std::map<std::string, std::map<std::string, std::string>>&
        seal_expects,
    const std::map<std::string, std::string>& seal_libane_expect) {
#ifndef MLX_OMARCHY_ANE_DEVICE
  (void)bundle_args;
  (void)libane_path;
  (void)deadline_ms;
  (void)iterations;
  (void)seal_expects;
  (void)seal_libane_expect;
  std::fprintf(
      stderr,
      "this binary was built without MLX_OMARCHY_ANE_DEVICE; no device "
      "backend is linked\n");
  return 70;
#else
  ResidentSession session = open_resident_session(
      bundle_args, libane_path, deadline_ms, iterations, seal_expects,
      seal_libane_expect);
  AneWorker& worker = *session.worker;

  AneWorkerReport opened = worker.open(session.bundles, session.session_names);
  if (opened.status != AneWorkerStatus::Completed) {
    std::fprintf(
        stderr, "resident open failed: %s\n", opened.detail.c_str());
    return 1;
  }
  std::printf(
      "relay-bypass ready pid=%lld deadline_ms=%ld detail=%s\n",
      static_cast<long long>(worker.resident_pid()), deadline_ms,
      opened.detail.c_str());
  std::fflush(stdout);

  int channel = worker.channel_fd();
  if (channel < 0) {
    std::fprintf(stderr, "relay-bypass: no resident channel\n");
    return 1;
  }

  // Byte pump, single thread. The wire protocol is strictly
  // half-duplex -- the runner writes one request, then reads one
  // response -- so one poll() loop moves bytes in whichever direction
  // is ready: splice(2) keeps it zero-copy, with no parsing anywhere.
  // Two threads splicing opposite directions of one socketpair was
  // measured unstable on this kernel (the response stalls with both
  // peers blocked), and threads buy nothing for a half-duplex
  // protocol. EOF on stdin is a half-close: shutdown(WR) so the
  // resident's read ends and its release token still travels back.
  // A dying peer surfaces as EPIPE; SIGPIPE is blocked for the whole
  // process, whose only remaining writer is this loop.
  sigset_t pipe_mask;
  sigemptyset(&pipe_mask);
  sigaddset(&pipe_mask, SIGPIPE);
  ::pthread_sigmask(SIG_BLOCK, &pipe_mask, nullptr);

  bool stdin_open = true;
  bool channel_open = true;
  for (;;) {
    struct pollfd fds[2] = {
        {STDIN_FILENO, static_cast<short>(stdin_open ? POLLIN : 0), 0},
        {channel, static_cast<short>(channel_open ? POLLIN : 0), 0},
    };
    int ready = ::poll(fds, 2, -1);
    if (ready < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (stdin_open && (fds[0].revents & (POLLIN | POLLHUP | POLLERR))) {
      ssize_t moved =
          ::splice(STDIN_FILENO, nullptr, channel, nullptr, 1 << 16, 0);
      if (moved > 0) continue;
      // EOF (the runner sent its last frame) or a dead socket: end the
      // request direction so the resident sees a clean end of input.
      (void)::shutdown(channel, SHUT_WR);
      stdin_open = false;
    }
    if (channel_open && (fds[1].revents & (POLLIN | POLLHUP | POLLERR))) {
      ssize_t moved =
          ::splice(channel, nullptr, STDOUT_FILENO, nullptr, 1 << 16, 0);
      if (moved > 0) continue;
      // The resident closed its end: the response is complete.
      channel_open = false;
    }
    if (!stdin_open && !channel_open) break;
  }

  // The pump ended. If the resident already exited (the runner's close
  // came through the pipe and the release token went back), reap it
  // directly -- a second close() would write into a dead socket and
  // report a failure the session never had. The child can close its
  // channel a moment before it is reapable, so give it a bounded
  // grace instead of a single WNOHANG. If it is still alive after
  // that, the runner vanished mid-flight: release through the worker
  // so the programs come off the device.
  pid_t pid = worker.resident_pid();
  int status = 0;
  pid_t reaped = 0;
  for (int tries = 0; tries < 2000; ++tries) {
    reaped = ::waitpid(pid, &status, WNOHANG);
    if (reaped == pid || reaped < 0) break;
    ::usleep(1000);
  }
  if (reaped == pid) {
    return (WIFEXITED(status) && WEXITSTATUS(status) == 0) ? 0 : 1;
  }
  AneWorkerReport closed = worker.close();
  if (closed.status != AneWorkerStatus::Completed) {
    std::fprintf(
        stderr, "relay-bypass: resident close failed: %s\n",
        closed.detail.c_str());
    return 1;
  }
  return 0;
#endif
}

// Resident daemon: same prologue as serve_resident_bypass (one private
// child, one bundle load, one device load, seals checked before device
// load), but successive clients reach the session over an owner-only
// unix socket. One client at a time; the loop ends on --idle-time-ms
// without a connection, SIGTERM/SIGINT, or a lost session. The device
// fd and resident buffer objects stay with this process; only wire
// protocol bytes cross the socket.
namespace {

// Async-signal-safe shutdown flag: SIGTERM / SIGINT handler writes 1,
// the accept loop reads it. A single atomic byte is enough because the
// only writer is a signal handler and the only reader is the listen
// loop on the same thread.
volatile sig_atomic_t g_daemon_stop = 0;

void g_signal_set_stop(int) {
  g_daemon_stop = 1;
}

// Bounded `flock` to keep a second --daemon invocation from claiming
// the same socket. Returns the held fd on success or -1 on failure;
// the caller owns the lifetime of the held fd.
int lock_at(const std::string& lock_path) {
  int fd = ::open(
      lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  if (fd < 0) {
    return -1;
  }
  if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
    int saved = errno;
    ::close(fd);
    errno = saved;
    return -1;
  }
  // Rewrite the lock file with this process's pid for diagnostics.
  if (::ftruncate(fd, 0) == 0) {
    std::string pid_line = std::to_string(::getpid()) + "\n";
    (void)::write(fd, pid_line.data(), pid_line.size());
  }
  return fd;
}

// Payload length from a "<tag> <name> <len>" frame header; 0 when the
// header is malformed (the resident refuses it with a named failure).
size_t frame_payload_length(const std::string& header) {
  auto space = header.rfind(' ');
  if (space == std::string::npos) return 0;
  try {
    return static_cast<size_t>(std::stoull(header.substr(space + 1)));
  } catch (...) {
    return 0;
  }
}

bool send_all(int fd, const char* data, size_t size) {
  while (size > 0) {
    ssize_t written = ::send(fd, data, size, MSG_NOSIGNAL);
    if (written < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    data += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

enum class DetachResult { SessionIntact, SessionLost };

// One client of the daemon. Bytes move with read/write, not splice:
// splice needs one end to be a pipe, and here both ends are sockets.
// The pump tracks the resident wire framing in both directions
// ("submit"/"in <len>"+payload/"run" requests; "iter"/"out
// <len>"+payload/"done"/"failed:" responses) so it knows where a
// request cycle stands when the client goes away:
//
//   - a client "close" at a cycle boundary is a detach: the daemon
//     answers "released" itself and never forwards it, because the
//     resident treats "close" as release-and-exit for the session;
//   - client EOF between cycles leaves the session intact;
//   - client EOF after "run" (client killed mid-submit) drains and
//     discards the in-flight response, bounded by the deadline, so the
//     next client never reads a stale reply;
//   - client EOF inside a partially written request poisons the
//     resident's input stream, so the session is torn down.
DetachResult run_framed_pump(
    int client_fd,
    int channel_fd,
    long deadline_ms,
    std::vector<char>& buffer) {
  enum class Cycle { Idle, Request, Await };
  Cycle cycle = Cycle::Idle;
  std::string client_line;
  std::string reply_line;
  size_t client_payload = 0;
  size_t reply_payload = 0;
  bool client_open = true;
  std::string forward;
  for (;;) {
    if (!client_open) {
      if (cycle == Cycle::Idle && client_payload == 0) {
        return DetachResult::SessionIntact;
      }
      if (cycle == Cycle::Request || client_payload > 0) {
        return DetachResult::SessionLost;
      }
    }
    int timeout_ms = client_open ? -1 : static_cast<int>(deadline_ms + 5000);
    struct pollfd fds[2] = {
        {client_fd, static_cast<short>(client_open ? POLLIN : 0), 0},
        {channel_fd, POLLIN, 0},
    };
    int ready = ::poll(fds, 2, timeout_ms);
    if (ready < 0) {
      if (errno == EINTR) continue;
      return DetachResult::SessionLost;
    }
    if (ready == 0) {
      return DetachResult::SessionLost;  // in-flight reply never finished
    }
    if (fds[1].revents & (POLLIN | POLLHUP | POLLERR)) {
      ssize_t got = ::read(channel_fd, buffer.data(), buffer.size());
      if (got < 0 && errno == EINTR) continue;
      if (got <= 0) {
        return DetachResult::SessionLost;  // resident exited
      }
      const char* cursor = buffer.data();
      size_t left = static_cast<size_t>(got);
      bool resident_failed = false;
      while (left > 0) {
        if (reply_payload > 0) {
          size_t take = std::min(reply_payload, left);
          reply_payload -= take;
          cursor += take;
          left -= take;
          continue;
        }
        auto* newline =
            static_cast<const char*>(std::memchr(cursor, '\n', left));
        if (newline == nullptr) {
          reply_line.append(cursor, left);
          break;
        }
        reply_line.append(cursor, static_cast<size_t>(newline - cursor));
        size_t consumed = static_cast<size_t>(newline - cursor) + 1;
        cursor += consumed;
        left -= consumed;
        if (reply_line.compare(0, 4, "out ") == 0) {
          reply_payload = frame_payload_length(reply_line);
        } else if (reply_line == "done") {
          cycle = Cycle::Idle;
        } else if (reply_line.compare(0, 7, "failed:") == 0) {
          cycle = Cycle::Idle;
          resident_failed = true;  // the resident exits after a failure
        }
        reply_line.clear();
      }
      if (client_open &&
          !send_all(client_fd, buffer.data(), static_cast<size_t>(got))) {
        client_open = false;
      }
      if (resident_failed) {
        return DetachResult::SessionLost;
      }
      continue;
    }
    if (client_open && (fds[0].revents & (POLLIN | POLLHUP | POLLERR))) {
      ssize_t got = ::read(client_fd, buffer.data(), buffer.size());
      if (got < 0 && errno == EINTR) continue;
      if (got <= 0) {
        client_open = false;
        continue;
      }
      forward.clear();
      const char* cursor = buffer.data();
      size_t left = static_cast<size_t>(got);
      while (left > 0) {
        if (client_payload > 0) {
          size_t take = std::min(client_payload, left);
          forward.append(cursor, take);
          client_payload -= take;
          cursor += take;
          left -= take;
          continue;
        }
        auto* newline =
            static_cast<const char*>(std::memchr(cursor, '\n', left));
        if (newline == nullptr) {
          client_line.append(cursor, left);  // held until complete
          break;
        }
        client_line.append(cursor, static_cast<size_t>(newline - cursor));
        size_t consumed = static_cast<size_t>(newline - cursor) + 1;
        cursor += consumed;
        left -= consumed;
        if (client_line == "close" && cycle == Cycle::Idle) {
          if (!send_all(client_fd, "released\n", 9)) {
            client_open = false;
          }
        } else {
          if (client_line.compare(0, 7, "submit ") == 0) {
            cycle = Cycle::Request;
          } else if (client_line.compare(0, 3, "in ") == 0) {
            client_payload = frame_payload_length(client_line);
          } else if (client_line == "run") {
            cycle = Cycle::Await;
          }
          forward.append(client_line);
          forward.push_back('\n');
        }
        client_line.clear();
      }
      if (!forward.empty() &&
          !send_all(channel_fd, forward.data(), forward.size())) {
        return DetachResult::SessionLost;
      }
      continue;
    }
  }
}

} // namespace

int serve_resident_daemon(
    const std::vector<std::pair<std::string, std::string>>& bundle_args,
    const std::string& libane_path,
    long deadline_ms,
    long iterations,
    long idle_time_ms,
    const std::map<std::string, std::map<std::string, std::string>>&
        seal_expects,
    const std::map<std::string, std::string>& seal_libane_expect,
    const std::string& socket_path) {
#ifndef MLX_OMARCHY_ANE_DEVICE
  (void)bundle_args;
  (void)libane_path;
  (void)deadline_ms;
  (void)iterations;
  (void)idle_time_ms;
  (void)seal_expects;
  (void)seal_libane_expect;
  (void)socket_path;
  std::fprintf(
      stderr,
      "this binary was built without MLX_OMARCHY_ANE_DEVICE; no device "
      "backend is linked\n");
  return 70;
#else
  // SIGPIPE is blocked for the whole process: a peer dying mid-pump must
  // surface as EPIPE on send(), not as a process-killing signal.
  sigset_t pipe_mask;
  sigemptyset(&pipe_mask);
  sigaddset(&pipe_mask, SIGPIPE);
  ::pthread_sigmask(SIG_BLOCK, &pipe_mask, nullptr);

  // Signal handlers: SIGTERM and SIGINT set a flag the accept loop
  // reads. Async-signal-safe; the only writer is the kernel's signal
  // delivery path.
  struct sigaction action {};
  action.sa_handler = g_signal_set_stop;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  ::sigaction(SIGTERM, &action, nullptr);
  ::sigaction(SIGINT, &action, nullptr);

  // Single-instance lock: refuse to start if another daemon already
  // holds this socket path. The lock is on `<socket>.lock`; the
  // socket itself is unlinked and re-bound below.
  const std::string lock_path = socket_path + ".lock";
  int lock_fd = lock_at(lock_path);
  if (lock_fd < 0) {
    std::fprintf(
        stderr,
        "[omarchy-ane] daemon: cannot take instance lock at %s: %s\n",
        lock_path.c_str(), std::strerror(errno));
    return 73;
  }

  // Stale socket: a previous run may have crashed before unlinking the
  // socket file. unlink(2) only the socket, not the lock file -- the
  // lock is held by THIS fd's file table entry, not by the on-disk
  // inode, so unlinking the socket does not release the lock.
  ::unlink(socket_path.c_str());

  // Open the resident session: same prologue as --serve / --relay-bypass.
  // The seal contract holds end to end: every bundle and the device
  // library are sealed and bound to the pin BEFORE device load.
  ResidentSession session = open_resident_session(
      bundle_args, libane_path, deadline_ms, iterations, seal_expects,
      seal_libane_expect);
  AneWorker& worker = *session.worker;

  AneWorkerReport opened = worker.open(session.bundles, session.session_names);
  if (opened.status != AneWorkerStatus::Completed) {
    std::fprintf(
        stderr, "resident open failed: %s\n", opened.detail.c_str());
    ::close(lock_fd);
    ::unlink(lock_path.c_str());
    return 1;
  }
  std::printf(
      "daemon ready pid=%lld socket=%s idle_time_ms=%ld\n",
      static_cast<long long>(worker.resident_pid()), socket_path.c_str(),
      idle_time_ms);
  std::fflush(stdout);

  // Bind the unix socket. Mode 0600: only the owning user (this
  // process's euid) can connect. umask is set before bind so the mode
  // is honored regardless of the inherited umask; restored after bind.
  mode_t saved_umask = ::umask(0077);
  int listen_fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listen_fd < 0) {
    int saved_errno = errno;
    ::umask(saved_umask);
    std::fprintf(
        stderr, "[omarchy-ane] daemon: socket(): %s\n",
        std::strerror(saved_errno));
    worker.close();
    ::close(lock_fd);
    ::unlink(lock_path.c_str());
    return 1;
  }
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (socket_path.size() >= sizeof(addr.sun_path)) {
    ::close(listen_fd);
    ::umask(saved_umask);
    std::fprintf(
        stderr, "[omarchy-ane] daemon: socket path too long: %s\n",
        socket_path.c_str());
    worker.close();
    ::close(lock_fd);
    ::unlink(lock_path.c_str());
    return 1;
  }
  std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
  if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    int saved_errno = errno;
    ::close(listen_fd);
    ::umask(saved_umask);
    std::fprintf(
        stderr, "[omarchy-ane] daemon: bind(%s): %s\n",
        socket_path.c_str(), std::strerror(saved_errno));
    worker.close();
    ::close(lock_fd);
    ::unlink(lock_path.c_str());
    return 1;
  }
  ::umask(saved_umask);
  // Belt-and-suspenders: chmod after bind in case the umask was wider
  // than expected (e.g. inherited from the caller).
  ::chmod(socket_path.c_str(), 0600);
  // Refuse to accept if our effective uid is not the socket's owner
  // -- the bind may have created a file owned by a different uid if
  // the parent dir is sticky or the caller had an unusual umask.
  struct stat socket_stat {};
  if (::stat(socket_path.c_str(), &socket_stat) == 0) {
    if (socket_stat.st_uid != ::geteuid() ||
        (socket_stat.st_mode & 0777) != 0600) {
      ::close(listen_fd);
      ::unlink(socket_path.c_str());
      worker.close();
      ::close(lock_fd);
      ::unlink(lock_path.c_str());
      std::fprintf(
          stderr,
          "[omarchy-ane] daemon: refusing to serve a socket with "
          "uid=%d mode=%o\n",
          socket_stat.st_uid, unsigned(socket_stat.st_mode & 0777));
      return 1;
    }
  }
  if (::listen(listen_fd, 1) != 0) {
    int saved_errno = errno;
    ::close(listen_fd);
    ::unlink(socket_path.c_str());
    worker.close();
    ::close(lock_fd);
    ::unlink(lock_path.c_str());
    std::fprintf(
        stderr, "[omarchy-ane] daemon: listen: %s\n",
        std::strerror(saved_errno));
    return 1;
  }

  int channel_fd = worker.channel_fd();
  if (channel_fd < 0) {
    ::close(listen_fd);
    ::unlink(socket_path.c_str());
    worker.close();
    ::close(lock_fd);
    ::unlink(lock_path.c_str());
    std::fprintf(
        stderr, "[omarchy-ane] daemon: no resident channel\n");
    return 1;
  }

  // Accept loop with idle timeout. The pump is single-threaded and
  // serves one client at a time, so concurrent callers serialize at
  // accept(); the listen backlog of 1 reflects the same.
  std::vector<char> pump_buffer(1 << 20);
  bool session_lost = false;
  while (!g_daemon_stop) {
    struct pollfd listen_poll = {listen_fd, POLLIN, 0};
    int ready = ::poll(&listen_poll, 1, static_cast<int>(idle_time_ms));
    if (ready < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (ready == 0) {
      std::fprintf(
          stderr, "[omarchy-ane] daemon: idle exit after %ld ms\n",
          idle_time_ms);
      break;
    }
    sockaddr_un client_addr{};
    socklen_t client_addr_len = sizeof(client_addr);
    int client_fd = ::accept4(
        listen_fd, reinterpret_cast<sockaddr*>(&client_addr),
        &client_addr_len, SOCK_CLOEXEC);
    if (client_fd < 0) {
      if (errno == EINTR) continue;
      // EMFILE / ECONNABORTED: transient, accept the next one.
      if (errno == EMFILE || errno == ENFILE || errno == ECONNABORTED ||
          errno == EAGAIN) {
        continue;
      }
      std::fprintf(
          stderr, "[omarchy-ane] daemon: accept: %s\n",
          std::strerror(errno));
      break;
    }
    // Owner check on the peer: a unix socket client is identified by
    // its connecting end's creds. Reject a cross-uid connection so a
    // single-user install does not become a co-tenant service.
    ucred creds{};
    socklen_t creds_len = sizeof(creds);
    if (::getsockopt(
            client_fd, SOL_SOCKET, SO_PEERCRED, &creds, &creds_len) == 0) {
      if (creds.uid != ::geteuid()) {
        std::fprintf(
            stderr,
            "[omarchy-ane] daemon: refusing peer uid=%d pid=%d "
            "(expected uid=%d)\n",
            int(creds.uid), int(creds.pid), int(::geteuid()));
        ::close(client_fd);
        continue;
      }
    }
    // Per-client handshake: the supervised worker's load banners went
    // to the daemon log at startup, so each client gets one composite
    // line before its first request.
    char banner[256];
    int banner_len = std::snprintf(
        banner,
        sizeof(banner),
        "daemon session pid=%u deadline_ms=%d bundles=%zu\n",
        static_cast<unsigned>(worker.resident_pid()),
        static_cast<int>(deadline_ms),
        session.bundles.size());
    if (banner_len <= 0 ||
        !send_all(client_fd, banner, static_cast<size_t>(banner_len))) {
      ::close(client_fd);
      continue;
    }
    DetachResult detach =
        run_framed_pump(client_fd, channel_fd, deadline_ms, pump_buffer);
    ::shutdown(client_fd, SHUT_RDWR);
    ::close(client_fd);
    if (detach == DetachResult::SessionLost) {
      // The resident exited, failed a submit, or saw a partial request
      // it can no longer frame. Serving another client on it could
      // hand out a stale or corrupt reply; exit so clients fall back
      // to the private path until the daemon is restarted.
      std::fprintf(
          stderr,
          "[omarchy-ane] daemon: session lost (resident pid=%lld); "
          "refusing further clients\n",
          static_cast<long long>(worker.resident_pid()));
      session_lost = true;
      break;
    }
  }

  // Cleanup on every exit path: unbind the socket, release the
  // instance lock, close the resident session. Order matters: the
  // resident close has to come before the lock release, otherwise a
  // racing --daemon would observe the lock free, try to bind the
  // still-resident socket, and race with this process's last writes.
  ::close(listen_fd);
  ::unlink(socket_path.c_str());
  AneWorkerReport closed = worker.close();
  ::close(lock_fd);
  ::unlink(lock_path.c_str());
  if (closed.status != AneWorkerStatus::Completed) {
    std::fprintf(
        stderr, "[omarchy-ane] daemon: resident close failed: %s\n",
        closed.detail.c_str());
    return 1;
  }
  std::printf(
      "daemon released pid=%lld programs=%d\n",
      static_cast<long long>(worker.resident_pid()),
      closed.released_programs);
  std::fflush(stdout);
  return session_lost ? 1 : 0;
#endif
}

// --stop: read `<socket_path>.pid` and SIGTERM the daemon so its
// cleanup path runs end to end (socket unlink, lock release, resident
// close). The actual exit value is reported by the daemon's exit
// message over the wire in a future revision; today we trust the pid
// file and the SIGTERM.
int stop_resident_daemon(const std::string& socket_path) {
  const std::string pid_path = socket_path + ".pid";
  std::ifstream stream(pid_path);
  if (!stream) {
    std::fprintf(
        stderr, "[omarchy-ane] stop: cannot read %s\n", pid_path.c_str());
    return 1;
  }
  std::string pid_line;
  std::getline(stream, pid_line);
  if (pid_line.empty()) {
    std::fprintf(
        stderr, "[omarchy-ane] stop: empty pid in %s\n", pid_path.c_str());
    return 1;
  }
  long pid = 0;
  try {
    pid = std::stol(pid_line);
  } catch (...) {
    std::fprintf(
        stderr, "[omarchy-ane] stop: malformed pid in %s\n", pid_path.c_str());
    return 1;
  }
  if (::kill(static_cast<pid_t>(pid), SIGTERM) != 0) {
    std::fprintf(
        stderr, "[omarchy-ane] stop: kill(%ld, SIGTERM): %s\n", pid,
        std::strerror(errno));
    return 1;
  }
  std::printf("daemon stop signalled pid=%ld\n", pid);
  std::fflush(stdout);
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  std::string bundle_dir;
  std::string libane_path;
  long deadline_ms = 2000;
  long iterations = 1;
  bool serve = false;
  bool relay_bypass = false;
  bool daemon_mode = false;
  bool stop_mode = false;
  long idle_time_ms = 60000;
  std::string socket_path;
  std::vector<std::pair<std::string, std::string>> resident_bundles;
  std::map<std::string, std::string> input_files;
  std::map<std::string, std::string> expect_files;
  std::map<std::string, std::string> save_files;
  std::map<std::string, std::map<std::string, std::string>> seal_expects;
  std::map<std::string, std::string> seal_libane_expect;

  for (int i = 1; i < argc; ++i) {
    std::string flag = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", flag.c_str());
        std::exit(64);
      }
      return argv[++i];
    };
    if (flag == "--serve") {
      serve = true;
    } else if (flag == "--relay-bypass") {
      relay_bypass = true;
    } else if (flag == "--daemon") {
      daemon_mode = true;
    } else if (flag == "--stop") {
      stop_mode = true;
    } else if (flag == "--socket") {
      socket_path = value();
    } else if (flag == "--idle-time-ms") {
      idle_time_ms = std::stol(value());
    } else if (flag == "--bundle") {
      auto assignment = value();
      std::string name;
      std::string dir;
      if (split_assignment(assignment, name, dir)) {
        resident_bundles.emplace_back(name, dir);
      } else {
        bundle_dir = assignment;
      }
    } else if (flag == "--libane") {
      libane_path = value();
    } else if (flag == "--deadline-ms") {
      deadline_ms = std::stol(value());
    } else if (flag == "--iterations") {
      iterations = std::stol(value());
    } else if (flag == "--input") {
      auto assignment = value();
      auto sep = assignment.find('=');
      if (sep == std::string::npos) {
        std::fprintf(stderr, "--input expects NAME=FILE\n");
        return usage();
      }
      input_files[assignment.substr(0, sep)] = assignment.substr(sep + 1);
    } else if (flag == "--expect") {
      auto assignment = value();
      auto sep = assignment.find('=');
      if (sep == std::string::npos) {
        std::fprintf(stderr, "--expect expects NAME=FILE\n");
        return usage();
      }
      expect_files[assignment.substr(0, sep)] = assignment.substr(sep + 1);
    } else if (flag == "--save") {
      auto assignment = value();
      auto sep = assignment.find('=');
      if (sep == std::string::npos) {
        std::fprintf(stderr, "--save expects NAME=FILE\n");
        return usage();
      }
      save_files[assignment.substr(0, sep)] = assignment.substr(sep + 1);
    } else if (flag == "--seal-expect") {
      auto assignment = value();
      auto colon = assignment.find(':');
      auto sep = assignment.rfind('=');
      if (colon == std::string::npos || colon == 0 || sep == std::string::npos ||
          sep <= colon + 1 ||
          !is_sha256_hex(assignment.substr(sep + 1))) {
        std::fprintf(
            stderr, "--seal-expect expects BUNDLE:FILE=SHA256\n");
        return usage();
      }
      seal_expects[assignment.substr(0, colon)][assignment.substr(
          colon + 1, sep - colon - 1)] = assignment.substr(sep + 1);
    } else if (flag == "--seal-expect-libane") {
      auto assignment = value();
      auto sep = assignment.rfind('=');
      if (sep == std::string::npos || sep == 0 ||
          !is_sha256_hex(assignment.substr(sep + 1))) {
        std::fprintf(
            stderr, "--seal-expect-libane expects FILE=SHA256\n");
        return usage();
      }
      seal_libane_expect[assignment.substr(0, sep)] =
          assignment.substr(sep + 1);
    } else {
      return usage();
    }
  }
  // --stop is independent of --libane / bundles; it only reads the
  // pid file at <socket_path>.pid.
  if (stop_mode) {
    if (socket_path.empty()) {
      std::fprintf(
          stderr, "--stop requires --socket PATH\n");
      return usage();
    }
    return stop_resident_daemon(socket_path);
  }
  if (libane_path.empty()) {
    return usage();
  }
  if (serve || relay_bypass || daemon_mode) {
    if (resident_bundles.empty()) {
      std::fprintf(
          stderr, "--serve / --relay-bypass / --daemon require at "
                  "least one --bundle NAME=DIR\n");
      return usage();
    }
    int mode_count =
        (serve ? 1 : 0) + (relay_bypass ? 1 : 0) + (daemon_mode ? 1 : 0);
    if (mode_count > 1) {
      std::fprintf(
          stderr, "--serve, --relay-bypass and --daemon are mutually "
                  "exclusive\n");
      return usage();
    }
    if (daemon_mode) {
      if (socket_path.empty()) {
        std::fprintf(
            stderr, "--daemon requires --socket PATH\n");
        return usage();
      }
      if (idle_time_ms <= 0) {
        std::fprintf(
            stderr, "--daemon requires --idle-time-ms N with N > 0\n");
        return usage();
      }
      // The instance-lock / connect protocol calls retain the seal
      // contract end to end. Write the pid file before binding so a
      // racing --stop sees a complete daemon entry.
      std::ofstream pid_file(socket_path + ".pid");
      if (pid_file) {
        pid_file << ::getpid() << "\n";
        pid_file.flush();
        pid_file.close();
        ::chmod((socket_path + ".pid").c_str(), 0600);
      }
      int rc = serve_resident_daemon(
          resident_bundles, libane_path, deadline_ms, iterations,
          idle_time_ms, seal_expects, seal_libane_expect, socket_path);
      ::unlink((socket_path + ".pid").c_str());
      return rc;
    }
    // A pinned name that is not part of this session would silently
    // leave that bundle unpinned: refuse instead of guessing.
    for (const auto& [name, files] : seal_expects) {
      if (!std::any_of(
              resident_bundles.begin(),
              resident_bundles.end(),
              [&](const std::pair<std::string, std::string>& bundle) {
                return bundle.first == name;
              })) {
        std::fprintf(
            stderr,
            "--seal-expect names unknown resident bundle '%s'\n",
            name.c_str());
        return usage();
      }
    }
    try {
      if (relay_bypass) {
        return serve_resident_bypass(
            resident_bundles, libane_path, deadline_ms, iterations,
            seal_expects, seal_libane_expect);
      }
      return serve_resident(
          resident_bundles, libane_path, deadline_ms, iterations,
          seal_expects, seal_libane_expect);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "error: %s\n", error.what());
      return 1;
    }
  }
  if (!seal_expects.empty() || !seal_libane_expect.empty()) {
    std::fprintf(
        stderr,
        "--seal-expect requires --serve, --relay-bypass or --daemon\n");
    return usage();
  }
  if (!resident_bundles.empty()) {
    std::fprintf(
        stderr,
        "--bundle NAME=DIR names a resident bundle and requires --serve\n");
    return usage();
  }
  if (bundle_dir.empty()) {
    return usage();
  }

  try {
    AneBundle bundle = load_bundle(bundle_dir);
    std::printf(
        "bundle name=%s programs=%zu driver_abi=%llu graph=%s\n",
        bundle.manifest.name.c_str(), bundle.programs.size(),
        static_cast<unsigned long long>(bundle.manifest.driver_abi_major),
        bundle.manifest.graph_hash.c_str());

    std::map<std::string, AneWorker::Buffer> inputs;
    for (const auto& entry : input_files) {
      inputs[entry.first] = read_file(entry.second);
    }
    for (const auto& tensor : bundle.manifest.inputs) {
      if (!inputs.count(tensor.name)) {
        std::fprintf(
            stderr, "missing --input for manifest input '%s'\n",
            tensor.name.c_str());
        return 65;
      }
    }

    AneWorkerOptions options;
    options.deadline = std::chrono::milliseconds(deadline_ms);
    options.iterations = static_cast<int>(iterations);

#ifdef MLX_OMARCHY_ANE_DEVICE
    AneWorker worker(
        [libane_path] { return make_libane_device(libane_path); }, options);
#else
    std::fprintf(
        stderr,
        "this binary was built without MLX_OMARCHY_ANE_DEVICE; no device "
        "backend is linked\n");
    return 70;
#endif

    std::map<std::string, AneWorker::Buffer> outputs;
    AneWorkerReport report = worker.run(bundle, inputs, &outputs);
    std::printf(
        "worker status=%d iterations=%d released=%d elapsed_ms=%lld\n",
        static_cast<int>(report.status), report.iterations,
        report.released_programs,
        static_cast<long long>(report.elapsed.count()));
    std::printf("worker detail=%s\n", report.detail.c_str());
    if (worker.quarantined()) {
      std::printf("worker quarantined: %s\n",
                  worker.quarantine_reason().c_str());
    }

    if (report.status != AneWorkerStatus::Completed) {
      return 1;
    }

    for (const auto& entry : save_files) {
      auto found = outputs.find(entry.first);
      if (found == outputs.end()) {
        std::fprintf(
            stderr, "cannot save missing output '%s'\n",
            entry.first.c_str());
        return 1;
      }
      std::ofstream stream(entry.second, std::ios::binary);
      stream.write(
          reinterpret_cast<const char*>(found->second.data()),
          static_cast<std::streamsize>(found->second.size()));
      if (!stream) {
        std::fprintf(stderr, "cannot write %s\n", entry.second.c_str());
        return 1;
      }
      std::printf("saved output %s (%zu bytes)\n", entry.first.c_str(),
                  found->second.size());
    }
    for (const auto& entry : expect_files) {
      auto found = outputs.find(entry.first);
      if (found == outputs.end()) {
        std::fprintf(
            stderr, "expected output '%s' was not produced\n",
            entry.first.c_str());
        return 1;
      }
      auto expected = read_file(entry.second);
      if (found->second.size() != expected.size() ||
          !fp16_values_equal(
              found->second.data(), expected.data(), expected.size())) {
        std::fprintf(
            stderr, "output '%s' does not match the expected fp16 values\n",
            entry.first.c_str());
        return 1;
      }
      std::printf("verified output %s exact\n", entry.first.c_str());
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
