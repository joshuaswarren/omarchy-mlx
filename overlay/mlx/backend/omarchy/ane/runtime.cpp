// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/runtime.h"

#include "mlx/backend/omarchy/ane/bundle.h"
#include "mlx/backend/omarchy/ane/runtime_detail.h"
#include "mlx/backend/omarchy/ane/runtime_ownership.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <linux/memfd.h>
#include <map>
#include <mutex>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char** environ;

#if !defined(MLX_OMARCHY_ANE_LIBRARY_BUILD_PATH) || \
    !defined(MLX_OMARCHY_ANE_WORKER_BUILD_PATH)
#error "private ANE build paths must be defined"
#endif

namespace mlx::core::omarchy::ane {
namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

std::string system_error(const std::string& operation, int error = errno) {
  return operation + " failed: " + std::strerror(error);
}

const char kRuntimeImageAnchor = 0;

std::filesystem::path loaded_runtime_image() {
  Dl_info image {};
  if (::dladdr(&kRuntimeImageAnchor, &image) == 0 || image.dli_fname == nullptr) {
    throw detail::runtime_error("cannot resolve loaded libmlx image");
  }
  std::error_code error;
  auto path = std::filesystem::canonical(image.dli_fname, error);
  if (error) {
    throw detail::runtime_error("cannot resolve loaded libmlx image");
  }
  return path;
}

size_t staging_size(const AneManifest& manifest) {
  uint64_t size = 0;
  auto add = [&](const std::vector<AneTensor>& tensors) {
    for (const auto& tensor : tensors) {
      if (tensor.byte_size > std::numeric_limits<uint64_t>::max() - size) {
        throw detail::runtime_error("host staging size overflows uint64");
      }
      size += tensor.byte_size;
    }
  };
  add(manifest.inputs);
  add(manifest.outputs);
  return detail::checked_size(size, "host staging size");
}

int remaining_milliseconds(TimePoint deadline) {
  auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
      deadline - Clock::now());
  if (remaining.count() <= 0) {
    return 0;
  }
  if (remaining.count() > std::numeric_limits<int>::max()) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(remaining.count());
}

bool wait_ready(int fd, short events, TimePoint deadline) {
  while (true) {
    pollfd descriptor{fd, events, 0};
    int result = ::poll(&descriptor, 1, remaining_milliseconds(deadline));
    if (result > 0) {
      return (descriptor.revents & events) != 0;
    }
    if (result == 0) {
      return false;
    }
    if (errno != EINTR) {
      throw detail::runtime_error(system_error("poll"));
    }
  }
}

void send_command(int fd, const detail::WorkerCommand& command, TimePoint deadline) {
  if (!wait_ready(fd, POLLOUT, deadline)) {
    throw detail::runtime_error("deadline expired before worker command could be sent");
  }
  ssize_t sent = ::send(fd, &command, sizeof(command), MSG_NOSIGNAL);
  if (sent != static_cast<ssize_t>(sizeof(command))) {
    throw detail::runtime_error(system_error("worker command send"));
  }
}

detail::WorkerReply receive_reply(int fd, TimePoint deadline) {
  if (!wait_ready(fd, POLLIN, deadline)) {
    throw detail::runtime_error("worker completion deadline expired");
  }
  detail::WorkerReply reply;
  ssize_t received = ::recv(fd, &reply, sizeof(reply), 0);
  if (received == 0) {
    throw detail::runtime_error("worker exited without a completion reply");
  }
  if (received != static_cast<ssize_t>(sizeof(reply))) {
    throw detail::runtime_error(system_error("worker reply receive"));
  }
  if (reply.version != detail::kWorkerProtocolVersion) {
    throw detail::runtime_error("worker protocol version mismatch");
  }
  reply.detail[detail::kWorkerDetailBytes - 1] = '\0';
  return reply;
}

bool wait_for_exit(pid_t pid, TimePoint deadline) {
  while (Clock::now() < deadline) {
    int status = 0;
    pid_t result = ::waitpid(pid, &status, WNOHANG);
    if (result == pid) {
      return true;
    }
    if (result < 0) {
      return errno == ECHILD;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return false;
}

int duplicate_owned_fd(int fd, int minimum) {
  int duplicate = ::fcntl(fd, F_DUPFD_CLOEXEC, minimum);
  if (duplicate < 0) {
    throw detail::runtime_error(system_error("file descriptor duplication"));
  }
  return duplicate;
}

class OwnedFd {
 public:
  explicit OwnedFd(int fd = -1) : fd_(fd) {}
  ~OwnedFd() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }
  OwnedFd(const OwnedFd&) = delete;
  OwnedFd& operator=(const OwnedFd&) = delete;
  OwnedFd(OwnedFd&& other) noexcept : fd_(other.fd_) {
    other.fd_ = -1;
  }
  OwnedFd& operator=(OwnedFd&& other) noexcept {
    if (this != &other) {
      if (fd_ >= 0) {
        ::close(fd_);
      }
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }
  int get() const {
    return fd_;
  }

 private:
  int fd_;
};

OwnedFd snapshot_file_at(
    int directory_fd,
    const std::string& name,
    TimePoint deadline) {
  OwnedFd source(::openat(
      directory_fd, name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (source.get() < 0) {
    throw detail::runtime_error(system_error("bundle snapshot open " + name));
  }
  struct stat status {};
  if (::fstat(source.get(), &status) != 0 || !S_ISREG(status.st_mode)) {
    throw detail::runtime_error("bundle snapshot source is not a regular file: " + name);
  }
  OwnedFd snapshot(::memfd_create(
      "mlx-omarchy-ane-bundle", MFD_CLOEXEC | MFD_ALLOW_SEALING));
  if (snapshot.get() < 0) {
    throw detail::runtime_error(system_error("bundle snapshot memfd_create"));
  }
  std::array<uint8_t, 64 * 1024> bytes{};
  while (true) {
    if (Clock::now() >= deadline) {
      throw detail::runtime_error("startup deadline expired while freezing bundle bytes");
    }
    ssize_t count = ::read(source.get(), bytes.data(), bytes.size());
    if (count == 0) {
      break;
    }
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw detail::runtime_error(system_error("bundle snapshot read " + name));
    }
    size_t written = 0;
    while (written < static_cast<size_t>(count)) {
      ssize_t result = ::write(
          snapshot.get(), bytes.data() + written, size_t(count) - written);
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result <= 0) {
        throw detail::runtime_error(system_error("bundle snapshot write " + name));
      }
      written += static_cast<size_t>(result);
    }
  }
  if (::lseek(snapshot.get(), 0, SEEK_SET) < 0 ||
      ::fcntl(
          snapshot.get(),
          F_ADD_SEALS,
          F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) != 0) {
    throw detail::runtime_error(system_error("bundle snapshot seal " + name));
  }
  return snapshot;
}

struct FrozenBundle {
  AneBundle bundle;
  OwnedFd manifest;
  std::vector<OwnedFd> payloads;
  std::string contract_sha256;
};

void validate_bundle_directory(
    int directory_fd,
    const AneManifest& manifest,
    TimePoint deadline) {
  int scan_fd = ::fcntl(directory_fd, F_DUPFD_CLOEXEC, 0);
  if (scan_fd < 0) {
    throw detail::runtime_error(system_error("bundle snapshot directory duplication"));
  }
  DIR* stream = ::fdopendir(scan_fd);
  if (stream == nullptr) {
    ::close(scan_fd);
    throw detail::runtime_error(system_error("bundle snapshot directory stream"));
  }
  auto close_directory = [](DIR* directory) { ::closedir(directory); };
  std::unique_ptr<DIR, decltype(close_directory)> entries(stream, close_directory);

  while (true) {
    if (Clock::now() >= deadline) {
      throw detail::runtime_error(
          "startup deadline expired while validating bundle directory");
    }
    errno = 0;
    dirent* entry = ::readdir(entries.get());
    if (entry == nullptr) {
      if (errno != 0) {
        throw detail::runtime_error(system_error("bundle snapshot directory read"));
      }
      break;
    }
    const std::string name(entry->d_name);
    if (name == "." || name == "..") {
      continue;
    }

    struct stat status {};
    if (::fstatat(
            directory_fd,
            name.c_str(),
            &status,
            AT_SYMLINK_NOFOLLOW) != 0) {
      throw std::runtime_error(
          "[omarchy-ane] bundle: cannot stat '" + name + "' inside bundle.");
    }
    if (S_ISLNK(status.st_mode)) {
      throw std::runtime_error(
          "[omarchy-ane] bundle: unexpected link '" + name + "' inside bundle.");
    }
    if (S_ISDIR(status.st_mode)) {
      throw std::runtime_error(
          "[omarchy-ane] bundle: unexpected directory '" + name + "' inside bundle.");
    }
    if (!S_ISREG(status.st_mode)) {
      throw std::runtime_error(
          "[omarchy-ane] bundle: unexpected non-regular file '" + name +
          "' inside bundle.");
    }
    if (name != "manifest.json" &&
        std::none_of(
            manifest.payloads.begin(),
            manifest.payloads.end(),
            [&](const AnePayload& payload) { return payload.path == name; })) {
      throw std::runtime_error(
          "[omarchy-ane] bundle: unknown payload file '" + name +
          "' not listed in manifest.");
    }
  }
}

FrozenBundle freeze_bundle(
    const std::filesystem::path& directory,
    TimePoint deadline) {
  OwnedFd directory_fd(::open(
      directory.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_DIRECTORY));
  if (directory_fd.get() < 0) {
    if (errno == ENOENT || errno == ENOTDIR) {
      throw AneBundleNotFound(
          "[omarchy-ane] bundle directory not found: " + directory.string() +
          " (the affected region stays on Vulkan)");
    }
    throw detail::runtime_error(system_error("bundle snapshot directory open"));
  }

  FrozenBundle frozen;
  frozen.manifest =
      snapshot_file_at(directory_fd.get(), "manifest.json", deadline);
  const auto manifest_path = std::filesystem::path("/proc/self/fd") /
      std::to_string(frozen.manifest.get());
  const AneManifest manifest = parse_ane_manifest(manifest_path);
  validate_bundle_directory(directory_fd.get(), manifest, deadline);
  frozen.payloads.reserve(manifest.payloads.size());
  std::map<std::string, std::filesystem::path> paths;
  for (const auto& payload : manifest.payloads) {
    frozen.payloads.push_back(
        snapshot_file_at(directory_fd.get(), payload.path, deadline));
    paths.emplace(
        payload.path,
        std::filesystem::path("/proc/self/fd") /
            std::to_string(frozen.payloads.back().get()));
  }
  frozen.bundle = load_bundle_snapshot(manifest_path, paths);
  frozen.contract_sha256 = sha256_file(manifest_path);
  return frozen;
}

void ensure_before(TimePoint deadline, const std::string& phase) {
  if (Clock::now() >= deadline) {
    throw detail::runtime_error("deadline expired before " + phase);
  }
}

class AneChildReaper {
 public:
  AneChildReaper() : thread_([this] { run(); }) {}

  void adopt(pid_t pid) {
    std::lock_guard<std::mutex> lock(mutex_);
    pids_.push_back(pid);
    ready_.notify_one();
  }

 private:
  void run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      ready_.wait(lock, [&] { return !pids_.empty(); });
      bool reaped = false;
      for (auto pid = pids_.begin(); pid != pids_.end();) {
        int status = 0;
        pid_t result = ::waitpid(*pid, &status, WNOHANG);
        if (result == *pid || (result < 0 && errno == ECHILD)) {
          pid = pids_.erase(pid);
          reaped = true;
        } else {
          ++pid;
        }
      }
      if (!reaped && !pids_.empty()) {
        ready_.wait_for(lock, std::chrono::milliseconds(10));
      }
    }
  }

  std::vector<pid_t> pids_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::thread thread_;
};

AneChildReaper& child_reaper() {
  static auto* reaper = new AneChildReaper();
  return *reaper;
}

} // namespace

struct AneRuntime::Impl {
  AneBundle bundle;
  std::filesystem::path diagnostic_path;
  int control_fd{-1};
  int staging_fd{-1};
  uint8_t* staging{nullptr};
  size_t staging_bytes{0};
  pid_t pid{-1};
  uint64_t serial{0};
  bool is_usable{false};
  bool current_operation_submitted{false};
  std::string identity;
  std::timed_mutex transaction_mutex;
  detail::RuntimeOwnership ownership;

  ~Impl() {
    if (control_fd >= 0) {
      ::close(control_fd);
    }
    if (staging != nullptr) {
      ::munmap(staging, staging_bytes);
    }
    if (staging_fd >= 0) {
      ::close(staging_fd);
    }
    if (pid > 0) {
      try {
        child_reaper().adopt(pid);
      } catch (...) {
      }
    }
  }

  void preserve_diagnostic(const std::string& reason, bool uncertain) noexcept {
    try {
      if (!diagnostic_path.parent_path().empty()) {
        std::filesystem::create_directories(diagnostic_path.parent_path());
      }
      std::ofstream output(diagnostic_path, std::ios::app);
      output << "reason=" << reason << '\n'
             << "graph_hash=" << bundle.manifest.graph_hash << '\n'
             << "worker_pid=" << pid << '\n'
             << "current_operation_submitted="
             << (current_operation_submitted ? "true" : "false") << '\n'
             << "runtime_identity=" << identity << '\n'
             << "recovery="
             << (uncertain
                     ? "stop ANE submissions; uncertain completion requires reboot"
                     : "worker control closed before another submission; hardware recovery not required")
             << '\n';
    } catch (...) {
    }
  }

  void mark_unusable(std::string reason, bool uncertain) noexcept {
    is_usable = false;
    if (control_fd >= 0) {
      ::close(control_fd);
      control_fd = -1;
    }
    bool requires_reboot = uncertain;
    if (requires_reboot) {
      ownership.quarantine();
    } else {
      try {
        ownership.release_cleanly();
      } catch (const std::exception& error) {
        requires_reboot = true;
        ownership.quarantine();
        reason += "; ownership state cleanup failed: ";
        reason += error.what();
      }
    }
    preserve_diagnostic(reason, requires_reboot);
    if (pid > 0) {
      try {
        child_reaper().adopt(pid);
        pid = -1;
      } catch (...) {
      }
    }
  }

  static std::unique_ptr<Impl> start(
      FrozenBundle frozen,
      TimePoint deadline,
      const std::filesystem::path& diagnostic) {
    ensure_before(deadline, "runtime initialization");
    auto implementation = std::make_unique<Impl>();
    implementation->bundle = std::move(frozen.bundle);
    implementation->diagnostic_path = diagnostic;
    if (implementation->diagnostic_path.empty()) {
      throw std::invalid_argument("ANE diagnostic path must not be empty");
    }
    implementation->ownership = detail::RuntimeOwnership::acquire();
    if (!implementation->diagnostic_path.parent_path().empty()) {
      std::filesystem::create_directories(
          implementation->diagnostic_path.parent_path());
    }
    {
      std::ofstream output(implementation->diagnostic_path, std::ios::app);
      if (!output) {
        throw detail::runtime_error("ANE diagnostic path is not writable");
      }
      output << "runtime_load_graph_hash="
             << implementation->bundle.manifest.graph_hash << '\n';
      if (!output) {
        throw detail::runtime_error("ANE diagnostic path write failed");
      }
    }
    implementation->staging_bytes = staging_size(implementation->bundle.manifest);
    rlimit file_limit{};
    if (::getrlimit(RLIMIT_NOFILE, &file_limit) != 0) {
      throw detail::runtime_error(system_error("worker descriptor limit"));
    }
    const uint64_t descriptor_limit = file_limit.rlim_cur == RLIM_INFINITY
        ? uint64_t{std::numeric_limits<int>::max()}
        : static_cast<uint64_t>(file_limit.rlim_cur);
    const int source_fd_floor = detail::worker_source_fd_floor(
        frozen.payloads.size(), descriptor_limit);
    OwnedFd worker_manifest(
        duplicate_owned_fd(frozen.manifest.get(), source_fd_floor));
    OwnedFd worker_lock(
        duplicate_owned_fd(implementation->ownership.lock_fd(), source_fd_floor));
    std::vector<OwnedFd> worker_payloads;
    worker_payloads.reserve(frozen.payloads.size());
    for (const auto& payload : frozen.payloads) {
      worker_payloads.emplace_back(
          duplicate_owned_fd(payload.get(), source_fd_floor));
    }

    OwnedFd memory(::memfd_create(
        "mlx-omarchy-ane", MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (memory.get() < 0) {
      throw detail::runtime_error(system_error("memfd_create"));
    }
    implementation->staging_fd = duplicate_owned_fd(memory.get(), source_fd_floor);
    if (::ftruncate(
            implementation->staging_fd,
            static_cast<off_t>(implementation->staging_bytes)) != 0) {
      throw detail::runtime_error(system_error("host staging allocation"));
    }
    void* mapping = ::mmap(
        nullptr,
        implementation->staging_bytes,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        implementation->staging_fd,
        0);
    if (mapping == MAP_FAILED) {
      throw detail::runtime_error(system_error("host staging mmap"));
    }
    implementation->staging = static_cast<uint8_t*>(mapping);
    std::string executable = detail::worker_executable_path(
        loaded_runtime_image(),
        MLX_OMARCHY_ANE_LIBRARY_BUILD_PATH,
        MLX_OMARCHY_ANE_WORKER_BUILD_PATH).string();
    int sockets[2];
    if (::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sockets) != 0) {
      throw detail::runtime_error(system_error("worker socketpair"));
    }
    OwnedFd parent_socket(sockets[0]);
    OwnedFd child_socket(sockets[1]);
    implementation->control_fd =
        duplicate_owned_fd(parent_socket.get(), source_fd_floor);
    OwnedFd child_control(
        duplicate_owned_fd(child_socket.get(), source_fd_floor));

    posix_spawn_file_actions_t actions;
    int spawn_error = ::posix_spawn_file_actions_init(&actions);
    if (spawn_error != 0) {
      throw detail::runtime_error(system_error("posix_spawn_file_actions_init", spawn_error));
    }
    auto destroy_actions = [&] { ::posix_spawn_file_actions_destroy(&actions); };
    spawn_error = ::posix_spawn_file_actions_adddup2(
        &actions, child_control.get(), detail::kWorkerControlFd);
    if (spawn_error == 0) {
      spawn_error = ::posix_spawn_file_actions_adddup2(
          &actions, implementation->staging_fd, detail::kWorkerStagingFd);
    }
    if (spawn_error == 0) {
      spawn_error = ::posix_spawn_file_actions_adddup2(
          &actions, worker_manifest.get(), detail::kWorkerManifestFd);
    }
    if (spawn_error == 0) {
      spawn_error = ::posix_spawn_file_actions_adddup2(
          &actions, worker_lock.get(), detail::kWorkerHardwareLockFd);
    }
    for (size_t i = 0; spawn_error == 0 && i < worker_payloads.size(); ++i) {
      spawn_error = ::posix_spawn_file_actions_adddup2(
          &actions,
          worker_payloads[i].get(),
          detail::kWorkerPayloadFdBase + static_cast<int>(i));
    }
    if (spawn_error == 0) {
      spawn_error = ::posix_spawn_file_actions_addclose(
          &actions, implementation->control_fd);
    }
    if (spawn_error != 0) {
      destroy_actions();
      throw detail::runtime_error(system_error("worker file actions", spawn_error));
    }

    std::string size = std::to_string(implementation->staging_bytes);
    char* arguments[] = {executable.data(), size.data(), nullptr};
    if (Clock::now() >= deadline) {
      destroy_actions();
      throw detail::runtime_error("deadline expired before worker spawn");
    }
    spawn_error = ::posix_spawn(
        &implementation->pid,
        executable.c_str(),
        &actions,
        nullptr,
        arguments,
        environ);
    destroy_actions();
    if (spawn_error != 0) {
      implementation->pid = -1;
      throw detail::runtime_error(system_error("ANE worker spawn", spawn_error));
    }
    implementation->ownership.arm();

    detail::WorkerReply reply;
    try {
      ensure_before(deadline, "worker readiness wait");
      reply = receive_reply(implementation->control_fd, deadline);
    } catch (const std::exception& error) {
      implementation->mark_unusable(
          std::string("worker startup failed: ") + error.what(), false);
      throw;
    }
    if (reply.kind != detail::WorkerReplyKind::ready) {
      implementation->mark_unusable(reply.detail, false);
      throw detail::runtime_error(std::string(reply.detail));
    }
    const std::string identity_prefix =
        "graph_hash=" + implementation->bundle.manifest.graph_hash +
        "\ncontract_sha256=" + frozen.contract_sha256 +
        "\nmodel_sha256=" +
        implementation->bundle.manifest.release_asset.model_sha256 + "\n";
    const std::string worker_detail(reply.detail);
    if (worker_detail.rfind(identity_prefix, 0) != 0) {
      implementation->mark_unusable(
          "worker loaded a different bundle identity", false);
      throw detail::runtime_error("worker loaded a different bundle identity");
    }
    implementation->identity = worker_detail;
    implementation->is_usable = true;
    return implementation;
  }
};

AneRuntime::AneRuntime(std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

std::unique_ptr<AneRuntime> AneRuntime::load(
    const std::filesystem::path& bundle,
    std::chrono::milliseconds startup_deadline,
    const std::filesystem::path& diagnostic_path) {
  const auto deadline = detail::checked_deadline(startup_deadline);
  FrozenBundle frozen = freeze_bundle(bundle, deadline.time);
  ensure_before(deadline.time, "worker startup");
  return std::unique_ptr<AneRuntime>(new AneRuntime(
      Impl::start(std::move(frozen), deadline.time, diagnostic_path)));
}

AneRuntime::~AneRuntime() {
  if (implementation_ && usable()) {
    try {
      shutdown(std::chrono::seconds(5));
    } catch (...) {
    }
  }
}

AneBufferMap AneRuntime::execute(
    const AneBufferMap& inputs,
    std::chrono::milliseconds deadline_duration) {
  const auto checked_deadline = detail::checked_deadline(deadline_duration);
  const TimePoint deadline = checked_deadline.time;
  std::unique_lock<std::timed_mutex> transaction(
      implementation_->transaction_mutex, std::defer_lock);
  if (!transaction.try_lock_until(deadline)) {
    throw detail::runtime_error("deadline expired waiting for runtime transaction");
  }
  if (!implementation_->is_usable) {
    throw detail::runtime_error("runtime is unusable; no ANE work was submitted");
  }
  const auto& expected = implementation_->bundle.manifest.inputs;
  if (inputs.size() != expected.size()) {
    throw detail::runtime_error("inputs do not exactly match the bundle manifest");
  }

  size_t offset = 0;
  for (const auto& tensor : expected) {
    auto found = inputs.find(tensor.name);
    if (found == inputs.end() || found->second.size() != tensor.byte_size) {
      throw detail::runtime_error(
          "input '" + tensor.name + "' byte count does not match the bundle manifest");
    }
    std::memcpy(
        implementation_->staging + offset,
        found->second.data(),
        found->second.size());
    offset += found->second.size();
  }

  ensure_before(deadline, "worker command submission");
  detail::WorkerCommand command;
  command.operation = detail::WorkerOperation::execute;
  command.serial = ++implementation_->serial;
  command.deadline_monotonic_nanoseconds = checked_deadline.monotonic_nanoseconds;
  implementation_->current_operation_submitted = false;
  try {
    send_command(implementation_->control_fd, command, deadline);
  } catch (const std::exception& error) {
    implementation_->mark_unusable(
        std::string("worker command was not sent: ") + error.what(), false);
    throw;
  }
  implementation_->current_operation_submitted = true;

  detail::WorkerReply reply;
  try {
    reply = receive_reply(implementation_->control_fd, deadline);
  } catch (const std::exception& error) {
    implementation_->mark_unusable(
        std::string("completion became uncertain: ") + error.what(), true);
    throw detail::runtime_error(
        std::string(error.what()) +
        "; runtime marked unusable; no signal was sent to the worker, and reboot is required before further ANE use");
  }
  if (reply.serial != command.serial) {
    implementation_->mark_unusable("worker reply serial mismatch", true);
    throw detail::runtime_error("worker reply serial mismatch; runtime marked unusable");
  }
  if (reply.kind == detail::WorkerReplyKind::uncertain) {
    implementation_->mark_unusable(reply.detail, true);
    throw detail::runtime_error(
        std::string(reply.detail) +
        "; runtime marked unusable and reboot is required before further ANE use");
  }
  if (reply.kind != detail::WorkerReplyKind::executed) {
    implementation_->current_operation_submitted = false;
    implementation_->mark_unusable(reply.detail, false);
    throw detail::runtime_error(
        std::string(reply.detail) + "; runtime marked unusable");
  }

  AneBufferMap outputs;
  for (const auto& tensor : implementation_->bundle.manifest.outputs) {
    AneBuffer data(detail::checked_size(tensor.byte_size, "output byte count"));
    std::memcpy(data.data(), implementation_->staging + offset, data.size());
    offset += data.size();
    outputs.emplace(tensor.name, std::move(data));
  }
  implementation_->current_operation_submitted = false;
  return outputs;
}

const std::vector<AneLogicalResult>& AneRuntime::output_layout() const {
  return implementation_->bundle.manifest.logical_results;
}

AneShutdownReceipt AneRuntime::shutdown(
    std::chrono::milliseconds deadline_duration) {
  const auto checked_deadline = detail::checked_deadline(deadline_duration);
  const TimePoint deadline = checked_deadline.time;
  std::unique_lock<std::timed_mutex> transaction(
      implementation_->transaction_mutex, std::defer_lock);
  if (!transaction.try_lock_until(deadline)) {
    throw detail::runtime_error("deadline expired waiting for runtime transaction");
  }
  if (!implementation_->is_usable) {
    throw detail::runtime_error("runtime is unusable and cannot claim clean shutdown");
  }

  ensure_before(deadline, "shutdown command submission");
  detail::WorkerCommand command;
  command.operation = detail::WorkerOperation::shutdown;
  command.serial = ++implementation_->serial;
  command.deadline_monotonic_nanoseconds = checked_deadline.monotonic_nanoseconds;
  implementation_->current_operation_submitted = false;
  try {
    send_command(implementation_->control_fd, command, deadline);
  } catch (const std::exception& error) {
    implementation_->mark_unusable(
        std::string("shutdown command was not sent: ") + error.what(), false);
    throw;
  }
  implementation_->current_operation_submitted = true;

  detail::WorkerReply reply;
  try {
    reply = receive_reply(implementation_->control_fd, deadline);
  } catch (const std::exception& error) {
    implementation_->mark_unusable(
        std::string("shutdown completion became uncertain: ") + error.what(), true);
    throw detail::runtime_error(
        std::string(error.what()) +
        "; runtime marked unusable and no worker signal was sent; reboot is required before further ANE use");
  }
  if (reply.kind != detail::WorkerReplyKind::stopped || reply.serial != command.serial) {
    implementation_->mark_unusable(
        "worker did not confirm clean resource release", true);
    throw detail::runtime_error("worker did not confirm clean resource release");
  }
  if (!wait_for_exit(implementation_->pid, deadline)) {
    implementation_->mark_unusable(
        "worker confirmed release but did not exit by deadline", true);
    throw detail::runtime_error("worker confirmed release but did not exit by deadline");
  }

  AneShutdownReceipt receipt{
      static_cast<int>(implementation_->pid), reply.released_programs};
  implementation_->pid = -1;
  implementation_->is_usable = false;
  implementation_->current_operation_submitted = false;
  implementation_->ownership.release_cleanly();
  return receipt;
}

bool AneRuntime::usable() const {
  std::lock_guard<std::timed_mutex> transaction(
      implementation_->transaction_mutex);
  return implementation_->is_usable;
}

int AneRuntime::worker_pid() const {
  std::lock_guard<std::timed_mutex> transaction(
      implementation_->transaction_mutex);
  return static_cast<int>(implementation_->pid);
}

std::string AneRuntime::runtime_identity() const {
  std::lock_guard<std::timed_mutex> transaction(
      implementation_->transaction_mutex);
  return implementation_->identity;
}

} // namespace mlx::core::omarchy::ane
