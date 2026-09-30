// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/backend/omarchy/ane/runtime_detail.h"

#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <cstdio>
#include <string>
#include <utility>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mlx::core::omarchy::ane::detail {

class RuntimeOwnership {
 public:
  RuntimeOwnership() = default;

  ~RuntimeOwnership() {
    if (lock_fd_ >= 0 && !preserve_quarantine_) {
      clear_state_noexcept();
    }
    close_fds();
  }

  RuntimeOwnership(const RuntimeOwnership&) = delete;
  RuntimeOwnership& operator=(const RuntimeOwnership&) = delete;

  RuntimeOwnership(RuntimeOwnership&& other) noexcept
      : lock_fd_(other.lock_fd_),
        state_fd_(other.state_fd_),
        preserve_quarantine_(other.preserve_quarantine_) {
    other.lock_fd_ = -1;
    other.state_fd_ = -1;
  }

  RuntimeOwnership& operator=(RuntimeOwnership&& other) noexcept {
    if (this != &other) {
      if (lock_fd_ >= 0 && !preserve_quarantine_) {
        clear_state_noexcept();
      }
      close_fds();
      lock_fd_ = other.lock_fd_;
      state_fd_ = other.state_fd_;
      preserve_quarantine_ = other.preserve_quarantine_;
      other.lock_fd_ = -1;
      other.state_fd_ = -1;
    }
    return *this;
  }

  static RuntimeOwnership acquire() {
    struct stat device {};
    if (::lstat("/dev/accel/accel0", &device) != 0 || !S_ISCHR(device.st_mode)) {
      throw runtime_error("qualified ANE device identity is unavailable");
    }
    const std::string boot_id = read_text_file("/proc/sys/kernel/random/boot_id");
    try {
      return acquire_at(kRuntimeOwnershipLockPath, kRuntimeQuarantinePath, boot_id, false);
    } catch (const std::runtime_error& error) {
      if (std::string(error.what()).find("cannot open ANE ownership state") == std::string::npos) {
        throw;
      }
      const char* runtime_dir = std::getenv("XDG_RUNTIME_DIR");
      if (!runtime_dir || runtime_dir[0] != '/') {
        throw;
      }
      const auto private_dir = std::filesystem::path(runtime_dir) / "mlx-omarchy-ane";
      if (::mkdir(private_dir.c_str(), 0700) != 0 && errno != EEXIST) {
        throw;
      }
      struct stat status {};
      if (::lstat(private_dir.c_str(), &status) != 0 || !S_ISDIR(status.st_mode) ||
          status.st_uid != ::geteuid() || (status.st_mode & 0777) != 0700) {
        throw runtime_error("per-user ANE ownership directory is unsafe");
      }
      const auto lock_path = private_dir / "device.lock";
      std::fprintf(
          stderr,
          "ANE ownership uses per-user lock %s\n",
          lock_path.c_str());
      return acquire_at(lock_path, private_dir / "quarantine", boot_id);
    }
  }

  static void validate_shared_paths_at(
      const std::filesystem::path& directory,
      const std::filesystem::path& lock_path,
      const std::filesystem::path& state_path,
      uid_t owner,
      gid_t group) {
    if (lock_path.parent_path() != directory ||
        state_path.parent_path() != directory) {
      throw runtime_error("ANE ownership files are outside the provisioned directory");
    }
    validate_shared_directory(directory, owner, group);
    validate_shared_file(lock_path, owner, group);
    validate_shared_file(state_path, owner, group);
  }

  static RuntimeOwnership acquire_at(
      const std::filesystem::path& lock_path,
      const std::filesystem::path& state_path,
      std::string boot_id,
      bool create = true) {
    trim_line_end(boot_id);
    if (!valid_boot_id(boot_id)) {
      throw runtime_error("current boot identity is invalid");
    }

    RuntimeOwnership ownership;
    ownership.lock_fd_ = open_regular(lock_path, create);
    if (::flock(ownership.lock_fd_, LOCK_EX | LOCK_NB) != 0) {
      const int error = errno;
      if (error == EWOULDBLOCK || error == EAGAIN) {
        throw runtime_error("another ANE runtime owns the host device");
      }
      throw runtime_error(
          "ANE ownership lock failed: " + std::string(std::strerror(error)));
    }
    ownership.preserve_quarantine_ = true;
    ownership.state_fd_ = open_regular(state_path, create);
    std::string current_state = read_fd(ownership.state_fd_);
    trim_line_end(current_state);
    if (!current_state.empty() && !valid_boot_id(current_state)) {
      ownership.preserve_quarantine_ = true;
      throw runtime_error("ANE runtime quarantine state is invalid");
    }
    if (current_state == boot_id) {
      ownership.preserve_quarantine_ = true;
      throw runtime_error(
          "ANE runtime is quarantined for this boot; reboot is required");
    }
    write_fd(ownership.state_fd_, boot_id + "\n");
    ownership.preserve_quarantine_ = false;
    return ownership;
  }

  int lock_fd() const {
    if (lock_fd_ < 0) {
      throw runtime_error("ANE runtime ownership is not active");
    }
    return lock_fd_;
  }

  void arm() {
    preserve_quarantine_ = true;
  }

  void quarantine() noexcept {
    preserve_quarantine_ = true;
  }

  void release_cleanly() {
    if (state_fd_ < 0) {
      throw runtime_error("ANE ownership state is not active");
    }
    write_fd(state_fd_, "");
    preserve_quarantine_ = false;
    close_fds();
  }

 private:
  static void trim_line_end(std::string& value) {
    while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
      value.pop_back();
    }
  }

  static bool valid_boot_id(const std::string& value) {
    if (value.size() != 36) {
      return false;
    }
    for (size_t i = 0; i < value.size(); ++i) {
      const bool separator = i == 8 || i == 13 || i == 18 || i == 23;
      if (separator ? value[i] != '-' : !std::isxdigit(
              static_cast<unsigned char>(value[i]))) {
        return false;
      }
    }
    return true;
  }

  static void validate_shared_directory(
      const std::filesystem::path& path,
      uid_t owner,
      gid_t) {
    struct stat status {};
    if (::lstat(path.c_str(), &status) != 0 || !S_ISDIR(status.st_mode) ||
        status.st_uid != owner || (status.st_mode & 07777) != 01777 ||
        ::access(path.c_str(), X_OK) != 0) {
      throw runtime_error(
          "ANE ownership directory is not a provisioned shared directory: " +
          path.string());
    }
  }

  static void validate_shared_file(
      const std::filesystem::path& path,
      uid_t owner,
      gid_t) {
    struct stat status {};
    if (::lstat(path.c_str(), &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_nlink != 1 || status.st_uid != owner ||
        (status.st_mode & 07777) != 0666 ||
        ::access(path.c_str(), R_OK | W_OK) != 0) {
      throw runtime_error(
          "ANE ownership state is not a provisioned shared regular file: " +
          path.string());
    }
  }

  static std::string read_text_file(const std::filesystem::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
      throw runtime_error(
          "cannot read boot identity: " + std::string(std::strerror(errno)));
    }
    std::string value;
    try {
      value = read_fd(fd);
    } catch (...) {
      ::close(fd);
      throw;
    }
    ::close(fd);
    return value;
  }

  static int open_regular(const std::filesystem::path& path, bool create) {
    const int flags = O_RDWR | O_CLOEXEC | O_NOFOLLOW | (create ? O_CREAT : 0);
    int fd = ::open(path.c_str(), flags, 0666);
    if (fd < 0) {
      throw runtime_error(
          "cannot open ANE ownership state " + path.string() + ": " +
          std::strerror(errno));
    }
    struct stat status {};
    if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_nlink != 1) {
      ::close(fd);
      throw runtime_error("ANE ownership state is not a private regular file: " + path.string());
    }
    if ((status.st_mode & 07777) != 0666) {
      if (status.st_uid == ::geteuid()) {
        if (::fchmod(fd, 0666) != 0) {
          const int error = errno;
          ::close(fd);
          throw runtime_error(
              "cannot set ANE ownership file mode: " +
              std::string(std::strerror(error)));
        }
      } else {
        ::close(fd);
        throw runtime_error(
            "cannot open ANE ownership state for mode repair: " + path.string());
      }
    }
    return fd;
  }
  static std::string read_fd(int fd) {
    if (::lseek(fd, 0, SEEK_SET) < 0) {
      throw runtime_error(
          "ANE ownership state seek failed: " + std::string(std::strerror(errno)));
    }
    std::string value;
    char buffer[128];
    while (true) {
      ssize_t count = ::read(fd, buffer, sizeof(buffer));
      if (count == 0) {
        return value;
      }
      if (count < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw runtime_error(
            "ANE ownership state read failed: " + std::string(std::strerror(errno)));
      }
      value.append(buffer, static_cast<size_t>(count));
      if (value.size() > 128) {
        throw runtime_error("ANE ownership state is invalid");
      }
    }
  }

  static void write_fd(int fd, const std::string& value) {
    if (::ftruncate(fd, 0) != 0 || ::lseek(fd, 0, SEEK_SET) < 0) {
      throw runtime_error(
          "ANE ownership state reset failed: " + std::string(std::strerror(errno)));
    }
    size_t offset = 0;
    while (offset < value.size()) {
      ssize_t count = ::write(fd, value.data() + offset, value.size() - offset);
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count <= 0) {
        throw runtime_error(
            "ANE ownership state write failed: " + std::string(std::strerror(errno)));
      }
      offset += static_cast<size_t>(count);
    }
    if (::fsync(fd) != 0) {
      throw runtime_error(
          "ANE ownership state sync failed: " + std::string(std::strerror(errno)));
    }
  }

  void clear_state_noexcept() noexcept {
    if (state_fd_ >= 0) {
      ::ftruncate(state_fd_, 0);
      ::fsync(state_fd_);
    }
  }

  void close_fds() noexcept {
    if (state_fd_ >= 0) {
      ::close(state_fd_);
      state_fd_ = -1;
    }
    if (lock_fd_ >= 0) {
      ::close(lock_fd_);
      lock_fd_ = -1;
    }
  }

  int lock_fd_{-1};
  int state_fd_{-1};
  bool preserve_quarantine_{false};
};

} // namespace mlx::core::omarchy::ane::detail
