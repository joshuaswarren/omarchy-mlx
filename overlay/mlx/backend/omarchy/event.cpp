// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// CPU-stream events stay host-only until a GPU consumer needs a timeline
// bridge. The CPU scheduler signals that bridge only after producer work;
// GPU-stream events keep queue-ordered timeline signals.

#include <unistd.h>
#include <sys/syscall.h>
#include "mlx/event.h"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/vulkan.h"
#include "mlx/scheduler.h"

namespace mlx::core {

namespace {

// Host-side counter for events on CPU streams.
struct HostCounter {
  uint64_t value{0};
  std::mutex mtx;
  std::condition_variable cv;

  void wait(uint64_t v) {
    std::unique_lock<std::mutex> lk(mtx);
    if (value >= v) {
      return;
    }
    cv.wait(lk, [this, v] { return value >= v; });
  }

  void signal(uint64_t v) {
    {
      std::lock_guard<std::mutex> lk(mtx);
      value = std::max(value, v);
    }
    cv.notify_all();
  }

  bool signaled(uint64_t v) {
    std::lock_guard<std::mutex> lk(mtx);
    return value >= v;
  }
};

// Device-side timeline semaphore for events consumed by GPU streams.
struct TimelineSemaphore {
  explicit TimelineSemaphore(omarchy::Device& device) : device(device) {
    signal_from_host_fn = reinterpret_cast<PFN_vkSignalSemaphore>(
        omarchy::vk::device_table().GetDeviceProcAddr(
            device.handle(), "vkSignalSemaphore"));
    if (!signal_from_host_fn) {
      throw std::runtime_error(
          "[omarchy] Vulkan device has no vkSignalSemaphore.");
    }
    VkSemaphoreTypeCreateInfo type{
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type.initialValue = 0;
    VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    ci.pNext = &type;
    VKX_CHECK(
        omarchy::vk::device_table().CreateSemaphore(
            device.handle(), &ci, nullptr, &semaphore));
  }

  ~TimelineSemaphore() {
    if (semaphore != VK_NULL_HANDLE) {
      omarchy::vk::device_table().DestroySemaphore(
          device.handle(), semaphore, nullptr);
    }
  }

  uint64_t counter() const {
    uint64_t counter_value = 0;
    VKX_CHECK(
        omarchy::vk::device_table().GetSemaphoreCounterValue(
            device.handle(), semaphore, &counter_value));
    return counter_value;
  }

  void signal_from_host(uint64_t value) {
    VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    info.semaphore = semaphore;
    info.value = value;
    VKX_CHECK(signal_from_host_fn(device.handle(), &info));
  }

  omarchy::Device& device;
  VkSemaphore semaphore{VK_NULL_HANDLE};
  PFN_vkSignalSemaphore signal_from_host_fn{nullptr};
};

struct EventImpl {
  std::atomic<Error*> error{nullptr};
  std::unique_ptr<TimelineSemaphore> gpu;
  std::unique_ptr<HostCounter> host;
  mutable std::mutex bridge_mtx;
  uint64_t host_value_on_gpu{0};
  mutable std::mutex signal_mtx;
  uint64_t signaled_completion{0};
  uint64_t signaled_value{0};
  std::atomic<bool> queued_signal{false};

  void publish_signal(uint64_t value, uint64_t completion) {
    std::lock_guard<std::mutex> lk(signal_mtx);
    if (value > signaled_value ||
        (value == signaled_value && completion > signaled_completion)) {
      signaled_value = value;
      signaled_completion = completion;
    }
  }

  uint64_t completion_for(uint64_t target) const {
    std::lock_guard<std::mutex> lk(signal_mtx);
    return signaled_value >= target ? signaled_completion : 0;
  }

  bool is_created() const {
    std::lock_guard<std::mutex> lk(bridge_mtx);
    return gpu || host;
  }

  void ensure_created(Stream s) {
    std::lock_guard<std::mutex> lk(bridge_mtx);
    if (gpu || host) {
      return;
    }
    if (s.device == Device::gpu) {
      gpu =
          std::make_unique<TimelineSemaphore>(omarchy::device(s.device.index));
    } else {
      host = std::make_unique<HostCounter>();
    }
  }

  VkSemaphore bridge_host_to_gpu(Stream s, uint64_t target_value) {
    std::lock_guard<std::mutex> lk(bridge_mtx);
    if (!gpu) {
      gpu =
          std::make_unique<TimelineSemaphore>(omarchy::device(s.device.index));
    }
    if (host->signaled(target_value) && target_value > host_value_on_gpu) {
      omarchy::allocator().flush_noncoherent(gpu->device.handle());
      gpu->signal_from_host(target_value);
      host_value_on_gpu = target_value;
    }
    return gpu->semaphore;
  }

  void signal_host(uint64_t target_value) {
    std::lock_guard<std::mutex> lk(bridge_mtx);
    if (gpu && target_value > host_value_on_gpu) {
      omarchy::allocator().flush_noncoherent(gpu->device.handle());
      gpu->signal_from_host(target_value);
      host_value_on_gpu = target_value;
    }
    host->signal(target_value);
  }
};

} // namespace

Event::Event(Stream s) : stream_(s) {
  event_ = std::make_shared<EventImpl>();
}

void Event::wait() {
  check_error();
  auto& event = cast<EventImpl>();
  event.ensure_created(stream_);
  if (value() == 0) {
    return;
  }
  if (event.host) {
    event.host->wait(value());
  } else {
    if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr) {
      uint64_t cnt = 0;
      omarchy::vk::device_table().GetSemaphoreCounterValue(
          event.gpu->device.handle(), event.gpu->semaphore, &cnt);
      fprintf(stderr,
              "[rtmod] EV-WAIT tid=%lu ev=%p val=%lu cnt=%llu resv=%llu\n",
              (unsigned long)syscall(SYS_gettid),
              (void*)event_.get(),
              (unsigned long)value_,
              (unsigned long long)cnt,
              (unsigned long long)event.gpu->device.completions().last_reserved());
    }
    omarchy::wait_for_timeline_progress(
        event.gpu->device.handle(),
        event.gpu->semaphore,
        value_,
        &event.gpu->device.completions(),
        [&event, target = value_] { return event.completion_for(target); },
        &event.gpu->device);
    for (int join_attempt = 0; join_attempt < 8; ++join_attempt) {
      uint64_t generation = event.completion_for(value_);
      if (generation != 0) {
        event.gpu->device.completions().wait(generation);
      }
      if (event.completion_for(value_) == generation) {
        break;
      }
      std::this_thread::yield();
    }
    event.gpu->device.join_completed_handlers();
    omarchy::allocator().invalidate_noncoherent(event.gpu->device.handle());
  }
  check_error();
}

void Event::wait(Stream s) {
  auto& event = cast<EventImpl>();
  event.ensure_created(stream_);
  if (value() == 0) {
    return;
  }
  if (s.device == Device::gpu) {
    VkSemaphore semaphore = event.host
        ? event.bridge_host_to_gpu(s, value())
        : event.gpu->semaphore;
    auto& encoder = omarchy::get_command_encoder(s);
    if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr) {
      fprintf(stderr,
              "[rtmod] EV-WAIT-ENC tid=%lu ev=%p val=%lu st=%lu\n",
              (unsigned long)syscall(SYS_gettid),
              (void*)event_.get(),
              (unsigned long)value(),
              (unsigned long)s.index);
    }
    encoder.add_semaphore_wait(semaphore, value(), event_);
  } else if (event.host) {
    uint64_t target_value = value();
    scheduler::wait_event(s, *this, [target_value](Event& self) {
      self.cast<EventImpl>().host->wait(target_value);
    });
  } else {
    uint64_t target_value = value();
    if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr) {
      fprintf(stderr,
              "[rtmod] EV-WAIT-TASK tid=%lu ev=%p val=%lu\n",
              (unsigned long)syscall(SYS_gettid),
              (void*)event_.get(),
              (unsigned long)target_value);
    }
    scheduler::wait_event(s, *this, [target_value](Event& self) {
      auto& impl = self.cast<EventImpl>();
      omarchy::wait_for_timeline_progress(
          impl.gpu->device.handle(),
          impl.gpu->semaphore,
          target_value,
          &impl.gpu->device.completions(),
          [&impl, target_value] {
            return impl.completion_for(target_value);
          },
          &impl.gpu->device);
      impl.gpu->device.join_completed_handlers();
    });
  }
}

void Event::signal(Stream s) {
  auto& event = cast<EventImpl>();
  event.ensure_created(s);
  const bool trace_dispatch =
      std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr;
  if (trace_dispatch) {
    fprintf(stderr, "[rtmod] EV-SIGNAL tid=%lu ev=%p val=%lu st=%d host=%d\n",
            (unsigned long)syscall(SYS_gettid), (void*)event_.get(), (unsigned long)value(), s.index,
            event.host ? 1 : 0);
  }
  if (event.host) {
    uint64_t target_value = value();
    if (s.device == Device::gpu) {
      EventImpl* impl = &event;
      std::shared_ptr<void> keepalive = event_;
      auto& encoder = omarchy::get_command_encoder(s);
      if (trace_dispatch) {
        fprintf(stderr, "[rtmod] EV-SIGNAL path=completed-handler\n");
      }
      encoder.add_completed_handler([impl, target_value, keepalive]() {
        impl->signal_host(target_value);
      });
      encoder.commit();
    } else {
      scheduler::signal_event(s, *this, [target_value](Event& self) {
        self.cast<EventImpl>().signal_host(target_value);
      });
    }
    return;
  }

  if (value() == 0) {
    return;
  }
  if (s.device == Device::gpu) {
    auto& encoder = omarchy::get_command_encoder(s);
    // Host-signal only when nothing on this stream is still executing:
    // a queue-ordered signal would have to outlive the event (it once
    // destroyed the semaphore under a live submit), and a host signal
    // with work in flight lets a GPU-stream waiter run ahead of it.
    if (!event.queued_signal.load(std::memory_order_acquire) &&
        encoder.idle() && encoder.synchronized()) {
      if (trace_dispatch) {
        fprintf(stderr, "[rtmod] EV-SIGNAL path=host-signal\n");
      }
      event.gpu->signal_from_host(value());
      event.publish_signal(value(), encoder.last_submitted_completion());
    } else {
      event.queued_signal.store(true, std::memory_order_release);
      encoder.add_semaphore_signal(event.gpu->semaphore, value(), event_);
      encoder.commit();
      event.publish_signal(value(), encoder.last_submitted_completion());
    }
  } else {
    uint64_t generation =
        event.gpu->device.signal_timeline(event.gpu->semaphore, value());
    event.publish_signal(value(), generation);
  }
}

bool Event::is_signaled() const {
  auto& event = cast<EventImpl>();
  if (!event.is_created()) {
    return false;
  }
  if (event.host) {
    return event.host->signaled(value());
  }
  return event.gpu->counter() >= value();
}

std::atomic<Error*>& Event::error() {
  return cast<EventImpl>().error;
}

} // namespace mlx::core
