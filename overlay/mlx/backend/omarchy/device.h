// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mlx/api.h"

namespace mlx::core::omarchy {

// Vendor ID for Apple Silicon GPUs on Omarchy Linux. Honeykrisp is the Mesa
// Vulkan driver that serves these GPUs.
constexpr uint32_t kAppleVendorId = 0x106b;

// VK_DRIVER_ID_MESA_HONEYKRISP spelled out for Vulkan 1.3 headers, which
// predate the enum entry (value verified against the 1.4 registry). Keep in
// sync with vk.xml; the runtime Vulkan >= 1.3 requirement is unchanged.
constexpr int32_t kMesaHoneykrispDriverId = 26;

// Result of the discovery policy applied to one physical device. The logic is
// pure so tests can exercise it without a Vulkan loader or GPU.
struct DeviceSupport {
  bool supported{false};
  // Set when the device was accepted only through the explicit
  // MLX_OMARCHY_ALLOW_NON_APPLE development override.
  bool non_apple_dev{false};
  // Human-readable refusal reason. Empty when supported.
  std::string reason;
  std::string device_name;
  uint32_t vendor_id{0};
  uint32_t device_id{0};
  uint32_t api_version{0};
  int32_t driver_id{-1};
};

// Classify one physical device against the Omarchy support policy:
//   1. Vulkan 1.3 is required (timeline semaphores, sync2).
//   2. Driver identity is authoritative: a Mesa Honeykrisp driverID accepts
//      the device (the M1 target reports vendor 0x10005, name "Apple M1").
//   3. A "honeykrisp" device name or Apple vendor 0x106b are alternate
//      signals.
//   4. Anything else is refused unless allow_non_apple is set, which accepts
//      any Vulkan 1.3 device for software development on non-Omarchy machines.
DeviceSupport classify_physical_device(
    const VkPhysicalDeviceProperties& props,
    int32_t driver_id,
    bool allow_non_apple);

// Physical-device capability facts collected at discovery time. Reported by
// mlx-omarchy-info and stored in hardware receipts.
struct CapabilityReport {
  std::string device_name;
  std::string driver_name;
  std::string driver_info;
  std::string driver_sha;
  std::string icd_path;
  // ICD selection origin: "packaged", "override", or "search".
  std::string icd_source;
  // Expected Mesa git SHA and where it came from: "env", "packaged file",
  // or empty when identity is recorded without enforcement.
  std::string expected_sha;
  std::string expected_sha_source;
  uint32_t vendor_id{0};
  uint32_t device_id{0};
  uint32_t driver_version{0};
  uint32_t api_version{0};
  int32_t driver_id{-1};
  std::array<uint8_t, VK_UUID_SIZE> pipeline_cache_uuid{};
  uint32_t queue_family_index{0};
  uint32_t queue_count{0};
  bool unified_memory{false};
  bool timeline_semaphore{false};
  bool shader_float16{false};
  bool shader_int16{false};
  bool shader_int64{false};
  bool storage_buffer_16bit_access{false};
  // True when the device exposes VK_EXT_shader_atomic_float with the
  // shaderBufferFloat32AtomicAdd feature: measured true on llvmpipe;
  // the M1 G13G B1 Honeykrisp does not advertise the extension at all
  // (an earlier "measured on both" note had read llvmpipe's feature
  // list on a box exposing both devices). Selects between the float
  // scatter Sum hardware-atomicAdd kernels and the FCAS
  // compare-exchange twins.
  bool shader_atomic_float_add{false};
  // True when VK_KHR_cooperative_matrix is present, the cooperativeMatrix
  // feature is on, and the device lists an 8x8x8 all-fp32 subgroup shape.
  // Honeykrisp advertises this behind AGX_SIMDMAT; llvmpipe does not.
  bool cooperative_matrix_f32_8{false};
  // True when the device lists VK_EXT_global_priority. Lets the backend
  // request a lower-than-default queue priority so MLX submissions yield
  // queue arbitration to the desktop compositor between submissions
  // (issue #19). Device creation additionally checks the compute queue
  // family's reported priorities and silently keeps the default when the
  // requested one is absent.
  bool queue_global_priority{false};
  size_t total_memory{0};
  VkDeviceSize max_allocation_size{0};
  VkDeviceSize max_buffer_size{0};
  VkDeviceSize max_storage_buffer_range{0};
  // Storage buffers bind at offsets aligned to at least this many
  // bytes; windowed bindings align their starts down to it and correct
  // the kernel's element offset by the moved bytes.
  uint32_t min_storage_buffer_offset_alignment{0};
  bool host_visible_coherent{false};
  // Storage-buffer descriptor limits reported by the physical device. These
  // bound the compute binding budget (compute.h kComputeBindingBudget).
  uint32_t max_per_stage_descriptor_storage_buffers{0};
  uint32_t max_descriptor_set_storage_buffers{0};
  uint32_t max_compute_work_group_invocations{0};
  uint32_t max_compute_shared_memory_size{0};
  std::array<uint32_t, 3> max_compute_work_group_size{0, 0, 0};
  float timestamp_period{0.0f};
  // Timestamp valid bits of the compute queue family; 0 when the queue
  // reports no timestamp query support. Read by the GPU profiling harness.
  uint32_t queue_timestamp_valid_bits{0};
  // Subgroup properties from VkPhysicalDeviceSubgroupProperties (Vulkan
  // 1.1+, queried at init via GetPhysicalDeviceProperties2 pNext chain).
  // Used by primitives.cpp to gate subgroup-reduction variants; see
  // DecodeGemvSubgroup. subgroup_size of 32 with ARITHMETIC bit set is
  // the load-bearing combination for the qmm_vec subgroup path.
  uint32_t subgroup_size{0};
  uint32_t subgroup_operations{0};
  // Capability simulation (MLX_OMARCHY_CAPS_SIM): set on a report that
  // is a named simulated profile, never on hardware discovery. The
  // profile name and the driver_variant axis it stands in for ride
  // alongside, so provenance output can stamp simulated runs. See
  // capability_sim.h for the axis vocabulary and gating contract.
  bool simulated{false};
  std::string simulation_profile;
  std::string simulated_driver_variant;
};

// A pending wait fails after 10 seconds with neither completion progress nor
// a submission that reached device execution. Recorded submissions set a
// VkEvent after their semaphore waits; vkGetEventStatus checks it without
// blocking. Vulkan has no portable intra-dispatch progress query, so started
// work remains subject to the 30-minute wall bound. Tests can override both
// limits with MLX_OMARCHY_HANG_NO_PROGRESS_NS and MLX_OMARCHY_MAX_WALL_NS.
inline constexpr uint64_t kSubmitHangNoProgressNsDefault =
    10ull * 1000 * 1000 * 1000;
inline constexpr uint64_t kSubmitMaxWallNsDefault =
    30ull * 60 * 1000 * 1000 * 1000;

// Parsed overrides. Resolved lazily on first call so unit tests can
// override the env before the process reads it.
uint64_t submit_hang_no_progress_ns();
uint64_t submit_max_wall_ns();

class CompletionDispatcher;
class Device;

// Wait for a timeline value with bounded, nonblocking progress observation.
// The callback returns the latest completion generation known to satisfy the
// waited value. Zero means no matching producer has been published.
// |recovery| (optional) enables the dropped-submission recovery ladder: on a
// no-progress interval with no evidence of execution, the device resubmits
// the stalled batches instead of throwing (Honeykrisp swallows submissions).
void wait_for_timeline_progress(
    VkDevice device,
    VkSemaphore semaphore,
    uint64_t target_value,
    CompletionDispatcher* progress = nullptr,
    std::function<uint64_t()> progress_generation = {},
    Device* recovery = nullptr);

class CommandEncoder;
class ComputeRuntime;

// Completion tracking for async submissions on the single Honeykrisp queue.
// Every encoder submission signals one strictly increasing value on a
// device-wide timeline semaphore; a dispatcher thread waits that timeline
// and runs each submission's completion handlers. Buffer temporaries and
// queued-semaphore ownership release one completion generation later,
// because Mesa signals a submission's semaphores before its submit-final
// cleanup retires the submission's timeline points. Submitters never
// block on the queue.
class CompletionDispatcher {
 public:
  explicit CompletionDispatcher(VkDevice device, Device* owner = nullptr);
  ~CompletionDispatcher();

  CompletionDispatcher(const CompletionDispatcher&) = delete;
  CompletionDispatcher& operator=(const CompletionDispatcher&) = delete;

  VkSemaphore semaphore() const {
    return semaphore_;
  }

  struct Completion {
    uint64_t value;
    std::vector<std::shared_ptr<void>> temporaries;
    std::vector<std::function<void()>> handlers;
    VkEvent started;
  };

  uint64_t reserve();
  // Newest value handed out by reserve() (0 when none): everything
  // reserved up to here is already submitted or submitting.
  uint64_t last_reserved();
  void enqueue(
      uint64_t value,
      std::vector<std::shared_ptr<void>> temporaries,
      std::vector<std::function<void()>> handlers,
      VkEvent started = VK_NULL_HANDLE);
  void wait(uint64_t value);
  uint64_t drained_value();
  void reset_progress_event(VkEvent event);
  bool has_active_submission(uint64_t through_value);
  void shutdown();

  // Everything a submission put on the queue, retained so the watchdog's
  // recovery ladder can resubmit it when Honeykrisp drops the batch (the
  // completion timeline never advances and the batch never started
  // executing). Retained under the dispatcher mutex, erased when the
  // completion drains.
  struct ResubmitBatch {
    uint64_t value{0};
    VkCommandBuffer cmd{VK_NULL_HANDLE};
    std::vector<VkSemaphore> wait_sems;
    std::vector<uint64_t> wait_values;
    std::vector<VkSemaphore> signal_sems;
    std::vector<uint64_t> signal_values;
  };
  void retain_for_resubmit(uint64_t value, ResubmitBatch batch);
  std::vector<ResubmitBatch> take_resubmit_batches(uint64_t through_value);

  // Attach the submission fence for |value|: the fence signals when the
  // driver has fully finished executing the batch, unlike the completion
  // timeline semaphore which Mesa signals before its submit-final cleanup.
  void attach_execution_fence(uint64_t value, VkFence fence);
  // Non-stalling: true when every submission <= |value| provably finished
  // executing (all attached fences signalled). The only sound safe-to-
  // recycle test; the timeline observation is not.
  bool execution_complete(uint64_t value);
  VkFence acquire_execution_fence();
  void mark_execution_complete_through(uint64_t value);

 private:
  void run();
  void drain_through(uint64_t max_value);

  VkDevice device_;
  Device* owner_{nullptr};
  VkSemaphore semaphore_{VK_NULL_HANDLE};
  std::deque<Completion> pending_;
  // Completion value -> the batch that carries it, retained until the
  // completion drains so a dropped submission can be resubmitted.
  std::map<uint64_t, ResubmitBatch> resubmitable_;
  // Submission execution fences, in value order: recycle decisions use
  // these, not the pre-cleanup timeline observation.
  struct ExecutionFence {
    uint64_t value;
    VkFence fence;
    bool done;
  };
  std::deque<ExecutionFence> execution_fences_;
  std::vector<VkFence> execution_fence_pool_;
  uint64_t execution_done_through_{0};
  std::mutex execution_mutex_;
  // Payloads of already-drained completions, released one completion
  // later. Mesa signals a submission's semaphores before its submit-final
  // cleanup releases timeline points, so a completion value on this
  // timeline does not prove the driver finished that submission.
  std::vector<std::shared_ptr<void>> retired_temporaries_;
  uint64_t next_value_{0};
  uint64_t drained_value_{0};
  std::mutex mutex_;
  std::mutex drain_mutex_;
  std::condition_variable cv_;
  std::thread thread_;
  bool stop_{false};
};

// A live Vulkan device. Created lazily per supported physical device index.
class Device {
 public:
  explicit Device(uint32_t physical_device_index);
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  VkDevice handle() const {
    return device_;
  }

  VkQueue queue() const {
    return queue_;
  }

  uint32_t queue_family() const {
    return caps_.queue_family_index;
  }

  const VkPhysicalDeviceMemoryProperties& memory_properties() const {
    return mem_props_;
  }

  const CapabilityReport& capabilities() const {
    return caps_;
  }

  // Hardware truth even when capabilities() is a simulated profile:
  // VkDevice creation and the simulation backing checks read this.
  // Without MLX_OMARCHY_CAPS_SIM the two reports are identical.
  const CapabilityReport& hardware_capabilities() const {
    return hardware_caps_;
  }

  std::mutex& queue_mutex() {
    return queue_mutex_;
  }

  // Dropped-submission recovery ladder, invoked by the watchdog when a
  // waited completion value stops advancing. Honeykrisp provably swallows
  // submissions (the in-tree empty signal-only QueueSubmit fix documents
  // the deterministic shape; burst observations add real command buffers
  // whose started event never fires). The ladder resubmits every retained
  // batch through |target_value| - legal because the single queue is
  // in-order, so no later completion can have signaled while an earlier
  // one is still stalled, and the started event proves the batch never
  // began executing.
  enum class RecoveryResult : int {
    // Ladder ran (rung 1 resubmit or rung 2 host-signal); the wait
    // should restart its no-progress clock and keep waiting.
    kRecovered = 1,
    // Nothing to recover right now: no retained batch exists at or
    // below last_reserved (another waiter's recovery already took the
    // batches, or the producing stream has not submitted yet). The wait
    // must NOT throw - it should keep waiting for the owner's recovery
    // or the wall deadline.
    kNotRecoverable = 0,
    // Attempt budget spent: a genuinely wedged device. The wait throws.
    kExhausted = -1,
  };
  RecoveryResult recover_stalled_submissions(
      uint64_t target_value,
      uint32_t round);

  void join_completed_handlers();

  CompletionDispatcher& completions() {
    return *completions_;
  }

  ComputeRuntime& compute() {
    return *compute_;
  }

  uint64_t signal_timeline(VkSemaphore semaphore, uint64_t value);

  CapabilityReport caps_;
  CapabilityReport hardware_caps_{};
  VkPhysicalDeviceMemoryProperties mem_props_{};
  VkDevice device_{VK_NULL_HANDLE};
  VkQueue queue_{VK_NULL_HANDLE};
  std::mutex queue_mutex_;
  std::unique_ptr<ComputeRuntime> compute_;
  std::unique_ptr<CompletionDispatcher> completions_;
};

// --- Process-wide runtime -------------------------------------------------

// Discover devices and prepare the runtime. Idempotent; never throws. On
// failure the reason is recorded and available through init_error().
bool init();

MLX_API bool is_available();
MLX_API const std::string& init_error();

// Number of devices that pass the support policy.
int device_count();

// Access the live device for a supported index. Initializes on first use and
// throws std::runtime_error with the discovery reason when unavailable.
MLX_API Device& device(uint32_t index = 0);
// False once the process-wide Vulkan runtime has been destroyed; callers
// that may run during static teardown (allocator::free) must not touch
// the device after this.
MLX_API bool runtime_alive();

// Capability facts for a supported index without creating a VkDevice.
// Throws std::runtime_error when the index is out of range or discovery
// failed.
MLX_API const CapabilityReport& capability_report(uint32_t index);
// Hardware-truth report for a supported index even when
// capability_report() is serving a simulated profile. Throws
// std::runtime_error when the index is out of range or discovery
// failed.
MLX_API const CapabilityReport& hardware_capability_report(uint32_t index);

// True when the named MLX_OMARCHY_* boolean environment variable is set
// to 1/on/true/yes (case-insensitive). False when unset or any other
// value. Read on every call: nothing here caches environment state.
bool env_flag(const char* name);

// Scoped diagnostic switches for the compiled-tape defect hunt
// (docs/install-omarchy.md, "Compiled-tape debug switches"). Between
// construction and destruction, tape_full_barriers() and tape_no_reuse()
// report the values given here so the encoder and the allocator can make
// the tape's dispatches heavier or its resources fresher. Out of scope
// both always read false, so no other work on the device changes shape.
// The tape runner is the only intended constructor; there is exactly one
// scope at a time by design.
class MLX_API TapeDebugScope {
 public:
  TapeDebugScope(bool full_barriers, bool no_reuse);
  ~TapeDebugScope();

  TapeDebugScope(const TapeDebugScope&) = delete;
  TapeDebugScope& operator=(const TapeDebugScope&) = delete;

 private:
  bool full_barriers_;
  bool no_reuse_;
};

// Current scoped diagnostic state. False outside a TapeDebugScope.
bool tape_full_barriers();
bool tape_no_reuse();
// True when MLX_OMARCHY_NO_BUFFER_CACHE was set at runtime init
// (diagnostic, docs/install-omarchy.md): the allocator destroys every
// freed buffer instead of recycling it, for the whole process.
bool buffer_cache_disabled();

} // namespace mlx::core::omarchy
