// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <deque>
#include <map>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "mlx/api.h"
#include "mlx/backend/omarchy/coopmat_select.h"
#include "mlx/backend/omarchy/device_local_activations.h"

namespace mlx::core::io {
class Reader;
}

namespace mlx::core::omarchy {

// Vendor ID for Apple Silicon GPUs on Omarchy Linux. Honeykrisp is the Mesa
// Vulkan driver that serves these GPUs.
constexpr uint32_t kAppleVendorId = 0x106b;

// VK_DRIVER_ID_MESA_HONEYKRISP spelled out for Vulkan 1.3 headers, which
// predate the enum entry (value verified against the 1.4 registry). Keep in
// sync with vk.xml; the runtime Vulkan >= 1.3 requirement is unchanged.
constexpr int32_t kMesaHoneykrispDriverId = 26;

// VK_DRIVER_ID_AURORA_BOREALIS (Khronos registry value 35): Aurora Silicon's
// Honeykrisp-derived driver (aurora-silicon/mesa), the only Vulkan driver that
// admits the M4 (G16G) and newer GPUs. Its deviceName ("Apple M4 (G16G ..)")
// names no driver and its vendorID is Mesa's, so the driver id is the signal.
constexpr int32_t kAuroraBorealisDriverId = 35;

// True for either accepted Apple-driver id: the Khronos enum entry that keeps
// its Honeykrisp name (26), or Aurora's own Borealis entry (35). Aurora
// renames the driver to Borealis (driverName, ICD and library file names) and
// keeps HK_ environment names as fallbacks for a time, so both spellings must
// keep working (Ryan 2026-11-20).
inline bool is_apple_driver_id(int32_t driver_id) {
  return driver_id == kMesaHoneykrispDriverId ||
      driver_id == kAuroraBorealisDriverId;
}

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

// The subgroup size every pipeline requests, or 0 to leave it to the
// driver. Vulkan 1.3 lets a driver vary it per pipeline within
// [min_size, max_size] (Intel runs small workgroups at 16 of 8-32 lanes),
// and every subgroup shader here assumes |subgroup_size| lanes.
uint32_t required_subgroup_size(
    bool size_control,
    VkShaderStageFlags required_stages,
    uint32_t min_size,
    uint32_t max_size,
    uint32_t subgroup_size);

// Device indices for |types| in discovery order: discrete GPUs first, so
// device 0 is the dGPU where an iGPU enumerates ahead of it. Stable.
std::vector<size_t> device_order(const std::vector<VkPhysicalDeviceType>& types);

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
  // The packaged ICD that enumerated no supported device before the
  // system ICD served the retry; empty when no fallback ran.
  std::string icd_fallback_from;
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
  // Same extension and feature, plus an 8x8x8 subgroup shape with fp16
  // A/B and an fp32 accumulator. The extension is enabled only with
  // cooperative_matrix_f32_8, so users gate on both.
  bool cooperative_matrix_f16_8{false};
  // VK_KHR_shader_bfloat16 with shaderBFloat16Type and
  // shaderBFloat16CooperativeMatrix on, plus an 8x8x8 subgroup shape with
  // bf16 A/B and an fp32 accumulator. Implies cooperative_matrix_f32_8.
  bool cooperative_matrix_bf16_8{false};
  // VK_KHR_shader_bfloat16 with both features on, plus ANY bf16 A/B
  // subgroup non-saturating shape with an fp32 accumulator (e.g. ANV's
  // 8x16x16 XMX shape on BMG, which advertises no f32 8x8x8 shape).
  // Unlike cooperative_matrix_bf16_8 this does not imply the fp32
  // shape; kernel selection still confirms the exact shape in
  // cooperative_matrix_shapes.
  bool cooperative_matrix_bf16{false};
  // Every cooperative-matrix shape the device advertised, recorded verbatim
  // (including mixed-operand shapes where AType != BType, which the booleans
  // above cannot express). Kernel selection confirms the exact shape a
  // coopmat kernel emits is listed here before taking the route; the booleans
  // remain as convenience summaries of the common same-operand 8x8x8 shapes.
  std::vector<CoopmatShape> cooperative_matrix_shapes;
  // True when the device lists VK_EXT_global_priority. Lets the backend
  // request a lower-than-default queue priority so MLX submissions yield
  // queue arbitration to the desktop compositor between submissions
  // (issue #19). Device creation additionally checks the compute queue
  // family's reported priorities and silently keeps the default when the
  // requested one is absent.
  bool queue_global_priority{false};
  // VK_EXT_memory_priority with the memoryPriority feature and
  // VK_EXT_pageable_device_local_memory with pageableDeviceLocalMemory:
  // enabled only as a pair (pageable memory is what the priority hint
  // acts on), matching lily's device init on Honeykrisp-M5. Lets a later
  // change pin weight allocations resident under memory pressure.
  bool memory_priority{false};
  bool pageable_device_local_memory{false};
  // VK_EXT_external_memory_host advertised. Probe only: the extension is
  // not enabled until the change that imports host weight memory.
  bool external_memory_host{false};
  // VK_EXT_calibrated_timestamps advertised. Enabled only under
  // MLX_OMARCHY_TRACE_DISPATCH, as lily gates it on trace::on().
  bool calibrated_timestamps{false};
  // VK_KHR_push_descriptor advertised. Lets a dispatch push its
  // descriptors into the command buffer instead of allocating a pool
  // set, calling vkUpdateDescriptorSets and binding it.
  bool push_descriptor{false};
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
  // Whether a per-submission timestamp compute submit can reach the device
  // on this board. False on boards whose admission check refuses such a
  // submit: the G18G (M6) check answers -EINVAL for a timestamp-bearing
  // compute submit, which the driver surfaces as VK_ERROR_DEVICE_LOST
  // (issue #526). The queue still advertises timestamp support there, so
  // the advertisement is not the gate. The GPU profiling harness reads
  // this and runs host-costs-only when it is false.
  bool submission_timestamps{true};
  // Subgroup properties from VkPhysicalDeviceSubgroupProperties (Vulkan
  // 1.1+, queried at init via GetPhysicalDeviceProperties2 pNext chain).
  // Used by primitives.cpp to gate subgroup-reduction variants; see
  // DecodeGemvSubgroup. subgroup_size of 32 with ARITHMETIC bit set is
  // the load-bearing combination for the qmm_vec subgroup path.
  uint32_t subgroup_size{0};
  uint32_t subgroup_operations{0};
  // Nonzero: pipelines pin this subgroup size (required_subgroup_size()).
  uint32_t required_subgroup_size{0};
  // Capability simulation (MLX_OMARCHY_CAPS_SIM): set on a report that
  // is a named simulated profile, never on hardware discovery. The
  // profile name and the driver_variant axis it stands in for ride
  // alongside, so provenance output can stamp simulated runs. See
  // capability_sim.h for the axis vocabulary and gating contract.
  bool simulated{false};
  std::string simulation_profile;
  std::string simulated_driver_variant;
};

// Boards whose compute submits cannot carry a per-submission timestamp.
// G18G (M6): the 7.1.12-aurora-m6usb-ARCH G18 compute admission check
// refuses a timestamp-bearing submit ("check d value 0", then "invalid
// resource metadata"), so the submit fails with -EINVAL and the driver
// reports VK_ERROR_DEVICE_LOST (issue #526). This board has no
// per-submission timestamp wheel at all
// (receipts/2026-11-13-g18g-batch-work).
inline constexpr std::string_view kBoardsWithoutSubmissionTimestamps[] = {
    "G18G",
};

// Pure board lookup: whether a per-submission timestamp compute submit can
// reach the device on the board with this device name.
inline bool submission_timestamps_supported(std::string_view device_name) {
  for (const auto board : kBoardsWithoutSubmissionTimestamps) {
    if (device_name.find(board) != std::string_view::npos) {
      return false;
    }
  }
  return true;
}

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
// Bounded watchdog/transfer poll cadence (100 ms): 100 samples per default
// 10 s no-progress window. Shared by timeline progress waits and staging
// transfer fence waits.
inline constexpr uint64_t kHangWatchPollNs = 100ull * 1000 * 1000;

// Parsed overrides. Resolved lazily on first call so unit tests can
// override the env before the process reads it.
uint64_t submit_hang_no_progress_ns();
uint64_t submit_max_wall_ns();
// Seconds without counter motion before the stuck-submit warning line
// fires (MLX_OMARCHY_SUBMIT_STUCK_S; 0 disables, default 300).
uint64_t submit_stuck_s();

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
  void shutdown(bool destroy_vulkan_handles = true);

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

  // Blocks until generation |value| can release under the one-generation
  // contract: the drain covers |value| and either a later generation
  // has also drained (submits drain in value order, so observing
  // |value + 1| proves cleanup for |value|) or the queue is idle and
  // the drain has been quiet for one poll interval - the idle tick's
  // rule for the newest generation. True when settled; false after
  // |budget| elapses (no later drain and no idle window) or on
  // shutdown.
  bool wait_settled(
      uint64_t value,
      std::chrono::steady_clock::duration budget);

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
  // Payloads of already-drained completions, released one completion
  // later. Mesa signals a submission's semaphores before its submit-final
  // cleanup releases timeline points, so a completion value on this
  // timeline does not prove the driver finished that submission.
  std::vector<std::shared_ptr<void>> retired_temporaries_;
  uint64_t next_value_{0};
  uint64_t drained_value_{0};
  // When drained_value_ last advanced. The idle tick releases the newest
  // drained generation only after this has settled for one poll interval:
  // while submissions are draining steadily (decode) the tick stays
  // strictly behind, and once the queue goes quiet the final generation
  // still recycles well inside the release contract.
  std::chrono::steady_clock::time_point last_drain_at_{
      std::chrono::steady_clock::time_point{}};
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

  VkPhysicalDeviceType physical_type() const {
    return physical_type_;
  }

  void upload_weights(
      CommandEncoder& encoder,
      VkBuffer destination,
      size_t bytes,
      io::Reader& reader,
      size_t file_offset);

  // Chunked staging transfers (<= 4 MiB host-visible staging buffer) for
  // opt-in device-local activation buffers on discrete GPUs. Runs on a
  // private transfer command buffer + fence serialized through queue_mutex_,
  // so callers never disturb any CommandEncoder's open command buffer.
  void upload_host_to_device(
      VkBuffer destination,
      const void* src,
      size_t bytes);
  void download_device_to_host(
      VkBuffer source,
      void* dst,
      size_t bytes);

  size_t device_local_heap_bytes() const;

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

  void probe_float_power(CommandEncoder& encoder);
  bool float_power_needs_lowering() const {
    return float_power_needs_lowering_;
  }

  uint64_t signal_timeline(VkSemaphore semaphore, uint64_t value);

  void ensure_transfer_resources_locked();
  void check_prior_transfer_locked(const char* direction);
  void wait_transfer_fence_locked(
      VkBuffer target_buffer,
      const char* direction);

  struct TeardownOutcome {
    bool device_destroyed{true};
    bool completion_handles_destroyed{true};
    bool compute_handles_destroyed{true};
  };
  // Retire or preserve this device's Vulkan handles coherently. Idempotent;
  // called by Runtime::teardown() before deciding whether VkInstance may be
  // destroyed, and again as a no-op from ~Device().
  TeardownOutcome teardown_vulkan();

  CapabilityReport caps_;
  CapabilityReport hardware_caps_{};
  VkPhysicalDeviceMemoryProperties mem_props_{};
  VkPhysicalDeviceType physical_type_{VK_PHYSICAL_DEVICE_TYPE_OTHER};
  std::mutex weight_upload_mutex_;
  VkBuffer weight_staging_{VK_NULL_HANDLE};
  VkDeviceMemory weight_staging_memory_{VK_NULL_HANDLE};
  void* weight_staging_data_{nullptr};
  bool weight_staging_coherent_{false};
  bool weight_staging_in_flight_{false};
  std::mutex transfer_mutex_;
  VkBuffer transfer_staging_{VK_NULL_HANDLE};
  VkDeviceMemory transfer_staging_memory_{VK_NULL_HANDLE};
  void* transfer_staging_data_{nullptr};
  bool transfer_staging_coherent_{false};
  VkCommandPool transfer_pool_{VK_NULL_HANDLE};
  VkCommandBuffer transfer_cmd_{VK_NULL_HANDLE};
  VkFence transfer_fence_{VK_NULL_HANDLE};
  StagingTransferState transfer_state_{};
  VkBuffer transfer_in_flight_buffer_{VK_NULL_HANDLE};
  VkDevice device_{VK_NULL_HANDLE};
  VkQueue queue_{VK_NULL_HANDLE};
  std::mutex queue_mutex_;
  std::unique_ptr<ComputeRuntime> compute_;
  std::unique_ptr<CompletionDispatcher> completions_;

 private:
  std::once_flag float_power_probe_;
  bool float_power_needs_lowering_{false};
  bool teardown_done_{false};
  TeardownOutcome teardown_outcome_{};
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

// Honeykrisp GPU-memory heap default for an MLX process (HK_SYSMEM):
// max(50% of |mem_total_bytes|, |mem_total_bytes| - 16 GiB), rounded down
// to 1 MiB like mesa, clamped to 62 GiB (the per-process user VA window
// is ~64 GiB, so a larger heap cannot map). Nullopt when the formula
// would not beat mesa's 50%-of-MemTotal DRIR default (MemTotal
// <= 32 GiB), which leaves 16 GB-class hosts byte-for-byte unchanged.
// Pure; the runtime applies it in configure_honeykrisp_icd().
std::optional<uint64_t> default_hk_sysmem_bytes(uint64_t mem_total_bytes);

// The per-client GPU memory limit of the kernel in bytes, from the
// `client_limit` line of a gpu_memory sysfs file. |parse_kernel_client_limit|
// reads the file text (pure); |kernel_client_limit_bytes| reads the first
// render node that has the file. Nullopt when the file, the key or a nonzero
// value is missing (0 means no limit), which is every kernel without the file.
std::optional<uint64_t> parse_kernel_client_limit(std::string_view gpu_memory_text);
std::optional<uint64_t> kernel_client_limit_bytes();

// The heap (|heap|, or mesa's stock MemTotal/2 when nullopt) never above
// |kernel_client_limit|, rounded down to 1 MiB; nullopt keeps the stock
// heap. A limit at or above the heap, no limit, or a limit under 1 MiB
// returns |heap| unchanged. Pure; the runtime applies it in
// configure_honeykrisp_icd() after default_hk_sysmem_bytes().
std::optional<uint64_t> cap_hk_sysmem_to_kernel_limit(
    std::optional<uint64_t> heap,
    uint64_t mem_total_bytes,
    std::optional<uint64_t> kernel_client_limit);

// One line, without prefix or newline, saying that |heap_bytes| is above
// |kernel_client_limit| and naming both numbers. Empty when the heap fits,
// when the kernel sets no limit or when the heap is unknown. An explicit
// HK_SYSMEM or drirc heap_memory_percent wins over the cap above, so this
// line is what explains the allocation failure that follows. Pure; the
// runtime prints it once per process at device creation.
std::string kernel_limit_heap_warning(
    uint64_t heap_bytes,
    std::optional<uint64_t> kernel_client_limit);

// GatedDeltaUpdate decode on the f32 A_log raw route (Qwen3.5-9B) with Hk != Hv: the parts whose
// default is the untiled prefetch kernel with native GQA (ComputeKernel::GatedDeltaDecodeBF16PfComposedGqa),
// because that route was measured bit-equal to the tiled one and faster on them (G14C and G13C:
// receipts/2026-10-18-gdn-pf-gqa; G16S, the M4 Pro: Qwen3.5-4B-MLX-4bit decode 55.27 -> 60.89 tok/s,
// identical tokens, receipts/2026-11-15-m4-pro-g16s-decode-defaults).
// Keyed on the device name so an unmeasured part keeps its previous default (a G13 part other than G13C
// already runs the untiled kernel with the expansion copies). Pure.
inline bool gdn_pf_gqa_default_part(const std::string& device_name) {
  if (const char* env = std::getenv("MLX_OMARCHY_GDN_PF_GQA")) {
    return env[0] != '\0' && std::strcmp(env, "0") != 0;
  }
  return device_name.find("G14C") != std::string::npos ||
      device_name.find("G13C") != std::string::npos ||
      device_name.find("G17C") != std::string::npos ||  // initial value matching G14C (same Max class) pending calibration on M5
      device_name.find("G16S") != std::string::npos;
}

// GatedDeltaUpdate decode on the bf16 A_log raw route: the (part, head ratio) pairs whose default is the untiled prefetch
// kernel (ComputeKernel::GatedDeltaDecodeBF16Pf); at Hk != Hv with the expansion copies. Measured faster in decode and not worse in teacher-forced NLL
// by the 2 SE rule, but not bit-equal to the tiled kernel:
//   G13C (M1 Max): ratio 2 (Qwen3.6-35B-A3B, 16 and 32 heads, about 8 percent), ratio 3 (Qwen3.8-27B, 16 and 48 heads, 5.8 percent),
//     receipts/2026-10-19-gdn-bf16a-untiled and receipts/2026-10-20-gdn-bf16a-ratio3, and equal head counts, ratio 1
//     (Qwen3.8-2B: 16 and 16 heads, about 3.3 percent with the plain prefetch kernel and 4.0 percent with the wide one), receipts/2026-10-22-gdn-wide-g13c;
//   G14C (M2 Max): ratio 3 (about 6 percent), receipts/2026-10-20-gdn-bf16a-g14c-ratio3, and equal head counts, ratio 1
//     (Qwen3.8-2B: 16 and 16 heads, about 2.8 percent with the plain prefetch kernel and 3.7 percent with the wide one, which is the default there),
//     receipts/2026-10-21-gdn-bf16a-g14c-ratio1; ratio 2 was not measured there.
// Keyed on the device name and the head ratio, so an unmeasured pair keeps the tiled default. Pure.
inline bool gdn_bf16a_untiled_default(const std::string& device_name, int hk, int hv) {
  if (hk <= 0) return false;
  if (const char* env = std::getenv("MLX_OMARCHY_GDN_BF16A_UNTILED")) {
    return env[0] != '\0' && std::strcmp(env, "0") != 0;
  }
  if (device_name.find("G13C") != std::string::npos) return hv == 2 * hk || hv == 3 * hk || hv == hk;
  if (device_name.find("G14C") != std::string::npos) return hv == 3 * hk || hv == hk;
  if (device_name.find("G17C") != std::string::npos) return hv == 3 * hk || hv == hk;  // initial value matching G14C (same Max class) pending calibration on M5
  return false;
}

// Where the default untiled prefetch kernel of gdn_bf16a_untiled_default is the wide one (ComputeKernel::GatedDeltaDecodeBF16PfWide, 32 state
// words in flight per round trip, bit-equal to the plain one): the (part, head ratio) pairs where the wide kernel was measured faster than the plain one:
//   G14C at equal head counts (Qwen3.8-2B: +3.7 percent against the tiled kernel, +2.8 with the plain one), receipts/2026-10-21-gdn-bf16a-g14c-ratio1;
//   G13C at equal head counts (Qwen3.8-2B: +4.0 percent against the tiled kernel, +3.3 with the plain one), receipts/2026-10-22-gdn-wide-g13c.
// Every other pair that takes the untiled kernel keeps the plain one. Pure.
inline bool gdn_pf_wide_default(const std::string& device_name, int hk, int hv) {
  if (const char* env = std::getenv("MLX_OMARCHY_GDN_PF_WIDE")) {
    return env[0] != '\0' && std::strcmp(env, "0") != 0;
  }
  if (!gdn_bf16a_untiled_default(device_name, hk, hv)) return false;
  return hv == hk;
}

// The fused GatedDelta conv + delta + gated norm op (fast::gdn_conv_delta_update with the gated norm folded in, the MLX_OMARCHY_GDN_CONV_DELTA value 2)
// runs by default on the 13-inch (Apple M1, G13G) and the 14-inch (Apple M2 Max, G14C). The 13-inch 2B decode measured +0.51 percent (5 of 5 pairs faster,
// identical tokens, dispatches per token 251 to 215, 5 pairs x 128 tokens at ctx 512).
// The 14-inch (receipts/2026-11-14-g14c-gdn-conv-delta, ABBA, 128 tokens, identical tokens in every pair): Qwen3.8-2B +2.00 percent at ctx 512 and +1.22 at
// ctx 2048; Qwen3.8-9B -0.08 (flat) and -0.14 (0.03 ms per token; both flag arms slower than both default arms: a small steady loss, not noise). The 2B gain is
// larger and one rule covers both models.
// The 16-inch (G13C) is unmeasured, so it keeps the stock route until its own numbers exist. An explicit MLX_OMARCHY_GDN_CONV_DELTA value overrides this
// everywhere.
inline bool gdn_conv_delta_norm_default_part(const std::string& device_name) {
  return device_name.find("G13G") != std::string::npos ||
      device_name.find("G14C") != std::string::npos ||
      device_name.find("G17C") != std::string::npos;  // initial value matching G14C (same Max class) pending calibration on M5
}

// The one resolver for the fused GatedDelta conv + delta op's mode: 0 = the stock route (the op is not called), 1 = conv + delta fused, 2 = conv + delta + gated
// norm fused. An explicit MLX_OMARCHY_GDN_CONV_DELTA value ("0", "1", "2"; anything else reads as 0) overrides the per-chip default. The C ABI
// (mlx_omarchy_gdn_conv_delta_default, eval.cpp) wraps this for the mlx-lm patch, and the op's dispatch consults it directly: both sides see the same answer.
inline int gdn_conv_delta_resolved_mode(const std::string& device_name) {
  const char* env = std::getenv("MLX_OMARCHY_GDN_CONV_DELTA");
  if (env != nullptr && env[0] != '\0') {
    if (env[0] == '1' && env[1] == '\0') {
      return 1;
    }
    if (env[0] == '2' && env[1] == '\0') {
      return 2;
    }
    return 0;
  }
  return gdn_conv_delta_norm_default_part(device_name) ? 2 : 0;
}

// Fused GatedDeltaUpdate on G18G (Apple M6): the fused update and raw
// decode routes miscompute there. On the 2026-11-24 PAR-35 A/B the fused
// path diverged from the ops reference from layer 1 prefill on, while the
// composed path was bit-identical at every layer and step. The part
// composes until the fused kernels match it.
// MLX_OMARCHY_GDN_G18G_FUSED=1 re-enables the fused route for
// measurement. Pure.
inline bool gdn_g18g_composed_default(const std::string& device_name) {
  if (device_name.find("G18G") == std::string::npos) {
    return false;
  }
  const char* env = std::getenv("MLX_OMARCHY_GDN_G18G_FUSED");
  return env == nullptr || env[0] == '\0' || std::strcmp(env, "0") == 0;
}

// ArgMax, ArgMin and LogSumExp over a long row (8192 to 1048576 columns, the vocabulary row of a decode step): the parts whose default is the chunked
// route of MLX_OMARCHY_ARGREDUCE_CHUNKED and MLX_OMARCHY_LSE_CHUNKED (one workgroup per 4096-column chunk plus a merge dispatch). Measured on G14C
// (M2 Max, Qwen3.5-9B 4-bit decode, both on against both off, same tokens): about 3.9 percent on one boot and 3.2 percent on another,
// receipts/2026-10-23-vocab-tail-g14c. On G13C (M1 Max, 2B and 4B) and G13G (M1, 4B) the same switches measured 0.6 percent or less, so those parts keep
// the single-workgroup kernels. Keyed on the device name, so an unmeasured part keeps its previous default. Pure.
inline bool vocab_tail_chunked_default(const std::string& device_name) {
  if (const char* env = std::getenv("MLX_OMARCHY_VOCAB_TAIL_CHUNKED")) {
    return env[0] != '\0' && std::strcmp(env, "0") != 0;
  }
  return device_name.find("G14C") != std::string::npos ||
      device_name.find("G17C") != std::string::npos;  // initial value matching G14C (same Max class) pending calibration on M5
}

// Wave scheduling (MLX_OMARCHY_WAVE_SCHED unset): the parts where the
// encoder records each batch in earliest-wave order by default. G16S (Apple
// M4 Pro): with aurora-silicon/mesa#353 and #343, Qwen3-8B decode +1.4%
// (wave) and +2.7% (both), tokens identical
// (receipts/2026-11-15-m4-pro-wave-sched). G14C (Apple M2 Max): Qwen3-4B-4bit
// decode three paired runs, wave off vs on, identical ids every arm; step logprob
// gate (gdn_logits.py, 32 steps) max |dlogprob|=0; Qwen3.8-2B flat decode,
// same logprob gate (receipts/2026-11-18-g14c-wave-sched-default).
// Other parts keep tape order until measured; MLX_OMARCHY_WAVE_SCHED=1 or =0
// overrides on any part.
inline bool wave_sched_default_part(const std::string& device_name) {
  if (const char* env = std::getenv("MLX_OMARCHY_WAVE_SCHED")) {
    return env[0] != '\0' && std::strcmp(env, "0") != 0;
  }
  // G16S and measured G14C default on; G17C (Apple M5 Max, Borealis):
  // Qwen3.8-2B decode wave off vs on 38.76 -> 40.23 tok/s (+3.8%), ids
  // digest identical (w7C M5-04, receipts/2026-11-23-*).
  return device_name.find("G16S") != std::string::npos ||
      device_name.find("G14C") != std::string::npos ||
      device_name.find("G17C") != std::string::npos;
}

// Two-pass block count for the fused f16 decode SDPA kernel (0 = one pass).
// The schedule native Metal uses on the M1 Max: 64 blocks at 1024 keys, 128 above.
inline uint32_t sdpa_decode_f16_two_pass_blocks(uint32_t keys) {
  return keys > 1024u ? 128u : (keys == 1024u ? 64u : 0u);
}

// The fused f16 decode SDPA kernel (sdpa_decode_native*.comp) uses no cooperative matrix, but its gate has always required one,
// and the M1/M2 parts keep that gate until their own numbers exist. On G16S (Apple M4 Pro, Borealis, no cooperative matrix) the
// gate is 32-wide subgroups with basic, arithmetic and shuffle operations, a 1024-lane workgroup and the shared stream; there the
// fused kernel measured 586 -> 177 microseconds per call at 1100 keys and 1872 -> 308 at 4096 keys against the three-dispatch
// composition (16 query / 8 kv heads, width 128). G15S (Apple M3 Pro, Borealis, no cooperative matrix) takes the fused kernel only
// in its one-pass range (below 1024 keys) and stays on the composition from 1024 keys up because two-pass pass 1 packs each block's
// unnormalized weighted value sum in f16 and can overflow to inf on finite inputs.
inline bool sdpa_decode_f16_without_coopmat_part(
    const std::string& device_name,
    uint32_t keys) {
  if (const char* env = std::getenv("MLX_OMARCHY_SDPA_F16_NO_COOPMAT")) {
    return env[0] != '\0' && std::strcmp(env, "0") != 0;
  }
  // G16S defaults on at every length; G15S defaults on in the one-pass range;
  // G14C and initial G17C (Max class) keep off pending calibration on M5.
  if (device_name.find("G16S") != std::string::npos) {
    return true;
  }
  if (device_name.find("G15S") != std::string::npos) {
    return sdpa_decode_f16_two_pass_blocks(keys) == 0u;
  }
  return false;
}

// Two-pass bf16 4-bit/group-64 prefill (MLX_OMARCHY_QMM_BF16W unset): the parts where a transposed quantized_matmul with enough
// rows (qmm_bf16w_rows_ok) dequantizes the weight to bf16 once (the affine Dequant kernel) and contracts it with the dense bf16
// matmul, instead of dequantizing to float32 tiles inside the cooperative-matrix qmm kernel. Rounding the weight to bf16 before the
// product is what the Metal backend does (its quantized loader stores T tiles for the MMA).
//   G16S (Apple M4 Pro, cooperative-matrix driver), Qwen3-8B-4bit prefill: receipts/2026-11-15-m4-pro-g16s-qmm-bf16w.
// An unmeasured part keeps the single-kernel route. Pure.
inline bool qmm_bf16w_default_part(const std::string& device_name) {
  if (const char* env = std::getenv("MLX_OMARCHY_QMM_BF16W")) {
    return env[0] != '\0' && std::strcmp(env, "0") != 0;
  }
  // G16S defaults on; G14C and initial G17C (Max class) keep off pending calibration on M5.
  return device_name.find("G16S") != std::string::npos;
}

// Intel-shaped bf16 cooperative-matrix prefill (MLX_OMARCHY_QMM_INTEL
// unset): QmmPrefillIntelBF16 (the 8x16x16
// bf16 x bf16 -> f32 subgroup shape) replaces the FMA route by
// default on the Intel Arc parts it was measured on; env=0 switches
// off, env=1 opts an unmeasured part in. Only a device advertising
// that exact shape can dispatch the kernel at all, so a wider
// default cannot reroute anything that lacks it.
//   BMG G21 (Intel Arc Pro B60 eGPU, ANV/xe driver), Qwen3-4B-4bit
//   prefill on jw16: kernel time 1649.6 -> ~770 ms (2.10x); quiet-host
//   5-pair ABBA wall +54% pp512, +94% pp2048, 5/5, PRED_SHA identical
//   (PAR-28, PR #836).
// Apple and NVIDIA parts keep the shipped default. Pure.
inline bool qmm_intel_default_part(const std::string& device_name) {
  if (const char* env = std::getenv("MLX_OMARCHY_QMM_INTEL")) {
    return env[0] != '\0' && std::strcmp(env, "0") != 0;
  }
  return device_name.find("Intel") != std::string::npos &&
      device_name.find("Arc") != std::string::npos;
}

// Short-pass GEMV (MLX_OMARCHY_QMV_SMALL_T) default row cap per part.
// G16S/G13C: 8 (measured ~3x at M=2..4 vs the coopmat tile). G13G: 8 —
// the short-pass GEMV beat the shipped routes at every measured row
// count on jwm1: +15..28% at M=4 vs the small-row chain, +4..20% at M=8
// vs the coopmat tile, each row bit-identical. G14C (M2): 8 — same
// sweep shape: +10..56% at M=8, ~+7% median at M<=4, bit-identical
// (receipts/2026-11-24-qmv-small-t-g13g.md). M=16 keeps the tile;
// M=5..7 ride the same single batch-8 dispatch M=8 wins with.
// Initial G17C (Max class) keeps 0 pending calibration.
inline int qmv_small_t_default_part(const std::string& device_name) {
  if (device_name.find("G16S") != std::string::npos ||
      device_name.find("G13C") != std::string::npos ||
      device_name.find("G13G") != std::string::npos ||
      device_name.find("G14C") != std::string::npos) {
    return 8;
  }
  return 0;
}

// gather_qmm_sub.comp subgroup routes — affine (bits 4/2, group 64/128),
// 3-bit (group 64), and mxfp4 (-DFP_MODE, group 32) — share one kernel
// template: a 256-lane workgroup whose subgroup-uniform SSBO reads
// (SCALE_WORD/BIAS_WORD) return 0 from lanes 128-255 on G18G (Apple M6;
// mesa#737). The fp route's wrong output is measured (omarchy-mlx#655:
// every output ~2^-125 low); the affine/q3 routes ride the same mechanism
// and stay off with it. Measured wrong on m6, correct on M1 (jwm1) with
// the same kernel dispatched. MLX_OMARCHY_GATHER_QMM_SUB=1 forces the
// route on for bisection; =0 forces it off everywhere, as before. Pure.
inline bool gather_qmm_sub_default_part(const std::string& device_name) {
  if (const char* env = std::getenv("MLX_OMARCHY_GATHER_QMM_SUB")) {
    return env[0] != '\0' && std::strcmp(env, "0") != 0;
  }
  return device_name.find("G18G") == std::string::npos;
}
// mxfp4 fp-mode subgroup gather decode (gather_qmm_sub.comp -DFP_MODE):
// the route measured wrong in omarchy-mlx#655. Named alias of the shared
// gate, kept for readability at the fp call site. Pure.
inline bool gather_qmm_sub_fp_default_part(const std::string& device_name) {
  return gather_qmm_sub_default_part(device_name);
}
// Row rule of the two-pass route: the dequantize pass costs one weight read and one bf16 write per call, which enough rows amortize.
// G16S, milliseconds per call, single-kernel against two-pass (best of three runs of six calls):
//   N x K 4096x4096: 256 rows 1.76 against 3.03, 512 rows 3.34 against 3.06, 2048 rows 12.65 against 10.66
//   N x K 12288x4096: 256 rows 4.84 against 8.14, 512 rows 9.42 against 8.60, 2048 rows 36.84 against 31.33
//   N x K 151936x4096 (vocabulary head): 768 rows 170.7 against 200.6, 1024 rows 257.9 against 250.4, 2048 rows 532.1 against 469.8
// so from 1024 rows on any weight, and from 512 rows when the weight has at most 2^26 elements. Pure.
inline bool qmm_bf16w_rows_ok(size_t rows, size_t n, size_t k) {
  if (rows >= 1024) {
    return true;
  }
  if (rows < 512 || k == 0) {
    return false;
  }
  constexpr size_t max_elems = size_t{1} << 26;
  return n <= max_elems / k;
}

// Split-K bf16 decode attention, head width 128 and 256 (MLX_OMARCHY_SDPA_DECODE_SPLIT unset; the thresholds were measured at head width 128; width 256 uses the same rule, receipts/2026-10-28-sdpa-split-hd256, except on the parts with their own sdpa_split_wide_min_keys row): the smallest cache length, in keys, from which the route is the default on a part.
// 0 means the fused kernel stays the default. The route costs 79 to 90 microseconds per call up to 512 keys (two dispatches) and 89 to 113 at 1024 keys, and it wins by more as the cache grows,
// so a short cache keeps the fused kernel. Per part, the smallest length of the measured sweep (128 to 1024 keys, 36-call chains and 36 independent
// calls, default against split on one tree, receipts/2026-10-23-sdpa-split-decode) where the split route is faster in both shapes of work, except G14C, which starts where it is faster in the chain shape of a decode step:
//   G13G (Apple M1): 128 keys (chain 91.5 against 78.6 microseconds; the shortest length measured);
//   G13C (Apple M1 Max): 320 keys (chain 107.3 against 82.5, independent 84.0 against 78.2); the decode default now starts at 256 keys on the 5-pair ABBA win below;
//   G14C (Apple M2 Max): 320 keys with the value-phase pass 1 (chain 89.7 against 64.3; in the independent shape the fused kernel is 2.6 microseconds faster, 55.6 against 58.2, receipts/2026-10-24-sdpa-split-pv); the plain pass 1 first won at 448 keys (chain 127.2 against 86.8, independent 79.4 against 78.6), and at 320 keys it still wins the chain (89.7 against 85.5).
// Qwen3-4B 4-bit decode, 477 prompt tokens and 128 generated: +21.9 percent (G13G), +19.5 percent (G13C), +12.8 percent (G14C). An unmeasured part keeps the
// fused kernel.
// The same sweep (receipts/2026-11-13-sdpa-split-g13s-g14g, chain and independent microseconds per call, fused against split):
//   G13S (Apple M1 Pro): 192 keys (128: chain 55.6 against 96.4; 192: chain 114.4 against 91.2, independent 117.2 against 90.6);
//   G14G (Apple M2): 192 keys (128: chain 86.7 against 88.1; 192: chain 154.8 against 89.2, independent 153.7 against 88.3).
// Qwen3-4B 4-bit decode at a 443-token prompt, 128 generated: +28.2 percent (G13S, 32.4 to 41.5 tokens per second), +20.0 percent (G14G, 17.6 to 21.1),
// tokens identical.
//   G16S (Apple M4 Pro): 320 keys with the value-phase pass 1 (288: independent 55.7 against 57.6, chain 56.9 against 56.5; 320: independent 61.9 against 59.7,
//   chain 61.6 against 56.2; two sweeps of 224 to 6144 keys, receipts/2026-11-15-m4-pro-g16s-decode-defaults/split-k); the plain pass 1 first wins at 1024 keys (147.9 against 84.2).
//   Qwen3-8B 4-bit decode, 128 generated, two rounds, default against the route with the value-phase pass 1: 443-token prompt 39.8 and 39.7 to 40.9 and 40.9 tokens
//   per second (+2.9 percent), 2048 tokens 31.5 to 38.2 (+21.6), 4000 tokens 25.2 to 35.5 and 35.0 (+39.6), tokens identical. Head width 256 starts later on this
//   part, sdpa_split_wide_min_keys.
//   BMG G21 (Intel Arc Pro B60 eGPU, xe driver): 512 keys. The fused kernel runs a fixed 32-workgroup grid on this part (one group per query
//   head) and reaches about 2 GB/s of 456 at ctx 512 — an occupancy wall, not bandwidth (EG-1 profile receipt). Qwen3-4B 4-bit decode,
//   128 generated, ABBA env-forced against default on one wheel: 512 keys 104.4 to 42.2 ms per token (+148 percent), 8192 keys 223.4 to 54.0 (+314 percent),
//   tokens identical (EG-2 receipt). The RTX 5060 Ti on the same host dispatches the same fused kernel but holds 112 tok/s against the
//   split route's 114 at 512 keys — parity, so it keeps the fused kernel.
//   G15S (Apple M3 Pro, 14-core GPU): 192 keys with the 64-key block pass 1 (176: independent 38.3 to 38.4 against 38.5 to 38.6, chain 66.0 to 66.1 against 41.2 to 41.3;
//   192: independent 40.2 to 40.3 against 38.6 to 38.8, chain 69.6 to 70.0 against 41.2 to 41.3; the same order in three sweeps spanning 128 to 16384 keys,
//   receipts/2026-11-22-m3-pro-g15s-sdpa-split); the plain pass 1 first wins at 448 keys (chain 136.4 to 137.0 against 79.6 to 83.0). The 64-key blocks need four
//   query heads per key-value head; other groups run the value-phase pass 1 from 192 keys and win the chain shape of a decode step there (192 keys, 8 key-value heads,
//   chain, the fused kernel against the route: one query head per key-value head 41.9 against 26.7, two 41.6 against 35.0) except eight (97.8 against 98.8, faster from
//   256 keys), and the independent shape from 320 keys (one query head per key-value head: from 640). Pure.
inline uint32_t sdpa_split_min_keys(const std::string& device_name) {
  if (const char* env = std::getenv("MLX_OMARCHY_SDPA_SPLIT_MIN_KEYS")) {
    return static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
  }
  if (device_name.find("G13G") != std::string::npos) return 128;
  // G13C: 5-pair ABBA Qwen3-4B decode (all three split knobs vs default) wins at ctx 256
  // (+9.32%, 5/5, k_len ~264..328), bit-identical tokens; ctx below 256 unmeasured,
  // so the threshold sits at the measured boundary.
  if (device_name.find("G13C") != std::string::npos) return 256;
  // G14C: 5-pair ABBA Qwen3-4B decode (64-key value-phase split vs default) wins at ctx 128 (+1.10%), 192 (+3.02%), and 256 (+7.99%), bit-identical tokens.
  if (device_name.find("G14C") != std::string::npos) return 128;
  if (device_name.find("G17C") != std::string::npos) return 320;  // initial M5 placeholder pending M5 calibration
  if (device_name.find("G13S") != std::string::npos) return 192;
  if (device_name.find("G14G") != std::string::npos) return 192;
  if (device_name.find("G16S") != std::string::npos) return 320;
  if (device_name.find("BMG") != std::string::npos) return 512;
  if (device_name.find("G15S") != std::string::npos) return 192;
  return 0;
}

// Head width 256 (the Qwen3.5 and Qwen3.8 full-attention layers): the smallest cache length from which the route is the default, where a part measured it apart from
// width 128; every other part uses sdpa_split_min_keys.
//   G16S (Apple M4 Pro): 768 keys. Width 256 runs the plain pass 1 there (the width-256 value-phase pass 1 has no default on this part), and Qwen3.5-4B 4-bit decode
//   with the route forced, 128 generated, two rounds: 443-token prompt 64.0 and 64.1 against 64.7 and 65.1 tokens per second; 562 tokens 64.3 and 57.0 against
//   64.4 and 64.9; 749 tokens 64.3 and 64.3 against 63.5 and 64.0; 1038 tokens 64.3 and 64.3 against 61.7 and 62.4; 2048 tokens 63.4 and 63.5 against 58.0 and 57.9; 4000 tokens
//   59.9 and 60.1 against 51.6 and 51.6 (receipts/2026-11-15-m4-pro-g16s-decode-defaults/split-k), tokens identical.
//   G15S (Apple M3 Pro, 14-core GPU): 1408 keys with the width-256 value-phase pass 1 (8 query and 2 key-value heads; 1344: independent 88.7 to 89.1 against 88.6 to 89.3;
//   1408: independent 90.9 to 91.8 against 89.2 to 89.9, chain 253.6 to 254.6 against 97.6 to 99.6; receipts/2026-11-22-m3-pro-g15s-sdpa-split). With 4 or 8 key-value
//   heads the width-256 value-phase limit is passed at 1408 keys and the plain pass 1 runs; it wins both shapes there too (16 query / 4 key-value heads, the fused
//   kernel against the route: independent 170.2 against 136.8, chain 254.9 against 145.7). Pure.
inline uint32_t sdpa_split_wide_min_keys(const std::string& device_name) {
  if (const char* env = std::getenv("MLX_OMARCHY_SDPA_SPLIT_WIDE_MIN_KEYS")) {
    return static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
  }
  // G14C head width 256: 5-pair ABBA Qwen3.8-2B decode loses at ctx 192 (-1.56%, 0/5) and wins at ctx 256 (+1.07%, 5/5), bit-identical tokens. G17C keeps 320 until M5 calibration.
  if (device_name.find("G14C") != std::string::npos) return 256;
  if (device_name.find("G17C") != std::string::npos) return 320;
  if (device_name.find("G16S") != std::string::npos) return 768;
  if (device_name.find("G15S") != std::string::npos) return 1408;
  // G13C head width 256 is unmeasured, so it keeps the previous effective 320 instead of following min_keys down to 256.
  if (device_name.find("G13C") != std::string::npos) return 320;
  return sdpa_split_min_keys(device_name);
}

inline bool sdpa_split_default(const std::string& device_name, uint32_t k_len, int head_dim = 128) {
  uint32_t min_keys = head_dim == 256 ? sdpa_split_wide_min_keys(device_name) : sdpa_split_min_keys(device_name);
  return min_keys != 0 && k_len >= min_keys;
}

// Value-phase variant of the split-K pass 1 (MLX_OMARCHY_SDPA_SPLIT_PV unset): the largest cache length, in keys, up to which the variant is the default on a part.
// 0 means the plain pass 1 stays the default. The variant takes about 19 microseconds off the flat 80 to 90 microsecond floor of the split route on short caches and costs more
// than it saves on long ones. Chain microseconds per call, split against split with the variant:
//   G14C (Apple M2 Max): 128 to 1024 keys 86 to 93 against 64 to 75; 2048 keys 103.8 against 89.1; 4096 keys 124.8 against 150.1; so the limit is 2048;
//   G13C (Apple M1 Max): 128 to 2048 keys 80 to 104 against 59 to 85; 4096 keys 155.6 against 149.4 (tie); 8192 keys 299.7 against 313.2; the chain holds through 2048 but decode ABBA turns past 9 blocks, so the default limit is 1152;
//   G13G (Apple M1): 128 and 512 keys 78 and 88 against 57 and 73, but 1024 keys 113 against 130 and 2048 keys 239 against 268, and decode of Qwen3-4B reads -0.5 percent at 477 prompt tokens and -2.4 percent at 1888, so the default stays off.
//   G16S (Apple M4 Pro): the variant is faster at every measured length, 224 to 16384 keys (2048 keys 145.8 against 109.8 independent, 4096 keys 212.4 against 164.5,
//   6144 keys 278.5 against 226.6, 16384 keys 633.4 against 507.8; receipts/2026-11-15-m4-pro-g16s-decode-defaults/split-k), so the limit is the longest length measured.
//   G15S (Apple M3 Pro, 14-core GPU), independent: 1280 keys 84.8 to 85.5 against 72.9 to 73.4, 1281 keys 87.2 to 87.3 against 106.8 to 107.0 (8 key-value heads, 80 workgroups of
//   128-key blocks against 88), so the limit is 1280 (receipts/2026-11-22-m3-pro-g15s-sdpa-split).
// Qwen3-4B 4-bit decode, 477 prompt tokens and 128 generated, one tree, default against the variant: +4.6 percent on G14C (63.1 to 66.0 tokens per second), +2.8 percent at 1888 tokens. An unmeasured part keeps the plain pass 1. Pure.
inline uint32_t sdpa_split_pv_max_keys(const std::string& device_name) {
  if (const char* env = std::getenv("MLX_OMARCHY_SDPA_SPLIT_PV_MAX_KEYS")) {
    return static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
  }
  // G14C 17-block (136 workgroups at 8 KV heads) ceiling: 5-pair ABBA Qwen3-4B decode wins at ctx 2048 (+3.77%, 5/5, k_len 2057..2120) and drops at ctx 2176 (-5.89%, 0/5, 18 blocks); 17 * 128 = 2176 keys, bit-identical tokens.
  if (device_name.find("G14C") != std::string::npos) return 2176;
  // G13C: 5-pair ABBA Qwen3-4B decode (value-phase variant vs default) wins at ctx 512
  // (+5.37%, 5/5) and ctx 1024 (+4.80%, 5/5, k_len ~1032..1096) and loses at ctx 2048
  // (-8.71%, 0/3 screen); 9 * 128 = 1152 keys, bit-identical tokens.
  if (device_name.find("G13C") != std::string::npos) return 1152;
  if (device_name.find("G17C") != std::string::npos) return 2048;  // initial value matching G14C (same Max class) pending calibration on M5
  if (device_name.find("G16S") != std::string::npos) return 16384;
  if (device_name.find("G15S") != std::string::npos) return 1280;
  return 0;
}

inline bool sdpa_split_pv_default(const std::string& device_name, uint32_t k_len) {
  uint32_t max_keys = sdpa_split_pv_max_keys(device_name);
  return max_keys != 0 && k_len <= max_keys;
}

// 64-key block variant of the value-phase split-K pass 1 (MLX_OMARCHY_SDPA_SPLIT_BK64 unset): the largest cache length, in keys, up to which the variant is the default on a part.
// 0 means the 128-key blocks stay the default. Four query heads per key-value head only. Chain microseconds per call, 128-key blocks against 64-key blocks, ten chains each:
//   G14C (Apple M2 Max): 512 keys 82.6 against 69.2; 1024 keys 81.5 against 75.7 (ranges overlap); 2048 keys 102.8 against 120.3; so the limit is 512.
//   G16S (Apple M4 Pro), independent and chain, two sweeps: 320 keys 54.3 to 55.0 against 39.2 to 41.4; 512 keys 55.9 to 58.7 against 43.8 to 44.9;
//   1024 keys 60.8 to 69.3 against 70.4 to 77.0; so the limit is 512 (receipts/2026-11-15-m4-pro-g16s-decode-defaults/split-k). Qwen3-8B 4-bit decode, 128 generated, two rounds, 128-key against 64-key blocks:
//   426-token prompt 41.6 and 41.5 against 41.9 and 42.6 tokens per second; past the limit the 64-key blocks lose (664 tokens 41.3 and 41.3 against 40.8 and 41.2,
//   970 tokens 41.1 and 41.1 against 39.6 and 40.3), tokens identical.
//   G15S (Apple M3 Pro, 14-core GPU), independent, value-phase 128-key blocks against 64-key blocks: 640 keys 59.3 to 59.6 against 49.1 to 50.7, 641 keys 59.2 to 60.2 against
//   67.3 to 67.7 (8 key-value heads, 80 workgroups of 64-key blocks against 88), so the limit is 640 (receipts/2026-11-22-m3-pro-g15s-sdpa-split).
// An unmeasured part keeps the 128-key blocks. Pure.
inline uint32_t sdpa_split_bk64_max_keys(const std::string& device_name) {
  if (const char* env = std::getenv("MLX_OMARCHY_SDPA_SPLIT_BK64_MAX_KEYS")) {
    return static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
  }
  // G14C 17-block (136 workgroups at 8 KV heads) ceiling: 5-pair ABBA Qwen3-4B decode wins at ctx 512 (+2.20%), 768 (+1.53%), and 1024 (+0.75%, 5/5) and drops at ctx 1088 (-4.67%, 0/5, 18 blocks); 17 * 64 = 1088 keys, bit-identical tokens.
  if (device_name.find("G14C") != std::string::npos) return 1088;
  // G13C: 5-pair ABBA Qwen3-4B decode (64-key blocks vs default) wins at ctx 512
  // (+3.25%, 5/5, k_len ~520..584) and loses at ctx 1024 (-4.52%, 0/3 screen);
  // 10 * 64 = 640 keys, bit-identical tokens.
  if (device_name.find("G13C") != std::string::npos) return 640;
  if (device_name.find("G17C") != std::string::npos) return 512;  // initial M5 placeholder pending M5 calibration
  if (device_name.find("G16S") != std::string::npos) return 512;
  if (device_name.find("G15S") != std::string::npos) return 640;
  return 0;
}

inline bool sdpa_split_bk64_default(const std::string& device_name, uint32_t k_len) {
  uint32_t max_keys = sdpa_split_bk64_max_keys(device_name);
  return max_keys != 0 && k_len <= max_keys;
}

// Width-256 value-phase pass 1 (shaders/sdpa_decode_split_p1v.comp -DSPLIT_R=4 -DHD=256): the Qwen3.5 and Qwen3.8 full-attention layers. Four query heads per
// key-value head only (the partial-sum tile for eight would not fit in shared memory), and V element offset, head stride and key stride multiples of 8 (16-byte
// loads). Flag values: unset follows the per-chip default below; "0" keeps the old pass 1 at every length; any other non-empty value selects the variant.
//
// Per-chip default: pass 1 launches one workgroup per key-value head and 128-key block, and the variant wins while that count is at most a per-chip limit and
// loses above it: 16 on G13G and 64 on G13C, two workgroups per GPU core (8 and 32 cores). The limit is the measured rule, not a fit: every op row (random bf16, one
// decode query, forced flag, ABBA blocks) sorts that way at 1, 2, 3, 4 and 8 key-value heads, and it is one block wide (1 key-value head on G13G: +26.68 percent at
// 2048 keys, 16 workgroups, then -6.91 at 2049, 17; on G13C: +22.48 at 8192 keys, 64, then -5.62 at 8193, 65). A decode step at context N attends N + 1 keys, so the
// model's own call crosses a block boundary one token early; measured in that call: Qwen3.8-2B (2 key-value heads) on G13C +28.42 percent at 3073 keys (50
// workgroups) and -7.06 at 4097 (66), Qwen3.8-9B (4 key-value heads) +30.99 at 1537 (52) and -7.78 at 2049 (68), on G13G the 2B +29.57 at 769 (14) and -8.19 at 1025 (18),
// the 9B +30.62 at 385 (16) and -13.95 at 769 (28). Decode rows and the full table are in receipts/2026-11-12-sdpa-hd256-kv-heads-budget. The count assumes 128-key blocks,
// which is what these parts run (the 64-key block route is off by default there).
//   G14C (M2 Max): 68. Op rows on the M2 Max (receipts/2026-11-14-g14c-sdpa-hd256-pv-limit) win +19 to +29 percent from 32 workgroups up to 68 and lose -10 to -16 percent
//   from 69 up. The edge is one block wide at 1 key-value head (+19.03 percent at 8704 keys, 68 workgroups, then -12.79 at 8832, 69); 4 key-value heads win at 64 and 68, 2 key-value
//   heads win at 64 (and 34 and 66 in the model rows) and both lose at 76, the first count tried above 68. The 68 is not two workgroups per core; it is the measured edge. Models: Qwen3.8-2B +3.4 percent at 512 and +4.66 at 2048 context, Qwen3.8-9B +1.57 and +1.47.
//   G15S (M3 Pro, 14-core GPU): 36. Op rows at 2 key-value heads (8 query heads): the variant is faster than the old pass 1 through 2304 keys (18 blocks, 36 workgroups;
//   independent 94.8 to 103.2 against 132.9 to 138.9) and slower from 2305 (19 blocks, 38 workgroups; 156.8 to 160.1 against 132.8 to 139.0) (receipts/2026-11-22-m3-pro-g15s-sdpa-split).
//   Every other part: no default (the width-128 variant's default is measured; the width-256 one is not).
inline uint32_t sdpa_split_pv_wide_max_workgroups(const std::string& device_name) {
  if (const char* env = std::getenv("MLX_OMARCHY_SDPA_HD256_PV_LIMIT")) {
    return static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
  }
  if (device_name.find("G13G") != std::string::npos) return 16u;
  if (device_name.find("G13C") != std::string::npos) return 64u;
  if (device_name.find("G14C") != std::string::npos) return 68u;
  if (device_name.find("G17C") != std::string::npos) return 68u;  // initial value matching G14C (same Max class) pending calibration on M5
  if (device_name.find("G15S") != std::string::npos) return 36u;
  return 0u;
}

inline bool sdpa_split_pv_wide(int head_dim, uint32_t group, uint32_t kv_heads, int pv_env, uint32_t k_len, const std::string& device_name,
                               uint32_t v_offset, int64_t v_head_stride, int64_t v_key_stride) {
  if (!(head_dim == 256 && group == 4u && v_offset % 8u == 0u && v_head_stride % 8 == 0 && v_key_stride % 8 == 0)) {
    return false;
  }
  if (pv_env > 0) {
    return true;
  }
  if (pv_env != 0) {  // unset: the per-chip workgroup limit
    uint64_t workgroups = static_cast<uint64_t>(kv_heads) * ((static_cast<uint64_t>(k_len) + 127u) / 128u);
    uint32_t limit = sdpa_split_pv_wide_max_workgroups(device_name);
    return workgroups != 0 && workgroups <= limit;  // no heads or no keys never selects it; a part with limit 0 never does either
  }
  return false;  // "0"
}

// Fused GEMV groups (MLX_OMARCHY_FUSED_GEMV) with the variable unset: the parts where the route defaults off because
// the per-node decode GEMV was measured faster than the fused groups:
//   G16G (Apple M4, Honeykrisp Apple9 port): 2026-11-01 calibration.
//   G14C (14-inch Apple M2 Max): Qwen3.8-2B 4-bit decode (5-pair ABBA, bit-identical tokens, FUSED_GEMV=0 vs unset):
//     ctx128 +3.51% (7.8867 -> 7.6186 ms/tok, 5/5), ctx512 +1.98% (7.9255 -> 7.7665 ms/tok, 5/5), ctx2048 +0.41% (8.1436 -> 8.1300 ms/tok, 5/5).
// The trio (MLX_OMARCHY_FUSED_TRIO) keeps its default on every part: on G16G the numerics gate found that the
// standalone trio kernels move Qwen2.5-0.5B perplexity away from the reference.
// Keyed on the device name so an unmeasured part keeps the route on; callers pass the executing device's name
// (fused_route_enabled). An explicit setting wins. Pure.
inline bool fused_gemv_off_default_part(const std::string& device_name) {
  return device_name.find("G16G") != std::string::npos ||
      device_name.find("G14C") != std::string::npos;
}

// Fused Qmm + ResidualAdd + RMSNorm decode epilogue
// (MLX_OMARCHY_QMM_RESIDUAL_NORM unset): the parts where the epilogue is
// on by default because matched ABBA decode measured faster with
// bit-identical output and token ids:
//   G13C (16-inch Apple M1 Max): Qwen3.8-2B 4-bit ctx 512 decode
//     7.689 -> 7.547 ms/tok (-0.142 ms, +1.84%, 3/3 ABBA pairs, chunk=1)
//     and 7.699 -> 7.564 ms/tok (-0.135 ms, +1.75%, 3/3 ABBA pairs, chunk=16),
//     -48 dispatches/tok, identical ids_sha 4784df039da0.
//   G14C (14-inch Apple M2 Max): Qwen3.8-2B 4-bit decode (5-pair ABBA, bit-identical tokens):
//     ctx128 +1.99% (7.8896 -> 7.7281 ms/tok, 5/5), ctx512 +2.17% (8.0410 -> 7.8628 ms/tok, 5/5), ctx2048 +2.06% (8.1166 -> 7.9517 ms/tok, 5/5).
// Other parts stay off until measured; MLX_OMARCHY_QMM_RESIDUAL_NORM=1/0
// overrides on any part, and MLX_OMARCHY_FUSED_CHAIN=0 turns it off. Pure.
inline bool qmm_residual_norm_default_part(const std::string& device_name) {
  return device_name.find("G13C") != std::string::npos ||
      device_name.find("G14C") != std::string::npos;
}

// Pruned greedy vocabulary head (MLX_OMARCHY_NO_GREEDY_PRUNE in mlx-lm's _greedy_head):
// the parts where greedy_quantized_argmax (8 sequential stages) defaults off because
// full projection + argmax (as_linear) was measured faster with bit-identical tokens:
//   G14C (14-inch Apple M2 Max): Qwen3.8-2B 4-bit decode (5-pair ABBA, NO_GREEDY_PRUNE=1 vs unset):
//     ctx128 +6.27% (7.8636 -> 7.4006 ms/tok, 5/5), ctx512 +6.68% (8.0408 -> 7.5261 ms/tok, 5/5), ctx2048 +6.89% (8.1427 -> 7.6195 ms/tok, 5/5).
// Combined G14C 2B decode stack (NO_GREEDY_PRUNE=1 + FUSED_GEMV=0 + QMM_RESIDUAL_NORM=1):
//   ctx128 +8.71% (7.8856 -> 7.2569 ms/tok, 5/5), ctx512 +7.05% (8.0290 -> 7.4991 ms/tok, 5/5), ctx2048 +5.45% (8.1478 -> 7.7367 ms/tok, 5/5).
inline bool greedy_prune_off_default_part(const std::string& device_name) {
  return device_name.find("G14C") != std::string::npos;
}

inline bool greedy_prune_enabled(const std::string& device_name) {
  if (const char* env = std::getenv("MLX_OMARCHY_NO_GREEDY_PRUNE")) {
    if (std::strcmp(env, "1") == 0) {
      return false;
    }
    if (std::strcmp(env, "0") == 0) {
      return true;
    }
  }
  return !greedy_prune_off_default_part(device_name);
}

// Dense bf16 matmul in the linear-layer orientation (row-major lhs, transposed rhs, alpha 1) where both the scalar-FMA kernel
// (MatmulBf16Fma) and the bf16 cooperative-matrix tile (MatmulBF16Coopmat) qualify: the parts where the FMA kernel is the default,
// because it was measured faster at the linear-layer shapes and in bf16 prefill:
//   G16G (Apple M4, Honeykrisp Apple9 port with VK_KHR_cooperative_matrix): receipts/2026-11-01-m4-cooperative-matrix-default.
// Elsewhere the cooperative matrix keeps the pick (receipts/2026-09-12-prefill-fma-qualify). Attention-shaped matmuls are not affected.
// Keyed on the device name so an unmeasured part keeps the cooperative matrix; dispatch_matmul passes the executing device's name. Pure.
inline bool matmul_bf16_fma_over_coopmat_part(const std::string& device_name) {
  return device_name.find("G16G") != std::string::npos;
}

// Forward RoPE in the half-split form with MLX_OMARCHY_NO_ROPE_HALF_SPLIT unset: the parts where the half-split builds
// (FastRope{,Freqs}HalfSplit{BF16,F16}) are the default, because the general kernel's code size was measured to dominate
// a decode-sized RoPE dispatch there:
//   G16G (Apple M4, Honeykrisp Apple9 port): receipts/2026-11-01-m4-rope-half-split.
// The output is bit-identical to the general builds on every part. Keyed on the device name so an unmeasured part keeps
// the general builds; RoPE::eval_gpu passes the executing device's name. An explicit setting wins. Pure.
inline bool rope_half_split_default_part(const std::string& device_name) {
  return device_name.find("G16G") != std::string::npos;
}

// The value of MLX_OMARCHY_NO_ROPE_HALF_SPLIT (nullptr when unset) -> 0: general builds on every part, 1: half-split builds
// on every part, -1: follow rope_half_split_default_part. Only 1, on, true and yes (any case) select the general builds, and
// only "0" forces the half-split builds; any other value, including an empty one, false, off and no, counts as unset, so a
// value that reads as "leave it alone" never puts an unmeasured part on the route. Pure.
inline int rope_half_split_setting(const char* value) {
  if (value == nullptr) {
    return -1;
  }
  std::string v(value);
  for (char& c : v) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  if (v == "1" || v == "on" || v == "true" || v == "yes") {
    return 0;
  }
  return v == "0" ? 1 : -1;
}

} // namespace mlx::core::omarchy
