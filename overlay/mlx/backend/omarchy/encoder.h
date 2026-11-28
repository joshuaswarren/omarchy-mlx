// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once
#include <cstdlib>
#include <atomic>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "mlx/array.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/dep_tracker.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/stream.h"

namespace mlx::core::omarchy {

// Storage-buffer binding for |value|: its buffer from byte 0 (kernels
// add the array's element offset themselves) through the last byte its
// data reaches, rounded up to 32-bit word granularity. Not the
// allocation size: allocator round_size bins every request above 1 MiB
// to one of 8 sizes per power of two (up to 12.5% above the request),
// so binding the whole allocation could reach past the device's
// maxStorageBufferRange for an array just under it
// (dispatch_compute_pipeline refuses those). The round exists because
// the byte-lane kernels address whole words: a take over a 5-byte uint8
// table reads input word (in_elem >> 2) and a 3-byte output merges its
// last byte through atomicOr into word (out_elem >> 2), so a logical
// 3-byte range would have the shader touch a byte outside the bound
// range. BYTE_AT extracts only lanes below the array's element count
// and unclaimed output lanes stay at the host pre-zero fill, so the
// widened window keeps results bit-exact.
inline ComputeBinding binding(const array& value) {
  auto* buffer = static_cast<const VulkanBuffer*>(value.buffer().ptr());
  // Bind the full allocation. The range must cover every byte a dispatch
  // might write: the concatenate's second copy writes the upper half of a
  // buffer whose first-half view would declare a shorter range, and
  // conformant drivers drop writes outside the descriptor range. The
  // full-allocation range removes that class entirely.
  ComputeBinding b{
      buffer->buffer,
      0,
      std::max<VkDeviceSize>(buffer->size, 1),
      buffer};
  // The dependency tracker registers the bytes the shader can name
  // through the view, not the whole allocation: two disjoint views of
  // one buffer must not collide (the M2 stall map's ResNorm -> Word
  // edge barriered on every occurrence for ranges that are provably
  // disjoint). The window derives from shape*strides, not data_size,
  // which understates a strided footprint (the concat column slice).
  const auto [lo, hi] = view_hazard_window(
      value.offset(),
      value.ndim(),
      value.shape().data(),
      value.strides().data(),
      static_cast<int64_t>(value.itemsize()),
      buffer->size);
  b.tracked_offset = lo;
  b.tracked_end = hi;
  return b;
}

// Per-stream command recorder over the device's single VkQueue. Recording
// is BATCHED: primitive evals append to an open command buffer and the
// buffer is submitted when a node/work budget is reached, a flush is
// demanded (semaphore operation, host read), or the in-flight ring is
// exhausted. Batching is order-safe because every dispatch is separated
// from its neighbors by a full dependency (MLX_OMARCHY_GATED_BARRIERS=0,
// the default: unconditional pre+post memory barriers per dispatch;
// =1: a barrier only when the node's ranges overlap work recorded since
// the last barrier, tracked per open batch), every submission waits on
// the completion-timeline value of this stream's previous submission
// (Vulkan defines no cross-submission dependency without a wait) and
// cross-submission waits use ALL_COMMANDS stage masks, so merging
// submissions weakens no dependency; it removes the per-eval host join
// between them. Temporaries and completion handlers released per
// submission still release exactly when that submission's GPU work
// finishes, via the device completion timeline.
//
// Command buffers come from a small ring so the device can execute one
// batch while the host records the next; the host blocks only when every
// ring slot is in flight (then it joins just the oldest) or when a host
// read demands it (synchronize()).
//
// Every submission signals a strictly increasing value on the device
// completion timeline; the device's CompletionDispatcher runs this
// submission's handlers and releases its temporaries and queued-semaphore
// ownership when the GPU work finishes. Cross-stream order comes from
// timeline semaphore waits and signals carried in the submissions;
// handler-only submissions (no recorded commands) still signal the
// timeline, so ordering with prior queue work is preserved.
// Caps recorded work and pinned buffers per submission. Larger batches reduce
// submit overhead but extend buffer lifetimes and watchdog exposure.
// Decode graphs (~585 nodes/token on jwm1) split into three submits per token
// at 256 (jwm1 decode profile, 2026-09-23, DecodeGap receipt); 4096 keeps a
// whole decode token on one submit. The byte budget below remains the real
// cap for large-tensor graphs, so prefill flush behavior is unchanged
// (2026-09-08 8.11 GB incident stays covered by the byte budget).
inline constexpr int kBatchNodeBudget = 4096;
// MLX_OMARCHY_BATCH_NODES=<n> overrides the node budget (default
// kBatchNodeBudget). A smaller budget submits the first nodes of a long graph
// while the host is still recording the rest, overlapping host record time
// with GPU execution; scheduling only, results are unchanged.
// MLX_OMARCHY_BATCH_FIRST=<n>: the FIRST batch of every graph evaluation is
// submitted after n nodes (default 0 = off, uses batch_node_budget()), the
// rest of the graph then rides the normal budget. The GPU starts on the head
// of a long graph while the host still records the tail (host record is
// ~45 us per dispatch on Linux and is otherwise fully exposed in synchronous
// paths), at the cost of one extra submit per graph. Scheduling only.
inline int batch_first_budget() {
  // Sampled once per graph eval via the env cache: mlx-lm's generate_step
  // sets the variable around prompt processing and clears it after the
  // first token, both eval-boundary changes.
  const char* e = env_str("MLX_OMARCHY_BATCH_FIRST");
  int n = e ? std::atoi(e) : 0;
  return n > 0 ? n : 0;
}
inline int batch_node_budget() {
  static const int v = []() {
    const char* e = std::getenv("MLX_OMARCHY_BATCH_NODES");
    int n = e ? std::atoi(e) : 0;
    return n > 0 ? n : kBatchNodeBudget;
  }();
  return v;
}
// Estimated GPU time per submission, in dispatch work-groups. A queue
// submission monopolizes the GPU until it finishes: the compositor gets
// the queue only between submissions, so one long submission makes the
// desktop hitch for its whole length (issue #19: ~78 ms submissions
// stutter, 2-6 ms stay smooth, same GPU busy fraction). Dispatch
// durations are not cheaply measurable at record time, so the evaluator
// flushes the open batch once its summed work-group counts reach the
// budget. Per-group GPU cost is chip-dependent (order 0.1 ms per 1000
// groups on the M2 Max, 2 ms per 1000 groups on the 13-inch M1), so the
// default is a per-board table calibrated from device-timestamp profiles.
// A board's budget is the largest swept value that keeps its Qwen3-4B
// decode submission p50 in the 2-6 ms smooth band. For G13S, G14G and
// G17P (receipts/2026-11-13-batch-work-g13s-g14g-g16s-g17p) it must also
// keep the p99 at or under ~40 ms (half the ~78 ms stutter threshold) with
// no sampled submission reaching 78 ms, so the tail cannot hitch the
// desktop and not only the median does. Where one prefill dispatch
// already holds the 5000 tail above ~40 ms (it runs whole at any budget),
// the tail must stay at that 5000 floor instead (G16S Qwen3-8B, G15S
// Qwen3-4B). Each row's receipt, with its tail data where the receipt
// has it:
// receipts/2026-10-02-submission-cap-19 (G14C),
// receipts/2026-10-12-submission-cap-calibration (G13C),
// receipts/2026-11-21-g13g-batch-work-20000 (G13G, prior 20000 row),
// receipts/2026-11-27-g13g-batch-work-35000.md (G13G, 2B single-submit decode),
// receipts/2026-11-13-g18g-batch-work (G18G),
// receipts/2026-11-23-g621-batch-budget-submits (G18G),
// receipts/2026-11-24-m4-g16g-batch-work (G16G),
// receipts/2026-11-15-m4-pro-g16s-batch-work (G16S, 5000 floor),
// receipts/2026-11-22-m3-pro-g15s-batch-work (G15S, 5000 floor) and
// receipts/2026-11-13-g14c-batch-work (G14C 100000: GPU time per
// submission p50 3.9 ms, p99 6.3 ms, max 8.6 ms, queue interval p50
// 4.2 ms; 0 frames over 33.3 ms beside decode, not tested on a real
// compositor). Board codes key the rows like the direct-matmul routes
// above (matmul_direct_select.h): G13C M1 Max, G13G 13-inch M1, G13S M1
// Pro, G14C M2 Max, G14G M2, G15S M3 Pro, G16G M4, G16S M4 Pro, G17P A18 Pro,
// G18G M6. A single dispatch whose group count alone exceeds the budget
// still runs whole: splitting happens between dispatches. Copies
// and fills are not counted; the node and byte budgets keep those
// batches bounded. MLX_OMARCHY_BATCH_WORK=<groups> overrides the table;
// =0 turns the work cap off (node and byte budgets still apply).
// Scheduling only: every submission already waits on this stream's
// previous completion and cross-submission waits use ALL_COMMANDS stage
// masks (batching note above), so splitting at any dispatch boundary
// preserves every dependency and results stay bit-identical.
struct BatchWorkCalibration {
  // CapabilityReport device-name substring, e.g. "Apple M1 Max (G13C)".
  std::string_view board;
  // Calibrated work-groups per submission (decode / general). A row is
  // added only once the board has been swept, so every row below is
  // measured; an unlisted board falls through to kBatchWorkFallback
  // (calibrated_batch_work() returns 0 for it).
  uint64_t budget;
  // Optional tighter work-group budget applied once an open batch records
  // a multi-token prefill / tiled-matrix kernel (is_prefill_work_kernel_name),
  // and used by dispatch_qmm_tile() for QMM prefill slicing. 0 reuses budget.
  uint64_t prefill_budget = 0;
};
inline constexpr BatchWorkCalibration kBatchWorkCalibration[] = {
    {"G13S", 10000},  // M1 Pro (16-inch, 16 GB, 16 GPU cores), Borealis Mesa be147e3625: Qwen3-4B submission p50 2.50 ms, p99 36.2 ms, max 43.2 ms; 20000 raised the p99 to 72.3 ms (max 86.0, at the #19 stutter); decode +17.4% 2B and +20.6% 4B against 5000, tokens identical (receipts/2026-11-13-batch-work-g13s-g14g-g16s-g17p, /tail)
    {"G13C", 80000},  // M1 Max (32 GPU cores): 40000 split every Qwen3-4B decode token mid-eval at layer 29 (~60k WG, 2 submits/tok); 80000 keeps the whole decode step in one submit (decode p50 10.61 ms, all-submissions including ctx=2048 prefill p99 32.05 ms / max 32.09 ms <= 33.3 ms; 200000 reached 63.72 ms prefill tail), +1.57% isolated / +2.30% with chunked vocab tail, greedy ids bit-identical to macOS (receipts/2026-11-26-g13c-batch-work-80000.md)
    {"G13G", 35000, 20000},  // 13-inch M1 (8 GPU cores): 20000 split every Qwen3.8-2B decode token across 2 submits (125 dispatches); 35000 keeps 2B@512 and 2B@2048 decode in 1 submit/tok (2B@512 +1.30% / 5/5 wins, decode p50 18.81 ms / p99 21.32 ms; Qwen3-4B decode 3 submits/tok, p99 21.53 ms at ctx=512 and 21.57 ms at ctx=2048, both <= 33 ms) while prefill_budget=20000 holds 4B prefill at baseline (4B@512 all p99 45.60 vs 45.34 ms; 4B@2048 all p99 83.31 vs 83.04 ms, avoiding the 128.36 ms uncapped-prefill tail), greedy ids bit-identical to macOS (receipts/2026-11-27-g13g-batch-work-35000.md)
    {"G14C", 200000},  // M2 Max: the 100000 row split every Qwen3.8-9B decode token mid-eval at ~181 of ~357 dispatches (1.969 submissions per token); 200000 keeps the whole step in one submit (0.984), greedy ids identical (receipts/2026-11-24-g14c-batch-work-200000). Earlier row: +0.5% (Qwen3-4B), +0.9% (Qwen3.8-2B), +1.4% (Qwen3.8-9B) decode against 40000, 0 frames over 33.3 ms beside decode (receipts/2026-11-13-g14c-batch-work)
    {"G14G", 8000},  // M2 (13-inch, 8 GB, 8 GPU cores), Borealis Mesa be147e3625: Qwen3-4B submission p50 3.64 ms, p99 37.9 ms, max 47.8 ms; 10000 raised the p99 to 48.3 ms (max 56.3); decode +2.7% 2B and +2.7% 4B against 5000, tokens identical (same receipt, /tail)
    {"G15S", 15000},  // M3 Pro (14-inch, 18 GB, 14 GPU cores), Borealis Mesa a196911147: decode +6.6% (Qwen3.8-2B, 12 -> 4 submissions per token; score_bench +6.5%) and +5.3% (Qwen3-4B-2507, 20 -> 8 submissions per token; score_bench +6.2%) against 5000, greedy ids identical; release-build submission GPU time (DRM scheduler) Qwen3-4B decode p50 3.98 ms / p99 4.31 ms / max 4.62 ms, all submissions p99 45.17 ms / max 45.52 ms (5000 floor: 44.80 / 44.89 ms), Qwen3.8-2B all p99 38.42 ms / max 38.51 ms (under 40 ms); 20000 raised 4B all-submissions p99 to 65.14 ms (2B to 43.85 ms) and 30000 reached 80.08 ms with 7.40 ms decode p50 (receipts/2026-11-22-m3-pro-g15s-batch-work)
    {"G16G", 60000},  // M4 (MacBook Pro 14-inch, Mac16,1), system Mesa, v0.7.49: decode +35.1% (Qwen3.5 2B text, 4-bit g64, 34 -> 4 submissions per token, 36.33 -> 49.07 tok/s median of 5) against 5000, greedy ids identical (digest 90280d19); time per token is linear in the submit count (19.52 ms + 0.238 ms per submit, 7 budgets 5000-1000000, residuals under 0.7%), so the row is set by the smooth band, not a knee: about 4.9 ms GPU per decode submission; release-build submission GPU time (DRM scheduler, all submissions incl. prefill) p50 6.14 ms / p99 11.86 ms / max 18.44 ms; 120000 leaves the band (about 9.8 ms) and 1000000 doubles prefill to 0.70 s (receipts/2026-11-24-m4-g16g-batch-work)
    {"G16S", 15000},  // M4 Pro, Borealis Mesa 05ddb00f5c, 2x4 decode Qmm geometry: decode +13.1% (Qwen3-4B), +7.7% (Qwen3-8B), +17.7% (granite-4.0-1b) against 5000, greedy ids identical; release-build submission GPU time (DRM scheduler) Qwen3-4B p50 2.5 ms decode, p99 27.8 ms and max 28.5 ms over all submissions (5000: 27.2 / 28.0), Qwen3-8B max 54.7 ms (5000: 54.2); 30000 reached 94.6 ms (receipts/2026-11-15-m4-pro-g16s-batch-work)
    {"G17C", 100000},  // M5 Max, Borealis Mesa: initial value matching G14C (same Max class) pending calibration on M5
    {"G17P", 5000},  // A18 Pro (MacBook Neo, 8 GB, 5 GPU cores), Borealis Mesa c208afc352: equals the fallback. Qwen3-4B submission p50 4.35 ms, p99 24.8 ms, max 47.2 ms; 8000 would raise the p99 to 48.9 ms (same receipt)
    {"G18G", UINT64_MAX},  // M6, uncapped: one submit per decode token. On the 4096e module the 40000 row mid-token submit boundary cost ~10% of 2B decode (6.49 -> 7.21 tok/s at one submit, ids identical; receipts/2026-11-23-g621-batch-budget-submits). The board runs headless, so the #19 stutter cap does not bind; node and byte budgets still bound large graphs.
    {"RTX 5060 Ti", 40000},  // NVIDIA RTX 5060 Ti 16 GB, driver 615.78.08, opt-in VRAM activations, no display attached: one command buffer per Qwen3-4B decode token, decode +5.8% at 256 tokens (+8.2% from token 6) against 5000, ids identical; decode buffer 9.68 ms p50, load/prefill p99 unchanged at 17.4 ms (receipts/2026-11-21-rtx-5060ti-batch-work)
};
inline constexpr uint64_t kBatchWorkBudget = 40000;

// The budget for a board with no calibrated row: the shortest submissions
// measured to stay in the 2-6 ms band. It is a constant, not the minimum of
// the table, so recalibrating one board cannot change the default of boards
// nobody has measured.
inline constexpr uint64_t kBatchWorkFallback = 5000;

// Pure table lookup for the runtime doctest: the calibrated budget for a
// device name; 0 when no row's board substring matches (an unlisted board).
inline const BatchWorkCalibration* calibrated_batch_work_row(
    std::string_view device_name) {
  const BatchWorkCalibration* best = nullptr;
  for (const auto& row : kBatchWorkCalibration) {
    if (device_name.find(row.board) == std::string_view::npos) {
      continue;
    }
    if (!best || row.board.size() > best->board.size()) {
      best = &row;
    }
  }
  return best;
}

inline uint64_t calibrated_batch_work(std::string_view device_name) {
  const BatchWorkCalibration* best = calibrated_batch_work_row(device_name);
  return best ? best->budget : 0;
}

inline uint64_t calibrated_prefill_batch_work(std::string_view device_name) {
  const BatchWorkCalibration* best = calibrated_batch_work_row(device_name);
  if (!best) {
    return 0;
  }
  return best->prefill_budget > 0 ? best->prefill_budget : best->budget;
}

// True when a ComputeKernel name belongs to a multi-token prefill or tiled
// matrix-multiply family (never 1-token decode GEMV/SDPA/GDN kernels). Used
// with BatchWorkCalibration::prefill_budget so a board can keep single-submit
// decode without enlarging prefill submissions.
inline constexpr bool is_prefill_work_kernel_name(std::string_view name) {
  return name.starts_with("QmmPrefill") ||
      name.starts_with("SdpaCausalFlash") ||
      name.starts_with("GatedDeltaPrefill") ||
      name == "QmmF32" || name == "QmmF16" || name == "QmmBF16" ||
      (name.starts_with("Matmul") && !name.starts_with("MatmulVec"));
}

// The default for an uncalibrated board: kBatchWorkFallback (shortest
// submissions). Under-budging costs a little submit overhead; over-budging is
// the stutter the cap exists to prevent.
inline uint64_t batch_work_fallback() {
  return kBatchWorkFallback;
}

// Test hook (issue 202): a value >= 0 replaces the budget for the whole process (0 turns the cap off), -1 (the default) leaves the
// environment variable and the board table in charge. The environment variable is read once per process, so a test that compares a
// split run with an unsplit one in the same process needs this. Atomic because primitives evaluate on the stream's worker thread.
inline std::atomic<long long>& batch_work_budget_test_override() {
  static std::atomic<long long> value{-1};
  return value;
}

inline uint64_t batch_work_budget(
    std::string_view device_name,
    bool has_prefill = false) {
  if (const long long forced = batch_work_budget_test_override().load(
          std::memory_order_relaxed);
      forced >= 0) {
    return static_cast<uint64_t>(forced);
  }
  // The env override is a process setting: read it once. A set but
  // non-positive value disables the cap, so "set" and its value stay
  // separate statics (issue 207).
  static const bool have_env =
      std::getenv("MLX_OMARCHY_BATCH_WORK") != nullptr;
  static const long long env_value =
      have_env ? std::atoll(std::getenv("MLX_OMARCHY_BATCH_WORK")) : 0;
  if (have_env) {
    return env_value > 0 ? static_cast<uint64_t>(env_value) : 0;
  }
  // Per-device calibration, cached on the name: the call sites run this
  // once per evaluated node, where the uncached path costs one getenv
  // plus an eight-row scan (75 ns measured on x86 with 81 environment
  // entries, issue 207 review), and the name comes from
  // hardware_capabilities(), a stable reference, so a one-entry
  // thread_local compare is enough. The name is copied so a caller
  // passing a temporary cannot leave a dangling view in the cache.
  thread_local std::string cached_name;
  thread_local uint64_t cached_budget = 0;
  thread_local uint64_t cached_prefill_budget = 0;
  thread_local bool cached = false;
  if (cached && cached_name == device_name) {
    return has_prefill ? cached_prefill_budget : cached_budget;
  }
  const uint64_t budget = calibrated_batch_work(device_name);
  const uint64_t prefill = calibrated_prefill_batch_work(device_name);
  cached_budget = budget > 0 ? budget : batch_work_fallback();
  cached_prefill_budget = prefill > 0 ? prefill : batch_work_fallback();
  cached_name.assign(device_name);
  cached = true;
  return has_prefill ? cached_prefill_budget : cached_budget;
}
// The one flush predicate for the open batch, shared by the eager
// evaluator (eval.cpp) and the compiled-tape recorder (compiled.cpp) so
// both close batches at the same points. omarchy_runtime_tests drives it
// deterministically with synthetic dispatch sequences and asserts the
// number of submissions and their boundaries; no GPU involved.
inline bool batch_over_budget(
    int nodes,
    uint64_t work,
    int node_budget,
    uint64_t work_budget) {
  return nodes >= node_budget || (work_budget > 0 && work >= work_budget);
}
// Byte budget for the same batch: freed intermediates stay pinned in the
// allocator quarantine until their batch submits and drains, so the open
// batch may hold at most 1/16 of the allocator memory limit in such bytes
// before the evaluator flushes it. Up to five generations can be pinned at
// once (four ring slots in flight plus the one-generation-late quarantine
// release), so the bound keeps batching-owed memory under a third of the
// limit. A 2,048-token Qwen2.5-0.5B forward held 8.11 GB in one 257-node
// batch against Honeykrisp's 7.56 GiB heap without this (2026-09-08).
inline constexpr size_t kBatchByteBudgetDivisor = 16;

// Deferred commits (DEC-23): keep the open batch across graph-eval
// boundaries so consecutive evals share one command buffer and one
// submission instead of one submission per eval. Under defer, finalize()
// returns without committing and Event::signal queues its semaphore signal
// onto the open batch instead of self-submitting; the batch still closes at
// the node/byte budgets inside eval() and whenever a waiter needs the work
// (Event::wait, host reads, GPU-side event waits flush every encoder that
// owes a deferred signal first). On the M6 (G18G), where each vkQueueSubmit
// costs driver helper work, the eval-boundary commits were ~4 submissions
// per decode token; deferral leaves the per-token host read as the single
// commit point. Default on for G18G; MLX_OMARCHY_DEFER_COMMIT=1 enables it
// elsewhere and =0 disables it everywhere.
inline std::atomic<long long>& defer_commits_test_override() {
  static std::atomic<long long> value{-1};
  return value;
}

inline bool defer_commits_enabled(std::string_view device_name) {
  if (const long long forced = defer_commits_test_override().load(
          std::memory_order_relaxed);
      forced >= 0) {
    return forced != 0;
  }
  static const bool have_env =
      std::getenv("MLX_OMARCHY_DEFER_COMMIT") != nullptr;
  if (have_env) {
    static const long long env_value =
        std::atoll(std::getenv("MLX_OMARCHY_DEFER_COMMIT"));
    return env_value != 0;
  }
  return device_name.find("G18G") != std::string_view::npos;
}

// Greedy earliest-wave schedule for one batch: level[i] is the smallest
// wave index such that every earlier tape node that node i hazards sits
// in a strictly smaller wave. A hazard is RAW (i reads a range an earlier
// node writes), WAW, or WAR (i writes a range an earlier node reads) over
// the exact tracked byte ranges. Nodes of equal level are pairwise
// hazard-free, so a stream that emits waves in tape order with one full
// dependency barrier between waves preserves every dependency of the tape
// order while letting hazard-free nodes run concurrently. Public so tests
// can assert the schedule contract without a device.
MLX_API std::vector<uint32_t> wave_levels(std::span<const WaveNode> nodes);

class MLX_API CommandEncoder
    : public std::enable_shared_from_this<CommandEncoder> {
 public:
  explicit CommandEncoder(Device& device);
  ~CommandEncoder();

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  // Record the array's buffer as referenced by the open batch. The raw
  // buffer is stamped kPendingCompletion now and with the submission's
  // completion value at submit(); the allocator quarantine keeps a
  // buffer whose array died mid-flight out of the reuse cache until the
  // driver provably finished that submission. No shared_ptr is held:
  // eager donation (is_donatable) must see the input's refcount at 1.
  void add_temporary(const array& arr) {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);
    auto data = arr.data_shared_ptr();
    if (!data) {
      return;
    }
    auto* buf = static_cast<VulkanBuffer*>(data->buffer.ptr());
    if (!buf) {
      return;
    }
    allocator().note_batch_buffer(buf);
    batch_buffers_.insert(buf);
  }

  // Record one dispatch binding's owning buffer exactly like
  // add_temporary records an array: ComputeBinding::owner covers the
  // buffers a dispatch binds that no add_temporary call registered
  // (plain input and output arrays), which otherwise could be freed
  // with completion == 0 while the queued commands still reference
  // them, and then be recycled or destroyed mid-flight.
  void note_binding_owner(const void* owner) {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);
    auto* buf = static_cast<VulkanBuffer*>(const_cast<void*>(owner));
    if (!buf) {
      return;
    }
    allocator().note_batch_buffer(buf);
    batch_buffers_.insert(buf);
  }

  // Handlers run on the device completion thread when this submission's
  // GPU work finishes.
  void add_completed_handler(std::function<void()> task) {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);
    completed_handlers_.push_back(std::move(task));
  }

  bool needs_commit() const {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);
    return node_count_ > 0;
  }

  // Deferred event signals (defer_commits_enabled): an Event::signal that
  // queued onto this batch without committing. Setting the flag also
  // registers the encoder in a process-wide set so flush_deferred_signal_
  // encoders() can reach encoders owned by other threads; the flag clears
  // at submit() and the entry is dropped by the flush or the destructor.
  void note_deferred_signals();

  bool deferred_signals_pending() const {
    return deferred_signals_pending_;
  }

  // True while eval_compiled_tape is recording a tape node's dispatches.
  // Set around each node's primitive.eval_gpu call; the GPU profiler
  // tags per-dispatch events with it so dispatch counts attribute
  // between the tape and eager paths. Recording is single-threaded per
  // encoder, so a plain bool is safe.
  bool in_tape_recording{false};

  // Nodes recorded in the open batch. The evaluator flushes the batch at
  // kBatchNodeBudget so a long graph cannot pin unbounded temporaries
  // behind one open command buffer.
  int nodes() const {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);
    return node_count_;
  }

  // Summed dispatch work-group counts of the open batch: the bounded
  // GPU-time proxy the work budget flushes on (batch_work_budget).
  uint64_t batch_work() const {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);
    return batch_work_;
  }

  // True once the open batch has recorded a prefill / tiled-matrix kernel
  // (is_prefill_work_kernel_name), so batch_work_budget() can apply a tighter
  // prefill cap without splitting single-submit decode steps.
  bool batch_has_prefill() const {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);
    return batch_has_prefill_;
  }

  // True when nothing is recorded, nothing is queued for submission, and
  // no batch buffers are pending stamping: the encoder owns no GPU work
  // and no lifetime obligations. An Event::signal issued on this state
  // moves the target timeline counter from the host instead of
  // submitting a signal-only command buffer; the batch-buffer check is
  // what makes the flush contract explicit — a signal on an encoder that
  // still owes a batch flush takes the queued path, which submits and
  // releases.
  bool idle() const {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);
    return !recording_ && pending_.empty() && wait_semaphores_.empty() &&
        signal_semaphores_.empty() && completed_handlers_.empty() &&
        batch_buffers_.empty();
  }

  // Record a device-to-device buffer copy. Both buffers must have
  // VK_BUFFER_USAGE_TRANSFER_* usage bits (all backend buffers do).
  void copy_buffer(
      VkBuffer src,
      VkBuffer dst,
      VkDeviceSize size,
      VkDeviceSize src_offset = 0,
      VkDeviceSize dst_offset = 0);

  // Record one compute dispatch. The binding count must not exceed the
  // device's runtime budget (ComputeRuntime::binding_limit()); kernels
  // needing more refuse by name at the primitive that builds them.
  void dispatch_compute(
      ComputeKernel kernel,
      std::span<const ComputeBinding> bindings,
      const ComputeParams& params,
      uint32_t group_count_x,
      uint32_t group_count_y = 1,
      uint32_t group_count_z = 1);
  void dispatch_compute(
      const std::string& cache_key,
      std::span<const uint32_t> spirv,
      std::span<const ComputeBinding> bindings,
      const ComputeParams& params,
      uint32_t group_count_x,
      uint32_t group_count_y = 1,
      uint32_t group_count_z = 1);
  // The metal2vk route: an explicit entry point and specialization
  // constants (clspv modules name entries after the kernel and take the
  // workgroup size from spec constants 0..2). The cache key must already
  // distinguish the specialization values.
  void dispatch_compute(
      const std::string& cache_key,
      std::span<const uint32_t> spirv,
      const std::string& entry_name,
      const VkSpecializationInfo* spec,
      std::span<const ComputeBinding> bindings,
      const ComputeParams& params,
      uint32_t group_count_x,
      uint32_t group_count_y = 1,
      uint32_t group_count_z = 1);

  // Record a four-byte-word fill. Size and offset must be multiples of 4.
  void fill_buffer(
      VkBuffer dst,
      uint32_t value,
      VkDeviceSize size,
      VkDeviceSize offset = 0);

  // Timeline semaphore operations for Event support. Waits apply to this
  // stream's next submission; signals are flushed by commit(). The keepalive
  // token (the owning Event's shared state) must stay held until the
  // submission completes: a queued wait or signal never stores a bare
  // VkSemaphore whose owner could be destroyed first.
  void add_semaphore_wait(
      VkSemaphore semaphore,
      uint64_t value,
      std::shared_ptr<void> keepalive) {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);
    std::lock_guard<std::mutex> lk(pending_sem_mutex_);
    wait_semaphores_.push_back({semaphore, value, std::move(keepalive)});
  }

  void add_semaphore_signal(
      VkSemaphore semaphore,
      uint64_t value,
      std::shared_ptr<void> keepalive,
      std::function<void(uint64_t)> on_submit = {}) {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);
    std::lock_guard<std::mutex> lk(pending_sem_mutex_);
    signal_semaphores_.push_back(
        {semaphore, value, std::move(keepalive), std::move(on_submit)});
  }

  // Submit recorded work, semaphore operations, and completion handlers.
  // Eager: every finalize flushes, one op-granular submission. Deeper
  // batching was measured (2026-09-03) and made generation 4.5x SLOWER:
  // the evaluator throttles at MAX_ACTIVE_TASKS with finalize plus
  // wait_for_one, so any flush granularity coarser than one op converts
  // its pipelined wait into a stop-and-wait proportional to batch
  // execution time — worse with deeper batches and longer models, never
  // better. A future attempt must flush when the scheduler would block,
  // not on a node budget (upstream-shape work). Safe to call repeatedly;
  // a no-op when nothing is pending.
  void commit();

  // Submit pending work and block (bounded) until it completes.
  void synchronize(const char* reason = "explicit");

  // Order this stream's next submission after every submission reserved so
  // far on any stream, without a host join: the ordering rides the next
  // submission as a timeline wait on the device completion semaphore. This
  // is the device-visible equivalent of a drain for GPU work. It cannot
  // order host writes to mapped memory, which still need synchronize().
  void wait_outstanding_submissions();

  // Persistent per-stream counter pool for the fused split-K merge (sdpa_decode_split_p1m.comp). Each head owns two
  // uint words — an arrival count and a publish count — and the merging workgroup resets both, so the pool is
  // self-cleaning: it needs zeroing only once, at (re)allocation. Sized to the largest head count seen.
  ComputeBinding sdpa_merge_counters(uint32_t heads);

  // True when every submission queued through this encoder has completed
  // and its handlers have run, and no batch is open with un-submitted
  // work. Host reads of input bytes are then sound.
  bool synchronized() const {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);
    return node_count_ == 0 &&
        (last_completion_ == 0 ||
         device_.completions().drained_value() >= last_completion_);
  }

  Device& device() {
    return device_;
  }

  uint16_t timeline_id() const noexcept {
    return encoder_id_;
  }

  uint64_t last_submitted_completion() const {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);
    return last_completion_;
  }

  // Return a 256-byte zero-initialized ticket buffer slice for the current
  // open batch so QmmVecQ4ResNorm* can elect its last finishing workgroup
  // without overlapping in-flight batches or competing encoders.
  ComputeBinding resnorm_ticket_binding();

  // Distinct buffers bound to the open batch. A persistent buffer bound
  // by many dispatches counts once (pre-binding), so this stays small on
  // weight/KV-heavy graphs.
  size_t open_batch_buffer_count() const {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);
    return batch_buffers_.size();
  }

 private:
  // A semaphore operation pending in this encoder, plus the ownership token
  // that keeps the semaphore's owner alive until GPU completion. Signals may
  // carry an on_submit hook: the encoder fires it with the submission's
  // completion value so the Event can publish the generation that actually
  // carries its signal (deferred signals do not know it at queue time).
  struct PendingSemaphore {
    VkSemaphore semaphore;
    uint64_t value;
    std::shared_ptr<void> keepalive;
    std::function<void(uint64_t)> on_submit;
  };

  // One command buffer and one reusable execution-start marker. The event
  // is reset from the host only after this slot's prior completion drained,
  // then set by the device at the head of the next command buffer. It is not
  // destroyed with the encoder because a watchdog error can leave it pending;
  // vkDestroyDevice releases it with the command pool.
  static constexpr int kInFlightCommandBuffers = 4;
  struct Slot {
    VkCommandBuffer cmd{VK_NULL_HANDLE};
    VkEvent started{VK_NULL_HANDLE};
    uint64_t in_flight{0};
  };

  // Descriptor-set pool cache: sets are allocated from a large pool that
  // is created once per kDescriptorSetsPerPool dispatches instead of a
  // pool create/destroy per dispatch. A retired pool is kept alive until
  // the submission recording at retirement time completes, which is after
  // every submission whose batches allocated from it (timeline order).
  static constexpr uint32_t kDescriptorSetsPerPool = 2048;
  VkDescriptorSet acquire_descriptor_set(ComputeRuntime& compute);
  void dispatch_compute_pipeline(
      VkPipeline pipeline,
      ComputeKernel profile_kernel,
      std::span<const ComputeBinding> bindings,
      const ComputeParams& params,
      uint32_t group_count_x,
      uint32_t group_count_y,
      uint32_t group_count_z);
  void upload_device_local_if_needed(
      VulkanBuffer* vb,
      bool read,
      bool write,
      VkDeviceSize offset,
      VkDeviceSize range,
      bool exact_write_coverage);
  static void commit_device_local_write(VulkanBuffer* vb);

  // Pick a completed ring slot (joining the oldest only when all are in
  // flight) and begin recording into it.
  void ensure_recording();

  // Join this encoder's newest in-flight submission (if any) through the
  // CompletionDispatcher, clear last_completion_, and invalidate
  // noncoherent host mappings. Guarantees the newest command buffer has
  // left the pending state.
  void join_last_completion(const char* reason);
  void submit();

  // Dependency-gated barrier state (MLX_OMARCHY_GATED_BARRIERS, default
  // ON since 2026-09-29; set 0 to restore the unconditional pre+post
  // barriers). Buffer ranges recorded by the open batch since the last
  // barrier: a node skips its barrier only when neither its reads nor
  // its writes overlap an unsynced range. Dispatch bindings carry the
  // SPIR-V reflected read/write split, so read-read pairs skip.
  // head_synced_ marks the batch-head dependency resolved: the first
  // node of a fresh batch decides the boundary barrier on the carried
  // tracker (overlap or the access-budget fallback -> barrier; provably
  // disjoint ranges -> skip, riding the submission's in-order wait).
  // M2D-6, phase 2 of docs/research/m2d6-boundary-barrier-design.md.
  static bool gated_barriers();

  // Wave scheduling (MLX_OMARCHY_WAVE_SCHED, per-part default; requires gated
  // barriers). Arrival-ordered nodes buffered by the open batch instead
  // of recording straight into the command buffer; submit() records them
  // in wave_levels() order with one full dependency barrier between
  // waves. A hazard-free node therefore runs concurrently with its wave
  // instead of serializing behind a barrier its tape neighbors induced.
  struct PendingNode {
    enum class Kind : uint8_t { Dispatch, Copy, Fill, Update };
    Kind kind{Kind::Dispatch};
    // Profiler attribution, captured at arrival (emit happens later, at
    // submit): the evaluating primitive's name and the tape flag.
    std::string_view prim{};
    uint32_t tape{0};
    uint64_t host_t0{0};
    // Dispatch payload:
    VkPipeline pipeline{VK_NULL_HANDLE};
    ComputeKernel profile_kernel{};
    std::array<ComputeBinding, kComputeBindingBudget> bindings{};
    uint32_t binding_count{0};
    ComputeParams params{};
    uint32_t group_count_x{1};
    uint32_t group_count_y{1};
    uint32_t group_count_z{1};
    VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
    // Copy/fill/update payload:
    VkBuffer src{VK_NULL_HANDLE};
    VkBuffer dst{VK_NULL_HANDLE};
    VkDeviceSize size{0};
    VkDeviceSize src_offset{0};
    VkDeviceSize dst_offset{0};
    VkDeviceSize write_offset{0};
    uint32_t value{0};
    std::vector<uint8_t> update_data{};
    // Hazard access sets (exact tracked ranges):
    std::array<TrackedRange, kComputeBindingBudget> rd{};
    std::array<TrackedRange, kComputeBindingBudget> wr{};
    uint32_t nr{0};
    uint32_t nw{0};
  };
  std::vector<PendingNode> pending_;
  static bool wave_sched();
  static bool wave_diag();
  // Record the buffered nodes in wave order (wave_levels), one full
  // dependency barrier between waves plus the batch-head barrier, and
  // update the tracker exactly like the tape-order path. Called from
  // submit() before the command buffer is closed.
  void emit_pending();

  /* Phase-1 dependency-export (design 22b395d): per-dispatch records
   * computed from the GATED_BARRIERS tracker, stored host-side, not
   * consumed yet (the driver learns nothing until phase 2). */
  struct DepRecord {
    uint32_t disjoint;
    uint32_t usc_changed;
    uint64_t seq;
  };
  static bool export_dep_masks();
  static bool dep_dump();
  static int dep_dump_n();
  std::vector<DepRecord> dep_records_;
  uint32_t dep_prev_signature_{0};
  uint32_t dep_have_prev_{0};
  uint64_t dep_seq_{0};
  void record_dependency_barrier();
  // Compute-to-compute barrier for a dispatch when MLX_OMARCHY_COMPUTE_BARRIER
  // is on and no transfer ran since the last full barrier; else the full one.
  void record_dispatch_barrier();
  static bool compute_barrier();
  void reset_dependency_tracking();
  DependencyTracker tracker_;
  bool head_synced_{false};
  // A copy, fill or inline update was recorded since the last full barrier.
  bool transfer_since_barrier_{false};
  // The carried tracker's access count crossed
  // kBoundaryTrackerAccessBudget at the last submission boundary: the
  // next recorded head node emits a real barrier and the table cleared
  // there, bounding carried state.
  bool boundary_forced_{false};

  Device& device_;
  uint16_t encoder_id_{0};
  VkCommandPool pool_{VK_NULL_HANDLE};
  std::array<Slot, kInFlightCommandBuffers> slots_{};
  int current_slot_{0};
  VkCommandBuffer cmd_{VK_NULL_HANDLE};
  static constexpr VkDeviceSize kResNormTicketStride = 256;
  mlx::core::allocator::Buffer resnorm_ticket_buf_{nullptr};
  std::array<uint64_t, kInFlightCommandBuffers> resnorm_ticket_in_flight_{};
  int resnorm_ticket_slot_{-1};
  // Compute pipeline bound on cmd_ this recording. Reset at command
  // buffer begin; lets emit skip a redundant CmdBindPipeline when a run
  // of dispatches reuses one pipeline.
  VkPipeline bound_pipeline_{VK_NULL_HANDLE};
  bool recording_{false};
  int node_count_{0};
  uint64_t batch_work_{0};
  bool batch_has_prefill_{false};
  uint64_t last_completion_{0};
  VkDescriptorPool desc_pool_{VK_NULL_HANDLE};
  uint32_t desc_pool_remaining_{0};
  // Serializes every mutation of encoder state: deferred signals let a
  // waiter on another thread commit this encoder, and a foreign commit
  // is only legal when recording, queueing, and submit() cannot overlap
  // (Vulkan's external-synchronization rule). Recursive because public
  // entry points nest (synchronize -> commit -> submit, dispatch ->
  // emit_pending). Readers lock too: a foreign submit() clears the STL
  // containers these methods walk.
  mutable std::recursive_mutex op_mutex_;
  // Guards the pending semaphore lists: deferred signals let a waiter on
  // another thread commit this encoder, so queueing and submit()'s
  // snapshot must serialize against each other.
  std::mutex pending_sem_mutex_;
  std::vector<PendingSemaphore> wait_semaphores_;
  std::vector<PendingSemaphore> signal_semaphores_;
  std::atomic<bool> deferred_signals_pending_{false};
  // Raw buffers referenced by the open batch, pre-bound once each: a
  // persistent buffer (weights, KV) touched by many dispatches inserts
  // on its first binding and dedups on every later one, so submit() and
  // the allocator scan it once per batch rather than once per dispatch
  // binding. Owned by their arrays or, after free(), by the allocator
  // quarantine; the set only carries the completion stamp at submit
  // time.
  std::unordered_set<VulkanBuffer*> batch_buffers_;
  mlx::core::allocator::Buffer sdpa_counter_buf_{nullptr};
  uint32_t sdpa_counter_words_{0};
  // Descriptor pools retired during recording. They ride the dispatcher
  // payload (one-generation retire) so a set allocated from them cannot
  // outlive its pool.
  std::vector<std::shared_ptr<void>> retired_pools_;
  std::vector<std::function<void()>> completed_handlers_;

  // Per-submission scratch (the per-token arena): this storage survives
  // submit(), so emit_pending() and submit() stop re-allocating the same
  // vectors on every batch.
  std::vector<WaveNode> wave_nodes_;
  std::vector<uint32_t> wave_levels_buf_;
  std::vector<uint32_t> wave_counts_;
  std::vector<uint32_t> wave_offsets_;
  std::vector<uint32_t> wave_order_;
  std::vector<uint8_t> wave_has_transfer_;
  WaveIndex wave_index_;
  std::vector<VkSemaphore> submit_wait_sems_;
  std::vector<uint64_t> submit_wait_values_;
  std::vector<VkPipelineStageFlags> submit_wait_stages_;
  std::vector<VkSemaphore> submit_signal_sems_;
  std::vector<uint64_t> submit_signal_values_;
};

MLX_API CommandEncoder& get_command_encoder(Stream s);
std::unordered_map<int, std::shared_ptr<CommandEncoder>>&
get_command_encoders();
std::unordered_map<int, std::shared_ptr<CommandEncoder>>&
get_global_command_encoders();
void synchronize_pending_encoders();

// Commit every encoder holding a deferred event signal. Waiters call this
// before blocking on a timeline a queued signal could satisfy: the signal
// rides the owning batch's submission, which must exist for the wait to
// ever observe it.
void flush_deferred_signal_encoders();

} // namespace mlx::core::omarchy
