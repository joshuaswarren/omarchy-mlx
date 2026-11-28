// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// GPU evaluation entry points (mlx/backend/gpu/eval.h). Mirrors the CUDA
// backend's flow: run the primitive's eval_gpu, register input buffers as
// encoder temporaries, and commit when work was recorded.

#include "mlx/backend/gpu/eval.h"

#include <cstdlib>
#include <chrono>
#include <vector>

#ifdef MLX_OMARCHY_GPU_PROFILING
#include <cstdio>
#endif

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/step_timeline.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/primitives.h"
#include "mlx/device.h"
#include "mlx/stream.h"
#include "mlx/backend/omarchy/fused_chain.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

void init() {
  // Discovery errors surface later through omarchy::init_error() when a caller
  // actually asks for the GPU device; initialization must not throw here.
  omarchy::init();
}

namespace {
// 0 = undecided for this graph, 1 = first batch commits early, 2 = normal.
// Whether early commit applies is decided per graph from
// MLX_OMARCHY_BATCH_FIRST, which mlx-lm's generate_step sets only around
// prompt processing and the first token (pipelined decode keeps one submit
// per token; an early submit there only adds submit cost).
thread_local int g_first_state = 0;

// MLX_OMARCHY_FOLD_TIMING: accumulates eval() host time per phase into
// omarchy::eval_phase_slots(); the eager-fusion scope dtor prints them.
struct EvalPhaseTimer {
  int phase;
  bool on;
  std::chrono::steady_clock::time_point t0;
  EvalPhaseTimer(int p, bool enabled)
      : phase(p),
        on(enabled),
        t0(enabled ? std::chrono::steady_clock::now()
                   : std::chrono::steady_clock::time_point{}) {}
  ~EvalPhaseTimer() {
    if (on) {
      omarchy::eval_phase_slots()[phase] +=
          std::chrono::duration<double, std::micro>(
              std::chrono::steady_clock::now() - t0)
              .count();
    }
  }
};
} // namespace

void eval(array& arr) {
  static const bool fold_timing =
      std::getenv("MLX_OMARCHY_FOLD_TIMING") != nullptr;
  EvalPhaseTimer total_timer(0, fold_timing);
  if (fold_timing) {
    omarchy::eval_phase_slots()[5] += 1.0;
  }
  omarchy::trace::counters().gpu_primitive_dispatches++;
  if (!arr.has_primitive() || arr.status() == array::Status::evaluated) {
    // A nested settle eval (rope-pair pre-settle) already ran this
    // node and detached it: its kernel is recorded and its data is
    // resident. Re-running it would dispatch on detached inputs.
    if (omarchy::env_set("MLX_OMARCHY_TRIO_TRACE")) {
      std::fprintf(
          stderr,
          "[omarchy-eval] skip id=%zu settled=%d status=%d prim=%d\n",
          arr.id(),
          static_cast<int>(arr.data_shared_ptr() != nullptr),
          static_cast<int>(arr.status()),
          static_cast<int>(arr.has_primitive()));
    }
    return;
  }
#ifdef MLX_OMARCHY_GPU_PROFILING
  ++omarchy::trace::prim_counts()[arr.primitive().name()];
#endif
  auto outputs = arr.outputs();
  if (omarchy::env_set("MLX_OMARCHY_TRIO_TRACE")) {
    std::fprintf(
        stderr,
        "[omarchy-eval] run id=%zu prim=%s ninputs=%zu settled=%d\n",
        arr.id(),
        arr.has_primitive() ? arr.primitive().name() : "<detached>",
        arr.inputs().size(),
        static_cast<int>(arr.data_shared_ptr() != nullptr));
  }
  auto& stream = arr.primitive().stream();
  auto& encoder = omarchy::get_command_encoder(stream);
  // Open-batch state BEFORE this op records: a batch spans every op
  // recorded between commits.
  bool batch_open = encoder.needs_commit();
  const uint64_t batch_open_t0 =
      (!batch_open &&
       MLX_OMARCHY_TIMELINE_UNLIKELY(omarchy::timeline::enabled()))
      ? omarchy::timeline::monotonic_ns()
      : 0;
  if (g_first_state == 0 && !batch_open) {
    g_first_state = omarchy::batch_first_budget() > 0 ? 1 : 2;
    static const bool trace_first = std::getenv("MLX_OMARCHY_BATCH_TRACE") != nullptr;
    if (trace_first) {
      std::fprintf(stderr, "[batchfirst] graph start state=%d last_completion=%llu drained=%llu\n",
                   g_first_state,
                   static_cast<unsigned long long>(encoder.last_submitted_completion()),
                   static_cast<unsigned long long>(encoder.device().completions().drained_value()));
    }
  }
  {
    EvalPhaseTimer dispatch_timer(1, fold_timing);
    // If the array is a tracer hold a reference
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }
    if (!omarchy::try_eval_eager_fusion(arr, stream)) {
      omarchy::trace::current_prim() = arr.has_primitive()
          ? std::string_view(arr.primitive().name())
          : std::string_view("<detached>");
      arr.primitive().eval_gpu(arr.inputs(), outputs);
      omarchy::trace::current_prim() = std::string_view("");
    }
  }

  // Temporaries flush contract. A buffer must be pinned against
  // incomplete GPU work if and only if some recorded dispatch references
  // it. The eval that records work pins its inputs, outputs, and
  // siblings here; the batch carries those pins to its submission, and
  // the dispatcher releases them one completion after that submission.
  // An eval whose primitive records nothing (host-materialized scalars,
  // view rearrangements) pins nothing: no in-flight dispatch can
  // reference its buffers, and every consumer that reads them records
  // its own eval, which pins them as inputs. Without this rule the
  // temporaries of workless evals would accumulate until an unrelated
  // submission flushed them.
  if (encoder.needs_commit()) {
    EvalPhaseTimer bookkeeping_timer(2, fold_timing);
    if (!batch_open) {
      if (MLX_OMARCHY_TIMELINE_UNLIKELY(batch_open_t0 != 0)) {
        omarchy::timeline::record_span(
            omarchy::timeline::Kind::StepBegin,
            omarchy::timeline::SubKind::BatchOpen,
            batch_open_t0,
            batch_open_t0,
            encoder.timeline_id());
      }
      // One scheduler task and one completion notification per batch,
      // attached when the batch opens so that every close path
      // (finalize, event flush contract, node budget, host read sync)
      // carries the pairing exactly once.
      scheduler::notify_new_task(stream);
      encoder.add_completed_handler(
          [stream]() { scheduler::notify_task_completion(stream); });
    }
    // Keep used buffers alive until the submitted work completes. The
    // output is retained too (covers donated storage, where the output
    // reuses an input's buffer): every backing buffer of the submitted
    // commands must outlive the caller's references.
    for (auto& in : arr.inputs()) {
      encoder.add_temporary(in);
    }
    for (auto& s : arr.siblings()) {
      encoder.add_temporary(s);
    }
    encoder.add_temporary(arr);
    // Node budget: flush the batch so a long graph cannot pin unbounded
    // temporaries behind one open command buffer. Byte budget: flush
    // once the intermediates freed under this batch, which nothing can
    // recycle before it submits, reach their share of the batch limit (the
    // memory limit, capped at 90% of the heap; kBatchByteBudgetDivisor).
    // Work budget (MLX_OMARCHY_BATCH_WORK, issue #19): flush once the
    // batch's summed dispatch work-groups reach the per-submission GPU-time
    // proxy, so one submission never holds the queue long enough to stutter
    // the desktop.
    auto& alloc = omarchy::allocator();
    // First-batch-early (MLX_OMARCHY_BATCH_FIRST): per host thread, reset at
    // every finalize (graph end).
    const int budget = (g_first_state == 1)
        ? omarchy::batch_first_budget()
        : omarchy::batch_node_budget();
    if (encoder.nodes() >= budget) {
      g_first_state = 2;
    }
    if (omarchy::batch_over_budget(
            encoder.nodes(),
            encoder.batch_work(),
            budget,
            omarchy::batch_work_budget(
                encoder.device().hardware_capabilities().device_name,
                encoder.batch_has_prefill())) ||
        alloc.pending_quarantine_bytes() >=
            alloc.batch_limit() / omarchy::kBatchByteBudgetDivisor) {
      EvalPhaseTimer commit_timer(3, fold_timing);
      encoder.commit("batch_over_budget");
    }
  }
}

void finalize(Stream s) {
  static const bool fold_timing =
      std::getenv("MLX_OMARCHY_FOLD_TIMING") != nullptr;
  EvalPhaseTimer finalize_timer(4, fold_timing);
  omarchy::trace::counters().omarchy_finalize_calls++;
  // MLX_OMARCHY_DEFER_COMMIT: keep the open batch across graph-eval
  // boundaries so consecutive evals share one command buffer and one
  // submit instead of one submit per eval. Batches still close at the
  // node/byte budgets inside eval() and whenever a waiter needs the work:
  // Event::signal queues onto the batch under defer, and every host and
  // GPU-side wait flushes the encoders that owe it first. Default on for
  // G18G (defer_commits_enabled); =1 opts in elsewhere, =0 opts out.
  auto& encoder = omarchy::get_command_encoder(s);
  if (omarchy::defer_commits_enabled(
          encoder.device().hardware_capabilities().device_name)) {
    return;
  }
  // Flush contract: the evaluator calls finalize at task-throttle points
  // and at graph end, and then waits on task-completion handlers. The
  // open batch must reach the queue here or those waits never complete.
  // Batching still happens: every dispatch recorded between finalizes
  // (one whole graph evaluation) shares one open command buffer.
  {
    static const bool trace_first = std::getenv("MLX_OMARCHY_BATCH_TRACE") != nullptr;
    if (trace_first) {
      std::fprintf(stderr, "[batchfirst] finalize\n");
    }
  }
  g_first_state = 0;
  omarchy::timeline::ScopedEvent tl_fin(
      omarchy::timeline::Kind::StepEnd,
      omarchy::timeline::SubKind::Finalize,
      encoder.timeline_id(),
      static_cast<uint64_t>(encoder.nodes()));
  encoder.commit("finalize");
}


void synchronize(Stream s) {
  omarchy::get_command_encoder(s).synchronize();
}

extern "C" __attribute__((visibility("default"))) void
mlx_omarchy_trace_snapshot(
    mlx::core::omarchy::trace::MlxOmarchyTraceSnapshot* out) {
  auto& c = mlx::core::omarchy::trace::counters();
  out->gpu_primitive_dispatches = c.gpu_primitive_dispatches.load();
  out->vk_submissions = c.vk_submissions.load();
  out->vk_buffer_copies = c.vk_buffer_copies.load();
  out->vk_buffer_fills = c.vk_buffer_fills.load();
  out->vk_compute_dispatches = c.vk_compute_dispatches.load();
  out->omarchy_finalize_calls = c.omarchy_finalize_calls.load();
  out->commit_calls_with_work = c.commit_calls_with_work.load();
  out->commit_calls_noop = c.commit_calls_noop.load();
  out->barriers_emitted = c.barriers_emitted.load();
  out->barriers_skipped = c.barriers_skipped.load();
}

extern "C" __attribute__((visibility("default"))) std::uint64_t
mlx_omarchy_trace_snapshot_abi_size(void) {
  return sizeof(mlx::core::omarchy::trace::MlxOmarchyTraceSnapshot);
}

extern "C" __attribute__((visibility("default"))) void
mlx_omarchy_step_timeline_mark(int is_end, std::uint64_t step_id) {
  mlx::core::omarchy::timeline::record(
      is_end ? mlx::core::omarchy::timeline::Kind::StepEnd
             : mlx::core::omarchy::timeline::Kind::StepBegin,
      is_end ? mlx::core::omarchy::timeline::SubKind::StepEndMarker
             : mlx::core::omarchy::timeline::SubKind::StepBeginMarker,
      0,
      step_id);
}

// The mode the fused GatedDelta conv + delta op runs in: 0 = the stock route (the op is not called), 1 = conv + delta fused, 2 = conv + delta + gated norm
// fused. Resolved by omarchy::gdn_conv_delta_resolved_mode (device.h): an explicit MLX_OMARCHY_GDN_CONV_DELTA value overrides, unset follows the per-chip
// default. The mlx-lm patch reads this once at import: without the symbol (an older wheel) the patch keeps the stock route.
extern "C" __attribute__((visibility("default"))) int
mlx_omarchy_gdn_conv_delta_default(void) {
  auto& encoder = omarchy::get_command_encoder(default_stream(Device::gpu));
  const std::string& name = encoder.device().capabilities().device_name;
  return omarchy::gdn_conv_delta_resolved_mode(name);
}

#ifdef MLX_OMARCHY_GPU_PROFILING
extern "C" __attribute__((visibility("default"))) void
mlx_omarchy_prim_dump(const char* path) {
  std::FILE* f = std::fopen(path, "w");
  if (!f) {
    return;
  }
  for (auto& [name, count] : mlx::core::omarchy::trace::prim_counts()) {
    std::fprintf(
        f, "%.*s,%llu\n", static_cast<int>(name.size()), name.data(),
        static_cast<unsigned long long>(count));
  }
  std::fclose(f);
}

extern "C" __attribute__((visibility("default"))) void
mlx_omarchy_prim_reset(void) {
  mlx::core::omarchy::trace::prim_counts().clear();
}
#endif

} // namespace mlx::core::gpu
