// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// GPU evaluation entry points (mlx/backend/gpu/eval.h). Mirrors the CUDA
// backend's flow: run the primitive's eval_gpu, register input buffers as
// encoder temporaries, and commit when work was recorded.

#include "mlx/backend/gpu/eval.h"

#include <vector>

#ifdef MLX_OMARCHY_GPU_PROFILING
#include <cstdio>
#endif

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/primitives.h"
#include "mlx/backend/omarchy/fused_chain.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

void init() {
  // Discovery errors surface later through omarchy::init_error() when a caller
  // actually asks for the GPU device; initialization must not throw here.
  omarchy::init();
}

void eval(array& arr) {
  omarchy::trace::counters().gpu_primitive_dispatches++;
  if (!arr.has_primitive() || arr.status() == array::Status::evaluated) {
    // A nested settle eval (rope-pair pre-settle) already ran this
    // node and detached it: its kernel is recorded and its data is
    // resident. Re-running it would dispatch on detached inputs.
    if (std::getenv("MLX_OMARCHY_TRIO_TRACE") != nullptr) {
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
  if (std::getenv("MLX_OMARCHY_TRIO_TRACE") != nullptr) {
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
  {
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
    if (!batch_open) {
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
    // recycle before it submits, reach their share of the memory limit
    // (kBatchByteBudgetDivisor).
    auto& alloc = omarchy::allocator();
    if (encoder.nodes() >= omarchy::kBatchNodeBudget ||
        alloc.pending_quarantine_bytes() >=
            alloc.get_memory_limit() / omarchy::kBatchByteBudgetDivisor) {
      encoder.commit();
    }
  }
}

void finalize(Stream s) {
  omarchy::trace::counters().omarchy_finalize_calls++;
  // MLX_OMARCHY_DEFER_COMMIT: keep the open batch across graph-eval
  // boundaries so consecutive custom-kernel dispatches (Parakeet TDT step:
  // 5 dispatches, one mx.eval per step) share one command buffer and one
  // submit instead of one submit per dispatch. Batches still close at the
  // node/byte budgets inside eval() and at every host-read sync; Event
  // flushes self-submit (encoder flush contract). Opt-in via env; off by
  // default everywhere.
  static const bool defer_commits =
      std::getenv("MLX_OMARCHY_DEFER_COMMIT") != nullptr;
  if (defer_commits) {
    return;
  }
  // Flush contract: the evaluator calls finalize at task-throttle points
  // and at graph end, and then waits on task-completion handlers. The
  // open batch must reach the queue here or those waits never complete.
  // Batching still happens: every dispatch recorded between finalizes
  // (one whole graph evaluation) shares one open command buffer.
  omarchy::get_command_encoder(s).commit();
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
