# Vulkan decode execution

Decode latency includes graph evaluation, Vulkan command recording, GPU execution, and host reads of completed results.
This guide maps those costs to the implementation at [`ec5bb701`](https://github.com/joshuaswarren/omarchy-mlx/tree/ec5bb701440b79734d03d0ff349a24c5935f07c2).
Use it to attribute a token's cost before changing kernels or submission policy.

## From a token to recorded work

The execution path has five stages:

1. **Schedule the token graph.** The mlx-lm patches call `mx.async_eval` for token results.
   The greedy-head path schedules token IDs without evaluating unused log probabilities.
   See the [0.32 generation patch](../patches/mlx-lm-0.32/mlx-lm-greedy-prune.patch).
2. **Evaluate GPU primitives.** [`gpu::eval`](../overlay/mlx/backend/omarchy/eval.cpp#L41-L160)
   tries eager fusion before calling the primitive's `eval_gpu` method.
   Layout handling, capability selection, copies, and output allocation occur inside these paths.
3. **Record compute and transfer commands.** The primitive calls the stream's
   [`CommandEncoder`](../overlay/mlx/backend/omarchy/encoder.cpp#L523-L892).
   A view can share storage without a compute dispatch; a composition can record several dispatches.
4. **Submit a batch.** Finalization and resource budgets close the open command buffer.
   A submission can contain many primitives, fused operations, and buffer copies.
5. **Publish completion.** The completion dispatcher runs handlers and releases eligible resources.
   A required host read waits for completion and refreshes noncoherent mappings.

A Python operation, an MLX primitive, a shader dispatch, a submission, and a generated token are different counting units.
Shapes, dtypes, cache state, device capabilities, and fusion determine their relationship.
No fixed dispatch count applies to every decode model.

### Compiled execution still has host work

[`eval_compiled_tape`](../overlay/mlx/backend/omarchy/compiled.cpp#L40-L185)
resolves tracing arrays against current inputs, derives shapes, identifies broadcasts, and counts consumers.
These maps and sets belong to each evaluation.
The loop also constructs input vectors and temporary arrays for unfused nodes
([compiled.cpp:311-456](../overlay/mlx/backend/omarchy/compiled.cpp#L311-L456)).

Eligible elementwise chains use one fused dispatch.
Other nodes call their own `eval_gpu` implementation.
`mx.compile` therefore does not imply one Vulkan dispatch or a reusable Vulkan command buffer.
A shapeless tape can serve different evaluation-time shapes; its tracing shape cannot substitute for the current shape.
The [fusion gate](../overlay/mlx/backend/omarchy/fused_chain.cpp#L34-L37) is enabled when its environment override is absent.

## Per-dispatch host work

For each ordinary nonzero compute dispatch, the encoder performs these steps:

| Component | Work in the source | Important boundary |
| --- | --- | --- |
| Pipeline lookup | Lock the compute-runtime mutex and look up the static kernel or dynamic cache key. Create a pipeline on a miss. | Cold creation differs from a warm cache hit. |
| Binding validation and ownership | Check binding count and range, register owners, and accumulate workgroup counts. | Bound buffers must survive queued work. |
| Descriptor setup | Allocate one descriptor set, populate buffer addresses/offsets/ranges, and call `UpdateDescriptorSets`. | Pool capacity does not eliminate per-dispatch set allocation. |
| Dependency recording | Classify binding reads/writes and compare ranges with outstanding accesses. | RAW, WAW, and WAR hazards need ordering. |
| Command recording | Bind the pipeline and descriptor set, push parameters, and record `CmdDispatch`. | Recording the command does not wait for its GPU completion. |

The [pipeline cache](../overlay/mlx/backend/omarchy/compute.cpp#L2108-L2142)
avoids repeated pipeline creation, but retains its lookup mutex on hits.

The normal [descriptor pool](../overlay/mlx/backend/omarchy/encoder.cpp#L444-L521)
holds 2,048 sets.
The encoder creates a replacement pool after exhausting that capacity.
It retires the old pool through completion-owned resources.
It does not reset or reuse a descriptor set while recorded commands can reference it.

At this source revision, [`dispatch_compute_pipeline`](../overlay/mlx/backend/omarchy/encoder.cpp#L590-L681)
submits one descriptor write structure per supplied binding in one `UpdateDescriptorSets` call.
For `D` dispatches with binding counts `b_i`, this means `D` set-allocation calls,
`D` descriptor-update calls, and `sum(b_i)` descriptor write structures.
Buffer copies and fills do not use this compute-descriptor path.
These are source-level call counts, not elapsed-time estimates.

### Buffer ownership also costs host work

Binding registration calls the allocator and appends owners to the batch buffer vector.
The evaluator also registers primitive inputs, outputs, and siblings
([encoder.h:275-307](../overlay/mlx/backend/omarchy/encoder.h#L275-L307),
[eval.cpp:112-132](../overlay/mlx/backend/omarchy/eval.cpp#L112-L132)).
The same buffer can appear through several references.

At submission, the allocator stamps the collected buffers with a completion value.
It recounts pending quarantined bytes under its mutex
([allocator.cpp:392-410](../overlay/mlx/backend/omarchy/allocator.cpp#L392-L410)).
On free, an in-flight buffer must remain outside the reuse cache.
Recycling requires the completion-generation rule and execution-fence check
([allocator.cpp:344-385](../overlay/mlx/backend/omarchy/allocator.cpp#L344-L385)).

### Dependency tracking and host records

Gated barriers are enabled by default.
The encoder tracks byte ranges and reflected SPIR-V read/write access.
An overlapping dependency emits a full memory barrier and clears the tracked ranges.
Independent reads can avoid that barrier
([encoder.cpp:158-227](../overlay/mlx/backend/omarchy/encoder.cpp#L158-L227),
[encoder.cpp:741-801](../overlay/mlx/backend/omarchy/encoder.cpp#L741-L801)).

The default dependency-export path also hashes binding signatures and appends a host-side `DepRecord` per compute dispatch.
Those records have no reader or clear operation in this implementation.
Their vector stays with the encoder
([encoder.cpp:137-143](../overlay/mlx/backend/omarchy/encoder.cpp#L137-L143),
[encoder.cpp:782-801](../overlay/mlx/backend/omarchy/encoder.cpp#L782-L801),
[encoder.h:567-581](../overlay/mlx/backend/omarchy/encoder.h#L567-L581)).
These records are separate from the range tracker that enforces GPU ordering.

Wave scheduling is a separate, opt-in path.
It buffers commands and orders them into dependency waves at submission.
Descriptor allocation and updates still occur when each dispatch arrives
([encoder.cpp:109-123](../overlay/mlx/backend/omarchy/encoder.cpp#L109-L123),
[encoder.cpp:683-713](../overlay/mlx/backend/omarchy/encoder.cpp#L683-L713)).

## Submission and synchronization

[`gpu::finalize`](../overlay/mlx/backend/omarchy/eval.cpp#L163-L195)
commits pending work by default; it does not itself join the completion timeline.
The evaluator can also commit at the node, workgroup, or pending-quarantine byte budget.
The compiled interpreter checks node and workgroup budgets inside its loop.
The node default is 4,096; the byte threshold is one-sixteenth of the allocator memory limit.
The workgroup budget depends on the physical device
([encoder.h:95-239](../overlay/mlx/backend/omarchy/encoder.h#L95-L239)).
These budgets preserve responsiveness and bound retained resources.
Removing a submission without preserving these constraints changes more than descriptor overhead.

The encoder uses [four command-buffer slots](../overlay/mlx/backend/omarchy/encoder.h#L474-L486).
`ensure_recording` reuses a completed slot.
When all four are busy, it waits for the oldest completion before recording into that slot
([encoder.cpp:302-344](../overlay/mlx/backend/omarchy/encoder.cpp#L302-L344)).

[`submit`](../overlay/mlx/backend/omarchy/encoder.cpp#L1116-L1398) closes the command buffer,
constructs semaphore arrays, and locks the device queue.
Inside that lock it flushes noncoherent host writes, reserves a completion value, stamps buffers,
and calls `QueueSubmit` with an execution fence.
It also retains the submission data used by bounded recovery.
The queue lock does not span a host completion wait.

A later submission waits on the stream's previous timeline value.
This is a GPU dependency carried by a semaphore, not a synchronous host join.
Removing it can expose data from incomplete earlier work.

### Where the host can wait

| Point | Host wait or maintenance |
| --- | --- |
| Command-buffer slot reuse | Wait for the oldest slot when the ring has no completed slot. |
| Explicit encoder synchronization | Commit pending work, wait for its last completion, then invalidate noncoherent host mappings. |
| Completion drain | Wait for timeline progress, run completion handlers, and serialize resource retirement. |
| Primitive readback or composition | A selector can require data on the host or explicitly settle a composed result. |
| Allocation pressure | An allocation retry can wait for submitted work before reusing quarantined memory. |

Examples of explicit primitive joins include RoPE offset/frequency validation,
GDN composition fallback, and chunked SDPA draining.
Their [call sites](../overlay/mlx/backend/omarchy/primitives.cpp) name the join reason.
A fused route can avoid a composition's intermediate work, but cannot omit a required dependency.

[`CompletionDispatcher::wait`](../overlay/mlx/backend/omarchy/device.cpp#L1851-L1891)
includes timeline progress, completion handlers, and the drained-value join.
The [drain](../overlay/mlx/backend/omarchy/device.cpp#L2027-L2085)
invalidates noncoherent mappings and releases eligible allocator-quarantine buffers through the previous generation
before it publishes its drained value.
Keepalive payloads, including retired descriptor pools
([encoder.cpp:1201-1205](../overlay/mlx/backend/omarchy/encoder.cpp#L1201-L1205)),
retire one generation later. The drain moves the previous payload aside, publishes the drained value,
and frees that payload when the function returns
([device.cpp:2077-2084](../overlay/mlx/backend/omarchy/device.cpp#L2077-L2084)).
The current batch's payload stays in `retired_temporaries_`.
The drained value is therefore not one point at which every resource is gone.
[`join_last_completion`](../overlay/mlx/backend/omarchy/encoder.cpp#L284-L300)
also refreshes mappings after its wait.
The allocator maintains a list of noncoherent allocations; coherent allocations need no Vulkan flush/invalidate range
([allocator.cpp:445-489](../overlay/mlx/backend/omarchy/allocator.cpp#L445-L489)).

The 0.32 [server log-probability patch](../patches/mlx-lm-0.32/mlx-lm-server-skip-logprob-read.patch)
already avoids a token-logprob gather and `.item()` when the request does not need log probabilities.
A serving trace therefore depends on request options as well as the model.

## Attribute a decode result

Keep release throughput and diagnostic attribution separate.
The [profiler](../overlay/mlx/backend/omarchy/gpu_profiler.h#L14-L49)
adds GPU timestamps and an execution barrier for each measured dispatch.
Those commands change scheduling.

| Signal | Meaning and limit |
| --- | --- |
| `gpu_primitive_dispatches` | GPU evaluator entries, including entries that return without recording work. Not a shader count. |
| `vk_compute_dispatches`, copies, fills, submissions | Different command and submission counts. Compare deltas over the same token interval. |
| `vk_descriptor_update_writes` | Supplied buffer-descriptor writes at this revision, not descriptor-update API calls. It is not in the C trace-snapshot struct. |
| `compiled_tape_node_evaluations`, `compiled_tape_dispatches` | Interpreter nodes versus compute commands attributed to compiled evaluation. They are not in the C trace-snapshot struct. |
| Profiler `b`, `d`, `q`, `s`, `j` events | Begin/slot cost, dispatch recording/GPU ticks, queue boundaries, submit cost, and completion-join cost. |

The [trace definitions](../overlay/mlx/backend/omarchy/trace.h#L18-L70)
and [C snapshot](../overlay/mlx/backend/omarchy/trace.h#L149-L160) have different fields.
Use the exported ABI-size function before reading the C snapshot.
GPU counters alone do not count explicit CPU-stream execution.

The dispatch event's host timer starts after descriptor allocation and update.
It therefore cannot measure those earlier costs
([encoder.cpp:660-721](../overlay/mlx/backend/omarchy/encoder.cpp#L660-L721)).
In the default path, compiled graph preparation and Python scheduling also sit outside that timer.
Separate CPU spans are required to attribute those costs.

With wave scheduling on, the timer starts when the node is buffered
([encoder.cpp:691](../overlay/mlx/backend/omarchy/encoder.cpp#L691))
and stops when `submit()` emits it
([encoder.cpp:1036](../overlay/mlx/backend/omarchy/encoder.cpp#L1036)).
That interval can include later graph preparation, other nodes' descriptor work, wave planning
and a command-buffer slot wait, and the intervals of several pending nodes can cover the same work.
Read it as arrival-to-emission latency, not per-dispatch recording cost.

For a comparable result, record source/wheel identity, model hashes, input IDs, request options,
cache state, device/driver, fusion policy, and materialization boundaries.
Report cold pipeline creation separately from warm decode.
Use token-completion boundaries consistently; a decode interval from token one to token 128 contains 127 intervals.
Host scheduling, GPU work, and completion waits overlap, so their measured durations must not simply be added.
