# Omarchy distributed primitives on the GPU stream (2026-10-04)

Status: dev-box and local two-rank M2 tests pass. The focused M2 tests passed AllReduce sum/max/min across fp32/fp16/bf16/int32 and confirmed upstream ring split refusal on both ranks. The wired CPU-stream cross-version gate now passes with Mac Nix Python 3.13.12 / MLX 0.32.2 and M2 MLX 0.32.4: AllReduce sum/max/min, AllGather, and Send/Recv values verified on both ranks for fp32/bf16/int32 at 4 KiB, 64 KiB, and 1 MiB. The earlier Homebrew-Python runs on wired and Tailscale completed the first TCP handshake but stalled before the reverse connection; the Nix-Python wired run completed both directions. Upstream RingGroup::sum_scatter returns not supported, so no cross-version ReduceScatter result is claimed. GPU interop, pipeline generation, and performance gates remain unrun; jwm1 still requires Main's OmarchyDistributed release.

## What changed

- `patches/mlx-ring-gpu-transport.patch` (new; applied by `scripts/prepare-mlx.sh`): the ring transport's communication stream always selects the GPU stream for the omarchy backend, so missing GPU hardware cannot fall back into CPU tensor evaluation. Non-omarchy builds retain the upstream CPU stream. Transport bodies run inline on GPU streams; the CPU command encoder remains available for upstream builds.
- `overlay/mlx/backend/omarchy/primitives.cpp`: the five `OMARCHY_UNSUPPORTED_MULTI` markers (AllReduce, AllGather, Send, Recv, ReduceScatter) are replaced by real `eval_gpu` implementations. The design keeps every tensor operation on the GPU:
  - `gpu::synchronize` ends/submits the stream and waits for the producing work before the transport reads mapped bytes.
  - AllReduce now follows the upstream RingGroup segment/chunk/direction schedule and runs each reduction step with GPU elementwise kernels between host byte exchanges. It still requires cross-version and expanded dtype/uneven-segment validation.
  - Reduction arithmetic stays on GPU. Upstream CPU `SumOp`/`MaxOp`/`MinOp` loops are not called.
  - Complex64 Min follows upstream's lexicographic real-then-imaginary ordering.
  - `ReduceScatter` remains implemented with gather + GPU sum and returns the owning rank's chunk. `Recv`/`AllGather` write peer bytes directly into the mapped output buffer; the next submit orders device reads. The allocator maps buffers persistently, prefers host-visible/coherent memory, and registers non-coherent fallbacks for full-range flush before queue submit and invalidate after completion waits.
  - And/Or/Prod all_reduce and non-sum reduce_scatter keep the exact compatibility error.
- `overlay/tests/omarchy/distributed/test_two_rank.cpp`: obsolete sum_scatter refusal assertion deleted; replaced with two-rank values and int32, bfloat16, complex64 lexicographic Min, and lazy-producer all_sum cases.
- `overlay/tests/omarchy/test_distributed_ops.cpp`: header contract comment updated; singleton short-circuit remains.

## Why the stream change is safe

Upstream 0.32.3 has no GPU implementation of these primitives (the Metal `eval_gpu` bodies throw). The op layer (`mlx/distributed/ops.cpp`) builds each primitive on `detail::communication_stream(group, s)`; with the stream now on the accelerator device, the omarchy eval_gpu bodies are the only distributed path for this backend. The omarchy ring group always selects the GPU stream; a missing GPU cannot silently route a tensor operation through the CPU backend. Non-omarchy MLX builds retain the upstream CPU transport stream.

## MLX-LM 0.31.3 pipeline check

The upstream v0.31.3 tag (commit ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd) contains `mlx_lm/examples/sharded_generate.py`, not a file named `pipeline_generate.py`. The example imports `sharded_load`, initializes `mx.distributed`, and selects pipeline mode with `--pipeline`. `mlx_lm/models/pipeline.py` `PipelineMixin` partitions model layers by group rank. Sources read from the tag's raw GitHub URLs during this task.

## Gates

| Gate | Result |
| --- | --- |
| Fresh prepare applies `mlx-ring-gpu-transport.patch` at fuzz 0 | pass (dev box) |
| Changed primitive translation unit compile | pass |
| Targeted `omarchy_two_rank_harness` build after complex64 Min | pass: 7 build steps, including shader generation |
| `omarchy_distributed_tests` singleton suite (llvmpipe) | pass: 9 tests, 36 assertions |
| Two-rank harness on llvmpipe (software Vulkan) | pass: both ranks, 8 cases / 23 assertions each, after explicit-stream fix for integer reduction temporary outputs; used an ephemeral `VK_DRIVER_FILES` manifest pointing to system Lavapipe because no packaged Honeykrisp ICD manifest is installed; not a Honeykrisp hardware verification |
| Fresh-dir aarch64 wheel build on the M2 (CPU, niced) | pass: 416316259 bytes; SHA256 `3cac729b301906a57e0a18040dc03683b59fe9690051262167bcf50a71060f8a`; build log `/tmp/od-m2-cpu-wheel-20261004203213.log`; source-commit receipt unknown (stamp contains `b59f4be`) |
| Fresh corrected-source M2 wheel build | pass: 416320199 bytes; SHA256 `b70f2885b843b732e093c7a4f026390a054131eb6a9abc3180f62d9032ee2b4e`; source `b59f4be`; `mlx_provenance.py` matched wheel RECORD |
| Corrected M2 wheel after int32 max/min stream fix | pass: 416320195 bytes; SHA256 `25eeaf516682d88675f809392afb1c4dfdb4864fc1a4c2dce80d954efa40e8ec`; wheel `mlx_omarchy-0.32.4.dev202610042215+b59f4be-cp314-cp314-linux_aarch64.whl`; provenance verified `match` |
| Updated fresh-dir aarch64 wheel build | pass: 416316314 bytes; SHA256 `ee023e6268fc02461449a0342fb4a72fc6e2a73af803dbbdffde557c7c735472`; wheel `mlx_omarchy-0.32.4.dev202610042051+b59f4be-cp314-cp314-linux_aarch64.whl`; source-commit receipt unknown (stamp contains `b59f4be`) |
| M2 two-rank upstream `ring_test_distributed.py` | 14 tests ran; both ranks had 2 errors in `test_all_reduce` and `test_send_recv`, both due missing Omarchy `Take` kernel for int8 `[1,7]`; complex64 Min no longer errors; selected applicable `test_groups`, `test_all_gather_extra`, `test_all_gather_vjp` each passed on both ranks |
| M2 two-rank direct GPU smoke (previous revision) | both ranks pass AllReduce sum/max/min, AllGather, Send/Recv; `mx.default_device()` reported `Device(gpu, 0)` |
| Corrected-schedule M2 Python 2-rank smoke, first wheel | failed: both ranks SIGSEGV in `dispatch_int_elementwise`; GDB showed the call chain `dispatch_int_elementwise` → `distributed_reduce` → `AllReduce::eval_gpu` |
| Corrected-schedule local llvmpipe harness after explicit-stream integer dispatch | pass: both ranks, 8 cases / 23 assertions each; includes AllReduce sum/max/min, gather, send/recv, sum_scatter, complex64 min, int32 sum, bf16 sum, lazy input |
| M2 two-rank extended GPU smoke | both ranks pass complex64 Min and int32 sum_scatter with expected per-rank values |
| M2 singleton GPU C++ suite | not run on M2; dev-box singleton suite above passed |
| M2 local ring, focused Python AllReduce | pass: both ranks, sum/max/min × fp32/fp16/bf16/int32, 3 elements per input (uneven 2+1 segment split) |
| Cross-version AllReduce, M2 0.32.4.dev202610042215 ↔ macOS 0.32.2 (Tailscale path) | pass both rank assignments; both ranks verified all 12 dtype/op combinations; macOS `Device(gpu, 0)`; memory pressure 70% free before test |
| macOS interop transport payload (derived from the 3-element inputs; not a throughput measurement) | each collective sends/receives 12 bytes for fp32/int32 or 6 bytes for fp16/bf16 per rank; 2 ring exchanges per collective (reduce + forward); 12 collectives / 24 exchanges per rank across the test |
| M2–macOS wired CPU-stream cross-version gate (Mac Nix Python 3.13.12 / MLX 0.32.2 ↔ M2 MLX 0.32.4) | pass on both ranks for AllReduce sum/max/min, AllGather, and Send/Recv; fp32/bf16/int32 × 4 KiB/64 KiB/1 MiB, all values asserted. M2 strace shows both connect/accept directions succeed; Mac en0 packet capture shows both handshakes and payload traffic. No throughput timing claimed. |
| Homebrew-Python cross-version CPU-stream attempts (wired and Tailscale) | blocked before collectives: TCP connect M2→Mac completed, but rank0 stayed in accept without the reverse connect; both ranks timed out. Replaced for the successful wired test with the allow-listed Nix Python. |
| M2–macOS two-host ring + sharded mlx-lm pipeline | not run; GPU-path interop remains untested, and jwm1 has not been released for the pipeline test |
| Remaining cross-version GPU / ReduceScatter gate | GPU-stream interop remains untested until M2 is released after the reboot window. Upstream ring 0.32.2 does not implement sum_scatter, so CPU-stream ReduceScatter interop is unavailable; Omarchy GPU ReduceScatter still needs its focused target-hardware run. |
| OMLX advisor alias smoke after ring test | direct OMLX alias returned model-not-found; canonical LiteLLM alias `curl --max-time 100` timed out with exit 28, HTTP 000, 0 bytes |

Measurement receipts are appended here as each gate runs. Hostnames are placeholders per the public-repo scrub rule.

## Design notes

- AllReduce mirrors upstream RingGroup: for each lane and phase/hop, exchange one bounded segment packet, perform phase-0 reduction on GPU, then phase-1 forwarding. The implementation reproduces upstream lane count, chunk size, packet size, direction, and segment progression; cross-version 0.32.4-dev ↔ macOS 0.32.2 tests passed for sum/max/min across fp32/fp16/bf16/int32, including uneven three-element segments.
- ReduceScatter still uses gather+GPU reduction and a zero-copy chunk view; upstream multi-host wire behavior has not been validated.
- Send/Recv keep upstream blocking semantics (send blocks until the peer receives); paired operations must be evaluated in receive-before-send order on the receiving rank. The first smoke harness tried to evaluate both ranks' sends first and timed out; correcting only the harness order made both ranks pass.


## 2026-10-05 correction

- Replaced the gather-all AllReduce with the upstream RingGroup schedule and GPU reductions between network exchanges. The changed translation unit compiles and links in the existing build. The first two-rank run exposed a SIGSEGV in `dispatch_int_elementwise` because the manually allocated intermediate output has no Primitive. Added an explicit Stream parameter to the existing integer dispatch path and passed the distributed stream through. Rebuild and fresh harness pass: both ranks, 8 cases / 23 assertions each.
- The corrected M2 wheel passed both local and cross-version two-rank AllReduce smoke. No large transfers were used.

- Mapped-memory check: `VulkanAllocator::malloc` keeps host-visible memory persistently mapped, prefers `HOST_COHERENT`, and registers non-coherent buffers for whole-range flush at queue submit and invalidate after completion waits (`allocator.cpp:132-168,306-349`; `encoder.cpp:1155-1156`; `encoder.cpp:296`).

## Follow-up gate run

| Gate | Result |
| --- | --- |
| M2 local two-rank focused Python AllReduce + ring group contract | pass: both ranks exit 0; sum/max/min across fp32/fp16/bf16/int32, 3-element inputs; upstream ring split refusal asserted on both ranks. Log recovered from GPU ticket `bg_6016`. |
| M2 0.32.4 ↔ macOS 0.32.2 two-host transport gate | timed out after 1200 seconds. Mac rank was terminated and checked absent; no wire compatibility conclusion. |
| macOS OMLX advisor alias smoke through LiteLLM | `curl --max-time 100` returned exit 28, HTTP 000, 0 response bytes. No service was restarted or reconfigured. |
| M2+jwm1 pipeline generation | not run; jwm1 has not been released for this task. |
| Device performance measurements | not run; no verified throughput or per-step communication timing is claimed. |
