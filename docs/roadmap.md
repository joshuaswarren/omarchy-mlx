# Roadmap

This roadmap uses proof gates, not calendar dates.
Each gate publishes a release before the next gate starts.

## v0.1.0: Vulkan runtime core

(1) Add the `MLX_BUILD_OMARCHY` build path.
(2) Select Honeykrisp through `mx.gpu`.
(3) Add allocation, copies, streams, events, queues, and bounded synchronization.
(4) Preserve buffer offsets, ownership, and handler order.
(5) Add backend traces and a capability report.
(6) Build without CPU primitive evaluation.

Exit receipt: Runtime and copy tests pass on M1.
The trace reports zero CPU primitive dispatches.
A fresh process reopens the device.

## v0.2.0: Vulkan kernel performance

(1) Add the minimum matmul and attention kernels for prefill and decode.
(2) Compare against the pinned `llama.cpp` Vulkan build on the same machine.
(3) Balance the case order across both programs.
(4) Run CPU references and negative controls before timing.

Exit receipt: The [full M1 run](https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.2.0/mlx-omarchy-v0.2.0-m1-kernel.json) passed every speed gate.
Every matched result exceeded the 0.80 release threshold.

## v0.3.0: Vulkan primitives and dtypes

(1) Add common elementwise, reduction, indexing, sort, scan, and normalization operations.
(2) Add convolution and attention support.
(3) Add matmul and quantized matmul.
(4) Preserve BF16 and low-bit semantics with Vulkan packing and conversion.
(5) Generate the compatibility matrix from test output.

Exit receipt: The fixed Qwen reference passes every required primitive and dtype test.
The backend trace reports no CPU tensor dispatch.

## v0.4.0: MLX transforms and compilation

(1) Connect Vulkan evaluation to lazy graphs and streams.
(2) Pass `grad`, `vjp`, `jvp`, and `vmap` tests.
(3) Add `mx.compile` fusion through cached SPIR-V.
(4) Run the supported operation and dtype matrix without a CPU tensor backend.
(5) List each exclusion and its named compatibility error.

Exit receipt: The supported M1 matrix passes.
Every excluded case fails with its operation, dtype, and shape.

## v0.5.0: Installable Vulkan compatibility

(1) Publish an `mlx-omarchy` distribution that provides the `mlx` Python module.
(2) Package the C++ library and headers.
(3) Publish `constraints/mlx-omarchy.txt` for ecosystem installs.
(4) Add an import guard for conflicting upstream `mlx` wheels.
(5) Prove a clean M1 Omarchy install.
(6) Run MLX-LM generation and serving with backend traces.
(7) Qualify one public workflow for each named ecosystem project.

Exit receipt: A clean machine installs the public wheel.
MLX-LM runs with zero CPU tensor dispatches.
Each ecosystem row has evidence or a failing conformance case.

## v0.6.0: ANE integration

ANE work starts after `v0.5.0`.
It uses the internal gates below.

### Complete the ANE model proof

(1) Run the corrected 13-layer Qwen graph with workspace buffer 3.
(2) Feed its live output and state into the 11-layer tail.
(3) Compare every boundary with the pinned macOS reference.
(4) Record descriptor, buffer, firmware, model, and graph hashes.

Exit receipt: One Linux token-to-logits run matches the accepted reference.

### Build versioned ANE bundles

(1) Build the open-source compiler on Linux, without Apple frameworks.
(2) Prove target-correct M1 code generation separately from the Linux host port.
(3) Compile one supported operation on Linux, validate the bundle, and execute it through the bounded M1 worker.
(4) Lower supported MLX regions through that path and publish the compiler and exact compatibility manifest.
(5) Keep private Apple software out of source and release assets.

Exit receipt: An MLX-built region exports and runs on Linux.
The runtime rejects each changed compatibility field.

### Stabilize the ANE runtime ABI

(1) Land proven buffer and kernel-window bindings in our `joshuaswarren/omarchy-ane` fork.
(2) Add capacity queries and state-indexed repeated execution.
(3) Return timeout and recovery state.
(4) Keep dma-buf APIs off until the full capability probe passes.

Exit receipt: The forked `libane` ABI passes its tests and an M1 consumer smoke.

### Enable ANE graph regions

(1) Partition exact supported regions before `mx.compile` fusion.
(2) Run ANE through a worker that owns its device and resident buffers.
(3) Preserve recurrent state and stream order.
(4) Exchange host stage buffers with the caller.
(5) Include copy and process IPC cost in each crossover result.
(6) Keep a region only when total median latency improves by 10 percent.

Exit receipt: Hybrid traces, numerical checks, state reuse, latency crossover, containment, and recovery all pass.

## Offline assistant: paired text and voice

The [assistant design](plans/2026-09-27-offline-assistant-design.md) defines a new application gate, not a completed backend milestone.

1. Prepare pinned Everyday and Quality model pairs through one setup flow.
2. Join Laya decisions and LLM conversation in one local chat UI.
3. Add arbitrary-input recognition and interruptible local speech synthesis.
4. Qualify installation, paired memory use, disconnected restart, and accessible interaction on each supported configuration.

Text can ship before voice, with voice explicitly unavailable. The complete assistant gate remains open until both pass.
Exit receipts must include real requests through both models, arbitrary recorded speech, audible TTS,
zero CPU tensor dispatches, cancellation/recovery, and a cold restart with outbound traffic denied.
Catalog model flags alone do not satisfy this gate.
