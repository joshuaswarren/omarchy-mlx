# VkDeviceCreateInfo feature-chain audit (PR-LILYVFT)

2026-11-22. aurora-lily measured sparse residency, robust
access, and buffer-device-address capture/replay as individually costly
on Honeykrisp-M5 and stripped them. This audit applies the same test to
the omarchy `VkDeviceCreateInfo` built in
`overlay/mlx/backend/omarchy/device.cpp` (`Device::Device`).

## Method

1. Read every `enabled*` struct the `dci.pNext` chain carries.
2. Searched shaders (`overlay/mlx/backend/omarchy/shaders/`), the
   allocator, the encoder, and cooperative-matrix paths for the
   capability each feature enables.
3. Named each feature's consumer; a feature with no consumer is a strip
   candidate.

## Enabled features and their consumers

| Feature | Set when | Consumer |
|---|---|---|
| `shaderInt16` | hardware reports it | int elementwise / reduction shaders |
| `shaderInt64` | hardware reports it | int64 buffer paths |
| `storageBuffer16BitAccess` | hardware reports it | bf16/f16 storage buffers |
| `timelineSemaphore` | always | submission completion tracking (encoder) |
| `shaderFloat16` | hardware reports it | fp16 compute shaders |
| `subgroupSizeControl` | required subgroup size | coopmat / fixed-subgroup kernels |
| `shaderBufferFloat32AtomicAdd` (EXT) | hardware reports it | scatter-add kernels |
| `cooperativeMatrix` (KHR) | hw f32 coopmat | `matmul_coopmat_*` kernels |
| `shaderBFloat16Type` + `shaderBFloat16CooperativeMatrix` | `OMARCHY_BF16_CMAT` build + hw | bf16 coopmat kernels |

## lily's costly features: status

| Feature | Requested? | Finding |
|---|---|---|
| sparse residency (all `sparseResidency*`) | **No** — absent from the chain | nothing to strip; no sparse-alloc path exists |
| `robustBufferAccess` | **No** | nothing to strip; bounds checks live in shaders |
| `bufferDeviceAddress` + `CaptureReplay`/`MultiDevice` | **No** | allocator binds memory explicitly; no BDA consumers |
| `robustImageAccess` (1.3) | **No** | not requested |
| `scalarBlockLayout`, `descriptorIndexing`, `runtimeDescriptorArray` | **No** | fixed descriptor sets; no indexing consumers |
| `dynamicRendering`, `maintenance4` (1.3) | **No** | not requested |

`dci.pEnabledFeatures` stays null; all enabled features ride the pNext
chain. Device extensions are limited to `VK_EXT_shader_atomic_float`,
`VK_KHR_cooperative_matrix`, `VK_KHR_shader_bfloat16`,
`VK_EXT_global_priority`, the lily pairing `VK_EXT_memory_priority` +
`VK_EXT_pageable_device_local_memory` (#668: memory-priority hints and
pageable local memory, enabled only when the driver advertises both),
and `VK_EXT_calibrated_timestamps` (gated on `MLX_OMARCHY_TRACE_DISPATCH`)
— each set only when the hardware reports the matching capability.

## Diff

No feature was requested, so the strip is a no-op on the chain itself:
the audit's deliverable is `omarchy_feature_chain_tests`
(`overlay/tests/omarchy/test_feature_chain.cpp`), which intercepts the
real `vkCreateDevice` call and fails if any feature outside the audited
whitelist — or any of lily's costly features — is ever enabled.

## Profiler calibration (the lily port)

`gpu_profiler.h` now emits one `{"k":"cal"}` record after `meta`:
`floor_ns` is the median host cost of 32 empty command buffers each
submitted alone and waited (after 2 untimed warmups) — the same
lone-submit job cost lily subtracts (~150 us on M5 Honeykrisp; measured
here per driver). The buffers stay strictly command-free: a
timestamp-only stream submits empty on compute-only chips (#526), so no
GPU-tick epoch is anchored here. `floor_ns` is for subtracting from
host-side spans (k=s/k=q/k=j); the k=d GPU spans already bracket the
dispatch inside the command buffer.
