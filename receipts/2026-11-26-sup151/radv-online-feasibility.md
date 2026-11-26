# SUP-151: RADV online-attention feasibility — 833fd2937's gates vs DEC-45 for the B60/5060-Ti executors

Scope: commit `833fd2937` ("omarchy: default 1-dispatch online-softmax bf16 decode SDPA on G13C
(k<=2304)") against the DEC-45 / #962 `requiredSubgroupSize = 32` context. Question: can the B60 and
RTX 5060-Ti executors of SUP-119/120 run the online-softmax decode route, and what must land first?

## The gates at 833fd2937 (decode SDPA route selection)

1. Shape gate: HD=128, batch 1, `q_len == 1` — decode only, one query head per 1024-thread workgroup
   (32 subgroups x 32 lanes), coalesced 256-byte uvec2 K+V loads, zero scratch/atomics/pass-2.
2. Route selection (`primitives.cpp`): `MLX_OMARCHY_SDPA_ONLINE` "0" disables, non-empty forces;
   unset defers to `sdpa_online_default(device_name, k_len)` — G13C-only, max_keys 2304, every other
   part returns 0 until measured — and only when no explicit split-K override is set.
3. Device-ready gate (all required): `decode_caps.subgroup_size == 32u`,
   `max_compute_work_group_invocations >= 1024`, `max_compute_work_group_size[0] >= 1024`,
   `max_compute_shared_memory_size >= 16640`, and subgroup operations BASIC | ARITHMETIC | SHUFFLE.
4. Kernel is fixed: `SdpaDecodeOnlineSg32BF16Hd128` (the SG=16 variant was removed by this commit).

## Vendor applicability (1024-thread WG, 16640 B shared memory, subgroup ops)

**RTX 5060 Ti (Blackwell, warp-32 subgroups) — applicable now, env-forced.** NVIDIA reports default
subgroup size 32, so the `subgroup_size == 32u` gate reads true with no pin; invocations 1024, work
group size 1024, shared memory well above 16640 B, and BASIC/ARITHMETIC/SHUFFLE are all supported.
The only blocker is route selection: `sdpa_online_default` is G13C-only, so run it with
`MLX_OMARCHY_SDPA_ONLINE=32` (or add a device-name entry after a measured result). The 2304-key
default is a G13C SLC number — NVIDIA L2 geometry differs, so measure the k_len ceiling per card
instead of importing 2304.

**Intel B60 (Battlemage, ANV) — applicable after the requiredSubgroupSize pin reaches this
pipeline.** Hardware gates pass (invocations 1024, work group 1024, SLM well above 16640 B, full
subgroup feature set), but Intel Xe2's default subgroup size is typically 8/16 — simd32 lanes are
reachable only through a `requiredSubgroupSize = 32` pin (VK_EXT_subgroup_size_control), the same
mechanism DEC-45/#962 uses for the QMV-family shaders. The pin plumbing exists
(`create_pipeline(..., required_subgroup_size_)` with the fail-closed
`subgroup_size_32_admitted_` check); the online route needs that pin to cover its pipeline, and the
`decode_caps.subgroup_size` read must then reflect the pinned size. Without the pin the route is
fail-closed, which is correct.

**AMD RADV RX 9070 (DEC-45's own card) — same pin prerequisite.** RDNA 3/4 RADV defaults to
subgroup 64; DEC-45/#962's `requiredSubgroupSize = 32` is the fix, and every non-size gate passes
on RADV (invocations 1024, work group 1024, 64 KiB shared memory, full subgroup ops). Verify the
pin is device-wide for omarchy pipelines (not only the QMV-family kernels) before reading the
online gate as open.

## What the executors should do (SUP-119 B60, SUP-120 5060 Ti)

1. 5060 Ti: run the 5-pair ABBA (2B decode, ids identical) with `MLX_OMARCHY_SDPA_ONLINE=32`
   today; record the k_len ceiling where it turns negative instead of assuming 2304.
2. B60: first confirm `VK_EXT_subgroup_size_control` + requiredSubgroupSize=32 on the online
   pipeline (one-line capability probe), then the same ABBA. Expect fail-closed until then.
3. Both: HD=128/batch 1/q_len 1 only; the HD=256 variant (477ca738b6) is a separate kernel with its
   own gates. Ids-identical greedy output stays the pass gate; tok/s is the comparison metric.
4. The env force works on any vendor with the caps above — the device-name default table in
   `sdpa_online_max_keys` is the only G13C-specific piece.
