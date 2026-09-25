# Chip portability audit: the Omarchy backend

2026-09-11. Baseline: origin/main `21710dfb`. Scope: `overlay/mlx/backend/omarchy`
— `primitives.cpp`, `compute.cpp`/`compute.h`, `device_info.cpp`, `device.cpp`
(discovery), `fused_chain.cpp`, `encoder.cpp` (dispatch), and `shaders/`
including `qmm_vec.comp`, `qmm_coopmat.comp`, `matmul_vec.comp`,
`matmul_coopmat_bf16.comp`. Every line number below was read from that commit.

The question this audit answers: the fleet has exactly one Apple-silicon Linux
box, an M1 (G13). What in this code assumes that chip, and what breaks — loudly
or silently — on an M2 today, or on M5/M6 when they arrive? No kernel
arithmetic was changed by this audit; the six canonical Q4 digests and the
per-driver BF16 pins in `parity-id-policy.md` are untouched contracts.

Machine-readable form: [`chip-capability-axes.json`](chip-capability-axes.json)
— the axis names are shared verbatim with the capability-simulation harness.

## 1. Findings, ranked by blast radius

**SILENT** = on a device where the assumption is false, the backend keeps
running and produces different numbers than the pins, without refusing. That is
the dangerous class; every member is marked.

| # | Location | Assumed property | How it is detected today | When it is false | Class |
|---|---|---|---|---|---|
| F-01 | policy in `parity-id-policy.md` | Nothing. A capability bit selects a route, it does not pin arithmetic | Digests are pinned per driver build, which is the policy, not a workaround | Two coopmat-capable Mesa builds on the same M1 produced different Q4-longctx and BF16-long digests. Expected: different driver builds compile the kernel differently, so the generated ids differ. Not a defect, and not something to gate on | Expected |
| F-02 | `primitives.cpp:619-630` gate; `shaders/matmul_vec.comp:105-151` | Subgroup size exactly 32 + `SHUFFLE_RELATIVE`, 4-byte-aligned offsets/gaps/strides, `k%128==0`, `n%4==0` mean the shuffle-down reduce produces the pinned dense BF16 decode stream | Capability query (subgroup properties) + host-side alignment arithmetic | A driver build that reports the same bits but lowers shuffles differently routes decode projections through a different accumulation order than the pinned one; no refusal fires | **SILENT** |
| F-03 | `compute.h:18` (`kComputeThreadsPerGroup = 256`); all `.comp` `local_size_x`; only exception `primitives.cpp:10299-10306` | `maxComputeWorkGroupInvocations >= 256` and `maxComputeWorkGroupSize[0] >= 256` | Never checked; the SDPA-native gate checks its own 1024 requirement, nothing checks the rest | Vulkan's spec floor is 128 invocations / 128×128×64 size. A conformant device reporting 128 makes every dispatch undefined — likely silent corruption, loud only under validation layers | **SILENT** (UB class) |
| F-04 | `primitives.cpp:6687-6689, 6999-7000, 6466-6468`; `shaders/qmm_vec.comp:70-79, 21-39` | `subgroupAdd` pairs bit-identically to the five-round shared tree ("measured that pairing bit for bit" — on Honeykrisp only) | Capability gate `subgroup_size == 32 && ARITHMETIC`; the bit-equality claim is a measurement, not a gate | On another driver build the two flavors may pair differently; the capability gate then silently picks the flavor whose bits differ from the pins. The shader's own comment says the dispatch gate is load-bearing because shaders cannot query subgroup size | **SILENT** |
| F-05 | `primitives.cpp:6577-6594, 6721-6754`; `shaders/qmm_coopmat.comp` | Coopmat presence + subgroup 32 + 4 KiB shared + even x/output offsets ⇒ Q4 prefill runs the coopmat tile; stock driver (no extension) runs `QmmTileRbF16` with a different accumulation order | Capability-keyed gate, queried every dispatch; alignment contract break refuses loudly (`6734-6738`) | Stock-vs-fork route split is real and already produces different pinned digests per driver (held to the policy). A third coopmat build gets its own pins, per the policy. If the alignment contract ever broke silently, the code's own comment says refuse-by-name beats rerouting — that guard is present | **SILENT** (route split; pinned per driver) |
| F-06 | `primitives.cpp:556-584`; `shaders/matmul_coopmat_bf16.comp:6-18, 103-105` | The staged BF16 coopmat tile stores identical bits to the eager-order kernel for the same tile values (ascending-k 8-wide MMA chain), needs 2-byte-aligned operands, `m>=32`, 4 KiB shared | Capability + alignment + shared-memory gate, queried every dispatch | On a driver build whose coopmat lowering reassociates, stored bits deviate with no refusal ; the equality claim has never been proven on a second driver build | **SILENT** (claimed result-preserving by construction, unproven elsewhere) |
| F-07 | `device_info.cpp:51-54` | `architecture` = "honeykrisp" iff `driver_name` contains the substring `Honeykrisp` | Name-keyed string match | A driver rename or a differently spelled `driver_name` mislabels the architecture as "vulkan". Cosmetic (info surface only, no arithmetic), but name-keyed where the driver id it should key on (`kMesaHoneykrispDriverId`, already collected) sits unused one field away | Loud (label only) |
| F-08 | `device.cpp:555-605` | Acceptance = Vulkan 1.3 + (driverID 26 ∨ vendor 0x106b ∨ name contains "honeykrisp") | Identity-keyed by design; M1 is accepted via driverID (it reports vendor `0x10005`, not 0x106b) | An M2 under Honeykrisp is accepted the same way — correct. But acceptance performs no capability floor check (see F-03): a device can be accepted and then dispatch UB. Inverse risk: a future Apple GPU on a non-Honeykrisp driver is refused loudly (safe) | Loud (acceptance side) / **SILENT** (via F-03) |
| F-09 | `compiled.cpp:37-58` | BF16 compiled tapes corrupt nondeterministically on Honeykrisp | Keyed on dtype, not on device: refused on every device, loudly | Over-refuses on drivers where the defect does not reproduce (llvmpipe is already proven correct). Safe direction — no wrong numbers — but the refusal message hard-codes a driver-empirical defect with no root cause pinned | Loud |
| F-10 | `fused_chain.cpp:291-297` | Fused chain runs only with `shader_float16`/`shader_int16` + `storage_buffer_16bit_access`; otherwise silently unfused | Capability gate; fallback is the unfused op stream | Fused-vs-unfused bit equality is a design contract, not a capability: if a driver broke it, fusion would silently change bits. Covered only by suite legs | **SILENT** (result-preserving by design, suite-verified) |
| F-11 | `primitives.cpp:7724-7728, 8256-8257`; `device.cpp:248-262` | Float scatter Sum uses hardware atomicAdd where `shaderBufferFloat32AtomicAdd`, else the FCAS compare-exchange twin | Capability-keyed, extension AND feature bit | The twins are exact-arithmetic equivalents by construction; route flip does not move bits. Cited as the model the other gates should converge to | Loud-safe (no risk) |
| F-12 | `compute.h:19`, `encoder.cpp:423-425` | `maxComputeWorkGroupCount[0] >= 65535` (spec floor); dispatch clamps and chunks past it | Spec guarantee + the fixed 2026-09-08 bool-tail defect (`known-defects.md`, regression `test_select_ops.cpp`) | The clamp-chunk pattern was once a silent wrong-results bug past 16,776,960 elements; the pattern is now regression-covered. Any new kernel using the clamp must chunk, not truncate | **SILENT** (historical; fixed) |
| F-13 | Shader corpus: `cast.comp:110-114`, `compare_bool.comp:14-20, 87-90`, `compare.comp:63-66`, `argreduce_suffix.comp:111-114`, `block_mask.comp:20-23`, `conv.comp:6-9`, `copy_general.comp:14-16`, `complex_elementwise.comp:90-94, 207-210`, `logical_or` atomicOr form (`primitives.cpp:1080-1084, 1150-1154`) | Honeykrisp miscompiles: divergent per-lane word loads, shift-then-mask with data-dependent shift, wide if/else selector store coalescing, `FLT_MAX/FLT_MAX = 0`, GLSL `/` one ulp off, inexact NIR constant reassociation | Empirical receipts (`known-defects.md`), upstream battery; workarounds are unconditional code shapes, keyed on nothing | Portable by construction today (they run everywhere and are harmless when the bug is absent). A future driver build can carry NEW miscompiles with no signal — silent wrong values until the battery runs on that build | **SILENT** (new-defect exposure) |

Deliberately not findings (checked, spec-floor-safe or already capability-keyed):
push constants are exactly `sizeof(ComputeParams)` = 128 bytes = the Vulkan
spec minimum (`compute.cpp:1320`); `fft_c2c.comp`'s 16 KiB shared and the
linalg/sort/reduce shared budgets all sit at or under the 16 KiB spec floor;
the 25-slot binding budget is `min(budget, device limits)` with a named throw
past the limit (`device.cpp:815-821`, `encoder.cpp:413-421`); the allocator's
non-coherent fallback does explicit flush/invalidate (`allocator.h:20-24`);
`kTrigArgumentLimit` is a device-independent contract, measured as such
(`primitives.cpp:4132-4136`); timeline-semaphore absence refuses the device
(`device.cpp:442-448`).

## 2. The smallest capability-keyed change for each finding

- **F-01**: no code change. The pin policy already keys digests to the driver
  build. The one useful piece of plumbing: key the pin lookup on driver build
  identity, since `driver_version` and `pipeline_cache_uuid` are already
  collected in `CapabilityReport` and unused for it.
- **F-02**: keep the capability gate; before enabling the shuffle route, run a
  one-time known-answer probe (small fixed K on both routes, compare bits).
  Cheaper minimum: make "route + driverInfo git hash" a required field of
  every digest receipt, so a silent route flip can never pass review unnoticed.
- **F-03**: in `ComputeRuntime`'s constructor, throw at device creation when
  `max_compute_work_group_invocations < kComputeThreadsPerGroup` or
  `max_compute_work_group_size[0] < 256` (and `< 1024` names the SDPA route
  as retired, which its gate already handles). Turns a UB class into a loud
  refusal, one place, once per device.
- **F-04**: add the same known-answer pattern to the subgroup gate: one
  32-element `subgroupAdd` vs shared-tree comparison at device creation; or
  document in `parity-id-policy.md` that the bit-equality claim is per driver
  build and must be re-proven (the qualification contract below requires it).
- **F-05**: nothing. The gate is already capability-keyed and the alignment
  break already refuses loudly.
- **F-06**: add a required qualification leg that proves tile-vs-coopmat bit
  equality on the target driver build (one fixed matmul, both routes via
  `MLX_OMARCHY_NO_COOPMAT`).
- **F-07**: key on `caps.driver_id == kMesaHoneykrispDriverId`, fall back to
  the name match. Two lines.
- **F-08**: after acceptance, apply F-03's floor check so "supported" implies
  "dispatchable".
- **F-09**: leave the refusal; add the observed driver build to the message.
  Retire it for a new chip only with a probe receipt proving the corruption
  does not reproduce there.
- **F-10**: no code change; the fused-vs-unfused digest-equality leg is a
  required qualification suite item (below).
- **F-13**: keep the workarounds unconditional; the defense is procedural —
  run the upstream battery on every new driver build before claiming support
  (qualification contract).
- **Gap (new)**: there is no env switch to force the subgroup route OFF
  (`MLX_OMARCHY_NO_COOPMAT`, `_QMM_TILE`, `_QMM_TILE_RB`, `_QMM_VEC_Q4_WORD`
  exist; subgroup has no equivalent). Cross-route bit verification on a capable
  device is impossible today. One `MLX_OMARCHY_NO_SUBGROUP` env in the three
  `subgroup_ready` expressions closes it.

## 3. Capability axes

Names agreed with the capability-simulation harness and used verbatim in
[`chip-capability-axes.json`](chip-capability-axes.json) (which also carries
the full gate map with line references). M1 reference values are measured or
gate-proven from the receipts; M2 values are unknown until the chip runs
Omarchy; M3 has no GPU driver (software rendering only); M4/M5/M6 are unknown
end to end. **Every axis for future silicon must come from runtime discovery
(`collect_capabilities`), never from a chip-name table.** The current code
already derives all of these at runtime; the audit's complaint (F-03) is that
one class of limits is never checked at all.

| Axis | M1 G13 (reference row) | M2 Max | M5/M6 |
|---|---|---|---|
| `driver_variant` | `honeykrisp_installed`, Mesa 26.3.0-devel git-6f6afc8968, driverID 26 | expected honeykrisp family (Asahi G14 acceleration is upstream); build unmeasured | unknown |
| `cooperative_matrix_fp32_8x8x8` | true (installed build) | unknown | unknown |
| `subgroup_size` | 32 (measured) | unknown; presumed 32 because it is an AGX-wide warp — presumption is not a measurement | unknown |
| `subgroup_ops_mask` | BASIC, ARITHMETIC, SHUFFLE, SHUFFLE_RELATIVE proven present (consumed by enabled routes); full mask not dumped | unknown | unknown |
| `shared_memory_limit_bytes` | ≥ 21504 (gate-proven: SDPA-native enabled); exact value not dumped in-repo | unknown | unknown |
| `workgroup_limits` | ≥ 1024 invocations / ≥ 1024 size_x (gate-proven); exact values not dumped | unknown | unknown |
| `atomic_float_add` | false (extension absent) | unknown | unknown |
| `memory_model` | `uma_coherent` | `uma_coherent` (Apple UMA is an architecture fact) | `uma_coherent` for Apple UMA parts; anything discrete would be new |
| `digest_policy` | `per_driver_build` — all six pins are functions of the driver build, not the chip | all six unknown until measured | unknown |

## 4. Per-chip qualification contract

A new chip is "supported" only when every item here has a receipt under
`receipts/`. Evidence keys on the **driver build** (`vulkaninfo` driverInfo
git hash + driverUUID + `mlx-omarchy-info` full dump), never on the chip name.

**Per-chip status (2026-09-20):**

- **M1 (T8103, m1-test-host) GPU/ANE**: previously qualified, historical only —
  m1-test-host fresh Arch boot reported (user-observed at login); Omarchy
  provisioning and benchmark recertification pending.
- **M1 Max (T6001, t6001-test-host) GPU**: measured, live.
- **M1 Max (T6001, t6001-test-host) ANE**: qualified (104/104 Parakeet E2E on v0.6.0;
  recerted 4773.8 ms `.ane` on v0.7.1 in [`ane-linux-experiments/receipts/2026-09-19-parakeet-e2e-v071-t6001-test-host.md`](https://github.com/joshuaswarren/ane-linux-experiments/blob/main/receipts/2026-09-19-parakeet-e2e-v071-t6001-test-host.md)).
- **M2 Max (T6021, t6021-test-host) GPU**: verified third-silicon on Honeykrisp /
  Vulkan 1.4.354 ([`ane-linux-experiments/receipts/2026-09-18-t6021-test-host-third-vulkan-device.md`](https://github.com/joshuaswarren/ane-linux-experiments/blob/main/receipts/2026-09-18-t6021-test-host-third-vulkan-device.md));
  t6021-test-host Linux reads kernel 7.1.13-3-1-ARCH stable, ANE_UNBOUND, no
  `/dev/accel/accel0`.
- **M2 Max (T6021, t6021-test-host) ANE**: **NOT live-inference-qualified** on the
  Linux driver (macOS ANE numbers cited in this tree are macOS CoreML /
  `aned` measurements on t6021-test-host / studio-host, not Linux-side execution).
  Apple GPU and Apple ANE are separate lanes.
- **M3 / M4**: not qualified by this project; no inference support claim.

The contract below governs what a *future* chip run must demonstrate to
be added to this list with status `qualified` rather than the historical
caveats above.

1. **Identity bundle**: `mlx-omarchy-info` dump (all `CapabilityReport`
   fields), `vulkaninfo` summary, driver package + git hash, ICD in use.
   This fills every axis in section 3 with measured values — no gate-proven
   floors accepted for a new chip.
2. **Capability floors** (from the measured dump): `subgroup_size == 32`;
   `subgroup_ops_mask` ⊇ BASIC, ARITHMETIC, SHUFFLE, SHUFFLE_RELATIVE;
   `shared_memory_limit_bytes >= 21504`; `workgroup_limits` ≥ 1024/1024;
   `shader_float16`, `shader_int16`, `storage_buffer_16bit_access`; UMA
   coherent. Anything less names which routes retire (SDPA-native, coopmat
   prefill, subgroup GEMV, fused chains) and the chip ships with those routes
   off, not claimed.
3. **Arithmetic-contract probes**: the `known-defects.md` probe set — fdiv
   rounding (80/25, 10/25, quantize boundary), `cos` at `fl(π/2)`,
   `FLT_MAX/FLT_MAX`, `log` host-ulp, shift-then-mask byte extraction, wide
   selector stores, divergent word loads. Record which reproduce; every hit
   is either a fixed-in-fork prerequisite or a new ledger entry.
4. **Suites**: the full upstream battery (the 24-binary / 407-case /
   828k-assertion pattern) green on the chip's driver build, plus
   fused-vs-unfused digest equality (F-10) and tile-vs-coopmat bit equality
   (F-06) as explicit legs. A software-driver run is not a substitute — the
   ledger exists because dev-box green missed real-M1 defects twice.
5. **Digest policy**: re-measure all six canonical legs (Q4 short / long /
   1K-ctx, BF16 short / 262 / 1K-ctx) on the chip's driver build and pin them
   **per driver build**. The native-matching legs (Q4 short, Q4 1K-ctx, BF16
   1K-ctx) must equal the native macOS digests; the others may take native's
   values or their own new per-driver-build values, per
   `parity-id-policy.md`.
6. **Benchmark legs**: the decode legs (subgroup vs tree GEMV, GB/s) and the
   prefill legs (coopmat vs register-blocked tile, TFLOP/s) run on their own
   capability-determined route; cross-route legs use the route-forcing env
   switches — which requires the `MLX_OMARCHY_NO_SUBGROUP` gap from section 2
   to be closed first.
7. **Native baseline**: capture native macOS MLX intermediates for the same
   model and legs (the `receipts/native-baseline-*` pattern) — required to
   apply rule 1 of the digest policy. For BF16 legs the oracle is the float64
   round-to-nearest reference, not the macOS capture (macOS deviates on the
   bias-cancellation elements; `known-defects.md`).
8. **Compiled-tape re-probe**: the BF16 tape refusal (F-09) must be re-probed
   on the new driver build; it retires only with a receipt proving the
   corruption does not reproduce.

Until items 1–5 exist for a chip, the honest claim is: "runs on Apple silicon
whose driver build passes the M1 contract; unverified on this chip." The
runtime is already built to make that statement checkable — every gate it
needs is a capability query away; the real hole is that the workgroup floor is
assumed rather than checked (F-03).
