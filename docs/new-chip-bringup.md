# New-chip bring-up procedure

How to bring a genuinely new GPU chip (M2 today, M5/M6 when they arrive)
into mlx-omarchy's qualification without guessing, and how to decide
fast whether the shipped kernels are usable on it or a new path is
required. The capability vocabulary here is shared with
`docs/chip-capability-axes.json`, `docs/chip-portability.md`, and the
simulation harness (`overlay/mlx/backend/omarchy/capability_sim.h`).

## 0. The capability axes

Every decision below keys off measured axes, never chip names:

| Axis | Meaning | Consumers in the dispatch |
|---|---|---|
| `driver_variant` | which driver BUILD serves the chip (honeykrisp_installed / honeykrisp_fork / stock_mesa_honeykrisp / llvmpipe / lavapipe / other) | selects the arithmetic contract and the digest pin set; pins are per driver build, not per chip |
| `cooperative_matrix_fp32_8x8x8` | extension present AND feature on AND an 8x8x8 all-fp32 subgroup-scope non-saturating shape enumerated; the 8x8x8 fp16 x fp16 -> fp32 shape is recorded beside it (`cooperative_matrix_f16_8`) and simulated by this axis; the 8x8x8 bf16 x bf16 -> fp32 shape (`cooperative_matrix_bf16_8`) also needs `VK_KHR_shader_bfloat16` with `shaderBFloat16CooperativeMatrix` and a build with `GL_EXT_bfloat16`, and simulation can only clear it | MatmulF32Coopmat, MatmulBF16Coopmat, MatmulDirectF32, QmmPrefillCoopmatF16, SdpaDecodeNative; MatmulDirectF16 when the fp16 shape is also enumerated; MatmulDirectBF16 (replacing MatmulBF16Coopmat) when the bf16 shape is |
| `subgroup_size` | physical + reported subgroup width; gates test `== 32` exactly | every subgroup-reduced kernel |
| `subgroup_ops_mask` | VkSubgroupFeatureFlags bitmask; ARITHMETIC, SHUFFLE, SHUFFLE_RELATIVE, BASIC are the consumed bits | qmm vec subgroup (ARITHMETIC), dense BF16 decode GEMV (SHUFFLE_RELATIVE), SdpaDecodeNative (BASIC\|ARITHMETIC\|SHUFFLE) |
| `shared_memory_limit_bytes` | maxComputeSharedMemorySize | 4096 for the coopmat kernels' staging, 21504 for SdpaDecodeNative, 16384 spec floor (fft) |
| `workgroup_limits` | {invocations, size_x, size_y, size_z} | SdpaDecodeNative needs 1024/1024; the local_size 256 kernels are validated only against the spec floor (audit finding F-03) |
| `atomic_float_add` | VK_EXT_shader_atomic_float + shaderBufferFloat32AtomicAdd | Scatter axis Sum selects FAdd vs FCAS |
| `memory_model` | uma_coherent / host_visible_incoherent / discrete | allocator mapped-pointer path vs explicit flush/invalidate |
| `digest_policy` | always `per_driver_build` | generated-id pins travel with the driver build (see `receipts/2026-09-10-prefill-qmm-isa/driver-portability-defect.json`: one M1, two coopmat-capable Mesa builds, two different Q4-longctx digests) |

Two structural facts drive the whole procedure:

1. **Subgroup-32 kernels need physical 32-lane subgroups.**
   `qmm_vec.comp` (subgroupAdd per 32-lane slot) and `matmul_vec.comp`
   (32-lane shuffle-down reduction) are only defined when the physical
   subgroup is the same 32 lanes the logical layout assumes. A device
   with 8- or 64-lane physical subgroups would silently sum the wrong
   lanes. The dispatch refuses by name when a simulated or real
   capability set claims 32 lanes the hardware cannot back.
2. **Digests are per driver build.** Passing the capability gates on a
   new driver build says nothing about generated-token identity. The
   six canonical digests must be re-measured on that build before
   anything is called a pin.

## 1. Measure (order matters)

Run `mlx-omarchy-info --json` on the new chip's driver and record, in
this order:

1. **Identity**: device name, vendor/device id, `driver_variant`
   (driverID + driverVersion + Mesa build id). This is the digest pin
   key — everything downstream is recorded against it.
2. **Capability report**: all axes above, from one discovery pass.
   `cooperative_matrix_fp32_8x8x8` is only trusted when the extension
   is listed AND the feature bit is on AND the 8x8x8 fp32 shape is
   enumerated (the backend's discovery already enforces this
   conjunction).
3. **Arithmetic contract probes** (the driver miscompile class Honeykrisp
   shipped with): fdiv ulp, cos near pi/2, FLT_MAX division flushing,
   log ulp, shift-then-mask, wide dynamic selectors. `docs/known-defects.md`
   lists the exact failures seen on stock builds. A new driver build
   that fails any probe gets a known-defects entry before anything
   else runs on it.
4. **Physical subgroup sanity**: the reported `subgroup_size` must equal
   the width a trivial `subgroupAdd` reduction actually sums (one
   256-wide workgroup, distinct lanes). A driver reporting 32 while
   physically executing 8-lane subgroups would flip every gate into
   undefined territory; the probe catches it in seconds.

## 2. Simulate before you touch hardware (optional but cheap)

If the chip is not in hand yet, encode its expected axis row as a
simulation profile (`capability_sim.cpp` registry) and run the
capability-simulation tests on any software host:

```bash
MLX_OMARCHY_ALLOW_NON_APPLE=1 MLX_OMARCHY_CAPS_SIM=<profile> \
  ./build/tests/omarchy/omarchy_capability_sim_tests <profile>
```

This proves which dispatch routes the chip would take and which tests
will refuse. It never validates the chip itself; simulated runs are
stamped and refuse benchmark/digest evidence.

## 3. Test ladder (run in this order; stop at the first red)

| Leg | What it proves | Where |
|---|---|---|
| 1. `omarchy_runtime_tests` | device discovery, timeline semaphores, buffer round trip | `overlay/tests/omarchy` |
| 2. `omarchy_capability_sim_tests` WITHOUT `MLX_OMARCHY_CAPS_SIM` (plus `MLX_OMARCHY_ALLOW_NON_APPLE=1` on non-Apple hosts) | the real capability set runs the full per-op battery through whatever routes the gates choose, with every result inside the documented accuracy envelopes | `overlay/tests/omarchy` |
| 3. The same five profile legs WITH `MLX_OMARCHY_CAPS_SIM` set to each profile | every alternative route the chip's row can select still produces correct results (or a named refusal where the chip cannot back a kernel) | same binary |
| 4. `omarchy_matmul_family_tests` decode-projection cases | the dense BF16 GEMV and qmm GEMV accuracy contracts on the chip's real routes | `overlay/tests/omarchy` |
| 5. Six canonical digests, re-pinned | greedy token identity on THIS driver build; legs that match native macOS are held to native | `scripts/bench_decode.py` via `scripts/bench_matrix.py` |
| 6. Decode/prefill benchmark legs per route | performance of the routes the chip actually takes | `scripts/sdpa-ab.sh`, bench matrix |

Leg 2 is the fast go/no-go: it exercises every shipped kernel family on
the real device in minutes on software, seconds on hardware. A green
leg 2 means the existing kernels are usable; the remaining legs are
qualification, not feasibility.

## 4. Reading a refusal

The simulation harness refuses by name when a capability set selects a
kernel the runtime hardware cannot back:

```
[omarchy] capability simulation '<profile>' (driver_variant <variant>)
dispatches <kernel>, which requires axis <axis> on the runtime device,
but the hardware ('<name>', <driver>) does not provide it. Refusing: ...
```

The same refusal fires without simulation whenever a device's real
report cannot back a selected kernel. A refusal is a routing fact, not
a bug: it names the exact axis that needs either different hardware, a
driver fix, or a new kernel path.

## 5. Existing kernels or new path?

Decide from the leg-2/3 results:

- **All green (incl. digests)**: the chip rides the existing kernels.
  Add its axis row to `docs/chip-capability-axes.json`, and add a
  simulation profile if its row differs from the five shipped ones.
- **Correct through fallbacks, refusals only on 32-lane-subgroup or
  coopmat kernels**: the chip works with the composed/tile fallbacks
  today. A new kernel path is needed only if the fallback's
  performance misses the decode/prefill targets (leg 6). The refusal
  names the axis; e.g. a 64-lane-subgroup chip needs a
  subgroup-width-agnostic reduction (shared-memory tree or
  workgroup-uniform layout) before `QmmVecSubgroup*` /
  `MatmulVecBF16` / `SdpaDecodeNative` can dispatch.
- **Any wrong numbers inside a claimed route**: a driver arithmetic
  defect (class: `docs/known-defects.md`). File it, gate the kernel
  behind the failing probe, keep the fallback. Never widen a gate to
  make a defect disappear.
- **CreateDevice / pipeline failures under a profile only**: the
  profile claims an axis the driver cannot execute; the refusal is the
  correct outcome and the chip needs that axis before the kernel can
  run.

## 6. Adding a simulation profile for the new chip

One entry in the `kProfiles[]` registry in
`overlay/mlx/backend/omarchy/capability_sim.cpp`, expressed as a delta
over the discovered hardware report (sentinels inherit):

- `name`: the chip's working row name (e.g. `m2-asahi-default`).
- `represents_driver_variant`: the audit's `driver_variant` enum value.
- Pin exactly the axes the chip is expected to define: `subgroup_size`,
  `subgroup_ops_mask` (OR/AND masks), `cooperative_matrix`,
  `shared_memory_limit_bytes`, `workgroup_invocations`, `workgroup_size_x`,
  `atomic_float_add`. Leave `memory_model` and everything else at
  hardware values.
- Register the profile in the test CMake `foreach` and in
  `test_capability_simulation.cpp` so the leg-3 battery covers it.

Then the one-M1 rule keeps working: every future chip's expected
behavior is reproducible on llvmpipe before the hardware exists, and
the real silicon only has to confirm.
