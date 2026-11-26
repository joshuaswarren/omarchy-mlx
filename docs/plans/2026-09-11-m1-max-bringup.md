# M1 Max bring-up plan — 2026-09-11

Bring the owner's 16-inch MacBook Pro (ssh alias `t6001-test-host`, Apple M1 Max,
32 GPU cores, 10 CPU cores) into mlx-omarchy qualification. This plan applies
the procedure in [`docs/new-chip-bringup.md`](../new-chip-bringup.md) to this
specific die and states the acceptance bar from
[`docs/chip-portability.md`](../chip-portability.md) section 4.

**Unverified premise, flagged 2026-09-11.** The first version of this plan
assumed the machine had a standing duty that made conversion impossible, and
built a dual-boot requirement plus an operational procedure on that
assumption. The check meant to establish it failed and the assumption was
carried anyway. Whatever this machine does outside this project is the owner's
to state; the plan no longer describes it. What stands: the hardware facts in
section 1 (measured with `sysctl` and `system_profiler`) and the Asahi
per-generation support statements (sourced from upstream). Dual-boot is kept
as the safe default for getting qualification evidence without committing the
machine, not as a constraint proven by this document.

Everything here is documentation and planning. The base M1 (`m1-test-host-linux`,
`/tmp/m1-gpu.lock`) is untouched by this work.

## 1. The machine and why it is a different die, not a bigger M1

| Fact | Base M1 (reference) | t6001-test-host (this plan) |
|---|---|---|
| SoC | t8103, `apple,j293`, AGX "G13" generation | t6001-class M1 Max, AGX "G14" generation |
| GPU cores | 8 | 32 |
| CPU | 4P+4E | 8P+2E (10 cores) |
| DRAM part | 68.25 GB/s class (measured roof 58.5 GB/s, `receipts/2026-09-11-q4-memory-roof`) | measured GPU blit roof 306.26 GB/s median, 64 GB LPDDR5 (`receipts/2026-09-11-m1max-macos-capture`) |
| System level cache | 8 MB | 48 MB public spec — verify |
| OS today | Omarchy/Linux (Asahi) | macOS |
| Driver | `mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1` | none yet — the question of section 5 |

Three consequences drive the whole plan:

1. **G14 is a different codegen.** Mesa's AGX backend compiles
   per-generation ISA. The same driver package that has only ever run on
   G13 will compile shaders differently for this die. That makes every
   digest pin and every miscompile probe a fresh measurement, not a copy.
2. **The performance envelope moves ~6x on paper.** Every GB/s number and
   every occupancy verdict measured against 8 cores sits on a 68 GB/s
   latency-bound part. Section 6 names the verdicts that do not transfer.
3. **Conversion is an open question, not a settled constraint.** Dual-boot is
   a safe way to get qualification evidence without committing the machine,
   but it is one option rather than the only shape available. Section 8 lists
   what would have to be true for full-time Omarchy, and defers the
   machine's other duties to the owner.

## 2. What dual-booting means for this machine

Asahi's installer shrinks the APFS container and installs Linux next to the
existing macOS install. Nothing about macOS is replaced: booting the machine
lands in the m1n1/U-Boot boot picker, and the selected OS runs exclusively
until the next reboot. The operating contract is:

- **macOS is the resting state.** The machine boots macOS by default and
  returns to it at the end of every window.
- **Omarchy is booted only inside an announced window** for bring-up and
  qualification work. A window has a start, a single top-level GPU session,
  receipts, and an explicit end that includes rebooting back to macOS.
- **While Omarchy is up, macOS is not running**, so anything that machine
  hosts is offline for the duration. Coordinating that is the owner's call
  and is deliberately not specified here.

### Window protocol (every Omarchy session)

1. Announce the window start and expected duration, and settle any hosting
   the machine does with its owner before booting.
2. Do the GPU work; receipts under `receipts/<date>-m1max-*/`.
3. End of window: shut down Omarchy, boot macOS, confirm the machine is back
   in its normal state, and record the receipt.

## 3. Before the first boot — what must be true

macOS side (do these before touching the installer):

- Full backup current (Time Machine or equivalent), recorded as done with a
  timestamp. The installer resizes the APFS container; that is the one
  destructive step in this plan, so its precondition is a backup that is
  known-good, not presumed-good.
- Disk layout and container sizes are now captured, not pending:
  [`receipts/2026-09-11-m1max-macos-capture`](../../receipts/2026-09-11-m1max-macos-capture/README.md)
  holds the pre-repartition partition table, container GUID, boot setting and
  snapshot list to compare against afterwards.
- **Free space is the blocker, measured 2026-09-11.** The 4 TB container is
  98.6% in use with 57.3 GB not allocated and 53 GiB available, FileVault is
  on for both the System and Data volumes, and 25 hourly Time Machine local
  snapshots hold purgeable space that `df` and the installer count
  differently. Asahi needs room for the Linux side plus its own staging, so
  the install cannot start from this state: free space first, verify the
  container's not-allocated figure moved, and only then run the installer.
  The resize runs against an encrypted, nearly-full container, which is the
  worst case for it - the backup precondition above is doing real work here,
  not ceremony.
- The native macOS baseline on this die is already banked, twice:
  [`receipts/2026-09-10-native-macos-metal-baseline`](../../receipts/2026-09-10-native-macos-metal-baseline/README.md)
  and
  [`receipts/2026-09-11-m1max-native-baseline`](../../receipts/2026-09-11-m1max-native-baseline/README.md),
  with all six canonical digests reproducing across both sessions. That is the
  rule-1 oracle and the same-die denominator; no other machine can produce it,
  and it no longer needs doing before the install.
- Settle with the owner what this machine currently hosts and how that is
  covered while it is booted into Omarchy. That coordination is outside this
  repository and is not described here.

Installer side:

- Use the current Omarchy-on-Asahi installer path
  ([`docs/install-omarchy.md`](../install-omarchy.md) is the in-repo
  reference). During first-boot setup: enable sshd and install the owner's
  key (windows are driven headless from the dev box), pick a hostname
  distinct from `m1-test-host`, record the sudo password with the owner's secrets.
- After install, set the default boot entry back to macOS and verify an
  unattended cold boot lands in macOS. The machine must fail safe to macOS.

## 4. After the first boot — what to verify

- Boots normally, whatever it hosts is serving again, receipt recorded. The
  dual-boot is only proven once a full macOS → Omarchy → macOS round trip has
  left the machine's normal duties intact.

Omarchy side, in this order:

1. `uname -m` is `aarch64`; AGX probe present in `dmesg`; the die reports
   as M1 Max / t6001-class — record what it reports, do not assume a name.
2. `vulkaninfo` enumerates exactly one ICD, Vulkan 1.3, and reports driver
   name, driverID, driverVersion, and driverInfo git hash. Record all four;
   the driverInfo hash is the digest pin key.
3. `pacman -Q mesa-honeykrisp-omarchy` — expect `26.3.0.devel.hk6f6afc8-1`;
   if the package differs or is absent, that is section 5's fork question
   and nothing else runs until it is resolved.
4. Build/install the aarch64 wheel, then run `mlx-omarchy-info --json` and
   save the full dump. That dump is the identity bundle of qualification
   contract item 1 and fills every capability axis in one pass.

## 5. The fork driver question

`mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8` has **only ever run on the
base M1** (t8103, G13). This die is G14. The question is empirical, not
architectural: a single Mesa build normally serves both generations (chip
selection is runtime), but the shader compiler emits different ISA per
generation, so "same package" does not mean "same behavior".

How to answer it, in order:

1. **Install the package as-is.** `vulkaninfo` driverInfo must read
   `git-6f6afc8968`. The device must pass the backend's acceptance check
   (`device.cpp:555-605`: Vulkan 1.3, and driverID 26 ∨ vendor `0x106b` ∨
   driver name contains `honeykrisp`). If the package installs and the
   device is accepted, proceed to the measurement order; the runtime gates
   will refuse by name anything the die cannot back.
2. **If the package is missing, fails to install, or the device is
   refused:** build the fork from the same pinned commit (`6f6afc8968`) for
   this machine and install that. Record the resulting build identity —
   a locally rebuilt binary is **its own driver build** for pin purposes
   even at the same git hash: pin to what `mlx-omarchy-info` reports
   (driverID + driverVersion + Mesa build id + `pipeline_cache_uuid`), per
   the `driver_variant` axis definition.
3. **If cooperative-matrix is absent on this die** (the extension is not
   listed, the feature bit is off, or no 8x8x8 all-fp32 shape is
   enumerated — the backend's discovery enforces the conjunction):
   - The gated routes retire by name: `QmmPrefillCoopmatF16` (falls back to
     the register-blocked tile `QmmTileRbF16`, a different accumulation
     order), `MatmulF32Coopmat`, `MatmulBF16Coopmat`, and `SdpaDecodeNative`
     (which also needs its own 1024/1024 workgroup + 21504-byte shared
     budget). Record which routes retired in the axis row.
   - Add the measured row to `docs/chip-capability-axes.json` and a
     simulation profile for this chip (delta over the discovered hardware
     report) in `capability_sim.cpp`, registered in the test CMake and
     `test_capability_simulation.cpp`, per
     [`docs/new-chip-bringup.md`](../new-chip-bringup.md) section 6.
   - This is a routing fact, not a failure: the fallbacks are shipped and
     suite-proven. A new kernel path is decided **only** by test-ladder
     leg 6 (benchmark targets), never by the absence itself.
4. **If cooperative-matrix is present but behaves differently** (bits
   deviate from the tile path where equality was claimed):
   - Route-split digest differences across driver builds are the expected
     class (F-01/F-05) — the die gets its own pins, per
     [`docs/parity-id-policy.md`](../parity-id-policy.md).
   - The F-06 tile-vs-coopmat bit-equality claim has **never been proven on
     a second driver build**; this machine is the first chance. Run that
     leg explicitly (both routes via `MLX_OMARCHY_NO_COOPMAT`).
   - Wrong numbers **inside a claimed route** are a driver arithmetic
     defect: known-defects entry, gate the kernel behind the failing probe,
     keep the fallback, never widen a gate.

## 5a. Measurement order (from `docs/new-chip-bringup.md` §1, applied)

Run `mlx-omarchy-info --json` on this die and record, in this order:

1. **Identity** — device name, vendor/device id, `driver_variant`
   (driverID + driverVersion + Mesa build id + `pipeline_cache_uuid`).
   Everything downstream is recorded against this key.
2. **Capability report** — all axes from the one discovery pass. The axes
   to characterise first, because the dispatch keys off them:

   | Axis | What to measure here | Why it is unknown on this die |
   |---|---|---|
   | `subgroup_size` | physical + reported width | AGX warps are presumed 32 family-wide; presumption is not a measurement (`chip-portability.md` §3) |
   | `subgroup_ops_mask` | dump the **full** mask | the reference row only ever proved the four consumed bits present |
   | `cooperative_matrix_fp32_8x8x8` | extension AND feature AND enumerated 8x8x8 fp32 shape | the fork build has never enumerated a G14 device; section 5 |
   | `shared_memory_limit_bytes` | exact value | reference row is a gate-proven ≥21504 floor, never dumped; a 32-core die does not imply a larger budget |
   | `workgroup_limits` | all four fields, plus `maxComputeWorkGroupCount[0]` vs the 65535 clamp | reference row is a gate-proven ≥1024/1024 floor; a bigger die does not imply bigger limits, and the F-03 audit finding is that nothing checks the 256 floor today |
   | `atomic_float_add` | extension + feature bit | false on the reference build; unmeasured here |
   | `memory_model` | confirm host-visible+coherent inside device-local | expected `uma_coherent` (architecture fact); still confirm the coherent mapped-pointer path is what discovery selects |

   The base M1 reference row for comparison: `subgroup_size` 32 (measured);
   ops mask BASIC|ARITHMETIC|SHUFFLE|SHUFFLE_RELATIVE proven present, full
   mask never dumped; shared ≥21504 and workgroup ≥1024/1024 both
   gate-proven floors, exact values never dumped; `atomic_float_add` false;
   `uma_coherent`; digests per driver build. **No gate-proven floors are
   accepted for a new chip** — this die dumps exact values for everything
   the reference row never did.

3. **Arithmetic-contract probes** — the `docs/known-defects.md` probe set:
   fdiv rounding, `cos` at `fl(π/2)`, `FLT_MAX/FLT_MAX` flushing, `log`
   host-ulp, shift-then-mask, wide dynamic selectors, divergent per-lane
   word loads. The driver binary is the same package that passed on G13,
   but G14 is a different codegen, so every probe runs fresh. Any failure
   gets a known-defects entry **before anything else runs on this die**.
4. **Physical subgroup sanity** — a trivial `subgroupAdd` over one
   256-wide workgroup must sum exactly the reported width. Seconds of work;
   catches a driver reporting 32 lanes it cannot physically execute, which
   would flip every gate into undefined territory.

Optional pre-leg, free if the machine is not in hand yet: encode the
expected axis row as a simulation profile and run the capability-simulation
battery on a software host. Simulated runs are stamped and refuse
benchmark/digest evidence — they prove routing, not the chip.

Then the test ladder, in order, stopping at the first red: runtime tests →
capability battery on the real device (leg 2, the go/no-go — the device is
Apple silicon, so `MLX_OMARCHY_ALLOW_NON_APPLE` stays unset) → per-profile
legs → matmul-family accuracy → **six canonical digests re-pinned on this
machine** → benchmark legs per the routes this die actually takes.

## 6. What does NOT transfer from the base M1

This is the part someone will get wrong. Two classes:

### 6a. Digest pins

- **Digest pins are per driver build and must be derived fresh on this
  machine before any parity claim.** The six canonical Q4 digests
  (installed build: `7fd25a869ff21678` / `4cc08910089477fd` /
  `7da83f06ec9f001d` Q4 short/long/1K, `f26175202f3dabe9` /
  `ad964232ee67fecd` / `ff502900d2a179a5` BF16 short/262/1K) are a
  **base-M1-plus-driver artifact**: G13 codegen compiled by that specific
  Mesa build. They are not a chip-independent contract — the repo's own
  proof is one M1, two builds, two different `q4_longctx` and `bf16_long`
  values (`receipts/2026-09-10-prefill-qmm-isa/driver-portability-defect.json`).
  G14 codegen plausibly moves all six. Measure them on this die, pin them
  per this driver build, and apply the parity rule to the new pins:
  native-matching legs (Q4 short, Q4 1K-context, BF16 1K-context) must
  equal the native macOS digests; the others may take native's values or
  their own fresh per-driver values, nothing else. The native oracle for
  this die is this machine's own macOS side (section 3).

### 6b. Performance verdicts measured against 8 GPU cores

All three were measured on a latency-bound 68 GB/s part. At 32 cores and
~6x bandwidth, the binding constraint moves; each verdict is a hypothesis
to re-measure, not a law to cite.

1. **The flat occupancy sweep** — workgroup count ×1→×16 flat at a ~30 us
   dispatch floor; conclusion "the kernel is latency-bound, not
   mapping-bound" (`receipts/2026-09-10-q4-gemv-native-map`). Why 32 cores
   could change it: the sweep proved extra concurrent workgroups buy
   nothing on a part where 8 cores saturate below the memory roof. With
   4x the cores and ~6x the bandwidth, production grids (112–1216
   workgroups) oversubscribe the die only 3.5–38x instead of 14–150x, the
   dispatch floor is a proportionally larger share of each kernel, and the
   balance between latency-hiding and bandwidth can flip the conclusion in
   either direction. Re-sweep before treating mapping changes as worthless.
2. **The rows-per-lane register-liveness loss** — R=2 costs +29.4/+29.6%
   us/layer, R=4 costs +46.2/+52.8%; 35→63→87 allocated GPRs; achieved
   bandwidth falls 41.0→31.6→27–28 GB/s
   (`receipts/2026-09-11-q4-decode-rows-per-lane`). Why 32 cores could
   change it: the loss mechanism was occupancy — more live registers per
   lane → fewer resident warps per core → less device-level memory-level
   parallelism — on a part where latency hiding was the bottleneck. On a
   400 GB/s part the latency-vs-bandwidth economics are different; per-lane
   MLP may pay where it lost, or the occupancy tax may bite harder. The
   measured negative must be re-established, not inherited.
3. **The workgroup-128 result** — wg128 was bimodal in the decision
   environment (+0.34% leg A / −4.43% leg B on the 202 MB widedep set;
   consistently −3.8/−4.0% only on the hot 8.4 MB iso4 set) and failed the
   ≥3%-repeating bar (`receipts/2026-09-11-q4-memory-roof`). Why 32 cores
   could change it: wg128 is a scheduling-granularity change (2x workgroups
   of 128 threads), and the bimodality was measured on a scheduler where
   every core was heavily oversubscribed. At 32 cores the workgroup-per-
   core ratio changes regime entirely; the bimodal result may collapse to
   a consistent answer in either direction.

What **does** transfer from today's measurement discipline, unchanged:

- Wall-anchored timing only: host `CLOCK_MONOTONIC` around whole submits.
  Per-dispatch device timestamps are a forbidden instrument
  (`receipts/2026-09-10-dispatch-floor`; the decode-gap receipt's
  decomposition of the bogus 98 us/layer is the cautionary tale).
- The copy-roof methodology for any bandwidth claim — but the roof values
  themselves (58.5 GB/s copy, 41.0 GB/s kernel, 60 GB/s layout roof) are
  base-M1 numbers; this die gets its own roof first.
- Peak probes must clamp to `maxComputeGroupCountX` (the 65535 invalid-
  dispatch lesson from the memory-roof receipt).
- The unconditional miscompile workarounds transfer as code, but their
  *necessity* is re-proven by the section 5a probe set on G14 codegen.
- The single-window flock discipline and hub announcements — this machine
  gets its own window discipline (section 2); the base M1's
  `/tmp/m1-gpu.lock` is not involved.

## 7. Acceptance bar: what lets the project claim "M1 Max supported"

Exactly the per-chip qualification contract
(`docs/chip-portability.md` §4), with receipts under `receipts/` keyed on
the **driver build** (driverInfo git hash + driverUUID + full
`mlx-omarchy-info` dump), never on the chip name:

1. **Identity bundle**: full `mlx-omarchy-info` dump, `vulkaninfo` summary,
   driver package + git hash, ICD in use.
2. **Capability floors, measured**: `subgroup_size == 32`; ops mask ⊇
   BASIC, ARITHMETIC, SHUFFLE, SHUFFLE_RELATIVE; shared ≥21504; workgroup
   ≥1024/1024; `shader_float16`, `shader_int16`,
   `storage_buffer_16bit_access`; UMA coherent. Anything less names which
   routes retire and the die ships with those routes off, not claimed.
3. **Arithmetic-contract probes**: the full known-defects probe set
   recorded; every hit is a fixed-in-fork prerequisite or a new ledger
   entry.
4. **Suites**: the full upstream battery green on this driver build, plus
   the fused-vs-unfused digest-equality leg (F-10) and the tile-vs-coopmat
   bit-equality leg (F-06) as explicit items. No software-driver run
   substitutes for this.
5. **Digest policy**: all six canonical legs re-measured on this machine
   and pinned per this driver build; the native-matching legs equal the
   native macOS digests captured on this die.
6. **Benchmark legs** per route, cross-route legs via the route-forcing
   env switches — which requires the `MLX_OMARCHY_NO_SUBGROUP` gap
   (chip-portability §2) closed first if cross-route legs are wanted.
7. **Native baseline** on this die (section 3's macOS-side capture; for
   BF16 legs the oracle is the float64 round-to-nearest reference).
8. **Compiled-tape re-probe**: the F-09 BF16-tape refusal re-probed on
   this build; it retires only with a receipt proving the corruption does
   not reproduce.

Until items 1–5 exist, the honest claim is the contract's own sentence:
"runs on Apple silicon whose driver build passes the M1 contract; unverified
on this chip." Documentation outcome on success: a new axis row in
`docs/chip-capability-axes.json`, a simulation profile for this die,
updated compatibility notes, and receipts under `receipts/<date>-m1max-*/`.

## 8. Separately: what full-time Omarchy on this machine would require

This is a different bar from section 7 and must not be reached by drifting:

- **Whatever the machine hosts today is the owner's decision to move, keep,
  or drop.** This plan takes no position on it and records no details about
  it; that reasoning belongs in the owner's private notes, not in a public
  repository.
- **Serving from Omarchy would be a new qualification**, not a byproduct of
  this plan: 24/7 stability of an alpha driver under serving load,
  serving-throughput legs, and a recovery story. Scope it separately.
- **Machine-level daily-driver checks**, each verified in a window before
  any default-boot switch: GPU-accelerated desktop stability, lid
  sleep/resume, Wi-Fi, battery life and thermals under sustained load,
  audio, and whatever Thunderbolt/external-display use the owner needs.
  Webcam and Touch ID have no Linux driver on this generation — that is a
  known, permanent gap to accept explicitly, not a bug to fix.
- **A trial period**: run the qualification windows back-to-back for long
  enough that suspend/resume and thermal behavior are observed, with macOS
  still the default boot entry, before the owner decides anything
  permanent.
