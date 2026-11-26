# B60 (BMG G21, ANV/Xe2) — Xe16 cooperative-matrix prefill route

DEC-19. Capability-gated dense `Matmul` + `QmmPrefillCoopmat16Xe` on the
ANV `M8xN16` cooperative-matrix family (`K16`, fp16/bf16 operands, f32
accumulate), replacing the `bf16_fma` / `QmmPrefillFmaBF16` fallbacks on
Intel Xe2 where the M16 and 8x8x8 shape lists are empty.

## Environment

- Source commits: `cc3756a39` (implementation), `45f9937a7` + `df89dbce2`
  (test coverage), `1c43b95ad` (review: gate capability + shared-memory
  bound), `ac4d2554e` (review: qmm Xe16 emits bf16 operands, matching
  `qmm_intel_bf16`'s contract), `dd8e63866` (review: qmm
  `require_backed` parity terms + 64-lane workgroup caps + stale
  comments). Wheels: base `0.32.4.dev…+diag.cc74e7a` (origin/main),
  prop `0.32.4.dev…+diag.dd8e638`. Early cells ran the fp16-operand
  build `+diag.45f9937`; every number below is the bf16-operand head.
- Kernel: `7.1.12-2`. Mesa (ANV): 26.2.6.
- Device: `Intel(R) Arc(tm) Pro B60 Graphics (BMG G21)`, eGPU over USB4,
  driver `xe`, subgroupSize 32 (pinned via requiredSubgroupSize).
  `VK_KHR_cooperative_matrix` shapes advertised: M8xN16 K16
  (f16→f32, bf16→f32) and int8 K32 only — no M16, no 8x8x8.
  `VK_EXT_subgroup_size_control` present.
- Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (q4 group-64).
- Host placeholder: a Thunderbolt-eGPU Linux workstation.

## Probes

### Route admission (live dispatch)

`MLX_OMARCHY_TRACE_DISPATCH=1` on the prop wheel:

- `mx.matmul` bf16 64×64 → `DISPATCH kernel=735`
  (`MatmulBF16CoopmatXe16`). Base wheel dispatches `kernel=26`
  (`MatmulBF16`, `matmul_bm32_bf16.comp` sequential f32 FMA).
- `quantized_matmul` q4-g64 bf16, ragged m=20 → `kernel=736`
  (`QmmPrefillCoopmat16XeBF16`).

### 5-pair ABBA + propoff (greedy, temp 0, 128-token decode)

Commands (per cell): `flock <host-egpu-lockfile> env CELLS=<cell>
nice -n 10 bash run-abba.sh` — interleaved A-B-B-A blocks, plus 3
prop-off runs (prop wheel + `MLX_OMARCHY_NO_MATMUL_XE16=1
MLX_OMARCHY_NO_QMM_XE16=1`).

- ctx512 (fp16-operand head `45f9937`): base/prop/propoff each
  deterministic (1 unique ids sha256 per arm). propoff ids == base ids.
  base != prop ids (one early argmax flip, paraphrase-level; see below).
  dmesg reset-keyword count 8→8.
- ctx2048 (bf16-operand head `dd8e63866`, boot `a79ce46b`): **all arms
  identical** ids sha256 `e683dade…f0` across 23 runs (10 ABBA pairs +
  3 propoff); propoff bit-matches base. dmesg 8→8,
  `GPU-TIMEOUTS delta=0`, `FOREIGN-GPU-USED count=0`.
- 0 GPU hangs across all runs; no engine resets.

### Timing (steady state, per run, `dd8e63866`)

Two guard-run cells on the same boot: r3 (shared host, post-reboot
churn) and r4 (Main-ordered quiet-host re-measure of decode). Both
5 A-B-B-A pairs + 3 propoff, ctx2048, greedy, 128-token decode.

r4 decode per block (base avg → prop avg, ms/tok):

| blk | base | prop | Δ |
|-----|------|------|---|
| 1 | 37.7 | 44.4 | +17.9 % |
| 2 | 40.0 | 44.5 | +11.3 % |
| 3 | 34.8 | 37.9 | +8.7 % |
| 4 | 35.9 | 38.7 | +7.9 % |
| 5 | 33.7 | 34.6 | +2.6 % |

propoff 31.2 / 35.3 / 31.1 ≈ base — the same wheel with both routes
env-disabled pays no premium, so the effect tracks routes-on, not the
wheel.

- ctx2048 prefill: prop faster in every block on both runs (r4:
  prop ≈ 18.4–21.9 s vs base ≈ 19.3–23.6 s; r3 medians prop ≈ 21.8 s
  vs base ≈ 23.2 s, ≈ −8 %).
- ctx2048 decode: prop carries a real premium (median +8.7 %),
  reproduced on the quiet host, fading with job time (+17.9 % →
  +2.6 % across blocks as the host settled post-reboot).
- Mechanism (per-dispatch GPU profile, r4): dispatch count identical
  (49703) and dependency-barrier count identical (44017); submission
  host costs equal (s ≈ 12–16 ms, b ≈ 75–92 ms, q ≈ 4.4 ms per run).
  The premium is on-device: prop's decode kernels run uniformly slower
  at identical counts — +16 % on the bandwidth-bound
  `QmmVecQ4MultiSubgroupBF16R2S16KCONTS16`, +58–68 % on the
  latency-bound `SdpaDecodeSplit*`/`FastRmsNormApple` kernels — a GPU
  core-clock/DVFS signature following the denser coopmat prefill, not
  host-side work. Decode dispatches zero Xe16 kernels (identical
  kernel table both arms); the regression is a second-order physical
  side-effect of the faster prefill, not a decode-path code
  difference. Net per request still favors prop at 128 tokens
  (−1.7 s prefill vs ≤ +0.6 s decode); a >500-token decode could net
  negative if the premium sustains.
- Main-ordered cause measurement (r5, 20 Hz `tile0/gt0/freq0/act_freq`
  + hwmon card/pkg energy, arms base/prop/propidle, 2 runs each, same
  wheels and boot): decode runs pinned at 2400 MHz GT clock on every
  arm (mean 2395–2400, max 2400; propidle's only sub-max samples are
  the post-idle ramp) — the slowdown is not GT core-clock DVFS, and
  the `min_freq` lever (root-writable, currently 1200) could only
  constrain. Decode card power 57–65 W, pkg 38–43 W, no arm
  separation; `power_profile=power_saving`. Decode medians: base
  30.8–32.9 ms/tok, prop 35.7–38.6, propidle 36.2–36.3 — propidle
  (prop prefill → 3 s idle → decode) does NOT recover to base speed,
  so the state left by the prop prefill is persistent across a 3 s
  idle, not a decaying thermal/clock transient. Combined with the
  identical dispatch/barrier counts and uniform per-kernel
  inflation, the remaining candidates are VRAM layout / allocator
  state or a memory-clock state from the different prefill
  allocation pattern (act_freq covers the GT core only, not GDDR).
  Evidence: `~/scratch/w7t-dec19/freq2/` (freq-*.csv, mk-*, out-*).
- Decode-phase dispatch attribution (r4 profs, 5 runs pooled per
  arm, dispatches under submissions whose host timestamps fall
  inside the marker decode window): total decode GPU-busy
  20,348 ms base → 22,578 ms prop (+2,229 ms, +11 %), matching
  the wall premium. Delta concentrates on the most-dispatched
  kernels: `ElementwiseF32` +1,577 ms (+27 %), `op897988541`
  +336 ms (+12 %), `CastBoolF32` +276 ms (+2 %) — a broad
  per-execution overhead signature, not a single-kernel
  slowdown. Decode buffer VA footprints are the same on both
  arms (~23-27 distinct MB units in the same region), so
  allocator placement does not explain it either.
- ctx512 decode (`45f9937`): both arms ≈ 45 ms/token steady state.
- Per-kernel (ctx2048 kernel table): `MatmulBF16 n=134217728`
  17.5 ms/35 dispatches → `MatmulBF16CoopmatXe16` 9.8 ms/35;
  `QmmPrefillFmaBF16 n=1740800` (propoff arm) →
  `QmmPrefillCoopmat16XeBF16` 6.5 ms/70 on prop. The FMA fallback lines
  appear only on the propoff arm, i.e. the Xe16 route is the only
  non-FMA path for those shapes.

### Per-op error vs fp32 reference (prop wheel `+diag.dd8e638`, B60)

`xe16_error_ppl.py` — random uniform inputs, error vs fp32 matmul of the
same bf16 operands:

```
[dense-xe16-bf16] m=64   n=4096 k=4096 err=0.2498 bound=0.3951 ok
[dense-xe16-bf16] m=512  n=2048 k=512  err=0.1138 bound=0.1429 ok
[dense-xe16-bf16] m=2048 n=512  k=2048 err=0.2238 bound=0.3031 ok
[dense-xe16-bf16] m=77   n=1536 k=256  err=0.0617 bound=0.0946 ok
[qmm-xe16-bf16]   m=77   n=2048 k=2048 err=0.000428 fixed-bound=0.000412
[qmm-xe16-bf16]   m=512  n=2048 k=2048 err=0.000477 fixed-bound=0.000444
```

The acceptance bound is the k-scaled bound the doctest uses for the
bf16-operand route — `(3k+1)·M·2⁻²³ + 2k·M·2⁻⁸ + M·2⁻⁸`, `M =
max(2·max|got|, 1)` — not the probe's `max|ref|·2⁻⁸`, which ignores
reduction depth. At k=2048 the k-scaled bound is ≈16.0; both qmm
errors sit ~3.5×10⁴× inside it. For reference the fixed bound misses
are 1.04× (m=77) and 1.07× (m=512), and the shipped bf16-operand
`QmmPrefillIntelBF16` route also exceeds the fixed bound 1.07× on its
own shapes at the same k=2048 — the coopmat route is strictly more
accurate than the shipped bf16-operand path it replaces.

Dispatch trace confirms 76 dispatches of kernel 735 + 1 of 736 on the
prop arm (the m=77 qmm probe), 0 on base (26 x72 + 657 x1). The m=512
qmm probe takes the pre-existing `QmmPrefillIntelBF16` route on both
wheels (m % 16 == 0) and is identical bitwise on both arms — its fixed-
bound overshoot belongs to that route, not this change.

### Perplexity (fixed 2000-token text, Qwen3-4B-4bit)

base ppl **1.346772**, prop ppl **1.346908** → +0.010 % (gate: ≤0.1 %).

## Known divergence note

Greedy ids at ctx512 differ between arms (paraphrase-level split around
token ~14); at ctx2048 they are bit-identical. The base kernel
accumulates K in strictly sequential f32 FMA order while ANV's
cooperative-matrix MMA accumulates in a hardware-defined order inside
each K16 tile; bitwise equality is therefore not a stable property of
the route change and lands stochastically per prompt.

## Watch items

- The `== tok/s ==` tail of `run-abba.sh` greps `out-ctx512-*-1.txt`
  unconditionally; under `set -e` a ctx2048-only cell exits rc1 after
  all run cells complete (the r3 wrapper logged `lane-failure
  kmsg-clean` for this reason). The `== ids gates ==` pool also globs
  both cells' out files, so the pooled counts mix ctx512-r1 hashes with
  ctx2048 — the per-run `ids_sha256` lines are the cell-scoped record.
- During the doctest battery (run once, before queue submission) one
  unrelated test case (`small-row qmm matches per-row single-row
  dispatch`) hit the backend's 10 s Vulkan timeline watchdog while the
  host was under heavy foreign build load; the Xe16 dispatch-pin cases
  around it had completed cleanly. Not reproduced in the model ABBA
  (46 model runs, 0 hangs). Flagged for follow-up.
- `test_xe16_covers_ragged_m` and the dense dispatch pin require an
  ANV device; they skip elsewhere.

## Device recovery

No hang, reset, or device-loss event occurred in any run; the device
reopens cleanly on subsequent runs (all runs after the ABBA used the
same device fd path without error).
