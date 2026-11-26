# Prefill gap on jwm1 (T8103/G13G): ledger reproduction, per-op profile, roofline, staged-A qmm lever

Date: 2026-10-03. Chip: Apple M1 (G13G B1), 8 GPU cores, 16 GB. Kernel
7.1.12-2-7-ARCH, boot `16b7c4a8…` for every timed run below — the release
baseline started ~15 min after that boot (the previous boot ended in
w71's H237 reboot at ~17:1xZ), settled by the recorded idle gates
(load1 0.01-0.46, PSI <= 0.05 per cell) rather than by uptime; numbers
match the prior settled-boot ledger within 0.3%. Host names and addresses
are redacted per the public-repo rule;
the private lab notebook holds unredacted transcripts and SHA256SUMS
(`entries/PrefillGap/20261003T152000Z-jwm1-prefill-gap-profile-lever.md`).
Provenance printed beside every measurement
(`scripts/mlx_provenance.py`, `version_match=true` for both wheels below);
idle gate load1 < 0.5 and PSI cpu avg10 <= 0.05 recorded per cell;
`flock /tmp/m1-gpu.lock` held per run only; mlx_lm 0.31.3 + vendored patch
series (incl. last-logits) in every venv; shipped uclamp default applied
via the bench wrapper in every arm.

Wheels:

| arm | wheel | stamp |
|---|---|---|
| release baseline | v0.7.21 public install into a throwaway HOME | `0.32.4.dev202610030937+e8a02e9b` |
| diag / lever | built on jwm1 from `agent/prefillgap-qmm-staged` (= origin/main `22cdc6da5` + lever `fcfc39e74`), glslc, pinned whole-encoder bundle (manifest `08769793…`, program `13c74423…` = the v0.7.21 pins) | `0.32.4.dev202610031742+diag.fcfc39e` |

## 1. What the ledger ratios refer to (decoded)

The v0.7.21 ledger ("losses reported as losses", H234) compares the **qwen38
protocol**: model = Qwen3.8-2B-class 4-bit affine group-64 (snapshot
`0867d98b`, `model.safetensors` sha256 `b0d5de68…`), prompt corpus
`qwen38-2b-prompts.jsonl` (sha256 `9299a3b2fc136a4c`), greedy temp 0, 32
new tokens, mlx_lm 0.31.3 + vendored patches **including last-logits on
both OSes**. Cells: pure-prefill leg (`--prefill-tokens N`, no generation;
`pure_prefill_tok_rate`) and median first-token over 30 short-prompt
greedy records in a back-to-back loop. H236 A1 decomposes the back-to-back
TTFT: ~19 ms of it is the previous request's in-flight lookahead decode
(clean rested TTFT ~109 ms); macOS runs the identical loop, so the ledger
TTFT comparison is protocol-consistent. The 0.918x / 1.043x ledger entries
are the pf512 cell against jwm1's own macOS reference (window5, same
last-logits patch both sides): prefill **457.6** (pf1024 **458.7**) tok/s,
TTFT **0.125 s**, decode 49.36/49.26/49.33 tok/s.

## 2. Release-wheel reproduction (v0.7.21, settled boot)

h234 cell args, n=3 mirrored ON/OFF blocks (ON = shipped uclamp default,
OFF = `MLX_OMARCHY_UCLAMP_MIN=0`), block medians:

| cell | arm | prefill tok/s | TTFT s | vs macOS |
|---|---|---|---|---|
| pf512 | ON | **419.4** (419.4-419.9) | **0.1307** (0.1305-0.1310) | **0.916x** / TTFT **1.046x** |
| pf512 | OFF | 418.1 (414.1-420.2) | 0.1309 (0.1307-0.1366) | — |
| pf1024 | ON | **420.9** (420.9-421.3) | 0.1305 | **0.918x** |
| pf1024 | OFF | 421.2 (420.7-421.4) | 0.1309 | — |

2B digest `bc519c03c4ef5fd1` in **all 24 cells** — the v0.7.21 pins
reproduce exactly. The reproduction matches the ledger within 0.2-0.3%
(prefill 0.916-0.918x, TTFT 1.046x). Context cells (release wheel, ON,
n=2-3): 4B prefill 157.2 (pf512) / 150.1 (pf1024) tok/s, TTFT 0.267 s,
digest `eac9fe326212ffa2`; 9B prefill 90.1 / 89.7 tok/s, TTFT 0.538 s,
digest `1523306ac5ddc9a2` (the 9B snapshot had been cleaned from the HF
cache mid-day and was restored byte-identical from the jw16 copy before
its cells ran). No macOS numbers exist for 4B/9B on jwm1 — not applicable.
The uclamp ON/OFF TTFT spread in this window (0.1307 vs 0.1309) is inside
noise, unlike H234's -7.8%: recorded as an unreplicated lever delta on a
warm 10-h boot, not a refutation.

w71's H238 (v0.7.22 verification, 9-20 min post-reboot + makepkg slot):
prefill 413.9 (0.905x), TTFT 0.1307 (1.046x) — consistent; this receipt's
settled-boot numbers are the cleaner read.

## 3. Per-op profile and roofline

Dispatch-count table is reliable; per-dispatch GPU-ms are NOT on this
driver (below). G13G, one 512-token prefill, 2B, diag wheel: **912
dispatches, 62 submissions, 1 join**; per-kernel dispatch counts: qmm
prefill family 186 (150 `QmmPrefillCoopmatBF16X32FullN` + 36 M16 twin),
`CastBF16F32` 222, `CopyGeneralBF16` 120, RMS/gated-norm family 103,
`GatedDeltaPrefillCoopmatBatchBF16` 18, attention coopmat matmuls + softmax
12+6, LM head via `QmmVecQ4WordSubgroupBF16` 1. Same shape mix at 1024
tokens. 4B (36 full-attn layers) and 9B (24 GDN + 8 full) tables have the
same families with proportionally more qmm/cast/attention dispatches
(1238 dispatches for 9B@512).

**Instrument finding (negative):** for a single-join prefill run the
`MLX_OMARCHY_GPU_PROFILE` GPU timestamps are unusable on this Mesa line —
the analyzer reports GPU busy 4.7% of span and qmm at 242 TFLOP/s
aggregate (physically impossible; the 1347 tok/s wall on jw16 proves the
GPU is near-saturated). H110's usable numbers used a 4-join pattern.
Per-op GPU-ms therefore come from prior censuses (H5: qmm = 76.7% of GPU
busy in the 512-token leg; H110: qmm 61%, casts 5.2%, GDN scan 12.6% at
M=11) and the wall-clock route evidence below, not from this window's
timestamps.

**Roofline reading (arithmetic + measured anchors):** at M=512/1024 the
quantized-matmul prefill is deep in the compute regime — FLOPs/byte =
3.56·M against a compute/BW crossover near M≈40 on this chip (68.25 GB/s
nominal, 65.4 pattern peak, 49.8 GB/s qmm layer mix measured by
q4-bw-bench; fp32 FFMA peak 2.62 TFLOP/s). H5 measured the qmm kernel at
~0.94 TFLOP/s in-model (36% of the fp32 peak) with GPU busy ~96% of wall:
the kernel is issue/occupancy-limited, not bandwidth-limited, which is why
every tiling knob (H6/H7/H13) and the small-M levers (H104-H111) failed to
move it. The remaining prefill gap vs macOS (0.916x) is inside the qmm
coopmat kernel's issue quality on G13G — a honeykrisp/compiler axis, not a
dispatch-glue axis: casts are ~5% of busy at small M and shrink relatively
at prefill M, and the A/B below shows removing them is a 22-25% LOSS.

## 4. Lever: staged-A bf16 qmm prefill — MEASURED NEGATIVE

`MLX_OMARCHY_QMM_BF16_STAGED=1` (commit `fcfc39e74`, branch
`agent/prefillgap-qmm-staged`, pushed) routes bf16 q4/g64 prefill to the
existing staged-A twins and skips the `CastBF16F32` pass. Same diag wheel
both arms, 5 alternating pairs per cell, ctl first:

| model | cell | ctl tok/s | staged tok/s | staged/ctl |
|---|---|---|---|---|
| 2B | pf512 | 413.9 | 316.9 | **0.766** |
| 2B | pf1024 | 414.5 | 319.8 | **0.772** |
| 4B | pf512 | 157.2 | 118.6 | **0.754** |
| 4B | pf1024 | 150.1 | 115.5 | **0.770** |
| 9B | pf512 | 89.1 | 69.0 | **0.774** |
| 9B | pf1024 | 88.8 | 69.3 | **0.780** |

Digests are **identical across arms in all 6 cells** (2B `a4ebce784981475a`,
4B `e21ec584a51e18b4`, 9B `7fc97dc29f2c1412` — the lever is exact, as
designed; the doctest pins staged-vs-X32 bit-identity). The lever loses
22-25% prefill everywhere: at prefill M the 21.8 us-class cast dispatch is
far cheaper than per-k-step bf16 shared-memory widening, the exact inverse
of the H111 small-M result in direction. **Not landed.** The branch stays
pushed as the negative record; the default route is byte-for-byte
unchanged with the env unset. Conclusion for the lane: the X32
cast+direct-global-A route is the right prefill route on G13G; the cast
pass is not prefill headroom.

Version fact for the release lane: on current main (`22cdc6da5`, after the
v0.7.22 cut at `58724762e`) the 2B pf digest on G13G is
`a4ebce784981475a`, not the v0.7.21/v0.7.22 pin `bc519c03c4ef5fd1`.
This is content between `58724762e` and `22cdc6da5` (GDN FLT_MIN floor +
wave-barrier hardening, ssm-maskless serve patch), NOT the lever: the
added block is a pure env-gated early return (with the env unset the
default path is byte-identical code), and ctl == staged in every cell.
Closed empirically on the post-macOS boot (`797d839e…`, settled): the
**published v0.7.22 wheel** (`0.32.4.dev202610031525+58724762`, private
venv, no env flags) reproduces the pin — pf512 digest
`bc519c03c4ef5fd1`, prefill 413.8 tok/s, TTFT 0.1306 s (0.904x / 1.045x
vs macOS; ~1.3% under the v0.7.21 settled read, single cell). The
`a4ebce784981475a` digest therefore belongs to main content after the
v0.7.22 cut and must be re-pinned by whoever ships it.

## 5. Next axes (proposed, in the order they should be tried)

1. **A-operand cache behaviour across row tiles** (bit-exact, no numerics
   gate): the coopmat grid is (n_groups, m_groups) with x-linear
   workgroup order, so all workgroups sharing a 32-row A tile are
   adjacent (good A reuse), but each weight slab is re-read by the
   m_groups=16 row tiles spread across the whole kernel — weights
   stream from DRAM m_groups times per op. Concrete experiment: (a)
   re-profile with the 4-join pattern (H110 style; this driver's
   single-join timestamps are unusable — §3) to get trustworthy per-op
   ms on the current wheel; (b) an m-major dispatch variant
   (`dispatch(m_groups, n_groups)` swap plus in-shader index swap;
   bit-exact) to flip which operand streams, 5 alternating pairs +
   digests on jwm1. If neither direction moves >= 1.5%, the streaming
   order is not the limiter and the axis closes.
2. **qmm coopmat issue quality** (numerator of the 0.916x): the shipped
   kernel runs one 32x32 output tile per workgroup with a single
   dependent K chain; the two untried shapes that keep the per-output k
   chain bit-exact are (a) a two-n-tile workgroup (64x32 output,
   4 subgroups, each subgroup owning its own n-tile's w_s staging —
   halves the weight re-reads without raising per-subgroup accumulator
   pressure, unlike the rejected 64-row H6), and (b) persistent
   workgroups looping over m tiles so the w_s dequant amortizes across
   row tiles. Both are env-gated shader variants + the standing A/B
   protocol (5 alternating pairs, digest identity, jwm1, >= 1.5% to
   land). If both lose like every predecessor, the residual gap is in
   honeykrisp's coopmat lowering and belongs to the Mesa lane
   (joshuaswarren/mesa-1).

## 6. Closed axes (do not re-run)

64-row tile 0.939x (H6), STEP_K 32/64 0.897/0.880x (H7), SG4 0.978x (H13),
staged-A at M<=16 +15.5 ms TTFT (H111), split-K M16 (H106/107/109),
NormApple G13G (decode +0.9%, prefill -0.6%, H231), staged-A at prefill M
(this receipt, 0.75-0.78x). Remaining prefill-parity axes: qmm coopmat
issue quality in honeykrisp (Mesa lane), A-operand cache behaviour across
row tiles (untested; needs the 4-join profile pattern or hardware
counters), and the ~27 us/hop dependent-dispatch turnover (Mesa lane).

## Artifacts

Private lab: `artifacts/PrefillGap/prefill-gap-jwm1/` (baseline/AB/profile
logs, runbook log, SHA256SUMS). Harnesses staged under `/var/tmp/pregap*`
on the hosts (cleaned after this receipt; archived to the private backup
host first).
