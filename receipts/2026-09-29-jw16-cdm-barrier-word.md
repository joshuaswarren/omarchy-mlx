# 2026-09-29 — jw16 (M1 Max) decode: per-launch CDM barrier word bisect — the installed 0x17f word is the minimal correct set; per-dispatch turnover breakdown

Lane: DecodeGap3. Host: jw16 (M1 Max, T6001/G13C "G13X"), Omarchy Linux 7.1.6-1-1-ARCH,
boot `cacd7b45` throughout. Driver: system ICD `9d949d4-vec2` (poll lineage
`jw16/hwmat-vec2-on` = `9d949d4ba48` + AGX_HWMAT_VEC2 default-on, installed by the
prefill lane at 16:58Z — all three windows of this lane ran on it; no VK_DRIVER_FILES
override). Serving venv `/var/tmp/v072-venv-fused` (mlx-omarchy 5ab5133 + mlx-lm
greedy-prune), model SiddhJagani/Qwen3.8-2B-mlx-4Bit snapshot 0867d98b. Zero builds:
every arm is `HK_CDM_BARRIER_MASK=<hex>` on the installed driver; the venv and ICD were
touched by nobody in this lane. Notebook: `apple-silicon-lab/entries/Jw16DecodeGap3/`
(pre-registered 17:24Z), artifacts `artifacts/Jw16DecodeGap3/w1..w3` with SHA256SUMS.

Assignment: find why a trivial dependent dispatch costs ~12 us on Linux Honeykrisp vs
4.8 us on macOS, and design a fix (barrier emission / batching / fusion).

## Method

- chain_costs_decode.py (sha `ad94a7a1`): dependent-op chain slopes, n=32..160, 7 reps.
- Cells: d64/d128/d256/d512, n=5 passes per invocation, arms interleaved (win1/w2/w3),
  digest = ordered_records_sha256. Production pins: `c84b3e7a / 07c515e0 / c6aabbf0 /
  5c120987`. Every window via gpuwin.sh (service restore: health 200 + finish=length).

## Word map (cells; digest = first 12 hex of ordered_records_sha256)

| word (bits) | d64 | d128 | d256 | d512 | verdict |
|---|---|---|---|---|---|
| ctl = 0x17f {0,1,2,3(usc),4,5,6,8} | 99.06-99.32 | 98.97-99.18 | 97.56-97.58 | 93.76-93.97 | all pins, 10 runs |
| 0x178 {4,5,6,8}+usc | 101.61/101.82 | - | - | - | d64 clean (w1 only) |
| 0x1F0 {4,5,6,7,8} | 101.88/101.61 | - | - | - | d64 clean (w1 only) |
| 0x170 {4,5,6,8} (no 0-2, no usc) | 101.57-101.99 (4 runs) | 101.58-101.72 (2) | **100.15 `08fe39753f2f` / 99.98 `72b695e83b6a`** | **96.12 `046ab1a06a06` / 95.88 `edf6ba95e771`** | clean at 64/128, CORRUPT at 256/512 — a DIFFERENT wrong digest every run |
| 0x187 {0,1,2,4,5,6,8} (drop usc only) | **60.27 `dd095c691cac` / 60.08 `43d5f2bb90cd`** | - | - | - | CORRUPT at d64, two distinct digests |
| 0x0F7 {0,1,2,4,5,6,7} (usc dropped, wait7) | 99.26/99.31 | - | - | - | clean, NULL (no gain) |
| nocdm (no barrier block; HK_PERFTEST) | 76.87 `6df38fc6675a` | - | - | - | corrupt AND slower e2e |

Controls in every window reproduced the production pins exactly. The d256/d512 corrupt
arms are the nondeterministic-corruption class (new wrong digest per run), not a
rounding class.

## Chain slopes (us/op; add = mx.add[1,2048] bf16 chain; gemv = q4 2048x2048)

| arm | add | gemv 2048^2 | gemv gbps | gemv 6144x2048 |
|---|---:|---:|---:|---:|
| ctl 0x17f | 12.16-12.23 | 22.9 | 102.8-102.9 | 39.3-39.7 |
| 0x170 | 7.45-11.9 (noisy) | 22.0-22.2 | 106.3-107.0 | 39.3-39.4 |
| nocdm (corrupt) | **1.34** | **13.30** | **177.4** | **29.12** |
| macOS (window 4, Jw16DecodeGap) | 4.77 | 14.24 | 166 | 34.0 (208 GB/s) |

## Findings

1. **The wait/drain bit (7 or 8, equivalent) is required AND is the cost.** Every
   wait-bearing word costs ~11-12 us/op on real add chains and ~10 us over the 1.3 us
   unordered floor, regardless of the other bits. macOS pays 4.77 us total on the same
   chain — ~3.4 us over the same unordered floor (attribution inference: Apple's
   dependency machinery is nearly free relative to overlap; Honeykrisp's CDM_BARRIER
   drain is not). The ~7 us/dispatch excess is firmware drain semantics, not reachable
   by bit selection inside this vocabulary.
2. **Bits {0,1,2}+usc are the 09-24 "rare race" — length-dependent corruption.** Without
   them (0x170) the stream is bit-exact at 64/128 tokens (+2.5% decode, 8 clean runs)
   and deterministically-corrupts-from-d256 on (a new wrong digest per run). The
   launch-sink lane's 2026-09-24 failure (2/7 divergences at 32-token passes) was this
   same corruption surfacing late; the 0-2 restore (2a9762ef6ec) was the correct fix.
   Short gates are structurally blind to it.
3. **The bit words are NOT independent.** Dropping ONLY usc while keeping 0-2 (0x187)
   corrupts immediately at d64 (slower AND wrong); dropping 0-2 AND usc is clean until
   d256. No compositional reasoning about these bits is valid; only whole-word gates
   count.
4. **The corrupt-arm speedups are dispatch overlap.** With no barrier block the chain
   slope of a 2.36 MB q4 GEMV falls 22.9 -> 13.3 us (102.9 -> 177.4 GB/s) and the
   norm+GEMV pair to 12.3 us — the GPU overlaps consecutive kernels. macOS's 166 GB/s
   on the same shape sits between our correct-word (102.9) and unordered (177.4)
   slopes: an unknown fraction of the "Linux GEMV kernel gap vs macOS" is boundary
   drain, not kernel code quality (attribution inference, consistent with the
   gemv_shapes probes).
5. **Verdict: the bit-exact CDM barrier trim on G13X is exhausted — again, and now on
   the current stack.** The installed 0x17f is the minimal correct word known. Nothing
   installed, nothing landed in code; the lever is closed with receipts. To capture the
   ~2.5% short-stream win someone must first find what bits 0-2+usc actually protect at
   length (candidates: cross-cluster L2/SLC coherence on the two-die M1 Max) — a
   firmware-semantics investigation, not a bit swap. Bits 9-19 have never been tested
   individually (only inside the kitchen sink).

## Per-dispatch cost breakdown (the measured answer to "why 12 vs 4.8 us")

| component | us/dispatch | evidence |
|---|---:|---|
| launch floor (empty, no memory write) | 0.5-0.7 | 2026-09-23 MaxDispatch floor matrix |
| unordered dispatch (no barrier; corrupt) | 1.3 | nocdm add slope, this lane |
| macOS full correct dependency | 4.8 | macOS chain_costs (window 4) |
| Linux minimal correct word (0x17f) | 12.2 | ctl add slope, this lane |
| **excess vs macOS** | **~7.4** | the CDM_BARRIER drain+cache-maintenance sequence |

At ~315 dispatches/token (~240 barrier-bearing after gating+RW-split) that excess is
~1.7-1.9 ms of the ~4.5 ms/token decode gap; the remainder is the same excess showing up
inside kernel slopes (GEMV drain/ramp) plus known kernel-level gaps.

## Production ratios for the record (this lane's windows, vec2 ICD)

d64 99.06-99.32 / 180.01 = 0.550-0.552; d128 98.97-99.18 / 179.28 = 0.552-0.553;
d256 97.56-97.58 / 178.28 = 0.547; d512 93.76-93.97 / 177.12 = 0.529-0.530.
Untouched by this lane (control cells reproduce the serving pins exactly).

## Artifacts

- `apple-silicon-lab/artifacts/Jw16DecodeGap3/w1,w2,w3/` (chain JSONs, cell JSONs/logs,
  order logs, SHA256SUMS, verified after pull). jw16:/var/tmp/dg3/ (scripts + raw).
- Window restore receipts: health 200 + completion finish=length after every window
  (w1-run.log, w2-run.log, w3-run.log RESTORE lines).
