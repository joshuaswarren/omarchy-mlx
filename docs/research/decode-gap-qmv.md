# Decode-gap qmv: per-shape Linux vs macOS on the same chip

Status: REFERENCE (complete for jw16, the 16-inch M1 Max / T6001; single
macOS window so far). This is the measurement record behind the
beat-macOS goal for quantized matmul (qmv). Receipts live on the lab CT
under `~/scratch/w7x-decode-gap/results/`; every number below traces to
a file named there.

## Question and method

Same machine, both operating systems: per-shape quantized-matmul
throughput in GB/s (median us/dispatch converted at bytes/dispatch),
one shape per 2B-decode matmul, distinct weights. Bench:
`bench_qmv.py` (w7P, sha256 33fcaf18), reps 15, independent and one
chained case.

- Linux: mlx-omarchy, jw16, kernel 7.1.12-2-20261103.2-lab-ARCH, GPU
  vote on, cpuset 2-9 (`results/qmv-bench-jw16-r2{.out,/}`).
- macOS: mlx 0.32.4, Darwin 25.6.0, same chip, quiet window
  2026-11-03T20:51-20:53Z, kit run alone, machine just back from a
  reboot (`results/qmv-macos-jw16/run{1,2}.jsonl`).
- Flap check: 3 back-to-back bench passes in separate processes plus
  one after 300 s idle, PTD words and four macsmc proxy temps sampled
  before/after every pass (`results/qmv-decay-jw16-r2{.out,/}`).

Linux "p1/p2" are two passes inside one bench process; macOS "r1/r2"
are two separate whole-bench processes. The structures differ - keep
that in mind everywhere below.

## Numbers

| shape (bytes/dispatch) | Linux p1/p2 GB/s | Linux 4-pass band | macOS r1/r2 GB/s | class |
|---|---|---|---|---|
| lm_head 248320x2048 (286.06 MB) | 269.1 / 268.6 | 268.3-270.7 | 337.9 / 340.4 | STABLE |
| chain proj 2048x2048 (2.36 MB) | 68.9 | 68.2-69.5 | 127.6 / 127.5 | STABLE |
| attn_kv 512x2048 (0.59 MB) | 50.5 | 50.1-66.3 | 77.5 / 76.3 | mostly stable |
| proj indep 2048x2048 (2.36 MB) | 77.3 | 70.3-77.5 | 95.6 / 100.9 | mostly stable |
| gdn_ab 16x2048 (0.02 MB) | 3.5 | 3.5-3.6 | 3.9 / 3.8 | stable (latency-bound) |
| attn_q 4096x2048 (4.72 MB) | 113.6 / 95.4 | 92.8-143.8 | 120.0 / 98.0 | FLAPS |
| mlp_down 2048x6144 (7.08 MB) | 166.1 / 119.0 | 138.1-158.2 | 110.2 / 203.1 | FLAPS |
| gdn_qkv 6144x2048 (7.08 MB) | 153.3 / 135.3 | 117.2-151.4 | 100.6 / 210.9 | FLAPS |
| mlp_gate_up 6144x2048 (7.08 MB) | 142.4 / 116.2 | 117.5-148.0 | 100.3 / 100.0 | FLAPS |

## What is stable and what flaps

Stable: lm_head is immune to everything measured - a 0.9 percent band
across four passes and both OSes never below 337. chain proj, gdn_ab,
and (mostly) attn_kv and proj also hold a band. macOS's lead on these
is regime-independent: even Linux's fastest observed draws stay below
macOS (lm_head 270.7 vs 340; chain 69.5 vs 127.6; attn_kv 66.3 vs
76.9; proj 77.5 vs ~98).

Flapping: the 4.72-7.08 MB shapes swing +-15-30 percent per pass in
BOTH directions, on both OSes, with no monotone law. macOS r1->r2
halves us/dispatch on mlp_down and gdn_qkv (warm-up; r1 was the first
run after reboot); Linux passes swing both ways (mlp_gate_up warms up
117->148 GB/s, attn_q falls 135->93 then recovers 144 after idle).
mlp_gate_up never enters the macOS fast regime in the median (r2 min
touched 33.3 us/dispatch, ~211 GB/s, so it flaps per-dispatch there
too). No DVFS correlate: DVFS-STATE w0 is constant across all 8
snapshots (w1 reads as a time counter), PMP-STATUS constant, proxy
temps moved under 1.5 C. LIMIT, stated: this kernel exposes no
GPU/SoC-die sensor, so a die-thermal mechanism is neither supported
nor excluded; and 300 s idle does not reset the regime consistently.

## Retraction (2026-11-03)

An earlier report claimed "jw16 pass 2 slower than pass 1 by 12-28
percent on the big shapes" from the r2 bench. The 4-pass row above
does not replicate it: that pattern was one draw from the flapping
distribution. Consequence: single-pass numbers on the flapping shapes
carry +-15-30 percent run-to-run risk on this boot/kernel; quote
single numbers only for the stable shapes. A regime-flap chase row is
explicitly deferred (Main, 2026-11-03 15:38 CST: not this week unless
a kernel change needs it).

## Target levers

The stable shapes carry the beat-macOS gap and are safe to optimize
against:

1. lm_head 248320x2048: macOS +26 percent (339 vs 269). Largest
   single-dispatch shape in decode (286 MB), immune to regime noise.
2. chain proj 2048x2048: macOS +85 percent (127.6 vs 68.9). The
   chained-dispatch path; biggest relative gap on the board.

Owners: w7P and w7Q, jointly (target list issued by Main,
2026-11-03 15:24 CST); per-lever split not yet assigned. Supporting
gaps for the same owners: attn_kv +52 percent, proj indep +27
percent.

## Receipts

- `~/scratch/w7x-decode-gap/results/qmv-bench-jw16-r2{.out,/}` -
  Linux two-pass bench (kernel, vote, load1 at start: load 0.24).
- `~/scratch/w7x-decode-gap/results/qmv-macos-jw16/run{1,2}.jsonl` -
  macOS runs (sha256 626a7e38... / b48af310...; no load1 in jsonl -
  kit gap, fixed in the next bundle, wrapper b999d6a6).
- `~/scratch/w7x-decode-gap/results/qmv-decay-jw16-r2{.out,/}` - the
  4-pass flap/retraction row with PTD words and proxy temps.
- Full working table with the jwm1 (13-inch M1) column:
  `~/scratch/w7x-decode-gap/results/QMV-TABLE-linux-20261103.md`.
