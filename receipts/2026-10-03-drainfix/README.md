# 2026-10-03 DrainFix — jw16 decode host/driver drain: timeline quantification + lookahead-depth A/B (negative result, no landing)

Lane: DrainFix (worker). Host: jw16 (M1 Max, T6001), ssh alias per fleet config.
Base: mlx-omarchy origin/main eaa697236; serving wheel 0.32.4.dev202610031046+b581d5c
and serving ICD UNTOUCHED. Branch `agent/drain-fix` carries this receipt and the lane
tools. Private notebook:
`entries/DrainFix/20261003T231000Z-jw16-decode-drain-quantify-lookahead.md`
(pre-registered before any window); artifacts `artifacts/DrainFix/20261003/{w1,w2-rerun-pending,w3,w4}`
with SHA256SUMS, plus a read-back-verified scratch archive off-repo.

## Question

Where do the ~12.7 ms/token of profiled host/driver drain in 2B decode go, and can any
host-side lever recover >= 1.5 % of decode throughput (the pre-registered landing bar)?

## Q1 — timeline (measured)

The 12.7 ms/token figure is a profiler artifact (DecodeBw W2 inserted a barrier per
dispatch; profiled wall 15.4 ms vs 9.2 ms cells wall). True per-token budget at d512:

| component | value | source |
|---|---|---|
| wall (unstraced) | 9.7 ms (102.7 tok/s); d64 9.18 ms | this lane ctl cells |
| GPU busy (true) | [2.70, 6.15] ms | DecodeBw W2 ring-sum bracket |
| intra-chain gaps | [3.05, 6.50] ms over ~228 dispatches (13-28 us/gap) | DecodeBw W2 + Turnover W5 (21.0 us/op ring rate) |
| token seam | 0.216 ms | DecodeBw W2 |
| cross-submit gap | 0.26 ms | DecodeBw W2 |

New syscall census (strace 7.2, first on this host — prior lanes lacked strace and
tracefs is blocked; counts exact, durations ptrace-inflated ~3x, window = last 14.96 s,
~520 tokens incl warmup):

| ioctl class | count/token | sum (inflated) |
|---|---:|---:|
| asahi submit (`_IOC(_IOC_WRITE, 0x64, 0x40, 0x18)`) | 1.92 | 118 us |
| SYNCOBJ_QUERY (fence poll) | 59.2 | 778 us |
| GEM_CPU_PREP-class query | 89.7 | 2770 us |
| GETPARAM-class query | 89.3 | 1099 us |
| SYNCOBJ_TIMELINE_WAIT (blocking) | 3.3 | 62 us |
| GEM_CLOSE | 4.3 | 230 us |

Reading: the host confirms 2 submits/token directly, blocks on the fence for only
~3 waits/token, and spends the rest of its margin in ~240 status-query ioctls/token.
Submit ioctls are cheap (~60 us) when the GPU is slowed by strace vs ~2 ms unstraced
(DecodeBw W2) — submit wall is ring backpressure, not fixed cost. The host is not the
token-time binder.

## Q2 — rank (with prior-lane evidence)

1. BATCH_WORK cap (0 or 100k): +1.5-1.8 % alone, +2.6 % with poll (Turnover W1-W3) —
   blocked by Main's cap decision (W8). Note: W8's prefill max criterion fails for the
   40k status quo itself (21.7 ms during 1024-token prefill), so the gate as written
   rejects every cap including the current one.
2. Decode-loop lookahead depth 2 (this lane): +0.3..+0.7 %, bit-exact.
3. HK_SUBMIT_POLL_US 5-20k: +0.4-0.6 % alone, +20-33 % process CPU (Turnover W7).
4. uclamp placement: already shipped (default 1024, v0717).
5. Refuted elsewhere, not re-run: HK_PERFTEST=batch (neutral), batch-open mesa branch
   (-8.1 %), pack64-only (neutral), T1/T2 levers, kernel geometry family.

## Q3 — A/B (negative, no landing)

Candidate: env-gated depth-2 lookahead in `mlx_lm.generate.generate_step`
(`MLX_OMARCHY_DECODE_LOOKAHEAD`, default off = upstream-identical behavior; candidate
venv = clone of the serving venv with only that file patched, sha256
bc4903b5...bef9c55 -> d4b95e1c...32f9f11; wheel stamps identical on both sides by
design, asserted via mlx_provenance version_match=true). Stub-harness proof: gate off is
behaviorally identical to upstream (yields + enqueue order, max_tokens in {1,2,3,5,64});
gate on enqueues exactly one chain further ahead, same order, no extra tail chains.

Protocol: paired alternating ctl (serving venv) / on (candidate venv, gate=2), one
model load per arm invocation, 5 rounds per depth split across two windows
(W3 on-first, W4 ctl-first), gates uptime>=360 s / load1<0.5 / PSI cpu some avg10=0.00
inside every window, greedy digest checked against per-(depth,passes) pins every run
(`ordered_records_hash` covers pass index, so pins are per-protocol; ids bit-identical
everywhere), plus in-process "patch-active" assertion on every candidate arm.

| depth | ctl med [min-max] | on med [min-max] | paired delta | windows |
|---|---|---|---|---|
| d64 | 109.0 [108.73-109.45] | 109.46 [109.25-109.57] | +0.28 % (n=3) / +0.43 % (n=2) | 2 |
| d128 | 108.3 [108.20-108.60] | 108.7 [108.56-108.84] | +0.38 % (n=3) / -0.09 % (n=2) | 2 |
| d256 | 107.3 [106.25-107.43] | 107.5 [107.32-108.13] | +0.18 % (n=2) / +0.75 % (n=3) | 2 |
| d512 | 102.55 [102.48-102.61] | 103.2 [102.91-103.29] | +0.66 % (n=5) / +0.48 % (n=2) | 2 |

d512 is disjoint in both windows; d64 overlaps; d128/d256 inconsistent. Stack arms
(lookahead-2 + HK_SUBMIT_POLL_US=10000/20000): 103.78 and 103.72 med vs ctl 102.53-102.57
= +1.18 % / +1.16 % at d512 — dose saturated, still below bar, and the poll dose carries
the known +20-33 % process CPU cost. Neutral arms (candidate venv, gate off): d64 109.31,
d512 102.68 — inert machinery confirmed.

VERDICT: the pre-registered landing rule (>= +1.5 % median, disjoint, >= 2 windows,
digests exact) is met by NO candidate. Nothing landed, nothing deployed; serving state
unchanged and health-probed after every window.

## Provenance

- mlx-omarchy branch `agent/drain-fix` @ HEAD of eaa697236; serving wheel
  0.32.4.dev202610031046+b581d5c on both A/B sides (mlx_provenance version_match=true);
  mlx_lm 0.31.3; kernel 7.1.13-3-2-ARCH; boot 4b353848-84fc-4dfc-bd7c-738309e45c60 for
  every window (same boot as all same-day prior-lane measurements).
- strace 7.2 installed on the host during prep (recorded; retained).
- Scratch deleted after archiving (read-back-verified off-repo archive; hashes in the
  notebook entry). llm-inference active, health probe finish_reason=length after every
  window, including one extra restore cycle post-cleanup.

## Where the remaining gap lives

The drain that is left is GPU-side: per-dispatch ring turnaround (~13-26 us measured
vs ~12.5 us implied by the macOS same-model figure) across ~229 dispatches/token. That
is outside host/python reach. Identified follow-ups, both outside this lane's no-reboot
scope: (a) the boot-time DT governor axis (`apple,perf-tgt-utilization`, base pstate —
method owned by the jwm1-parity lane on T8103; untested on T6001 decode); (b) the
BATCH_WORK=0 lever (+1.5-1.65 % measured, bit-exact) pending an owner decision on the
cap rule.
