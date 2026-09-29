# 2026-09-29 — M1 Max decode: GDN state tile, logsumexp/argreduce prefetch (bit-identical, +5-6% over the serving wheel)

Host: T6001 (M1 Max, 32-core GPU), Omarchy Linux 7.1.6-1-1-ARCH, Mesa 26.3.0-devel
Honeykrisp (installed ICD lineage jw16/submit-poll-on-flush17f, `poll9d949d4`,
sha256 `eaab047e…`), Vulkan device "Apple M1 Max (G13C C0)". Model
SiddhJagani/Qwen3.8-2B-mlx-4Bit snapshot 0867d98b (model.safetensors sha256
`b0d5de68…`). mlx-lm 0.31.3 with the serving venv's patches, Python 3.14.7.

## What changed (commits 4f291fec7, a773ea948)

- `shaders/gated_delta_decode.comp` + dispatch: 4 workgroups per head (32 Dv
  rows each) instead of 1; the 32x128 f32 state tile is loaded and stored
  coalesced through padded shared memory. The per-row walks keep the shipped
  kernel's arithmetic exactly (see "Exactness" below).
- `shaders/logsumexp_suffix.comp`, `shaders/argreduce_suffix.comp`: each
  thread's strided walk issues 16 loads before consuming them in the original
  order. Same per-thread sequence, same tree.

## Attribution that picked these (diagnostics wheel af95191, 64-token decode)

- 315 dispatches/token. The profiler's per-dispatch t0/t1 pair does not
  bracket execution on this driver (every kernel reads ~10 us, a 14 MB GEMV
  would be 650 GB/s); completion-to-completion (t1[i] - t1[i-1]) is
  self-consistent and was used, with a ~25 us profiled floor per dispatch.
- Outliers against their byte cost: LogSumExp over the 248320-wide logits
  706 us, ArgReduce 483 us (one 256-thread workgroup, 970 strided steps per
  thread, one load per memory round trip), GatedDeltaDecode 121 us x 18
  (16 workgroups, each lane walking a 512-byte-strided state row).
- macOS on the same laptop (Main's window 4, mlx 0.32.2 Metal), dependent-op
  chain slope: add 4.8 us, rms_norm 4.6 us, q4 GEMV 2048x2048 14.2 us,
  6144x2048 34.0 us. Linux (af95191): add 12.1, rms_norm 26.4, GEMV 18.3-20.4,
  39.6-40.8 us. Host record+submit is 1-3 us/op, so the per-op excess is GPU
  side: ~7 us per trivial dependent dispatch and ~14 us more inside
  fast_norm.comp. Script: ane-linux-experiments
  `scripts/jw16-macos-parity/chain_costs_decode.py`.

## Exactness

The first cut (a773ea948) moved the GDN row walks to shared memory with the
same source text and was NOT bit-identical (state rows 10-17% of elements
off by rounding; y bits identical; tokens identical). AGX NIR dumps
(`AGX_MESA_DEBUG=shaders`) showed why: Honeykrisp runs nir_opt_reassociate on
non-exact float chains, and the shipped kernel's `kv += s * g * k[i]`
compiled to kv = ffma(s, fmul(k[i], g), kv), `ns = s * g + delta * k[i]` to
ffma(s, g, fmul(delta, k[i])), out = ffma(ns, q[i], out). The same source
reading a shared tile regroups differently. GLSL `fma()` lowers to
`ffma_weak(contract)`, which NIR splits and re-fuses, so it pins nothing.
4f291fec7 makes every step a two-operand chain with one possible
contraction: k[i] * g is a per-workgroup table (the same single fmul),
delta * k[i] is `precise`, and the walks and gate prologue sit in separate
functions because glslang propagates `precise` backward through everything
that feeds it inside one function (SPIR-V: NoContraction only on delta and
delta * k[i]).

Bit check (`dg_bitcheck.py`, same numpy-seeded inputs in both venvs,
SHA-256 of raw output bits): 3084 of 3084 rows identical between the base
wheel (af95191) and 4f291fe — gated_delta_update_raw decode 300 seeds (state
scales 0..1e3), bf16-gate decode 300, inf/nan/huge state rows 4, 64-step
chains 10, logsumexp 270 (15 shapes x 6 value sets incl. NaN/inf/-inf rows/
ties/+-0 x f32/f16/bf16), argmax/argmin 720 (float and 6 integer dtypes),
the composed decode head `argmax(logits - logsumexp(logits))` 40, fast
rms_norm/layer_norm 1440 (unchanged kernel, control).

## Cells (run-linux-cells protocol: prompt 1, warmup 1, 5 passes, prefill leg 512; one GPU window, arms interleaved per cell; service stopped, lock held, restored with health 200 + completion finish=length)

| cell | serving rw10 | af95191 | 4f291fe | vs serving | macOS | Linux/macOS before -> after | digest (all three arms) |
|---|---:|---:|---:|---:|---:|---|---|
| d64 | 88.67 | 90.00 | 93.52 (92.91-93.54) | +5.5% | 180.01 | 0.493 -> 0.520 | c84b3e7af6401c64 |
| d128 | 87.66 | 89.87 | 93.03 (92.76-93.07) | +6.1% | 179.28 | 0.489 -> 0.519 | 07c515e0338b9108 |
| d256 | 86.68 | 89.16 | 92.06 (91.83-92.26) | +6.2% | 178.28 | 0.486 -> 0.516 | c6aabbf0a51de38d |
| d512 | 84.20 | 85.64 | 88.26 (88.21-88.45) | +4.8% | 177.12 | 0.475 -> 0.498 | 5c120987f0e5869d |

The af95191 column is origin/main before this change (its gain over the
serving wheel is the decode-SDPA prefetch lineage and the removed CCTRACE
prints); this change alone is +3.1% to +3.9%.

Host split of the greedy loop (host_split.py, 96 tokens): graph build 0.81 ms,
record+submit 4.56 ms, wait 5.36 ms, token 10.69 ms (base: 0.81 / 4.56 /
5.74 / 11.10). Decode stays GPU-bound; recording 315 dispatches costs
~14.5 us each on the host.

No synchronization, barrier, or submission code changed; the race battery
does not apply.

## Remaining gap (ranked, not done here)

1. fast_norm.comp (61 dispatches/token, ~14 us over a trivial dispatch each
   in a dependent chain): same prefetch treatment; its 8-step unrolled sum is
   a candidate for the reassociation trap above, so verify with a NIR dump.
2. Per-dependent-dispatch GPU turnover (~7 us over macOS x 315): fusion
   (norm into the GEMV prologue, conv+silu+q/k norms into the GDN kernel).
3. q4 GEMV efficiency: Main's standalone shapes read Linux 113-193 GB/s vs
   macOS 186-299 GB/s (lm_head 193 vs 299 on one 286 MB dispatch).

## Addendum (same day, window 4, 14:11-14:15Z, fresh boot 38371487)

Arms: c2 = main 4f291fe wheel (this receipt's change), c3 = c2 + a
fast_norm.comp load-prefetch candidate (branch `agent/jw16-decode-attrib`
53ad7cf), gp = c3 wheel + `patches/mlx-lm-greedy-prune.patch` applied to the
test venv's mlx_lm (the serving venv does not carry that patch; the dispatch
census showed the full 286 MB tied head + logsumexp + argmax every token).

| cell | c2 | c3 | gp | gp vs serving (88.67/87.66/86.68/84.20) | gp / macOS | digest (all arms) |
|---|---:|---:|---:|---:|---:|---|
| d64 | 93.36 | 93.92 | 100.18 (100.06-100.33) | +13.0% | 0.557 | c84b3e7af6401c64 |
| d128 | 92.96 | 93.75 | 100.01 (99.89-100.32) | +14.1% | 0.558 | 07c515e0338b9108 |
| d256 | 92.27 | 93.03 | 98.70 (97.71-98.84) | +13.9% | 0.554 | c6aabbf0a51de38d |
| d512 | 88.49 | 89.24 | 94.64 (94.41-94.87) | +12.4% | 0.534 | 5c120987f0e5869d |

- The greedy-prune route is already on main (wheel op + mlx-lm patch, listed
  in `scripts/apply-mlx-lm-patches.sh`); applying it to the serving venv is
  worth +6.1..6.7% on top of this change with every digest unchanged.
- c3 (fast_norm prefetch) is NOT landed: +0.6..0.8% (below the
  pre-registered +1% bar) and not bit-identical for f32 weighted rms_norm
  (86 of 360 rows differ; bf16/f16, unweighted and layer_norm rows identical).
  AGX NIR of the shipped kernel: sum of squares acc = ffma(v, v, acc) in a
  loop; bf16 output fmul(v, fmul(norm, w)). A `[[dont_unroll]]` loop over a
  preloaded array keeps the sum-of-squares chain at length 2 (NIR of c3:
  ffma(v, v, acc) per iteration); the f32 output grouping differs from the
  bf16 one, so pinning it needs the f32 NIR too.
