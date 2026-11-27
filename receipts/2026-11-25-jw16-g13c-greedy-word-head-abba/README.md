# G13C fused greedy word head ABBA (jw16)

Date: 2026-11-25. Receipt for the DEC-46 word-column fused greedy
quantized-argmax head (branch `w7y-dec46-greedy-word`, head `63087be59`).

## Host and provenance

- Machine: 16-inch MacBook Pro, Apple M1 Max (`Apple M1 Max (G13C C0)`).
- Vulkan driver: `Mesa 26.3.0-devel (git-1650fc1a33)` (Honeykrisp/Borealis).
- Wheel under test: `mlx-omarchy 0.32.4.dev202611260056` (prod venv
  `/usr/lib/omarchy-mlx/venv`), built from `63087be59`.
- Model: `mlx-community/Qwen3.5-4B-MLX-4bit`, snapshot
  `32f3e8ecf65426fc3306969496342d504bfa13f3`, 4-bit g64, hidden K=2560,
  vocab n=248320. Tied `QuantizedEmbedding` lm_head, `qwen3_5` arch —
  the greedy head route is active on this model.
- Harness: `dec46_abba.sh` → `dec46_decode.py`, 64-token greedy decode,
  ABBA-interleaved legs, each leg one process; whole run a single
  `gpu-turn` ticket on the jw16 GPU queue.
- Arms: A = fused head (default); B = composed 5-dispatch tail
  (`MLX_OMARCHY_NO_GREEDY_PRUNE=1` disables the fused route on the same
  binary — recorded as `prune_env` in the per-leg JSON).

## Contract check

`ids_sha` = `294463eb6dba3a58` on all 20 legs — token-identical greedy
decode on both arms. First 8 ids `[279, 15217, 5388, 13, 561, 3841,
13477, 37550]` identical on both arms.

## Result (5 pairs ABBA, tps per leg)

| pair | leg order | A (fused) | B (composed) |
|---|---|---|---|
| 1 | B A B A | 43.48, 43.30 | 43.19, 43.04 |
| 2 | B A B A | 43.30, 43.52 | 43.05, 43.10 |
| 3 | B A B A | 43.37, 43.28 | 43.18, 43.12 |
| 4 | B A B A | 43.47, 43.39 | 43.12, 43.01 |
| 5 | B A B A | 43.38, 43.24 | 43.01, 43.08 |

- A median 43.38 tps, B median 43.10 tps → **+0.65% decode**, fused wins
  10/10 legs.
- Micro-bench on the same shape (n=248320, K=2560): fused primitive
  1.502 ms vs composed tail 1.500 ms — the tails are at parity in
  isolation; the win is 3 dispatches vs 5 plus no full-vocab argreduce
  read.

## Root-cause record (three rounds to get here)

1. Round 1 (2-dispatch, per-chunk single-WG GEMV): identical ids, ~14%
   slower — one workgroup per 4096-vocab chunk serialized the GEMV.
2. Round 2 (3-dispatch, row-parallel shared-tree GEMV): still ~12%
   slower — the composed GEMV picks the subgroup word kernel
   (`qmm_geometry().word`) on this device; my shared-tree chain could
   not match it.
3. Round 3 (dispatch the picked lm_head vec kernel into a bf16 scratch,
   then chunk partials + combine/select): still ~12% slower. The
   primitive itself was already at parity — the gap was the python
   head's `int(token)` overflow-sentinel check inside `_step`, which
   syncs the device every token and breaks `generate_step`'s one-token
   lookahead. `cand_cap = n` makes the sentinel impossible by
   construction; the check now runs only on k==2048 where the staged
   subgroup route can still emit it.

## Per-leg JSON

See `pairs.md` (verbatim harness output).
