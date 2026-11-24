# G13C SDPA split-K decode ABBA screen and 5-pair confirms (jw16)

Date: 2026-11-24. Receipt for PR #840 (G13C split-K decode defaults).

## Host and provenance

- Machine: 16-inch MacBook Pro, Apple M1 Max (`Apple M1 Max (G13C C0)`), 32 GPU cores.
- Kernel `7.1.12-2-20261103.2-lab-ARCH`; Vulkan driver Honeykrisp (`DRIVER_ID_MESA_HONEYKRISP`), package `vulkan-asahi-omarchy 1:26.2.4.git69bde15340-2`.
- Wheel under test: `mlx-omarchy 0.32.4.dev202611210550+bf5631d` (prod venv `/usr/lib/omarchy-mlx/venv`). At this commit G13C defaults were `min_keys=320`, `pv_max=2048` (PV default-on below 2048 keys), `bk64_max=0`.
- Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit`, snapshot `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, bf16 4-bit.
- Harness: `abba_4b.py` (3-pair screen then 5-pair confirms), ABBA-interleaved legs, 8 warmup + 64 generated tokens, k_len during decode ≈ ctx .. ctx+72. Each cell ran as one `gpu-turn` ticket on the jw16 GPU queue (serialized against other GPU users).
- Output check: token ids hashed per leg (`ids_sha`); every pair reported `id_match=true`. Same-ctx cells share a sha: ctx256 `902e12e22d1d8367`, ctx512 `a6fad159dfa90047`, ctx1024 `445cb2aab50cf666`, ctx2048 `9115b5280c1d7e33`.
- Env knob semantics (`overlay/mlx/backend/omarchy/primitives.cpp`): `PV=0` forces the plain 128-key split pass 1, `PV=1` forces the value-phase variant; `BK64=1` selects 64-key blocks only when `pv && split_group==4`; `DECODE_SPLIT=1` forces the split route (a no-op wherever the default already engages).

## Cell matrix (env vs env, exact arms)

| cell | env_a | env_b | A config | B config |
|---|---|---|---|---|
| pv_ctxN | `PV=0` | `PV=1` | split, plain pass 1 | split, PV variant |
| bk64_ctxN | `BK64=0` | `BK64=1` | split+PV (pv default-on ≤2048, 128-key) | split+PV, 64-key blocks |
| bk64pv_ctxN | `{}` (default) | `BK64=1,PV=1` | split+PV at ≤2048 | split+PV, 64-key blocks |
| splitall_ctxN | `{}` (default) | `DECODE_SPLIT=1,PV=1,BK64=1` | default route | split+PV, 64-key blocks |

At ctx512 `bk64_ctx512`, `bk64pv_ctx512`, and `splitall_ctx512` all force the identical B-side kernel (split + PV + 64-key blocks) and their A sides all run split+PV (bk64 was off by default): three independent measurements of the same A/B.

## 5-pair confirms (all id_match, per-pair detail in `pairs.md`)

| cell | A→B median ms | gain (tok/s) | wins |
|---|---|---|---|
| pv_ctx512 | 14.23 → 13.50 | +5.37% | 5/5 |
| pv_ctx1024 | 14.36 → 13.69 | +4.80% | 5/5 |
| bk64_ctx512 | 13.54 → 13.10 | +3.25% | 5/5 |
| splitall_ctx256 | 14.09 → 12.90 | +9.32% | 5/5 |
| splitall_ctx512 | 13.47 → 13.04 | +3.29% | 5/5 |
| bk64pv_ctx512 | 15.19 → 17.00 | +0.60% | 3/5 (noisy, see below) |
| pv_ctx256 | 14.09 → 14.12 | −0.30% | 2/5 (did not confirm) |

The `bk64pv_ctx512` confirm is the outlier of the three same-config cells: its per-pair deltas swing −20.6% to +9.2% with pair-2 A/B both ~21 ms (a ~50% whole-machine slowdown mid-run), while `bk64_ctx512` (+3.25%) and `splitall_ctx512` (+3.29%) and the earlier `bk64pv_ctx512` screen (+3.17%, 3/3) agree tightly. Treated as a corrupted run, not a regression signal.

## 3-pair screen (16 cells; losers set the ceilings)

| cell | gain | wins | note |
|---|---|---|---|
| splitall_ctx256 | +14.75% | 3/3 | → confirmed +9.32% 5/5 |
| pv_ctx256 | +8.42% | 2/3 | noisy; → confirm −0.30% 2/5, NOT defaulted |
| bk64_ctx512 | +3.49% | 3/3 | → confirmed +3.25% 5/5 |
| bk64pv_ctx512 | +3.17% | 3/3 | → noisy confirm +0.60% 3/5 |
| splitall_ctx512 | +3.10% | 3/3 | → confirmed +3.29% 5/5 |
| pv_ctx512 | +5.02% | 3/3 | → confirmed +5.37% 5/5 |
| pv_ctx1024 | +4.96% | 3/3 | → confirmed +4.80% 5/5 |
| bk64_ctx256 | −0.06% | 0/3 | wash |
| bk64pv_ctx256 | +0.18% | 2/3 | wash |
| bk64_ctx1024 | −4.52% | 0/3 | ceiling: bk64 off above ctx512 |
| bk64_ctx2048 | −0.13% | 1/3 | wash at 2048 |
| bk64pv_ctx1024 | −4.42% | 0/3 | loses |
| bk64pv_ctx2048 | −8.08% | 0/3 | loses |
| pv_ctx2048 | −8.71% | 0/3 | ceiling: PV off above ctx1024 |
| splitall_ctx1024 | −4.60% | 0/3 | loses (bk64 leg) |
| splitall_ctx2048 | −7.80% | 0/3 | loses |

## Defaults derived

- `sdpa_split_min_keys` 320 → **256**: splitall wins at ctx256 (+9.32%, k_len 264–328); nothing measured below.
- `sdpa_split_pv_max_keys` 2048 → **1152**: PV confirmed at ctx512/1024 (+5.37%/+4.80%, k_len ≤ ~1096) and loses at ctx2048 (−8.71%); 9 × 128 blocks.
- `sdpa_split_bk64_max_keys` 0 → **640**: BK64 confirmed at ctx512 (+3.25%, k_len ~520–584) and loses at ctx1024 (−4.52%); 10 × 64 blocks.
- `sdpa_split_wide_min_keys` pinned at **320**: head width 256 is unmeasured on G13C; keeps the previous effective threshold rather than inheriting the new `min_keys`.
