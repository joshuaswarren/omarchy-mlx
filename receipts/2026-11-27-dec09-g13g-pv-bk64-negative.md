# DEC-09 negative receipt — G13G re-screen of SDPA split PV + BK=64 (w7H, 2026-11-27)

Row: DEC-09 (M1/M2). Question: `sdpa_decode_split_pv_max_keys("G13G") = 0` at
`device.h` predates the BK=64 kernel and the #933-era G13G word route — does
the value-phase PV pass 1 (with or without 64-key blocks) close the final
0.40–0.62 ms/tok gap to macOS on 2B and 4B decode?

**Answer: no.** The `PV = 0` gate on G13G is correct as it stands; the screen
proves a negative receipt for the mainline context lengths.

## Screen

13-inch M1 (G13G, 8-core, `jwm1`, boot `ebea5b2b`), installed wheel
`0.32.4.dev202611271621+aeb462a`, 5-pair interleaved arms, greedy 64 tokens,
`ids` digest per leg. Arms: **A** unset (live default: fused pass 1, no PV),
**B** `MLX_OMARCHY_SDPA_SPLIT_PV=1` (value-phase pass 1, 128-key blocks),
**C** `SPLIT_PV=1 + SPLIT_BK64=1` (value-phase, 64-key blocks).

### Medians (tok/s), `%` = arm vs A

| model | ctx | A | B | B vs A | C | C vs A | ids |
|---|---|---:|---:|---:|---:|---:|---|
| 2B | 128 | 54.61 | 53.996 | −1.12% | 56.44 | **+3.35%** | identical |
| 2B | 512 | 56.44 | 55.45 | −1.76% | 55.43 | −1.78% | identical |
| 2B | 2048 | 55.07 | 53.02 | −3.74% | 52.18 | −5.25% | identical |
| 4B | 128 | 28.41 | 28.76 | +1.24% | 28.89 | **+1.68%** | identical |
| 4B | 512 | 28.29 | 28.30 | +0.01% | 28.36 | +0.22% | identical |
| 4B | 2048 | 25.79 | 25.88 | +0.36% | 25.58 | −0.78% | identical |

Greedy token ids are bit-identical across all 90 legs (2 models × 3 ctx ×
5 pairs × 3 arms) — the variants are exact re-plans, the deltas are pure time.

## Findings

1. **The row's targets fail**: at ctx512 and ctx2048 — the lengths that carry
   the 0.40–0.62 ms/tok gap — PV (B) and PV+BK64 (C) lose or wash on both
   models (−0.78% to −5.25%). The 8-core G13G has neither the SLC nor the
   workgroup count to benefit; the 2026-10-24 pre-BK64 measurement (value
   phase losing above 512 keys on this part) stands confirmed post-BK64.
2. **A narrow short-context window exists but is thin**: at ctx128, C wins
   +3.35% (2B) and +1.68% (4B). k_len there is ~192 keys (3 × 64-key blocks);
   B alone (PV, 128-key blocks) loses at 2B ctx128, so the win requires both
   flags. A `pv/bk64 max_keys ≈ 192` cap on G13G would serve only very short
   prompts — not worth a default change on its own, recorded here as an
   optional micro-window.
3. The `sdpa_decode_split_pv_limit("G13G") = 0` line should stay. No PR.

## Run mechanics

Ticket `1b98b9e0` on the jwm1 GPU guard queue (boot `ebea5b2b`), runner
`~/scratch/w7h-m1d2/m1d2-pv-screen.py` (per-leg subprocess, env-only arm
flip, `HF_HUB_OFFLINE=1`, `MLX_DISABLE_COMPILE=1`), raw jsonl on jwm1
`~/scratch/w7h-m1d2/out/dec09-pv-*.jsonl`. One pre-existing host-env failure
class (status/CLI legs needing the full engine runtime) is documented in the
#796/#804 verdicts and is not touched by this screen.
