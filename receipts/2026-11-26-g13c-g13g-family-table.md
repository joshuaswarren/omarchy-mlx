# G13C vs G13G per-kernel family table - jw16 + jwm1 captures, night of 2026-11-26

Main's SUP-142. Sources: jw16 = wB1's SUP-5 5-pair ABBA (`scratch/main/sup5-kcont-lever-g13c.md`,
wheel 0.32.4.dev202611260056, boot ca3505a3, Qwen3-4B-Instruct-2507-4bit ctx512, 64 greedy
tokens, ids sha a6fad159dfa90047 identical on all 20 legs); jwm1 = w7P's par1-matched-706bbe28
ABBA set (33 g13g-gemv captures, boots fb9ebb8c + 815d8eb4, matched wheel
0.32.4.dev202611241944+61b0bff3 ON vs shipped OFF, Qwen3-4B/2B, 128 greedy tokens, token
digests match per ABBA). ms/token = 1000 / decode_tps, rounded to one decimal.

## ms/token per family per chip

### jw16 (16-inch, M1 Max, ctx512, 4B)

| family | kernel | ms/token | tok/s | vs G13G 2x8w2 |
|---|---|---|---|---|
| G13C | 2x16kcont | 54.1 | 18.50 | -46.2% |
| G13G | 2x16 (R2S16 tile) | 30.4 | 32.93 | -4.2% |
| G13G | 2x8w2 (shipping default) | 29.1 | 34.37 | baseline |

### jwm1 (13-inch, M1, Qwen3-4B, G13G gemv only - no G13C capture tonight)

| ctx | ON (candidate wheel) ms/token | OFF (shipped wheel) ms/token | ON vs OFF |
|---|---|---|---|
| 512 | 43.9 | 47.7 | +8.6% tps |
| 1024 | 45.9 | 51.1 | +11.3% tps |
| 1536 | 47.9 | 56.1 | +17.0% tps |
| 2304 | 49.9 | 63.9 | +28.0% tps |
| 2560 | 51.4 | 64.3 | +25.0% tps |

### jwm1 (13-inch, Qwen3-2B, G13G gemv, same ABBA)

| ctx | ON ms/token | OFF ms/token | ON vs OFF |
|---|---|---|---|
| 1920 | 35.4 | 35.8 | +1.2% tps |
| 2304 | 35.0 | 36.1 | +3.3% tps |
| 4096 | 36.1 | 38.0 | +5.1% tps |

Boot 815d8eb4 (post-restart) reproduces the fb9ebb8c numbers at 2B ctx1920
(28.42 vs 27.90 tps) - the wheel effect survived tonight's restarts.

## Levers: shared vs chip-specific

1. KCONT lever (contiguous-K load, the G13C family): measured on jw16 ONLY
   tonight. It loses -46.2% against the shipping G13G 2x8w2 on the M1 Max; the
   loss decomposes as -43.8% from the contiguous-K load swap itself and only
   -4.2% from the 2x16 tile. The wAZ ids-gate passed (bit-identity holds at
   K>=1024), so parity is fine and the lever is simply slower here. CHIP-SPECIFIC
   SO FAR: no G13C capture on jwm1 tonight, and wB1's earlier m6 9B divergence
   was shape/device-dependent. Shared recommendation: the knob stays off on the
   M1 Max; the 13-inch needs its own A/B before any generalization.
2. Matched-wheel lever (dev202611241944+61b0bff3 vs shipped): measured on jwm1
   ONLY tonight, G13G gemv, and it WINS everywhere: 4B +8.6% at ctx512 growing
   to +28% at ctx2304, 2B +1.2-5.1%. The gain grows with context on 4B. CHIP
   COVERAGE: unmeasured on jw16 tonight (SUP-5's wheel was a different dev
   build) - treat as jwm1-positive, jw16-open.
3. Tile lever (2x16 vs 2x8w2, inside G13G): jw16-only tonight, -4.2%; 2x8w2
   stays the shipping choice. jwm1 unmeasured.
4. Cross-family observation: G13C was never competitive on the one chip where
   it was measured tonight, and the constant-fail-band incident on the M2 (PR
   joshuaswarren/omarchy-ane#486) was a harness golden bug, not a family effect;
   no G13C-vs-G13G claim exists for the M2 from tonight.

## Coverage gaps (stated, not hidden)

- jwm1 G13C: unmeasured tonight (all 33 captures are g13g-gemv).
- jw16 matched-wheel lever: unmeasured tonight.
- jwm1 tile lever (2x16 vs 2x8w2): unmeasured tonight.
- ms/token here are decode-only decode_tps reciprocals; prefill and time-to-
  first-token are in the source receipts and are not folded into this table.
