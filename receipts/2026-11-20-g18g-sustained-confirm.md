# G18G (M6) batch-work 40000 — sustained confirm, 2026-11-20

Confirms receipts/2026-11-13-g18g-batch-work (`{"G18G", 40000}`) with a sustained
 guarded A/B on the stock M6 mini Linux boot and adds the raw DRM-submit counts
 that board could not produce before.

## Setup
- M6 mini, Linux boot `008978c9-9a55-4291-98df-219cedc45ff2`, kernel
  `7.1.12-aurora-m6usb-ARCH`, omarchy-mlx **0.7.40-3** (v0.7.40, which predates
  the G18G row — stock therefore runs the 5000 fallback; verified below).
- Mesa mesa-g18 1:26.1.4.git6db3ae9140-27 (`12e05a5250…`), HoneyKrisp ICD.
- Qwen3-4B-Instruct-2507-4bit (snapshot `50d42775…`), batch 1, prompt 35,
  128 generated tokens, greedy; both arms share one private RoPE host-offset
  overlay (the overlay is the control constant; it changes both arms equally).
- Idle-guard supervised, GPU timeouts 0, every receipt rc=0 clean.

## Sustained timing (35+128, three alternating pairs, 128 exact IDs in all arms)
| budget | median tok/s |
|---|---|
| 5000 (forced stock fallback) | 2.0046 |
| 40000 (this table row) | 2.6873 |

40000 is **+34.1%** over 5000 with the overlay held constant. The Nov 13 row's
+48.6% (Qwen3-4B) was measured without the overlay; both point the same way.

## Why 40000 and not higher (35+16 probe, raw trace, 123 steps stock)
- MLX submits/token: 5000 → 28, 40000 → 4, 120000 → 2.
- Raw DRM submit ioctls/token: 5000 → 156, 40000 → 120, 120000 → 117;
  but *all* DRM requests/token rise from 622 at 40000 to 764 at 120000.
  40000 is the last step that pays.

## Stock-fallback identity
At 11 aligned probe steps the stock runtime's MLX submit counts equal the
forced `MLX_OMARCHY_BATCH_WORK=5000` counts exactly — direct evidence the
installed 0.7.40-3 falls through to `kBatchWorkFallback` and has no G18G row.

## Packaging gap
The row landed on main Nov 13; the M6 image still ships 0.7.40-3. Until the M6
package is rebuilt past v0.7.40, the board stays on the 5000 fallback and pays
24 MLX submits and 36 DRM submits per token more than the table intends.

Raw artifacts: CT `~/scratch/w7g-rope128-008978c9/` (guard ids `f3ff0548`,
`e22b0cfc`, `b4a7e1b7` timing; `91d8fc67` probe; `1451bdbb` count run).
