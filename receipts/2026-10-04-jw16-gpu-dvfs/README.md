# 2026-10-04 Jw16Dvfs — GPU firmware DVFS DT A/B on jw16 (T6001): base pstate / perf-tgt-utilization vs decode+prefill

Lane: Jw16Dvfs (worker). Host: jw16 (M1 Max T6001), ssh `jw16mbp1-linux`, kernel 7.1.13-3-2-ARCH.
Pre-registered: private notebook `entries/Jw16Dvfs/20261004T002600Z-jw16-gpu-dvfs-dt-ab.md`
(before any host change). Artifacts: `artifacts/Jw16Dvfs/20261004/` + SHA256SUMS.

## Question

Do `apple,perf-base-pstate` (stock 1 = 388.8 MHz) and `apple,perf-tgt-utilization` (stock 85)
on `/soc/gpu@406400000` move jw16's back-to-back decode/prefill cells (the 0.58-0.60x macOS
parity gap axis), or is DVFS null for serving throughput on T6001?

## Facts (read-only phase)

- Live DT: GPU `/soc/gpu@406400000`, compatible `apple,agx-t6001/g13c/g13s`, base pstate 1,
  tgt-util 85, min-sram 790000 µV. OPP (live `opp-table-gpu`): off/388.8/486/648/777.6/972/1296 MHz.
- Driver: `drivers/gpu/drm/asahi/hw/mod.rs:593,624` parses both properties for every SoC
  (`perf_base_pstate` default 1, `perf_max_pstate` = OPP count - 1 = 6) and passes them to the
  firmware — every pre-registered arm is honoured, no per-chip clamp.
- jw16 boot: `update-m1n1` + omarchy-ane-dt swapped board copy
  `/var/lib/omarchy-ane/dtbs/7.1.13-3-2-ARCH/t6001-j316c.dtb` (`omarchy-ane-dt status`:
  ane overlay at /soc/ane@284000000). NO omarchy-mac-boot opt-in consumer on this host —
  experiment path = fdtoverlay onto that copy + `update-m1n1`; product path = this overlay
  package (opt-in `gpu-pstate-t6001`) for omarchy-mac-boot systems.
- Dry run (no install): merged DTB differs from stock by exactly one line
  (`apple,perf-base-pstate 0x01 -> 0x06`), ane node intact (`apple,t6000-ane`).

## Arms and protocol

S0 stock (pre-experiment boot 4b353848) -> P6 -> P4 -> U40 -> U60 -> S1 stock return; one
Linux->Linux reboot per arm (announced), `/var/tmp/JW16_MAINTENANCE` up during each reboot,
hang recovery = macstudio `sudo ~/src/macvdmtool/macvdmtool reboot` (documented
Jw16AneAutosuspend 2026-10-02; executed on jw16 2026-10-02, Linux return verified).
Per arm, one gpuwin window: d64/d128/d256/d512 (`--passes 5`), pf512/pf1024 (`--passes 3`),
idle-gap TTFT probe (H236 method), `scripts/mlx_provenance.py` under the serving venv.
Serving venv and env UNTOUCHED (`MLX_OMARCHY_NORM_APPLE=1 MLX_OMARCHY_GDN_BATCH=1
MLX_OMARCHY_GDN_F16_STATE=0`, DecodeBw W1 footing; recorded d-baselines
108.72/108.56/107.50/102.70 tok/s).

## Decision rule (pre-registered)

Ledger-moving lever iff decode d512 AND prefill medians gain >= +1.0 % vs S0 with bit-exact
digests; +-0.5 % = null, axis closed; between = report only. Boot-to-boot drift bounded by S0 vs S1.

## Results (2026-10-04T00:34-02:22Z, six arms, five Linux->Linux reboots, all announced; per-arm gates uptime>=450 s / load1<0.5 / PSI 0 / provenance verified=match / digests exact)

Medians, tok/s (TTFT) prefill; gap probe = median TTFT ms at idle gap, n=24/gap.

| arm (live DT) | d64 | d128 | d256 | d512 | pf512 | pf1024 | gap100 | gap500 | gap2000 |
|---|---|---|---|---|---|---|---|---|---|
| S0 stock (base 1, util 85) | 109.58 | 108.55 | 107.48 | 102.86 | 1378.94 | 1392.99 | 88.59 | 88.62 | 91.71 |
| P6 (base 6) | 109.20 | 108.66 | 107.38 | 102.72 | 1377.13 | 1400.51 | 84.35 | 83.89 | 86.62 |
| P4 (base 4) | 108.18 | 108.06 | 107.23 | 102.53 | 1381.20 | 1395.52 | 86.19 | 86.71 | 89.03 |
| U40 (util 40) | 108.94 | 108.24 | 101.65* | 102.68 | 1378.87 | 1400.31 | 88.85 | 87.89 | 89.40 |
| U60 (util 60) | 109.17 | 108.55 | 107.30 | 102.94 | 1381.13 | 1396.58 | 89.27 | 88.30 | 92.18 |
| S1 stock return | 108.62 | 108.05 | 107.28 | 102.57 | 1380.81 | 1394.30 | 88.31 | 88.33 | 92.57 |

\* one noisy pass, digest exact. Boot-to-boot drift S0->S1: d64 -0.88 %, d512 -0.29 %, pf +0.1-0.2 %.

- PRIMARY (pre-registered): every variant arm is NULL on the serving cells — d512 deltas
  -0.32..+0.08 %, prefill within +-0.2 % of stock, all greedy digests bit-exact in every arm.
  The DVFS DT axis does not explain any part of the 0.58-0.60x decode parity gap; that gap
  stays dispatch-drain (see receipts 2026-10-03-drainfix / -turnover / -decode-bandwidth).
- SECONDARY: base pstate 6 removes the post-idle TTFT ramp penalty: -4.2/-4.7/-5.1 ms at
  100/500/2000 ms gaps (exceeds the 1-3 ms cross-boot probe variance; S1 reproduces stock).
  P4 recovers roughly half (directional). U40/U60 do nothing: the governor target does not
  change the clock a burst starts from. TGT-UTIL AXIS CLOSED for T6001.
- Restore verified: boot.bin sha256[:16] 6e8f90c895d83d9f = recorded stock, checked after
  rebuild and again on the S1 boot; omarchy-ane dtb copy byte-identical to pristine
  (7b6ac97aa67de564); llm-inference active + finish_reason=length at every gate.

## Recommendation

The DVFS axis is CLOSED for jw16 serving throughput. The only real effect is user-facing:
`gpu-pstate-t6001` (base pstate 6) cuts TTFT after a 100 ms-2 s pause by 4-5 ms at no
measured throughput/digest cost (T8103 energy evidence: none at idle, +26 mW at 1 Hz,
H247). Ship as OPT-IN only (the overlay here), alongside t8103's; the parity ledger does
not move. An idle-power measurement on jw16 would need a new SMC sampler (none exists).

## Deliverable

Opt-in overlay branch `agent/jw16-gpu-dvfs` (`packaging/dt/t6001-gpu-pstate.dts`,
`packaging/build-dtbo.sh`, `tests/test_gpu_pstate_t6001.py`, `docs/gpu-base-pstate-t6001.md`).
Nothing lands as default.
