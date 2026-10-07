# v0.7.31 M2 battery summary — 12.1 evidence set (see the wheel column)

Provenance note (w7K review, 2026-10-07): run-1 (the 09:18–09:30Z battery)
installed the v0.7.30-cut wheel `0.32.4.dev202610070939+cf71ea2` from a stale
`SHA256SUMS`, so EVERY gate that reused its gate-home ran the wrong wheel.
Its raw per-gate logs were not kept (only `gates.done`); the RC column below
is therefore run-1 record only and is NOT v0.7.31 evidence. The v0.7.31
12.1 evidence is ONLY the gates rerun later with the correct wheel
(`0.32.4.dev202610071347+9b5c938`, sha a2f8c83e…) and with logs in this
directory: g1, g7b, g16 legs P+B, g16b legs P+B, g17.

| gate | host | wheel actually used | RC | log kept? |
|---|---|---|---|---|
| g1-clean-install | M2 | run-1: cf71ea2 (wrong wheel); RERUN: 0.32.4.dev202610071347+9b5c938 (WHEEL_IDENTITY match) | run-1 0; rerun 0 | rerun: g1-install.log |
| g2-online-9b | M2 | cf71ea2 (wrong wheel) | 0 (run-1 gates.done only) | NO — not v0.7.31 evidence |
| g3-online-4b-card | M2 | cf71ea2 (wrong wheel) | 0 (run-1 gates.done only) | NO — not v0.7.31 evidence |
| g4-offline | M2 | cf71ea2 (wrong wheel) | 0 (run-1 gates.done only) | NO — not v0.7.31 evidence |
| g5-laya | M2 | cf71ea2 (wrong wheel) | 0 (run-1 gates.done only) | NO — not v0.7.31 evidence |
| g6-codec | M2 | cf71ea2 (wrong wheel) | 0 (run-1 gates.done only) | NO — not v0.7.31 evidence |
| g7a-packaged-icd | M2 | n/a (system-ICD fixture; driver sha e7631595df) | 0 (run-1 gates.done only) | NO — not v0.7.31 evidence |
| g7b-system-install | M2 | run-1: RC=1 (stale vendor tar); RERUN: driver-staged tag tree HEAD=9b5c938fe…, 7/7 staged | rerun 0 | rerun: g7b-system-install.log |
| g8-kokoro | M2 | cf71ea2 (wrong wheel) | 0 (run-1 gates.done only) | NO — not v0.7.31 evidence |
| g9-speak-queue | M2 | cf71ea2 (wrong wheel) | 0 (run-1 gates.done only) | NO — not v0.7.31 evidence |
| g10-kokoro-primer | M2 | cf71ea2 (wrong wheel) | 0 (run-1 gates.done only) | NO — not v0.7.31 evidence |
| g11-card-9b | M2 | cf71ea2 (wrong wheel) | 0 (run-1 gates.done only) | NO — not v0.7.31 evidence |
| g12-kokoro-stream | M2 | cf71ea2 (wrong wheel) | 0 (run-1 gates.done only) | NO — not v0.7.31 evidence |
| g14-routing | M2 | cf71ea2 (wrong wheel) | 0 (run-1 gates.done only) | NO — not v0.7.31 evidence |
| g16-qmm-batch leg P | M2 | 0.32.4.dev202610071347+9b5c938 (rerun) | 0 | rerun: g16-legP-jwm2.log |
| g16-qmm-batch leg B | M2 | 0.32.4.dev202610071347+9b5c938 (rerun) | 0 | rerun: g16-legB-jwm2.log |
| g16b-route-probe leg P | M2 | 0.32.4.dev202610071347+9b5c938 (rerun) | 0 | rerun: g16b-legP-jwm2.log |
| g16b-route-probe leg B | M2 | 0.32.4.dev202610071347+9b5c938 (rerun) | 0 | rerun: g16b-legB-jwm2.log |
| g17-patch-series | M2 | 0.32.4.dev202610071347+9b5c938 (rerun) | 0 | rerun: g17-result.json |

Earlier g16/g16b probes ran on jwm1 (not the M2) and are superseded by the
reruns above; the jwm1 JSONs stay in this directory as jwm1 records.

Run-1 truth: `install.sh` fetched `SHA256SUMS` from
`MLX_OMARCHY_RELEASE_BASE=file://$ASSETS_DIR`; the `SHA256SUMS` in the shared
assets dir was the v0.7.30-era file and named the cf71ea2 wheel. The battery
exported `EXPECTED_WHEEL_SHA256` but nothing consumed it (fixed on main:
env.sh pins assets by sha; g1 asserts WHEEL_IDENTITY; g16/g16b/g7b/g17 were
rerun with the correct wheel and logs).

This file describes the 12.1 (pre-w7J-upgrade) kernel
(`7.1.12-2-12.1-sep-ARCH`). The 12.2 set lives in `m2-aurora-12.2/` with its
own SUMMARY-m2-12.2.md.
