# 2026-09-29 — qmm coopmat FULL_N x32 variants: paired prefill measurement (M1 Max, jw16)

Host: jw16 (M1 Max T6001), Linux 7.1.6-1-1-ARCH, Vulkan Honeykrisp (Mesa
26.3.0-devel, mesa-1 9d949d4, AGX_SIMDMAT on). Model
SiddhJagani/Qwen3.8-2B-mlx-4Bit snapshot 0867d98bfb174b042d88461c0e7c97b86b34b381,
mlx-lm 0.31.3 with the shipped patches. Measured by the Jw16PrefillGap2
worker (private notebook entries/Jw16PrefillGap2/20260929T1439Z); the change
itself is `b612e4f63` (H75), already on main.

## What FULL_N changes

When the GEMM's output width N is a multiple of 32, a `-DFULL_N` qmm
specialization compiles the per-column predication out of the K loop
(`column_ok = true`; every scale/bias/weight load drops its exec-mask
region). Arithmetic and accumulation order are unchanged, so the route is
bit-exact by construction and the dispatch (compute.cpp) selects it only
when `N % 32 == 0` — the model's lm_head (N = 151936) and every MLP GEMM
qualify.

## Paired cells (same window, n=5 processes per cell, boot-B jw16)

| cell | base +b10013f (no FULL_N) | +b612e4f63 | delta |
|---|---|---|---|
| pf512 | 810.0 [804.0-812.2] | 828.3 [824.7-831.1] | +2.3% |
| pf1024 | 910.1 [907.6-911.0] | 930.6 [928.5-933.1] | +2.3% |
| pf2048 | 953.0 [951.7-954.2] | 975.3 [974.9-975.6] | +2.3% |

tok/s medians, min-max across the 5 processes; `ordered_records_sha256`
100a61b62470 in all 30 runs; teacher-forced full-logit digests identical to
the deployed serving wheel at T=512/1024/2048
(`f771c4265f88…`, `ce24f3b4ce42…`, `b8c4e14f8f8a…`) — the FULL_N wheel is
bit-exact end to end.

Module breakdown at T=2048 (ms): lm_head(all T) 465.11 -> 447.75 (-3.7%,
the largest single module in the prefill profile), gdn.mlp 35.23 -> 34.92,
attn.mlp 35.31 -> 34.91. The +2.3% pf gain repeats at every length and the
candidate min-max never overlaps the base min-max at 512/2048, so it clears
the paired-win bar. (The same-window arm that carried the refuted GDN
barrier-swap lever ran -1.1/-1.3/-2.5% — the FULL_N gain is what remains
once that regression is reverted; the module numbers above are from the
clean arm.)

Run-to-run context: same-boot medians drift about +-0.5% between windows
(the base arm re-measured 810/910/953 in this window vs 813/910/949 in the
morning serving window), an order of magnitude below the +2.3%.

Follow-ups on the same harness (`d955aebed`): n64 and tile-major-A variants
are bit-exact but slower; they stay off main.

## Verdict

FULL_N stays (it is main). Deploy note: the serving venv
(/var/tmp/v072-venv-fused) still runs `+b10013f`, so production does NOT
have this gain yet; the Jw16PrefillGap2 worker deploys the FULL_N wheel with
a rollback copy at the next GPU window (Main's 2026-09-29 control period
paused GPU work).
