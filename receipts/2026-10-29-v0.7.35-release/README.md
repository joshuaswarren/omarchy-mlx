# v0.7.35 release receipt

Tag `v0.7.35` at `42b30a2de2f8b42e673cc50eaca9f7b145f04ac6` (main, CI green at that commit). Published 2026-10-30 00:31:05Z, latest, not a prerelease.

## Assets

| File | Bytes | sha256 |
| --- | --- | --- |
| `mlx_omarchy-0.32.4.dev202610292153+42b30a2d-cp311-cp311-linux_x86_64.whl` | 10538243 | `1a946f9dc1225e13da336bf7e3b4817f86d937229d3d8aa100c34a25bd4baab1` |
| `mlx_omarchy-0.32.4.dev202610292155+42b30a2d-cp314-cp314-linux_aarch64.whl` | 422425505 | `39671c8a1e34c52cb2d575514e3b6bb00b2d1d8bbd2b21ae2f5dfad9c19fba82` |
| `omarchy-mlx-wheel-cp314-aarch64.whl` (same bytes) | 422425505 | `39671c8a1e34c52cb2d575514e3b6bb00b2d1d8bbd2b21ae2f5dfad9c19fba82` |
| `omarchy-mlx-vendor-wheels-v0.7.35-cp314-aarch64.tar` | 467558400 | `568b312cb297f951200a2d341ede04ec3641ccab2fc3732f8ab7cbb50d1cc1a7` |
| `omarchy-mlx-vendor-wheels-v0.7.35-cp314-aarch64.tar.sha256` | 118 | `57bd358daabcbbee350ae10dfb1cb50f8e8bb043859e472883881aa4bdf13877` |
| `SHA256SUMS` (covers the other five) | 624 | `4e19a32808794893c6be3adb88e31c543b65e0c382980e8ad6da8706d9847eda` |

The x86_64 wheel was built on the x86_64 development machine. The aarch64 wheel and the vendored wheel set were built on a 16-inch M1 Max from a detached checkout of the tag commit (`git status` empty, build log stamped `42b30a2d`). `python3 scripts/check-version-stability.py .work/mlx` passed for the x86_64 build.

`python3 scripts/verify-release-assets.py v0.7.35` printed `VERIFIED: every uploaded asset matches what the release claims` on the draft (twice, the second time after the notes were edited) and on the published release.

## Gates

The draft assets were the input of every gate.

| Gate | Host | Result |
| --- | --- | --- |
| g1 clean install, g2 online 9B, g3 online 4B card, g4 offline, g5 Laya, g6 codec, g7a packaged ICD, g7b system install, g8 Kokoro, g9 read-aloud, g10 Kokoro primer, g11 card 9B, g12 Kokoro stream, g14 routing | 14-inch M2 Max, stock kernel 7.1.12-2-12.6-sep-ARCH | all exit 0 (about 5.5 minutes) |
| g7c packaged ANE worker verify, g7d fresh-image transcribe | 13-inch M1 | both exit 0; g7d golden transcript match (`db501a8c...`), `ane_mode` true, `cpu_tensor_events` 0, every check passed |
| g13 GDN coopmat correctness | 16-inch M1 Max | 6 of 6 cases, 1394 of 1394 assertions |
| g15 trig contract | 16-inch M1 Max | 7 of 7 cases (128 assertions) and 4 of 4 cases (49 assertions) |

Details of the M2 run: the fresh Hugging Face cache held 6,824,246,720 bytes after g2 and 9,103,542,663 bytes after g3. g3 and g11 passed the card check. g4 ran with outbound traffic denied. g14 passed the routing-on and the kill-switch legs.

g13 and g15 ran on the 16-inch M1 Max, not on the 13-inch M1 as for v0.7.34: the 13-inch M1 had about 6 GB of free disk and the wheel build tree that g13 and g15 compile in needs about 2 GB on top of the assets. The test binaries were built from the tag commit on the 16-inch M1 Max (`omarchy_gdn_maskless_correctness_tests`, `omarchy_eq_math_tests`, `omarchy_trig_reduction_tests`). The case and assertion counts equal the v0.7.34 run.

Attempts before these results. All failures came from my harness, not from the release. The logs of the first two attempts were not kept; the third is in the run's own output:

- First M2 battery run: the wrapper took the GPU lock with `flock` inside a queue ticket that already held it, waited its 600 second limit and exited 1 before any gate ran (no gate log, no download). The second run removed the nested lock and passed.
- First g13 and g15 run: the queue admitted it before the test binaries existed (a reboot of the host had killed the detached build), so both gates refused ("binary not built"). The configure step then failed with `BLAS_INCLUDE_DIRS NOTFOUND` until `CMAKE_INCLUDE_PATH=/usr/include/openblas` was set, as in the v0.7.34 run on the other host. Both gates passed on the next run.
- The pinned-decode wrapper called `mlx_provenance.py` with `--wheel`, which that script does not accept (`--expect-wheel`). The call printed its usage and the run continued; the provenance check that counts is the one inside `bench_decode.py`, which printed `verified=match` on every run.

## Installed decode on the uploaded wheel

The uploaded aarch64 wheel (sha256 `39671c8a...`, also the copy inside the vendored wheel set) was installed into a fresh venv on a 14-inch M2 Max (stock kernel) with `pip install --no-index --find-links` on the vendored wheel set, from the draft bytes.

Qwen3.8-2B 4-bit (`SiddhJagani/Qwen3.8-2B-mlx-4Bit@0867d98b`), greedy (`--temp 0`), seed 0, 128 tokens requested with EOS suppressed, `MLX_DISABLE_COMPILE=1`:

| Run | decode tok/s over 127 tokens |
| --- | --- |
| 1 | 72.26 |
| 2 | 72.03 |
| 3 | 72.00 |
| 4 | 72.17 |
| 5 | 72.16 |

Median 72.16. Generated token ids identical in all five runs (`sha256:775bf034e185c8f9`, the digest the v0.7.34 run gave on a 13-inch M1). Prefill 0.196 s, reported separately.

## Known cost carried by this release

Batch 2 on a 14-inch M2 Max takes 5.6 percent longer per step than with the previous kernel (29.1 ms against 27.6 ms, 0.947x speed; `receipts/2026-10-29-qmv-batch-hoist/`). The cause is not isolated; issue 304 tracks it.
