# v0.7.34 release receipt

Tag `v0.7.34` at `5e40ecf953e8b907dcae1ce6caadce6ad62b0e0f` (main). Published 2026-10-29 01:04:41Z, latest, not a prerelease.

## Assets

| File | Bytes | sha256 |
| --- | --- | --- |
| `mlx_omarchy-0.32.4.dev202610282034+5e40ecf9-cp311-cp311-linux_x86_64.whl` | 10537942 | `827137c1f14f52802254029020510f72c7d63f31b7854b3b938f24624d92a697` |
| `mlx_omarchy-0.32.4.dev202610281952+5e40ecf9-cp314-cp314-linux_aarch64.whl` | 422425305 | `98b6d12ead2e28461c1a8591fd6aa73201dfeaa7b38089841dd9308784b712dc` |
| `omarchy-mlx-wheel-cp314-aarch64.whl` (same bytes) | 422425305 | `98b6d12ead2e28461c1a8591fd6aa73201dfeaa7b38089841dd9308784b712dc` |
| `omarchy-mlx-vendor-wheels-v0.7.34-cp314-aarch64.tar` | 467558400 | `89a93417003578225aa5758b9ce7b3884ee782def57f0d407083e6b42c687033` |
| `omarchy-mlx-vendor-wheels-v0.7.34-cp314-aarch64.tar.sha256` | 118 | `429d53ce151c7ec70b7f62e46f93dcfc4d27b4f013ffe08087b069e01fa2f84b` |
| `SHA256SUMS` (covers the other five) | 624 | recorded in the release |

`python3 scripts/verify-release-assets.py v0.7.34` printed `VERIFIED: every uploaded asset matches what the release claims` on the draft and again on the published release.

## Gates

The draft assets were the only input. All exit codes 0.

| Gate | Host | Result |
| --- | --- | --- |
| g1 clean install, g2 online 9B, g3 online 4B card, g4 offline, g5 Laya, g6 codec, g7a packaged ICD, g7b system install, g8 Kokoro, g9 read-aloud, g10 Kokoro primer, g11 card 9B, g12 Kokoro stream, g14 routing | 14-inch M2 Max | pass |
| g7c packaged ANE worker verify, g7d fresh-image transcribe | 13-inch M1 | pass |
| g13 GDN coopmat correctness | 13-inch M1 | 6 of 6 cases, 1394 of 1394 assertions |
| g15 trig contract | 13-inch M1 | 7 of 7 cases (128 assertions) and 4 of 4 cases (49 assertions) |

Attempts before these results. All failures came from the gate harness or from how it was run, not from the release:

- First run on the M2: g2 and g3 failed because the raw-mode `flock` wrapper named no lock file, so `flock` tried to execute `-k` (`flock: failed to execute -k: No such file or directory` in their server logs). g4 then failed with no local model snapshot, because g2 and g3 had not filled the cache, and g5 failed (its log gives no cause). g6 to g9 passed. g10 hit the same `flock` error and I stopped the run there; g11, g12 and g14 have the same wrapper line and were not reached. Fixed in #298. The second run held the lock with `flock -x` and `GPU_TURN_TICKET=1` and passed every local gate.
- First ANE-host stage, run 1: the g13 and g15 build stage returned RC 1 with `REFUSING: build tree ... missing (build the wheel first)`, because the harness's `GATE_WORKTREE` named a tree that did not exist on the ANE host. The second attempt pointed it at the prepared tag tree there.
- First g7c start: RC 137. The run was started by my own clock hold before the ANE host was free, and I killed its `ssh` session about 20 seconds in (SIGKILL), so 137 is that kill, not a timeout or an out-of-memory kill. g7c then ran to completion and passed (RC 0).
- ANE build stage, run 2: g13 and g15 built without the Vulkan headers (cmake: `Could NOT find Vulkan`). I installed the `vulkan-headers` package (1:1.4.363.0-1) on the ANE host, and the g13 and g15 builds found the headers through a local include directory (`CMAKE_PREFIX_PATH`, a local `vksdk/include` tree), not the system package. They passed on the next attempt.

## Installed decode on the uploaded wheel

The uploaded aarch64 wheel (sha256 `98b6d12e...`, also the copy inside the vendored wheel set) was installed into a fresh venv on a 13-inch M1 with `pip install --no-index --find-links` on the vendored wheel set, as the wrapper script (sha256 `c7de6be118b4754fac60e3820bdda91e16f087cb54dc159dd36f4d34dc63ab5d`) does; the decode log itself records the installed version `0.32.4.dev202610281952+5e40ecf9`, not the install command. Then `scripts/bench_decode.py` ran with the wheel provenance check (`verified=match`, libmlx.so `sha256:7ca9b27ebb32c378`).

Qwen3.8-2B 4-bit (`SiddhJagani/Qwen3.8-2B-mlx-4Bit@0867d98b`), greedy (`--temp 0`, the script default), seed 0, 128 tokens requested with EOS suppressed. The wrapper script sets `MLX_DISABLE_COMPILE=1` for the runs, as `scripts/bench_decode.py` documents for its baseline; the log does not print the variable.

| Run | decode tok/s over 127 tokens |
| --- | --- |
| 1 | 27.73 |
| 2 | 27.73 |
| 3 | 27.82 |
| 4 | 27.70 |
| 5 | 27.77 |

Median 27.73. Generated token ids identical in all five runs (`sha256:775bf034e185c8f9`). Prefill 0.47 s, reported separately.
