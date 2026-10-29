# v0.7.34 release receipt

Tag `v0.7.34` at `5e40ecf953e8b907dcae1ce6caadce6ad62b0e0f` (main). Published 2026-10-29 01:04:41Z, latest, not a prerelease.

## Assets

| File | Bytes | sha256 |
| --- | --- | --- |
| `mlx_omarchy-0.32.4.dev202610282034+5e40ecf9-cp311-cp311-linux_x86_64.whl` | 10537942 | `827137c1f14f52802254029020510f72c7d63f31b7854b3b938f24624d92a697` |
| `mlx_omarchy-0.32.4.dev202610281952+5e40ecf9-cp314-cp314-linux_aarch64.whl` | 422425305 | `98b6d12ead2e28461c1a8591fd6aa73201dfeaa7b38089841dd9308784b712dc` |
| `omarchy-mlx-wheel-cp314-aarch64.whl` (same bytes) | 422425305 | `98b6d12ead2e28461c1a8591fd6aa73201dfeaa7b38089841dd9308784b712dc` |
| `omarchy-mlx-vendor-wheels-v0.7.34-cp314-aarch64.tar` | 467558400 | `89a93417003578225aa5758b9ce7b3884ee782def57f0d407083e6b42c687033` |

`python3 scripts/verify-release-assets.py v0.7.34` printed `VERIFIED: every uploaded asset matches what the release claims` on the draft and again on the published release.

## Gates

The draft assets were the only input. All exit codes 0.

| Gate | Host | Result |
| --- | --- | --- |
| g1 clean install, g2 online 9B, g3 online 4B card, g4 offline, g5 Laya, g6 codec, g7a packaged ICD, g7b system install, g8 Kokoro, g9 read-aloud, g10 Kokoro primer, g11 card 9B, g12 Kokoro stream, g14 routing | 14-inch M2 Max | pass |
| g7c packaged ANE worker verify, g7d fresh-image transcribe | 13-inch M1 | pass |
| g13 GDN coopmat correctness | 13-inch M1 | 6 of 6 cases, 1394 of 1394 assertions |
| g15 trig contract | 13-inch M1 | 7 of 7 cases (128 assertions) and 4 of 4 cases (49 assertions) |

Two failed attempts came before these results, both from the harness and not from the release:

- The first run on the M2 failed g2, g3, g4, g5 and g10 to g14: the raw-mode `flock` wrapper named no lock file (fixed in #298). The second run held the lock and passed.
- The first ANE run built g13 and g15 without the Vulkan headers on the ANE host; the headers are installed there now and the second run passed.

## Installed decode on the uploaded wheel

The aarch64 wheel was installed offline from the vendored wheel set into a fresh venv on a 13-inch M1, then `scripts/bench_decode.py` ran with the wheel provenance check (`verified=match`, libmlx.so `sha256:7ca9b27ebb32c378`).

Qwen3.8-2B 4-bit (`SiddhJagani/Qwen3.8-2B-mlx-4Bit@0867d98b`), greedy, seed 0, 128 tokens requested with EOS suppressed, `MLX_DISABLE_COMPILE=1`:

| Run | decode tok/s over 127 tokens |
| --- | --- |
| 1 | 27.73 |
| 2 | 27.73 |
| 3 | 27.82 |
| 4 | 27.70 |
| 5 | 27.77 |

Median 27.73. Generated token ids identical in all five runs (`sha256:775bf034e185c8f9`). Prefill 0.47 s, reported separately.
