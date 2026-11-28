# omarchy-mlx v0.7.55

The fix that puts the last parity cell ahead of macOS — the 13-inch
Apple M1 (G13G) 2B decode at 52.3 tok/s against 49.06 in the matched
test — a blocked-online-softmax decode kernel for the Apple M1 Max
(G13C), eight new `aten` lowerings in torch-mlx-vulkan, and the macOS
parity card. 11 non-merge commits since v0.7.54.

## Decode and kernel speedups

- Apple M1 (G13G): the K=2048 greedy sentinel sync is skipped on the
  word route — the C++ routes G13G to the word path while the Python
  patch matched only G14C — for +8.42% 2B decode (#1138). The 13-inch
  M1 2B cell now measures 52.3 tok/s against 49.06 in the matched
  macOS test (#1140 table rows).
- Apple M1 Max (G13C): HD=128 SdpaDecodeOnlineRope 2-key/4-key blocked
  online softmax, +1.92% ctx=2048 and +0.36% ctx=512 (#1135).

## torch-mlx-vulkan and omarchy_torch

- 8 new `aten` lowerings: fft_fft2/fft_ifft2 (#1129), fft_rfft/fft_irfft
  (#1128), glu (#1126), elu/selu/celu (#1125), histc collapsed range
  and histogram per-bin density widths (#1096).

## Scoreboards and tooling

- docs: macOS parity card — Mesa 53ad9af196f5, decode and prefill
  (#1122); 13-inch M1 parity rows on the #1138 fix (#1140).
- scripts: one-command wheel card-row qualification, Q-a (#1136).

## Note on Apple M1 Max PMP memory DVFS

The 16-inch M1 Max rows in the parity card were measured with the Apple
PMP (`apple_pmp`) memory DVFS driver active, which the shipped Linux
kernel does not yet turn on for M1 Max (`t6001`).
[`aurora-silicon/linux#3401`](https://github.com/aurora-silicon/linux/pull/3401)
is the kernel change that turns it on. Matched decode rates on the same
16-inch M1 Max, same build (`cccad65` + `#1203`), and Mesa `53ad9af196f5`
(`MLX_OMARCHY_DEFER_TOKENS=0`, identical greedy SHA-256) without PMP vs
with PMP:

- `Qwen3-4B-Instruct-2507-4bit` (`50d42775`) `ctx=512`: `98.48 tok/s`
  without PMP vs `107.87 tok/s` with PMP (`+9.5%` / `−8.7%` without;
  `94.2%` of macOS).
- `Qwen3.5-9B-MLX-4bit` (`938d8919`) `ctx=512`: `58.56 tok/s` without
  PMP vs `64.09 tok/s` with PMP (`+9.4%` / `−8.6%` without; `101.4%` of
  macOS).
- `Qwen3.5-9B-MLX-4bit` (`938d8919`) `ctx=2048`: `57.74 tok/s` without
  PMP vs `63.49 tok/s` steady with PMP (`+10.0%` steady / `−9.1%`
  without; `101.2%` of macOS; second consecutive alias leg `60.44 tok/s`
  under fan ramp).

## Cut

Cut commit and assets recorded in the release receipt at tag time.

## Release verification

`verify-release-assets.py v0.7.55`: VERIFIED — every uploaded asset
matches (SHA256SUMS coverage exact, both wheels stamp d75a3d0a). Pinned
4-bit decode on the installed aarch64 wheel, Apple M2 Max:
`decode 70.19 tok/s over 63 tokens` (mlx-community/Qwen3.5-9B-MLX-4bit, bench_decode.py,
MLX_DISABLE_COMPILE=1, ids sha256 e17044fd1d6f0656).
