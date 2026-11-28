# v0.7.55 release receipt

- Tag: v0.7.55 on d75a3d0a9bf598a5c90c5defd0acb6661739ab5f (main head).
- qualified-mesa-commit in tag: 53ad9af196f5b748cb67b5d847a43dbafa6aef86.

## Wheels

| file | bytes | sha256 |
|---|---|---|
| mlx_omarchy-0.32.4.dev202611272152+d75a3d0a-cp314-cp314-linux_aarch64.whl | 424,887,790 | ec78267506d17f77bd82ac126d54782838c3ff060cabfdaaacccaf5badd66f40 |
| mlx_omarchy-0.32.4.dev202611272153+d75a3d0a9-cp311-cp311-linux_x86_64.whl | 12,778,561 | 262e474c7a12ddbe3c702c958532088bccfe4f94fb0bb2a1b577df58ae24c918 |
| omarchy-mlx-wheel-cp314-aarch64.whl | 424,887,790 | ec78267506d17f77bd82ac126d54782838c3ff060cabfdaaacccaf5badd66f40 |
| omarchy-mlx-vendor-wheels-v0.7.55-cp314-aarch64.tar | 615,628,800 | bbefd96cf12cc1a5da0b5fc07d9a2ca556e22bca7f101abc3e2bee992ba2ad5c |

aarch64 wheel built natively on an Apple M2 Max (G14C B1) aarch64 host
with glslc and the pinned whole-encoder ANE bundle
(MLX_OMARCHY_WHOLE_BUNDLE_DIR, manifest sha256 08769793f8ee...), from a
detached worktree at the tag commit. Vendored wheel set produced on the
same host by packaging/vendor-wheels.sh --wheel: nodeps closure checked,
offline venv built under unshare -rn, 23 voice modules imported offline,
en_core_web_sm loads.

## Gate

`python3 scripts/verify-release-assets.py v0.7.55` against the draft
upload: VERIFIED — every uploaded asset matches what the release claims
(SHA256SUMS exact coverage, both wheels stamp d75a3d0a, feature strings
stable, platform tags agree, fixed-name copy byte-identical).

## Installed-wheel check

install.sh run on the same Apple M2 Max via
MLX_OMARCHY_RELEASE_BASE=file://<draft assets>: matmul OK, venv stamped
0.32.4.dev202611272152+d75a3d0a. ANE unavailable on this host
(no /dev/accel/accel0); GPU-only install.

## Pinned decode

`MLX_DISABLE_COMPILE=1 .local/share/mlx-omarchy/venv/bin/python
bench_decode.py --model mlx-community/Qwen3.5-9B-MLX-4bit
--prompt "What is the capital of France?" --tokens 64
--wheel <aarch64 wheel>`

    decode 70.19 tok/s over 63 tokens (64 requested, EOS suppressed)
    prefill 0.258s (excluded from decode)
    generated_ids sha256:e17044fd1d6f0656
    provenance: mlx-omarchy +d75a3d0a verified=match, libmlx.so
    sha256:c93db8b2da4c1daa

## Note on Apple M1 Max PMP memory DVFS

The 16-inch M1 Max rows in the parity card published with v0.7.55 were
measured with the Apple PMP (`apple_pmp`) memory DVFS driver active,
which the shipped Linux kernel does not yet turn on for M1 Max
(`t6001`).
[`aurora-silicon/linux#3401`](https://github.com/aurora-silicon/linux/pull/3401)
turns it on. Matched decode rates on the same 16-inch M1 Max and Mesa
`53ad9af196f5` (`MLX_OMARCHY_DEFER_TOKENS=0`) without PMP vs with PMP:

- `Qwen3-4B-Instruct-2507-4bit` (`50d42775`) `ctx=512`: `98.48 tok/s`
  without PMP vs `107.87 tok/s` with PMP (`+9.5%` / `−8.7%` without).
- `Qwen3.5-9B-MLX-4bit` (`938d8919`) `ctx=512`: `58.56 tok/s` without
  PMP vs `64.09 tok/s` with PMP (`+9.4%` / `−8.6%` without).
- `Qwen3.5-9B-MLX-4bit` (`938d8919`) `ctx=2048`: `57.74 tok/s` without
  PMP vs `63.49 tok/s` steady with PMP (`+10.0%` steady / `−9.1%`
  without; second consecutive alias leg `60.44 tok/s` under fan ramp).

