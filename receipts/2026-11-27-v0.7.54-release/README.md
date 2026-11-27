# v0.7.54 release receipt

- Tag: v0.7.54 on a6a8333d19bacd53856852ed0a68552351250b55 (main head).
- qualified-mesa-commit in tag: 53ad9af196f5b748cb67b5d847a43dbafa6aef86.

## Wheels

| file | bytes | sha256 |
|---|---|---|
| mlx_omarchy-0.32.4.dev202611271928+a6a8333d-cp314-cp314-linux_aarch64.whl | 424,879,169 | be3089e28172e43172d9237d36b702b190fe93daafbee95c5449d3f9b64db014 |
| mlx_omarchy-0.32.4.dev202611271928+a6a8333d1-cp311-cp311-linux_x86_64.whl | 12,761,229 | 9887cf969ad03f83e97938caa03eb38a47f2f738e294642f43cff7c9cb1f5d72 |
| omarchy-mlx-wheel-cp314-aarch64.whl | 424,879,169 | be3089e28172e43172d9237d36b702b190fe93daafbee95c5449d3f9b64db014 |
| omarchy-mlx-vendor-wheels-v0.7.54-cp314-aarch64.tar | 615,618,560 | 0cfa725b0e83782ccc70e9cf244b7e6732f31b57d2d9a6dcb6b70d7a4df5548c |

aarch64 wheel built natively on an Apple M2 Max (G14C B1) aarch64 host
with glslc and the pinned whole-encoder ANE bundle
(MLX_OMARCHY_WHOLE_BUNDLE_DIR, manifest sha256 08769793f8ee...), from a
detached worktree at the tag commit. Vendored wheel set produced on the
same host by packaging/vendor-wheels.sh: 80 wheels vendored, nodeps
closure checked, offline venv built under unshare -rn, 23 voice modules
imported offline, en_core_web_sm loads.

## Gate

`python3 scripts/verify-release-assets.py v0.7.54` against the draft
upload: VERIFIED — every uploaded asset matches what the release claims
(SHA256SUMS exact coverage, both wheels stamp a6a8333d, feature strings
stable, platform tags agree, fixed-name copy byte-identical).

## Installed-wheel check

install.sh run on the same Apple M2 Max via MLX_OMARCHY_VERSION=v0.7.54
with MLX_OMARCHY_RELEASE_BASE=file://<draft assets>: matmul OK, venv
stamped 0.32.4.dev202611271928+a6a8333d. ANE unavailable on this host
(no /dev/accel/accel0); GPU-only install.

## Pinned decode

`MLX_DISABLE_COMPILE=1 .local/share/mlx-omarchy/venv/bin/python
scripts/bench_decode.py --model mlx-community/Qwen3.5-9B-MLX-4bit
--prompt "The capital of France is" --tokens 64`

    decode 64.56 tok/s over 63 tokens (64 requested, EOS suppressed)
    prefill 0.284s (excluded from decode)
    generated_ids sha256:9049b0a7101188a0
    provenance: mlx-omarchy +a6a8333d verified=match, libmlx.so
    sha256:a5a957f495244577

A first decode read 27.02 tok/s immediately after the install while the
host was still under post-build thermal load; the steady-state rerun
above is the recorded number. Identical generated_ids on all runs.
