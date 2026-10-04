# Draft branch descriptions (prepared 2026-10-04; NO PRs opened — owner go required)

## drowzeys/TensorFold (fork @ ea9b6372), branch `omarchy-gate-probe`

Gate custom kernels on a canary probe instead of `mx.metal.is_available()`.
On macOS Metal this answers exactly as before. Other MLX builds route
`mx.fast.metal_kernel` through their own shader path while
`mx.metal.is_available()` stays false; on those the same kernels are enabled
through a one-time canary compile, so family kernels activate without
weakening the Metal path. The M5 tensor-unit detection in `generation()` is
untouched (Metal-specific by design). Six gate sites rewired; all
syntax-checked. Net chain: 7 files, +74/-8 (device.py helper + five gate
files + import lines).

## drowzeys/TensorFold (fork), branch `omarchy-h3-int8` (on top of gate-probe)

H3 int8: run W8A8 through the backend-native op where the build ships one.
MLX builds with `mx.fast.int8_matmul` (the Omarchy Vulkan W8A8 op) answer
`available()` through it and dispatch `int8_linear` / `int8_swiglu` to it with
the same per-row / per-1024-channel activation scales and per-output-channel
weight scales; bias is widened to float32 exactly as the Metal kernel reads
it. macOS Metal keeps the MPP tensor-unit kernels unchanged; `available()`
there still compiles the real kernel.

## keys-Mac-TensorFold-MiniMax-H3-MLX, branch `omarchy-oneshot-gates`

oneshot-setup: platform and memory checks become warnings. The engine's own
gates (`available()`, server startup) already refuse what a host cannot run;
the installer no longer hard-dies on non-macOS or under-128 GB hosts and
reports the measured memory need instead. `bash -n` clean.

## License note

TensorFold is Apache-2.0 (relicensed from MIT at v0.6.0). These patches are
original work; no TensorFold code is copied into the MIT omarchy-mlx repo.

## keys-Mac-TensorFold-MiniMax-H3-MLX, branch `omarchy-text-rows` (added 2026-10-04 late)

`--dump-text-rows` / `--text-rows`: encode the prompt once on a big host
(the 63 GiB encoder never becomes resident on the small one) and
`--dump-latents` / `--latents-in`: denoise on a big host, decode on a small
one. Latents serialize the full `Latents` dataclass (arrays in the
safetensors payload, scalars — including the packed `Layout` — in a JSON
sidecar) and rebuild bit-identically on load. Verified live on the reference
host: `text: 519 rows in 0.0s` with `--text-rows`, latents round trip 7
tensors + sidecar.

## drowzeys/TensorFold (fork), branch `omarchy-h3-int8` (added)

`Int8MLP.from_state`: wrap an already-quantized state saved after a swap on
another host without re-quantizing (bit-identical weights, same
`quantize_weight` code, only the swapping host differs). Enables the
~25.8 GiB int8 DiT checkpoint (saved on the reference host) to run on hosts
that cannot hold the 62 GiB bf16 DiT at swap time.
