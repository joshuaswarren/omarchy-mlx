# Linux-port patches for the TensorFold H3 video engine

This directory contains the local H3 video-engine patch set that lets the MiniMax-H3 video pipeline run on Linux through the omarchy-mlx Vulkan backend (Honeykrisp). The base is the [drowzeys/TensorFold](https://github.com/drowzeys/TensorFold) fork at commit `ea9b63728b690e511722a18ace3b43521a750789` (TensorFold 0.6.5, Apache 2.0).

This directory also carries unrelated TensorFold LLM-engine port material under `01-add-compat-gate.patch` and `02-gate-sites.patch` (and the build helper files `_build_patch.py` and `tensorfold-compat-gate-source.py`); that work is the LLM lane and is not part of the H3 video pipeline. The H3 video patches below are scoped and named `0010-` and later.

## What is in here

- `0010-kernel-gate-linux-custom-kernels.patch` — adds a `custom_kernels()` probe and routes the `mx.metal.is_available()` gates through it, so the engine's custom-kernel paths activate on the Honeykrisp backend.
- `0020-h3-int8-native-dispatch.patch` — extends the H3 int8 kernel (`mlp_int8`, `qkv_int8`) with native dispatch to `mx.fast.int8_matmul` and the `from_state` classmethods that wrap the already-quantized state without re-quantizing.
- `0030-h3-dit-lazy-adaln-dtype-pin.patch` — pins the SDPA q/k to the v dtype in `Attention.mix` and adds the `lazy_tables` mode to the DiT, which projects the per-block AdaLN tables on demand (saves about 24 GiB of heap).
- `0040-h3-int8-state-loader.patch` — adds `load_int8_dit` to assemble the DiT from the saved int8 shards with a strict, from-state per-block load and the 8-bit AdaLN requantization.
- `0050-h3-sampler-resume-checkpoints.patch` — adds per-step safetensors checkpoints and `start_index`-driven resume to the sampler, plus the `check_noise` gating on resume.
- `0060-vae-video-query-chunked-sdpa.patch` — chunks the video VAE's SDPA over 1024-row query blocks (the unchunked dispatch binds 3.1 GiB of scores per call, over Vulkan's 2 GiB `maxStorageBufferRange`); the per-row softmax keeps each block bit-identical to the single call.

## Apply

```bash
git clone https://github.com/drowzeys/TensorFold
cd TensorFold
git checkout ea9b63728b690e511722a18ace3b43521a750789
for p in /path/to/omarchy-mlx/packaging/tensorfold-linux/patches/*.patch; do patch -p1 < "$p"; done
```

The patch set reconstructs the render's local engine tree exactly (verified with a `diff -r` against the M2 extraction, excluding `__pycache__`).

## Upstream

These patches are noted here for review only. No PR, issue, or comment is opened against the drowzeys/TensorFold fork, the ashhart/TensorFold upstream, or any other upstream repository as part of this work. The patches are local modifications of a fork that the upstream TensorFold project declined to merge (`ashhart/TensorFold#384`, closed as out of scope for the engine's token-lane design).

## License

Each modified file inherits the Apache 2.0 license of the upstream drowzeys/TensorFold fork (the LICENSE file at the pinned commit is Apache 2.0 text; a copy of the upstream LICENSE and the LICENSE files of the related projects live under `LICENSES/` in the HF repo at https://huggingface.co/joshuaswarren/MiniMax-H3-int8-omarchy). The drowzeys/TensorFold LICENSE file at commit `ea9b6372` is reproduced here as `LICENSE` in this directory.
