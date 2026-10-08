# TensorFold MiniMax H3 on Omarchy Linux (int8)

This recipe runs the MiniMax H3 video and audio DiT with int8 weights on an
Apple Silicon Linux machine. The software stack is the TensorFold engine, the
minimax-h3-mlx package, and the Omarchy MLX wheel with the Honeykrisp Vulkan
driver. The int8 path uses the Omarchy native `mx.fast.int8_matmul` operator.

## Status and provenance

Read this before you spend disk space:

* The pipeline runs end to end on Linux. The output is finite and the run is
  deterministic. The same seed produces the same latents bit for bit.
* Quality against the macOS bf16 reference is NOT validated. The latents
  differ from the macOS bf16 reference and the control run is pending. Use
  the int8 path for engineering work, not for final renders.
* The H3 engine family (DiT, sampler, int8 weights) and the `rows` driver
  script are NOT in a public repository yet. This recipe runs only with that
  engine tree; its publication is pending. The public pieces today are the
  wrapper repository (Apache-2.0, holds the checkpoint tooling) and the
  `minimax-h3-mlx` package mirror.

## Hardware and disk

* RAM: 94 GB class. The measured peak was 28.32 GiB for a 31-block resume
  probe. The default heap setting (`HK_SYSMEM`) is enough.
* Disk: 44 GiB for the int8 state plus the bf16 checkpoint for the swap
  (measured 62 GiB). Keep 120 GiB free before you start.
* GPU: Apple M2 Max or M2 Ultra. The Honeykrisp driver comes from the
  `omarchy-mlx-vulkan` package; see docs/install-omarchy.md.

## Install

Python 3.14, aarch64. The wheel ships the GPU kernels; the driver package
ships the ICD.

```sh
# 1. The wheel from the release, verified against the release SHA256SUMS.
gh release download v0.7.31 -R joshuaswarren/omarchy-mlx -p '*.whl'
shasum -a 256 -c SHA256SUMS
pip install mlx_omarchy-0.32.4.dev202610071347+9b5c938-cp314-cp314-linux_aarch64.whl

# 2. Check that the device appears.
python3 -c "import mlx.core as mx; print(mx.default_device())"

# 3. The wrapper repository (Apache-2.0): checkpoint tooling and prompts.
git clone https://github.com/drowzeys/keys-Mac-TensorFold-MiniMax-H3-MLX.git
```

## Build the int8 state

The public bf16 checkpoint is `MiniMaxAI/MiniMax-H3` on Hugging Face. The
wrapper repository ships the swap procedure. It quantizes the weights (per
output channel, max/127) and writes base and delta shards with manifests.

The state directory holds `base-manifest.json`, `delta-manifest.json`, the
shards, and `transformer-config.json`. Verify it before a long run:

```sh
python3 - << 'PY'
import hashlib, json
for manifest in ("base-manifest.json", "delta-manifest.json"):
    for entry in json.loads(open(manifest).read())["files"]:
        digest = hashlib.sha256(open(entry["name"], "rb").read()).hexdigest()
        assert digest == entry["sha256"], entry["name"]
print("state OK")
PY
```

## Run

Encode the prompt once, then denoise from the int8 state. The driver script
and the H3 engine family are not published yet (see Status), so these
commands need the unpublished engine tree:

```sh
# One-time: text rows (uses the Qwen3VL text encoder from the checkpoint).
python3 h3_generate-rows.py <checkpoint-dir> \
  --prompt-file prompts/black-mirror-scene.txt \
  --dump-text-rows text-rows.safetensors

# Denoise: points 2 is one forward pass. Seed 0 for the reference.
python3 h3_generate-rows.py <checkpoint-dir> \
  --text-rows text-rows.safetensors \
  --width 768 --height 448 --frames 124 --points 2 --seed 0 \
  --int8-from-state <state-dir> \
  --dump-latents latents.safetensors -o clip.mp4
```

Measured times on an M2 Max (94 GB), from the run logs:

* int8 state load: 37 s.
* One points-2 forward, 124 frames, 48 blocks: 1529 s.
* VAE decode of the latents: 103 s video plus 1.4 s audio.
* A 31-block resume probe with per-stage checks: 13 to 15 min.

The VAE decode needs the `minimax_h3_mlx` package on the PYTHONPATH. A small
machine can dump latents and decode them on another host with `--latents-in`.

## Check success

1. The run prints a report line. Confirm `nan_video: 0` and `nan_audio: 0`.
2. Determinism: run the denoise twice with the same seed. The latents file
   must have the same SHA-256. Two independent runs on one machine matched
   bit for bit. The published latents reference (SHA-256 starting
   `e3166dda`) was measured on the development wheel
   `0.32.4.dev202610061912+3c1c7bc` with the packaged Honeykrisp ICD, seed 0,
   points 2, 768x448x124. Reproducibility on the release wheel is not
   claimed.

## Known limits

* macOS has no int8 path for this pipeline. The Omarchy native operator is
  Linux only, and the composed fallback needs a working Metal quantized
  matmul compile, which fails on current macOS builds.
* A small heap causes `vkAllocateMemory` failures with the message
  `VK_ERROR_OUT_OF_DEVICE_MEMORY (cache released)`. This was observed when
  non-finite values grew through the network and inflated the allocations.
  Keep the default heap setting and check the finite flags in the report.
* Quality versus the macOS bf16 reference is not validated. See Status.
