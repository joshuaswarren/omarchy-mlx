# TensorFold MiniMax H3 on Omarchy Linux (int8)

This recipe runs the MiniMax H3 video and audio DiT with int8 weights on an
Apple Silicon Linux machine. The software stack is the TensorFold engine, the
minimax-h3-mlx package, and the Omarchy MLX wheel with the Honeykrisp Vulkan
driver. The int8 path uses the Omarchy native `mx.fast.int8_matmul` operator.

## Status

Read this before you spend disk space:

* The pipeline runs end to end on Linux. The output is finite and the run is
  deterministic. The same seed produces the same latents bit for bit.
* Quality against the macOS bf16 reference is NOT validated. The latents
  differ from the macOS bf16 reference and the control run is pending. Use
  the int8 path for engineering work, not for final renders.

## Hardware and disk

* RAM: 94 GB class. The measured peak was 28.32 GiB for a 48-block resume
  ladder. The default heap setting (`HK_SYSMEM`) is enough.
* Disk: 44 GiB for the int8 state, plus the bf16 checkpoint for the swap
  (about 62 GiB). Keep 120 GiB free before you start.
* GPU: Apple M2 Max or M2 Ultra with the Honeykrisp driver from the
  `omarchy-mlx-vulkan` package.

## Install

```sh
# 1. The wheel. Use the newest release tag; the release ships SHA256SUMS.
pip install mlx-omarchy-0.7.31-py3-none-any.whl

# 2. The driver package ships with Omarchy. Check that the device appears.
python3 -c "import mlx.core as mx; print(mx.default_device())"

# 3. The sources. TensorFold (engine), the H3 wrapper, and minimax-h3-mlx.
git clone https://github.com/drowzeys/TensorFold.git
git clone https://github.com/drowzeys/keys-Mac-TensorFold-MiniMax-H3-MLX.git
git clone https://github.com/drowzeys/minimax-h3-mlx.git
export PYTHONPATH=$PWD/minimax-h3-mlx:$PWD/TensorFold
```

## Build the int8 state

The public bf16 checkpoint is `MiniMaxAI/MiniMax-H3` on Hugging Face. The
wrapper repo ships the swap script. It quantizes the weights (per output
channel, max/127) and writes base and delta shards with manifests.

```sh
cd keys-Mac-TensorFold-MiniMax-H3-MLX
bash swap-save-int8.sh   # writes the state shards; see the script header
```

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

Encode the prompt once, then denoise from the int8 state:

```sh
# One-time: text rows (uses the Qwen3VL text encoder from the checkpoint).
python3 h3_generate-rows.py <checkpoint-dir> \
  --prompt-file wrapper/prompts/black-mirror-scene.txt \
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
   bit for bit.
3. The inputs are public (the Hugging Face checkpoint and the prompt file in
   the wrapper repo), so the SHA-256 above is reproducible by anyone with the
   same wheel and driver.

## Known limits

* macOS has no int8 path for this pipeline. The Omarchy native operator is
  Linux only, and the composed fallback needs a working Metal quantized
  matmul compile, which fails on current macOS builds.
* A small heap causes `vkAllocateMemory` failures with the message
  `VK_ERROR_OUT_OF_DEVICE_MEMORY (cache released)`. This was observed when
  non-finite values grew through the network and inflated the allocations.
  Keep the default heap setting and check the finite flags in the report.
* Quality versus the macOS bf16 reference is not validated. See Status.
