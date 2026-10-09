# Render MiniMax-H3 video on Linux (Apple Silicon, Vulkan)

This guide renders a MiniMax-H3 clip end to end on Linux on Apple Silicon, using the omarchy-mlx Vulkan backend (Honeykrisp), the TensorFold engine (H3 family) with a small set of local patches, and the mrbizarro minimax-h3-mlx port.

Model bundle (int8 state, compact text-tower export, video VAE, audio VAE, tokenizer, processor, configs, sample clip, license): https://huggingface.co/joshuaswarren/MiniMax-H3-int8-omarchy

## Hardware you need

- 96 GB of unified memory. On a 96 GB M2 Max the render peaks at 33.9 GiB of omarchy-mlx allocations, with the 44 GB int8 DiT resident on the Honeykrisp heap (the default heap budget on 96 GB machines is about 47 to 60 GiB).
- 64 GB machines can't run it. On a 64 GB M1 Max, loading the 50.3 GB compact text-tower export fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.

## Software you need

- Omarchy Linux on Apple Silicon, kernel `linux-aurora` with the Honeykrisp Vulkan ICD. Install via Install > AI > MLX + Core ML (the `omarchy-mac-ml` meta package pulls in `omarchy-mlx` and `omarchy-mlx-vulkan`); the system Mesa package stays installed for the desktop. See `docs/install-omarchy.md` for the stack and the Honeykrisp ICD.
- An omarchy-mlx wheel that ships the native `mx.fast.int8_matmul` op (introduced in the v0.7.27 series; the sample clip was rendered on the dev wheels `0.32.4.dev202610090236+53bc1e3` and `0.32.4.dev202610090724+95e7b6f`).
- TensorFold pinned to commit `ea9b63728b690e511722a18ace3b43521a750789` (the drowzeys fork of the engine, TensorFold 0.6.5), plus the local patch set under `packaging/tensorfold-linux/patches/`.
- minimax-h3-mlx at commit `79190205258454b43e6c9e50e577de234222419c` (the mrbizarro fork), the MLX (Apple Silicon) port of MiniMax-H3, supplying the compact text-tower export loader, the audio VAE, and the media writer.

## What is in the HF repo

| path | what it is | size | sha256 |
|---|---|---|---|
| `int8-dit/` | DiT quantized to per-channel int8: 7 base shards + 5 delta shards + per-shard sha256 manifests + `transformer-config.json` | 44 GB | per-shard in `int8-dit/base-manifest.json` and `delta-manifest.json` |
| `text_encoder/` | compact text-tower export: byte-exact key-subset copy of the released text encoder (Qwen3-VL-32B text layers 0-49, 552 tensors) + configs | 50.3 GB | `SHA256SUMS` |
| `video_vae/` | video VAE weights | 9.8 GB | `SHA256SUMS` |
| `audio_vae/` | audio VAE weights | 578 MB | `SHA256SUMS` |
| `tokenizer/`, `processor/`, `model_index.json` | tokenizer, processor, and pipeline index | small | `SHA256SUMS` |
| `LICENSE`, `NOTICE`, `LICENSES/` | license and attribution files | small | n/a |
| `sample/demob-linux-full-768x448-s1.mp4` | the sample clip rendered entirely on Linux on an M2 Max | 526 KB | `36bd5df0a7b4b060304944beeeebb28e89de75ede22b1eaf3e31ce165fafee07` |

Download the bundle:

```bash
hf download joshuaswarren/MiniMax-H3-int8-omarchy --local-dir hf-bundle
```

## Install the engine

Clone and patch the engine, then install it editable. No other GitHub PR or issue is opened anywhere as part of this work.

```bash
# The render driver lives in this repo; clone it so the script paths resolve.
git clone https://github.com/joshuaswarren/omarchy-mlx
cd omarchy-mlx
git checkout origin/main

# The H3 engine itself, with the six local patches.
git clone https://github.com/drowzeys/TensorFold
cd TensorFold
git checkout ea9b63728b690e511722a18ace3b43521a750789
# The patches directory holds two unrelated series: the older 01- and 02- pair targets a
# different upstream; the six 0010-0060 files target this drowzeys pin. Pin the glob:
for p in /path/to/omarchy-mlx/packaging/tensorfold-linux/patches/{0010,0020,0030,0040,0050,0060}-*-*.patch; do
  patch -p1 < "$p"
done
pip install -e .
pip install numpy tqdm safetensors requests pillow huggingface_hub mlx-vlm
```

The patch set:

- `0010-kernel-gate-linux-custom-kernels.patch` adds a `custom_kernels()` probe and routes the `mx.metal.is_available()` gates through it, so the engine's custom-kernel paths activate on the Honeykrisp backend.
- `0020-h3-int8-native-dispatch.patch` extends the H3 int8 kernel (`mlp_int8`, `qkv_int8`) with native dispatch to `mx.fast.int8_matmul` and the `from_state` classmethods that wrap the already-quantized state without re-quantizing.
- `0030-h3-dit-lazy-adaln-dtype-pin.patch` pins the SDPA q/k to the v dtype in `Attention.mix` and adds the `lazy_tables` mode to the DiT, which projects the per-block AdaLN tables on demand (saves about 24 GiB of heap).
- `0040-h3-int8-state-loader.patch` adds `load_int8_dit` to assemble the DiT from the saved int8 shards with a strict, from-state per-block load and the 8-bit AdaLN requantization.
- `0050-h3-sampler-resume-checkpoints.patch` adds per-step safetensors checkpoints and `start_index`-driven resume to the sampler, plus the `check_noise` gating on resume.
- `0060-vae-video-query-chunked-sdpa.patch` chunks the video VAE's SDPA over 1024-row query blocks (the unchunked dispatch binds 3.1 GiB of scores per call, over Vulkan's 2 GiB `maxStorageBufferRange`); the per-row softmax keeps each block bit-identical to the single call.

## Install the omarchy-mlx wheel

The engine's int8 path needs the native `mx.fast.int8_matmul` op. Install the wheel last so the `pip install -e .` step (and any later `pip install mlx`) cannot overwrite it; `pip install` does not detect the override because both packages share the `mlx` import name.

```bash
cd /path/to/omarchy-mlx
WHEEL_URL=https://github.com/joshuaswarren/omarchy-mlx/releases/download/tensorfold-video-wheel-95e7b6f/mlx_omarchy-0.32.4.dev202610090724%2B95e7b6f-cp314-cp314-linux_aarch64.whl
WHEEL_SHA=4dcae17fc998256018623dc73e65e8e82492865525f542905e3f93f2ddd3d291
curl -L -o /tmp/omarchy-mlx.whl "$WHEEL_URL"
echo "$WHEEL_SHA  /tmp/omarchy-mlx.whl" | sha256sum -c -
pip install --force-reinstall --no-deps /tmp/omarchy-mlx.whl
python -c "import mlx.core as mx; assert hasattr(mx.fast, 'int8_matmul'), 'omarchy-mlx wheel was overwritten by PyPI mlx; reinstall with --force-reinstall --no-deps'"
```

The python one-liner is mandatory. If it raises `AssertionError`, the wheel was overwritten; the render will then fall back to the MSL bf16 path and fail at the first `mx.eval` after a forward.

## Disk and memory required

The render needs roughly 160 GB free on the target filesystem before `hf download` starts: the HF bundle is 101 GB and the HF xet reconstruction layer writes the file in halves before assembling the final safetensors, peaking at roughly 60 GB of cache for the 50 GB text_encoder-compact.safetensors alone. Per-step denoise checkpoints add another ~1 GB at the directory passed to `--checkpoint-dir`.

## Install minimax-h3-mlx

The mrbizarro fork supplies the compact text-tower export loader, the audio VAE, and the media writer. It is not a pip package (no `pyproject.toml`); use a checkout on the Python path.

```bash
git clone https://github.com/mrbizarro/minimax-h3-mlx
cd minimax-h3-mlx
git checkout 79190205258454b43e6c9e50e577de234222419c
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Render

The render driver is `scripts/tensorfold/h3_generate_rows.py` in this repository. It derives from the engine's `tools/h3_generate_dev.py` at the pinned commit (Apache 2.0, drowzeys); the local changes are limited to the documentation and command-line defaults.

Before the three render stages, point the driver at the int8 DiT location. The HF bundle ships the int8 shards and the DiT config under `int8-dit/` (the config is `int8-dit/transformer-config.json`); the render driver expects a `transformer/config.json` inside the model directory:

```bash
mkdir -p "$MODEL_DIR/transformer"
ln -s "$MODEL_DIR/int8-dit/transformer-config.json" "$MODEL_DIR/transformer/config.json"
```

The three stages of a reproducible run (768x448, 56 frames, 20 sampler steps, seed 1, the prompt below):

```bash
MODEL_DIR=/path/to/hf-bundle
OUT=clip.mp4
CKPT=ckpt
LAT=clip.latents.safetensors
TEXT=clip.text.safetensors

# Run from the omarchy-mlx checkout so `scripts/tensorfold/h3_generate_rows.py` resolves.
cd /path/to/omarchy-mlx

# 1. Text encode (one-time per prompt): writes 35-row text encoder output. The DiT
#    is not loaded for this step; --int8-from-state is not needed here. `-o /dev/null`
#    is required by the argparse even though the script exits before any video write.
python scripts/tensorfold/h3_generate_rows.py "$MODEL_DIR" \
  --prompt "Slow dolly across a dark desk at night: a terminal window on a Hyprland desktop, green text scrolling, rain on the window behind, warm desk lamp glow." \
  --width 768 --height 448 --frames 56 --points 21 --seed 1 \
  --dump-text-rows "$TEXT" \
  -o /dev/null

# 2. Denoise + decode + mux (20 forwards; resumable per-step checkpoints; writes the mp4).
#    The driver always decodes and writes the output file: there is no flag that skips the
#    decode, so denoise and mux are one command. If the run is interrupted, the same
#    command resumes from the newest checkpoint in $CKPT.
mkdir -p "$CKPT"
python scripts/tensorfold/h3_generate_rows.py "$MODEL_DIR" \
  --text-rows "$TEXT" \
  --width 768 --height 448 --frames 56 --points 21 --seed 1 \
  --int8-from-state "$MODEL_DIR/int8-dit" \
  --checkpoint-dir "$CKPT" --resume \
  --dump-latents "$LAT" \
  -o "$OUT"
```

`--dump-latents "$LAT"` also saves the denoised latents next to the mp4. If a rerun only needs the mp4 from an existing latents file, pass `--latents-in "$LAT"` and drop the `--text-rows`, `--int8-from-state`, `--checkpoint-dir`, and `--resume` flags.

The packaged `sample/demob-linux-full-768x448-s1.mp4` is the output of stages 1 and 2 on the M2 with the same prompt and seed.

## What you should see

Per-stage wall times on the 96 GB M2 Max (Linux, Honeykrisp Vulkan):

| stage | seed 1 wheel `53bc1e3` | seed 1 wheel `95e7b6f` (swiglu coop) | seed 0 wheel `95e7b6f` (swiglu coop) |
|---|---|---|---|
| text encode (35 rows, 50.3 GB load) | 17 s | 17 s | 16 s |
| int8 DiT load (44 GB, 12 shards) | < 1 s (page cache) | < 1 s (page cache) | < 1 s (page cache) |
| denoise (20 forwards) | 5195.7 s (258.8 s/step) | 3399.0 s (168.9 s/step) | 3398.8 s (168.9 s/step) |
| video VAE decode | 190.0 s | 90.2 s | 90.3 s |
| audio VAE decode + mux | 5.2 s | 5.1 s | 5.1 s |
| **total** | **90.3 min** | **58.7 min** | **63.2 min** |

- omarchy-mlx allocator peak: 33.9 GiB.
- The mp4 sha256 was identical across the two wheels on the M2 (the int8 kernels are exact through the whole pipeline): the `clip.mp4` in the HF sample is `36bd5df0a7b4b060304944beeeebb28e89de75ede22b1eaf3e31ce165fafee07`, and the same seed-1 render on wheels `53bc1e3` and `95e7b6f` produced the same hash. The sha holds across the two wheels on the same chip.
- This holds only on the same chip family (M2 Max / G14C) with the same seed, because the same-host rerun of the sampler is deterministic (the latents themselves were reproduced bit-identically on the M2). Other chips will produce a visually equivalent clip with a different hash; judge those by per-frame relative error against the sample, not by sha256.

## One-time preprocessing

The HF repo ships the int8 DiT state and the compact text-tower export already prepared, so the render needs neither step. The upstream MiniMax-H3 release has only the bf16 weights; the Omarchy M team made both files for the Linux port:

- The compact text-tower export is a byte-exact key-subset copy of the released `text_encoder/`: it is a CPU-only `safetensors` operation and runs on any platform with no model use.
- The int8 DiT state is a per-output-channel int8 transform of the bf16 transformer shards (weights int8, per-output-channel fp32 scales, group 1024 for the fc2 input, 8-bit AdaLN requantized at load time). The shipped state is 44 GB across 12 shards with per-shard sha256 in `int8-dit/base-manifest.json` and `delta-manifest.json`.

## License and credits

The model files in the HF repo are released under the MiniMax H3 Community License Agreement (see `LICENSE`). The card in the HF repo (`README.md`) lists every notice, attribution, and the territory restriction the agreement places on use, distribution, and display (sections I.3, I.5, V.4, and Exhibit A item 1 are quoted verbatim there).

The engine and ports:

- The engine is [ashhart/TensorFold](https://github.com/ashhart/TensorFold) (Apache 2.0, by Ash Hart / X @ashxhart). The H3 family base is the [drowzeys/TensorFold](https://github.com/drowzeys/TensorFold) fork at commit `ea9b6372` (Apache 2.0; the macOS MLX H3 pipeline our Linux run starts from is by drowzeys / X @u1tra_instinct, [keys-Mac-TensorFold-MiniMax-H3-MLX](https://github.com/drowzeys/keys-Mac-TensorFold-MiniMax-H3-MLX), Apache 2.0; drowzeys proposed the H3 family upstream as `ashhart/TensorFold#384`, which was closed as out of scope for TensorFold's token-lane engine).
- The MLX port that supplies the text-tower export, the audio VAE, and the media writer is [mrbizarro/minimax-h3-mlx](https://github.com/mrbizarro/minimax-h3-mlx) at commit `79190205` (Apache 2.0, X @AIBizarrothe), a fork of the original [PipeNetwork/minimax-h3-mlx](https://github.com/PipeNetwork/minimax-h3-mlx) (Apache 2.0, X @pipenetwork).
- The text encoder is Qwen3-VL-32B (Apache 2.0).
- The Linux port on Apple Silicon, the omarchy-mlx Vulkan backend (Honeykrisp), the int8 kernels, the chunked VAE attention for the 2 GiB storage-buffer limit, and the int8 state are by Joshua Warren and the Omarchy M team.

The sample clip is a machine-generated Output of MiniMax H3.

## Limitations

- The commands above render 768x448, 56 frames, 20 sampler steps.
- The render needs a Honeykrisp Vulkan backend (omarchy-mlx). lavapipe or llvmpipe are not supported.
- The chunked VAE attention makes the decode's per-row softmax identical to a single call; the chunking does not change the result. There is no command-line option to disable it.
- The int8 DiT replaces the bf16 transformer at load time; the bf16 transformer is not in the HF repo.
- The HF repo is the HF-side identity; mirrors and forks are not provided here. The license's territory restriction is the only authorization the agreement gives, and it is quoted in the HF card.
