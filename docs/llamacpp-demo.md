# llama.cpp meetup demo: the fast Vulkan driver, built from source

This demo runs the same llama.cpp binary and the same model twice. The only
thing that changes between the two runs is the Vulkan driver. On an Apple M1
the decode speed of a small quantized model jumps by an order of magnitude
when the current driver branch is used. The whole demo needs about 15 minutes
on decent internet, mostly the 4 GB model download, and installs nothing
system-wide.

Two drivers appear in this demo:

- **System Mesa** (what `pacman` ships today). It works, but it is an older
  revision of the Apple-GPU Vulkan driver, before the change that makes
  quantized models fast.
- **Honeykrisp v3 built from source**, with the large-constants change turned
  on by default. The driver lives in a folder under your home directory and is
  selected per command with `VK_DRIVER_FILES`, so your system stays untouched.

## What you need

- An Omarchy Mac, M1 or newer.
- About 12 GB of free disk space and internet access.
- Build tools (one-time):

```bash
sudo pacman -S --needed meson ninja cmake shaderc python-mako python-yaml pkgconf git
```

## Step 1: build the driver, llama.cpp, and the model

One command; the script does the rest and is safe to rerun:

```bash
git clone https://github.com/joshuaswarren/omarchy-mlx && cd omarchy-mlx && bash scripts/llamacpp-demo.sh
```

Run it from wherever you like; the script creates an `llamacpp-demo/`
directory there and puts everything inside it. On an 8-core M1 expect about
2 minutes for the driver and about 3 minutes for llama.cpp (a fanless Air
runs a bit slower). The script prints both timings and ends with the exact
benchmark commands and the model's SHA-256.

## Step 2: benchmark both drivers

From the demo directory the script prints at the end, first the system
driver:

```bash
llama-build/bin/llama-bench -m Qwen_Qwen3.5-9B-IQ2_M.gguf -p 512 -n 64 -ngl 99 -t 8
```

Then ours:

```bash
env VK_DRIVER_FILES="$PWD/mesa-install/share/vulkan/icd.d/"*.json \
  llama-build/bin/llama-bench -m Qwen_Qwen3.5-9B-IQ2_M.gguf -p 512 -n 64 -ngl 99 -t 8
```

Each run prints a table with prompt processing (pp512) and generation (tg64)
tokens per second. Compare the `t/s` columns.

## Measured on a base M1

Measured 2026-10-08, 8-core Apple M1, model `Qwen_Qwen3.5-9B-IQ2_M.gguf`
(3.74 GiB at 2.7 bpw), same llama.cpp build, `-p 512 -n 64 -ngl 99 -t 8`,
one driver per process, runs stable across repeats:

| driver | prefill pp512 | decode tg64 |
|---|---|---|
| system Mesa 26.2.3 | 28.54 tok/s | 0.34 tok/s |
| Honeykrisp v3 66cb84fd431d | 39.77 tok/s | 4.28 tok/s |

Decode is 12.6x faster, prompt processing 1.4x. The model file needs about
4 GB free; close other apps before the demo. On a bigger machine the same
stack does far better still: the same binary ran Qwen3-30B-A3B at 27.4 tok/s
decode on an M1 Max, 17.9x over the system driver, see
[the llama.cpp stack receipt](../receipts/2026-10-08-llamacpp-mesa-stack/README.md).

## Step 3: put it on stage

One command, and the audience watches tokens appear:

```bash
env VK_DRIVER_FILES="$PWD/mesa-install/share/vulkan/icd.d/"*.json \
  llama-build/bin/llama-cli -m Qwen_Qwen3.5-9B-IQ2_M.gguf -ngl 99 -t 8 --temp 0.7
```

That opens an interactive chat running on the GPU (measured 4.3 tok/s
generating on the M1 above). For contrast, open a second terminal and run the
same command without the `env` line to land on the system driver.

## Why the new driver is faster

Quantized models (the IQ2 and IQ1 formats) carry large lookup tables. The old
driver lowering kept those tables in scratch memory that every shader thread
re-read; the new branch moves them into constant memory. In the driver's own
microbenchmark one quantized matrix-vector kernel drops from 2,290 to 102
microseconds, and the quant kernels run 5 to 24 times faster. That single
change is most of the decode gap in the table above.

## Notes

- Model chosen 2026-10-08: Qwen3.5-9B is the current small dense generation,
  its format is supported by the pinned llama.cpp, and any IQ2/IQ1 quantized
  model shows the same driver effect. Swap in another GGUF if you prefer.
- `-t 8` sets CPU threads (nproc on the M1). Keep it the same for both runs;
  only the driver should change.
- Packaged drivers do not have the speedup yet; building the pinned commit
  from source is the supported path for now.
- The `VK_DRIVER_FILES` variable only affects commands where you set it.
  Close the terminal and your system behaves exactly as before.
