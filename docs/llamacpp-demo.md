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

## Step 0: update the system (required)

```bash
omarchy update
```

This is not optional. It refreshes the package database (a stale mirror 404s
the build packages) and brings every installed package to the same release.
Installing the build tools on an out-of-date system is a partial upgrade: for
example `llvm` jumps to version 23 while `llvm-libs` stays at 22, and the
driver build then fails with a missing `libLLVM.so.23.1`. The update needs
about 10 GB free. The script in Step 1 checks for pending updates and
refuses to run until the system is current.

If you hold some packages back on purpose (a lab machine, a pinned kernel)
and you know the rest is current enough for the build tools, you can skip
that check with `LLAMACPP_DEMO_SKIP_UPDATE_CHECK=1 bash scripts/llamacpp-demo.sh`.
Use it only then: it is the same partial-upgrade risk the check exists to prevent.

Build tools (one-time). Step 1 checks for them and prints the exact `pacman`
line to run if anything is missing. Run that line only after Step 0 and only
when the script asks for it; it is not needed on a machine that already has
the tools:

```bash
sudo pacman -S --needed meson ninja cmake make bison flex shaderc glslang python-mako python-yaml pkgconf git expat libdrm libelf libunwind zstd zlib llvm spirv-tools spirv-llvm-translator libclc spirv-headers
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

Change into the demo directory the script printed (`cd llamacpp-demo`), then
first the system driver:

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

## Measured on a 16 GB M1 Pro (base M1 GPU)

Measured 2026-10-08, 8-core Apple M1, 16 GB RAM, model
`Qwen_Qwen3.5-9B-IQ2_M.gguf` (3.74 GiB at 2.7 bpw), same llama.cpp build,
`-p 512 -n 64 -ngl 99 -t 8`, one driver per process, runs stable across
repeats:

| driver | prefill pp512 | decode tg64 |
|---|---|---|
| system Mesa 26.2.3 | 28.54 tok/s | 0.34 tok/s |
| Honeykrisp v3 66cb84fd431d | 39.77 tok/s | 4.28 tok/s |

Decode is 12.6x faster, prompt processing 1.4x. The system Mesa baseline
ran with 16 GB available; on an 8 GB machine the system driver can fail to
allocate its compute buffer (542 MB) and refuse to load, while our build
still fits. See the 8 GB section below. On a bigger machine the same stack
does far better still: the same binary ran Qwen3-30B-A3B at 27.4 tok/s
decode on an M1 Max, 17.9x over the system driver, see
[the llama.cpp stack receipt](../receipts/2026-10-08-llamacpp-mesa-stack/README.md).

## Measured on an 8 GB-class machine (ballooned from 16 GB)

Same 8-core Apple M1, memory ballooned to about 4.5 GB MemAvailable
(mimicking an 8 GB Air running a desktop and the model). The 9B model
needs `-ub 256 -b 256` to fit under the old driver's compute buffer; the
4B model fits with the defaults.

| model | driver | prefill pp512 | decode tg64 |
|---|---|---|---|
| Qwen3.5-9B IQ2_M, `-ub 256 -b 256` | system Mesa 26.2.3 | 27.65 tok/s | 0.34 tok/s |
| Qwen3.5-9B IQ2_M, `-ub 256 -b 256` | Honeykrisp v3 66cb84fd431d | 39.17 tok/s | 4.27 tok/s |
| Qwen3.5-4B IQ2_M (1.81 GiB) | system Mesa 26.2.3 | 51.50 tok/s | 0.44 tok/s |
| Qwen3.5-4B IQ2_M (1.81 GiB) | Honeykrisp v3 66cb84fd431d | 73.68 tok/s | 7.42 tok/s |

Use the 4B model if you want a 4 GB model file that fits on 8 GB; use the
9B model only on 16 GB or larger. The system Mesa failure on 8 GB is
because the old driver tries to allocate a single 542 MB compute buffer
up-front; our build allocates per-ubatch (256 tokens here) and survives
the same memory pressure.

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
