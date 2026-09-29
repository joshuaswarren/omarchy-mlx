# Install mlx-omarchy

mlx-omarchy is MLX with the Omarchy Vulkan backend. The distribution name is
`mlx-omarchy`. The Python module is `mlx`.

## Supported target

Apple-silicon Honeykrisp GPUs are the supported target. The wheel builds on
Linux only.

The repository installer, `install.sh`, exposes `mlx-omarchy-info` in
`~/.local/bin`. It launches the native capability reporter from the installed
wheel; uninstall removes the launcher. Command presence records installation,
not a working GPU or qualified ANE support. The reporter also prints ANE
visibility (FDT `apple,*-ane` compatible, `/dev/accel/accel0`, and the `ane`
module) and Core ML frontend/cache presence. GPU smoke always runs. ANE smoke
runs only when `/dev/accel/accel0` exists and refuses the install if that
node is present without a matching FDT node and loaded module. The installer
does not install `kmod-ane`.

Every wheel also installs `mlx-omarchy-coreml` and `mlx-omarchy-parakeet`
launchers beside it, with the Core ML runtime modules under
`site-packages/mlx/coreml/`. aarch64 wheels additionally ship the Parakeet
ANE runtime: the standalone `mlx-omarchy-ane-worker` (fd-protocol serve CLI)
in `mlx/bin/`, and the pinned island bundles, strict `libane-strict.so`, and
hash pin manifest under `mlx/share/mlx-omarchy/parakeet-1/`. Other
architectures install the CLIs without the arm64 payloads. The Parakeet
product surface — `download`, `verify`, `transcribe` — is documented in
[docs/parakeet.md](parakeet.md#installed-product-wheel).

## Development override

`MLX_OMARCHY_ALLOW_NON_APPLE=1` allows a desktop or software Vulkan driver,
for example llvmpipe, on a development machine. Do not set it on a supported
machine. Receipts from a development run must record that the device is a
development device, not Honeykrisp.

Compiled tapes run by default on Apple GPUs: the stale-shape
corruption that closed them is root-caused and fixed
([docs/known-defects.md](known-defects.md)). The former
`MLX_OMARCHY_ALLOW_UNSAFE_COMPILE` override was retired with the fix;
setting it now does nothing.
To enable compilation, unset `MLX_DISABLE_COMPILE`; setting it to `0`
still disables compilation because upstream checks its presence.

Development builds use tiled quantized prefill by default. Set
`MLX_OMARCHY_QMM_TILE=0` to compare with the untiled path; single-row
decode still uses GEMV. This change is not in the v0.3.5 wheels.
The experimental `MLX_OMARCHY_ROPE_BF16_DIRECT` and
`MLX_OMARCHY_SDPA_BF16_FAST` flags remain off: both changed generated
token IDs on M1. See the [hardware gate receipt](../receipts/2026-09-04-m1-performance-gates.md).

Compiled-tape elementwise chains and exact eager SwiGLU graphs
(`gate * sigmoid(gate) * up`) fuse into one dispatch by default. The eager
path and compiled-tape chains support f32, f16, and bf16 with per-instruction
rounding to the storage dtype, bit-exact against the per-node path
(see docs/known-defects.md). Set `MLX_OMARCHY_FUSED_CHAIN=0` to use
the per-node path for every dtype.

Eager single-row 4-bit/group-64 quantized projections that read one x
(q/k/v, gate/up) dispatch as one multi-weight GEMV, and the bias or
residual `Add` that is a projection's only consumer is folded into that
GEMV's store: a Qwen2 decode layer drops from 22 dispatches to 14 with
every array still materialized and every value bit-identical to the
per-node path. Eager dense BF16 decode projections that share one
evaluated input (q/k/v or gate/up) dispatch as one grouped GEMV with
independent per-output accumulation, also bit-identical to the per-node
path. Set `MLX_OMARCHY_FUSED_GEMV=0` to keep either grouping on the
per-node path (`MLX_OMARCHY_FUSED_CHAIN=0` disables it too).

Dispatches record dependency-gated barriers by default: the encoder tracks
per open batch which buffer ranges were read or written since the last
barrier and records one barrier only when a dispatch, copy, or fill overlaps
an unsynced range. `MLX_OMARCHY_GATED_BARRIERS=0` restores the historic
unconditional pre+post barriers. Each recorded submission ends with a
device-to-host visibility barrier before its completion signal; waiting and
invalidating host caches do not replace that memory-domain transfer. The mode
became the default on 2026-09-29 after an 8-pair interleaved battery of the
full 10-prompt x 10-pass contract on the M1 Max (16 runs, identical ordered
records digest `dbf70497`, +2.9% decode) on the dependency-tracked Honeykrisp
driver; other SoCs should re-run their own digest gate. Skip and emit counts
appear in the GPU profile and in the runtime-test trace counters.

Submit batching (scheduling only; results are bit-identical): the open batch is
submitted at 4096 nodes or at the byte budget. `MLX_OMARCHY_BATCH_NODES=<n>`
overrides the node budget. `MLX_OMARCHY_BATCH_FIRST=<n>` submits the first batch
of each graph after n nodes so the GPU starts while the host is still recording
the rest of a long graph; host record time is otherwise fully exposed in
synchronous paths (about 45 us per dispatch on the M1). It is read at every graph
start. The mlx-lm patch `mlx-lm-ttft-early-submit.patch` sets it to 128 only
around prompt processing and the first token of `generate_step`, so pipelined
decode keeps one submit per token (a constant early first batch cost decode
about 1%); `MLX_OMARCHY_NO_TTFT_EARLY_SUBMIT=1` disables the patch's behavior.
Measured on jwm1 (T8103): qwen38 protocol TTFT 0.1844 -> 0.1590 s (-13.8%),
decode64 41.62 -> 41.48 tok/s, digests unchanged.

## Build the wheel

1. Install the build tools: Python 3.10 or newer with `venv`, `cmake` 3.25 or
   newer, Vulkan development headers, a C++ compiler, and the BLAS/LAPACK
   development packages the CPU backend links (`liblapack-dev libblas-dev
   liblapacke-dev` on Debian-family distributions).
   On Arch/Omarchy, install `blas-openblas` and set
   `CMAKE_INCLUDE_PATH=/usr/include/openblas` for the build; both BLAS and
   LAPACK headers live in that package-owned directory. No source or linker
   flag changes are needed.
2. Run `./scripts/build-wheel.sh`
3. Read the wheel path, size, and sha256 from the receipt lines.

The script prepares the pinned upstream tree, builds with
`MLX_BUILD_OMARCHY=ON`, the CPU backend on, and the Metal and CUDA backends
off, and writes one wheel into `dist/`. Keep the selected BLAS/LAPACK
provider installed at runtime: `libblas`/`liblapack` on Debian-family
distributions, or `openblas` on Arch/Omarchy.

## Install and smoke-test

1. Run `./tools/ci/run-clean-omarchy-install.sh`
2. Expect `clean install verified` as the last line.

The script creates a fresh venv and runs an import check, an add, a matmul,
and a gradient check. It installs the newest wheel from `dist/` by default;
set `MLX_OMARCHY_WHEEL` to an existing wheel path to test that exact build.
On a non-Apple development host, set `MLX_OMARCHY_ALLOW_NON_APPLE=1`
explicitly before running it. Do not set that override on a supported M1 host.

## Install by hand

1. `python3 -m venv ~/.venvs/mlx-omarchy`
2. `~/.venvs/mlx-omarchy/bin/pip install dist/mlx_omarchy-*.whl`
3. `~/.venvs/mlx-omarchy/bin/python -c 'import mlx.core as mx; print(mx.default_device())'`

Do not install the upstream `mlx` package beside this wheel. The module name
is the same, so the two distributions conflict. Remove upstream `mlx` before
you install `mlx-omarchy`.

## Honeykrisp driver with the fork fixes

Stock Mesa 26.1.7 Honeykrisp has four driver-side defects that this
backend works around in its shaders (data-dependent byte extraction
miscompiles, one-ulp float division, one-ULP `log`, and `sin`/`cos` range
reduction above 1e5), and it does not expose `VK_KHR_cooperative_matrix`.
The fork branch [`honeykrisp-omarchy`](https://github.com/joshuaswarren/mesa-1/tree/honeykrisp-omarchy)
fixes all four in the compiler and turns the G13 8x8x8 matrix unit on by
default, so an unmodified wheel runs dense f32 matmul on cooperative
matrices. Every workaround stays in the shaders for stock Mesa; the fork
only removes the need for them. Driver receipts: the
[integration receipt](../receipts/hk/2026-09-08-honeykrisp-omarchy-integration.json)
(25/25 suites, every reproducer) and the
[package receipt](../receipts/2026-09-08-honeykrisp-package.json).

`packaging/mesa-honeykrisp-omarchy/PKGBUILD` builds the fork as a pacman
package that replaces `mesa`. It is the asahi-alarm `mesa` recipe
(AsahiLinux/PKGBUILDs `28229b8`, the PKGBUILD that produced the installed
`mesa 26.1.7-1`) with the source pointed at fork commit `6f6afc89`, so GL,
EGL, GBM, llvmpipe, zink, rusticl, teflon, and VA are built with the same
options as the stock package. The package is Mesa `26.3.0-devel`; the
desktop runs on Mesa main plus the fork's Asahi changes.

### Build

On the M1 (Omarchy on Asahi Arch, `base-devel` installed), 2 minutes 31 seconds wall time on 8 cores (clean `makepkg -C -f`, 09:19:55–09:22:26 UTC-5):

```sh
mkdir -p ~/src/mesa-pkg && cp packaging/mesa-honeykrisp-omarchy/* ~/src/mesa-pkg/
cd ~/src/mesa-pkg
sudo pacman -Sy            # the makedepends list needs a current package db
makepkg -s --noconfirm     # installs missing makedepends, clones the fork, builds
ls mesa-honeykrisp-omarchy-*.pkg.tar.xz
```

The source is a git clone pinned to the commit, not a tarball, because
Mesa derives the `git-<sha>` in `driverInfo` from the checkout; that
string is how you tell the fork from stock later.

### Install

Keep the stock package for rollback (pacman already has it in
`/var/cache/pacman/pkg/`), then replace the conflicting `mesa` package in
one interactive pacman transaction. Confirm the `Remove mesa?` prompt:

```sh
ls /var/cache/pacman/pkg/mesa-26.1.*-aarch64.pkg.tar.xz
sudo pacman -U mesa-honeykrisp-omarchy-*.pkg.tar.xz
# answer y to pacman's exact `Remove mesa?` conflict prompt
env -u VK_ICD_FILENAMES -u AGX_SIMDMAT vulkaninfo --summary | grep -E 'driverName|driverInfo'
```

`driverInfo` must read `Mesa 26.3.0-devel (git-6f6afc8968)`. Running GL
clients keep the old libraries mapped until they restart; log out and back
in (or reboot) for the compositor to pick up the new GL.

### Normal use

Nothing to set. No `VK_ICD_FILENAMES`, no `AGX_SIMDMAT`, no private ICD
json; the wheel detects `VK_KHR_cooperative_matrix` from the device
extension list and uses the coopmat matmul kernel on its own.
`AGX_SIMDMAT=0` turns the extension off again for A/B comparison. The
`flock /tmp/m1-gpu.lock` wrapper in this project's receipts is a
multi-agent convention for the shared test machine, not a driver
requirement.

### Rollback

```sh
sudo pacman -U /var/cache/pacman/pkg/mesa-26.1.7-1-aarch64.pkg.tar.xz
```

pacman removes `mesa-honeykrisp-omarchy` as the conflict and restores the
stock driver. `sudo pacman -S mesa` does the same from the asahi-alarm
repository. Because the fork package `conflicts=('mesa')` and provides
`mesa`, `pacman -Syu` never silently swaps it back for a stock release;
moving to a newer stock Mesa is always this explicit step.

### Distribution

The `[omarchy-aarch64]` pacman repository that Omarchy Mac installs
(`github.com/omarchy-mac/omarchy-pkgs-aarch64`, release tag `edge`) is
owned by the `omarchy-mac` organization; this project has read access
only, so nothing is published there. That repository does accept in-tree
PKGBUILDs (`pkgbuilds/` plus a `source: local`, `category: compile` entry
in `packages.json`, built on a native ARM runner), so the path is a pull
request carrying `packaging/mesa-honeykrisp-omarchy/`. Until then, this
section's `makepkg` is the supported route, and a personal pacman
repository is the self-hosted alternative: `repo-add
mesa-honeykrisp-omarchy.db.tar.gz *.pkg.tar.xz`, upload the package and
db files to a GitHub release, and point a `Server =` line at the
release's download URL, exactly as `[omarchy-aarch64]` does.

## Benchmark matrix

`scripts/bench_matrix.py` runs the declared workload matrix (models x
prompts x pinned-length decode) through `scripts/bench_decode.py`. It
never downloads models: a snapshot missing from the local Hugging Face
cache is reported `skipped`, never passing, and revisions are read from
the cache, never guessed.

On Linux, inside the venv that holds the mlx-omarchy wheel, add `mlx-lm`
to the same venv, then:

```sh
python3 scripts/bench_matrix.py --mode plan
python3 scripts/bench_matrix.py --mode run \
  --python ~/.venvs/mlx-omarchy/bin/python --wheel dist/mlx_omarchy-*.whl
```

`--wheel` hands the file to `bench_decode`'s provenance gate, which
refuses to emit numbers from a mismatched binary. On a development
machine without an Apple GPU, add `--allow-non-apple`; llvmpipe results
are correctness checks, never performance claims.

On macOS (16-inch M1 Max baseline), keep the benchmark in its own venv
and never install into system Python, Homebrew, or an existing venv:

```sh
/opt/homebrew/bin/python3.12 -m venv ~/src/mlx-bench-$(date +%Y%m%d)
~/src/mlx-bench-<date>/bin/pip install "mlx" "mlx-lm==0.31.3"
python3 scripts/bench_matrix.py --mode metadata \
  --python ~/src/mlx-bench-<date>/bin/python
python3 scripts/bench_matrix.py --mode run \
  --python ~/src/mlx-bench-<date>/bin/python --host-label <label>
```

`metadata` records chip, core count, memory, OS version and build, MLX
and mlx-lm versions, Metal identity, source commit and dirty state, power
state, and any running model-serving processes. Hostnames, user names,
and serial numbers are excluded. A run while `llama-server`, `ollama`, or
similar processes are serving is labeled contended; contended timings are
never compared against clean numbers.

The matrix covers ~262, ~1024, and ~4096 prompt-token prefill plus
32/128-token pinned decode; exact prompt token counts are recorded per
leg from bench_decode's own measured generation response, never assumed
or probed separately. The ~4096 workload is explicit selection only, to
bound normal runs:

```sh
python3 scripts/bench_matrix.py --mode run --select longctx-4096-decode-32
```

Every run records a pins map: each ready model with its exact revision,
labeled `pinned` (manifest SHA) or `resolved-from-cache` (optional
models). To compare two machines, pass machine A's pins map to machine
B with `--expect-pins MODEL_ID=REVISION`; a different resolved revision
refuses the run with exit 4 before anything executes, because the same
model id with different weights is not a comparison.
