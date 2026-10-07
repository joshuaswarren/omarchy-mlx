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

Two fixes shipped in v0.7.17 (`receipts/2026-10-02-parakeet-deps`,
`receipts/2026-10-02-libane-pin`; installs from v0.7.17 or later carry
both):

- `mlx-omarchy-parakeet transcribe` runs on an installed system without
  manual pip installs. The pre-fix launcher carried a source-tree shebang and
  had no staged venv launcher, so it resolved the system `python3` and failed
  with missing `numpy` / `google.protobuf` even though the venv contained
  both. On an installed 0.7.x system built before the fix, run the CLI
  through its owning interpreter:
  `/usr/lib/omarchy-mlx/venv/bin/python .../mlx/bin/mlx-omarchy-parakeet transcribe`.
  The packaging recipe's `check()` now imports the transcribe deps so a
  broken venv fails at build time.
- The whole-encoder share tree is seal-verified: the v0.7.14 aarch64 wheel's
  11 pinned files (libane, pin json, bundles) match `parakeet-runtime-pin.json`
  with no mismatches, and a seal mismatch now reports the parsed mismatch by
  name instead of a raw traceback (`scripts/verify_runtime_assets.py`).

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
token IDs on M1. The hardware gate receipt for that A/B is not in this
checkout.

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

Wave scheduling (scheduling only; results are bit-identical):
`MLX_OMARCHY_WAVE_SCHED=1` buffers the open batch's dispatches, copies, and
fills instead of recording them in tape order, and records them at submit in
greedy earliest-wave order — one full dependency barrier per wave, tape order
preserved within a wave. A node joins the current wave only when it has no
RAW/WAW/WAR overlap (exact byte ranges, SPIR-V reflected read/write split)
with any earlier node, so hazards stay ordered exactly as in tape order and
outputs are unchanged; hazard-free nodes run concurrently instead of
serializing behind barriers their tape neighbors induced. Requires gated
barriers. Off by default until its jw16 A/B lands.

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
decode64 41.48 tok/s, digests unchanged.

Per-submission GPU-time cap (issue #19): one queue submission holds the GPU
until it finishes and the desktop compositor only gets the queue between
submissions, so a ~78 ms submission hitches the desktop for about nine frames
at 120 Hz while 2-6 ms submissions stay smooth (same GPU busy fraction;
reporter's OpenGL measurement). `MLX_OMARCHY_BATCH_WORK=<groups>` bounds each
submission to an estimated GPU cost: the open batch is submitted once its
summed dispatch work-group counts reach the budget. The default
(`40000` groups) is calibrated on the M2 Max so a 4B decode step splits
into ~2-6 ms submissions; a single dispatch larger than the budget still runs
whole (splitting happens between dispatches), and copies/fills ride the node
and byte budgets. `MLX_OMARCHY_BATCH_WORK=0` disables the cap (headless
boxes). Scheduling only: every submission already waits on the stream's
previous completion, so splitting preserves order and results are
bit-identical (greedy ids digests identical in every A/B pair;
`receipts/2026-10-02-submission-cap-19`). Measured on the M2 Max at the
default: a 4B decode step drops from p50 20.9 ms to 5.6 ms per submission
(decode tok/s −0.15 %), the 9B from 49.6 ms to 4.4 ms (+6 % decode).
Honest limit: frame pacing was not measured on a real logged-in compositor —
none was available; the evidence is submission-length histograms plus a
60 Hz tiny-submit probe whose worst host latency was 2.7 ms even with the
cap off on the idle M2.

Queue priority (issue #19, where the driver supports it): when the device
lists `VK_EXT_global_priority` and the compute queue family reports the
requested priority, the backend chains `LOW` into the queue so desktop work
wins arbitration between MLX submissions; otherwise the default priority is
kept silently. `MLX_OMARCHY_QUEUE_PRIORITY=medium` requests MEDIUM instead;
`off`, `default`, or `0` keeps the unchained queue. Measured exposure:
Honeykrisp lists `VK_EXT_global_priority` rev 2 on the M2 Max (T6021) and
M1 Max (T6001).

GPU-memory heap default (`HK_SYSMEM`): Honeykrisp sizes its GPU heap at
50% of MemTotal (mesa `heap_memory_percent` 0.5) and fails every
allocation past it, which on 32 GB+ hosts strands half the RAM: on the
62 GB M1 Max the stock heap (31.18 GiB) rejects Qwen3-32B-8bit's 34.8 GB
of weights outright. MLX processes therefore set the heap themselves
before Vulkan loads: `max(50% of MemTotal, MemTotal − 16 GiB)`, rounded
down to 1 MiB, clamped to 60 GiB. The clamp follows the per-process GPU
virtual-address window (~64 GiB; the 2026-10-05 heap-budget probe
committed 63 GiB before the 64th allocation failed to map, so a larger
heap cannot be used). On MemTotal ≤ 32 GiB the formula never beats the
50% default and those hosts (16 GB machines included) keep the stock
heap byte-for-byte. An explicit `HK_SYSMEM` (absolute bytes, as before)
and a `heap_memory_percent` set in a drirc file (`$DRIRC_CONFIGDIR`,
`drirc.d`, `/etc/drirc`, `~/.drirc`) still win — the default is applied
only when neither is present, and only inside MLX processes: every other
Vulkan application keeps the 50% heap.

Serving placement (`MLX_OMARCHY_UCLAMP_MIN`, default 1024): the serve raises
`uclamp_min` on its startup thread through an unprivileged
`sched_setattr` call, so every serving thread inherits the scheduler's
P-cluster boost hint under the stock `schedutil` governor — first-token
latency recovers most of the performance-governor gain without touching
any governor or needing root (mechanism: the submit thread otherwise
lands on the E cluster and pays the frequency ramp at request start;
see `receipts/2026-10-02-perf-hold`). `MLX_OMARCHY_UCLAMP_MIN=0` disables;
any other value sets the clamp. On kernels that refuse the call
(`EPERM`/`EINVAL` — e.g. hosts without uclamp support) the serve logs
one stderr line and serves unchanged. Measured A/B on both Macs with the
published v0.7.14 wheel: prefill +1.0–2.3 % and first-token rate +0.3–2.7 %
by route, decode unchanged, greedy digests identical everywhere; the w71
lane's ≥6 % first-token numbers were taken against an idle-decayed P cluster,
a machine state that did not recur in this A/B. The power-profiles-daemon
route this replaces is infeasible on Apple Silicon — ppd 0.30 offers only
placeholder profiles (no `performance`), holds are polkit-gated to
seat-active sessions, and a successful hold moves no governor — so the
in-process clamp is the supported route and a narrow sudoers+helper governor
flip remains the documented fallback for a kernel that refuses
`sched_setattr` (`receipts/2026-10-02-perf-hold`). The payoff is
conditional on the P cluster's idle state: an independent jwm1 run on an
idle-decayed cluster measured pf512 first-token 0.1424 to 0.1303 s
(−8.7 %) and ttft rate +10.5 % with exact digests, reported in the
receipt's addendum ("reported by the jwm1 lane, notebook entry H219; not
re-measured here").

## Build the wheel

1. Install the build tools: Python 3.10 or newer with `venv`, `cmake` 3.25 or
   newer, Vulkan development headers, a C++ compiler, and the BLAS/LAPACK
   development packages the CPU backend links (`liblapack-dev libblas-dev
   liblapacke-dev` on Debian-family distributions).
   On Arch/Omarchy, install `blas-openblas` and set
   `CMAKE_INCLUDE_PATH=/usr/include/openblas` for the build; both BLAS and
   LAPACK headers live in that package-owned directory. No source or linker
   flag changes are needed. `build-wheel.sh` checks for `lapacke.h` first: it
   uses `/usr/include/openblas/lapacke.h` when no other copy exists, and stops
   with the package to install when there is none.
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

## Honeykrisp driver

The supported driver is the `omarchy-mlx-vulkan` package. It installs a
private Honeykrisp Vulkan ICD under `/usr/lib/omarchy-mlx/vulkan/` and
leaves the system Mesa package installed for the desktop. `omarchy-mlx`
depends on the same version of `omarchy-mlx-vulkan`. The Omarchy menu
entry Install > AI > MLX + Core ML installs the `omarchy-mac-ml` meta
package, which pulls both in. A wheel or an `install.sh` venv ships GPU
kernels only.

The Vulkan package builds the Honeykrisp Vulkan driver alone: the
driver library and the ICD JSON live under `/usr/lib/omarchy-mlx/vulkan/`,
outside the loader's default ICD directories. GL, window-system, and
Gallium drivers stay with the system Mesa package. The package
conflicts with `mesa-honeykrisp-omarchy`.

`packaging/mesa-honeykrisp-omarchy/` is not a supported install. Do not
build or install it. That recipe replaces the system Mesa package.

### Package layout

Names are fixed in `serve/mlx_omarchy_paths.py` and
`overlay/mlx/backend/omarchy/honeykrisp_identity.h`:

| Path | Contents |
|---|---|
| `/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json` | ICD JSON. `library_path` is an absolute path to the driver library installed in the same directory. |
| `/usr/lib/omarchy-mlx/vulkan/mesa-git-sha` | One line: the hex string the driver reports after `git-` in `driverInfo`. |

`OMARCHY_MLX_SYSTEM_PREFIX`, when set and non-empty, replaces the
`/usr/lib/omarchy-mlx` prefix. Tests and staged trees use that seam.
The ICD and SHA paths are then `<prefix>/vulkan/honeykrisp_icd.aarch64.json`
and `<prefix>/vulkan/mesa-git-sha`.

### Selection

Before Vulkan loads, the backend selects the ICD for this process
(`configure_honeykrisp_icd` in `overlay/mlx/backend/omarchy/device.cpp`).
`is_honeykrisp_icd` in `honeykrisp_identity.h` accepts a path whose
name contains `honeykrisp`, in any case. It also accepts the stock
system ICD JSON and the driver library name, so an explicit override
of the system driver still resolves. The packaged file
`honeykrisp_icd.aarch64.json` matches.

1. A non-empty `VK_DRIVER_FILES` is the override. `VK_ICD_FILENAMES` is
   the override only when `VK_DRIVER_FILES` is unset or empty. A
   colon-separated value uses the first entry that is a Honeykrisp
   ICD. That file must exist; a later entry is not tried. An
   override does not fall through to the packaged ICD and does not
   inherit the packaged SHA. `icd_source` is `override`.
2. Otherwise, if the packaged ICD JSON exists, it is selected
   (`icd_source` `packaged`).
3. Otherwise the backend searches `/etc/vulkan/icd.d`, then
   `/usr/local/share/vulkan/icd.d`, then `/usr/share/vulkan/icd.d`, and
   selects the first Honeykrisp ICD JSON it finds (`icd_source`
   `search`). Without the package, that is usually the system Mesa
   driver, and no packaged SHA applies.

In cases 2 and 3 this process sets both `VK_DRIVER_FILES` and
`VK_ICD_FILENAMES` to the selected path. An override leaves those
variables as the caller set them.

A missing override file fails initialization with
`Honeykrisp ICD selection refused: user Vulkan ICD JSON does not exist: `
plus the path. A value that excludes Honeykrisp fails with
`Honeykrisp ICD selection refused: user Vulkan ICD value excludes Honeykrisp: `
plus the value. If no ICD is found, initialization fails with
`Honeykrisp Vulkan ICD JSON was not found`.

Leave `VK_DRIVER_FILES` and `VK_ICD_FILENAMES` unset on a packaged
install. The loader variables are set inside the MLX process. Other
programs keep the loader's normal ICD search and the system Mesa driver.

### Identity check

`mlx-omarchy-info` and `mx.device_info()` report `icd_path`,
`icd_source` (`packaged`, `override`, or `search`), `driver`,
`driver_info`, `driver_sha`, `expected_sha`, and
`expected_sha_source`. `driver_sha` is the hex run after `git-` in
driver info when that run is at least 7 digits; otherwise it is empty.

Expected SHA (`resolve_expected_sha_policy`):

- `MLX_OMARCHY_EXPECTED_HK_SHA`, when set and non-empty, is the expected
  value. `expected_sha_source` is `env`.
- When that variable is unset and the packaged ICD was selected, the
  expected value is the trimmed contents of `mesa-git-sha`.
  `expected_sha_source` is `packaged file`. A missing or unreadable
  file leaves the expectation empty.
- The check runs when the device's Vulkan driver id is the Honeykrisp
  id (`26`). A non-empty expectation that differs from `driver_sha`
  fails initialization with
  `Honeykrisp Mesa git SHA mismatch: expected <expected>, found <actual>`
  (`found unavailable` when `driver_sha` is empty).
- With no expectation, the runtime records identity and does not reject
  the driver.

On a packaged install with the variables unset, `mlx-omarchy-info`
reports `icd_source` `packaged`, `icd_path`
`/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json`, and
`expected_sha` equal to `driver_sha` with `expected_sha_source`
`packaged file`.

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
