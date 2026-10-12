# Run omarchy-mlx on any Vulkan GPU

Back to the [README](../README.md). Results so far:
[TESTED-HARDWARE.md](TESTED-HARDWARE.md).

omarchy-mlx targets Apple GPUs through Honeykrisp. On any other Vulkan
1.3 device (NVIDIA, Intel, AMD, or Apple silicon without the
Honeykrisp ICD), the backend refuses at device discovery on purpose.
`MLX_OMARCHY_ALLOW_NON_APPLE=1` lifts that refusal for development.
The kernels and tools are unchanged; what changes is which device the
loader hands the backend.

This is a development path. A non-Apple GPU has no qualification
contract (that concept is defined for Apple silicon in
[docs/new-chip-bringup.md](new-chip-bringup.md)). Do not set the flag
on an Apple-GPU receipt.

## One command

`scripts/run-on-your-hardware.sh` runs the whole path in this document:
it checks the prerequisites and names the Omarchy/Arch packages to
install, stages the pinned MLX source, builds and runs the runtime
suite, and prints the quick capability report.

```bash
./scripts/run-on-your-hardware.sh            # check, build, run, collect
./scripts/run-on-your-hardware.sh --submit   # also publish the report
```

A run on a terminal offers the same publish step as an exact-word
`SUBMIT` prompt after the report prints. The published row is the
printed report and nothing else; the dataset deduplicates identical
reports by content hash, and [contribute-data.md](contribute-data.md)
describes the dataset and the redaction. The suite log is not
published; it stays in the output directory the script prints, and a
failure report needs it (see "Attach your results").

Verification status of the script:

| Verified | Where |
|---|---|
| Prerequisite checks, device detection, `--help`, `--dry-run` | x86_64 CPU-only container (Debian 12), device llvmpipe |
| Stage, build, runtime suite, quick collector | lavapipe software Vulkan 1.3, same container |
| AMD, Intel, NVIDIA hardware | not verified |
| Fresh Omarchy install from the ISO | not verified |

## Prerequisites

- Linux with a Vulkan 1.3 driver: Mesa ANV (Intel), Mesa RADV (AMD),
  the NVIDIA proprietary driver, or lavapipe/llvmpipe for software.
  Check with `vulkaninfo --summary` (package `vulkan-tools` on most
  distros).
- A C++ toolchain, CMake, and Python 3.
- Stock Mesa stays your system driver. The Honeykrisp ICD package
  (`omarchy-mlx-vulkan`, built from joshuaswarren/mesa-1) is only for
  Apple GPUs; a non-Apple box does not need it.

## Build

```bash
git clone https://github.com/joshuaswarren/omarchy-mlx.git
cd omarchy-mlx
./scripts/prepare-mlx.sh
.work/mlx/tools/ci/run-omarchy-runtime.sh
```

The repository root has no `CMakeLists.txt`. `prepare-mlx.sh` downloads
the pinned MLX source, applies the overlay and the patch set, and stages
the buildable tree at `.work/mlx` (it prints that path when it
finishes). Build and test commands run in the staged tree.
`run-omarchy-runtime.sh` takes its CMake source directory from its own
location, so the runnable copy is
`.work/mlx/tools/ci/run-omarchy-runtime.sh`; the copy under `overlay/`
cannot configure a build.

The script configures `build-omarchy/` (CMake, `MLX_BUILD_OMARCHY=ON`,
`MLX_BUILD_TESTS=ON`), builds, and runs `omarchy_runtime_tests`,
`omarchy_copy_offset_tests`, and `mlx-omarchy-info` with a `--json`
receipt. Its header says it directly: on a non-Omarchy development
machine, export `MLX_OMARCHY_ALLOW_NON_APPLE=1` first.

The full battery instead of the runtime slice:

```bash
cmake -S .work/mlx -B .work/mlx/build-omarchy \
  -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON
cmake --build .work/mlx/build-omarchy -j"$(nproc)"
MLX_OMARCHY_ALLOW_NON_APPLE=1 \
  ctest --test-dir .work/mlx/build-omarchy --output-on-failure
```

## Run the test suite

```bash
MLX_OMARCHY_ALLOW_NON_APPLE=1 \
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/<driver>_icd.json \
./.work/mlx/build-omarchy/tests/omarchy/omarchy_runtime_tests
```

- `VK_DRIVER_FILES` selects the ICD: `nvidia_icd.json` (proprietary
  NVIDIA), `intel_icd.x86_64.json` (Mesa ANV), `radeon_icd.x86_64.json`
  (Mesa RADV), `lvp_icd.x86_64.json` (lavapipe). `ls
  /usr/share/vulkan/icd.d/` lists yours. The loader default is fine
  when only one driver is installed.
- More than one GPU: `MLX_OMARCHY_DEVICE_INDEX=<n>` picks the device.
- With several drivers installed, name one with `VK_DRIVER_FILES` so
  the suite measures the device you mean.

## Reading a pass/fail

doctest prints the summary at the end:

```
[doctest] test cases:   55 |   55 passed |   0 failed | 0 skipped
```

Exit code 0 is pass. A failure prints the case name, the assertion,
and `file:line`; paste all of it. A case that skipped says why
(usually "no qualifying Vulkan device"): that means the flag or the
ICD is wrong, not that the test passed.

## Attach your results

Open an issue on
[joshuaswarren/omarchy-mlx](https://github.com/joshuaswarren/omarchy-mlx/issues)
with:

- the full test-suite output, including the doctest summary and exit code
- `vulkaninfo --summary`
- driver package versions (`pacman -Q mesa`, `nvidia-smi`, or your
  distro's equivalent)
- `uname -r` and the distro
- `git rev-parse HEAD` of this checkout

## Submit through the collector where it works

- `python3 scripts/collect_quick.py --submit` publishes host facts
  and, when `vulkaninfo` is installed, the Vulkan device section to
  the public community dataset. Without `vulkaninfo` there is no GPU
  identity on an x86_64 host, so the collector refuses to publish
  (see the decision below); install `vulkan-tools` and rerun.
- `python3 scripts/collect_deep.py --out FILE` adds correctness and
  benchmark sections, but only where a `mlx` module is importable.
  Current releases ship aarch64 wheels only, so on x86_64 non-Apple
  boxes those sections report `available: false` (verified on a GPU-less
  x86_64 host on 2026-10-11). Export `MLX_OMARCHY_ALLOW_NON_APPLE=1`
  before running it so a source-built module may start.

## Non-Apple results stay issue-only

Decided 2026-10-12. The payload schema has no field for test-suite
output, and adding one means a schema version bump and a Worker
deploy. Decision: non-Apple test results stay issue-only. Follow
"Attach your results" above; the non-Apple tables in
[TESTED-HARDWARE.md](TESTED-HARDWARE.md) grow from those issues.

The quick and deep collectors also refuse to publish a row that names
no hardware (no Vulkan device and no chip or model). The dataset held
10 such rows from 2026-09-17 and 2026-09-18; they stay as they are.
Install `vulkan-tools` so `vulkaninfo` runs, then submit. The refusal
exits with status 3 and sends nothing; the report still prints and
`--out` still writes it.

The collector always previews first; nothing uploads without
`--submit`.
