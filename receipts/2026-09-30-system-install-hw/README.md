# System install hardware check (vendored wheels, offline `--system` stage)

Commit under test: `42c0f25f6` ("packaging: venv discovery, offline vendored
build, legacy retire command"), verified on Apple M1 Max / T6001 hardware
against the packaged DKMS `ane.ko` (`0.2.0.r14.g87f427f`). The full raw logs,
window transcripts, and `SHA256SUMS` live in the private notebook under
`artifacts/SysInstallCheck/20260930T2231Z-*/`; this receipt carries the
public-safe results only.

## Vendored wheel set

`packaging/vendor-wheels.sh` ran on the T6001 host with the pinned
interpreter (Python 3.14.7) and resolved `requirements-lock.in` for aarch64:

- 36 wheels, `packaging/verify-vendor.sh` green ("verified 36 vendored
  wheel(s)"), zero sdist builds.
- No lock input lacks an aarch64-installable binary wheel: C extensions ship
  as `cp314` (numpy, pyyaml, markupsafe, regex, cffi, sentencepiece) or as
  stable-ABI `abi3` wheels (protobuf `cp310-abi3`, tokenizers
  `cp310-abi3`, safetensors `cp310-abi3`, hf-xet `cp38-abi3`); the rest are
  pure-Python.
- Total 441 MB. Largest: the repo wheel 416,224,396 B
  (`mlx_omarchy-0.32.3.dev202609302237+sysinst.42c0f25-cp314-cp314-linux_aarch64.whl`,
  SHA-256 `ac362ac7706b423f08cc6560a31c6174e5f454e26a33a424975fb67f78d0f36a`,
  built from `42c0f25f6` with glslc 2026.3 and the exact pinned
  whole-encoder bundle), numpy 15.7 MB, transformers 12.1 MB.

## Offline system stage

`install.sh --system --dest-root ... --vendor ... --lock ...` ran inside a
user + network namespace (netlink showed loopback only and any outbound
socket failed immediately), and exited 0: the venv imports `mlx.core`,
`mlx_lm`, `mlx_omarchy_paths`, and `mlx_omarchy_serve` with no network; the
staged `/usr/bin` launchers, unit, and desktop entry embed only final system
paths.

Two bugs were found on hardware, both fixed test-first and pushed in this
lane:

1. **Staging prefix leaked into venv bin scripts.** `build-venv.sh` creates
   the venv at the staging absolute path, so 39 files under `venv/bin/`
   (pip, activate, every `mlx_lm.*` and third-party entry point) embedded
   the `--dest-root` prefix and would die after a PKGBUILD's `package()`
   copies the tree to `/` — contradicting `packaging/PKGBUILD.example`'s
   "no shebang fixups are needed". `install.sh` now rewrites the staging
   prefix to the final venv path in every regular `venv/bin` file; a
   regression test asserts the staged tree carries only final paths
   (`a48e20c52`).
2. **Staged retire script could not find its name table.** The staged
   `mlx-omarchy-retire-legacy` exited 1 with "paths.sh (the name table) not
   found" because its candidate list had no entry that exists in a staging
   tree. It now resolves `usr/bin/../share/omarchy-mlx/paths.sh`, which is
   the same relative path on an installed system (`241353240`).

A third packaging defect was fixed in the wheel build itself: the documented
`MLX_OMARCHY_WHOLE_BUNDLE_SKIP=1` opt-out never worked (the runtime pin
always declares the whole bundle, so the flag only ever reached the
refuse-and-exit branch). The gate moved verbatim to
`packaging/stage-whole-bundle.sh` with six behavior tests, the trap flag is
gone, and the header states the real contract: exact pinned bundle bytes or
refusal (`baa38b453`).

## Hardware checks from the staged venv

Inside one GPU window (serving stack stopped and restored with a health 200
plus authenticated completion probe around it):

- `mlx-omarchy-info` from the staged venv: `Apple M1 Max (G13C C0)`, driver
  `Honeykrisp` (`Mesa 26.3.0-devel`), system ICD
  `asahi_icd.aarch64.json`, Vulkan 1.4.359, device-tree `apple,t6000-ane`,
  ANE module `0.2.0.r14.g87f427f`.
- `scripts/mlx_provenance.py --expect-wheel <built wheel>`: `version_match:
  true`, dist == mx == `0.32.3.dev202609302237+sysinst.42c0f25`, and the
  loaded `mlx/lib/libmlx.so` hash matches the wheel RECORD.
- A real matmul computed on `Device(gpu, 0)`.

Inside a second window (GPU lock plus the ANE lane lock), the h13 add-mul
bundle ran against the packaged DKMS module: exit 0, `abi=1`,
`dt_compatible=apple,t6000-ane`, driver `0.2.0.r14.g87f427f`, both
iterations `exact_fp16=PASS`, clean shutdown
(`released_programs=2 process_released=true`), no ANE errors in the kernel
log. (The smoke binary is the runtime-gate lane's stage-B build; this lane
reused it by design, so the packaged module and gate are exercised here,
not this commit's worker binary.)

## Discovery precedence and retire

- `$OMARCHY_MLX_VENV` beats the system prefix; with the env unset the
  helper picks the system venv (`'system'` origin).
- With only a legacy home venv present, `discover_venv` prints the one-line
  warning naming `mlx-omarchy-retire-legacy` and returns the legacy root.
- The dry-run retire on a real home that has only the data root (no venv,
  launchers, or unit) reports "Legacy install already retired. User data
  remains under ...", removes nothing, and exits 0.

## Not covered

An actual `makepkg`/pacman install (the stage-copy equivalence is what the
tests assert), GPU performance numbers, and the `--voice` extras (system
package scope decision: voice stays install.sh-only for v1).
