# Muse 2-bit conversion: provenance statement

Full provenance for the 2026-10-15 conversion runs behind `tools/muse-2bit/`.
The public build guide states plain facts only. See AGENTS.md, "Public writing".
This receipt keeps the complete detail.

## What the records state

- Source revision: `52cace63f43ed9fc6acef6dcd391adb02f4bfbe1`.
  It is recorded in `BUILD.md` and `recipe.json`.
- Converter: `requant_stream.py`.
  sha256 `9a2fa17a762c8806da6297376088a7dcefce794e15a3ce73d537dc13cb77f3b8`.
- Finalizer: `finalize2.py`. Manifest: `mac-manifest.py`.
- mlx on the Studio macOS: `0.32.3`.
  `recipe.json` keys it to layers 0-13 and the shared-part check.
- mlx on the `M2 aarch64 Linux CPU`: `0.32.4`.
  `BUILD.md` Hashes places layers 14-36, 39 and 41-61 on it.
  That machine is the "M2 laptop".
- Python on the Studio: `3.13.16`.
  Python for the `CT shared part`: `3.11.2`.
  The `CT` entry names no machine.
- Layer 3 was converted twice on the Mac Studio.
  It was converted once on the M2 machine.
  All 39 per-tensor sha256 are equal.
  Sources: `recipe.json` (`equivalence`) and `BUILD.md` (Equivalence rule).

## What the records do not state

- The machine model behind `M2 aarch64 Linux CPU` and "M2 laptop" is not
  recorded. Neither `recipe.json` nor Hashes names a model or screen size.
- Layers 37, 38 and 40 are placed by no record in the repository.
  No machine, date, or operator is recorded for them.
  The 45-layer list in Hashes skips them.
- The layer 1, expert 5 independent dequant/requant check records no machine.
  The check is described only as CPU backend.
  Sources: `BUILD.md` (Equivalence rule) and `recipe.json` (`measured_so_far`).
- No mlx-omarchy wheel is recorded for these runs.
  `recipe.json` carries no omarchy dev-build version string.
- `0.32.4` is a bare base version.
  It is also the base of the omarchy-mlx development wheels.
  Dev wheels add a local segment to that base.
  A real dev-wheel string is
  `mlx_omarchy-0.32.4.dev202610160214+39f00215-cp314-cp314-linux_aarch64.whl`.
  It comes from `receipts/2026-10-16-gather-qmm-subgroup-3bit-2bit`.
  The recorded bare `0.32.4` cannot distinguish stock mlx from an omarchy dev
  build.
- What was installed on each machine beyond the recorded version numbers is
  not captured.
- No installed-binary check was recorded for the runs behind the Result
  numbers and `recipe.json`'s `measured_so_far`.
  No `scripts/mlx_provenance.py` line exists for those conversions.
  They cannot be rerun to extend the record.
  Those published numbers carry no installed-binary provenance.

## What a rebuild must record

Run the provenance check with the same interpreter that runs
`requant_stream.py`. Do not use a different environment.
Put its line beside the published numbers.
`--self-test` must exit 0 on that interpreter first:

    <interpreter-that-runs-requant_stream.py> scripts/mlx_provenance.py

AGENTS.md "Test rules" requires the provenance line beside every measurement.
This is how a muse-2bit rebuild satisfies it.
