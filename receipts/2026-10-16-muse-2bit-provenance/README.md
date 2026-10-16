# Muse 2-bit conversion: provenance statement

Full provenance for the 2026-10-15 conversion runs behind `tools/muse-2bit/`
(`BUILD.md`, `recipe.json`). This receipt keeps the complete detail; the
public build guide states plain facts only (AGENTS.md, "Public writing").

## What the records state

- Source, part/tensor counts, bits per group: `tools/muse-2bit/BUILD.md` and
  `recipe.json` (source revision `52cace63f43ed9fc6acef6dcd391adb02f4bfbe1`).
- Tools: converter `requant_stream.py` sha256
  `9a2fa17a762c8806da6297376088a7dcefce794e15a3ce73d537dc13cb77f3b8`;
  finalizer `finalize2.py`; manifest `mac-manifest.py`.
- mlx, as `recipe.json` keys it: `0.32.3` on the Studio macOS ("layers 0-13
  and shared checked"); `0.32.4` on the `M2 aarch64 Linux CPU` (its key reads
  "layers 14-36, 37-61 except as noted"). `BUILD.md` Hashes places layers
  14-36, 39 and 41-61 (45 layers) on that machine, the "M2 laptop".
- Python, as `recipe.json` keys it: `3.13.16` (Studio), `3.11.2`
  (`CT shared part`). The `CT` entry names no machine.
- Layer 3 was converted twice on the Mac Studio and once on the M2 machine,
  with all 39 per-tensor sha256 equal (`recipe.json`, `equivalence`;
  `BUILD.md`, Equivalence rule).

## What the records do not state

- The machine model behind `M2 aarch64 Linux CPU` / "M2 laptop" is not
  recorded: neither `recipe.json` nor Hashes names a model or screen size.
- Layers 37, 38 and 40 are placed by no record in the repository: no machine,
  date, or operator is recorded for them; the 45-layer list in Hashes skips
  them.
- Which machine ran the layer 1, expert 5 independent dequant/requant check
  is not recorded; the check is described only as CPU backend (`BUILD.md`,
  Equivalence rule; `recipe.json`, `measured_so_far`).
- No mlx-omarchy wheel is recorded for these runs and `recipe.json` carries
  no omarchy dev-build version string. `0.32.4` is a bare base version and is
  also the base of the omarchy-mlx development wheels, which add a local
  segment (a real dev-wheel string from
  `receipts/2026-10-16-gather-qmm-subgroup-3bit-2bit`:
  `mlx_omarchy-0.32.4.dev202610160214+39f00215-cp314-cp314-linux_aarch64.whl`);
  the recorded bare `0.32.4` therefore cannot distinguish stock mlx from an
  omarchy dev build. What was installed on each machine beyond the recorded
  version numbers is not captured.
- No installed-binary check was recorded for the runs behind the Result
  numbers and `recipe.json`'s `measured_so_far`: no `scripts/mlx_provenance.py`
  line exists for those conversions, and they cannot be rerun to extend the
  record. Those published numbers carry no installed-binary provenance.

## What a rebuild must record

Run the provenance check with the same interpreter that runs
`requant_stream.py` — not a different environment — and put its line beside
the published numbers, after `--self-test` exits 0 on that interpreter:

    <interpreter-that-runs-requant_stream.py> scripts/mlx_provenance.py

AGENTS.md "Test rules" requires the provenance line beside every
measurement; this is how a muse-2bit rebuild satisfies it.
