# Parakeet transcribe missing runtime deps (numpy, google.protobuf)

Defect: on an installed omarchy-mlx 0.7.10 venv (jwm1, fresh Omarchy, M1),
`mlx-omarchy-parakeet transcribe` exits with
`missing runtime dependencies: numpy, google.protobuf` while `verify` works.
Fixed so a normal installed package transcribes without manual pip.

Raw evidence (private notebook, not public): `apple-silicon-lab`
`entries/ParakeetDeps/20261002T113618Z-jwm1-parakeet-deps-repro.md` and
`20261002T120500Z-jw16-parakeet-deps-fix-verify.md`. This receipt carries the
public-safe results.

## Root cause

Not packaging. `packaging/requirements-lock.in` has listed `numpy` and
`protobuf` since `42c0f25f6` (pre-v0.7.10), and jwm1's 0.7.10 system venv
physically contains them (numpy 2.5.3, protobuf 7.36.2 dist-infos on disk;
both import in the venv python). The v0.7.14 vendor tar carries them too:
`numpy-2.5.3-cp314-cp314-manylinux_2_27_aarch64...whl` and
`protobuf-7.36.2-cp310-abi3-manylinux2014_aarch64.whl` (tar sha256
`24a5bef313eeb6b25345c34e907f8ff2737a31fe2ae373f49bb7209074adac16`, matches
the published `.sha256`).

The deps are lost at **interpreter binding**:

1. The wheel ships the CLI as a data file,
   `site-packages/mlx/bin/mlx-omarchy-parakeet`, with the source-tree
   shebang `#!/usr/bin/env python3`.
2. Every staged launcher in `/usr/bin` routes through
   `/usr/lib/omarchy-mlx/venv/bin/python` — except there was **no parakeet
   launcher at all** (7 launchers staged, none for parakeet).
3. So direct execution resolves `python3` → the system interpreter, which
   has no numpy/protobuf (jwm1: `ModuleNotFoundError: No module named
   'numpy'` / `'google'`).
4. `verify` works because its handler only hash-verifies the reference
   cache and fixture — it never imports numpy; the golden-e2e leg it would
   share with transcribe is skipped/refused before the import matters.

Repro on jwm1 (read-only; CLI run under `/usr/bin/python3`, i.e. the
shebang's resolution):

```
error: missing runtime dependencies for transcribe: numpy, google.protobuf;
install them with `pip install numpy protobuf` (ffmpeg handles FLAC decode
when soundfile is absent)
```

## Fix (this branch)

1. **CLI self-heal** (`overlay/tools/mlx-omarchy-parakeet/
   mlx_omarchy_parakeet.py`): when the dependency probe fails, derive the
   interpreter of the environment that owns the installed file
   (`<env>/lib/python*/site-packages/mlx/bin/` → `<env>/bin/python` — five
   parents up), probe that it can import the missing modules, and `execve`
   into it once (env-guard `MLX_OMARCHY_PARAKEET_REEXEC` prevents loops).
   The identity check compares **raw paths**: a venv python symlinks to the
   same CPython binary the system `python3` uses (jw16:
   `/usr/bin/python3 → python3.14`), so a `resolve()` comparison collapses
   the two environments — that was the one real bug found during
   verification (first overlay, `875cc27f…`, did not re-exec; fixed and
   covered by `test_installing_venv_python_accepts_symlinked_venv_python`).
2. **Actionable refusal**: the message now names the interpreter that ran,
   the exact pip line for it (`<sys.executable> -m pip install …`, module
   → distribution names mapped: `google.protobuf` → `protobuf`), and — when
   present — the system package's runtime interpreter
   (`/usr/lib/omarchy-mlx/venv/bin/python`). The
   "missing runtime dependencies for transcribe:" prefix is unchanged.
3. **Launcher staging** (`install.sh`, both lanes): the wheel now gets a
   `mlx-omarchy-parakeet` launcher (`/usr/bin` in the `--system` lane,
   `$BIN` in the release-script lane) that execs the installed CLI through
   the owning venv's python — the same pattern as the info launcher,
   skipped with a note on wheels without the aarch64-only runtime.
   `packaging/PKGBUILD.example` layout table updated accordingly.
4. **Deliberately unchanged**: wheel `install_requires` stays empty (the
   upstream packaging contract documented in the CLI docstring); every
   sanctioned install path (vendor lock, system package, install.sh pins)
   already carries numpy+protobuf. Adding metadata deps would not have
   prevented this defect — the venvs always had them.

## Tests

- `overlay/tests/omarchy/coreml/test_parakeet_cli.py` — new
  `ParakeetDependencyProbeTest` (11 tests green):
  - probe message names the pip line and interpreter;
  - refusal fires when no re-exec target exists;
  - venv discovery: none in the dev checkout, stub env found, **symlinked
    venv python still found** (the raw-path regression);
  - `transcribe --help` under an interpreter without the deps, inside a
    fake installed layout, re-execs into the owning venv and exits 0
    (the jwm1 defect staged locally);
  - **import-closure contract**: AST closure of the CLI + vendored
    `coreml` tree must be exactly {mlx, numpy, google, soundfile} — the
    two declared deps are required to appear, soundfile stays optional,
    anything else fails the test.
- `tests/test_install_system_layout.py` — fake wheel now carries
  `mlx/bin/mlx-omarchy-parakeet`; staged tree asserts the launcher exists,
  embeds the final `/usr/lib/omarchy-mlx/venv/bin/python` and the staged
  CLI path, and that the staged CLI runs through the staged venv python
  (16 tests green).
- Suites touching the CLI: `tests/coreml/test_parakeet_runtime_pins.py`
  and `tests/test_install_sh_contract.py::SocGateTests::
  test_serve_catalog_arches_cover_the_m1_family` fail identically on clean
  `origin/main` (`96c91547d`) — pre-existing, not from this branch.

## Before/after on published v0.7.14 (jw16, M1 Max, cp314 aarch64, CPU-only)

Throwaway venv from the published wheel + the two wheels out of the
published vendor tar (no manual pip beyond that). `MLX_OMARCHY_CACHE_DIR`
pointed at an empty dir for the after-runs so the pipeline can never start
(cache hashing refuses first) — zero GPU/ANE execution.

| invocation | v0.7.14 (before) | fix branch (after) |
|---|---|---|
| `"$CLI" transcribe` (shebang → system python3) | `missing runtime dependencies for transcribe: google.protobuf` | `the reference cache does not verify against the lock; run mlx-omarchy-parakeet download first` |
| `/usr/bin/python3 -S "$CLI" transcribe` (bare interpreter, verbatim jwm1 form) | `missing runtime dependencies for transcribe: numpy, google.protobuf` | same cache refusal |
| `"$CLI" transcribe --help` | rc 0 (argparse exits before the handler) | rc 0 |

The only interpreter holding numpy+protobuf is the owning venv, so reaching
the cache gate proves the dependency boundary was crossed by the re-exec.

Fix-branch CLI sha256 at verification: `15213009d5831dea9f1d09f4b429ed2bacfd
2737adad87cae41dde56fcc4cf21` (source and installed bytes identical).

## Recipe side (omarchy-pkgs owner — no PR opened from here)

The pacman `depends=()` list is **not** the problem and needs no change:
the venv is private and self-contained; numpy/protobuf come from the
vendored wheels, not pacman. To pick the fix up, the recipe needs one new
release of mlx-omarchy (see below) and nothing else — `install.sh --system`
stages `/usr/bin/mlx-omarchy-parakeet` itself. Optional hardening for the
recipe's `check()`:

```bash
"stage/usr/lib/omarchy-mlx/venv/bin/python" \
  -c 'import mlx.core, mlx_omarchy_paths, mlx_omarchy_serve, numpy, google.protobuf'
```

(the same line as the updated `packaging/PKGBUILD.example`) so a venv
missing the transcribe deps fails at build time instead of at first
transcribe.

## Release needed?

**Yes.** Code (CLI) and install.sh changed; the fix reaches users only
through a new wheel + vendor tar release and a rebuilt `omarchy-mlx`
package. Until then, installed 0.7.x systems can be unblocked without any
install: run the CLI through its owning venv python
(`/usr/lib/omarchy-mlx/venv/bin/python …/mlx/bin/mlx-omarchy-parakeet
transcribe`), or `pip install numpy protobuf` into the interpreter the
error names.
