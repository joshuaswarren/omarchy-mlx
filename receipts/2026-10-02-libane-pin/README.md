# 2026-10-02 — omarchy-mlx libane pin: root cause + seal diagnosis fix

## Identity and status

- LibanePin worker lane (jwm1 + dev box; read-only on jwm1; no ANE run by
  this actor). Continues H211 (omarchy-mlx 0.7.10-1 seal refusal on jwm1)
  and the w71 fresh-image qualification (`apple-silicon-lab/entries/jwm1-
  parity/20261001T193000Z-jwm1-qualify-local-ai-fresh-image.md`).
- Fix delivered on `joshuaswarren/omarchy-mlx` branch `LibanePin` (worktree
  `~/.config/superpowers/worktrees/mlx-omarchy/LibanePin`), rebased on
  origin/main `6cbf55d` and pushed to origin/main per repo rules.
- Notebook entry `LibanePin/20261002T123027Z-jwm1-libane-pin-defect.md`.

## Question and acceptance

- Defect: H211 reported omarchy-mlx 0.7.10-1 ships `libane-strict.so`
  `6e20168d…` while the wheel's own pin demands `d06222a8…`; the worker
  seal refuses; `mlx-omarchy-parakeet` exits with a raw traceback
  (`ane_resident.ResidentWorkerError`).
- Question: where did the 6e20168d bytes come from? Is the published
  release wheel itself broken, or did the recipe (omarchy-pkgs) ship a
  different stage? Decide the right fix in THIS repo (mlx-omarchy), keep
  ANE safety semantics, name the recipe recipe instruction, and tell Main
  whether the v0.7.14 release assets pass the seal.

## Root cause (every sha cited)

| Artifact | sha256 |
| --- | --- |
| Shipped installed libane (jwm1, on-disk) | `6e20168d4924689a6e70cca51098713ca25330c3aa2db62fe29d26942889e812` (67728 B) |
| Shipped installed pin json (jwm1, on-disk) | `d06222a86f3bff26aaf1cec1223ade32f27cad84b994dccb1af9cca965a7da8c` |
| Pacman mtree for libane (jwm1) | `6e20168d…` / mtime `1790900315.0` (2026-10-01 19:18:35 -0500) |
| Pacman Build Date (`pacman -Qi omarchy-mlx`) | Thu Oct  1 19:18:35 2026 |
| Installed RECORD libane entry (wheel's own metadata) | sha256 base64 `0GIiqG87…` = `d06222a8…`, size 71456 |
| Published v0.7.10 aarch64 wheel sha | `16a14856cf7ca7617750024518d307ee8…` |
| Published v0.7.14 aarch64 wheel sha | `60f175e0afde9568cd3e20eecca2e4daa03b54e85041619193ce2e3a7b8640a2` |
| libane inside v0.7.10 wheel | `d06222a8…` (matches pin) |
| libane inside v0.7.14 wheel | `d06222a8…` (matches pin; ELF BuildID `73750a9d049662538a6a7699b2c64e95957f79c3`; GCC 16.1.1 20260430) |
| Whole-encoder program seal in H211 receipt | `13c744231524d440b0a774155343df9ade0bbcbc37edc4b1ccf9698e580d5453` (matched its pin — only libane drifted) |
| Verbatim seal error (H211 `logs/smoke-v1.log`) | `error: [omarchy-ane] sealed libane-strict.so sha256 6e20168d… does not match the pin d06222a8…; refusing to load unverified ANE userspace` |

**The published v0.7.10 and v0.7.14 release wheels are internally
consistent: every pinned asset hash matches its byte.** `scripts/verify_runtime_assets.py` on the fully extracted v0.7.14 share tree reports `OK: pinned runtime assets verified` (11 files).

**The 0.7.10-1 pacman package on jwm1 was not built from a release
wheel** despite the wheel version stamp matching
`0.32.4.dev202610012048+6cff5ea`. The wheel pip-installed into the
package's venv contained d06222a8/71456 (RECORD proves the wheel), and the
package shipped 6e20168d/67728 (mtree proves the package). The stage was
swapped between `install.sh --system` and `package()`. The recipe
(`pkgbuilds/omarchy-mlx/PKGBUILD` in `joshuaswarren/omarchy-pkgs-ml-wt`,
an omarchy-pkgs checkout we do not own) does not contain a step that
swaps libane in either its current shape (`d4736cd`, v0.7.14) or the
v0.7.10 repin (`5a49a3fab`); `install.sh` does not either.

What did swap it: the w71 fresh-image qualification
(`apple-silicon-lab/entries/jwm1-parity/20261001T193000Z-jwm1-qualify-
local-ai-fresh-image.md`) built `omarchy-mlx 0.7.10-1 + omarchy-mlx-vulkan
0.7.10-1` LOCALLY on jwm1 with `--nocheck`, Packager "Unknown Packager",
to align the ANE surface with the v0.4.0 DKMS driver on the T8103 arm
(H210/H211/abi-smoke thread). The packaged worker sha `e51b585f`
(134352 B) ≠ the release v0.7.10 wheel's worker sha `816c096af9…` (the
abi-smoke entry records `The omarchy-mlx 0.7.10 package ships its own
mlx-omarchy-ane-worker (134 KB, sha e51b585f...)`). Both `worker` and
`libane-strict.so` were replaced in the stage against the same v0.4.0
DKMS ABI; the pin json was left untouched. The worker seal caught the
mismatch at first transcribe (H211's CLI smoke, 11:47 UTC), exactly as
designed — safety check working against an unpinned ANE userspace swap.

## Fix (pushed on origin/main via branch LibanePin)

1. `scripts/verify_runtime_assets.py` (new, 70 lines, stdlib only): for any
   `mlx/share/mlx-omarchy/parakeet-1` tree, fail with expected/actual
   digests and the legal-fix line if the pin and the bytes disagree (forward
   + strict reverse — extra unpinned file under `bundles/` or `libane/` is
   itself a failure).
2. `scripts/build-wheel.sh`: run it on the wheel tree right before `pip
   wheel` — a dirty builder tree now fails the wheel build with the
   expected/actual digests and the two legal fixes (rebuild from pinned
   omarchy-ane commit per provenance, or move the pin in the same change).
3. `install.sh --system`: run it on the staged venv share tree right after
   the wheel install (line ~101 of the `--system` branch) — closes the gap
   the w71 local-build exploited. The home install path doesn't need it:
   the release wheel there is checksum-verified at download.
4. `packaging/PKGBUILD.example` `check()`: same check against the staged
   venv; also fixed the stale `cd mlx-omarchy-$pkgver` to `cd
   omarchy-mlx-$pkgver` (the renamed repo extracts as
   `omarchy-mlx-$pkgver`; the example was the same cd that bit the w71
   local build). Real omarchy-pkgs `pkgbuilds/omarchy-mlx/PKGBUILD` lives
   in `joshuaswarren/omarchy-pkgs-ml-wt` (not ours) — see the recipe
   recipe section.
5. `overlay/tools/mlx-omarchy-parakeet/mlx_omarchy_parakeet.py`: catch the
   worker's `ResidentWorkerError` in `__main__` (it previously escaped as a
   raw traceback), parse the seal text with `_SEAL_MISMATCH_RE`, and emit
   actionable diagnosis using `_wheel_record_sha` (`importlib.metadata`) and
   the on-disk sha:

   - RECORD matches the pin, disk drifted → "installed file was modified
     after install; reinstall omarchy-mlx or rebuild the venv from the
     release wheel; never package a stage its own pin does not name."
   - RECORD matches the actual bytes → "the installed wheel itself shipped
     bytes against its own pin; the release artifact is internally
     inconsistent. Re-cut the release; the wheel build must run
     `scripts/verify_runtime_assets.py` on the share tree it ships."
   - else → mixed state; reinstall from a verified release wheel.

6. `tests/coreml/test_parakeet_runtime_pins.py`: 9 new tests cover the
   checker (accepts repo tree, rejects tampered with both digests, reports
   `MISSING`/`UNPINNED`, skips a tree without a pin manifest) and the seal
   diagnosis (regex parse, reinstall branch, broken-release branch,
   end-to-end monkeypatched advice). Also added two `pytest.skip` guards
   to pre-existing tests whose 458 MB `parakeet-encoder-whole` payload is
   gitignored and not present in a plain checkout (the release pipeline
   stages it via `MLX_OMARCHY_WHOLE_BUNDLE_DIR`).

7. `docs/parakeet.md`: one paragraph at the pin description naming the
   enforcement points and the pin-move-with-the-change rule.

**Safety semantics unchanged.** The worker seal still refuses any
`libane-strict.so` byte mismatch before device load — that is the load-
time safety check the original pin was added for. The new checks fail the
wheel build, the system staging, and the recipe check() BEFORE a tree
disagreeing with its own pin ever reaches a package; the CLI just turns
the existing refusal text into an actionable diagnosis instead of a
traceback.

## Verification

- `bash -n install.sh scripts/build-wheel.sh` ok.
- `python3 -m py_compile scripts/verify_runtime_assets.py overlay/tools/mlx-omarchy-parakeet/mlx_omarchy_parakeet.py` ok.
- `pytest tests/coreml/test_parakeet_runtime_pins.py -q`: 15 passed, 3 skipped (skip guards fire only when the 458 MB whole bundle is absent from the checkout, which is the default in a plain `git worktree add`).
- `python3 scripts/verify_runtime_assets.py /tmp/libanepin-assets/x-202610020636+…/mlx/share/mlx-omarchy/parakeet-1` (the fully extracted v0.7.14 release share tree, 11 files): `OK: pinned runtime assets verified`.
- Cross-builder libane reproducibility: not run end to end (would require a jwm1 rebuild in the w71 GPU window). The release v0.7.10 and v0.7.14 wheels were both built on the project M2 (jw14m2-linux, GCC 16.1.1 20260430) and ship byte-identical libane with the same BuildID `73750a9d049662538a6a7699b2c64e95957f79c3` — a binary sha pin IS the right guard against release-tree drift; the defect was not a builder-nondeterminism issue, it was a recipe-side stage swap that the new check refuses.

## Recipe instructions (omarchy-pkgs — not ours)

In `pkgbuilds/omarchy-mlx/PKGBUILD` (omarchy-pkgs repo):

- Make `check()` refuse a stage whose ANE bytes disagree with the pin.
  Add at the end of `check()`:
  ```bash
  python3 "$srcdir/../omarchy-mlx-$pkgver/scripts/verify_runtime_assets.py" \
    "$srcdir/stage/usr/lib/omarchy-mlx/venv/lib/python3.14/site-packages/mlx/share/mlx-omarchy/parakeet-1"
  ```
  (the script is shipped in the `omarchy-mlx-$pkgver` repo tarball
  source). The example in `packaging/PKGBUILD.example` now demonstrates
  this.
- Never swap `libane-strict.so` or `mlx-omarchy-ane-worker` into the
  staged venv after `install.sh --system` to align with a local driver
  ABI. If the packaged ANE surface has to match a host-side ABI, that
  belongs in `mlx-omarchy`'s own wheel build (and a moved pin in the
  same change), not in a pacman-side patch. The w71 lane that built
  0.7.10-1 on jwm1 for the local-ai image qualification hit exactly
  this gap; its fresh-image qualification entry recorded the
  `--nocheck` run.
- The current `cd "omarchy-mlx-$pkgver"` in the real recipe already
  matches the renamed repo; the example was stale.

## Release needed?

**Yes.** `scripts/verify_runtime_assets.py`, `install.sh`, `build-wheel.sh`,
`PKGBUILD.example`, `mlx_omarchy_parakeet.py`, the tests, and the docs all
changed; a release ships the wheel/install surface that uses the new
checks. Without the release, the jwm1 failure recurs the moment someone
re-builds the 0.7.10-1 lineage.

## What the v0.7.14 release assets contain

`mlx_omarchy-0.32.4.dev202610020636+07729f40-cp314-cp314-linux_aarch64.whl`
(sha `60f175e0…`, 416258372 B) plus `omarchy-mlx-vendor-wheels-v0.7.14-
cp314-aarch64.tar` (sha `24a5bef3…`, 461578240 B). The wheel's share tree
contains 11 files: libane-strict.so (`d06222a8…`), pin json, the three
island bundles (manifest + program-0/1.anec), and the whole-encoder
parakeet-encoder-whole bundle (manifest + 458 MB program-0.anec). The
checker verifies all 11 against `parakeet-runtime-pin.json` with no
mismatches. A package-less `pip install` of the wheel into a fresh venv
on an aarch64 host with `/dev/accel/accel0` and the staged whole bundle
will not trip the seal at first transcribe; the v0.7.14 cut is clean.

## Artifact manifest

`apple-silicon-lab/artifacts/LibanePin/libane-pin/SHA256SUMS`:
- `60f175e0…` v0.7.14 aarch64 wheel (extracted share tree passes the checker)
- `827464321d55…` `parakeet-runtime-pin.json` extracted from v0.7.14
- `16a14856…` v0.7.10 aarch64 wheel (same `d06222a8` libane; included for
  the "release wheels are consistent" claim)
- `24a5bef3…` v0.7.14 vendor tarball (461578240 B; not extracted — `make_vendor`
  carries the v0.7.14 SHA lock from the release notes)
- `14698f9a…` `scripts/verify_runtime_assets.py` (the fix)