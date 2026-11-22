# G402 staging notes — qualification kit r5 (WHEEL_TAG bump)

Branch (staged): `w71/qual-kit-r5` off `main` 79d99c90c1, in
`/tmp/mo-r5` (commit lands when the r5 wheel sha arrives from the jw16
build row 6338d909).

## The diff

1. `qualification-kit/qualify.py` — `WHEEL_TAG = 'qualification-r5'`
   (was `qualification-r4`).
2. `qualification-kit/m1-max-reference.json` — `"release": "qualification-r5"`
   (the runner's consistency check `baseline.get('release') != WHEEL_TAG`
   requires this to match), and `wheel_sha256` → the r5 wheel (filled from
   the build receipt; r4's `05a4802b…` retained in git history).
3. `qualification-kit/README.md` — "Linux verifies the r4 wheel" → r5.

Not carried, per the card's "if merged" clause: `#611` (card-row print) and
`#608` (Borealis ICD manifest discovery) are both still OPEN on
joshuaswarren/omarchy-mlx — they land in r6, not r5. The E1 scalar-load fix
chain (`#561/#570/#574/#578/#580`) is already in main at the pin
(`b9017ff660` is an ancestor of 79d99c90c1), so the r5 wheel carries it by
construction.

## TAG vs WHEEL_TAG (so nobody reads the dry-run wrong)

`qualify.py` has two tag constants: `TAG = 'qualification-r6'` is the
receipt schema/contract version and appears in receipts as `release`;
`WHEEL_TAG` is the release wheel the Linux arm verifies. The dry-run receipt
below says `release: qualification-r6` — that is the contract version, not
a stale wheel tag.

## CPU dry-run (this host: omp-studio-local, a Proxmox VM, no Apple silicon)

- `python3 qualify.py --machine-id 0123456789abcdef --out …` → exit 2,
  `set --icd to the installed Apple Vulkan JSON manifest` (the Linux-arm
  gate fires as designed on a host without an Apple ICD).
- With a stub ICD: exits 1, receipt written with `status: incomplete`,
  `error: apple_chip_unidentified` — args parse, the ICD gate passes, the
  chip probe refuses where it must. The staged diff is live through the
  whole preflight chain; everything past the chip probe needs real Apple
  hardware, which is G403's reference run on jw16.

python3.14 is not installed on this studio host, so the cp314 wheel cannot
be built here; the wheel build runs on jw16 (python3.14.7, the established
`scripts/build-wheel.sh` path), under guard row 6338d909.

Wheel build provenance: `MLX_OMARCHY_WHOLE_BUNDLE_DIR` points at
`~/scratch/w7m-r49/bundle/parakeet-encoder-whole` on jw16 (manifest.json +
program-0.anec; compiler pinned at `6a963f975ae1`, target h13,
driver_abi_major 1) — the same bundle layout the gate demands. First build
attempt (05:0xZ) refused with "runtime pin declares parakeet-encoder-whole
but MLX_OMARCHY_WHOLE_BUNDLE_DIR is unset"; the env is the fix, and the
bundle's currency itself is G408's audit, not this staging's.

## Deliverables

- Staged diff: this branch (commit + sha when the wheel sha lands).
- Dry-run logs: `q-dryrun.log` (ICD-gate refusal) and `q-dryrun2.log` +
  `q-dryrun2.json` (chip-probe refusal) committed under
  `receipts/2026-11-22-qual-kit-r5-staging/`.
- SHA256SUMS for the staged artifacts accompanies the commit; the release
  `SHA256SUMS` covering the uploaded wheel is created at publish time by
  G403 (w7B) per `docs/release.md`.
