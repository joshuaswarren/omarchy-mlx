# G402 staging notes — qualification kit r5 (WHEEL_TAG bump)

Branch (staged): `w71/qual-kit-r5` off `main` 79d99c90c1, in
`/tmp/mo-r5` (commit lands when the r5 wheel sha arrives from the jw16
build row 6338d909).

## The diff

1. `qualification-kit/qualify.py` — `WHEEL_TAG = 'qualification-r5'`; the
   Linux arm's release-asset pin moves to the r5 wheel: `WHEEL` names the
   built artifact (`…dev202611220732+c15a0d4f2…`) and `ASSETS[WHEEL]` is its
   sha256 `f78eff3d…cc4bc769`. The vendor-wheel pin (`omarchy-mlx-vendor-
   wheels-v0.7.47-cp314-aarch64.tar`, `31a9ed20…`) is unchanged. This is
   what the first refresh attempt exposed: without it the kit still
   verified the r4 asset and curled a nonexistent r5 release URL.
   (was `qualification-r4`).
2. `qualification-kit/m1-max-reference.json` — `"release": "qualification-r5"`
   (the runner's consistency check `baseline.get('release') != WHEEL_TAG`
   requires this to match), and `wheel_sha256` → the r5 wheel (filled from
   the build receipt; r4's `05a4802b…` retained in git history).
3. `qualification-kit/README.md` — "Linux verifies the r4 wheel" → r5.

Not carried, per the card's "if merged" clause: none — `#608` (Borealis
packaged Vulkan ICD manifest discovery, `f06d179190`) and `#611`
(card-row print + `--row`, `4223c1fb4d`) both merged between staging and
the wheel build, so the branch rebased onto the built pin `c15a0d4f24`
("pipeline ANE island submits…", #613) and r5 carries both. The E1
scalar-load fix chain (`#561/#570/#574/#578/#580`) is on main at the
pin, so the r5 wheel carries it by construction.

## The built wheel (G421/G402 close-out)

`R5WHEEL DONE head=c15a0d4f24` at 2026-11-22T07:32Z on jw16 (guard job
`mlx-wheel-build-r5`, class build/no-gpu, kernel 7.1.12-2-20261103.2-lab-ARCH,
no foreign GPU usage per the receipt header; 120 patch hunks applied from the
staged whole-encoder bundle; both staging env vars present — attempt 2 had
refused without them). Artifact:
`mlx_omarchy-0.32.4.dev202611220732+c15a0d4f2-cp314-cp314-linux_aarch64.whl`,
423,645,975 bytes,
sha256 `f78eff3dcd1062f8dc4e3113dafc9cdcaea2fd9ddf57c526896a4755cc4bc769`,
pinned in `m1-max-reference.json`. The stale dist wheel from the aborted
attempt 2 (stamp 8d92e094b) was removed before this build and never pinned.

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

## G421: post-558 contents + the lab's timeout

`qualification-kit/qualify.py` now defaults `PER_FILE_TIMEOUT` to **900 s**
(was 300 s on main; the lab's own history went 35 → 900). Provenance: the
r3/r4 lab rows stopped at `test_quantized.py` on the short limit (exit 124)
and only completed all 34 files with 80/80 known failures and 81/81 known
skips matching once the limit was 900 s (omarchy-mlx#537, M5 Max G17C r3
row, 2026-11-20) — that was the lab's only ad hoc kit patch; no other
deviation was needed to match every reference.

The r5 wheel carries **#558** by construction: `dd58d2e33ab3c2833a32e9760f6
f8e56222d5d69` ("publish scalar loads synchronously so the evaluator fence
is truthful", merged 2026-11-21) is an ancestor of the staging pin
`bb66022715`, together with the follow-up regression tests (#561/#570/
#574/#580) — the exact fix the lab's `test_export_unique` failure was
pointing at. `QUALIFY_PER_FILE_TIMEOUT` remains an override for slower
hosts.

## G415 companion tarball (Main's instruction, for the G403 prerelease)

The #537 rerun ask ships one extra asset next to the r5 kit:
`g415-j616s-macos-baseline-v2.tar.gz` — 754,214 bytes, sha256
`75ffd0fc62aab3989a13e868286cd3907546048d7535e4303504581d950fe91f`,
staged at `~/scratch/main/g415/` beside its `.sha256` sidecar. Contents
(exact, nothing else): `j616s-macos-baseline.v2.sh`,
`encoder_bench.swift`, `j616s-macos-baseline.v2.sha256`,
`parakeet-inputs/{SHA256SUMS,feat_f32.bin,gold_f16.bin,mask_i32.bin}`.
Verified: the six internal checksums pass after extraction
(`sha256sum -c --strict`). Privacy: no private data — the only
identifying strings are a public HF repo id
(`mweinbach1/parakeet-tdt-0.6b-v3-coreml`), a public provenance sha, and
Swift syntax; the `.bin` files are numeric fixtures (mel features f32,
golden output f16, mask i32). Reason: Ryan's J616s cannot pull from our
hosts and the harness commit is not on GitHub. The v1 script
(`j616s-macos-baseline.sh`) is deliberately excluded — the .sha256 does
not cover it.

## Deliverables

- Staged diff: this branch (commit + sha when the wheel sha lands).
- Dry-run logs: `q-dryrun.log` (ICD-gate refusal) and `q-dryrun2.log` +
  `q-dryrun2.json` (chip-probe refusal) committed under
  `receipts/2026-11-22-qual-kit-r5-staging/`.
- SHA256SUMS for the staged artifacts accompanies the commit; the release
  `SHA256SUMS` covering the uploaded wheel is created at publish time by
  G403 (w7B) per `docs/release.md`.
