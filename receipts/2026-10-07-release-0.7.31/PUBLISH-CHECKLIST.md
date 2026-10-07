# v0.7.31 publish checklist (execute only on Main's GO; drafts prepared, nothing published)

## Pre-publish state (2026-10-07)
- Draft release v0.7.31 exists on joshuaswarren/omarchy-mlx (isDraft: true) with 4 assets, all `sha256sum -c SHA256SUMS` OK (receipt: publish/PKGBUILD-verify.md).
- Draft PR omacom/omarchy-pkgs #839 (branch mac/v0.7.30-bump) retargeted to 0.7.31: pkgver 0.7.31, wheel/vendor/tag shas verified above, mesa pin bbbfa36dce7 + tarball sha 7677613c… unchanged. Prepared PKGBUILD text: publish/PKGBUILD-pkgbuild-v0.7.31.
- Draft PR stays DRAFT (not ready, no build-approved label) until the GateM2/GateJw16 reruns report GREEN on the v0.7.31 wheel and Main says GO.

## Publish sequence (in order, stop on first failure)
1. `gh release edit v0.7.31 --repo joshuaswarren/omarchy-mlx --draft=false --latest`
2. Post-publish asset verification:
   `python3 scripts/verify-release-assets.py v0.7.31` (from a tag checkout; confirm names + shas match SHA256SUMS over the public URLs)
3. Public install test (fresh venv, public URL only):
   `curl -fsSLO https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.7.31/omarchy-mlx-vendor-wheels-v0.7.31-cp314-aarch64.tar` etc., venv install with `--no-index --find-links`, import mlx.core, run the digest probe.
4. `gh pr edit 839 --repo omacom/omarchy-pkgs --title "omarchy-mlx 0.7.31 (+ honeykrisp-omarchy-v3 ICD pin bbbfa36dce7)"` and mark ready ONLY after Main's explicit GO + the build-approved label decision (owner: Main/Ryan per omarchy-pkgs rules).
5. Land publish receipts in receipts/2026-10-07-release-0.7.31/ on main (fetch+rebase, never force).

## Rollback
- `gh release edit v0.7.31 --draft=true` re-hides the release (assets stay).
- #839 stays draft; nothing to roll back on omarchy-pkgs until merge.
