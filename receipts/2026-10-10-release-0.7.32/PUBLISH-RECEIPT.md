# v0.7.32 publish receipt (2026-10-10, all times UTC)

Freeze commit: `ae5d0950bc4f2d47843c65112a212e21eb5e0060`. Gate results: `GATE-SUMMARY.md` in this directory.
Release notes as published: `NOTES.md` in this directory.

1. **Preflight.** The freeze commit is an ancestor of `origin/main`. `sha256sum -c SHA256SUMS` passed for the wheel, the
   vendor tar and the sidecar. No `v0.7.32` tag and no release existed.
2. **Tag.** Annotated tag `v0.7.32` (object `90ebe012197096aa1504a7f6157cb47468bd7d96`) pushed with a normal push.
   `git ls-remote origin refs/tags/v0.7.32^{}` returns `ae5d0950bc4f2d47843c65112a212e21eb5e0060`.
3. **Draft.** `gh release create v0.7.32 --draft --verify-tag` with four assets: the wheel (421,859,122 bytes), the
   vendor tar (466,933,760 bytes), its `.sha256` sidecar (118 bytes) and `SHA256SUMS` (383 bytes).
4. **Asset check on the draft.** The downloaded draft files are byte-identical to the files the gates ran (`cmp`).
   `python3 scripts/verify-release-assets.py v0.7.32 --platforms linux_aarch64` printed
   `VERIFIED: every uploaded asset matches what the release claims`. Every line was PASS: sha256 against
   `SHA256SUMS`, version across filename, `dist-info` and `METADATA` (`0.32.4.dev202610101057+ae5d0950`), build commit
   `ae5d0950` equals the tag, feature strings of a stable build, platform tag `linux_aarch64`. The platform override
   is recorded in the run because the release ships an aarch64 wheel only, as v0.7.31 did.
5. **Publish.** `gh release edit v0.7.32 --draft=false --latest` at 2026-10-10T20:18:25Z. `releases/latest` returns
   `tag_name=v0.7.32`.
6. **Public download check.** The wheel, the vendor tar and the sidecar were downloaded from the public release URLs.
   `sha256sum -c SHA256SUMS` reports OK for all three.

Wheel sha256 `a91fbe5ec2a8e4c197106d749739f696b2689f5926a97349d17e8827cc2f4712`.
Vendor tar sha256 `f0fcdf0fb0bd67f3898df338b460afcc9af9de1281e96c12c82164bca1539db0`.

## What this receipt does not cover

- The notes were checked with the hard lint for dashes, banned words and bold density, and passed those checks. The
  average sentence length check fails because the lint merges hard-wrapped list items into one sentence. No
  rewrite pass ran over the notes.
- The release has no x86_64 wheel.
- The Arch package update for this release (omarchy-pkgs #839) is a separate change. It pins the Mesa commit the
  gates ran on, `e7631595df6281748ea5e643d74db59c5f783b01`.
