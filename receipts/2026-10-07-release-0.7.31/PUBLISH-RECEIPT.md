# v0.7.31 publish receipt (2026-10-07, all times UTC)

Executed per PUBLISH-CHECKLIST.md on Main's pre-authorization (w7K GREEN at 8ae198be7).

1. Pre-publish verification: `python3 scripts/verify-release-assets.py v0.7.31 --platforms linux_aarch64` -> rc=0, "VERIFIED: every uploaded asset matches what the release claims". Tag assert: `git ls-remote origin refs/tags/v0.7.31` -> annotated tag object 9f7c710a2f -> commit 9b5c938fe236e767df9545de301cfc74fdbc3395 (git cat-file confirms type commit, target 9b5c938fe).
2. Publish: `gh release edit v0.7.31 --draft=false --latest` -> released. `gh release view` -> {"isDraft": false, "publishedAt": "2026-10-07T21:13:25Z", "tagName": "v0.7.31"}; `gh api releases/latest` -> tag_name v0.7.31. (isLatest JSON field does not exist in gh; releases/latest is the authority.)
3. Publish-time install test from PUBLIC URLs:
   - Dev box (x86_64): sha256 of the downloaded wheel == a2f8c83e5c5f635d00702565a8557d87c40a9eccac885329dcec4692d85d300d; python-level import OK, version 0.32.4.dev202610071347+9b5c938 (native extensions are aarch64; no matmul there).
   - M1 base host (aarch64, 12.2): public-URL wheel sha256 prefix a2f8c83e5c5f635d0070 (full sha == a2f8c83e...00d), pip install into a fresh venv rc=0, import OK, version 0.32.4.dev202610071347+9b5c938, CPU matmul OK 4096.0.
4. omarchy-pkgs #839: `gh pr ready 839` -> marked ready for review. `--add-label build-approved` -> PERMISSION DENIED ("joshuaswarren does not have the correct permissions to execute AddLabelsToLabelable"). Not retried with other tokens, per instruction. The label needs an omacom-side maintainer.
5. Release notes: appended "Known pending at publish time" section naming jw16 g-g13c leg B on the v0.7.31 wheel as supplemental (leg P + negative control are in receipts).
