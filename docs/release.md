# Cutting a release

Releases are cut by hand. One rule produced this procedure. Nothing
verifies the artifact after upload by default. A local check does not
verify what users download. Three releases in one day (2026-09-03) needed
manual static-diff investigations because of it. The same day, v0.3.3 was
first published with only the x86_64 wheel. Its metadata was correct and
the gate was green. The wheel was unusable on every Apple Silicon machine
the project serves. So the procedure now names the required platforms, and
the gate enforces them.

1. Pick the commit. Tag only what you verified. If a performance or
   correctness commit lands after your target commit, re-cut from the new
   commit. Or accept a slower release on the record.

2. Build EVERY required platform. A stable release carries both:

   | platform | wheel | built on |
   |---|---|---|
   | linux_x86_64 | `cp311` | the dev box, `scripts/build-wheel.sh` |
   | linux_aarch64 | `cp314` | an aarch64 Apple Silicon host (natively, with `glslc` and the pinned whole-encoder ANE bundle), same script from a detached worktree at the tag commit |

   Cutting on the dev box alone produces only the x86_64 wheel. That wheel
   does not run on Apple Silicon, the project's target hardware. scp the
   aarch64 wheel to the dev box for upload. Re-verify its sha256 after the
   transfer. Build with `scripts/build-wheel.sh`. Add `--diagnostics` only
   for a `-diag.*` prerelease. That build requires only the aarch64 wheel.
   The receipt prints the source commit the wheel is stamped with. Record
   wheel name, bytes, sha256, and the installed version string in the
   receipt. Development versions use the prepared `setup.py` modification
   time in UTC. `SOURCE_DATE_EPOCH` wins when it is set. Metadata and wheel
   build hooks then agree. Check this with
   `python3 scripts/check-version-stability.py .work/mlx`. A stable release
   also carries the vendored wheel set for the system install.
   `packaging/vendor-wheels.sh` runs once on an aarch64 `cp314` host. The
   result ships as `omarchy-mlx-vendor-wheels-v<TAG>-cp314-aarch64.tar`
   plus a `.sha256` sidecar. `SHA256SUMS` (flat filenames) must cover every
   uploaded asset.

3. Create the GitHub release as a DRAFT. Upload the assets to it. Run the
   gate and the installed-from-release gates against the DRAFT's assets.
   The distro packager watches published releases with a 24 h hold. Publish
   ONLY after every gate passes. Never publish and fix afterwards.
   `gh release view/download` resolve drafts for the owner.
   `scripts/verify-release-assets.py` checks the draft's bytes. That
   includes the SHA256SUMS coverage of every uploaded asset. The installer
   on a draft resolves the tag's scripts and patches through
   raw.githubusercontent.com once the tag is pushed. It takes the release
   assets from a local directory:

   ```bash
   MLX_OMARCHY_VERSION=<tag> \
   MLX_OMARCHY_RELEASE_BASE="file://$(pwd)/draft-assets" \
     bash install.sh
   ```

   Fill `draft-assets/` with `gh release download <tag>` first.

4. Run the gate against the UPLOADED (draft) bytes:

   ```bash
   python3 scripts/verify-release-assets.py <tag>
   ```

   The gate downloads every asset from the release. It checks the sha256
   against `SHA256SUMS`, then the release notes or `receipts/`. It checks
   the version metadata against the filename and any recorded version. It
   checks the stamped build commit against the tag's commit. It checks the
   feature strings. A `-diag` release must carry the profiling harness in
   `libmlx.so`. A stable release must not. A positive control keeps an
   empty result from reading as a pass. The gate also FAILS a release whose
   wheels do not cover the required platforms above. It fails a release
   whose `SHA256SUMS` does not cover exactly the uploaded asset set. It
   checks each wheel's filename platform against its `dist-info/WHEEL` Tag.

5. The release is announced as usable only after the gate prints
   `VERIFIED`. The installed-from-release gates must also pass on the
   draft. `VERIFIED WITH FINDINGS` is acceptable only for the traceability
   finding. That finding covers assets built before the stamping change.
   Publish with `gh release edit <tag> --draft=false`. Then promote with
   `gh release edit <tag> --prerelease=false --latest`. Verify the
   promotion through the API.

6. Run one pinned 4-bit decode on the UPLOADED aarch64 asset. Install it
   from the release on Apple Silicon hardware. Put that number in the
   notes. The gate checks hashes, platforms and feature strings. None of
   that sees speed. v0.3.4 passed the gate and shipped 4-bit decode at
   0.21 tok/s. That was 70x below the number in its notes. That number had
   been measured on the commit before the one that was tagged
   (`receipts/2026-09-04-v0.3.4-decode-regression.md`).

7. Write the dated receipt under `receipts/`. Include the gate output and
   the decode number.

8. For runtime measurement work, install the exact wheel under test. Then
   run `scripts/mlx_provenance.py`, or let `scripts/bench_decode.py` do it.
   A run whose loaded `libmlx.so` does not match the installed wheel's
   RECORD refuses to emit a number. Never pass a requirements input
   containing an mlx-omarchy pin to provisioning.
   `scripts/check-wheel-pins.py` rejects them in any form.
