# omarchy-mlx v0.7.48 release receipt

[Agent for Joshua Warren]

Tag `v0.7.48` points at `e78558049804aaf74552c089446d5ad80ccf36f3`. `RELEASE-NOTES.md` is the published body. The release was drafted first, every gate ran against the draft's assets, and it was published only after all of them passed (published 2026-11-21 00:44 UTC).

## Assets (SHA256SUMS of the release)

```
834d08a92b68aec08cf092a045f0c2155ee2a088d9654e3905ef3579de4b05bb  mlx_omarchy-0.32.4.dev202611202212+e78558049-cp314-cp314-linux_aarch64.whl
92e840f1c5498fd6e5b1af3239509db0f1e44a61a97d58bd6e4a46519efd77ac  mlx_omarchy-0.32.4.dev202611210023+e78558049804aaf74552c089446d5ad80ccf36f3-cp311-cp311-linux_x86_64.whl
6ab0af6b2d75284103362fea3e945e5cfe0a0dd1c0357f0c9f5dd62cab692ed1  omarchy-mlx-vendor-wheels-v0.7.48-cp314-aarch64.tar
bde3e374320ff428e77fe89cdd2f8641db9189fe94e5614560aa05ae2ef42345  omarchy-mlx-vendor-wheels-v0.7.48-cp314-aarch64.tar.sha256
834d08a92b68aec08cf092a045f0c2155ee2a088d9654e3905ef3579de4b05bb  omarchy-mlx-wheel-cp314-aarch64.whl
```

- `scripts/verify-release-assets.py v0.7.48` (the script from the tag) run against the draft's uploaded bytes printed: `VERIFIED: every uploaded asset matches what the release claims`. It checked every asset's sha256 against `SHA256SUMS`, the version in the filename and in `dist-info/METADATA`, the stamped build commit against the tag, the feature strings, the platform tags, and that the fixed-name copy is byte-identical to the aarch64 wheel.
- The aarch64 wheel, the vendor tar and the fixed-name copy were built on an aarch64 Apple Silicon host from the tag's tree. The x86_64 wheel was built on a Debian 12 host with Python 3.11.2 from the same tag commit, with the whole-encoder bundle taken from the verified aarch64 wheel and `cblas.h` staged from `libopenblas-pthread-dev` 0.3.21. Its build tree was deleted after the wheel was hashed.

## Decode on the uploaded aarch64 asset

Run on a 13-inch M1 MacBook Pro under the pinned Mesa build `439dc032d7`, from the installed release asset, with the wheel's own provenance verified as a match for the tag.

- median 29.23 tokens per second over four runs of 128 tokens (29.10, 29.22, 29.23, 29.28)
- all four runs returned the same 128 token IDs (`sha256:775bf034e185c8f9`)
- prefill 0.44 s per run, reported separately from decode
- GPU timeouts: 0 before and after; foreign GPU users: 0

## Known limit: speech on M6

Speech is not qualified on M6. Kokoro produced 4.00 seconds of audio in 13.05 seconds on Mesa `12e05a5250`. The gate requires synthesis in under 5.5 seconds. The published notes carry that limit.
