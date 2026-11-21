# omarchy-mlx v0.7.49 release receipt

Tag `v0.7.49` points at `bf5631dc34b26644d16901383b9fbde19c838bff`, the same
commit both wheels name. `RELEASE-NOTES.md` is the published release body.
The release was drafted first, the asset gate ran against the draft's
uploaded bytes, and it was published only after that gate printed its
verdict (published 2026-11-21 06:23 UTC).

## Assets (SHA256SUMS of the release)

```
d41df12441b34b4e90b27283ea52034e2d53a94db2a54119acfbbf38c4caac75  mlx_omarchy-0.32.4.dev202611210550+bf5631d-cp314-cp314-linux_aarch64.whl
1dac6c95ca7b43547715bbbab632b86b1544ea1ed6f891c45609648ee292f38c  mlx_omarchy-0.32.4.dev202611210601+bf5631d-cp311-cp311-linux_x86_64.whl
45981c07da273f4d0b1a1f997f188c814eb8d334b1d5ace144763851d6b25d51  omarchy-mlx-vendor-wheels-v0.7.49-cp314-aarch64.tar
8ea4058ed4bc9ebea9701c6ed4d6d795c544ca23d515c5b354ecacbce5313d44  omarchy-mlx-vendor-wheels-v0.7.49-cp314-aarch64.tar.sha256
d41df12441b34b4e90b27283ea52034e2d53a94db2a54119acfbbf38c4caac75  omarchy-mlx-wheel-cp314-aarch64.whl
```

## Gate record (release-gates g1–g15)

The release-assets gate (`scripts/verify-release-assets.py v0.7.49`, the
script from the tag) ran against the uploaded draft bytes on the aarch64
build host and printed: `VERIFIED: every uploaded asset matches what the
release claims`. The published body's Gates section carries that verdict
with the same checks: every asset hash matches `SHA256SUMS`, both wheels
carry the tag commit in their version and in the library, the fixed-name
aarch64 copy is byte-identical to the aarch64 wheel, both required
platforms are covered, and the Mesa pin is in the tag.

The v0.7.49 gates battery (g1–g15) covers the runtime gates that need a
machine; for this release the battery did not run, because the release
window was moved up before the battery could be scheduled. The published
`RELEASE-NOTES.md` carries the same scope statement. Recorded outcomes:

- g1–g8, g9, g13, g15: **not run** for this release. The last full
  battery on record is v0.7.46
  (`receipts/2026-11-15-v0.7.46-release/README.md`), and the asset gate
  above is the only gate status carried for v0.7.49. This is a scope
  statement, not a pass.
- The released aarch64 wheel was exercised on hardware after publishing:
  a guarded inventory-and-default-route row on the 13-inch M1 installed
  from the same uploaded wheel (provenance `verified=match`, harness
  `bf5631d`), reported `PASS inventory_and_bm32_default_route` with zero
  GPU timeouts and zero foreign GPU users. Its guard log stays in the
  host queue records; the committed decode logs below are the
  measurement receipt this directory adds.
- The pinned decode on the uploaded asset is its own measured section
  below and the `logs/decode-001/` records.

## Measured decode on the uploaded aarch64 asset

Run on the 13-inch M1 (the same machine and recipe as v0.7.46 and
v0.7.48), from the draft asset set downloaded before publishing and checked
strictly against the release `SHA256SUMS`, in a fresh virtual environment
built from the uploaded vendored wheels. Pinned model snapshot
`SiddhJagani/Qwen3.8-2B-mlx-4Bit` at revision
`0867d98bfb174b042d88461c0e7c97b86b34b381`, hash-verified file by file at
staging (its `tokenizer.json` and `model.safetensors` sha256 match the
staging script's pins). Greedy, seed 0, 128 tokens requested, Mesa
`439dc032d7`. Guarded: zero GPU timeouts before and after; no foreign GPU
user in the window.

| Run | decode (tok/s) |
|---:|---:|
| 1 | 28.9075 |
| 2 | 29.1198 |
| 3 | 29.2155 |
| 4 | 29.0785 |
| 5 | 29.0394 |

- Median 29.0785 tok/s over five runs (redacted run logs under `logs/`,
  `logs.sha256` carries their sha256 values; the originals stay in the lab
  queue records). Every run returned the same token-id digest
  `775bf034e185c8f9` with first ids 760, 1156, 369 and last 248046, 198,
  248046, matching the published notes.
- Prefill per run: 0.448 s to 0.452 s (the logs carry the exact values;
  the published notes quote run 1).
- The two-token pilot (2 tokens, digest `e9540e197675516d`) is committed
  as `logs/decode-001/pilot.log`.
- The vendored mlx-lm serve series was applied to a second environment and
  measured there too (`logs/decode-patched-001/`): 8.08, 29.94, 45.82,
  17.08 and 20.54 tok/s with the same token digest, and a second staged
  attempt never finished its pilot (its empty pilot log rides the queue
  record on the host; the committed logs carry the five series runs).
  The published notes record this as an open item, not a release number,
  and the run logs here back those quotes.

## Known limit: speech on M6

Speech is not qualified on M6 for this release either. The gate requires
synthesis in under 5.5 seconds; the v0.7.48 measurement of 13.05 seconds on
Mesa `12e05a5250` still describes the M6 state, and the published notes
carry that limit.
