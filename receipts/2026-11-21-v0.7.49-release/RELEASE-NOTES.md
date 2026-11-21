# omarchy-mlx v0.7.49

Release receipt for v0.7.49. Cut point `86fcc43cef`; assets, gates and the
measured decode are recorded below.

## What changed

- **Nine bounded CBQ custom kernels now dispatch on the Vulkan backend
  (#524).** The shipped CBQ inventory (seven gather kernels and the two
  upstream atomic gradient kernels) translates from the byte-identical
  shipped MSL, validates as SPIR-V without int8/8-bit-storage or atomics,
  and dispatches on the Apple GPU. Runtime receipt on the 16-inch M1 Max
  (receipt `2026-11-21-cbq-applegpu`, PR #538): the seven gather kernels
  pass per element against the integer oracles; the two atomic grad
  kernels pass upstream's own relative-L2 bar (3.6e-08 both, bar 1e-5)
  with two-boot, four-sample exact reproducibility; float32 and float64
  CPU references agree exactly. The signed-lookup-bytes oracle fix rides
  the same patch. The ABI/launch refusals stay fail-closed: near-match
  bodies and contract mutations still refuse (17-case CI acceptance in
  the syntax gate).
- **Opt-in BM32 BF16 GEMM for the Nemotron Labs Diffusion projections
  (#523).** 32x64x16 tile, FP32 accumulation, round-to-nearest-even BF16
  stores, complete-tile dense NT gate with
  `MLX_OMARCHY_MATMUL_BM32=1`; other shapes keep the general matmul
  route. The projection probe pins bit-exact integer-oracle outputs and
  exactly one (dense) or four (chunked) dedicated dispatches.
- **Fused masked batch-2 BF16 decode is on by default (#531).** This uses
  the fused route introduced in v0.7.48 (#521).
  `MLX_OMARCHY_SDPA_DECODE_BOOL_MASK=0` restores the composed route.
  Measured on a 16-inch M1 Max over five AB and BA pairs: two streams
  53.895 -> 68.934 tokens per second, token digests identical, zero GPU
  timeouts and no other GPU in use. The receipt is
  `receipts/2026-11-20-fused-bool-mask-decode/`.
- **The backend accepts the Borealis driver name, id, ICD file and env
  prefix (#532)** so installs on the Borealis stack pick our ICD instead
  of failing the system-ICD retry.
- **Batching under mlx-lm 0.31.3 preserves host RoPE offsets (#520)** so
  batched decode keeps the exact single-request rotation behavior.
- **Packaging fixes (#540-#543):** the wheel ships the GCC runtime libs
  beside the vendored OpenBLAS and pins the library directory to `lib`
  (refusing `lib64`); the qualification kit imports the r3/r4 sources,
  pins the r4 wheel and keeps the suite running past a timeout or abort.
- **The mlx-lm BitNet CPU fallback keeps FP32 scaling (#548).** This is
  the first stable release that ships the fix. The installed series that
  already carry the earlier patch get a separate precision migration
  patch, and the upgrade path is idempotent: a second install skips the
  patch instead of applying it twice. Both installer regressions are
  green in the pinned series.
- **The qualification kit captures and compares native generated token
  IDs (#546)**, so a kit run compares the tokens the runtime produced
  instead of a rendered string.

This is the first release that ships the BitNet fix, the nine CBQ kernels
and the BM32 route. The BM32 route stays off by default and is enabled
with `MLX_OMARCHY_MATMUL_BM32=1`.

Source: main `86fcc43cef` (PR #548 the mlx-lm BitNet FP32 fallback
correction and PR #546 the native generated token IDs). The release commit
adds only this receipt. Mesa qualification pin: unchanged from v0.7.48
(`439dc032d70ab9ec7e3b40afd4efd920f75c56f1`) unless the release owner
repins.

## Asset, gates, measured numbers

Assets (published 2026-11-21 06:23 UTC, tag `v0.7.49`): both wheels carry the
cut point in their version and in the library; the fixed-name copy is
byte-identical to the aarch64 wheel; the vendor tar and its checksum ride the
same release. The release-assets gate printed `VERIFIED: every uploaded asset
matches what the release claims` against the uploaded bytes.

The g1-g15 runtime gates battery did not run for this release. The
release-assets gate is the only gate status carried for v0.7.49, and the
last full battery on record is v0.7.46
(`receipts/2026-11-15-v0.7.46-release/`). This is a scope statement, not
a pass.

Measured decode on the uploaded aarch64 asset: one pinned 4-bit decode
installed from the uploaded vendored set on a 13-inch M1 MacBook Pro, model
Qwen3.8-2B-mlx-4Bit at revision `0867d98bfb174b042d88461c0e7c97b86b34b381`,
greedy, seed 0, Mesa `439dc032d7`. Five 128-token runs: 28.91, 29.12, 29.22,
29.08 and 29.04 tokens per second, a median of 29.08; every run returned the
same token id digest `775bf034e185c8f9`; prefill 0.45 s, reported separately.
The loaded wheel reports `verified=match` and `harness=bf5631d`. The same
recipe with the vendored mlx-lm serve patch series applied is not a stable
number on this machine and is recorded as an open item, not a release number.

## Release notes

- Cut point: merged main `86fcc43cef` (PR #548 the BitNet FP32 correction
  and PR #546 the native token IDs). The release commit adds this receipt
  and nothing else.
- Qualification kit: this release ships the kit as it stands on main
  `86fcc43cef`, which includes PR #543 (the r4 wheel pin and the measured
  reference), PR #545 (results are kept after a GPU correctness failure)
  and PR #546 (native generated token ids are captured and compared). The
  `qualification-r4` pre-release tag was cut from `57486b3ccd`, which is
  the tree of PR #543 alone, so that tag does not carry PR #545 or PR #546.
  A kit pre-release for this tree has to be cut separately.
- Version: **v0.7.49** follows v0.7.48 with no skipped number;
  `scripts/check-version-stability.py` stays the release-owner check.
- PR #538 (the CBQ Apple GPU receipt) is merged, so the release receipt
  can cite its rows as landed evidence.
- The fused decode numbers in the notes come from
  `receipts/2026-11-20-fused-bool-mask-decode/`.
