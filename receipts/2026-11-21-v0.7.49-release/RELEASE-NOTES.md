# omarchy-mlx v0.7.49

Draft. Entries accumulate as changes land on main; the release owner fills
the asset, gate and measured sections at cut time.

## What changed

- **Nine CBQ custom kernels run on the Vulkan backend (#524).** Seven gather
  kernels pass the per-element comparison. The two gradient kernels pass the
  `1e-5` relative-L2 bound on the 16-inch M1 Max. The Vulkan port keeps the
  source, buffer signature and launch checks for all nine kernels.
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

To be filled at cut time by the release owner: wheel asset and SHA-256,
gate statuses (release-gates g1-g15 on the release wheel), and the
per-machine decode/prefill medians, following the v0.7.45/v0.7.46/v0.7.48
receipt conventions.

## Draft notes

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
