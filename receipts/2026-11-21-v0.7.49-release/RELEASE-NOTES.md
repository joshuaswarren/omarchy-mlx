# omarchy-mlx v0.7.49

Draft. Entries accumulate as changes land on main; the release owner fills
the asset, gate and measured sections at cut time.

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
- **The fused masked batch-2 BF16 decode route is the default (#531,
  already in v0.7.48's notes as shipped code; its default flips here).**
  `MLX_OMARCHY_SDPA_DECODE_BOOL_MASK=0` restores the composed route.
  Measured on the 16-inch M1 Max over five AB and BA pairs: two streams
  53.895 -> 68.934 tokens per second, token digests identical.
- **The backend accepts the Borealis driver name, id, ICD file and env
  prefix (#532)** so installs on the Borealis stack pick our ICD instead
  of failing the system-ICD retry.
- **Batching under mlx-lm 0.31.3 preserves host RoPE offsets (#520)** so
  batched decode keeps the exact single-request rotation behavior.
- **Packaging fixes (#540-#543):** the wheel ships the GCC runtime libs
  beside the vendored OpenBLAS and pins the library directory to `lib`
  (refusing `lib64`); the qualification kit imports the r3/r4 sources,
  pins the r4 wheel and keeps the suite running past a timeout or abort.

Source: `57486b3cc` (main at draft time). Mesa qualification pin: unchanged from
v0.7.48 (`439dc032d70ab9ec7e3b40afd4efd920f75c56f1`) unless the release owner
repins.

## Asset, gates, measured numbers

To be filled at cut time by the release owner: wheel asset and SHA-256,
gate statuses (release-gates g1-g15 on the release wheel), and the
per-machine decode/prefill medians, following the v0.7.45/v0.7.46/v0.7.48
receipt conventions.

## Draft notes

- Cut point: merged main `57486b3cc`; the qualification-kit r4 pre-release
  (tag `qualification-r4`, cut from the same commit) already exercises the
  kit side.
- Version: **v0.7.49** follows v0.7.48 with no skipped number;
  `scripts/check-version-stability.py` stays the release-owner check.
- PR #538 (the CBQ Apple GPU receipt) should merge before or with the cut
  so the release receipt can cite its rows as landed evidence.
