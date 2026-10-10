# gather_qmm tiled sorted-expert route (MoE prefill)

Source commit under test: `4dbeedf4bb2319483ebab96eb4acaf204628fc56` (two tile commits cherry-picked onto main `714cddc9f`).
Hardware: M1 Max (G13C) only. Linux kernel 7.1.12-2. Private Honeykrisp build exported through `VK_ICD_FILENAMES` (Mesa git `6543eeb7df7`, driver library
sha256 prefix `3546bcafe8ed3995`). The logs do not print the device name. Build and home paths in the logs are replaced by `<build>`, `<work>` and `$HOME`.

## Change

`MLX_OMARCHY_GATHER_QMM_TILE` (default on, `=0` turns it off). For a sorted-expert MoE prefill (`gather_qmm` with `right_sorted`, m = 1, 64 or more rows, 4-bit, group size 64, K and N
multiples of 64, f16 or bf16 output) the new `gather_qmm_tile.comp` dequantizes a weight tile once per run of up to 32 rows that share an expert, instead of once per row.
Any other case takes the existing kernels. The route also requires `max_compute_shared_memory_size` of at least 25,216 bytes (the kernel's shared tiles); a smaller limit falls back to the per-row kernels.
There is no no-bias variant: affine quantization always carries biases (`ops.cpp` refuses affine without them), and the bias-free modes are floating-point modes the gate excludes. A first draft shipped two no-bias kernels that no input could reach; they are deleted. The route is correct for any index order and fast only when the rows are sorted.

## Speed (`g13c/h64-repeated-prompt.log`, `g13c/h69-natural-text.log`)

Qwen3-30B-A3B-Instruct-2507 4-bit, prefill T = 512, wheel `0.32.4.dev202610100224+4797d4f7` (the same two tile commits on an older base), four fresh processes ON OFF ON OFF,
one warm-up and three timed prefills each.

| Prompt | ON median tok/s | OFF median tok/s | Ratio |
|---|---|---|---|
| One sentence repeated (H64) | 97.75 | 24.12 | 4.05 |
| Repo prose, 324 distinct token ids (H69) | 94.64 and 95.23 | 24.11 | 3.94 |

In H69 the second OFF arm started with load1 above 5 because another job ran on the host (the arm header logs 5.07, the provenance line 5.70, both far above the limit). The pre-registered rule voids an arm above 1.5; that arm is void and was not rerun, so the ratio above uses the
first OFF arm only (the void arm read 24.15, within 0.2 percent). The lowest ON rep (94.57) beats the highest OFF rep (24.14).

## Numerics

H69, last-position logits, ON against OFF, for both ON/OFF pairs: same argmax token and same top-5 set (`[49978, 6985, 5800, 49891, 29699]`), maximum absolute logit difference 0.3125 with
logit absolute maximum 21.0, no non-finite values. The top logit is 21.125 with the route and 21.0 without. This is one position of one prompt; it is not a perplexity or generation test.

## Tests (`g13c/tile-ticket.log`, `g13c/tile-suite.log`)

New doctest `gather_qmm sorted-expert tile route matches the f32 reference and the per-row route`: 32 assertions, 0 failed. Shapes include bf16 and f16, a shuffled (unsorted) index order,
and tile-versus-f32-reference errors of 0.0006 to 0.0048. Whole `omarchy_matmul_family_tests` (binary `191398e11eaa85d0`): 34 cases, 82,942,953 assertions, 0 failed.

## Not covered

- M1 (G13G) and M2 Max (G14C): no run. Only the doctest could run there; the speedup is measured on G13C only.
- Models other than Qwen3-30B-A3B were not run. gpt-oss-20b is MXFP4 (a floating-point mode), which the route's gate excludes.
- Prefill other than T = 512. Decode (m = 1 with few rows) does not use the route.
- Generation quality, perplexity.
- The two tile commits ran on an older base for the speed rows; the build and tests above are on the rebased head.
