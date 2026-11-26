# M1 standing battery — receipt (2026-09-30, fix-bearing rerun)

## Result
- Host completed: `<project-m1>` (M1 / T8103, 13-inch, Asahi Linux).
- Host not completed: `<project-m1-max>` (M1 Max / T6001, 16-inch). Another job held the
  machine for the whole window. Not preempted.
- All 26 listed suites pass on `<project-m1>`. 0 failures across 28 (cases) × ~97.0 M (assertions).

## Pinned source (verified, four proofs satisfied)
- Source commit: `db74f11adfb2c6fcee674d46635083b56a30f9c8`
  ("ModelBench: append 2B row to summary table", 2026-09-29 15:48 CDT).
  This is `origin/main` HEAD at the moment of `git fetch origin main` (2026-09-29T20:52Z).
- A fresh `git worktree add <scratch>/m1battery -b feat/m1battery origin/main` on the device:
  not another user's checkout, not a copy.

### (a) HEAD equals the recorded hash; tree is clean
```
$ git rev-parse HEAD
db74f11adfb2c6fcee674d46635083b56a30f9c8
$ git status --porcelain | wc -l
0
```

### (b) sha256 of overlay/mlx/backend/omarchy/primitives.cpp is identical before and after `scripts/prepare-mlx.sh`
```
8e71abc6a7973cef1f43631e5db0e0610732d5709a79ff8ccef315952ee2c024  overlay/mlx/backend/omarchy/primitives.cpp
... prepare-mlx.sh ...
8e71abc6a7973cef1f43631e5db0e0610732d5709a79ff8ccef315952ee2c024  .work/mlx/mlx/backend/omarchy/primitives.cpp
```

### (c) `omp_binary_op_type` appears >= 6 times in the prepared copy
```
$ grep -c omp_binary_op_type .work/mlx/mlx/backend/omarchy/primitives.cpp
6
```
The six occurrences are the function definition and the five call sites in `Binary::evaluate`,
`Binary::eval_cpu`, `Broadcast::eval_gpu` (the ternary-quotient and ternary-remainder branches),
and `Select::eval_gpu`. Matches the parent commit `3ab075ae1d37ef5c0c51c88ba192686003e46d1d`'s
`overlay/mlx/backend/omarchy/primitives.cpp | 25 ++-`.

### (d) The regression test for the fix is in the built `omarchy_primitive_tests`
```
$ ./.work/build/tests/omarchy/omarchy_primitive_tests --list-test-cases | grep -i 'col-contiguous'
col-contiguous view Add forces General output storage
```
This is the doctest case added in `3ab075ae1d37ef5c0c51c88ba192686003e46d1d`'s
`overlay/tests/omarchy/test_primitives.cpp | 84 ++++++++++`. Built into the binary. Runs as part
of `omarchy_primitive_tests` and passes (104 cases, 2,743,003 assertions, 0 failed).

## Toolchain identity (`<project-m1>`)
- `glslc 2026.3` (`/usr/bin/glslc`) — M1 has `glslc`; dev box only has `glslangValidator`.
- `libvulkan_asahi.so` from `/usr/lib/`; active ICD `asahi_icd.json.honeykrisp-7faf04c.active`.
- GCC 16.1.1 (libstdc++ shared_ptr atomic deprecation warnings are cosmetic; no test failures).
- mlx backend version 0.32.3, mlx commit `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`,
  archive sha256 `425905d1c2b7c21c35cb86f0eeb5b6acfe5ae55ff7aa4413d002f39508f433a6` (locked in
  `mlx.lock`; `sha256sum --check --status` verified at prepare time).
- Build flags: `MLX_BUILD_TESTS=ON`, `MLX_BUILD_OMARCHY=ON`, `MLX_BUILD_METAL=OFF`,
  `BUILD_SHARED_LIBS=OFF`. Static linking is required because the omarchy backend's symbols
  are not in the `MLX_API` export set; the same workaround the `m1-integrate-20260922` tree
  uses for these suites.

## Per-suite table (`<project-m1>`, 2026-09-29T21:08:09Z..21:09:47Z = 98 s wall)

| suite                              | cases | pass        | fail | sec |
|------------------------------------|------:|------------:|-----:|----:|
| omarchy_runtime_tests              | 41    | 22,694      | 0    | 40  |
| omarchy_primitive_tests            | 104   | 2,743,003   | 0    | 1   |
| omarchy_matmul_family_tests        | 22    | 82,940,463  | 0    | 42  |
| omarchy_fast_ops_tests             | 35    | 1,104,350   | 0    | 7   |
| omarchy_kv_ops_tests               | 16    | 781         | 0    | 0   |
| omarchy_indexing_ops_tests         | 53    | 4,093       | 0    | 0   |
| omarchy_reduce_ops_tests           | 34    | 7,152       | 0    | 0   |
| omarchy_shape_ops_tests            | 25    | 898         | 0    | 1   |
| omarchy_linalg_ops_tests           | 30    | 181,118     | 0    | 0   |
| omarchy_copy_offset_tests          | 26    | 629         | 0    | 0   |
| omarchy_distributed_tests          | 9     | 36          | 0    | 0   |
| omarchy_compiled_tape_tests        | 12    | 2,096       | 0    | 0   |
| omarchy_fft_ops_tests              | 19    | 1,379       | 0    | 2   |
| omarchy_fft_general_tests          | 14    | 4,245       | 0    | 3   |
| omarchy_eig_ops_tests              | 9     | 264         | 0    | 0   |
| omarchy_take_fill_tests            | 8     | 1,109       | 0    | 0   |
| omarchy_conv_tests                 | 13    | 3,450       | 0    | 0   |
| omarchy_complex_ops_tests          | 34    | 1,715       | 0    | 0   |
| omarchy_select_layout_tests        | 13    | 4,500       | 0    | 1   |
| omarchy_fast_regression_tests      | 2     | 16          | 0    | 0   |
| omarchy_scatter_determinism_tests  | 21    | 127         | 0    | 0   |
| omarchy_eq_math_tests              | 7     | 116         | 0    | 0   |
| omarchy_fused_chain_tests          | 36    | 346,272     | 0    | 0   |
| omarchy_error_contract_tests       | 3     | 14          | 0    | 0   |
| omarchy_ane_bundle_tests           | 39    | 7,055       | 0    | 1   |
| omarchy_capability_sim_tests*      | 30    | 12,304      | 0    | <1  |

\* `omarchy_capability_sim_tests` is a profile-matrix driver. Each profile returns RC=0:
m1-honeykrisp-fork (6 cases / 2462 assertions), m1-stock-no-coopmat (6/2460),
subgroup-size-64 (6/2460), small-shared-memory (6/2460), no-cooperative-matrix (6/2462).

Total assertions across the 26 suites: ~97.0 M.

## Acceptance per assignment
- All 26 suites accounted for on `<project-m1>`: ✓
- Per-suite table (suite, tests, pass/fail, seconds): ✓
- Commit hash: `db74f11adfb2c6fcee674d46635083b56a30f9c8` (verified above)
- Toolchain identity: glslc 2026.3, Vulkan/Asahi ICD, MLX_BUILD_OMARCHY=ON,
  MLX_BUILD_TESTS=ON, BUILD_SHARED_LIBS=OFF
- Receipt path: this file
- Hosts completed verbatim: `<project-m1>` only.

## Correction note (vs. an earlier public draft)
An earlier draft of this receipt documented a build from `024d4fe60` (a checkout on the
device that was not this repository's release tree). That commit predates the elementwise-add fix
`3ab075ae1d37ef5c0c51c88ba192686003e46d1d` and was not source-frozen. The earlier draft was
removed; the build was rerun at the verified `db74f11a` (above) and this receipt reflects that
rerun. The earlier raw logs remain under a private notebook entry with `SHA256SUMS` for audit.

## What did NOT run and why
- `<project-m1-max>`: held by another job for the entire window. Did not preempt.
