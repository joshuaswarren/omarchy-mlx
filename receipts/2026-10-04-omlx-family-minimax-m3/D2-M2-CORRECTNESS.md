# D2 correctness receipt — MiniMax M3 JIT patch kernels on the M2 (T6021 Honeykrisp)

Date: 2026-10-04/05 (M2 window). Ticket: gpu-turn -m 12, 402 s wall including
fresh private-venv copy + install + verify; battery itself 0.667 s.

## Golden re-run (2026-10-05, rebase onto main @ 9f4f4f747)

After KernelBattery's translator landed on main (60f80d2 golden base), this
lane rebased and re-ran everything on the combined translator:

- Wheel: `mlx_omarchy-0.32.4.dev202610050436+60f80d2` via the golden recipe
  (`golden-clone-tree.sh` + `golden_rebuild.sh`, 42 s incremental),
  installed `libmlx.so` sha256 `1d086f3263edbcc1d95a8fa0ea4ea273e8fccf4009a786c021980ee461bdd825`
  (harness-verified == wheel-embedded), on a reflink clone of
  `/var/tmp/golden-venv` (`MLX_OMARCHY_BASELINE_TAG=60f80d2` guard).
- D2 battery: **6/6 OK** (0.854 s) on Device(gpu, 0).
- `tests/custom_kernel_smoke.py` from main tip: **19/19 OK** (1.338 s) —
  the translator combination (their cast scanner + pointer/alias passes,
  my `_mlx_arg` param aliases + uint→int decl casts + INFINITY/NAN +
  fast::/thread strips) passes main's full smoke suite.

Landing gate met.

## Wheel provenance

- Wheel: `mlx_omarchy-0.32.4.dev202610050059-cp314-cp314-linux_aarch64.whl`
- Installed `libmlx.so` sha256 == wheel-embedded sha256 (harness-verified):
  `b2fba0205d1de17912bb3477cea73e841c47da5febb51bb0e9bf4ab15a5e4593`
- `mx.__file__` resolved under `/var/tmp/fmm3-venv` (private cp -a of
  `/var/tmp/shared-omarchy-venv`, shebangs rewritten by
  `python -m venv --upgrade` before any pip call).
- Base tree: OmarchyDistributed complete build `/var/tmp/od-distributed-wheel-20261004`
  (+ its two unstaged distributed deltas: `mlx/distributed/ring/ring.cpp`,
  `mlx/distributed/distributed_impl.h` copied verbatim so the shared ring
  symbols link; disclosed to Main).
- Lane delta vs origin/main @ 19952daef: `overlay/mlx/backend/omarchy/custom_kernel.cpp`
  (translator) only; branch `agent/fam-minimax-m3` @ 017876602.

## Result: 6/6 PASS (real GPU, Device(gpu, 0))

| Kernel (upstream site) | Check | Result |
|---|---|---|
| K1_SCALAR (msa.py `_MSA_CSR_K1_SCALAR` body) | full MSL body through translator, shape (1,16,8,32) + finite | ok, 0.2032 s |
| K2 combine (`_MSA_CSR_K2`) | LSE combine vs numpy, rtol 1e-4 | ok |
| TOPK_SELECT (`_MSA_TOPK_SELECT`) | block selection vs numpy replica, exact | ok |
| D1 int scalar | values + shift | ok |
| D1 single float scalar (`scale`) | smoke contract | ok |
| D1 three float scalars | multi-scalar | ok |

`Ran 6 tests in 0.667s  OK`

## Defects the first ticket caught (both translator-level, fixed same day)

1. Parameter macro swizzle clobbering: param named `x` → `#define x _b0.data`
   expanded inside `gl_GlobalInvocationID.x`. Fixed by `_mlx_arg<binding>`
   aliases (body token rewrite refuses `.`-preceded positions). Reproduced on
   dev-box glslang 11:12.0/16.6 before fixing — deterministic GLSL bug.
2. MSL implicit uint→int declarations (`int d = elem % D;`) rejected by GLSL;
   initializer now wrapped in the matching constructor cast (no-op when
   already correct type).
3. Lane test bug: synthetic CSR fixture wrote `row_ptr` out of bounds;
   rewritten to the real contract (cursor walk + cumsum assertion).

Steel-MMA K1 (`msa.py:344`) remains refused with the exact `simdgroup_matrix`
error by design; the coopmat reimplementation (D3) is the next deliverable.

## Environment honesty

- Numerics tolerances: K2 rtol 1e-4/atol 1e-5 vs float64-ordered numpy;
  TOPK exact; K1 finite+shape only (per-element parity for K1 vs a CPU
  reference is D4 scope with the model-level run).
- No timing claims: wall-clock lines are informational only.
- Cleanup: `/var/tmp/fmm3-wheel` (~5 GB) and `/var/tmp/fmm3-venv` deleted
  after this receipt; `/var/tmp/shared-omarchy-venv` never written by this
  lane (guard verified `+b8af62c` before each ticket).
