# 1007b: upstream mlx backports #4366 #4599 #4635 #4637 #4638 #4626

Land: 72a62ea67 on origin/main (one commit, five verbatim patches, all
wired into scripts/prepare-mlx.sh, patch-test-applied at the pin
9c3d35571 before staging).

## GPU reproducer matrix (jw16, M1 Max, Honeykrisp, wheels)

| PR | baseline wheel 3db5cc1 (pre-fix main) | fix wheel 7dd21cb (72a62ea67 tree) |
|---|---|---|
| #4366 sort/argsort transposed views | PASS on omarchy (upstream pinned the failure on CUDA/Metal; the shared collapse bug did not alter our sort result) | PASS |
| #4599 dynamic slice, non-contiguous start | PASS on omarchy GPU (the landed part is the CPU path that schedules our copies) | PASS |
| #4637 vmap odd-length irfftn | **FAIL: vmap (3,8,10) != expected (3,8,11) at n=11** | **PASS** |
| #4635 0-dim boolean mask scatter | NAMED REFUSAL (bool MaskedScatter has no omarchy Vulkan kernel by design — compatibility contract) | same refusal (kernel work out of scope) |
| #4626 fp16 clip_grad_norm (CPU/python) | norm=inf, clipped=[nan, nan] with fp16 grads [30000, 40000] | norm=50000.0, clipped=[30000.0, 40000.0] |
| #4638 CPU compile-cache race | not reproducible single-process; race fix landed (unique temp name + rename) | same |

## Notes

- #4366/#4599 pass both before and after on omarchy: the backports are
  upstream-parity hardening of shared code (collapse_contiguous_dims is
  consumed by our backend; the CPU slice path schedules our copies).
- #4637 is the failing-before/passing-after pair on omarchy GPU.
- Probe: tools/u1007b_repro.py; wheels mlx_omarchy-0.32.4.dev202610071211+7dd21cb
  (fix) and dev202610071228+3db5cc1 (baseline), both rc=0 builds.
- #4635 residual: bool MaskedScatter kernel for Vulkan is open work
  (new kernel, out of scope for a backport).

## Part B audit (2026-10-07 15:4xZ, dev box, no GPU): gather_qmm_sub coverage

gather-qmm-sub-coverage.md in this directory: layouts uncovered (bits8,
g32/g128, f16, m>1, non-transposed with file:line), minimal
spec-constant/gate change proposal in ship order, and the host-reference
test matrix. No new shader files; per-layout parity proof required
before each gate widening (the scalar kernel stays the fallback).

## State at the 2026-10-07 15:50Z GLM quiet-down

- 1007b backports landed 72a62ea67 (+892ea7e4 probe); battery 9 suites
  green at 72a62ea67 (lane line sent 12:46Z).
- #4635 bool MaskedScatter: kernel + tests committed upstream-1007b
  62aed7883-lineage (v2 per-element atomics + cast_int->uint32 +
  int32 exclusive scan); the last GPU run showed 4 failing sub-cases:
  2 are wrong test expectations (now corrected to the NumPy reference in
  the same commit lineage) and 2 are real trailing-true-position
  misses with hypothesis + byte-dump plan written into the test file.
- Resumes after 'GLM session done': one GPU debug cycle
  (offsets dump -> fix), battery rerun, known-defects update, land.
