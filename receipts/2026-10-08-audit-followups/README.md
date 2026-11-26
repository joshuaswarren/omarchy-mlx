# 2026-10-08 audit follow-ups: Scan guard, ArgReduce NaN, empty Sort, matrix row

Main's three audit items, all landed on `main` (jwm1 G13G, coreglass ICD;
binary/test receipts below; no hostnames/IPs/serials).

## (1) Compatibility-matrix row 338 — MaskedScatter: partial → native
- Row now reads native with dtypes `u32, i32, f32, f16*, bf16*, bool` and
  links the bool test (`overlay/tests/omarchy/test_indexing_ops.cpp:1098`,
  33 NumPy-reference assertions). Remaining named refusals unchanged
  (`MaskedScatter dtype`, `MaskedScatter row count`,
  `MaskedScatter mask alignment` — verified still present at
  primitives.cpp:7168/7138/7198).
- Commit: the audit stack carried it (in the Scan-guard commit's tree;
  see f3dc90c94 lineage + a3ccf3a90 stack).

## (2) Scan::eval_gpu allocation guard — CONFIRMED REAL, fixed
- Suspect was `out.set_data(allocate_omarchy(...))` unconditionally at both
  Scan::eval_gpu sites (primitives.cpp:9604, 9651) — the LongSdpaCoop3 class
  fixed for dispatch_matmul in 84d7598e.
- Failing-before: test commit `7c0317160` ("CumSum with a pre-allocated
  shared-buffer output": parent-aliased output via array::unsafe_weak_copy +
  direct UnaryPrimitive::eval_gpu) FAILED on the pre-fix tree — parent read
  0 where the inclusive scan should read 15 (5/5 positions wrong: the scan
  landed in detached scratch). Run: omarchy_primitive_tests
  -tc="CumSum scans suffix rows against host references" on 183f4d096/24f3b2c85.
- Fix commit `a3ccf3a90`: `data_shared_ptr() == nullptr` guard at both sites
  (same pattern as dispatch_matmul:611/dispatch_softmax:2192).
- Passing-after: same case 5038/5038 assertions; [scan] tag 104/104 cases,
  2743008 assertions.
- The M1 Max dev host scan-zeros verdict remains INFERENCE (labelled in the receipt);
  the guard defect is now proven independently of it.

## (3) arg_reduce failures — NOT order dependence; two real defects, both fixed
- Correction of the earlier report: `test arg reduce NaN` fails
  DETERMINISTICALLY IN ISOLATION (my "each passes individually" check had
  run the other three arg_reduce cases only). Root cause:
  argreduce_suffix.comp raw `<`/`>` compares never let NaN win, so NaN rows
  resolved to the first non-NaN extremum ([3,NaN,1] → index 2; 1024-wide
  NaN@17 → 0). Identical on pure main ec2cd9dc and the branch tree.
- Fix `b42203b66`: NaN-wins compare (first NaN index wins; float dtypes
  only; integer compares and tie rules preserved) + omarchy regression case
  "float argreduces propagate NaN like the reference" (14 assertions).
  Upstream `test arg reduce NaN`: 6/6 assertions PASS after.
- Two omarchy tests had pinned the OLD NaN-skipping behavior
  (test_primitives.cpp argmax/argmin cases); updated to the mlx semantics
  in `152a23fdd` (expected {1,0,1,2,0}→{1,0,1,1,0}, {0,2,0,0,0}→{0,2,0,1,0}).
- Full-suite runner exposed a second defect: `test sort and scan on empty
  arrays` THREW "[omarchy] Sort dtype is not implemented (bool, shape=[0])".
  Fix `e7f56fc3d`: empty Sort/ArgSort short-circuit (size 0 → allocate-0 +
  return) before the dtype refusal.
- Defects recorded in `docs/known-defects.md` (OMARCHY-ARGREDUCE-NAN entry).

## Final verification (jwm1 G13G, e7f56fc3d+152a23fdd stack)
| suite | result |
|---|---|
| tests/tests (upstream full) | 264/264 cases, 3543/3543 assertions PASS (was 262/264) |
| omarchy_primitive_tests | 104/104 (2743008 assertions) |
| omarchy_reduce_ops_tests | 36/36 (incl. the new NaN case) |
| omarchy_indexing_ops_tests | 59/59 (35904 assertions) |
| omarchy_shape_ops_tests | 25/25 |

## Landing shas
- `7c0317160` tests: CumSum pre-allocated output (failing-before)
- `a3ccf3a90` omarchy: shared-buffer-view allocation guard in Scan eval_gpu
- `b42203b66` omarchy: ArgReduce float scans propagate NaN (+ regression case)
- `e7f56fc3d` omarchy: empty Sort/ArgSort short-circuit
- `152a23fdd` tests: argmin/argmax NaN rows follow mlx semantics
- All on `main`; branches upstream-scan-guard / upstream-argreduce-nan /
  upstream-sort-empty hold the per-fix lineage.

## 2026-10-09 addendum: upstream 1-bit affine (#3161) vs Bonsai Q1

Upstream merged 1-bit affine quantization (#3161, merge `e0408d473`): affine
semantics with bit 0 mapping to the group minimum and bit 1 to the group
maximum, `scale = max(w_max - w_min, eps)`, `bias = w_min`, per-group scale
and bias arrays of shape `[N, K/group_size]` with group sizes {32, 64, 128},
weights packed uint32 LSB-first (`[N, K/32]` words). Our Bonsai Q1 path
(`bonsai_qmv_q1`, `bonsai_dequant_q1` in `overlay/mlx/backend/omarchy`) uses
the same affine semantics and the same LSB-first bit order, but stores the
oMLX checkpoint's uint8 byte pack (`[N, K/8]`, 8 one-bit codes per byte,
byte e bit i carries `w[e*8+i]`), addressed per byte through a uint32 view;
the two packs are byte-identical when K is a multiple of 32. Both paths
round at the midpoint of the group range and clamp the scale at eps.
Interoperation needs only: identical per-group scales/biases (dtype and
group_size agreement) and K % 8 == 0 for the Bonsai byte addressing; a
word-aligned upstream weight dumps bit-identically into the Bonsai layout
(INFERENCE from code reading, not yet exercised on hardware). The modes stay
on separate paths by design; before the next pin bump, either unify the
packs behind one layout or keep them apart with named refusals, and the
existing Bonsai parity gates pin whichever choice lands.
