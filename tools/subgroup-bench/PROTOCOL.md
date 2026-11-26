# Subgroup-vs-tree microbenchmark protocol for m1-test-host (BenchQueueM1)

## Why this exists

The repo's `qmm_vec.comp` and `matmul_vec.comp` source comments claim
"Honeykrisp hardware scan/reduce covers integer ops only, and its float
subgroupAdd lowers to a software shuffle chain" (the comments are at
`overlay/mlx/backend/omarchy/shaders/matmul_vec.comp:18-20` and
`overlay/mlx/backend/omarchy/shaders/qmm_vec.comp:17-18`). The same
claim is restated in `receipts/2026-09-02-gemv-decode/README.md` and
`receipts/2026-09-04-subgroup-finding.md`. No receipt established it
with measurement: every instance is an inference from "the comments
say so." If the inference is wrong, the production paths lose the five
barrier rounds of the shared-memory tree for no reason; if it is right,
the kernel is dead on arrival and no amount of equivalence-testing will
get it shipped.

The bench in `tools/subgroup-bench/` answers the question directly: two
byte-identical compute kernels that differ ONLY in the reduction body
(one does `subgroupAdd(float)` over 32 lanes, the other does the
five-round shared-memory tree). Both run on the same buffers, on the
same device, in the same process. GPU-timestamped. Equivalence check
against a CPU f32 reference for both. The leg's `ratio_gpu` (subgroup
ns / tree ns) is the answer. < 1.0 means subgroup wins; > 1.0 means
software lowering is real and the tree is the right path.

## Provenance line beside every number

The bench prints the device name, API version, subgroupSize, ARITHMETIC
support bit, and timestampPeriod at startup, every run. The receipt
cites this line for any quoted number.

## Build

Run from the worktree root (`~/src/mlx-omarchy-subgroup-qmm` on m1-test-host):

```sh
bash scripts/prepare-mlx.sh    # only if .work/mlx missing
g++ -std=c++17 -O2 -o /tmp/subgroup-bench tools/subgroup-bench/bench.cpp
```

Shaders are compiled by the program at startup via `glslc` if present
(the M1 has it; the repo hard rule says compile where glslc exists),
else `glslangValidator`. If neither is on PATH, the program exits with
a name error.

## Run

```sh
/tmp/subgroup-bench                 # full legs (65536 groups)
/tmp/subgroup-bench --quick         # shorter legs (4096 groups, 3 repeats)
```

The bench outputs NDJSON on stdout, one line per result. Lines:

- `{"k":"dev",...}` — device info (one line)
- `{"k":"eq",...}` — equivalence check, 256 groups × 32 lanes vs CPU f32 reference
- `{"k":"subgroup_skip",...}` — only when the device lacks ARITHMETIC or subgroupSize != 32
- `{"k":"leg",...}` — the timed leg: sub_host_ns, sub_gpu_ns, tree_host_ns, tree_gpu_ns, ratio_gpu

## Keep rule for a deciding verdict

The assignment says "commit provisional until the M1 leg returns;
revert with numbers if it does not pay." The keep rule:

- **Equivalence must be 0 mismatches on the M1.** Any mismatch means
  the subgroup variant is broken on Honeykrisp and the kernel is
  killed. Ratio irrelevant.
- **ratio_gpu < 1.00** (subgroup faster than tree on GPU ns): keep
  variant. Convert qmm_vec.comp to subgroup, guarded by
  subgroupSize == 32 + ARITHMETIC bit.
- **ratio_gpu >= 1.00** (subgroup same or slower than tree on GPU ns):
  drop variant. The five-round barrier tree is the right path; the
  "software lowering" claim is now measured, not inferred. The receipt
  cites the measured number and the source-comment claim gets updated
  to cite the measurement.

## Why the equivalence check is strict

The CPU reference computes `sum(input[i*32..i*32+31])` in f32 with
straight-line accumulation. The subgroup variant uses
`subgroupAdd(accumulator)`, where each lane accumulated the same way.
Both should match the CPU reference within 1-2 ulps. On llvmpipe
(software Vulkan, Mesa), the bench records the expected failure: with
subgroupSize=8 but the kernel hardcoded to LANES=32, subgroupAdd only
sums across the 8-lane hardware subgroup and leaves 24 lanes unsummed
(`max_diff` up to ~8 absolute on [-1,1] random inputs). The bench does
NOT exit nonzero on equivalence failure — it records the failure and
skips the timed leg. On the M1 (subgroupSize=32) the equivalence must
hold or the receipt names that as the answer.

## Shaders used

- `tools/subgroup-bench/shaders/reduce_subgroup.comp` — `subgroupAdd(float)` over 32 lanes.
- `tools/subgroup-bench/shaders/reduce_tree.comp` — five-round shared-memory tree, identical input/output.

Both compiled with `glslc -fshader-stage=compute` per the repo hard
rule that the M1 leg uses glslc.

## What is NOT measured here

- The k-loop body: the bench reduces only the post-loop 32-element sum
  step, which is the exact step qmm_vec.comp replaces. The full matmul
  (with the k-loop read) is what the live profiling run in the
  follow-on A/B bench measures.
- bf16 / f16 paths: the bench uses f32 throughout. qmm_vec.comp
  promotes to f32 inside the dot product before reduction, so the f32
  reduction cost is the right number to measure. The bf16/f16
  equivalence story lives in the existing test_matmul_family.cpp cases.
- SubgroupSize other than 32: the bench reports and skips. A device
  with subgroupSize != 32 is not the M1 and not the support target.

## Receipt

The receipt (`receipts/2026-09-04-subgroup-vs-tree-microbench.md`) is
written by SubgroupQmm after the M1 leg returns and carries the
provenance line, the `eq` line, the `leg` line, the device line, and
the verdict (keep / drop with measured numbers).
