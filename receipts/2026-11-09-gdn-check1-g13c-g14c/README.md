# GDN decode check 1 (per-op fp64 error) for the untiled and wide defaults on G13C and G14C

Closes the receipt gap named in issue 369: the untiled prefetch kernel (default on G13C at head
ratios 1, 2 and 3 and on G14C at ratios 1 and 3) and the wide kernel (default at equal head counts
on both parts) shipped without the numerics gate's check 1, and the two prior receipts
(`2026-10-21-gdn-bf16a-g14c-ratio1`, `2026-10-22-gdn-wide-g13c`) each say so.

## What ran

The check-1 TEST_CASE added in the merge of PR 384 (`b2e22ded40f5d56258d60d869537ee6b14f05d43`):
`omarchy_gdn_gqa_bf16a_tests --test-case='*per-op fp64*'`. The test computes an independent
double-precision reference (gate `exp(-exp(A_log) * logaddexp(0, a + dt_bias))`, beta `sigmoid(b)`,
recurrence `S = S*g + ((v - S.k)*beta) outer k`, `y = S.q`) from the bf16-rounded operands, runs the
tiled, untiled-plain and (at Hv = Hk) wide kernels on the same pinned operands with a nonzero fp32
initial state, and requires each candidate kernel's max-abs output and state error to be no worse
than the tiled kernel's on every pair (16 key heads; 16, 32 and 48 value heads).

## Results

| Part | Device | Driver | Pair | tiled err | untiled err | wide err |
|---|---|---|---|---|---|---|
| G13C | Apple M1 Max (G13C C0) | Mesa 26.3.0-devel (git-69bde15340), mlx 0.32.4 | 16/16 | 0.0832068 | 0.0832068 | 0.0832068 |
| G13C | Apple M1 Max (G13C C0) | same | 16/32 | 0.064196 | 0.064196 | n/a (not a default pair) |
| G13C | Apple M1 Max (G13C C0) | same | 16/48 | 0.0502079 | 0.0502079 | n/a |
| G14C | Apple M2 Max (G14C B1) | Mesa 26.3.0-devel (git-e7631595df), mlx 0.32.4 | 16/16 | 0.0832068 | 0.0832068 | 0.0832068 |
| G14C | Apple M2 Max (G14C B1) | same | 16/32 | 0.064196 | 0.064196 | n/a |
| G14C | Apple M2 Max (G14C B1) | same | 16/48 | 0.0502079 | 0.0502079 | n/a |

doctest Status: SUCCESS on both parts, exit 0. The three kernels agree exactly with each other on
these operands (identical errors against the fp64 reference), so the gate bar (candidate no worse
than the deployed tiled kernel) holds with margin zero. The identical per-pair errors on the two
parts match the chip-comparison result of `2026-10-22-gdn-wide-g13c`: these kernels give the same
numbers on both chips. Kernel ids in the output confirm the right kernel per arm (455 tiled, 480
untiled plain, 638 wide).

## Procedure

One run per part, a single fresh process per invocation, `scripts/run-mac369.sh` (checked out at
origin/main `b2e22ded`, prepared tree, Ninja Release build of `omarchy_gdn_gqa_bf16a_tests`, then
the filtered run). Kernel 7.1.12-2-20261103.2-lab-ARCH on G13C with the performance governor,
7.1.12-2-7-ARCH on G14C with the schedutil governor. No timing or thermal claims in this receipt;
these are correctness runs (each finished in seconds of GPU work).

## Files

`SHA256SUMS` covers every file. `g13c-*` and `g14c-*` hold each part's check-1 output, run log,
cmake log and prepared-tree log; hostnames in the logs are replaced with G13C-HOST and G14C-HOST.
`scripts/` holds the runner script that produced them.
