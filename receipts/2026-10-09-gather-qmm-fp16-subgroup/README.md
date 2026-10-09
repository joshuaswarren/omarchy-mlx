# fp16 decode gather_qmm on the subgroup kernel

Date 2026-10-09. At the routed-MoE decode shape (`m == 1`, affine, 4-bit, group size 64, transposed packed weights) `mx.gather_qmm` in float16 ran the scalar kernel, one thread per output element, 24 to 64 workgroups, latency-bound. The subgroup kernel for that layout (`GatherQmmSubF16`) was already built; only a dtype test in `dispatch_gather_qmm` kept float16 off it (bfloat16 was already routed). This change widens that test to float16.

## Change
- `primitives.cpp`: `use_sub` accepts `out.dtype() == float16` as well as `bfloat16`. Every other condition stays (transposed, 4 bits, group 64, m == 1, subgroup size 32, subgroup arithmetic, 16-bit storage access); float32 and other layouts keep the scalar kernel; `MLX_OMARCHY_GATHER_QMM_SUB=0` forces the scalar kernel.
- `test_matmul_family.cpp`: the existing case "gather qmm subgroup kernel matches scalar at decode shapes" gains a float16 axis at the same five shapes (k 128, 192, 2048; index_count 2, 8, 48; the 48 x 1408 case is z-chunked).
- `docs/compatibility.md`: the route, its layout class and the measured speed.

## What was wrong with the first reading
A report measured 63.7 ms per layer for bf16 `gather_qmm` against 4.29 ms for a device `take` plus one batched `quantized_matmul`, on the M1. That run used a wheel stamped 2026-10-03, before the subgroup kernel landed (2026-10-05, re-landed 2026-10-06): on a current wheel the bf16 decode route is already the subgroup kernel (4.37 ms on the same chip) and only float16 stayed on the scalar kernel (64 ms). The dispatch traces in this receipt show it: the old wheel's fp16 default runs `GatherQmmF16` (24 / 24 / 64 workgroups); bf16 default runs `GatherQmmSubBF16` (6144 / 6144 / 16384); the new wheel's fp16 default runs `GatherQmmSubF16`.

## Not bit-identical to the scalar kernel (and not claimed to be)
The subgroup kernel sums the K loop as a lane tree (256 lanes, subgroup add, then eight subgroup totals) where the scalar kernel runs one serial loop, so the float32 accumulation order differs and the outputs differ in the last bits. The repository's gate for this kernel is the error against an fp64 host reference, no worse than 1.5x the scalar kernel's. It holds with room: per case the two errors agree to 3 digits or better (float16 relative L2 3.10e-4 against 3.10e-4 at k 2048; 3.166e-4 against 3.165e-4 on the 67,584-output case), and on the full float16 decode layer the maximum error against a float32 reference is 6.23e-4 for both kernels. The output hashes of the two kernels differ (that is expected, not a defect).

## Results
Probe: one gate / up / down layer of the Qwen3-30B-A3B decode shape (hidden 2048, expert intermediate 768, 4-bit group 64, top-8 of 40 slots, one token), a full `mx.eval` after every layer, 60 iterations per arm after warm-up, median; spread = interquartile range over the median, a cell is void above 10 %; error against a float32 reference built from the dequantized routed experts. Old wheel `0.32.4.dev202610081236+50e40389`, new wheel `0.32.4.dev202610090028+7d31d68e`; `tools/gqmm_probe2.py`, runner `tools/h45_ticket.sh`.

| chip | arm | old wheel ms (IQR) | new wheel ms (IQR) |
|---|---|---|---|
| M1 (G13G) | fp16 `gather_qmm` default | 64.078 (1.7 %) | **4.129 (2.3 %)** |
| M1 (G13G) | fp16 forced scalar (`SUB=0`) | 64.121 (1.9 %) | 64.110 (4.3 %) |
| M1 (G13G) | fp16 device `take` + batched `quantized_matmul` | 4.013 (2.2 %) | 3.893 (7.4 %) |
| M1 (G13G) | bf16 `gather_qmm` default | 4.367 (0.4 %) | 4.368 (0.4 %) |
| M1 Max (G13C) | fp16 `gather_qmm` default | 7.594 (2.9 %) | **1.184 (1.4 %)** |
| M1 Max (G13C) | fp16 forced scalar | 7.606 (0.3 %) | 7.565 (2.8 %) |
| M1 Max (G13C) | fp16 `take` + batched `quantized_matmul` | 2.173 (**43.3 %, void**) | 2.423 (**33.3 %, void**) |
| M1 Max (G13C) | bf16 `gather_qmm` default | 1.267 (2.0 %) | 1.263 (4.1 %) |

The registered rules (private notebook entry `20261009T0030Z-gather-qmm-fp16-subgroup-route-h45`, committed before any code or data): fp16 default new / old <= 0.5 (M1 0.064, M1 Max 0.156), new default / take <= 1.5 (M1 1.06; **not readable on the M1 Max, the take arm is void**), fp16 error <= 1.5x the scalar error (1.00 on both chips), bf16 default within 5 % (M1 0.0 %, M1 Max 0.3 %), device tests pass (3377 of 3377 assertions on both chips). All readable rules are met.
Identity per chip is at the top of `*/h45-identity-tests-probe.log`: device name, driver (Honeykrisp, Mesa 26.3.0-devel git-6543eeb7df), driver library sha256 starting 3546bcafe8ed3995, kernel 7.1.12-2-12.6-sep-ARCH, family test binary sha256 starting 842a1fd3b9fcb73f (the same file on both chips), and the loaded-library check of both wheels (`verified=match`, version match). The loaded-library check ran in the same ticket as the measurements, before them.

## Not measured, not claimed
- The M2 Max (G14C): the chip was held for another job; the same ticket is queued for it.
- float32, 8-bit, group sizes 32 and 128, non-transposed layouts: unchanged on the scalar kernel; their subgroup variants have no device test here.
- End-to-end model speed: this is one layer of the decode shape, not a model run. A real MoE decode step also pays routing, the top-k and the rest of the layer.
- The M1 Max `take` arm is void (bimodal timing), so no ordering of `gather_qmm` and `take` is claimed on that chip.
- Device reopen after the runs and GPU firmware identity were not recorded.
