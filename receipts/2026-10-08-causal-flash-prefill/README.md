# Causal cooperative-matrix flash prefill (bf16, head_dim 128), opt-in

Date 2026-10-08. A new SDPA arm for causal bf16 prefill at head_dim 128, off by default. `MLX_OMARCHY_SDPA_CAUSAL_FLASH=1` forces it (for `q_len >= 64`); unset or `0` keeps the composed route (f32 scores, softmax, PV). This receipt carries the numerics gate and the kernel timing on all three chips; the end-to-end gate was run by the parity lane (see "End to end" below).

## Change
- `shaders/sdpa_prefill_flash_causal_coopmat.comp`: one workgroup of four subgroups per (batch, head, 32 query rows). Each subgroup owns 8 query rows over the whole key walk and uses no workgroup barrier. Q panels stay in registers as cooperative-matrix A operands; K tiles load column-major and V tiles row-major straight from the buffers. Per 32-key tile: S = alpha * Q K^T on the 8x8x8 matrix unit (f32 accumulators), the causal mask and the online softmax on a small per-subgroup shared tile (f32 max, exp, sum, correction), P carried as two bf16 values (hi and lo, so the weights are not rounded to bf16), O += P_hi V + P_lo V, and a final division by the running sum. Causal skip: key tiles beyond the last row's key are not walked; offsets `k_len > q_len` are supported.
- `primitives.cpp`: the route, the engagement conditions, the kill switch. `compute.h`, `compute.cpp`, `CMakeLists.txt`: one pipeline (appended id).
- `overlay/tests/omarchy/test_sdpa_prefill_flash.cpp`: a device test that runs the route against the composed route and a float64 reference on 20 cells (plain and outlier-channel data, 10 shapes: q_len 64 to 2048, k_len up to 2048, `q_len < k_len` offsets, GQA repeat factors 1, 2, 4 and 8), checks that the route engages with one dispatch where eligible, that three runs are bit-identical, and the tolerance gate below.
- `docs/compatibility.md`: the arm and its measured ratios.

## What changed on the way (stated because two early design points were wrong)
- First kernel: P rounded to a single bf16, on the assumption (taken from the spec, not checked) that the composed route also rounds P. It does not: `probs` and the PV operand are float32 in the composed bf16 path. With single-bf16 P the device test failed on G13C: 120 of 200 assertions passed, 80 failed, all of them the two error checks (flash max abs error up to about +45 % over the composed route's). Fixed by the hi/lo carry; the second run is `m1max-g13c/device-test-hilo-first-run.txt` (200 of 200 with the tolerance gate; it ran both row-tiling variants).
- A 16-rows-per-subgroup variant was built and measured against the 8-row one at L 768 and 1536 on the M1 Max: 5.789 and 20.219 ms against 2.743 and 9.330 ms (2.1x slower, spread 0.1 to 1.2 %). It was deleted (`m1max-g13c/rowp-selection-micro.log`). The shader still carries a `ROWP` macro fixed at 1 and the loops over it; removing them is a refactor left as debt.
- The strict test "flash no worse than composed in max abs and in relative L2 on every cell" was written first and fails by rounding-boundary flips: both routes sit on the bf16 output-rounding floor, so a one-ulp flip of a few elements moves max abs by about 2 %. The gate in the repository is therefore a tolerance (max abs <= 1.05x, relative L2 <= 1.03x of the composed route's, per cell), registered before the data of the final kernel was read. The strict reading holds in 5 of 20 cells per chip; the worst cell is 1.0221x (max abs) and 1.000058x (relative L2). The test prints the strict result per cell. The relative-L2 tolerance (3 %) is much looser than the data (worst 0.006 %): it catches the single-bf16-P class of defect (+45 %), not drift of a few percent or below; the max-abs tolerance (5 %, worst 2.2 %) is the tighter of the two. The tolerance was registered before the final kernel's data was read: private notebook entry `20261008T1255Z-coopmat-causal-flash-prefill-h35`, amendment 2 (committed 12:41 CDT, before the 17:45Z run it governs).

## Numerics (device test, final kernel, 100 of 100 assertions on each chip)
`m1-g13g/device-test.log`, `m1max-g13c/device-test.log`, `m2max-g14c/device-test.log`. These three logs are byte-identical (the test prints error figures only, no device name, driver or binary hash, so by themselves they prove no device; the identity of the M1 Max run is in `m1max-g13c/head-build.log`, a run of this change's head (wheel `0.32.4.dev202610090004+2073579c`, provenance `verified=match`): it prints device name, Honeykrisp driver string, driver library sha256 and test binary sha256 beside the doctest lines (100 of 100 assertions passed) and three repeats of the L 512/1024/2048 micro. The M1 run is in `m1-g13g/head-build.log` (same wheel, same test binary sha256, 100 of 100 assertions, three micro repeats: flash / composed 0.607, 0.629, 0.624 at L 512 / 1024 / 2048, the same as the gate-2 figures); the M2 Max identity is the one in the run ticket, no head-build log exists for it). The error figures against the float64 reference agree to every printed digit on the three chips.

## Speed (the dense-attention micro, 4B shape: 1 batch, 32 query heads, 8 KV heads, head_dim 128, bf16, causal)
Median of 5 repetitions of 20 calls per process; 3 processes per arm, arms alternated, idle gap before each (30 s on the M1 and M1 Max, 60 s on the M2 Max); spread across the 3 processes 0.0 to 1.7 %. Output error against an fp32 numpy reference (head 0) is identical for both routes at every length (max abs 6.08e-3 / 3.88e-3 / 3.90e-3).

| chip | L | composed ms | flash ms | flash / composed |
|---|---|---|---|---|
| M1 (G13G) | 512 | 6.233 | 3.782 | 0.607 |
| M1 (G13G) | 1024 | 21.640 | 13.590 | 0.628 |
| M1 (G13G) | 2048 | 82.224 | 51.282 | 0.624 |
| M1 Max (G13C) | 512 | 1.976 | 1.800 | 0.911 |
| M1 Max (G13C) | 1024 | 6.133 | 4.713 | 0.768 |
| M1 Max (G13C) | 2048 | 22.975 | 15.806 | 0.688 |
| M2 Max (G14C) | 512 | 1.675 | 1.223 | 0.730 |
| M2 Max (G14C) | 1024 | 5.150 | 3.553 | 0.690 |
| M2 Max (G14C) | 2048 | 18.087 | 12.097 | 0.669 |

Raw: `*/gate2-micro.log`; harness `tools/sdpa_micro.py`, runner `tools/cf_ticket.sh`, summary `tools/cf_select_an.py`.
The registered bars were a target of 0.5x and a minimum of 0.8x at every length. The target is missed on all three chips. The minimum is met on the M1 and the M2 Max at every length and **missed on the M1 Max at L 512 (0.911)**. A tile-order change that launches the heaviest causal tiles first (bit-identical output on the M1 Max; 0.874 at L 512 there) is measured and kept out of this change until its M1 and M2 Max legs are read. The reason for the M1 Max shortfall at short L is not known.

## End to end (not measured in this receipt)
The parity lane ran the gate on the M1 only (wheel `0.32.4.dev202610081925+3c767b30`, the code in this change, kernel 7.1.12-2-12.6-sep-ARCH, private Honeykrisp ICD): 4B pf512 stock logits flash against composed, 5 interleaved pairs with non-overlapping ranges, 174.29 against 169.05 tok/s (+3.10 %; bar +2.0 %); pf1024 +4.9 % (one pair); decode equal; token digests equal; 2B pf512 +0.26 % and 9B pf512 -0.17 % (bar: not worse than -0.3 %, 2 pairs each); 4B last-position logits flash against composed max abs difference 0.094, mean 0.032, argmax equal. Those figures come from that lane's report; their raw receipt is not in this directory. M1 Max and M2 Max end-to-end replicas are queued and not read.

## Identity
- Wheel `0.32.4.dev202610081925+3c767b30` (a build of the branch commit this change was cherry-picked from, before the rebase onto main; the shader and the device-test source are byte-identical to this change's, the wheel is not a build of this PR's head; the head build checks are `m1max-g13c/head-build.log` and `m1-g13g/head-build.log`). Test binary sha256 starting ae64c79afbd842a4 on all three chips. Loaded-library provenance (`scripts/mlx_provenance.py`, run after the measurements, not beside them): `verified=match`, version match, on the M1 and M1 Max; on the M2 Max the loaded check was not run (the chip was held for another job) and the on-disk `libmlx.so` and `mlx.core` hashes equal the other two (`provenance-wheel.txt`).
- Kernel 7.1.12-2-12.6-sep-ARCH (aurora 12.6 prerelease) on all three; Honeykrisp ICD from the omacom/mesa stack at commit `6543eeb7df` (driver library sha256 starting 3546bcafe8ed3995).
- Device reopen after the runs and GPU firmware identity were not recorded.

## Not claimed
- Any default-on behaviour: the arm is opt-in. A per-chip or per-length engagement floor is open (the M1 Max loses the bar at L 512; nothing below L 512 is measured yet).
- Chips other than these three, models other than the 4B attention shape in the micro, head_dim other than 128, f16 or f32 attention, decode, `q_len < 64`.
- Why the kernel stops at about 0.6x to 0.9x of the composed time: no phase profile has been read.
