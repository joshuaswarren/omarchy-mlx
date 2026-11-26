# Causal flash prefill: launch the heaviest q tiles first

Change: in `sdpa_prefill_flash_causal_coopmat.comp` the query-row base is computed from `gl_NumWorkGroups.x - 1 - gl_WorkGroupID.x` instead of `gl_WorkGroupID.x`. The tiles are the same, each tile does the same arithmetic over the same keys in the same order, only the launch order of the workgroups changes. Under a causal mask the work of a tile grows with its row index, so the old order started the heaviest workgroups last and left cores idle at the end of the grid.

This stacks on the causal flash route (the causal flash PR, default off) and changes nothing for a user who leaves `MLX_OMARCHY_SDPA_CAUSAL_FLASH` unset.

## Pre-registered rules (in the lane notebook before any run)
1. Bit identity: the flash output sha256 of the old and the new build are equal at L 512, 1000, 1024 and 2048 on the same chip.
2. The device test (tolerance gate, 3-run identity) passes on the new build.
3. Speed, 4B attention shape (batch 1, 32 query heads, 8 KV heads, head_dim 128, bf16, causal), 3 repeats per arm, arms alternated, idle gap before each run (30 s, 60 s on the M2 Max): kept if the new build is at least 3 percent faster than the old at L 512 on the M1 Max and not slower than the old by more than 3 percent at any length on any chip. A cell is void if an arm's spread over the repeats exceeds 5 percent.

## Results (raw logs in `m1/`, `m1max/`, `m2max/`; host names replaced by chip labels)
Same two wheels on every chip: old = build of the causal flash route, new = that build plus this change. Private Honeykrisp build, kernel 7.1.12-2-12.6 prerelease.

| chip | bit identity (4 lengths) | device test | ms at L 512 / 1024 / 2048: composed | old | new | new / old | new / composed |
|---|---|---|---|---|---|---|---|
| M1 | equal | 100 of 100 | 6.235 / 21.642 / 82.214 | 3.790 / 13.597 / 51.300 | 3.758 / 13.501 / 51.521 | 0.992 / 0.993 / 1.004 | 0.603 / 0.624 / 0.627 |
| M1 Max | equal | 100 of 100 | 1.967 / 6.161 / 22.993 | 1.801 / 4.718 / 15.676 | 1.719 / 4.625 / 15.515 | 0.952 / 0.980 / 0.990 | 0.874 / 0.751 / 0.675 |
| M2 Max | equal | 100 of 100 | 1.691 / 5.150 / 18.109 | 1.226 / 3.546 / 12.088 | 1.201 / 3.442 / 11.713 | 0.979 / 0.971 / 0.969 | 0.710 / 0.668 / 0.647 |

All three rules are met. The gain is 4.8 percent at L 512 on the M1 Max (bar 3 percent), 2.1 to 3.1 percent on the M2 Max, and within noise on the M1 (worst cell 0.4 percent slower at L 2048). Spreads were 0.0 to 2.8 percent, no cell void. The M1 Max ratio at L 512 is still above the 0.8 minimum bar (0.874): this change closes part of the gap, not all of it.

## Evidence fields (what this receipt records, what it does not)
Recorded:
- Source: shader change in commit 2e1c712eb on the causal flash branch (the one-line diff above); the measured wheels are builds of the branch commits `3c767b30` (old) and `2e1c712e` (new), which differ by that change.
- Exact commands: `tools/h40_ticket.sh` is the ticket that produced every log (identity lines, flash output hashes per length via `tools/sdpa_hash.py`, the device test, then the micro `tools/sdpa_micro.py causal 512,1024,2048` three times per arm with the arms alternated, idle gap before each run). Paths and the private driver directory are replaced by `$WORK` and `$PRIVATE_ICD_DIR`.
- Kernel `7.1.12-2-12.6-sep-ARCH` on all three chips (printed in each log); Vulkan driver `Mesa 26.3.0-devel (git-6543eeb7df)` and its library sha256 prefix `3546bcafe8ed3995` (in the logs); device names `Apple M1 (G13G B1)`, `Apple M1 Max (G13C C0)`, `Apple M2 Max (G14C B1)` (in the logs and in the flash hash line).
- Numerical result: the flash output sha256 per length, old against new, per chip (`m1`, `m1max`, `m2max` logs, line `hash venv-cf` and `hash venv-cf2`), equal on all three chips; device test 100 of 100 assertions.
- Timing procedure: median of 5 reps x 20 calls per process, 3 repeats per arm, arms alternated, 30 s idle before each run (60 s on the M2 Max), spread per cell printed; cells over 5 percent spread are void (none were).
Not recorded, and not claimed:
- Backend dispatch trace for these runs. The device test of the causal flash route asserts one flash dispatch per eligible cell and fails when the route is disabled (shown in the causal flash receipt), so the route is exercised, but this ticket did not print a trace.
- Firmware identity of the boot firmware (not captured by the ticket), thermal state beyond the idle gaps (no temperature or clock log), and a Vulkan device-reopen result (no reset or hang path was exercised; no reopen test was run).
- Model and quantization hashes: not applicable, this is a kernel micro on synthetic bf16 inputs from a fixed seed.
The performance claim is therefore limited to the numbers in the table on those builds; reproducing it needs the two wheels, which are builds of the branch commits named above.

## Identity
Wheels `0.32.4.dev202610081925+3c767b30` (old) and `0.32.4.dev202610082132+2e1c712e` (new), both from the branch this change was taken from; the shader differs by the two lines above. Driver library sha256 starting 3546bcafe8ed3995 on all three chips. The M1 Max leg was read on 2026-10-08, the M1 and M2 Max legs on 2026-10-09. Not measured here: end-to-end model throughput (the causal flash route's own end-to-end result is in its receipt).
