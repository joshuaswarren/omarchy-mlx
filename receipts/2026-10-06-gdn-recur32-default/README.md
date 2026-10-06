# GDN prefill: macOS-shape recurrence (recur32 mode 2) becomes the default on G13 parts other than G13C

Date 2026-10-06. Decision: Joshua (standing rule: a change that only reorders arithmetic and keeps fp64-reference error equal or better lands without asking).
Hardware evidence: jwm1 (Apple M1, G13G B1, Honeykrisp e7631595df fork ICD), Aurora 7.1.12-2-11.36-sep-ARCH, native wheels from this branch, idle gates (load1 < 0.3, PSI 0), GPU lock per cell. Lab entries (notebook): jwm1-parity H318 (candidates), H320 (mode 2 / TTFT), H323 (this flip). Artifacts: artifacts/jwm1-parity/{h318,h320,h323} in the notebook.

## What changes
`fast::gated_delta_update` prefill (maskless, scalar-per-head g, B=1, Hk=Hv, Dk=Dv=128, bf16, T >= 2, subgroup size 32) now takes the one-dispatch per-token recurrence (one 32-lane subgroup per (hv, dv) row, 4 f32 state elements per lane, two `subgroupAdd` reductions per token; the Metal `gated_delta_step` order) by default on G13 parts other than G13C (`g13_legacy_part`: jwm1 M1 G13G is the measured part). Other chips keep the previous routes until measured there. `MLX_OMARCHY_NO_COOPMAT_GDN=1` also keeps the exact scan.
`MLX_OMARCHY_GDN_RECUR32` : `0` (also off/false/no) = previous routes exactly (exact 4-lane scan for T < 64, chunked coopmat / HOIST for T >= 64), `1` = T >= 64 only, `2` = force mode 2 on any chip.

## Numerics (independent fp64 per-token reference, L2-normalised q/k, zero and non-zero initial state)
- T = 512/519/1024: y error 1.95e-3 .. 3.68e-3, state 1.2e-7 .. 3.7e-7 (the previous default HOIST: y 1.95e-3 .. 3.68e-3, state 1.1e-7 .. 6.1e-7).
- T = 2,3,5,11,16,31,32,33,63: worst y 2.70e-3, state 2.0e-7 (the exact scan: y 2.70e-3, state 2.5e-7).
- Deterministic (10/10 identical hash probes at T=512), long-prompt greedy tokens identical to the old routes at n=1031 and n=2051, omarchy_gdn_maskless_correctness (5 cases, 632 assertions, includes the recur32 and mode-2 fp64 cases) and omarchy_gdn_fast_route_repeat green on G13G.

## Pinned digests (jwm1, qwen38 protocol, temp 0) changed by the summation order; the off switch restores the old pins
| cell | old pin (RECUR32=0) | new default |
|---|---|---|
| 2B d64 | eee1cf9635d6d4eb | 1acd076784c2ab76 |
| 2B d128 | 0756351401f5b3fb | cf000162ad39e381 |
| 2B d256 | 393a1cf303e9f362 | 0e2b6fbddfb45521 |
| 2B pf512 | 509c19201275dfcc | d0de4df2662a17e3 |
| 4B d64 | 42d27a8cbe93df49 | 42d27a8cbe93df49 (unchanged) |
| 9B d64 | 80274aa790426468 | f1133a78dacbee1e |
The 2B d64 tokens move toward the macOS tokens for the same 5 prompts (3/5 prompts identical to macOS vs 2/5 before; prompt 1 stops diverging at token 28). The old pins stay valid under `MLX_OMARCHY_GDN_RECUR32=0`.

## Performance (jwm1, mirrored cells)
- pf512 pure prefill 443.9 vs 427.2 tok/s (+3.9 %, n=1 each in H323, +4.0 % n=4 in H318); kernel at T=512 4.08 vs 6.87 ms/call (-41 %), T=1024 -45 %, T=64 0.55 vs 1.05 ms, T=128 1.14 vs 2.00 ms.
- TTFT (2B d64, 4+4 mirrored cells, H323): 0.1193 vs 0.1308 s mean of cell medians (-8.8 %; H320: -8.4 %); decode unchanged (45.7-45.8 tok/s).
- 9B TTFT 0.450 vs 0.478 s (-5.9 %, n=1); 4B 0.2749 vs 0.2764 s (noise).
- Versus jwm1's own macOS: pf512 tier B 443.9 / 457.6 = 0.970x (still a LOSS), tier A not re-measured; decode 0.926x unchanged (LOSS); TTFT 0.1193 / 0.125 s = 0.954x latency (PASS candidate, needs the same-window paired macOS cell before a parity claim).

## Battery on G13G (static test build of this branch, jwm1)
All suites green except the known ones: omarchy_fast_ops 26 failed of 1,307,001 (the two `sdpa vjp ... (known defects)` cases, same as v0.7.28 and earlier today), and omarchy_matmul_family 2 cases / 288 assertions in `gather qmm` (introduced by the routed-MoE subgroup gather_qmm commit ef2d4477e, since reverted on main in 1a6db44fc: one case throws "shapes of the weight and scales are incompatible (3,64,12) vs (3,64,1)" at test_matmul_family.cpp:2159, the other fails numerically at :146); verified IDENTICAL with MLX_OMARCHY_GDN_RECUR32=0 (3575 assertions, 288 failed in both) so it is not caused by this change. omarchy_ane_runtime_tests does not link in the static test configure (pre-existing).
