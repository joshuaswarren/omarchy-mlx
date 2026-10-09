# GatedDeltaUpdate with Hv a multiple of Hk (Qwen3.5-9B: 16 key heads, 32 value heads) on the fused path

Claim: an `mx.fast.gated_delta_update` call that passes un-repeated q and k (Hk != Hv) now runs the fused kernels instead of MLX's composed per-token fallback, for prefill and for decode. The result equals the call that repeats q and k first, bit for bit.

Before: `GatedDeltaUpdate::use_fallback` returned true for any `Hk != Hv`, so a caller that does not repeat q and k (stock mlx-lm; the vendored mlx-lm patch series repeats them in Python) ran the per-token fallback.

## Evidence (M1 Max, G13C, Linux; one run each)

`omarchy_gdn_fast_route_repeat_tests`, five shapes: decode (T=1), prefill T=32, prefill T=96 (chunked route), prefill T=32 and decode with q/k as slices of a longer array (nonzero buffer offset). Q/K have 16 heads, V has 32, head dim 128.

| run | binary (sha256 prefix) | source | result |
|---|---|---|---|
| red (`m1max-red.log`) | `d98c4d8341938e8d` | tests only, `91004b98b` | FAILED: 11 of 3,863 assertions. The un-repeated route took 1,634 dispatches at T=96 and 546 at T=32 against 1 for the repeated route, and its output and state differ from the repeated route. |
| green (`m1max-green.log`) | `95a3dd8e8c211702` | fix, `d858fccbc` | 2 test cases, 3,863 of 3,863 assertions passed. The un-repeated route takes 3 dispatches at every shape (two expansion copies plus the kernel) against 1 for the repeated route; outputs and final states are bit-identical. |

| red (`m1max-red.log`) | `d98c4d8341938e8d` | tests only, `91004b98b` | FAILED: 11 of 3,863 assertions. In the visible part of the log the un-repeated route took 1,634 dispatches at T=96 and 546 at T=32 with offset views, against 1 for the repeated route, and its output and state differ from the repeated route (the log keeps only its last 40 lines, so the plain T=32 and decode lines are not in it). |

## Change

- `use_fallback` accepts `Hv % Hk == 0`.
- `eval_gpu` expands q and k from Hk to Hv heads on the device with one general copy each (value head j reads key head j / rep, the order of `mx.repeat` on the head axis), after the existing strided-input materialization and before the decode, scan, coopmat and recur32 kernels. The copy adds each array's own buffer offset, so the expansion passes `i_offset = 0`.
- The two old cases in `test_gdn_fast_route_repeat.cpp` pinned the old behavior (un-repeated route must cost more than 4x the dispatches; fused vs composed agreement above 95 percent) and are replaced by one case on the new contract.

## Not covered

- Speed on the model: the stock-mlx-lm one-hunk rerun (H53) was queued and not run when this receipt was written.
- The raw-gates decode op (`gated_delta_update_raw`) shares the expansion block but has no test in this PR.
- Per-channel decay (`g` with 4 dimensions) with `Hk != Hv`, masked prefill with `Hk != Hv`, and `B > 1` decode with `Hk != Hv` are not tested.
- No M1 or M2 Max run. Kernel and Mesa versions, dispatch trace beyond the counts above and thermal procedure are not recorded.
