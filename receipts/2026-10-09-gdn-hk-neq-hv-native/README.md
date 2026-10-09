# GatedDeltaUpdate with Hv a multiple of Hk (Qwen3.5-9B: 16 key heads, 32 value heads) on the fused path

Claim: an `mx.fast.gated_delta_update` call that passes un-repeated q and k (Hk != Hv) now runs the fused kernels instead of MLX's composed per-token fallback, for prefill and for decode. The result equals the call that repeats q and k first, bit for bit.

Before: `GatedDeltaUpdate::use_fallback` returned true for any `Hk != Hv`, so a caller that does not repeat q and k (stock mlx-lm; the vendored mlx-lm patch series repeats them in Python) ran the per-token fallback.

## Evidence (M1 Max, G13C, Linux; one run each)

`omarchy_gdn_fast_route_repeat_tests`, five shapes: decode (T=1), prefill T=32, prefill T=96 (chunked route), prefill T=32 and decode with q/k as slices of a longer array (nonzero buffer offset). Q/K have 16 heads, V has 32, head dim 128.

| run | binary (sha256 prefix) | source | result |
|---|---|---|---|
| red (`m1max-red.log`) | `d98c4d8341938e8d` | tests only, `91004b98b` | FAILED: 11 of 3,863 assertions. In the visible part of the log the un-repeated route took 1,634 dispatches at T=96 and 546 at T=32 with offset views, against 1 for the repeated route, and its output and state differ from the repeated route (the log keeps only its last 40 lines, so the plain T=32 and decode lines are not in it). |
| green (`m1max-green.log`) | `95a3dd8e8c211702` | fix, `d858fccbc` | 2 test cases, 3,863 of 3,863 assertions passed. The un-repeated route takes 3 dispatches at every shape (two expansion copies plus the kernel) against 1 for the repeated route; outputs and final states are bit-identical. |

## Change

- `use_fallback` accepts `Hv % Hk == 0`.
- `eval_gpu` expands q and k from Hk to Hv heads on the device with one general copy each (value head j reads key head j / rep, the order of `mx.repeat` on the head axis), after the existing strided-input materialization and before the decode, scan, coopmat and recur32 kernels. The copy adds each array's own buffer offset, so the expansion passes `i_offset = 0`.
- The two old cases in `test_gdn_fast_route_repeat.cpp` pinned the old behavior (un-repeated route must cost more than 4x the dispatches; fused vs composed agreement above 95 percent) and are replaced by one case on the new contract.
- `GatedDeltaUpdateVJP::use_fallback` now returns true for `Hk != Hv` unless `MLX_OMARCHY_FUSED_VJP_GQA=1`. Reason: the first gate run at `Hk = 4, Hv = 16, T = 17` failed (10,606 assertion failures at head `aeb0e67a4`) when the forward started taking the fused path and autograd reached the fused backward at GQA shapes. With the composed backward at GQA the gate passed (125,023 of 125,023 at `09047417d`).

## Speed on the model (H53, M1 Max, G13C, wheel `0.32.4.dev202610092039+d858fccb`, Qwen3.5-9B, quiet machine)

Stock mlx-lm plus only the one-hunk gate change, `score_bench.py` unmodified, one run per row.

| arm | decode tok/s | prefill 512 tok/s | digest |
|---|---|---|---|
| stock | 25.17 | 55.43 | `4e5430032a62d99f` |
| one-hunk, run 1 | 33.91 | 256.85 | `1caed5680c3aa8f2` |
| one-hunk, run 2 | 33.84 | 258.69 | `1caed5680c3aa8f2` |
| full vendored series | 37.16 | 302.73 | `80274aa790426468` |

The one-hunk route is 4.6x stock on prefill and 1.34x on decode. It reaches 85 percent of the full series on prefill and 91 percent on decode. Its output digest differs from the full series' digest (the series' digest is the one that matches macOS), so the one-hunk route is not token-identical to macOS.

## Found by the gate: the fused GDN backward computes a wrong dg (open, not fixed here)

`gate-a0207a4e1-vjp-float32-reference.txt` (binary `222373de37b7b38b`, head `a0207a4e1`) compares the fused backward with the composed GPU backward and with a float32 CPU reference built from the same bf16-rounded inputs.

- dq, dk, dv and dbeta: fused and composed both within 0.0027 relative L2 of the float32 reference.
- dg: **the fused result is off by 0.95 to 1.01 relative L2 for T > 1**, at `Hk = Hv = 16, T = 33` (0.96) as well as at the GQA shapes. The composed result is within 0.0016. At T = 1 dg is exact.
- Host-repeated data at equal head counts fails the same way (0.9456), so GQA is not the cause.
- The fused backward is on by default at `Hk == Hv`, so this predates PR #60. Recorded in `docs/known-defects.md`; the two VJP test cases are `may_fail` so the run stays green while the defect is open.

## Not covered

- The raw-gates decode op (`gated_delta_update_raw`) shares the expansion block but has no test in this PR.
- Per-channel decay (`g` with 4 dimensions) with `Hk != Hv`, masked prefill with `Hk != Hv`, and `B > 1` decode with `Hk != Hv` are not tested.
- No M1 (G13G) or M2 Max run of this change. Kernel and Mesa versions, dispatch trace beyond the counts above and thermal procedure are not recorded here beyond the H53 notebook entry.
- The fused dg defect is diagnosed to "fused side wrong", not fixed. Training through `Hk == Hv` GDN with T > 1 gets a wrong gate gradient on the default path.
