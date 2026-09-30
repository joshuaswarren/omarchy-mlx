# 2026-09-30 — GDN prefill fast-route fix: Hk != Hv repeat

Host: M2 Max (T6021) Omarchy test host (LAN TBD), Linux 7.1.13-ARCH-polltx,
Vulkan Honeykrisp on the M2. Wheels (live venv at
`~/.local/share/mlx-omarchy/venv`, stamped by the install pipeline;
`mlx-lm` patched in place by scripts/apply-mlx-lm-patches.sh against
pinned mlx_lm 0.31.3):

| arm | mlx-omarchy wheel stamp | mlx-lm patch state |
|---|---|---|
| baseline | `0.32.3.dev202609282218+29cba8e` | fast-route applied (line 286), no Hk repeat |
| candidate | `0.32.3.dev202609282218+29cba8e` | fast-route + mlx-lm-gated-delta-fast-route-repeat.patch applied |

Both arms use the same wheel — the patch is purely in mlx_lm/models/gated_delta.py.
Both arms use the same model snapshot
`938d8919941c6e7efd3c7150eff7fe9d12afa631` and the same prompt synthesis.
Every GPU run under `flock /tmp/m2-gpu.lock` (no nested flock); both arms
back-to-back in the same session window.

## Defect: the fused coopmat prefill path is bypassed on Qwen3.5-9B

`mlx_lm/models/gated_delta.py:286` calls `mx.fast.gated_delta_update(q, k,
v, g, beta, state, mask)` with the q/k shapes the model carries. For
Qwen3.5-9B those shapes have `linear_num_key_heads=16,
linear_num_value_heads=32`, so `q.shape = [B, T, 16, 128]` and `v.shape =
[B, T, 32, 128]`. The omarchy backend's
`GatedDeltaUpdate::use_fallback(Hk, Dk, Hv, Dv, ...)` returns
`Dk!=128 || Dv!=128 || Hk!=Hv` (overlay/mlx/backend/omarchy/primitives.cpp:10622-10633),
so `Hk=16, Hv=32` makes the check return true and the dispatch falls
through to the composed Python fallback `gated_delta_ops` (mlx_lm/models/
gated_delta.py:236-256), which is a Python `for t in range(T):` loop of
individual Multiply / Sum / Subtract / Add dispatches. The same
`gated_delta_ops` does its own `mx.repeat(q, Hv//Hk, -2)` for the
fallback's own arithmetic (line 242), but the fast path on line 286
calls the backend BEFORE that repeat and so never sees it.

Qwen3.8-2B has `linear_num_key_heads=16, linear_num_value_heads=16`, so
`Hk==Hv` already — the repeat would be a no-op and the model takes the
fast path naturally. That is exactly the 1,400 tok/s vs 42 tok/s gap
noted in the assignment.

## Fix

`patches/mlx-lm-gated-delta-fast-route-repeat.patch`: a 4-line addition
inside `gated_delta_update`, immediately before the `mx.fast.gated_delta_update`
call, that mirrors the repeat the composed fallback already does:

```python
if Hv != Hk:
    q = mx.repeat(q, Hv // Hk, -2)
    k = mx.repeat(k, Hv // Hk, -2)
```

The patch self-guards on `Hv != Hk` (no-op when equal; bit-exact identity
on Qwen3.8-2B). It composes with the existing fast-route patch
(`mlx-lm-gated-delta-fast-route.patch`) and is applied immediately after
it by `scripts/apply-mlx-lm-patches.sh`.

## A/B on the same machine, same model, same prompt

Both arms back-to-back on the M2, MLX_DISABLE_COMPILE=1
(receives/2026-09-30-chat-model-bench parity), gpu-turn -m 5 serial:

| metric | baseline | candidate | delta |
|---|---|---|---|
| Qwen3.5-9B prefill T=421 tok/s | **46.8** | **316.9** | **+577%** |
| Qwen3.5-9B prefill T=421 wall | 10.94 s | 1.62 s | -85% |
| Qwen3.5-9B prefill T=2048 tok/s | (extrapolated 42) | **312.5** | +644% |
| Qwen3.5-9B prefill T=2048 wall | (extrapolated ~49 s) | 6.55 s | -87% |

Wall times use the synthesize-a-prompt helper from
scripts/bench/chat_model_bench.py and a single warmup (one short prefill
to settle caches) before the 5 measured turns; this run reports a single
measurement (the assignment's cell shape). Tokenizers: same as the
receipts/2026-09-30-chat-model-bench/{qwen3.5-9b.json,ministral3-8b.json}
baseline.

Decode (32-token greedy after the prefill) was unchanged within noise
(±1 tok/s) — confirmed separately: the decode path goes through
`gated_delta_update_raw` (T==1, raw gates) which already does the repeat
inside the C++ fallback; the patch does not touch that call site.

## Doctest gate (overlay/tests/omarchy/test_gdn_fast_route_repeat.cpp)

Drives the exact Qwen3.5-9B prefill shape (B=1, T=32, Hk=16, Hv=32,
Dk=128, Dv=128) on both routes and asserts:

  * fused route (q/k expanded to Hv): <= 16 dispatches (the coopmat
    path is one dispatch + dtype/state materialization; bounded small)
  * fallback route (q/k NOT expanded): > fused * 4 (the per-token loop
    produces many more dispatches)
  * numerical agreement: max abs diff between fused and fallback <=
    `max(2 * 1e-3 * max(out), 1e-2)` bf16 quanta — verified bit-exact at
    the output scale on the test shape.

Measured on the M2 (profiling wheel build):

```
[gdn_fast_route_repeat] fused (Hk=Hv=32): 1 dispatches;
                              fallback (Hk=16, Hv=32): 546 dispatches
test cases: 1 | 1 passed
```

## 7-battery slice (standing M2 slice from AGENTS.md)

| test | result |
|---|---|
| omarchy_fast_ops_tests | 35 cases / 1,104,350 assertions / PASS |
| omarchy_fast_regression_tests | 2 / 16 / PASS |
| omarchy_runtime_tests | 41 / 22,694 / PASS |
| omarchy_primitive_tests | 104 / 2,743,003 / PASS |
| omarchy_kv_ops_tests | 16 / 781 / PASS |
| omarchy_conv_tests | 13 / 3,450 / PASS |
| omarchy_error_contract_tests | 3 / 14 / PASS |

Total: 214 test cases / 3,874,308 assertions, all green.

## Side artifact (profile only, not a code change)

overlay/tests/omarchy/test_gdn_prefill_profile.cpp: profile harness that
asserts the Qwen3.5-9B GDN prefill dispatch count (proves the fused
coopmat path is taken; <= 16 dispatches per layer) and that no CPU
dispatches leak. Built against the diagnostics wheel
(MLX_OMARCHY_GPU_PROFILING=ON) and shipped as
omarchy_gdn_prefill_profile_tests so future regressions of the same
class trip on a single test name.

The diagnostics wheel captures a 145 MB NDJSON profile of one full
512-token prefill at artifacts/GdnPrefill/2026-09-30-qwen-prefill.jsonl
(sha256 2117e2c01b421fcc3e885dd949e94d474b7d4bee094ac1f861fb4b409e9a4a59);
on the unpatched model this profile shows ZERO `GatedDeltaUpdate` prim
dispatches and 188,576 Multiply / 75,264 Sum dispatches — exactly the
per-token Python loop signature. Per the rule that diagnostic NDJSON is
never copied into receipts, this is referenced by SHA-256 only; the
the notebook holds the raw file pointer at
`~/.local/share/apple-silicon-lab/artifacts/GdnPrefill/`.

## Followups (not in this delivery)

* The fast-route repeat fix is a one-line patch; the same repeat belongs
  inside `mx.fast.gated_delta_update` itself (so the C++ backend
  contract reads "Hv != Hk is allowed and the backend does the repeat
  for you"). That would remove the need for the mlx-lm patch entirely
  and harden against any other model with mismatched K/V head counts.
  The change is small (~10 lines in
  `gated_delta_update` in mlx/fast.cpp, with the same `repeat_factor`
  the composed fallback already does) and is the right long-term home;
  deferred because it touches a public API contract.

* A 9B prefill of 1.62 s is now the dominant term in TTFT for chat
  workloads (the previous bottleneck). With the fast path enabled the
  remaining dispatch counts (per the captured NDJSON) shift to
  QuantizedMatmul (4-bit linears) and the elementwise prologue of the
  GDN layers' gate chain; targeted followups are the raw-gates prefill
  kernel (eliminates 24 compute_g + 24 sigmoid dispatches per prefill)
  and qmm Q4_0 bf16 scales/biases lifting (already in the coopmat-direct
  lane at +6.2% prefill-512 on jw16). Both are out of scope for the
  fast-route fix.
