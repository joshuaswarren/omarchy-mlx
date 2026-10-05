# oMLX DFlash speculative decoding on omarchy (M2) — matrix row A7 (2026-10-05)

**Verdict up front:** DFlash **runs end-to-end** on the omarchy-mlx wheel on the
M2 (DFlashEngine serves real completions, acceptance 69.5% at block 16), but it
is **~5.9x slower than plain batched decode on this backend** (10.59 vs 62.95
tok/s mean over 8×128-token greedy requests) and **fails the pre-registered
greedy token-identity gate (6/8 prompts diverge from plain decode)**. Static
analysis: the pure_attention pair needs no Metal-only API — the run confirms
that (no translator/MSL errors anywhere). Follow-up: numerics parity of the
batched verify pass on Vulkan (KernelBattery/NumericsGate class), not a gate or
platform gap.

## Provenance

- oMLX pin `v0.7.0` = `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`
  (dev clone `~/src/omlx`, verified `git log -1`)
- dflash-mlx pin `71f7c2c` = `0.1.10+omlx.9` (required dep, pyproject.toml:133);
  wheel built on the dev box (`dflash_mlx-0.1.10+omlx.9-py3-none-any.whl`,
  pure Python), installed `--no-deps` into a **copy** of OmlxLinux's M2 venv
- omarchy wheel `mlx_omarchy-0.32.4.dev202610042317+b8af62c`; libmlx.so sha256
  identical source vs copy (`af7df8850b3c…5901787`)
- Target: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (2.21 GB, already in the
  M2 HF cache); drafter: `z-lab/Qwen3-4B-DFlash-b16` (1.08 GB, downloaded
  2026-10-05 after the announced lane note; snapshot `b74e3a329c4d…`)
- Host: jw14m2-linux (T6021), GPU via one `gpu-turn -m 10` correctness ticket;
  server `127.0.0.1:8931`, `curl --max-time 150`

## Static analysis — what DFlash needs from mlx, and its omarchy disposition

Read at the pins: `omlx/engine/dflash.py` (2582 L), `omlx/speculative/dflash_drafter.py`
(1027 L), dflash-mlx `runtime/loading.py`, `runtime/registry.py`, `runtime/bundle.py`,
`engine/target_ops.py`, `engine/target_qwen_gdn.py`, `kernels.py`, `metal_limits.py`,
`docs/experimental/dflash_mlx_integration.md`.

| Requirement (site) | Engaged for Qwen3-4B pair? | omarchy disposition |
|---|---|---|
| `mx.fast.metal_kernel` GDN kernels ×8 (dflash_mlx/kernels.py:113–829, each behind `if not mx.metal.is_available(): return`) | no — hybrid_gdn only; `family()` returns `pure_attention` (target_qwen_gdn.py:792–798) | n/a on this path |
| `mx.fast.metal_kernel` DFlash2 conv/select (`omlx/speculative/dflash_drafter.py:105,799`) | no — scheduler-side DFlash2/MTP drafter (engine_pool.py:3654, engine/vlm.py:2865), not DFlashEngine | n/a |
| Wired-limit ownership `mx.metal.is_available()` → `mx.set_wired_limit` (engine/dflash.py:523–548) | no — only GLM-5.3 targets (`_load_with_wired_limit`, dflash.py:545); gate closes on omarchy anyway; omarchy `set_wired_limit` is an honest no-op (bce97ef2a) | skipped honestly |
| `mx.device_info()['max_recommended_working_set_size']` (dflash.py:526) | only behind that same gate | key landed on main (bce97ef2a) |
| `apply_metal_limits()` wired+cache (dflash_mlx/metal_limits.py:62) | no — called from dflash-mlx CLI/server/benchmark only, not the `stream_dflash_generate` serve path; early-returns when Metal unavailable | n/a |
| `mx.distributed.all_sum` (target_qwen_gdn.py:387,660; `sharding_group is not None`) | no — shard-gated, single host | n/a |
| BatchGenerator ownership | bypassed by design (dflash.py:493) | no mlx-lm batching dependency on the speculative path |
| Drafter load — bundled `DFlashDraftModel` classes (loading.py:64–75), no `trust_remote_code` | yes | pure mlx nn; 5-layer qwen3-shape drafter, bf16, block 16 |
| Verify-linear kernels (`dflash_mlx/verify_linear.py`) | eligible for pure_attention? — no `metal_kernel`/`is_available` hits in the module | stock mx ops |
| Target ops resolution (`resolve_target_ops`, target_ops.py:129) | yes — `QwenGdnTargetOps.supports_model` accepts `model_type=qwen3` (`"qwen" in model_type` + text shape, target_qwen_gdn.py:756–762) | pure Python dispatch |
| Engine routing | `dflash_batched_supported` False for llm/batched (utils/model_loading.py:1231–1237) → else-branch creates `DFlashEngine` (engine_pool.py:3361–3392) | verified |

Conclusion: **zero Metal-only requirements on the Qwen3-4B pure_attention
DFlash path** — confirmed by the run (no translator/MSL errors; the only
start failure was the missing `dflash-mlx` package, below).

## M2 install notes (honest)

- venv copy `/tmp/omlxdflash-home/.venvs/omlx` (`cp -a` of
  `/tmp/omlx-home/.venvs/omlx`, OmlxLinux's tree untouched) +
  `python -m venv --upgrade` + `pip install --no-deps` of the wheel;
  `mx.__file__` and libmlx.so sha verified.
- **The cp-a shebang hazard bit exactly as the hardware rules warn**: the first
  ticket run failed with `No module named 'dflash_mlx'` because the `omlx`
  console script's shebang still pointed into OmlxLinux's original venv.
  Fixed by rewriting the copy's `bin/*` shebangs to the copy path; `omlx
  --version` and `import dflash_mlx` then verified through the console script.
- The first enqueue (`-m 20`) died before GPU use on the `--hf-cache <path>`
  CLI misuse (`--hf-cache` is a BooleanOptionalAction flag); resubmitted at
  `-m 10` per the queue-sizing directive. Effective ticket GPU time ≈ 4 min.

## M2 run (2026-10-05 01:47–01:49 Z, ticket on the live GPU)

Server: `omlx serve --model-dir $HF_HUB --host 127.0.0.1 --port 8931`
(discovered id `mlx-community--Qwen3-4B-Instruct-2507-4bit`). Phase OFF: no
model settings (batched engine). Phase ON: `model_settings.json` written via
omlx `ModelSettingsManager` (`dflash_enabled=true`, `dflash_draft_model=<local
snapshot>`); server restarted. Same 8 prompts × 128 max tokens, temperature 0,
non-stream + one stream probe per phase.

Engine engaged (server-on.log):

```
omlx.engine_pool    - INFO - DFlash enabled for mlx-community--Qwen3-4B-Instruct-2507-4bit,
                             draft=/home/…/models--z-lab--Qwen3-4B-DFlash-b16/snapshots/b74e3a…
omlx.engine.dflash  - INFO - DFlashEngine loaded: target=/home/…/Qwen3-4B-Instruct-2507-4bit…
[dflash] prefix cache enabled (max_entries=4, max_bytes=8589934592)
[dflash] end-of-request snapshot saved (154 tokens)   # ×9 requests
```

Throughput (server-reported `usage` fields, 8 non-stream requests each):

| Phase | mean gen tok/s | mean total s/request | ttft |
|---|---|---|---|
| OFF (batched) | **62.95** | 2.24 | 0.20 s |
| ON (DFlash)   | **10.59** | 12.76 | 0.31–0.79 s |

DFlash summary line (streaming request; the only path that logs acceptance —
non-stream responses carry no acceptance field, a small omlx surfacing gap):

```
DFlash generation complete: 128 tokens, 19.2 tok/s, acceptance=69.5%, cycles=39,
phases[prefill=347.2ms draft=187.8ms(first=9.3/incr=178.5) verify=130.0ms replay=0.3ms commit=5.2ms]
```

Prefix snapshot cache is live (9 snapshots, TTFT drops 0.79→0.36 s across
requests). SSE streaming works through DFlashEngine (132 `data:` chunks).

## Correctness gate — FAIL (6/8 prompts), with analysis

Pre-registered gate: speculative greedy output token-identical to plain decode
for ≥8 prompts × 128 tokens. Result: **2/8 identical, 6/8 diverge** (sha256 of
tokenized continuation ids, deterministic OFF digest across both runs).

| prompt | identical | first diff (byte offset) | bytes differing in common prefix |
|---|---|---|---|
| 1 | no | 74 | 481 (tail rewrite) |
| 2 | no | 145 | 419 |
| 3 | no | 333 | 195 |
| 4 | **yes** | — | 0 |
| 5 | **yes** | — | 0 |
| 6 | no | 364 | **1** (single-token flip, then re-converges) |
| 7 | no | 142 | 374 |
| 8 | no | 300 | 209 |

All divergences are single argmax flips followed by legitimate rewrite (prompt
6 differs by exactly one byte in 585). This is the numerics-gate shape: a tiny
logit difference flips a near-tie, then text diverges. On the M2/Vulkan backend
the DFlash verify pass computes logits for 16-token blocks (batched GEMM
tiling, sdpa over 16 positions) while plain decode is 1-token steps; the two
routes are not bitwise identical on this backend, and near-ties flip. The
macOS stock reference below shows whether the same pair is identity-clean on
Metal — that contrast isolates "omarchy verify-path numerics" from "DFlash
losslessness claim".

Per docs/numerics-gate.md this is a route change that would need the
teacher-forced top-1 ≥ 99% + first-divergence ULP-gap analysis before being
called acceptable; that logit capture on the M2 is the named follow-up
(KernelBattery/NumericsGate class, ~1 ticket).

## macOS stock reference (macstudio, M1 Ultra)

Environment: throwaway venv `/tmp/omlxdflash-v2` (homebrew python3.13), **stock**
PyPI mlx 0.32.2 (Metal true) + stock mlx-lm + the same dflash-mlx wheel — no
omarchy patches. Same pair downloaded into the macstudio HF cache (2.21 +
1.08 GB, announced). Memory rules held: pre-check 83 GB free+inactive, during
74–79% free, pressure normal throughout (`mem-during.log`); advisor untouched.

Baseline here is stock `mlx_lm.stream_generate` (the dflash runtime has no
draftless mode — `SpeculativeSession.open` requires a drafter;
`draft_backend.make_cache(draft_model=None)` raises `AttributeError`), DFlash
path is `load_runtime_bundle` + `stream_dflash_generate`, both greedy, same 8
prompts × 128 tokens, tokenized-id comparison.

| Metric | result |
|---|---|
| greedy token-identity vs plain | **0/8 identical** (fails on stock macOS too) |
| mean acceptance (per prompt) | 41.3% (range 24–55%, cycles 57–97) |
| plain tok/s / DFlash tok/s | 4.8 / 5.5 mean (**contention caveat** below) |
| in-run speed ratio | DFlash ≈ 1.15x plain (vs 0.17x on the M2) |

**Contention caveat:** absolute macstudio tok/s are unreliable — a control
plain-decode probe measured 5.7 tok/s for the same 4-bit 4B (expected ~100 on
an idle M1 Ultra) while ChatGPT-Codex and Helium helper processes held 100–150%
CPU each. Identity and acceptance are compute-count results, not timing
results, and stand; speed numbers are only meaningful as the within-run ratio.

**Reading of the identity result:** the strict gate (speculative greedy output
token-identical to plain decode) fails on stock macOS as well — 8/8 diverge
there vs 6/8 on the M2, and the two prompts that matched on the M2 (4, 5)
diverge on macOS. Batched verify logits (16 positions per pass) are simply not
bitwise-equal to 1-token-step decode logits on either backend, so near-tie
argmax flips are expected everywhere. The claim "temp=0 is bit-for-bit
reproducible" in `docs/experimental/dflash_mlx_integration.md` does not hold
against plain decode on either backend; it may only compare DFlash runs to
DFlash runs. The honest replacement bar is the repo's numerics gate
(docs/numerics-gate.md): teacher-forced top-1 ≥ 99% plus first-divergence
top-2-gap ≤ 1 bf16 ULP — needs one captured-logits ticket (KernelBattery/
NumericsGate class).

**Secondary observation:** acceptance differs across backends for the same
pair and prompts (one M2 server sample 69.5% vs macstudio mean 41.3%). The M2
serves through omlx (patched mlx-lm 0.31.4.dev + omarchy context defaults);
macstudio runs stock offline defaults — backend numerics and/or runtime-config
delta, not separated by this run.

Post-run advisor smoke (required by macstudio rules): real completion from the
advisor's loaded `mlx-community--Qwen2.5-0.5B-Instruct-4bit` alias,
`"Red, Blue, Yellow."` (6 tokens) inside `curl --max-time 30` — advisor
healthy, no starvation.

Housekeeping: `/tmp/omlxdflash-venv` (abandoned python3.9 venv attempt) could
not be removed (rm guard); macOS /tmp will reclaim it.

## Perf decomposition (measured, 2026-10-05 post-reopen probes)

Cycle-trace instrument results (69-cycle 128-token DFlash run): the cycle wall
is **GPU-route time, not host time** — verify 152 ms/cycle (80%), draft 34.5,
accept 2.6, hidden 0.7, rollback 0.02, other 0.17 ms (host/Python negligible;
yield pause 0.03 ms). The runtime's adaptive policy shrank blocks 16→4 and
mean commit fell to 1.86 tok/cycle. Follow-up probes isolated the cause:

| measurement | result |
|---|---|
| raw mlx-lm forward, q_len=1 | 25.4 ms |
| raw forward, q_len=4 / q_len=16 | 144 / 130 ms (**flat**) |
| prefill 154 tokens (server) | 347 ms (2.3 ms/token amortized) |
| isolated mx.quantized_matmul M=16 (4-bit, gs64, K=2560→N=4096) | 0.99 ms |
| isolated dense matmul M=16 | 2.7 ms |
| isolated mx.fast.sdpa masked, q16 × kv160 | 0.41 ms |
| swapping 252 qmm linears → dense (dflash verify-linear) | verify unchanged 136.6→137.1 ms; committed tokens bit-identical |
| dispatch counts (MLX_OMARCHY_TRACE_DISPATCH) | decode step ~377, q16 step ~1,064 → **~118 µs/dispatch end-to-end (M≥2 route) vs ~45-66 µs (decode)** |

Root cause: **the multi-row (2≤M≤~64) forward route on the omarchy Vulkan
backend is dispatch-bound** — per-dispatch end-to-end cost ~2.5x decode's,
with trivial GPU compute per dispatch (all isolated ops are sub-2 ms; the
whole q16 forward's compute is <30 ms). DFlash verify is one such forward per
cycle, hence 137-152 ms, hence DFlash ON ~5.9x slower than plain despite 69.5%
acceptance. This is a backend/KernelBattery-class finding, not a dflash or
numerics bug. KernelBattery item: the M16 ktmpl qmm kernel
(dflash-mlx verify_qmm.py, `_resolve_m16_ktmpl_variant`, 71f7c2c) crashes with
the exact named error `[omarchy] fast::CustomKernel MSL subset: unsupported
MSL feature 'simdgroup_matrix' … (dtype=bfloat16, shape=[16,1024])`.

Candidate fixes, ranked: (1) cut dispatch count on the multi-token forward
(mx.compile/fusion of elementwise chains — dflash runtime patch layer or
backend); (2) backend dispatch batching on Honeykrisp; (3) ~~block-size
amortization~~ **measured NOT viable**: the z-lab b16 drafter clamps the block
at its trained 16 (`block_tokens` 32/64 requested → `block_seen_max` 16), and
verify stays ~146-149 ms from block 8 through 16 — in-process throughput
9.8-10.0 tok/s across the whole sweep (blk_sweep.json), matching the server's
10.59. The dispatch-bound route can only be fixed by reducing dispatches
(MidM's multi-row fused kernel lane) or backend batching. Secondary: the
runtime's adaptive block policy shrank 16→4, the wrong direction under a flat
per-cycle cost — a dflash-side policy note for the upstream draft. Also
recorded: offline in-process acceptance 46% vs the omlx server's 69.5% for
the same pair+prompts — the offline runtime config differs from the serve
path (draft window/sink or capture config), worth one look before anyone
compares acceptance numbers across the two paths.

### Post-MidM A/B (wheel +28c87d5, SDPA rows + qmm token rows 1..8 landed; rows 9..16 still compose)

Ran in the 09:00Z-window ticket (03:07 CDT). **No measurable improvement on
this path**: plain decode 57.8-58.0 tok/s (was 59-63 on b8af62c); forward
walls q1 25-27 ms, q4 ~132-136 ms (first rep 218 ms), q8 ~134 ms, q16
~127 ms — q8 is inside the landed rows-1..8 range yet still costs ~134 ms;
DFlash block 8: verify 147.6 ms/cycle, 9.39 tok/s (was 146.4/9.76); block
16: 148.3 ms, 10.0 tok/s (was 148.1/10.01); committed tokens bit-identical
to all prior runs. Reading: either the fused rows do not engage for this
model's shapes (GQA 32/8 heads, 4-bit qmm) or the mid-M cost is dispatch
COUNT/submission overhead, which per-row kernel fusion does not reduce —
consistent with the flat-vs-M dispatch math above. Numbers for MidM:
midm_ab.json (+28c87d5 stamp recorded). The full re-check repeats when rows
9..16 land (then M=16 verify should engage the fused path end to end).

## Artifacts

- dev box: `/tmp/a7-artifacts/` (16 responses, both server logs, compare.json,
  divergence table, stream captures)
- notebook: `artifacts/OmlxDflash/dflash-a7/` (+SHA256SUMS) and entry
  `entries/OmlxDflash/20261005T011156Z-m2-jw14m2-omlx-dflash-a7.md`
