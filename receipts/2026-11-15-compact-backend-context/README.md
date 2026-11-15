# Compact pair chat model at a 16384-token backend context - 2026-11-15

[Joshua's agent]

**Question.** Qwen3-4B-Instruct-2507-4bit (the `compact` pair's chat model) had no
`extension.backend_context_qualified_tokens`, so the pair admitted the 4096-token default.
omp sends an 8.6 KB system prompt and 52 tool schemas (about 5.7k tokens); the pair server
refused it (`prompt (5724 tokens) + max_tokens (30) exceeds the admitted context budget of 4096`)
and the engine showed the user a 502. Does the chat worker serve 16384 tokens, and what does it
cost in memory?

**Method.** `scripts/qualify_backend_context.py` on the 14-inch M2 Max (96 GB, Linux), installed
omarchy-mlx 0.7.41-2 (before the allocator cache cap of #436), mlx-lm 0.31.3, no other GPU job running.
It adopts the saved `compact` lock (4096), starts the pair, calls `ensure_context(16384)` (the dev
flag `MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1` lifts the catalog cap for this run only), then sends three
requests straight to the chat worker. A sampler (every 0.5 s) records the lowest system MemAvailable
and the chat worker's highest RssAnon, RssFile, RssShmem and VmHWM. Two runs: `receipt-run1.json`
(no memory sampler) and `receipt.json` (run 2, with the sampler and the launch flags).

| Step (run 2) | Result |
|---|---|
| `ensure_context(16384)` from the 4096 lock | ok, context 4096 -> 16384, chat worker restarted at the new cap |
| omp-shaped request (8.6 KB system + 52 generated tool schemas, max_tokens 30) | HTTP 200, 9740 prompt tokens, 40.6 s, reply "ok" |
| needle at the start of a 15488-token prompt, max_tokens 64 | HTTP 200, 89.0 s, the code was returned (7316) |
| prompt over the cap (21259 tokens + 30) | HTTP 404 `... exceeds the admitted context budget of 16384 ...`, 0.03 s |
| small request right after the refusal | HTTP 200 |

Run 1 gave the same results (40.4 s, 88.5 s). The generated tool schemas are heavier than omp's
(9740 tokens against 5724), so this covers omp.

**Prefill step.** The worker was launched with `--prefill-step-size 511` (`prefill_step_size` in the
receipt): the 512 default, cut by `effective_prefill_step` so one fp32 score tensor fits a storage
buffer binding at 16384 tokens.

**Memory (GiB, 1024^3 bytes).**

| Quantity | GiB |
|---|---|
| Chat worker peak VmHWM, run 2 (run 1: 16.67) | 16.88 |
| of which peak RssShmem (GPU buffers) | 16.55 |
| Chat worker RssShmem right after the restart at 16384, idle | 0.27 |
| System MemAvailable lowest minus its baseline before the pair started (both workers) | 17.54 |
| Catalog estimate at 16384 without prefill scratch | 5.36 |
| Catalog estimate at 16384 with the step-511 scratch | 7.74 |

The scratch formula still reads less than half of the measured peak: the memory is taken during the
long prefill and is shared memory, not anonymous. This build also keeps freed buffers in an
allocator cache; #436 caps that cache and this run predates it, so the peak may be lower on a build
with #436 (not measured; unverified).

**What the catalog does with it.** `extension.backend_context_peak_bytes` = 18,123,325,440 bytes
(16.88 GiB, the run-2 VmHWM). Above the 4096 default, `pairs.chat_estimate` takes the larger of the
stepped estimate and that peak, so admission, the reservation and the worker claim all use it. A
16 GiB machine (assuming 1.5 GiB for the decision worker) stays at the default; 24 GiB admits 16384.
The 1.5 GiB is an assumption of the test, not a measurement.

**Not measured.** Other chat models (each keeps the 4096 default), other machines (including a 16 GB
or 24 GB one: the admission outcome there comes from the arithmetic above, not a run), concurrent
requests at 16384, quality beyond the one needle, any context above 16384, a build with #436.

**Reproduce.** `MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1 MLX_OMARCHY_OFFLINE=1 MLX_OMARCHY_HOME=<home with
the saved compact lock> PYTHONPATH=<serve tree> python scripts/qualify_backend_context.py --home <home>
--pair compact --tokens 16384 --out receipt.json`
