# Compact pair chat model at a 16384-token backend context - 2026-11-15

[Joshua's agent]

**Question.** Qwen3-4B-Instruct-2507-4bit (the `compact` pair's chat model) had no
`extension.backend_context_qualified_tokens`, so the pair admitted the 4096-token default.
omp sends an 8.6 KB system prompt and 52 tool schemas (about 5.7k tokens); the pair server
refused it (`prompt (5724 tokens) + max_tokens (30) exceeds the admitted context budget of 4096`)
and the engine showed the user a 502. Does the chat worker serve 16384 tokens?

**Method.** `scripts/qualify_backend_context.py` on the 14-inch M2 Max (96 GB, Linux), installed
omarchy-mlx 0.7.41-2, mlx-lm 0.31.3, kernel 7.3.0-2-20261110.1-lab-ARCH, quiet-gpu solo (idle-guard
ticket, class=validation). It adopts the saved `compact` lock (4096), starts the pair, calls
`ensure_context(16384)` (the dev flag `MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1` lifts the catalog cap for this
run only), then sends three requests straight to the chat worker. Raw output: `receipt.json`.

| Step | Result |
|---|---|
| `ensure_context(16384)` from the 4096 lock | ok, context 4096 -> 16384, chat worker restarted at the new cap |
| omp-shaped request (8.6 KB system + 52 generated tool schemas, max_tokens 30) | HTTP 200, 9740 prompt tokens, 40.4 s, reply "ok" |
| needle at the start of a 15488-token prompt, max_tokens 64 | HTTP 200, 88.5 s, the code was returned (7316) |
| prompt over the cap (21259 tokens + 30) | HTTP 404 `... exceeds the admitted context budget of 16384 ...`, 0.02 s |
| small request right after the refusal | HTTP 200 |

The generated tool schemas are heavier than omp's (9740 tokens against the 5724 omp sent), so
this covers omp.

**Memory, not reconciled.** The chat worker's peak RSS was 17.5 GB and MemAvailable fell by about
10 GB over the run, against the catalog estimate of 5.4 GiB at 16384. RSS counts mapped weights
and driver-mapped buffers and other processes may have moved MemAvailable, so neither number is a
clean per-worker figure (unverified). The limit applies the existing fit check
(`ensure_context` / `admissible_context` still refuse a context the machine cannot hold by the
estimate); this run does not re-measure that estimate.

**Not measured.** Other chat models (each keeps the 4096 default), other machines, concurrent
requests at 16384, quality beyond the one needle, any context above 16384.

**Reproduce.** `MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1 MLX_OMARCHY_OFFLINE=1 MLX_OMARCHY_HOME=<home with
the saved compact lock> PYTHONPATH=<serve tree> python scripts/qualify_backend_context.py --home <home>
--pair compact --tokens 16384 --out receipt.json`
