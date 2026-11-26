# serve-bench — isolated multi-server serve benchmark

Audit harness for the 2026-09-19 serve receipts
(`receipts/2026-09-19-mlxserve-linux-port-t6001-test-host.md`,
`receipts/2026-09-19-serve-options-bench-t6001-test-host.md`). Client-side only; it
starts and probes real servers and ships no serving code.

## Files

- `serve_bench.py` — harness (stdlib only). `--selfcheck` runs the full
  client path against an in-process stdlib mock; no GPU needed.
- `run-window.sh` — GPU-window wrapper: stop `llm-inference` → flock
  `/tmp/m1-gpu.lock` → bench → release lock → restart + health-check the
  service. Announces TAKE/RELEASE with the lock inode for peer
  deconfliction. The lock is released BEFORE the service restart
  (ExecStart holds `flock --nonblock`; restarting while held fails —
  observed 2026-09-19).

## What it measures (per leg)

- `end_to_end_tok_s` — completion tokens / HTTP wall, non-streaming
  request. The only metric comparable with the 09-19 receipt numbers
  (mlx_lm 10.36 / oMLX 33.5 / mlx-serve 5.65).
- `ttft_s` — first streamed chunk of a `stream=true` request.
- `decode_tok_s` — `(completion_tokens - 1) / (stream_wall - ttft)`
  (decode-slope approximation).
- `concurrency_tok_s` — optional `--concurrency N` phase, aggregate
  tokens / wall.
- Server RSS after load and after rounds + total safetensors bytes
  (weights→peak factor for admission control).
- Optional `--engine-control`: bare `mlx_lm` `stream_generate` with the
  same prompt/max_tokens, no HTTP — splits backend decode cost from
  serving-layer overhead.

## Differences from the 09-19 method

- Legs run sequentially, one fresh server process per leg (zero
  cross-leg GPU/cache bias; four 15 GB residents do not fit 62 GB).
  `--resident` restores the legacy co-resident mode for comparability.
- Identical warmup (count, tokens, prompt) for every leg; `prompt_tokens`
  and any server-reported `cached_tokens` recorded per request.
- Runtime pins fatal-checked before any request: mlx-omarchy version,
  libmlx sha256-16, mlx-serve binary sha256-16.

## Example

```sh
run-window.sh \
  --model-snapshot ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/<rev> \
  --hf-id mlx-community/Qwen3.8-27B-4bit \
  --expect-version 0.32.3.dev202609190758+50eeb29 \
  --expect-libmlx df3d4e74c597956c \
  --legs mlxlm,omlx,mlxserve --rounds 5 --concurrency 4
```

Results land in `--outdir` (default `/tmp/servebench`):
`results-servebench.json` + per-leg server logs.
