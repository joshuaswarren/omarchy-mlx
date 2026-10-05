# oMLX API-surface parity — rows A1, A2, A3, A12, A13, A18, A31

Owner: OmlxApi. Pin: oMLX v0.7.0 (`4d4f5a28`). Pre-registration:
notebook `entries/OmlxApi/20261005T0915Z-devbox-omlx-api-parity.md`.

## Evidence classes

- **Dev-box protocol tests (SUPPORTING ONLY — this host has no mlx wheel):**
  - `packaging/omlx-linux/test_api_protocol.py` → 7/7 PASS
    (artifacts `OmlxApi/api-parity/devbox_protocol_tests.json`):
    T1 profile validation/slugify (A13), T2 settings-manager
    alias/profile/expose/apply — exposed id composes as `<alias>:<profile>`
    = `alias-x:par` (A13), T3 usage-history sqlite round-trip (A31),
    T4 MCP config loader on the repo's own mcp.example.json (A18),
    T5 websearch providers + error payloads + honest-refuse without the
    optional markitdown dep (A31), T6 json_schema validation + fenced-output
    parsing (A18), T7 Anthropic→internal request mapping (A2).
  - `packaging/omlx-linux/api_parity.sh` (harness) exercised end-to-end against
    `packaging/omlx-linux/mock_omlx_server.py` (wire-format stub copied from
    omlx/server.py @ pin): **27 PASS / 0 FAIL / 1 SKIP** (skip = MCP section
    by design without `--start`). Log: `devbox_harness_mockrun.log`.
  - `packaging/omlx-linux/mcp_echo_server.py` verified against real `mcp` 2.x
    (omlx pins `mcp>=2,<3`): tool registers, call returns `echo:parity-ok`.

## REAL-run receipts (M2/jwm1) — pending the M2 FREE window

One command inside a gpu-turn ticket, against the shared venv:

```
packaging/omlx-linux/api_parity.sh --start \
  --venv /tmp/omlx-home/.venvs/omlx \
  --hf-cache /tmp/omlx-home/.cache/huggingface/hub \
  --model mlx-community--Qwen3-4B-Instruct-2507-4bit \
  all
```

Per-row acceptance (pre-registered; full detail in the notebook entry):

| Row | What the receipt must show |
|---|---|
| A1 | /v1/models 200 with real max_model_len; chat completions stream+non-stream real text; /v1/completions text |
| A2 | /v1/messages Anthropic shape + usage; count_tokens; stream lifecycle message_start→…→message_stop with usage; thinking request accepted (block presence recorded — Qwen3-4B-Instruct-2507 is non-thinking) |
| A3 | include_usage final chunk real counts; `: keep-alive` SSE comments in a long stream; max_model_len == checkpoint config.json context |
| A12 | embeddings dim>0 + cosine ordering; rerank top = relevant doc — **blocked on model download approval** (see below) |
| A13 | alias in /v1/models + request-by-alias; profile create/expose/apply (`<model>:<profile>` serves) |
| A18 | tool_calls parsed; response_format json_schema conforms; MCP echo round-trip (server started with --mcp-config) |
| A31 | /api/web-search/test real results (ddgs, M2 internet); chat search path; /api/usage rows for this run |

## A12 blocker (needs Main)

No embedding/rerank model exists in any fleet HF cache (greps on M2, jwm1,
macstudio 2026-10-05 ~09:07Z). Native loaders mean TWO small downloads suffice:
`mlx-community/all-MiniLM-L6-v2` (~90 MB, BERT arch) and
`mlx-community/Qwen3-Reranker-0.6B-4bit` (~0.6 GB, CausalLM reranker path,
`omlx/models/reranker.py:6-11`). Requested from Main; a12 skips honestly until
approved.

## Status

- A1 A2 A3 A13 A18 A31: harness green on dev box; REAL receipt pending M2 run.
- A12: harness green (mock); REAL receipt pending download approval + M2 run.
