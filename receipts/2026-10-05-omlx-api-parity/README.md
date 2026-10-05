# oMLX API-surface parity — rows A1, A2, A3, A12, A13, A18, A31

Owner: OmlxApi. Pin: oMLX v0.7.0 (4d4f5a28). Pre-registration: notebook entry OmlxApi/20261005T0915Z-devbox-omlx-api-parity.md.

## Evidence classes

- Dev-box protocol tests (supporting only; no mlx wheel): test_api_protocol.py, 7/7 PASS; api_parity.sh against mock_omlx_server.py, 27 PASS / 0 FAIL / 1 expected skip (MCP not started). MCP echo helper separately checked with upstream mcp 2.x. Supporting results only, not parity evidence.
- Real run: M2 ticket B5, 2026-10-05 11:40Z. oMLX 0.7.0 on real Apple Silicon, mlx-lm 0.31.4.dev132+g94cdcae13 and mlx-omarchy 0.32.4.dev202610050725+5c15fba. Qwen3-4B-Instruct-2507-4bit, all-MiniLM-L6-v2-bf16, Qwen3-Reranker-0.6B-4bit, and live MCP echo server. **28 PASS / 0 FAIL / 0 SKIP; exit=0.**
- Raw evidence and checksums: lab notebook artifacts/OmlxApi/api-parity/run5-final/ (parity_run5.log, server.log, per-check JSON, SHA256SUMS).

## Row results

| Row | Status | Real M2 observation |
|---|---|---|
| A1 | DONE | /v1/models listed the served model; chat nonstream and stream returned text/usage; legacy /v1/completions returned text. |
| A2 | DONE | Anthropic Messages shape and usage; stream lifecycle (20 events); count_tokens=10; thinking request accepted. Qwen3-4B-Instruct is non-thinking; response contained text only. |
| A3 | DONE | include_usage counts after 8 chunks; max_model_len=262144; one OpenAI noop keepalive chunk arrived before last data chunk. |
| A12 | DONE | 384-dimensional embeddings; cosine(sim)=0.998 vs cosine(dis)=0.996; relevant rerank score 0.9765625 vs 0.0004043571 and ranked first. |
| A13 | DONE | Alias listed and served; exposed parity-alias-model:parity served; profile apply returned overlaid setting; model type override accepted/persisted. Harness reset settings and removed profile. |
| A18 | DONE | Tool-call JSON parsed; JSON schema output name=Ada, age=36; MCP connected with one tool, called echo__echo with text parity-ok, then returned echoed assistant response on follow-up turn. |
| A31 | DONE | Admin web-search test and /v1/web/search each returned 3 results; usage history contained 3 model rows and 74 requests. |

## Pre-registered acceptance

| Row | Criterion exercised |
|---|---|
| A1 | /v1/models, streamed and non-streamed chat completions, legacy completions. |
| A2 | Anthropic Messages, token counting, message stream lifecycle/usage, and thinking request acceptance. |
| A3 | Streaming usage, keepalive wire form, and model context reporting. |
| A12 | Embedding dimension and semantic ordering; relevant reranker document ranked first. Embedding model substitution: all-MiniLM-L6-v2-bf16 because the unquantized repo lacked safetensors. |
| A13 | Per-model alias, exposed profile, request by alias/profile, profile application, type override. |
| A18 | Tool-call parsing, schema-constrained output, and two-turn live MCP tool round-trip. |
| A31 | Real results from admin test and server-side /v1/web/search; usage recorded for requests. Web search is a server-side endpoint, not injected into /v1/chat/completions tools. |

The harness is packaging/omlx-linux/api_parity.sh; use --list for sections. Real-run invocation:

    packaging/omlx-linux/api_parity.sh --start --venv /path/to/omlx-venv --hf-cache /path/to/hf-cache --model mlx-community--Qwen3-4B-Instruct-2507-4bit all

No upstream omlx source changes were made. No Metal-only code patch was required for these rows.
