# Chat-model benchmark — 2026-09-30

**Question.** Why does Everyday use a 2B chat model on the M2 (96 GB M2 Max),
and which chat model should each default pair use? Data only, no product
code changes.

**Method.** Run each candidate model on the idle M2 through the project's
mlx-lm HTTP shim path (`serve/mlx_omarchy_serve/_mlxlm_server.py`) with the
patched mlx-lm 0.31.3 from the live venv. Greedy, thinking disabled, 2
warmups then 5 measured turns. Per-model process (clean cold/warm load),
frozen prompt corpora (hashed before the run), and the coordinator's
`SCHEMA_PROMPT_COMPACT` for ordinary chat (full `SCHEMA_PROMPT` when the
prompt names chart/graph/form/decision/options/facts/sources) with
`repetition_penalty=1.1` and `max_tokens=1200`. Peak memory recorded via
`mx.get_peak_memory`, process RSS, and `/proc/meminfo` MemAvailable delta.

## Machines and host identity

- M2 Linux: `jw14m2-linux` (T6021, 96 GB, kernel 7.1.13-ARCH-polltx,
  Honeykrisp Vulkan, driver Mesa). boot_id captured in the notebook.
- M2 macOS: not used (no Tailscale AC required for this bench).
- GPU lock: `flock /tmp/m2-gpu.lock` per the shared fleet rule.

## Candidate models and revisions

| Catalog ID | HF repo | Revision | Cached? |
|---|---|---|---|
| qwen3.8-2b-4bit | SiddhJagani/Qwen3.8-2B-mlx-4Bit | 0867d98bfb174b042d88461c0e7c97b86b34b381 | yes (snapshots) |
| qwen3.5-9b-mlx-4bit | mlx-community/Qwen3.5-9B-MLX-4bit | 938d8919941c6e7efd3c7150eff7fe9d12afa631 | yes |
| ministral-3-8b | mlx-community/Ministral-3-8B-Instruct-2512-4bit | 182f003f01daa75f9de0f2c4d379722fd0bc1c61 | yes |
| qwen3.8-27b-4bit | mlx-community/Qwen3.8-27B-4bit | 10c35caafbb80f7dc6a7a432cdd11af10a6d4818 | yes |
| gemma-4-31b-it-4bit | mlx-community/gemma-4-31b-it-4bit | 696d436c404745a59f30e4939a658162b0a9e57f | yes |
| qwen3.8-27b-mxfp4 | mlx-community/Qwen3.8-27B-mxfp4 | 97ab0819817ab1c61d7d39f9169fc71999915641 | downloaded (15 GB) |
| novaeon-35b-moe | NovaeonStudio/Qwen3.8-35B-A3B-Distill-oQ8-fp16-mtp | 5315bc8a40d751f5bded428eadd6a22007a3bf17 | downloaded (37 GB) |

## Frozen prompt corpora (hashed before the run)

```
138d8c371041dde0dd729d871dbc502555a17c733b6b940b76bf39f995ebd7ee  cards_16.jsonl
48593e7db1ed6c253dfea7ce565abe4fb53c05c9555e094870a7d9296ec14dac  gsm8k_20.jsonl
ea83563f77235360c64b19010e0c7e14a6d889b6bbf613b709aa67afc0af52bc  ife_20.jsonl
```

## Headline table

(populated after all per-model runs complete; see `summary.csv`,
`summary.md`, and individual `runs/<label>.json` files)

## Quality proxy disclaimer

GSM8K and IFE are reproducible quality proxies — **not** benchmarks. The
IFE set is 20 deterministic instruction-following checks written for this
study (hashed before the run). GSM8K uses 20 hand-picked items from the
public GSM8K test set with gold answers. Both are small and biased toward
short-form tasks; use them to rank models, not to claim SOTA.

## Notebook

`~/.local/share/apple-silicon-lab/entries/ModelBench/2026-09-29T19-58Z-jw14m2-linux-chat-model-bench.md`
(persisted by Main; artifacts under `~/.local/share/apple-silicon-lab/artifacts/ModelBench/chat-model-bench/`)

## Status (filled by Main on completion)

- [ ] All 5 cached models measured end-to-end
- [ ] mxfp4 + MoE smoke tested (load + one generation each)
- [ ] summary table built (`aggregate.py`)
- [ ] recommendations: 96 GB M2 default and 16 GB default chosen by data
- [ ] public-download eligibility called out per model
- [ ] data says 2B is or is not the right Everyday on this hardware