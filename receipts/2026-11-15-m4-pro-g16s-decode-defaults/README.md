# M4 Pro (G16S): GDN native GQA and split-K decode attention by default

Backs the two commits of `m4/g16s-followup-v2` on top of #439 (`a356ccda`):

- `5a20c641` omarchy: GDN decode with native GQA by default on the M4 Pro
- `efadfbb4` sdpa: split-K decode attention defaults on G16S

## Host

Apple M4 Pro (J616s, T6040, G16S B1), 14 CPU cores, 48 GB, kernel 7.3.0, Mesa 26.3.0-devel (`git-05ddb00f5c`) Borealis, stock clocks, no Mesa or MLX environment variables set (`icd=system`). Full host facts: `receipts/2026-11-13-m4-pro-g16s-gpu/host.txt`.

Wheels (`wheels.sha256`): base `0.32.4.dev202611150046+a356ccd`, head `0.32.4.dev202611150048+efadfbb`, same build script, same venv otherwise.

## Model decode (`models.txt`)

`tools/gen2.py` (`mlx_lm` greedy, chat prompt "Explain in three sentences why the sky is blue.", 128 tokens, granite stops at 53; a 4-token warm-up generate first), one process per run, three interleaved rounds base/head, median. Each run also records prefill at about 300 and 2000 tokens and peak memory; the line before each round records the CPU/GPU temperature.

| model | base tok/s | head tok/s | change | text sha |
|---|---|---|---|---|
| Qwen3.5-4B-MLX-4bit | 55.27 | 60.89 | +10.2% | identical (`8d04f5ffd6c2`) |
| Qwen3-8B-4bit | 42.23 | 42.42 | +0.4% (noise) | identical (`9ee73549c37f`) |
| granite-4.0-1b-4bit | 111.84 | 111.79 | 0 | identical (`288995c87e1f`) |
| Qwen2.5-1.5B-Instruct-4bit | 138.29 | 136.73 | -1.1% (round spread 132.8-142.4) | identical (`6fa09084770d`) |

Prefill is unchanged within 0.3% on every model. Qwen2.5-1.5B and granite take neither route (no linear attention, short cache), so their rows are the noise floor.

## Long context decode (`long/`)

`receipts/2026-11-13-sdpa-split-g13s-g14g/tools/longctx.py` (the llmstep.py text repeated to the prompt length, one warm-up), 128 greedy tokens, two rounds of two runs, median decode tok/s:

| model | prompt tokens | base | head | change |
|---|---|---|---|---|
| Qwen3-8B-4bit | 970 | 36.60 | 40.98 | +12.0% |
| Qwen3-8B-4bit | 1939 | 31.07 | 37.47 | +20.6% |
| Qwen3-8B-4bit | 3775 | 25.38 | 35.18 | +38.6% |
| Qwen3.5-4B-MLX-4bit | 970 | 53.34 | 57.94 | +8.6% |
| Qwen3.5-4B-MLX-4bit | 1939 | 49.98 | 57.61 | +15.3% |
| Qwen3.5-4B-MLX-4bit | 3775 | 44.87 | 55.41 | +23.5% |

Generated token hashes (`tokens_sha`) are identical between base and head in every file. Prefill is unchanged.

## Split-K thresholds (`split-k/`)

`split-k/sweeps`: `sdpa_bench_kv.py`, bf16 decode attention 224 to 16384 keys at head width 128 (a) and 256 (b), independent calls and dependent chains, per variant (default, split, value-phase pass 1 `pv`, 64-key blocks `bk64`), two runs each. The thresholds in `sdpa_split_min_keys` and the `pv`/`bk64` limits are the smallest lengths where the variant wins in both shapes.

`split-k/decode`: model decode with each variant forced at 443 to 4000 tokens (`long-*.json`), and `hd256-pv-tokens-1400.txt` (`tokens.py`): at head width 256 the value-phase pass 1 changes the generated tokens from token 17 on at a 1327-token prompt, so that variant stays off at width 256.

## Test battery (`battery.txt`)

Head of the branch, one process per suite under `flock /tmp/m1-gpu.lock timeout 1800`, same runner as `receipts/2026-11-13-m4-pro-g16s-gpu`: 79 of 79 units exit 0 (suites, the 8 capability-simulation profiles, and the 16 flash prefill cases one per process). `omarchy_ane_runtime_tests` is not built here: it links against the ANE runtime sources, as in the earlier receipt.

`SHA256SUMS` covers every file in this directory.
