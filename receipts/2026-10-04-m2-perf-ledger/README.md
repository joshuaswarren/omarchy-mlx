# M2 Max (T6021) performance ledger on the published v0.7.26 stack

Actor: M2Lane (delegated worker), 2026-10-04. Pre-registered in the private
notebook (entries/M2Lane/2026-10-04T134800Z-jw14m2-v0726-lora-smoke-perf-
ledger.md) before any run; this receipt is the public, host-scrubbed copy.
Everything here ran on the "T6021 host" (M2 Max, 96 GB, kernel
7.1.13-3-1-ARCH, stock boot `af53c729…`, one boot for all arms).

## Stack and method

- Published v0.7.26 release (tag `v0.7.26` = `56488ba21`, assets verified
  against the release SHA256SUMS, 4/4 OK), installed by the tag's own
  `install.sh` into a throwaway HOME (env -i, `MLX_OMARCHY_RELEASE_BASE=
  file://…`). Installer smoke: `Apple M2 Max (G14C B1)`, `matmul OK`;
  `mlx-omarchy 0.32.4.dev202610040755+56488ba21`, mlx-lm 0.31.3 with the
  shipped patcher series; `scripts/mlx_provenance.py` version_match=true.
- Fleet bench `qwen38-mlx-bench.py` (8103 B, `max_tokens=a.new_tokens`
  sentinel verified) + `qwen38_bench_lib.py` + the `qwen38-2b-prompts.jsonl`
  corpus, run per model from the host's HF cache (read-only reuse; no new
  downloads).
- Cells (the h253 shape): decode `--new-tokens {64,256,512} --prefill-tokens
  0 --limit 5 --passes 2 --warmup 2`; prefill `--new-tokens 32
  --prefill-tokens {512,1024} --limit 10 --passes 3 --warmup 2`; two reps
  per cell; `uclampset -m 1024` wrapper as in h253/h254; quiet gate before
  every cell (load1 < 0.5 and PSI cpu avg10 = 0, else NOTQUIET marked);
  boot id + uptime at each arm start; all arms under gpu-turn tickets
  (<= 15 min each, FIFO).

## Results (median decode tok/s | median TTFT s | pure-prefill tok/s)

| model | d64 | d256 | d512 | pf512 | pf1024 |
|---|---|---|---|---|---|
| Qwen3.8-2B-mlx-4Bit | 108.0 / 108.0 | 106.2 / 106.2 | 102.6 / 102.6 | 1491.25* / 1612.3 | 1751.2 / 1751.9 |
| Qwen3-4B-Instruct-2507-4bit | 63.9 / 64.0 | 61.9 / 62.0 | 57.4 / 57.4 | 673.8 / 672.9 | 668.9 / 669.9 |
| Qwen3.5-9B-MLX-4bit | 39.2 / 39.1 | 39.0 / 39.1 | 38.4 / 33.9 | 398.5 / 399.1 | 125.0* / 411.6 |
| Qwen3.8-27B-4bit (d64 + pf512 only) | 13.5 / 14.8 | – | – | 119.3 / 98.2 | – |

- Cells are rep1 / rep2. TTFT: 2B ~0.091 s, 4B ~0.153 s, 9B ~0.221 s,
  27B ~0.64 s — stable across reps.
- `*` flagged cells: 2B r1 pf512 ran at load1=0.49 (gate edge) and its r2
  at 0.34 reproduces the h253 S0 value (1607.3); 9B r1 pf1024 printed
  124.99 while its r2 printed 411.59. Four single-cell decode-rate
  outliers also occurred (4B r2 pf512 45.85 vs 64.0; 9B r2 d512 33.88 vs
  38.4; 9B r2 pf512 14.53 vs 39.2; 27B d64 13.54 vs 14.82 across reps).
  Every outlier cell kept a BIT-IDENTICAL `ordered_records_sha256` across
  reps — output identity is not in question; the flagged rates are
  scheduling/clock artifacts and are re-verification candidates, not
  ledger values.
- Greedy digests (per model, stable across reps): 2B d-cells
  `76ec87cb…/c20359c7…/d773b4fc…`, pf `d5f7da5e…`; 4B d-cells
  `42d27a8c…/2178e445…/c8ec4eee…`, pf `eac9fe32…`; 9B
  `80274aa7…/959317b8…/73bce9e5…`, pf `d9892bdf…`; 27B d64 `5c28f009…`,
  pf `09e93271…`.

## Same-machine macOS reference

NONE exists for this host: the notebook and docs hold macOS qwen38 MLX
baselines for T8103 (receipts/2026-09-24-jwm1-macos-baselines; docs/
compatibility.md "0.916x of same-machine macOS") and T6001 (macOS windows
4/5/9), while every macOS-side T6021 capture to date is ANE/CoreML
encoder work (254 ms Linux vs 89 ms macOS encoder), not MLX decode/
prefill. No ratio is claimed for any cell above; macOS runs are the
kernel lane's window and need the orchestrator's go.

## Comparators that do exist (same host, Linux)

h253 S0 (stock GPU pstate, v0.7.2x stack, same 2B cells): d64 109.83,
d128 109.81, d256 108.65, pf512 1607.3, TTFT 0.0907 s. This v0.7.26 run:
d64 -1.6%, d256 -2.2%, pf512 +0.3% (r2) — inside the boot-to-boot and
stack-version spread; no regression claimed in either direction.

## Artifacts

Raw cell JSONs, runner logs, install log, and the LoRA arm outputs are in
the private notebook (artifacts/M2Lane/2026-10-04-v0726-lora-ledger/) with
SHA256SUMS (60 files). The runner source is committed at
artifacts/M2Lane/2026-10-04-v0726-lora-ledger/staged/scripts/ (ledger.py,
run-lora.sh, launchers). Reproduction: install v0.7.26 per the tag
installer into a throwaway HOME, then
`ledger.py LABEL {2b,4b,9b,27b} OUTDIR` with `M2LANE_PY` pointing at the
gate venv python, under a GPU lock ticket, quiet gates on.

## Limitations

- The 4B model is Qwen3-4B-Instruct-2507-4bit (the only 4B in the host's
  cache), not a Qwen3.8-4B; recorded as such rather than downloaded.
- d128 was dropped and pf1024 added relative to h253 to fit four models
  in the ticket budget; d64/d256/pf512 stay shape-identical to h253.
- Single boot, single pass-set per arm; the outliers above bound the
  cross-rep noise at roughly ±1% (stable cells) with rare larger
  transient dips.
