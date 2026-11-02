# Numerics gate (docs/numerics-gate.md) for #317: INCOMPLETE: gate not yet evaluated

This directory is a checkpoint of a run in progress. Nothing here is a gate result until every check has run; `RESUME.md` lists what is done and the exact commands for the rest.

## Protocol (fixed before the run)

- Arms: `main` `a69d9e6` (baseline, the deployed route) against this change's code commit `91d9f8a`, each a stamped release wheel built on the host from that commit (`wheels.txt`, `commits.txt`); every GPU process printed the `scripts/mlx_provenance.py` line and refused unless its stamp matched (`scripts/prov.py`). Five runs per arm, interleaved by `scripts/ab.py`.
- Reference: stock upstream MLX 0.32.3 (`mlx` and `mlx-cpu` wheels from PyPI, the `mlx.lock` version) with stock mlx-lm 0.31.3, in a venv used only for the reference, CPU stream, every floating parameter in float32 (the CPU quantized matmul dequantizes in fp32), 64-token passes. Each wheel's sha256 matched the hash PyPI publishes before install, and every installed file matched its distribution's RECORD (`ref_setup.log`). It shares no GPU code with the omarchy wheel.
- Models: the route models already on the host, not the parity-contract model (`SiddhJagani/Qwen3.8-2B-mlx-4Bit`, which is not on the host and was not downloaded). The gate document names no model; these are the models whose decode or prefill goes through the changed route. Hashes of every model file are in `model-sha256.txt`.
- Text: the GPL-3 licence text kept in the tree (`receipts/2026-10-21-gdn-bf16a-g14c-ratio1/numerics/reference-gate/tfref-text.txt`, sha256 `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986`), tokenized by each model's tokenizer, ten non-overlapping 513-token segments, BOS first where the tokenizer has one; 512 scored positions per segment, 5,120 per model (`teacher-*.json`), a fresh cache per segment.
- Pass shape: 128-token passes, because the dense bf16 linear route only runs on multi-row passes; an S=1 pass never reaches it.
- Harness check (before any arm): S=1 teacher-forced PPL of the baseline wheel against one-pass prefill PPL on segments 0 and 1; pass when the relative difference is at most 1e-2 (`validate-*.json`).
- Check 1 (per-op): the changed ops on captured operands (residual stream, attention output and MLP input recorded by the reference forward at the first, middle and last attention layer, two text positions blocks, stored in the model's dtype) on both wheels, relative L2 error of each output against an fp64 numpy forward of the same operands and dequantized weights; the candidate passes a cell when its mean error is no worse than the baseline's (`perop/`, `scripts/perop.py`).
- Check 2 (token choices): teacher-forced top-1 agreement with the reference; a flip whose reference top-2 margin is at most 2 bf16 ULP at the reference top logit is a near tie (`tools/dfuse/bf16_ulp.py`, inclusive); median candidate agreement with near ties must be at least the lower bound of the two-sided 95% Wilson interval on the median baseline agreement. Raw counts are kept.
- Check 3 (perplexity): deviation `abs(log(PPL / PPL_ref))`; noise = max minus min over the five baseline runs; median candidate deviation must be at most median baseline deviation plus noise.
- Old against new agreement is reported for information only (`compare_tf9-*.json`).
- One harness change during the run: the first harness check stopped because the M4 driver rejects the `mx.topk` partition pipeline at vocabulary width (`validate-attempt1-topk-rejected.jsonl`, `opprobe.out`: `topk` fails, `max`, `argmax`, `logsumexp`, `take_along_axis` run). The harness now takes the top two as the maximum, its index, and the maximum with that index masked out: the same two values, a tie giving a gap of 0. Every arm and the reference use that same code; nothing had been measured before the change.

## Status

| model | harness check | baseline runs | candidate runs | reference | checks 2, 3 | check 1 |
|---|---|---:|---:|---|---|---|
| Llama-3.2-1B-Instruct-bf16 | pass | 5/5 | 5/5 | 4/10 segments | not run | not run |

Harness check Llama-3.2-1B-Instruct-bf16: S=1 PPL 7.53524, one-pass PPL 7.54124, relative difference 7.95e-04 (pass).

## Files

- `scripts/`: the harness (`gate_tf.py`), per-op check (`perop.py`), analyzer (`analyze_gate.py`), the phase script as run (`run_gate.sh`; `run_gate.v1.sh` ran the facts, harness-check and first reference phases), the runners, `ref_setup.sh`, `setup.sh` (host rebuild of the wheels) and `opprobe.py`. `ab.py`, `once.py` and `prov.py` are in the receipt's `scripts/`.
- Records: `runs/<pr>-<model>/*.json.gz` (one per GPU run, compare_tf9 format), `ref-<model>.json.gz`, `ab-*.jsonl` (each process: stamp, environment, load, temperatures, provenance line), `validate*.json*`, `teacher-*.json`, `gate-*.txt|json`, `perop/`.
- Host: `host.txt`, `wheels.txt`, `commits.txt`, `model-sha256.txt`.
