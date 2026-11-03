# Numerics gate (docs/numerics-gate.md) for #321: evaluated, every check ran

## Protocol (fixed before the run)

- Arms: `main` `a69d9e6` (baseline, the deployed route) against this change's code commit `89b54a0`, each a stamped release wheel built on the host from that commit (`wheels.txt`, `commits.txt`); every GPU process printed the `scripts/mlx_provenance.py` line and refused unless its stamp matched (`scripts/prov.py`). Five runs per arm, interleaved by `scripts/ab.py`.
- Reference: stock upstream MLX 0.32.3 (`mlx` and `mlx-cpu` wheels from PyPI, the `mlx.lock` version) with stock mlx-lm 0.31.3, in a venv used only for the reference, CPU stream, every floating parameter in float32 (the CPU quantized matmul dequantizes in fp32), 64-token passes. Each wheel's sha256 matched the hash PyPI publishes before install, and every installed file matched its distribution's RECORD (`ref_setup.log`). It shares no GPU code with the omarchy wheel.
- Models: the route models already on the host, not the parity-contract model (`SiddhJagani/Qwen3.8-2B-mlx-4Bit`, which is not on the host and was not downloaded). The gate document names no model; these are the models whose decode or prefill goes through the changed route. Hashes of every model file are in `model-sha256.txt`.
- Text: the GPL-3 licence text kept in the tree (`receipts/2026-10-21-gdn-bf16a-g14c-ratio1/numerics/reference-gate/tfref-text.txt`, sha256 `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986`), tokenized by each model's tokenizer, ten non-overlapping 513-token segments, BOS first where the tokenizer has one; 512 scored positions per segment, 5,120 per model (`teacher-*.json`), a fresh cache per segment.
- Pass shape: 64-token passes with HK_APPLE9_COOPMAT=0 on both arms, because the route runs on multi-row bf16 Q4 passes where the cooperative matrix is unavailable.
- Harness check (before any arm): S=1 teacher-forced PPL of the baseline wheel against one-pass prefill PPL on segments 0 and 1; pass when the relative difference is at most 1e-2 (`validate-*.json`).
- Check 1 (per-op): the changed ops on captured operands (residual stream, attention output and MLP input recorded by the reference forward at the first, middle and last attention layer, two text positions blocks, stored in the model's dtype) on both wheels, relative L2 error of each output against an fp64 numpy forward of the same operands and dequantized weights; the candidate passes a cell when its mean error is no worse than the baseline's (`perop/`, `scripts/perop.py`).
- Check 2 (token choices): teacher-forced top-1 agreement with the reference; a flip whose reference top-2 margin is at most 2 bf16 ULP at the reference top logit is a near tie (`tools/dfuse/bf16_ulp.py`, inclusive); median candidate agreement with near ties must be at least the lower bound of the two-sided 95% Wilson interval on the median baseline agreement. Raw counts are kept.
- Check 3 (perplexity): deviation `abs(log(PPL / PPL_ref))`; noise = max minus min over the five baseline runs; median candidate deviation must be at most median baseline deviation plus noise.
- Old against new agreement is reported for information only (`compare_tf9-*.json`).
- One harness change during the run: the first harness check stopped because the M4 driver rejects the `mx.topk` partition pipeline at vocabulary width (`validate-attempt1-topk-rejected.jsonl`, `opprobe.out`: `topk` fails, `max`, `argmax`, `logsumexp`, `take_along_axis` run). The harness now takes the top two as the maximum, its index, and the maximum with that index masked out: the same two values, a tie giving a gap of 0. Every arm and the reference use that same code; nothing had been measured before the change.

## Status

| model | harness check | baseline runs | candidate runs | reference | checks 2, 3 | check 1 |
|---|---|---:|---:|---|---|---|
| granite-4.0-1b-4bit | pass | 5/5 | 5/5 | done | done | done |

Harness check granite-4.0-1b-4bit: S=1 PPL 6.54881, one-pass PPL 6.56134, relative difference 1.91e-03 (pass).

## Result

**Gate: passed.** Every check passes on every model.

### granite-4.0-1b-4bit

| arm | runs | top-1 vs reference (median) | flips | near ties | PPL (median) | mean abs dlogprob |
|---|---:|---:|---:|---:|---:|---:|
| baseline `a69d9e6` | 5 | 97.266% | 140 | 122 | 7.58016 | 0.07272 |
| candidate `89b54a0` | 5 | 97.266% | 140 | 122 | 7.58016 | 0.07272 |
| reference (stock MLX CPU fp32) | 1 | | | | 7.58552 | |

Every run of an arm gave the same values (deterministic), so the five-run PPL noise is 0 and check 3 allows no difference from the baseline's deviation.
- Check 2: candidate agreement with near ties 99.648% against the baseline Wilson lower bound 99.445% (baseline median 99.648%): **pass**.
- Check 3: candidate PPL deviation 0.000707 against baseline 0.000707 + noise 0.000000: **pass**.
- Old against new (information): 5120/5120 top-1 the same (100.00%).
- Check 1: candidate no worse in 21 of 21 cells (worst candidate/baseline ratio 1.000000): **pass** (`perop/score-321-granite-4.0-1b-4bit.txt`).

## Deviations from docs/numerics-gate.md, and notes

- Pass shape and environment (a deviation from the document's wording). `docs/numerics-gate.md` check 3 names route-sensitive S=1 perplexity. This change's route runs only on multi-row bf16 4-bit passes where the cooperative-matrix qmm route is unavailable. An S=1 pass takes the decode GEMV instead, and on this host's driver the cooperative route is available by default, so neither would reach the change. Both arms therefore ran teacher-forced 64-token passes (each 512-token segment as eight passes sharing one cache) with `HK_APPLE9_COOPMAT=0`, which hides the driver's cooperative matrix. The baseline arm is the tile route `QmmTileBF16`, the original arm this change replaces. The harness check is the documented one on the baseline wheel, run with the matrix both exposed and hidden (`validate-*.json`). The reference ran in 64-token passes on the CPU in float32; it is the same reference used for every PR in this series.
- The arms are bit-identical. All ten runs give the same values at all 5,120 positions (PPL 7.58016; reference 7.58552), and both wheels give identical outputs in every per-op cell. The gate passes because the candidate's error equals the baseline's, not because it is smaller. This matches the receipt's `qmm_fma_bits.py` result (21/21 bit-identical to the tile route on this host).
- The changed route ran. The per-op cells of check 1, repeated on profiling builds of both commits with the same environment, dispatched `QmmTileBF16` on `main` and `QmmPrefillFmaBF16` on this change, 42 times each (`perop/321-*-dmain.kernels.txt`, `perop/321-*-dp6.kernels.txt`; `scripts/run_gatek.sh`). Each profiling build's 84 outputs equal its release build's bit for bit (`perop/321-*-release-vs-profiling.txt`). This is evidence for the M4 with this compiler only. On stock-Mesa M1/M2 this becomes the default bf16 Q4 prefill route, and whether it keeps the tile kernel's bits there is not measured here.
- Nothing else departs from the protocol fixed before the run: five runs per arm, all with exit status 0 and no new GPU driver error, no run repeated or dropped, and the thresholds as documented. The main-arm runs took about 250 s each against 40 s for this change (the tile kernel is slow); that is the run time of the harness, not a speed measurement.

## Files

- `scripts/`: the harness (`gate_tf.py`), per-op check (`perop.py`), analyzer (`analyze_gate.py`), the phase script as run (`run_gate.sh`; `run_gate.v1.sh` ran the facts, harness-check and first reference phases), the runners, `ref_setup.sh`, `setup.sh` (host rebuild of the wheels) and `opprobe.py`. `ab.py`, `once.py` and `prov.py` are in the receipt's `scripts/`.
- Records: `runs/<pr>-<model>/*.json.gz` (one per GPU run, compare_tf9 format), `ref-<model>.json.gz`, `ab-*.jsonl` (each process: stamp, environment, load, temperatures, provenance line), `validate*.json*`, `teacher-*.json`, `gate-*.txt|json`, `perop/`.
- Host: `host.txt`, `wheels.txt`, `commits.txt` (the first round's builds), `wheels-all.txt`, `commits-all.txt` (every build used on 2026-11-02, including the profiling builds `dp2` and `dp6` for the per-op kernel lists and the narrowed #316 builds `p1n`/`dp1n`), `model-sha256.txt`.
