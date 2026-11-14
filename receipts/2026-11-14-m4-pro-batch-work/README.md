# M4 Pro (G16S) submission budget: 5000 to 15000

MacBook Pro 16-inch, M4 Pro (J616s, T6040), 48 GB, Linux 7.3.0 aurora, mlx-lm 0.31.3, one boot for
every run. Mesa aurora-wip with aurora-silicon/mesa#319 (55c06da6), wheel of this branch at fed73df.
`Apple M4 Pro (G16S B1)` matched no row of `kBatchWorkCalibration`, so it took the 5000 fallback.

## Verdict

`kBatchWorkCalibration` gets the row `{"G16S", 15000}`. Decode medians (tok/s) of three interleaved
rounds, one fresh process per run, 128 greedy tokens (granite stops at 53):

| Model | 5000 (fallback) | 10000 | 15000 | 20000 | 40000 | off | 15000 against 5000 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct 4-bit | 138.08 | 143.45 | 154.88 | 157.02 | | | +12.2% |
| Qwen3.5-4B 4-bit | 64.86 | 66.17 | 66.70 | 66.65 | 66.63 | 66.71 | +2.8% |
| Qwen3-8B 4-bit | 41.98 | 42.46 | 42.66 | 42.54 | 42.62 | 42.25 | +1.6% |
| granite-4.0-1b 4-bit | 119.34 | 120.55 | 119.78 | 118.06 | 110.39 | 110.53 | +0.4% |

5000/10000/15000/20000 from `ab-batchwork2.txt`; 40000 and off (`MLX_OMARCHY_BATCH_WORK=0`) from
`ab-batchwork.txt`, whose 5000 and 20000 arms agree with the second file within 0.5% (the largest gap is 4B/20000, 66.35 versus 66.65, 0.452%). Every run of a
model produced the same text (`text_sha` column), at every budget.

Above 15000 granite loses: -1.4% at 20000 and -7.8% at 40000 against 15000. Its decode step is short
enough that larger batches stop overlapping host encoding with the GPU. 15000 is the largest budget
that does not cost any of the four models. Prefill (pf300, pf2k) at 15000 is within the spread of
5000; with the cap off, Qwen3.5-4B pf300 drops from 491 to 432 tok/s.

## Submission length

`subcap.txt`: `scripts/subcap_calibrate.py`, Qwen3.5-4B, 512-token prompt, 64 decode tokens, run on the
profiling wheel (about 100 us added per dispatch, 10.6 tok/s instead of 65), so its absolute times
are inflated and only the counts carry over. 602 submissions at 5000, 159 at 20000, 84 at 40000:
9.4, 2.5 and 1.3 per token. At the real 4B rate (about 15 ms per token) that estimates about 1.6 ms per
submission at 5000 and 6 ms at 20000. The 15000 length is not directly measured: about 4.5 ms is an
interpolation, not a latency percentile or proof of the 2-6 ms band.

`probe.txt`: `scripts/submit-latency.c` at 60 Hz beside decode. 0 frames over 33.3 ms at every
measured budget (5000, 20000, 40000 and off), p99 0.19-0.21 ms, the same as idle. There is no
15000 arm in this probe: any 15000 frame-latency claim is an estimate bracketed by those arms,
not a direct measurement.

## Build provenance

`provenance.txt` records the wheel and private Mesa prefix inspected after these runs, before either
was replaced. The installed core and libmlx bytes match the archived wheel, and the prefix library
carries Mesa commit 55c06da699 in its version string. The ICD path in every raw A/B header points to
that prefix. This is a post-run binding: the raw benchmark runner did not hash the loaded objects
per process, so it is not per-process loaded-object attestation. No binary has been rebuilt or
reinstalled in either location between the recorded runs and this inspection.
