# qwen3 dense rope-norm fold on the 13-inch M1: decode +1.09%, prefill +0.59% (8 paired A/B)

`scripts/patch-mlx-lm-qwen3-rope-norm.py` folds the per-head q/k RMSNorm into `mx.fast.rope_rms_norm` at the qwen3 dense attention site. It is on by default; `MLX_OMARCHY_ROPE_NORM_FUSE=0` turns it off. This receipt measures the fold on a MacBook Pro 13-inch (M1, 16 GB).

**Scope: Qwen3-4B-Instruct-2507 4-bit, a 512-token prompt, 128 decode tokens. Other models, prompt sizes and chips are not measured here.**

## Result

Eight paired runs, 3 rounds per arm. A pair value is median(fold ON) / median(fold OFF) over its 3 rounds. Ratios are computed by `analyze.py` from `pairs.tsv` (tok/s rounded to 0.01 in the table).

| pair | order | decode ON/OFF |
|---|---|---|
| 1 | ON first | 1.0063 |
| 2 | ON first | 1.0102 |
| 3 | ON first | 1.0126 |
| 4 | ON first | 1.0121 |
| 5 | OFF first | 1.0116 |
| 6 | OFF first | 1.0111 |
| 7 | OFF first | 1.0102 |
| 8 | OFF first | 1.0106 |

| | decode | prefill (512) |
|---|---|---|
| pooled median of the 8 pair ratios | 1.0109 | 1.0059 |
| range | 1.0063 to 1.0126 | 1.0048 to 1.0072 |
| median, ON-first pairs / OFF-first pairs | 1.0111 / 1.0109 | 1.0061 / 1.0056 |

Mean decode over all rounds: 20.83 tok/s ON, 20.65 tok/s OFF. Prefill is about 173 tok/s on both arms. Every pair favors the fold. Both order groups have a median ratio above 1 (1.0111 ON-first, 1.0109 OFF-first).

Prefill and decode of one run come from the same pass: prefill rate is 512 over the time to the first token, decode rate is 127 tokens over the time from the first to the last token, at depth 512.

## Output is identical

Greedy generation of the 128 tokens from the fixed 512-token prompt gives the same token digest with the fold ON and OFF (`CAPTURE_TOKENS_SHA256 fc31fd0e7a886c3608be5d351fdb2159e0581539b447cc581ae056a9ce62408e`; prompt digest `bf856d91…`). The validation output of the two arms hashes identically (`ebaa4b508feea75e…`). Each arm was captured once, on boot A before pair 1, and not repeated on boot B; `validation.txt` holds both captures.

## Procedure

- ON arm: `MLX_OMARCHY_ROPE_NORM_FUSE=1`. OFF arm: `MLX_OMARCHY_ROPE_NORM_FUSE=0`. Same venv, same model files, same seeded prompt ids.
- One harness invocation per arm: `t7b-mlx-omarchy.py --sizes 512 --rounds 3 --gen 128 --prefill-step 512 --pause 30`. The harness is in this directory (sha256 `5fe9de4903f8e1586060d4bbdfa7ca4d5e115f17a473f4bb03fe4bda080bae32`).
- Pairs 1 to 4 ran the ON arm first, pairs 5 to 8 the OFF arm first: two blocked order groups, not a randomized order design. `runs.tsv` records the start time of each arm.
- Before each run the harness waits until the load average is under 1.0 (its default gate), with a 30 s pause between runs. The `load_avg_after_run` column of `pairs.tsv` is the load average read after that run's generation loop (highest 0.34); the harness does not record the value that passed the gate. Peak memory was 2.34 GiB in every run.
- Preflight before every arm, which a run must pass to start: the interpreter, the `mlx.core` extension (`7542a0bc…`) and `libmlx.so` (`6236fae2…`) are hashed and compared with pinned values, together with the hostname, kernel release and boot id. All 16 retained arms carry the same interpreter, extension and `libmlx.so` pins, on both boots.
- Raw per-round values: `pairs.tsv`. The 16 raw result files carry identical environment records (package version, Honeykrisp and Mesa versions, kernel release, model and weight hashes) apart from timestamps. Run date 2026-11-06.

Run notes, pair 8 (the other seven pairs ran once each, in order, on boot A):

- First attempt, boot A: the OFF arm completed at 05:29 to 05:31 UTC (prefill 172.2 / 172.1 / 171.9, decode 20.66 / 20.65 / 20.64 tok/s). The run watchdog ended the ON arm after 2 of its 3 rounds at 05:32 UTC (prefill 172.9 / 174.4, decode 20.84 / 20.88). The incomplete pair was set aside, not pooled, and re-taken. On those values the decode ratio is 1.0102.
- Second attempt, boot A: the OFF arm completed at 06:02 to 06:04 UTC and is the retained OFF arm. The ON arm started at 06:04 UTC; 8 s in, the kernel journal logged a GPU timeout and no data was produced. Boot A ended at 06:12 UTC. The host stayed at the bootloader's kernel-changed prompt until a person confirmed it; boot B began at 11:57 UTC. The cause of the timeout is not established.
- ON arm repeat, boot B, 12:41 to 12:43 UTC, retained. The kernel release string is unchanged, and the preflight check above passed on boot B against the same pins; the kernel image file on disk was replaced between the boots and was not hashed. Pair 8's two arms therefore ran 6.6 hours apart, across a reboot.

Pair 8's ratio (1.0106) sits inside the range of the other seven. Pairs 1 to 7, all on boot A, have a median decode ratio of 1.0111. Start and finish times, boot and the raw result file hash of every retained arm are in `runs.tsv`.

## Provenance

```
mlx-omarchy 0.32.4.dev202611040253+a39f0251 (mx.__version__ equal), mlx_lm 0.31.3
libmlx.so sha256 6236fae222c6c966a0b096638f650671323848d9aba1f3d3fb0c595fdabae77b  RECORD match
mlx.core extension sha256 7542a0bcff502c00f2c48a2395367e9d49e3405b13f59b17f54abb78b9a0444d  RECORD match
(scripts/mlx_provenance.py run on boot B after pair 8: verified "match")
Apple M1 (G13G B1), 16 GB, Honeykrisp (Vulkan api 1.4.362), Mesa 26.3.0-devel (git e7631595df)
kernel release 7.1.12-2-20261013.18-sep-ARCH on both boots
model Qwen3-4B-Instruct-2507-4bit, snapshot 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b,
  model.safetensors sha256 2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f
```

## Reproduce

```
python3 analyze.py            # prints the table above from pairs.tsv
env -u MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE MLX_OMARCHY_ROPE_NORM_FUSE=1 python t7b-mlx-omarchy.py --model <Qwen3-4B-Instruct-2507-4bit dir> \
    --host-label "<machine label>" --out on.json --sizes 512 --rounds 3 --gen 128 --prefill-step 512 --pause 30
env -u MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE MLX_OMARCHY_ROPE_NORM_FUSE=0 python t7b-mlx-omarchy.py ... --out off.json   # same arguments
```

Run the harness with the interpreter of the packaged install, on an idle machine. The specific override `MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE` takes precedence over the general toggle, so the run commands clear it; the preflight chains of the timed runs pinned the patched `qwen3.py` and cleared it the same way.

## Relation to the release figure

The v0.7.25 release receipt (`receipts/2026-10-04-v0725-release.md`) reports +4.18 to +4.86% decode for the same fold on another machine, at decode depths 64 to 512. That procedure and chip differ from this one, so the two figures are not compared.
