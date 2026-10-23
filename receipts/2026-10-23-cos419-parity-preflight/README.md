# COS-419: parity measurements and execution gate

## Result

COS-419 is blocked at the hardware preflight. No speedup or new performance measurement is claimed. The assigned environment is Linux x86_64 on an AMD EPYC 7443P; `/dev/dri`, `/dev/nvidia0` and `/usr/lib/omarchy-mlx` are absent. The required Apple Silicon GPU model run cannot execute here. CPU or software-Vulkan timings would measure a different execution path.

| Requested measurement | Result |
|---|---|
| Before throughput | unmeasured |
| After throughput | unmeasured |
| Before / after real model runs | 0 / 0 |
| Model, settings and exact measurement command | unset; no model process launched |
| Source base | `48908ca1825d5c80feb5edaa1eb04c5c410fe9c0` |

## Largest retained timing gap

The largest ratio among the paired MLX timing JSON files in `receipts/2026-10-09-linux-vs-macos/{linux,macos}/` is **gpt-oss-20b prefill: Linux 1.862 versus macOS 581.120 tok/s**, or 312.095 times as long per token on Linux. Both files identify model `f356f2747216d7e98fee755df25987459fc19089`, the same corpus hash and a 512-token prefill. Linux has two prefill samples and macOS five. Decode in these same files is 2.177 versus 79.234 tok/s, with three versus ten samples.

Sources: [Linux JSON](../2026-10-09-linux-vs-macos/linux/gpt-oss-20b.json), [macOS JSON](../2026-10-09-linux-vs-macos/macos/gpt-oss-20b.json), and [benchmark definitions](../../docs/BENCHMARKS.md). These are historical measurements, not this task's before/after runs. The decode digests differ. The benchmark document therefore prints no macOS share for this model. The older `table-full.md` reports an earlier failed Linux run and is not the current timing source.

Among the MLX rows with matching output digests, the largest published deficit is **Qwen3.5-9B decode on the M1 Max: 37.052 versus 59.378 tok/s**, Linux at 62.400% of macOS. Its prefill is 299.356 versus 339.021 tok/s. Sources: [Linux JSON](../2026-10-09-linux-vs-macos/linux/qwen35-9b.json) and [macOS JSON](../2026-10-09-linux-vs-macos/macos/qwen35-9b.json). This output-matched comparison must not replace the much larger raw GPT-OSS timing deficit without saying why.

## PR 269 baseline

[PR 269](https://github.com/joshuaswarren/omarchy-mlx/pull/269) is merged at the source base above. Its diff and comments were read. It enables the existing chunked ArgMax/ArgMin and LogSumExp vocabulary-tail routes by default on G14C (M2 Max), preserving explicit environment overrides and other parts' defaults.

Its [measurement record](../2026-10-23-vocab-tail-g14c/README.md) reports Qwen3.5-9B decode gains of 3.16–3.85% on the M2 Max. Those numbers belong to PR 269, not COS-419. They do not establish a GPT-OSS prefill gain or justify enabling the default on an unmeasured part. This change leaves backend code and kernel defaults unchanged.

## Local safety preflight

Read-only checks: `uname -a`, `lscpu`, `ps -eo pid,ppid,pcpu,pmem,etime,args --sort=-pcpu`, `top -b -n 1`, device-path existence checks, `cat /proc/loadavg`, `herdr agent list`, and read-only output inspection of a neighboring MLX pane. No input was sent to another pane. The supported `top` invocation succeeded after an initial invocation rejected `-w`.

At 2026-10-23T13:35:10Z, load averages were 51.236 / 53.159 / 50.746. No GPU device is exposed; GPU utilization is unavailable, not zero. The local kernel reports `6.17.2-1-pve`. No remote hardware was accessed. No session or service was changed, no model downloaded and no other worktree edited. The existing upstream-watch timer is active.

## Resumption requirement

The same-machine before/after protocol requires an Apple Silicon Linux execution environment with the target GPU exposed, a pinned model and runtime, and an exclusive benchmark window. Select and preregister the single change there, then execute exactly one real before and one real after run with identical command/model/settings and verified build provenance. Leave the existing defaults intact until the numerical and performance gates pass.
