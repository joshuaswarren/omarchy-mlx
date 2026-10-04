# 2026-10-04 — zero-CPU gdb trace on the published v0.7.24 stack (release-gate close)

Gate: trace every ecosystem workflow at least once and require zero CPU
tensor dispatches (AGENTS.md). This closes the gate that had never run to
completion: a gdb trace on the shipped stack proving **zero** CPU tensor
dispatches while real workloads run, with a positive control proving the
trace can see CPU dispatches. Parakeet is the ANE/other lane and is out of
scope.

## Stack under test (provenance)

The PUBLISHED v0.7.24 release (tag commit `eaa697236519c56c…`), installed by
the tag's own `install.sh` into a throwaway HOME on the M2 gate host
(Apple M2 Max, T6021, kernel 7.1.13-3-1-ARCH, gdb 17.2, python3.14.7):

| file | sha256 (head) | match vs wheel RECORD |
|---|---|---|
| `mlx/lib/libmlx.so` | `801c989b…` | yes |
| `mlx/core.cpython-314-aarch64-…so` | `133e15fc…` | yes |
| compiled `mx.__version__` | `0.32.4.dev202610032142+eaa69723` | equals dist version |

`scripts/mlx_provenance.py` verdict: `verified=match`. The mlx-lm serve
patch series (gated-delta fast route x3, raw, greedy-prune, qwen35
qk-scaled, gdn-conv, conv-silu, gated-norm, ttft-early-submit,
last-logits; conv-ring OFF) was applied by the installer; voice floor
`mlx-audio==0.5.6`, `misaki==0.7.4`, `phonemizer>=3.2.1`. Release assets
verified against the release `SHA256SUMS` before install (4/4 OK).

## Trace target and harness

Breakpoint symbol `_ZN3mlx4core3cpu19get_command_encoderENS0_6StreamE`
(`mlx::core::cpu::get_command_encoder(Stream)`) — the exported dispatch
entry every CPU-stream primitive's eval path takes. The release `libmlx.so`
exports it by construction, so the trace must resolve and every hit is
observable. Harness: `receipts/2026-09-30-speech-input-gpu/harness/
count_cpu.gdb.py` (silent Python breakpoint, one `CPU_COUNT` JSON line);
spawned workers traced by attaching gdb to the live worker PID with
textual silent breakpoints (`Breakpoint 1 at …` recorded per arm). All
arms on one boot (`a51618f7…`, load 0.06–0.3), one gpu-turn ticket at a
time.

## Arms and results

| arm | workload (as served) | CPU encoder hits | trace resolution | verdict |
|---|---|---|---|---|
| control-cpu | `mx.add(a, a, stream=mx.cpu)`, 4096 elements | **2** | resolved, libmlx loaded | positive control >= 1 PASS |
| control-gpu | same op on the default stream | **0** | resolved, ran on `Device(gpu, 0)` | PASS |
| kokoro streamer | shipped streamed decoder (default), pack `kokoro-82m-bf16`, voice `af_heart`, TWO sentences in-process (4.000 s + 5.200 s audio, wall 8.755 s, RTF 0.952) | **0** | resolved, both sentences `Device(gpu, 0)` | PASS |
| GDN doctest | `omarchy_gdn_maskless_correctness_tests` built from the tag tree (HEAD `eaa697236…`, binary sha256 `7c6911d1…`): 2/2 cases, 152/152 assertions, exit 0 | **0** | resolved at load (`Breakpoint 1 at 0x5ec310`, static libmlx) | PASS |
| 9B chat | pair `everyday`, worker serving `Qwen3.5-9B-MLX-4bit`, one turn `status=complete 3.15s max_tokens=32 text='Ready'` under the attached trace | **0** | resolved (`Breakpoint 1 at 0xfffecd3ea590`) | PASS |
| 27B chat | pair `quality`, worker serving `Qwen3.8-27B-4bit`, first tokens: `status=complete 3.12s max_tokens=4 text='ready'` under the attached trace | **0** | resolved (`Breakpoint 1 at 0xffffae44a590`) | PASS |

Every GPU arm: 0 hits. Positive control: 2 hits on the same harness, same
venv, same boot. The gate's pass condition (every named workflow arm 0
with a positive control >= 1) is met.

The 2-hit control count (the 2026-09-30 battery recorded 3 on an older
core) is a property of the v0.7.24 core build, not a harness change — same
op, shape, and buffer.

## Incident owned during the run (upstream-mlx contamination)

A voice-dependency install copied from `g8-kokoro.sh` (`pip install
mlx-audio==0.5.6 …` without `--no-deps`) resolved mlx-audio's `mlx>=0.31.1`
requirement and pulled **upstream `mlx` 0.32.3** over the vendored build —
the exact trap `install.sh --voice` warns about. Detected by
`scripts/mlx_provenance.py` (loaded extension hash `3a4d2137…` vs wheel
RECORD `133e15fc…`; +88 bytes, different symbol table), before any arm
counted. Repair: removed the conflicting upstream package, force-reinstalled
the published wheel, re-added `mlx-audio` with `--no-deps` per install.sh's
own recipe, and re-verified `verified=match`. Every number above comes from
the post-repair venv. The early control run against the contaminated
environment (2 hits of the UPSTREAM core) is voided evidence.

Product/tooling note: `scripts/release-gates/g8-kokoro.sh` still installs
mlx-audio without `--no-deps` and can reproduce this contamination from a
voice-less venv; `install.sh --voice` has the correct recipe.

## Method notes and limitations

- The trace covers the process in which the workload's tensor dispatches
  run: the Kokoro arm is a single in-process render; the GDN arm is the
  doctest binary itself; the chat arms trace the `_mlxlm_server` shim that
  owns the model (worker cmdline captured per arm). The chat coordinator
  and Laya decision worker were not separately traced (not among the
  gate's named workflows).
- The chat arms required `kernel.yama.ptrace_scope=0` on the gate host for
  the attach window (reboots had reset it to 1); restored to 1 at close.
- Two owned failures preceded the clean 9B arm: a ptrace denial (yama=1)
  and a setup timeout (the first arm's ticket still held the GPU flock —
  FCFS queue). Both cleaned up by exact PID; queue verified empty before
  the rerun.
- Command lines, per-arm gdb logs, worker cmdlines, provenance JSON, and
  asset verification live in the private lab notebook
  (`artifacts/ZeroCpu24/2026-10-04-zero-cpu-v0724/`, SHA256SUMS +
  COMPLETE per arm).

## Verdict

The v0.7.24 release stack runs the Kokoro streamer, the GDN doctest, 9B
chat, and 27B chat entirely on GPU streams: zero CPU tensor dispatches
observed at the CPU command-encoder entry, with a live positive control.
`docs/compatibility.md` records the row.
