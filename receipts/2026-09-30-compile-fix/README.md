# mx.compile: Snake "hang" and "eval an array without a primitive" (2026-09-30)

Two failures of `mx.compile` were reported on the omarchy Vulkan wheel while
the Kokoro engine was evaluated (receipts/2026-09-30-speech-output-kokoro).
One did not reproduce. The other was a real tape-interpreter defect that is
now fixed in `ec94a4bb5`.

Host: Apple M2 Max (T6021) test host `<host>`, Linux 7.1.13-ARCH-polltx,
mesa / vulkan-asahi 26.2.3-1, Honeykrisp. Every GPU command ran through the
host's FIFO GPU queue with a `timeout -k 10 90` guard. The host rebooted
several times during the session (a separate project's kernel-module work).
No number below comes from a run that a reboot cut. Every log carries its
provenance line.

| Wheel | Source | libmlx.so sha256 |
|---|---|---|
| live (pre-fix) | 0.32.3.dev202609282218+29cba8e | d931f8c5600bc6f0… |
| fix | 0.32.3.dev202609300006+ec94a4b | 543f453dd9e469a2… |

## (a) Pure Snake under mx.compile: does not hang

`repro_snake_compile.py` (19 lines) compiles `x + (1/a)*sin(a*x)**2` with
lazy inputs, including the exact reported line (`(1,8,256)` normal x,
`mx.ones` alpha). Live wheel: all five cases trace in 0.1-0.2 ms and
evaluate in 2.7-26.6 ms, with max abs error 0 against eager (f32 and bf16,
up to `(1,512,4096)`). The fix wheel gives the same result. The hang did not
reproduce. The run that reported it wrote no output file, and the host was
rebooting every 30-60 min in that window [INFERENCE: a reboot or queue wait
was read as a hang].

## (b) "Attempting to eval an array without a primitive": fixed

Repro: `repro_adain_snake_compile.py` (19 lines) is Kokoro's
AdaIN1d followed by Snake1D with a captured alpha (the `AdaINResBlock1`
body). On the live wheel it raises the named RuntimeError. On the fix wheel
it prints `ok max_abs_err_vs_eager 0.0`.

Narrowing on the live wheel (`probe_resblock_parts.py`): AdaIN alone,
Snake alone, and conv alone compile correctly. AdaIN followed by Snake
fails. With `MLX_OMARCHY_FUSED_CHAIN=0` every part passes. Kokoro blocks
(`probe_kokoro_blocks.py`, T=96): `generator.resblocks[0]` and
`noise_res[0]` fail on the live wheel and pass on the fix wheel with error
0. `decoder.encode/decode[]` (no Snake) pass on both.

Root cause (gdb `catch throw` on a local lavapipe build of the same
source): `Sin::eval_gpu` calls `trig_argument_gate`
(`overlay/mlx/backend/omarchy/primitives.cpp:4413-4439`). The gate builds
`max(abs(input))` and calls `settle()`, which is a nested `eval_impl`. Two
facts in the tape interpreter made that nested eval unsafe:

1. `FusedChain::evaluate` built the fused output with the tail's operands
   (`fused_chain.cpp:692-696` before the fix). For a chain extension, one of
   those operands is the previous member's tracing-graph array, which the
   interpreter passes through for the chain continuation
   (`compiled.cpp:320-322`). The nested eval walked from the fused output
   into the trace. It reached a tracer placeholder, which has no primitive
   and is unscheduled, and `transforms.cpp` threw.
2. Per-node tape outputs (`compiled.cpp:437-443` before the fix) stayed
   unscheduled with live inputs. The same gate therefore re-dispatched
   their whole upstream sub-tape. On lavapipe, the doctest graph recorded 28
   Vulkan dispatches compiled vs 24 eager with the chain gate off.

Fix (`ec94a4bb5`): a computed tape value now looks like an eager node after
the evaluator runs it. `FusedChain::evaluate` returns an evaluated,
graph-free array (no primitive, no inputs). `eval_compiled_tape` marks each
per-node output evaluated and detaches it after its dispatch. Buffer
lifetimes do not change: `add_temporary` holds data pointers, not graphs.
The unused `tail_primitive` and per-member `inputs` fields are removed.
After the fix, the lavapipe doctest graph records 20 dispatches (chain on)
and 21 (chain off). Eager records 24.

Doctest: `compiled AdaIN + Snake: a Sin after a fused chain matches eager
(no-primitive regression)` in `overlay/tests/omarchy/test_compiled_tape.cpp`.
It compares against eager at f32 1e-5 and bf16 bit-exact. On lavapipe it
THREW the named error on the pre-fix backend and passes after (1028/1028
assertions).

Battery slice on the M2, test binaries built from `ec94a4bb5`:

| Suite | Cases | Assertions |
|---|---|---|
| omarchy_compiled_tape_tests | 13/13 (new case included, 0 skipped) | 3124 |
| omarchy_fused_chain_tests | 36/36 | 346272 |
| omarchy_primitive_tests | 104/104 | 2743003 |
| omarchy_runtime_tests | 41/41 | 22694 |

All work runs on the default GPU device. A compiled tape runs inside
`gpu::eval`, and no CPU stream is created.

## Kokoro speed effect

`bench_kokoro_compile.py` ran on the fix wheel, in one boot, with no other
GPU job (load average 0.45-0.48). Sentence: "Your meeting starts at nine,
and the review follows at eleven." Voice af_heart, 3.975 s of audio, trig
reduction installed as the product does. Each variant ran 1 warmup and 3
measured runs. RTF = audio s / wall s around the pipeline call, with G2P
included, as in the Kokoro receipt. Each compiled variant was then checked
against eager with the same random seed.

| Variant (fix wheel) | RTF runs | Median RTF | GPU primitives | Vulkan dispatches | Seeded vs eager |
|---|---|---|---|---|---|
| eager | 0.746, 0.735, 0.735 | 0.735 | 21976 | 15189 | reference |
| Snake blocks compiled (`generator.resblocks`, `noise_res`) | 0.744, 0.743, 0.725 | 0.743 | 19672 | 14877 | same length, max abs diff 7.5e-8, SNR 139.8 dB |
| all blocks compiled (+ `decoder.encode`, `decode[]`) | 0.744, 0.742, 0.738 | 0.742 | 19335 | 14837 | same length, max abs diff 8.9e-8, SNR 139.8 dB |

Reference: eager on the live wheel (29cba8e) with the same script and boot
measured RTF 0.669-0.675 (median 0.671). The live wheel comes from an
older tree, so the difference between 0.671 and 0.735 is not the effect of
this fix.

Verdict: compilation now works on Kokoro, and it is numerically identical
to eager at 140 dB. It does not make Kokoro faster: the median RTF is
+1.1% (0.735 to 0.743), which is inside the run-to-run spread, and it
removes only 2% of the Vulkan dispatches. The default fused elementwise
chains already fold the Snake runs, so what remains is conv-bound, which
matches the Kokoro receipt's profile. Kokoro stays about 7x below its
RTF >= 5 threshold, and the path to speed is the conv kernels, not
`mx.compile`. Qwen3-TTS was not measured here.

## Files

- `repro_snake_compile.py`, `repro_adain_snake_compile.py`: the minimal repros.
- `probe_kokoro_blocks.py`, `probe_resblock_parts.py`: block and sub-block narrowing.
- `bench_kokoro_compile.py`: resumable eager vs compiled Kokoro bench.
- Raw logs with SHA256SUMS are in the private lab notebook (CompileFix run-001 to run-005).
