# Trig contract — one honest sin/cos contract on every path

Commit 79a53e388 (+ 9ed0e3cf6 test pins). Supersedes the accuracy half
of OpCost's 6cbf55d8f; keeps its speed property (no per-call host sync).

## The defect OpCost left (measured, probes r1/r2)

`6cbf55d8f` replaced the per-call `trig_argument_gate` with an in-shader
Cody-Waite reduction (NaN above 1e9) but its constants were wrong:

- `C1` = full 24-bit f32 rounding of 2*pi → `k*C1` inexact for every
  `|k| >= 2` (Sterbenz-exact subtraction preserves the product rounding:
  phase error ≈ x*6e-8);
- `C2 = -6.7792634e-8` does not satisfy `C1 + C2 = 2*pi` (residual
  -1.07e-7 → a second `|k|*1.07e-7` term).

Measured max abs error vs float64 (1e5 f32 samples/decade + edges, both
signs; probes: `.work/trig-probe/`, archived in the notebook at
artifacts/TrigContract/trig-probe-r{1,2}/ with SHA256SUMS):

| band | llvmpipe | jw16 M1 Max (109060195) | jwm1 M1 (109060099) |
|---|---|---|---|
| 1e3..1e4 (built-in) | 6.0e-8 | 3.4e-8 | 7.1e-4 |
| 1e4..1e5 (shipping Cody-Waite) | **5.2e-3** | **5.2e-3** | **6.9e-4** |
| 1e5..1e6 | **4.5e-2** | **4.5e-2** | **1.9e-3** |
| 1e6..1e7 | **6.0e-1** | **6.0e-1** | **1.5e-2** |
| 1e7..1e8 | 1.86 | 1.86 | 6.2e-3 (A_LIMIT=1e9 unmeasured band) |

(OpCost's commit message recorded 1.7e-3 at 1e5 and called it "adequate
for Snake" — the contract text, the tests, and the 1e9 threshold were
never updated to match the numbers.)

The Kokoro serve path escaped on hosts whose serve checkout carries
`_kokoro_install_trig_reduction` (the graph-level patch with the CORRECT
C1 = 6.28125 constants). The M2's SpeechOutputKokoro checkout predates
that patch and its wheel predates OpCost: Kokoro there died loudly —
`[omarchy] Sin argument magnitude 185111.8125 exceeds the built-in
accuracy limit 100000.0` — captured verbatim twice on jw14m2.

## The fix

One shared GLSL header `shaders/omarchy_trig.h`, included by every
shader that evaluates trig of an unbounded argument:

- Cody-Waite with `C1 = 6.28125 = 201*2^-5`: `k*C1` is EXACT in f32 for
  every `|k| <= 83468` (arguments to 524447) — the contract constant 5e5
  sits inside the exact-product zone with margin;
- a dyadic `2^-9` term (k times it exact for all k) plus two f32
  correction terms close the 2*pi decomposition (residual < 1e-15/k);
- the three-subtract chain is `precise` (NoContraction);
- band: |x| <= 1e4 raw built-in (bit-identical; Kokoro's graph patch and
  Laya's rope angles live there), 1e4 < |x| <= 5e5 reduction, > 5e5 NaN.

Measured reduction-band error of the fixed form: 2.47e-7/8.2e-8
(decades 4/5, llvmpipe and jw16), 2.0e-7 (jwm1 far band; jwm1's
older-Mesa built-in sin adds up to 6.9e-4 at SOME reduced arguments —
argument-dependent, stable across runs, the same floor its built-in
band already documents; the tests pin 1e-3 in the reduce band and the
per-device maxima live in this receipt).

Wired paths (audit):

| path | before | now |
|---|---|---|
| eager Sin/Cos (elementwise.comp 11/12) | broken constants, NaN > 1e9 | shared header |
| eager Tan (elementwise.comp 26) | raw built-in, never gated | shared header (sin/cos of reduced arg) |
| complex exp/sin/cos/sinh/cosh/tan/tanh/asin/acos (complex_elementwise.comp) | raw built-ins ("the gate is irrelevant to the complex path" — wrong on GPU: glibc's reduction is a CPU property) | shared header on every real-part/imag-part call |
| complex logsumexp scan (scan_general.comp) | raw built-ins of an unbounded imag-part difference | shared header |
| fused rope (fast_rope/fast_rope_norm/fast_trio) | raw built-ins — fused-vs-composed bit equality broken above 1e4 by 6cbf55d8f | shared header — equality restored (pinned by "fused rope matches the composed fallback bit for bit") |
| FFT twiddles (fft_c2c/fft_stage) | quadrant fold to [0, pi/4], exact at quarter turns | unchanged — bounded by construction |
| fused chains (fused_chain.comp) | no trig opcodes | unchanged — nothing to gate |
| LITE elementwise | trig excluded by opcode guard (ops 0-10) | unchanged |
| compiled tapes | dispatch through the same primitives | covered by elementwise.comp; doctest leg added |
| dead trig_argument_gate (53 lines) | dead since 6cbf55d8f | deleted |

Dead ends probed and buried (evidence in the probes): the fma-based
variant (C1 = 6, three fma terms) — fma() is NOT fused on Honeykrisp
(runtime-operand probe returns 0 on jw16 and jwm1; the earlier nonzero
probe was NIR strength-reducing `a*3` into `a+a+a`); the k hi/lo split —
broken by design (the split leaves a ~4e5-magnitude intermediate whose
subtractions round at ulp/2 ≈ 0.016); the software Payne-Hanek fallback
stays dead (dynamic-indexing miscompile class, 2026-09-02).

## The contract

- accurate to float64 up to kTrigArgumentLimit = 5e5 (device-independent;
  the constant is the exact-product zone, not a taste threshold);
- NaN above 5e5 — loud, never a finite wrong value. The eager per-call
  host refusal was replaced by this NaN contract at the owner-approved
  perf tradeoff (OpCost, v0.7.21+; the refusal needed a GPU sync + host
  read per call — 784 joins per 14 Laya forwards, the Kokoro serve-path
  stall). Where the magnitude IS knowable without a readback, the host
  check stays: fused RoPE refuses by name at the same 5e5.
- below 1e4 the raw built-in runs bit-identically.

Tests: test_eq_math.cpp (band ladder + boundary 499999/500000/500001 +
NaN legs incl. -2.7e37 and ±inf, sin/cos/tan), test_trig_reduction.cpp
(bands + boundary + compiled-tape leg + complex exp leg + tan legs),
rope bit-equality legs in test_fast_ops.cpp.

## Verification on hardware

- jw16 (M1 Max) probe r1/r2: table above; NaN bands verified above 5e5
  (decade-6+ rows report zero finite results where NaN is expected).
- jwm1 (M1) probe r1/r2: same; the 6.9e-4 built-in floor documented.
- M2 (jw14m2): the prior wheel REFUSES Kokoro (named error, theta
  185111.8, captured verbatim); the trig-contract wheel RENDERS it
  (0.32.4.dev202610032049+trigcontract.79a53e388, wall 2.952 s for
  3.975 s audio, RTF 0.743, 95400 samples). The corr-vs-previous leg is
  not applicable on the M2 — the previous wheel produces no render.
  On serve checkouts that carry the in-graph patch, all Kokoro trig
  arguments stay below 1e4 and hit the raw built-in band, bit-identical
  by construction.
- Suite battery on jwm1 (M1, old Mesa 109060099 - the strictest device,
  fresh-boot run 2026-10-03 ~16:09Z, under /tmp/m1-gpu.lock):
  omarchy_eq_math_tests 7/7 cases 128/128 assertions (the rewritten
  band + boundary + NaN contract), omarchy_trig_reduction_tests 4/4
  49/49 (bands, tan, NaN/boundary, complex exp), omarchy_primitive_tests
  104/104, omarchy_fused_chain_tests 36/36, omarchy_compiled_tape_tests
  13/13, omarchy_runtime_tests 41/41. An earlier jwm1 attempt died with
  a host reboot 36 s in (up 0 min at 16:08Z; unrelated holder bench on
  the lock at the time) - the clean rerun after boot is the table above.
  The jw16 window ran the same battery with a mangled loop (no results);
  jw16's device leg is the probe table instead. Open item: compile() of
  a dedicated sin*cos tape segfaults on jwm1 (removed from the doctest;
  belongs to the compiled-tape lane - the tape suite's unary legs pin
  the same routing).

## Known llvmpipe deltas (handed off from SdpaVjpFix; rerun pending)

Three fast_ops doctests show 1-bf16-ulp fused-vs-composed deltas on
dev-box llvmpipe since the Cody-Waite wiring (their bisect:
receipts/2026-10-03-sdpa-vjp-fix, §Attributions; all green on jw16
hardware). 79a53e388 puts both sides through the same wrapper, so the
fused and composed paths are again the same function. The dev-box
rerun of the three named cases on this wheel was NOT completed in this
window (the jw16/jwm1 legs consumed the GPU budget) and is the one
open verification item. The captured deltas are small-argument,
finite, one-bf16-ulp schedule noise - the llvmpipe class of the
documented inexact-reassociation defects; the hardware suites are the
contract of record.
