# Differential harness: compiled vs eager, first-divergence localization

This harness hunts the silent wrong-value defect: with compilation enabled,
the compiled path computes wrong numbers at normal speed, exit code 0, no
warning. Observed on m1-test-host-linux (Apple M1, Honeykrisp Vulkan) at commit
`ff4b05a` and later: a Qwen2.5-0.5B-Instruct-4bit greedy decode emitted
garbage while `MLX_DISABLE_COMPILE=1` produced the correct answer. The C++
batteries pass on llvmpipe and cannot see this class of defect.

Two tools, run in this order:

1. `scripts/probe_tape_eager.py` - the mechanism probe. Seconds to run.
   Tests the mechanism (compiled tape read by small eager ops in a
   data-dependent loop), not the model.
2. `scripts/differential_compile.py` - the differential runner. Runs the
   same computation twice with identical inputs and seed, once compiled and
   once eager, and compares every unit boundary bit-for-bit. Reports the
   FIRST divergence: operation, shapes, dtypes, the flat index of the first
   differing element, and both values. Then shrinks: it emits the smallest
   standalone reproduction of the diverging operation.

## Equality policy

The default comparison is BITWISE. Both runs execute the same primitives on
the same device with identical inputs, so identical bits is the only
legitimate outcome. Any tolerance would hide exactly the corruption this
harness exists to find. `--atol X` switches to tolerance comparison; it
prints a loud warning and exists only for explicit cross-path numerics
debugging.

## Prerequisites

1. A wheel built at the commit under test. The defect is commit-sensitive
   (`ff4b05a` changed the submission timeline); never test a wheel whose
   source commit you cannot name:

   ```bash
   git -C ~/src/mlx-omarchy log --oneline -1          # name the commit
   cd ~/src/mlx-omarchy && ./scripts/build-wheel.sh   # builds from HEAD
   python3 -m venv --system-site-packages .work/venv-run
   .work/venv-run/bin/pip install dist/mlx_omarchy-<version>.whl mlx-lm==0.31.3
   ```

2. Compilation ENABLED. Unset `MLX_DISABLE_COMPILE` before every run below.
   `MLX_DISABLE_COMPILE=1` is the eager side; the harness sets the eager
   pass up itself with `mx.disable_compile()`.

3. Device environment:
   - On the Apple M1 (m1-test-host): never set `MLX_OMARCHY_ALLOW_NON_APPLE`.
   - On a non-Apple Vulkan device (llvmpipe dev box): export
     `MLX_OMARCHY_ALLOW_NON_APPLE=1`.

4. Compiled tapes. They run by default on every device class since the
   stale-shape fix (docs/known-defects.md); no override is needed and
   `MLX_OMARCHY_ALLOW_UNSAFE_COMPILE` is retired (setting it does
   nothing). The former auto-eager gate and backstop were removed with
   the fix; `probe_compile_ordering.py` was retired with them. Keep
   `MLX_DISABLE_COMPILE` unset for the compiled legs; the eager pass is
   set up by the tools themselves.

Exit codes for both tools: `0` match, `3` divergence (or injected
divergence detected), `2` usage or environment error.

## 1. Mechanism probe (run this first; seconds)

```bash
unset MLX_DISABLE_COMPILE
python3 scripts/probe_tape_eager.py
```

Each iteration submits one compiled tape (fusable, with a data-dependent
`mx.where` select inside) and then small EAGER operations that read the
tape's outputs. The eager read size sweeps over `64, 17, 2, 1` - the
one-element output and one-element write match the shape of the earlier
abort at a trigonometric domain gate on a value near 9.1e8. The next tape
input is derived from the eager result, so a stale read cannot produce a
plausible number; it shifts the whole chain and fails the bitwise compare
against an eager-only reference loop.

Variants: `interleave` (sync every iteration), `depth2` (two tape
submissions in flight before the eager read), `nosync` (eight tape+eager
pairs, one sync per batch).

Prove the detector detects (injected stale read at iteration 5):

```bash
python3 scripts/probe_tape_eager.py --inject 5   # expect exit 3, iteration 5
```

Useful flags: `--size`, `--iters`, `--dtype float16|float32`, `--seed`.
`--self-test` runs the comparator checks without mlx.

## 2. Differential runner

Four modes. All compare BITWISE and report the first divergence with the
operation name, shapes, dtypes, first differing flat index, and both values
(float reading and raw bits).

### 2a. Graph mode (synthetic; no model needed)

```bash
python3 scripts/differential_compile.py --mode graph --fn chain   --dtype float16
python3 scripts/differential_compile.py --mode graph --fn swiglu  --dtype float16
```

`chain` mixes fusable elementwise chains with an eager matmul and a
one-element reduction. `swiglu` replicates the mlx-lm swiglu fragment shape
(`silu(gate) * up`), the fragment implicated by the bf16 probes.

Prove the detector detects, and shrink in the same run:

```bash
python3 scripts/differential_compile.py --mode graph --fn chain \
  --inject-unit 2 --shrink --shrink-dir /tmp/repro_shrunk   # expect exit 3
```

`--inject-unit 2` perturbs unit 2's output in the compiled pass only, so
the corruption propagates exactly like the real defect. The report must
name `matmul` as the first divergence. That is the proof the detector
works; rerun without `--inject-unit` for the honest result.

### 2b. Real-path mode (the native mlx-lm compile structure)

Runs the model exactly as mlx-lm does: native per-fragment shapeless
compiles (`mlx_lm.models.qwen2.swiglu`, `nn.silu`, compiled samplers) and
the decode loop, as two env-isolated subprocesses - one with
`MLX_DISABLE_COMPILE` popped, one with `MLX_DISABLE_COMPILE=1` - then
compares per-step logits and tokens bit-for-bit.

```bash
unset MLX_DISABLE_COMPILE
python3 scripts/differential_compile.py --mode realpath \
  --model /path/to/Qwen2.5-0.5B-Instruct-4bit-mlx \
  --prompt "What is the capital of France?" --steps 32
```

Prove the detector detects:

```bash
python3 scripts/differential_compile.py --mode realpath \
  --model /path/to/model --steps 8 --inject-step 3   # expect exit 3, step 3
```

This is the reproduction pass. If the hardware defect is real at this
commit, this mode shows it as a step index plus the two token sequences.

### 2c. Model mode (layer-by-layer localization, then op descent)

Runs the Qwen forward pass layer by layer with full recompute (no KV
cache), so every layer output is a plain function output that is evaluable
under both settings. Pass 1 wraps each layer whole in `mx.compile` for the
compiled leg and records every boundary. On the first diverging layer,
pass 2 descends: it wraps `input_layernorm`, `q/k/v/o_proj`, `rope`,
`gate/up/down_proj`, the module-level `swiglu` and
`scaled_dot_product_attention` fragments, and the layernorms inside that
layer only. It names the first diverging OPERATION.

```bash
python3 scripts/differential_compile.py --mode model \
  --model /path/to/Qwen2.5-0.5B-Instruct-4bit-mlx \
  --prompt "What is the capital of France?" --steps 4
```

Coverage note, stated plainly: the wrapped units are every callable
attribute boundary plus the two module-level fragments. Non-wrapped glue -
residual adds, reshape/transpose chains, the tied-embedding `as_linear`
head - stays inside its parent unit. If every wrapped unit matches but the
parent diverges, the harness says so instead of guessing.

If nothing localizes (real-path diverges, ladder clean), the defect is
specific to the native tape structure or its submission timeline. That is
a result, not a failure of the harness: hand the realpath step index plus
the mechanism probe output to the hardware owner.

### 2d. Shrink mode

Run any diverging mode with `--shrink --shrink-dir DIR`. The harness:

1. Re-runs the diverging unit standalone: eager vs `mx.compile` on the
   captured inputs.
2. If it still diverges, shrinks the inputs: it halves axis 0 of the
   largest input while the divergence survives (bounded at 24 tries).
3. Emits `DIR/repro.py` + `DIR/repro.npz`: a standalone script that
   rebuilds only that operation and its saved inputs and compares compiled
   vs eager. Model-mode repros reload the callable from the pinned model
   by attribute path; graph-mode repros embed the builder.

```bash
unset MLX_DISABLE_COMPILE
python3 DIR/repro.py [--model /path/to/model]   # exit 3 = still diverges
```

Honest limits, printed by the tool itself: if the standalone unit does not
reproduce, the emitted repro says so and the harness reports
`context-only`. A divergence that needs the loop, the cache, or the exact
native tape structure will not survive shrinking to one op; the mechanism
probe is the tool for that class.

## Software-driver limits and the shapeless-retrace refusal - read before trusting a green run

llvmpipe executes submissions synchronously inside `vkQueueSubmit`, so it
cannot expose asynchronous or cross-submission corruption at all. The
entire class of defect this harness hunts is invisible to it. A clean
llvmpipe run proves the harness logic and the detectors only. The defect
reproduction, the localization, and the shrink all must come from the M1.

Shapeless-retrace refusal, recorded 2026-09-03 and CONFIRMED ON BOTH
DEVICES - this is not an llvmpipe quirk:

- llvmpipe x86-64, commit `e7a6542`, wheel
  `0.32.2.dev202609031517+e7a6542`, Mesa lavapipe: with compilation
  enabled, an mlx-lm Qwen2 forward refuses at shapes that retrace the
  shapeless fragments. A 7-token prompt passes; a 16/36/64-token prompt
  refuses at the OLD shape `[1,7,...]`; a first trace at 36 tokens refuses
  immediately at `[1,36,...]`. An unquantized tiny f16 Qwen2 refuses
  identically, so quantization is not a factor. Standalone `nn.silu`,
  `mx.compile(a * mx.sigmoid(a))`, and eager sigmoid on a broadcast view
  all pass in a fresh process, so the gap needs the shapeless-fragment
  tape context (`mlx_lm.models.qwen2.swiglu` is the decorated fragment to
  interrogate), not the Sigmoid kernel in isolation.

- Honeykrisp, Apple M1 (m1-test-host), commit `d1a6bfd`, provenance-gated wheel
  `0.32.2.dev202609031604+d1a6bfd` (installed `libmlx.so` sha256 equal to
  the wheel member): the France-prompt realpath (36 tokens) refuses with
  the same error at `[1,36,...]`, and a 30-token "Hi" prompt refuses at
  `[1,30,4864]`. No silent execution window was reachable through this
  harness on current main. The mechanism probe ran bitwise-clean on the
  same device.

```
RuntimeError: [omarchy] broadcast Sigmoid is not implemented for the Omarchy
Vulkan backend (dtype=float16, shape=[1,7,172]). No GPU kernel exists for it;
no silent CPU fallback occurs.
```

This refusal is the backend contract working: a loud named gap instead of
wrong numbers. Provenance context from the same hardware run: the silent
corruption reported earlier on 2026-09-03 was produced by a STALE
`dev20260903` wheel generation (contaminated venv, pre-CPU-backend
`libmlx`), not by current main. Whether the silent corruption still exists
at current main is unproven in both directions; it is fenced behind this
refusal on the mlx-lm native path. This is why the llvmpipe verification
sweep below uses `--no-chat-template` and `--steps 0` / `--steps 1`
(single compile trace, no retrace).

## llvmpipe verification sweep (2026-09-03, commit e7a6542)

All commands ran with `MLX_DISABLE_COMPILE` unset,
`MLX_OMARCHY_ALLOW_NON_APPLE=1`, `HF_HUB_OFFLINE=1`, wheel at HEAD, against
the tiny f16 Qwen2 fixture and the real 4-bit snapshot:

| # | Run | Result |
|---|-----|--------|
| 1 | probe clean, 32 iters | all 3 variants bitwise-clean, exit 0 |
| 2 | probe `--inject 5` | all 3 variants diverge at exactly iteration 5, exit 3 |
| 3 | graph chain clean | all units match, exit 0 |
| 4 | graph swiglu clean | all units match, exit 0 |
| 5 | graph `--inject-unit 2 --shrink` | FIRST DIVERGENCE names `matmul`, shapes/dtypes/index/values, exit 3; repro emitted |
| 6 | emitted repro.py standalone | runs, reports standalone-clean honestly, exit 0 |
| 7 | realpath tiny clean, 1 step | logits+tokens bitwise match, exit 0 |
| 8 | realpath tiny `--inject-step 0` | logits diverge at step 0, exit 3 |
| 9 | model ladder tiny, 1 forward | all layer boundaries match, exit 0 |
| 10 | model ladder real Qwen 4bit, 1 forward | all layer boundaries match, exit 0 |

Both self-tests (`--self-test`) pass without mlx: comparator bitwise
semantics, NaN payloads, first-index search, and injected-divergence
detection.

## On-device results (Honeykrisp, m1-test-host, 2026-09-03, commit d1a6bfd)

Provenance-gated run by the hardware owner (installed `libmlx.so` sha256
equal to the wheel member; upstream `mlx` absent):

- Mechanism probe: rc=0, all three variants bitwise-clean over 32
  iterations (float16, 64x64). The isolated tape->eager loop with
  one-element writes is clean on real hardware at this commit.
- Realpath, France prompt (36 tokens): compiled child refuses with the
  shapeless-retrace Sigmoid error at `[1,36,...]`; a 30-token prompt
  refuses at `[1,30,4864]`. Reproduction of the earlier silent garbage is
  not reachable on current main through the native mlx-lm path; it was
  last observed on a stale `dev20260903` wheel generation.
- Localization next targets the refusal, not divergence: pin it with a
  direct `q2.swiglu` fragment call, no model, per the constraints above.

## File map

- `scripts/probe_tape_eager.py` - mechanism probe
- `scripts/differential_compile.py` - differential runner (graph, model,
  realpath modes + shrink)
- `docs/differential-harness.md` - this document
