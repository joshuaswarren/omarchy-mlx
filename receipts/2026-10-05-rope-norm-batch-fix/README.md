# rope_rms_norm: per-request array offsets (v0.7.25-27 B>1 crash fix)

Date: 2026-10-05. Lane: RopeNormBatch. Hosts: the M2 (T6021, Apple GPU,
Linux) and the x86 development box (llvmpipe, dev-only).

## Defect

The default-ON qwen3 dense q/k-norm+rope fold (`mx.fast.rope_rms_norm`)
crashed every batched leg with B>1:

```
ValueError: [broadcast_shapes] Shapes (B,8,L,64) and (B,1,L,128) cannot be broadcast.
```

Observed with oMLX-style BatchGenerator serving (4 concurrent requests,
Qwen3-4B). Any HTTP client issuing 2+ concurrent requests to a default
`python -m omlx.server` hit it (`omlx/config.py` `completion_batch_size=8`
default); the managed `mlx-omarchy-serve` mlx-lm backend was safe (pins
decode/prompt-concurrency 1), bare `python -m mlx_lm.server` was exposed
(defaults 32/8).

## Root cause

`patches/mlx-rope-rms-norm.patch` parks the norm weight in `inputs[2]`
(the freqs slot) and pushes it into the inputs vector only after
validation. The shared `rope()` fallback composition then selected

```cpp
auto inv_freqs =
    inputs.size() == 3 ? reciprocal(inputs[2], s) : default_inv_freqs();
```

which reciprocated the norm weight (D elements) as rotary frequencies
(D/2 expected). Every fallback leg - per-batch vector offsets, CPU
streams, VJP composition - built D-wide trig tables and broadcast them
against D/2 rotation halves. B=1 never hit the fallback because the
omarchy GPU primitive fuses (scalar offsets and size-1 array offsets).

## Fix

- New hunk in `patches/mlx-rope-rms-norm.patch`: `norm_weight ?
  default_inv_freqs()` in the fallback composition. The fused kernel
  itself already reads per-batch offsets (`rhs_offset + b * matrix_m`)
  and needed no change.
- The C++ op keeps its internal routing: fused kernel for scalar and
  size-1 array offsets, the composed fallback inside the op for vector
  offsets.
- The interim v0.7.27 patcher fence (`isinstance(cache.offset, int)`)
  and `B == 1` conjuncts are removed; both kill switches stay and the
  fold stays default ON.

## Evidence

Op-level matrix (B{1,2,4} x L{1,17} x offset{int 0, int 5, per-request
array}, H_q=32, H_kv=8, D=128, bf16, fused vs the unfused
`rms_norm -> rope` chain, bitwise via bf16-as-uint16):

| wheel | stamp | bit-exact | errored |
|---|---|---|---|
| shipped v0.7.27-era | `0.32.4.dev202610042317+b8af62c` | 14/18 | 4 (all B>1 array-offset rows, the exact field error) |
| fix | `0.32.4.dev202610050627+885509f` | **18/18** | 0 |

The 14 rows that passed before are bitwise identical across the two
wheels (sha256-pinned fused outputs per row): the B=1 fused decode win
is preserved with zero numerical change.

End to end (M2, Apple GPU, wheel stamped with the fix commit): running
the repo patcher against a stock mlx_lm 0.31.3 `qwen3.py` and forwarding
a tiny random Qwen3 config through `BatchKVCache` with left-padding
offsets: patched logits bit-equal to the eager chain for B in {1,2,4};
the fold dispatches by default (kill switches skip it, bit-identically);
patcher stays wired into `apply-mlx-lm-patches.sh`. 5/5 passed
(`tests/test_rope_norm_batch_behavior.py`).

C++ doctest (dev box, llvmpipe, tests-ON build):
`fused rope_rms_norm serves per-batch array offsets bit-exactly` -
645,132/645,132 assertions green across B{1,2,4} x T{1,17} x N{8,32}
with distinct per-batch offsets against concatenated scalar-offset
references. On Apple GPU the same contract is covered by the op matrix
above.

Notes for reproducibility:

- The omarchy ICD resolver only accepts ICD JSONs whose filename
  contains "honeykrisp"; on the dev box, point `VK_DRIVER_FILES` at a
  honeykrisp-named copy of the lavapipe ICD. A suite run printing
  "passed" with `assertions: 0` is an all-skipped run, not evidence.
- The older doctest case `fused rope_rms_norm is bit-exact against the
  composed chain` fails one cell on llvmpipe (width=204, rows=9, one
  bf16 ULP at element 1586) identically with and without this fix (A/B
  receipts) - a pre-existing llvmpipe-only divergence, not from this
  change; routed to the known-defects owner.

## Land

- main: `ea2657f52` (fix) .. `f6d59324e` (v0.7.28 release candidate tip).
- Interim fence removal: the qwen3 patcher in this landing; the
  qwen3_next patcher fence comes off in a follow-up by its owner once
  this wheel ships.
