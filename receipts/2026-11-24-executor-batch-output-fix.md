# Compiled-executor batch>1 output corruption — root cause + fix

## Symptom (wAW, #808)

Conv-only module through `torch.compile(backend="mlx_vulkan")`:
batch=1 maxdiff 4.8e-7, batch=8 maxdiff 3.717. `mx.conv2d` alone bit-exact.

## Bisect

- `mx.compile` of the bare `_conv2d` lowering: exact at batch-8.
- `_eager_run` path: exact at batch-8.
- Uncompiled `build` (mx.compile→identity): **corrupt** — not codegen.
- Placeholder→input mapping verified: [w(16,3,3,3), b(16), x(8,3,32,32)] correct.
- `outs[0]` mx-side is byte-identical to an eager `_conv2d` on the same
  runtime args and matches torch to ~bf16 noise — mx compute fine.
- `run()` returns `(8,16,32,32)` but the caller sees `(16,32,32)`:
  the batch-0 slice.

## Root cause

`run()` returned a **bare tensor** (`outs[0]`/`_torch(logits)`) for
single-output graphs. Dynamo's calling frame treats the backend return as a
sequence of graph outputs and unpacks it — a 4-D tensor unpacks over dim-0
and the frame takes element 0. Batch=1 masked it (leading dim 1); decode
logits `(1,1,V)` masked it too ((1,V) argmax is unaffected).

## Fix

Always return a sequence: `out = _torch(tuple(logits))` where
`logits = outs[:n_out] if n_out else outs[:1]`.

## Verify (jwm1, Qwen3-0.6B env, conv-only nn.Conv2d(3,16,3,padding=1))

```
bs=1 shape=(1,16,32,32)  diff=4.768e-07
bs=8 shape=(8,16,32,32)  diff=7.153e-07   (was 3.717)
bs=4 shape=(4,16,32,32)  diff=4.768e-07
```

Decode MATCH check unchanged (False at first diff 5) — the #787 decode
regression is a separate, pre-existing issue on main (c3f4fb4f passes,
0365a18a0 fails identically with/without this fix).
