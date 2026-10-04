# Draft upstream changes for the M2 Omarchy H3 run (prepared 2026-10-04; NO PRs opened — Lead review required)

## Patch 1: keys-Mac-TensorFold-MiniMax-H3-MLX — pre-encoded text conditioning

`h3_generate.py` gains two options:

- `--dump-text-rows PATH`: after `encode_text`, save the conditioning with
  `mx.save_safetensors(PATH, {"text": text, "tags": tags})` and exit before
  the DiT loads (a cheap "encode-only" mode: `--dump-text-rows` implies
  skipping load_dit entirely — the encoder is 63 GiB and the DiT load is the
  expensive part on small-memory hosts).
- `--text-rows PATH`: load the saved tensors instead of calling
  `encode_text`; the text encoder is never instantiated (63 GiB never
  resident).

`scripts/generate.sh` gains `TEXT_ROWS=` and `DUMP_TEXT_ROWS=` env plumbing.

Why: the M2 host has ~48 GiB free; the text encoder alone is 63 GiB bf16.
Prompt conditioning is 519 rows of small tensors (the encoder pass is
23.7 s warm on the reference host), so encoding once on a big host and
shipping the rows makes small-host runs possible without touching the
encoder.

## Patch 2: drowzeys/TensorFold (fork, H3 family) — load a saved int8 state without re-quantizing

After `int8_mlp(model)` / `int8_attention(model)` the blocks hold
`Int8MLP` / `Int8Linear` / `Int8QKV` whose weights are plain mx arrays
(int8 `wq`, float32 `ws`, optional float32 `b`). `mx.save_safetensors` on the
swapped model therefore produces a ready-to-run int8 checkpoint, but loading
it back rebuilds plain linear layers and the int8 wrappers are lost.

Add a load-side flag, e.g. `int8_mlp(model, from_state=True)` (and the same
for attention), that wraps projections whose saved state already carries the
int8 shapes (`wq` int8 [N, K], `ws` float32 [N]) WITHOUT re-quantizing, so a
host that cannot hold the bf16 DiT (62 GiB) can still run the int8 kernels
from a ~31 GiB int8 checkpoint prepared on a bigger host.

Numerics: identical to the normal swap path — the int8 weights and scales
are bit-identical because they were produced by the same
`quantize_weight`/`quantize_rows` code on the big host; only the host that
performs the swap changes.

## M2 run shape that follows

1. Reference host: `--dump-text-rows` + swapped-model save (patch 2) →
   ships ~31 GiB int8 DiT + ~2 MiB text rows + VAEs (11 GiB) ≈ 43 GiB —
   at the edge of the M2's ~48 GiB free; VAE decode can stay on the
   reference host if it does not fit (ship latents instead).
2. M2 host (gpu-turn window): patched wrapper + omarchy-mlx wheel;
   `--text-rows` + int8-from-state; int8 kernels route to
   `mx.fast.int8_matmul` (the Vulkan W8A8 op, landed on omarchy-mlx main).
3. Compare against the bf16 and int8 macOS references (frames 0/60/123,
   PSNR/rel-L2 + audio metrics).
