# Muse pipeline rank prune: tiny four-rank run

This is a memory and token check on a tiny random-weight Muse model. It says nothing about Muse Spark Open speed or answers.
It runs on omarchy-mlx via omarchy-cluster (mlx-lm), not the oMLX server.

## What changed
`MLX_OMARCHY_PIPELINE_PRUNE=1` makes the adapter swap two tensors for stubs after the pipeline split.
- `embed_tokens` stays only on the rank that runs the first layers (rank size-1). The other ranks read only the shape and dtype of the embedding.
- `lm_head` stays only on rank 0, the rank that samples. The other ranks already swap it for the token all_gather in omarchy-cluster.
The setting is off by default. The vendored class file is unchanged.

## Run
- Setup: four CPU ranks on one x86 machine. Stock mlx 0.32.3. mlx-lm 0.32.0. Ring backend.
- Split: layers 2,2,2,2. Token sharing from omarchy-cluster main at `11c6e34` (`_pipeline_split`, `_share_token`).
- Model: hidden size 256. 8 layers. Vocab 32000. fp16 with 2-bit experts. Random weights.
- Load order: one safetensors file loaded lazily, then the split, then `mx.eval(parameters)`.
- Greedy decode. 12 tokens. Prompt of 8 ids.
- Files: `prune_probe.py`, `run_prune_probe.sh`.

## Result
Tokens: identical on all four ranks and identical with the setting on and off: `[8371, 28438, 10492, 31308, 437, 3614, 25398, 24825, 5749, 13764, 4423, 22877]`.

Active memory in bytes, `mx.get_active_memory()`:

| Rank | Layers | Off, after eval | Off, after token swap | On, after eval | On, after token swap |
|---|---|---|---|---|---|
| 0 | 6-7 | 35,248,688 | 35,248,688 | 18,864,680 | 18,864,680 |
| 1 | 4-5 | 35,248,680 | 18,864,680 | 2,512,680 | 2,480,680 |
| 2 | 2-3 | 35,248,680 | 18,864,680 | 2,512,680 | 2,480,680 |
| 3 | 0-1 | 35,248,680 | 18,864,680 | 18,896,680 | 18,864,680 |

Each of `embed_tokens` and `lm_head` is 16,384,000 bytes here (32000 x 256 x 2).
- Steady state: ranks 1 and 2 save the embedding. Rank 0 saves the embedding. Rank 3 saves nothing.
- Load peak, setting off: ranks 1 to 3 hold both tensors until the token swap.
- Load peak, setting on: one tensor less on rank 0 and both gone on ranks 1 and 2.
- For the real build, the byte count of the shared shard (2.28 GB) sets the saving.

## Limits
CPU ranks. One run each. Random weights. The real `sharded_load` path from an indexed directory was not run with the setting. The setting needs one rank to sample and share the token.
