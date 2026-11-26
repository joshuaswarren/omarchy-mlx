# Muse pipeline rank prune: tiny four-rank run

This is a memory and token check on a tiny random-weight Muse model. It says nothing about Muse Spark Open speed or answers.
It runs on omarchy-mlx via omarchy-cluster (mlx-lm), not the oMLX server.

## What changed
`MLX_OMARCHY_PIPELINE_PRUNE=1` makes the adapter swap two tensors for stubs after the pipeline split.
- `embed_tokens` stays only on the rank that runs the first layers (rank size-1). The other ranks read only the shape and dtype of the embedding.
- `lm_head` stays only on rank 0, the rank that samples. The other ranks already swap it for the token all_gather in omarchy-cluster.
The setting is off by default. The vendored class file is unchanged. A model with `tie_word_embeddings` keeps every tensor, because every rank then computes logits with `embed_tokens`.

## Run
- Setup: four CPU ranks on one x86 machine. Stock mlx 0.32.3. mlx-lm 0.32.0. Ring backend.
- Split: layers 2,2,2,2. Token sharing from omarchy-cluster main at `11c6e34` (`_pipeline_split`, `_share_token`).
- Model: hidden size 256. 8 layers. Vocab 32000. fp16 with 2-bit experts. Random weights.
- Load order: one safetensors file loaded lazily, then the split, then `mx.eval(parameters)`.
- Greedy decode. 12 tokens. Prompt of 8 ids.
- Files: `prune_probe.py`, `run_prune_probe.sh`. Set `MUSE_SERVE_DIR` and `OMARCHY_CLUSTER_DIR`, then run the shell script.

## Provenance
`scripts/mlx_provenance.py` on the Python that ran this: `dist` mlx-omarchy, `verified` no-metadata, `mx_version` 0.32.3.
This is stock mlx from PyPI. No mlx-omarchy wheel was installed, so there is no installed-binary check. The adapter under test is `serve/mlx_omarchy_muse/__init__.py` with sha256 `6d2e261baaeef128eab42f2c919a03746b2fb8097f9621083c57028bf8241d1b` on main at `8af52fb4e`.

## Result
Tokens: identical on all four ranks and identical with the setting on and off: `[1841, 8239, 30859, 8928, 24700, 10261, 4155, 19032, 9265, 1545, 393, 4651]`.
Rank 0 log-probabilities: the sha256 of the float32 bytes of each of the 12 steps matches with the setting on and off. The first eight hex digits per step are `5196ba2a 37d6c284 27d29e2e d6b14643 fc4b7c43 a667e40c 026adab4 a7a722e7 6890427e cc543f00 e0864e20 6dcb032f`.
Ranks 1 to 3 hold a one-hot head by design, so only rank 0 shows real log-probabilities.
The packed expert weights are not seeded, so the tokens differ from run to run. Each comparison above uses one checkpoint file for both settings.

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
