# Gemma 4 E4B 4-bit in the serve catalog (13-inch M1 MacBook Pro)

Entry `gemma-4-e4b-it-4bit`: `mlx-community/gemma-4-e4b-it-4bit` at revision `475b9088d29754a3379866cf5aeb6b41acd313c2`.
All three qualification gates are `untested`. `recommended` is `false`. The entry lets a caller name the model by catalog id instead of a path.

## Pinned facts

- `model.safetensors` is 5,146,800,534 bytes. Its SHA-256 `932b8271...` equals the Hugging Face LFS digest and the file downloaded on the test machine.
- All 10 files of the snapshot are listed with size and SHA-256 (5,179,241,512 bytes in total).
- License `gemma` (Gemma terms of use, from the pinned model card). Not Apache.
- `kv_bytes_per_token` is 16384: four full-attention layers own a cache (the last 18 of 42 layers share earlier caches), 2 KV heads, 512 global head dim, bf16. The 20 owning sliding-attention layers keep a fixed 512-token window of about 21 MB.
- `plan gemma-4-e4b-it-4bit --context 4096 --offline`: weights 4.79 GiB, KV 0.06 GiB, model requirement 6.72 GiB, verdict FITS.

## Run on the 13-inch M1 (15 GB, Mesa 26.3.0-devel 439dc032d7, omarchy-mlx 0.7.40)

`lb-mlx.py` loads the snapshot with the packaged `mlx_lm` and streams 120 tokens at temperature 0 on `Device(gpu, 0)`. Load 5.6 s, peak 4.36 GB. Output files: `lb-mlx.out` (short answer, 3 runs), `lb-mlx2.out` (long answer, table below).

| Run (49-token prompt, 120 tokens out) | Prompt tok/s | Decode tok/s |
|---|---|---|
| 1 | 85.5 | 8.54 |
| 2 | 84.4 | 8.60 |
| 3 | 90.1 | 10.83 |
| 4 | 115.9 | 14.65 |
| 5 | 116.0 | 13.99 |

Decode climbs from 8.5 to 14 tok/s over the five runs (GPU clock ramp).

## Not claimed

No qualification gate was run. The `mlx-omarchy-serve serve` path for this model is not measured here.
