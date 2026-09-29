# Chat-model benchmark summary

| label | hf_repo | cold_load_s | warm_load_s | peak_mem_gb | ttft_p50_s | prefill_512_p50_s | prefill_2048_p50_s | decode_512_tps | decode_2048_tps | card_valid | gsm8k | ife |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2b-baseline | SiddhJagani/Qwen3.8-2B-mlx-4Bit | 1.15 | 1.16 | 1.06 | 0.30 | 0.41 | 1.45 | 69.20 | 49.83 | 9/13 | 0/0 | 0/0 |
| qwen3.5-9b | mlx-community/Qwen3.5-9B-MLX-4bit | 2.73 | 1.95 | 5.04 | 6.18 | 13.25 | 56.81 | 18.26 | 15.74 | 5/16 | 3/20 | 0/19 |
