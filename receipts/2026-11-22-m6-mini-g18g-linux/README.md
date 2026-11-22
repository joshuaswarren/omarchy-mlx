# M6 mini (G18G, j873g), Linux, 2026-11-22

First standard receipt on the M6 mini. Procedure: `score_bench.py` with the
standard `prompts.jsonl`, 2 warmups, 10 greedy decodes of 64 tokens
(5 prompts x 2 passes), 5 prefills of 512 tokens. Same procedure as the
2026-11-21 m3 Pro G15s Linux receipt.

- Machine: M6 mini (apple,j873g / apple,t8152), 12 cores, 31 GB RAM.
- OS: Omarchy Linux, kernel 7.1.12-aurora-m6usb-ARCH, boot 09c18d28.
- Governor: performance.
- Wheel: installed release wheel `mlx_omarchy-0.32.4.dev202611181701+83cf322`
  (0.7.47 series) in `/usr/lib/omarchy-mlx/venv`.
- Driver: Vulkan ICD sha256 `ee5fbac9500bfbf71c09a0d20153e9cad66712a0a2337580b85de2bac4fb2b6d`
  (local build; mesa git sha for this ICD not recorded). Kernel module
  kernel module sha256 `3f192f53ea1fc1b6784283d91fd07b78dc343464632d08d458783fbe8f9fbc46`.
- Models: `SiddhJagani/Qwen3.8-2B-mlx-4Bit`,
  `mlx-community/Qwen3-4B-Instruct-2507-4bit` (both 4-bit, HF cache).
- Queue: idle-guard m6 GPU ticket `3d8aec7c`, class perf, cap 25 min;
  started 00:57Z, finished 01:02Z on 2026-11-22.

| Model | Decode (tok/s) | Prefill 512 (tok/s) |
|---|---|---|
| Qwen3.8-2B (4-bit) | 7.087 | 196.656 |
| Qwen3-4B-Instruct-2507 (4-bit) | 5.230 | 74.922 |

Decode is the median of 10 greedy 64-token generations; prefill is the
median of 5 runs of one 512-token prompt. Decode stdev 0.038 (2B) and
0.031 (4B); prefill stdev 4.9 (2B) and 0.3 (4B).

Files: the two result JSONs and `provenance.json` (wheel, kernel, boot id,
governor, kernel module sha, ICD sha, model config shas). `SHA256SUMS` covers all
four files.
