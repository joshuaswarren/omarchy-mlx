# Gated-delta prefill route (MLX_OMARCHY_GDN_RECUR32) on G13G: output tokens and speed

Question: on G13G (M1) the default gated-delta route (mode 2, per-token subgroup-reduction kernel) produced different greedy tokens from the route used on G13C (M1 Max) and G14C, and from macOS. Is the difference this kernel, and what does the macOS-identical route (mode 0) cost?

Setup: Qwen3.5-9B 4-bit, the installer's patched mlx-lm 0.32.0, wheel `0.32.4.dev202610090923+a48b7cc5`, greedy, 10 decode samples of 64 tokens (5 prompts x 2 passes), 512-token prefill median of 5 runs, quiet machine (load below 0.5, no compile running), private Vulkan driver build `3546bcafe8ed3995`.

Output digest of the 10 decodes (first 16 hex of the ordered-records SHA-256):

| Chip | Route | Digest |
|---|---|---|
| G13G | default (mode 2) | `83043aa001f15d1b` (3 runs) |
| G13G | `MLX_OMARCHY_GDN_RECUR32=0` | `80274aa790426468` (3 runs) |
| G13C | default (mode 0) | `80274aa790426468` (3 runs); equals the macOS digest on the same model |
| G13C | `MLX_OMARCHY_GDN_RECUR32=2` | `83043aa001f15d1b` |

The digest follows the route, not the chip: predictions written before the runs held on both chips. The first 16 tokens of all five prompts are identical between the routes; the difference appears later.

Speed on G13G (tokens per second, interleaved runs, 3 per route; one mode-0 run was lost to a download timeout so mode 0 has 2):

| Route | Prefill 512 | Decode |
|---|---|---|
| mode 2 (old default) | 99.80 / 99.68 / 99.81 | 12.37 / 12.39 / 12.39 |
| mode 0 (macOS-identical) | 96.97 / 97.02 | 12.37 / 12.35 |

Mode 0 costs 2.8% of prefill (ranges do not overlap) and nothing measurable on decode.
