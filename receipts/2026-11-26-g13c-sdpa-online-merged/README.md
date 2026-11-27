# G13C one-dispatch online-softmax decode SDPA — merged perf fold (PR #997)

Date: 2026-11-26. Canonical post-merge citation for `SdpaDecodeOnlineSg32BF16Hd128`
(single-dispatch 32-subgroup online-softmax bf16 decode SDPA, HD=128), merged in
`d3a3eaf5dc39d549075c62164ed065e810bdc722`.

## Lineage

- `w8e/g13c-sdpa-online @ 5e26721a6` (2026-11-26 early screen): 16-inch 4B decode
  **84.45 -> 91.77 tok/s** at ctx512, 5 of 5 pairs — the measurement that opened
  the kernel row.
- PR #997 at `833fd293768a9b183a35b2d940ebfc41d1d022f2`: full 4-context
  same-wheel ABBA (table below), plus the ctx4096 cap-safety check.
- Merged `d3a3eaf5`: the numbers below are the citation set for "the merged
  kernel's decode gain on G13C".

## Host and provenance

- Machine: 16-inch MacBook Pro, Apple M1 Max (`Apple M1 Max (G13C C0)`), 32 GPU cores.
- Driver: Honeykrisp Vulkan ICD; `MLX_DISABLE_COMPILE=1`, `HF_HUB_OFFLINE=1`.
- Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit`, bf16 4-bit.
- Harness: matched same-wheel interleaved ABBA (A = `MLX_OMARCHY_SDPA_ONLINE=0`,
  B = unset so the new default fires), 8 warmup tokens, greedy decode, 0 GPU
  timeouts; per-leg token ids hashed.
- Wheel: the PR-head build (`833fd293`); the PR body records the method but not
  the version stamp — see the 5-pair tables in the PR body for per-pair legs.

## Result (medians, 5-pair ABBA per context; ids identical 10/10)

| ctx | gen | base tok/s (ms/tok) | default tok/s (ms/tok) | gain | wins | ids sha |
|---:|---:|---:|---:|---:|---:|---|
| 256 | 64 | 85.32 (11.72) | 92.97 (10.76) | +8.97% (-0.97 ms) | 5/5 | `2b1a1b75ab45` |
| 512 | 64 | 84.50 (11.83) | 91.71 (10.90) | +8.53% (-0.93 ms) | 5/5 | `f35142981fe6` |
| 1024 | 64 | 80.35 (12.44) | 88.96 (11.24) | +10.71% (-1.20 ms) | 5/5 | `09a8d2e13cff` |
| 2048 | 32 | 75.49 (13.25) | 83.72 (11.94) | +10.91% (-1.30 ms) | 5/5 | `05b2eac932f5` |
| 4096 (cap check) | 32 | 71.01 (14.08) | 70.92 (14.10) | -0.13% (+0.02 ms) | 1/3 | `0c00a9bf8c06` |

macOS reference at ctx512: 104.53 tok/s / 9.57 ms/tok (wAW's 2026-11-26 macOS
cell, `receipts/2026-11-26-jw16-macos-4b/` in the private journal stream).

## Default gating (what "merged" means for each part)

| part | k_len <= 2304 | k_len > 2304 |
|---|---|---|
| G13C (M1 Max) | one-dispatch online softmax | split-K, unchanged |
| all other parts | unchanged | unchanged |

The cap exists because a forced run at ctx4096 regressed -7.25% (71.03 -> 65.88)
past the 48 MB SLC. `MLX_OMARCHY_SDPA_ONLINE` `0`/`1` and
`MLX_OMARCHY_SDPA_ONLINE_MAX_KEYS` override; explicit split-K override vars set
before start keep the old route as default.

## Verification carried in the PR

- `glslc` + `spirv-val` clean; `check_compute_kernel_wiring.py` clean;
  `privacy_check.py origin/main..HEAD` clean.
- `omarchy_sdpa_split_tests` 18/18 (1754/1754 assertions) on Lavapipe, f64
  accuracy across k_len 1..2048, one-hot, single-dispatch count, 2304/2305
  boundary, non-G13C refusal, env-override precedence.
- On-device f64 parity k_len in {1,15,16,17,31,32,33,65,513,544} within
  `<= 3 * bf16_rounding + 1e-5`, identical greedy digests.
- Follow-ups tracked separately: HD=256 shapes (Qwen3.8-2B / Qwen3.5-9B) and
  per-shape digest tests.
