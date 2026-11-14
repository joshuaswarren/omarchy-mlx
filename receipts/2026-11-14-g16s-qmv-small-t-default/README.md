# Short-pass GEMV on by default on G16S (M4 Pro) and G13C (16-inch M1 Max)

`MLX_OMARCHY_QMV_SMALL_T` gated the batch-1 `[1, T, K]` short-pass GEMV
(`QmmVecQ4WordSubgroupBatch4BF16`, 2 <= T <= cap) behind the environment
variable, so an unset variable left a 2-to-8-row forward on the coopmat
prefill tile. On G16S and G13C the short pass is now the default when the
variable is unset; every other part stays on the tile (PR #296: a second
GEMV block was slower than the tile from 5 rows on G13G).

Contract: `qmv_small_t_max_rows()` returns -1 when the variable is unset, and
`QuantizedMatmul::eval_gpu` resolves that to the part cap (8 on G16S/G13C, 0
elsewhere). An explicit value still forces the route (2..8, clamped to the
cap); `0`, empty or `MLX_OMARCHY_QMV_BATCH=0` keep the tile.

## Forward A/B, Qwen3-8B 4-bit, G16S (M4 Pro)

`mscan.py` median of runs 1-5, fresh cache each run. `defON` is the new
default; `off` is `MLX_OMARCHY_QMV_SMALL_T=0`.

| rows T | defON (ms) | off (ms) | speedup |
|-------:|-----------:|---------:|--------:|
| 1      | 26.5       | 26.5     | 1.00x   |
| 2      | 35.3       | 143.1    | 4.05x   |
| 4      | 46.7       | 153.6    | 3.29x   |
| 8      | 75.2       | 151.7    | 2.02x   |
| 13     | 145.3      | 145.9    | 1.00x   |
| 16     | 128.8      | 132.9    | 1.03x   |
| 32     | 138.0      | 142.0    | 1.03x   |
| 64     | 244.4      | 241.8    | 0.99x   |
| 128    | 419.1      | 421.9    | 1.01x   |

T=1 (decode) and T>=13 (above the cap, back on the tile) are unchanged. The
win is at T=2..8, which is prompt-prefill of short prompts and the
speculative-verify pass. Full logs: `forward-ab.txt`.

## Row-for-row identity (width invariance)

`width.py` compares each row of a T-row short pass against the single-row
GEMV of that row, K=N=4096, group 64, 4-bit, bf16.

    small-T on : M=2,4,8 width_invariant_vs_M1 True  maxdiff 0.0
    small-T off: M=2,4,8 width_invariant_vs_M1 False maxdiff up to 7.8e-3

The short pass is bit-for-bit the single-row decode kernel; the tile it
replaces is not. So greedy prefill and greedy speculative verify produce the
same tokens as plain single-row decode. Greedy 64-token generation is
byte-identical with the route on and off (`tokens.txt`, SHA 2e31ec84).

## Reproduce

    python width.py          # row identity, on then off
    MLX_OMARCHY_QMV_SMALL_T=0 python mscan.py mlx-community/Qwen3-8B-4bit  # off arm
    python mscan.py mlx-community/Qwen3-8B-4bit                            # default arm

Device: Apple M4 Pro (G16S), omarchy-mlx. Source commit recorded in the
wheel used for the forward A/B.
