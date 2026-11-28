# 16-inch M1 Max (G13C) 3-Dispatch Greedy Word Head (`MLX_OMARCHY_GREEDY_WORD=1`) 5-Pair ABBA

Date: 2026-11-27. Hardware: 16-inch MacBook Pro (`Apple M1 Max (G13C C0)`, 32 GPU cores), Linux `7.1.12-2-20261103.2-lab-ARCH`, pinned Mesa `53ad9af1` (`libvulkan_asahi` sha256 `c161301c6590ab10...`), `main` `d56f6af03` (`core.cpython-314-aarch64-linux-gnu.so` sha256 `2e58c140b1e83b95...` + `patches/mlx-lm-greedy-prune.patch` `word_route` overlay `e9691d89...`), `MLX_OMARCHY_DEFER_TOKENS=0`, `MLX_OMARCHY_DEFER_COMMIT=0`, `MLX_DISABLE_COMPILE=1`. All runs executed exclusively under the GPU queue lock with zero GPU timeouts (`delta=0`) and zero foreign GPU usage (`count=0`).

## Question

On Apple M1 Max (`G13C`), both `overlay/mlx/backend/omarchy/device.h` (`greedy_word_over_staged_part`) and `patches/mlx-lm-greedy-prune.patch` (`word_route`) defaulted off. As a result, `Qwen3.8-2B-mlx-4Bit` (`K=2048`) took the 8-stage `QmmVecGreedyBF16` path and executed the synchronous per-token `int(token) >= emb.weight.shape[0]` sentinel check inside `_step()` before `mx.async_eval(next_y)`, draining the GPU every token and triggering firmware GPU idle power-off cycles. Does enabling the 3-dispatch `QmmVecGreedyWordBF16` route (`word_route=True`, sentinel unreachable) win on `G13C` with bit-identical greedy token IDs?

## Verdict

Enable `G13C` by default in both `greedy_word_over_staged_part()` (`device.h`) and `word_route` (`patches/mlx-lm-greedy-prune.patch`):

- **`Qwen3.8-2B-mlx-4Bit` `ctx=512` (5 pairs x 3 inner, `gen=64`):** `116.595 -> 168.342 tok/s` (**`+44.38%`**, `+51.75 tok/s`, **5/5 wins**), prefill `1045.9 -> 1059.7 tok/s` (`+1.32%`, 5/5), GPU firmware `power_off` cycles per 63 decode tokens **`38 -> 0`** (`0/15`), bit-exact greedy IDs SHA-256 `2f4697d06db73dba`.
- **`Qwen3.8-2B-mlx-4Bit` `ctx=2048` (5 pairs x 3 inner, `gen=64`):** `111.240 -> 160.651 tok/s` (**`+44.42%`**, `+49.41 tok/s`, **5/5 wins**), prefill `1685.5 -> 1690.2 tok/s` (`+0.28%`, 5/5), GPU firmware `power_off` cycles per 63 decode tokens **`55 -> 0`** (`0/15`), bit-exact greedy IDs SHA-256 `93a61074d50a2788`.
- `MLX_OMARCHY_GREEDY_WORD=0` remains available to restore the staged `K=2048` path.

## Method

- Model: `SiddhJagani/Qwen3.8-2B-mlx-4Bit` (snapshot `0867d98bfb174b042d88461c0e7c97b86b34b381`), prompt `"The history of France begins in antiquity. " * 200` sliced to `ctx=512` or `ctx=2048` tokens, `gen=64` tokens (`63` timed decode tokens after prefill sync).
- Arms (single binary, zero rebuild):
  - **Base:** `MLX_OMARCHY_GREEDY_WORD` unset (default off on `G13C`: 8-stage `QmmVecGreedyBF16` + per-token `int(token)` sentinel sync).
  - **Cand:** `MLX_OMARCHY_GREEDY_WORD=1` (3-dispatch `QmmVecGreedyWordBF16` + `word_route=True`, skipping the unreachable sentinel sync).
- Order: 5 interleaved ABBA pairs (`base->cand`, `cand->base`, `base->cand`, `cand->base`, `base->cand`), 3 inner trials per leg (`15` timed trials per arm per context), 3.0 s cooldown between legs.
- Unperturbed boundary power-off counter: persistent root helper reads `/sys/kernel/debug/dri/406400000.gpu/asahi_stats` once right after prefill sync and once right after final decode sync (zero mid-decode polling).

## Per-Pair Results

### `Qwen3.8-2B-mlx-4Bit` `ctx=512`

| Pair | Order | Base median tok/s (trials) | Cand median tok/s (trials) | Decode delta | Prefill (base -> cand) | `power_off` / 63 tok (base -> cand) | Greedy SHA-256 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `base->cand` | `114.376` (`120.489, 113.090, 114.376`) | `168.081` (`167.960, 168.081, 168.334`) | `+46.96%` | `1044.1 -> 1059.6` | `[33, 42, 40] -> [0, 0, 0]` | `2f4697d06db73dba` |
| 2 | `cand->base` | `116.595` (`115.959, 116.595, 117.232`) | `169.148` (`169.148, 168.314, 169.388`) | `+45.07%` | `1045.9 -> 1061.8` | `[39, 38, 37] -> [0, 0, 0]` | `2f4697d06db73dba` |
| 3 | `base->cand` | `115.046` (`117.762, 113.977, 115.046`) | `167.748` (`167.771, 167.748, 167.683`) | `+45.81%` | `1046.2 -> 1059.7` | `[35, 45, 38] -> [0, 0, 0]` | `2f4697d06db73dba` |
| 4 | `cand->base` | `118.573` (`118.573, 119.048, 109.461`) | `168.508` (`168.682, 168.508, 168.493`) | `+42.11%` | `1046.7 -> 1057.0` | `[29, 29, 51] -> [0, 0, 0]` | `2f4697d06db73dba` |
| 5 | `base->cand` | `118.326` (`118.326, 119.967, 107.648`) | `168.342` (`168.342, 168.326, 168.512`) | `+42.27%` | `1042.9 -> 1060.0` | `[32, 30, 56] -> [0, 0, 0]` | `2f4697d06db73dba` |
| **Median** | **5/5 wins** | **`116.595`** | **`168.342`** | **`+44.38%`** | **`1045.9 -> 1059.7` (`+1.32%`, 5/5)** | **`38 -> 0`** | **`2f4697d06db73dba` (bit-exact)** |

### `Qwen3.8-2B-mlx-4Bit` `ctx=2048`

| Pair | Order | Base median tok/s (trials) | Cand median tok/s (trials) | Decode delta | Prefill (base -> cand) | `power_off` / 63 tok (base -> cand) | Greedy SHA-256 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `base->cand` | `107.868` (`108.535, 107.868, 102.433`) | `160.630` (`160.913, 159.937, 160.630`) | `+48.91%` | `1685.8 -> 1687.8` | `[59, 56, 63] -> [0, 0, 0]` | `93a61074d50a2788` |
| 2 | `cand->base` | `111.240` (`106.630, 111.240, 112.437`) | `161.681` (`161.856, 161.681, 161.560`) | `+45.34%` | `1680.5 -> 1693.2` | `[62, 55, 52] -> [0, 0, 0]` | `93a61074d50a2788` |
| 3 | `base->cand` | `110.696` (`110.636, 111.506, 110.696`) | `160.651` (`160.933, 160.651, 157.596`) | `+45.13%` | `1682.7 -> 1690.0` | `[53, 52, 56] -> [0, 0, 0]` | `93a61074d50a2788` |
| 4 | `cand->base` | `115.519` (`115.590, 115.519, 112.596`) | `160.510` (`160.510, 160.597, 159.665`) | `+38.95%` | `1688.5 -> 1694.5` | `[29, 41, 51] -> [0, 0, 0]` | `93a61074d50a2788` |
| 5 | `base->cand` | `111.759` (`111.759, 105.391, 111.784`) | `161.152` (`161.372, 161.083, 161.152`) | `+44.20%` | `1685.5 -> 1690.2` | `[57, 53, 61] -> [0, 0, 0]` | `93a61074d50a2788` |
| **Median** | **5/5 wins** | **`111.240`** | **`160.651`** | **`+44.42%`** | **`1685.5 -> 1690.2` (`+0.28%`, 5/5)** | **`55 -> 0`** | **`93a61074d50a2788` (bit-exact)** |
