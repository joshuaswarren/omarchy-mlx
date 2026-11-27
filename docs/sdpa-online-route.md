# Online-softmax decode SDPA route

The 16-inch M1 Max decode path can run scaled_dot_product_attention as **one
dispatch per query head** instead of the split-K pass-1/pass-2 pair. This page
says when the online route fires and how to override it, for operators who see
`SdpaDecodeOnlineSg32BF16Hd128` in a dispatch trace and for anyone calibrating
a new part.

Implementation: `sdpa_decode_online.comp -DSUBGROUPS=32`
(`overlay/mlx/backend/omarchy/shaders/`) — one 1024-thread (32-subgroup)
workgroup per query head, coalesced 256-byte `uvec2` K/V loads, FP32
online softmax in registers, zero scratch/atomics/pass-2. Selection lives in
`ScaledDotProductAttention::eval_gpu`
(`overlay/mlx/backend/omarchy/primitives.cpp`, the
"Single-dispatch online-softmax bf16 decode route" block); the per-part cap
lives in `sdpa_online_max_keys`
(`overlay/mlx/backend/omarchy/device.h`).

## When it dispatches

All of these must hold at one call:

**Route wanted** — the first match wins:

- `MLX_OMARCHY_SDPA_ONLINE` set to anything except `0` → forced on
  (skips the cap; shape/device gates below still apply).
- `MLX_OMARCHY_SDPA_ONLINE=0` → never taken.
- Unset → taken only when **no** explicit split-route override is set
  (`MLX_OMARCHY_SDPA_DECODE_SPLIT`, `MLX_OMARCHY_SDPA_SPLIT_PV`,
  `MLX_OMARCHY_SDPA_SPLIT_BK64`, `MLX_OMARCHY_SDPA_SPLIT_MERGE`,
  `MLX_OMARCHY_SDPA_SPLIT_P1KV`) **and** the part default says yes:
  `sdpa_online_default(device, k_len)`, i.e. `k_len <= max_keys(part)`.

**Call shape** — all bf16 (`q`, `k`, `v`, `out`); 3 inputs; no sinks; no
`output_logsumexp`; `batch == 1`; `q_len == 1`; `head_dim == 128`;
`v_dim == 128`; `k_len > 0`; `heads % kv_heads == 0`; innermost strides 1;
K and V head/key strides multiples of 4; K/V element offsets multiples of 4.

**Device** — subgroup size 32; `max_compute_work_group_invocations >= 1024`;
`max_compute_work_group_size[0] >= 1024`; `max_compute_shared_memory_size >=
16640`; subgroup BASIC|ARITHMETIC|SHUFFLE. And `MLX_OMARCHY_SDPA_DECODE_NATIVE`
unset or non-`0` (the `0` kill-switch covers every native decode variant,
online included).

## Per-part default cap

| part | `max_keys` | basis |
|---|---|---|
| G13C (16-inch M1 Max) | **2304** | Qwen3-4B 4-bit decode ABBA vs shipped defaults: +8.53% to +10.91% (−0.93 to −1.30 ms/tok, bit-identical greedy ids) at ctx 256–2048; −7.25% at ctx 4096 once redundant KV reads spill SLC. 2304 covers ctx 2048 + 256 generated tokens; ctx 4096 stays on split-K. |
| every other part | 0 (never default) | unmeasured; add a row only with an ABBA receipt |

## Overrides

| flag | effect |
|---|---|
| `MLX_OMARCHY_SDPA_ONLINE` | `0` disables the route; any other non-empty value forces it where the shape/device gates pass; unset follows the part cap. Read live per call. |
| `MLX_OMARCHY_SDPA_ONLINE_MAX_KEYS` | replaces the part cap for the unset-default path (`strtoul`; `0` means never default). |
| `MLX_OMARCHY_SDPA_DECODE_SPLIT` (and the other split knobs) | when set to anything, an unset `SDPA_ONLINE` defers to the split family; an explicit `SDPA_ONLINE` non-`0` still wins. |
| `MLX_OMARCHY_SDPA_DECODE_NATIVE=0` | disables every native decode variant — online, split, fused — and returns the call to the composed route. |

## When a gate misses

The call falls through to the split-K decode route (its own gates: HD 128/256,
128-key blocks, 16-byte K rows / even V strides), then to the composed fused
path — never to a copy. The online route is output-neutral only against its
own reference: `test_sdpa_split.cpp`'s online case gates the cap table,
checks float64-referenced accuracy, and asserts the single-dispatch count.

Related: `kernel-flags.md` for the full env table; `decode-execution.md` for
where this dispatch sits in token cost.
