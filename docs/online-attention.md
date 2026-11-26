# Online attention kernels

When each online SDPA kernel dispatches on the Metal backend, the measured
defaults per part, and the environment overrides. The full flag rows live in
[kernel-flags.md](kernel-flags.md); this page is the dispatch summary.

All online routes are single-dispatch online-softmax bf16 decode kernels for
Apple GPUs: one workgroup per query head, softmax streamed over the key range,
zero scratch, no atomics, no merge pass, bitwise identical to the split-K and
composed fallbacks at every key length. They require bf16 q/k/v/out, batch 1,
one query token, a 32-lane subgroup with basic, arithmetic and shuffle
operations, and coalesced K/V loads.

## When the online decode kernel dispatches

Default-on per part and head width (PRs #1012, #1016):

| part | head width 128 | head width 256 |
|---|---|---|
| G13C (M1 Max) | keys <= 2304 | off |
| G13G (M1 Ultra) | keys <= 2304 | off |
| G14C (M2 Max) | keys <= 1024 | keys <= 1024 |
| other parts | off | off |

A decode step at context N attends N + 1 keys, so the model call crosses the
cap one token before the context count reaches it.

Device gates beyond the part table: 1,024 threads (32 subgroups) for width
128, 512 threads (16 subgroups) for width 256; shared memory 16,640 B or
8,320 B; K/V strides aligned to 4 bytes (width 128) or 8 bytes (width 256);
no sinks, no logsumexp output. Outside the gates the split-K pair or the
composed route serves the call.

The fused rope variant (PR #1013, G13C) folds Q/K RMSNorm, forward RoPE and
the `k_cache` write into the same online kernel at HD=128, removing two
dispatches and two barriers per layer; it follows the same 2,304-key cap.

## Environment overrides

- `MLX_OMARCHY_SDPA_ONLINE=0` disables the online route everywhere (this also
  disables the fused rope variant).
- `MLX_OMARCHY_SDPA_ONLINE=16` forces the 16-subgroup width-256 kernel;
  any other non-empty value forces the 32-subgroup kernel.
- An explicit split-K override (`MLX_OMARCHY_SDPA_DECODE_SPLIT`,
  `MLX_OMARCHY_SDPA_SPLIT_PV`, `MLX_OMARCHY_SDPA_SPLIT_BK64`,
  `MLX_OMARCHY_SDPA_SPLIT_MERGE`, `MLX_OMARCHY_SDPA_SPLIT_P1KV`) defers the
  unset case to the split route.
- `MLX_OMARCHY_SDPA_ONLINE_MAX_KEYS` overrides the per-part key cap.
- `MLX_OMARCHY_SDPA_ONLINE_ROPE_MAX_KEYS` overrides the rope variant cap.
- `MLX_OMARCHY_SDPA_SPLIT_ROWS=0` keeps verify-style multi-row calls on the
  composed route.

Measured basis (PRs #1012, #1013, #1016): Qwen3-4B 4-bit decode, five-pair
ABBA blocks, bit-identical greedy ids, +8.55 percent (80.26 to 86.84 tok/s)
at 512 keys on the default-on parts; HD=256 on G14C +2.58 percent at 128 keys
and +3.34 percent at 512, turning -1.02 percent at 2048, which is why the
G14C cap stays at 1024.

## What runs when

1. bf16 decode, one query token, inside the part cap: the online kernel
   (rope-fused on G13C at HD=128, plain otherwise).
2. bf16 decode, one query token, past the cap or off-part: the split-K pair
   while its own window holds, then the composed route.
3. bf16 verify calls with 2 to 16 query rows: the split-K pair one row at a
   time (`MLX_OMARCHY_SDPA_SPLIT_ROWS=0` composes).
4. f16, sinks, logsumexp, batched decode past the batch gate, or anything the
   gates refuse: the composed route.
