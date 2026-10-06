# Staging-cache design: persistent packed [scales|biases] buffer for affine gather_qmm

Status: design only (Lead, 2026-10-06: implementation waits for the DescRange
allocator/completion-tracking fix; this document is the 14:00Z deliverable).
Author: MoeLayer2. Baseline: dispatch_gather_qmm, overlay/mlx/backend/omarchy/primitives.cpp
(the SEP variant of this idea failed numerics and is deleted; see receipts here
and 25bf62da9 in history — the failure was in its SEPARATE-BINDINGS kernels,
not in the staging idea).

## Cost being attacked

Per affine gather_qmm call today (m==1 decode): 1 buffer allocation +
`fill_buffer` + up to 4 `copy_buffer` (scales bytes, bias bytes, lhs words,
rhs words) + 1 compute dispatch. 78 calls/token => ~390 recorded
copy/fill commands per token. Old-wheel ledger: CopyGeneralBF16+U32
~0.9 s of a 31.9 s span (n=14852+3646) ~= 15-20 ms/token class. After the
subgroup kernel removes the gather math cost, staging is the next-largest
identified item.

## Design

Cache ONLY the immutable weight-parameter region (scales+biases). The two
index words change every call (top-k ids) and stay on the existing
per-call copy path (2 small copies instead of 4 + fill + alloc).

- Cache object: per-device map in the omarchy allocator-adjacent layer
  (device.h/device.cpp), guarded by the encoder lock discipline already
  used for the descriptor pools.
- Key: {scales VkBuffer handle, scales byte offset, scales byte size,
  biases buffer handle, biases byte offset, biases byte size, group_size,
  bits, transpose, no_bias, dtype}.
- Value: a persistent VulkanBuffer of `packed_bytes` sized exactly as
  today (align4 rules unchanged), holding [scales | biases] at the same
  offsets the scalar kernels already decode (aux_offset/shape[1] routing
  is unchanged), plus the recorded {index_base, index_count}.
- Hit path: bind the cached buffer as binding 1; record ONLY the two
  index `copy_buffer` commands into the cached buffer's index region;
  skip allocation, skip the fill (the fill today only zeroes <=6 padding
  bytes plus regions that are then fully overwritten — the kernel never
  reads padding), skip both weight copies. 6 recorded commands -> 2.
- Miss path: build exactly as today, insert, record as today.
- Alias safety: the key pins the SOURCE arrays' buffer handle+offset+size.
  A recycled buffer with a reused handle is the false-hit hazard — this is
  the exact hazard class the DescRange allocator/completion-tracking fix
  closes; implementation is gated on that fix landing and will additionally
  hold a reference on the source VulkanBuffer so a cached entry pins its
  sources alive (no recycle while cached).
- Capacity: LRU capped at 64 entries / 256 MB (decode models touch 26
  layers x 3 projections = 78 hot entries at the cap boundary; prefill
  shapes share the same keys so no growth). Eviction frees the cached
  buffer via the allocator.
- Env gate: `MLX_OMARCHY_GATHER_QMM_STAGE_CACHE=1`, default OFF (the
  default flips only after the same TF/PPL/agreement evidence as the Sub
  kernel — staging changes WHICH bytes the kernel reads; a cache bug is a
  numerics bug, so the gate applies).
- Tests: extend test_matmul_family.cpp — counters assert fills 1->0 and
  copies 4->2 on a repeat call with identical weights; bit-exact pairing
  vs the uncached path (same staging bytes => identical outputs);
  eviction + aliasing tests keyed on the reuse-lag fix's guarantees.

## Expected effect

Removes ~3 of ~6 recorded commands per call and the per-call allocation:
old-wheel profile arithmetic says ~12-18 ms/token of a 106-135 ms/token
step on the Sub wheel (and ~20 ms/token of the 283 ms/token baseline).
A/B protocol: 5 alternating pairs, medians, against the same wheel with
the env off.
