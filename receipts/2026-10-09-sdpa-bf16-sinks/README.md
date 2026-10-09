# SDPA with bf16 attention sinks (gpt-oss-20b refusal)

Claim: `mx.fast.scaled_dot_product_attention` with bf16 q/k/v and bf16 sinks at the gpt-oss shape `[1, 64, 11, 64]` ran into the named refusal
`attention sinks dtype ScaledDotProductAttention is not implemented for the Omarchy Vulkan backend (dtype=bfloat16, shape=[1,64,11,64])`.
After this change the same call returns the host-reference result. Unit test level only: gpt-oss-20b end to end on Linux is not run here.

## Evidence

| run | binary (sha256 prefix) | source | result |
|---|---|---|---|
| red (`m1max-red.log`) | `2036b60d4beeca03` | tests only, `dced93c2b` | test case throws the refusal above at the first case; 258 assertions passed before it |
| intermediate (`m1max-green1-api-rejected-f32-sinks.log`) | `f00ccf3dda1e5659` | fix plus first test, `35751e8d7` | the four bf16-q cases pass (139,526 assertions); the next case throws `Type of sinks must promote to output type bfloat16`. That is the public API rejecting f32 sinks with bf16 q, so the test was wrong, not the fix |
| green (`m1max-green2.log`) | `381e82808f1520f7` | fix plus corrected test, `92caa0b2f` | 2 test cases passed, 188,680 assertions passed, 0 failed |

- Command: `omarchy_fast_ops_tests -tc='*sinks*'` (two test cases match: the existing f32/f16 sinks test and the new bf16 test).
- Chip: M1 Max (G13C), Linux, private Honeykrisp ICD build. The runs went through the lab's GPU guard queue, one run at a time.
- Tolerances: 3e-2 for bf16 q/k/v (output rounding), 1e-4 for f32 q with bf16 sinks. The host reference is double precision and reads the rounded inputs back.
- The two f32-q cases (new in `92caa0b2f`) have no red run of their own: the red binary predates them. Before the change they would take the same refusal path (f32 composition, non-f32 sinks).

## Not recorded

Kernel and Mesa versions, ICD identity hash, dispatch trace, thermal procedure, device reopen result, the model quantization hash (no model was loaded).
