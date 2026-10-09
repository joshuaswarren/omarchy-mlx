# Masked gated-delta prefill on the single-pass route (receipt)

A padded batched prefill (four prompts of different lengths, left padding, a real mask) ran 1.46x to 1.65x of the same four
prompts run one at a time, after PR 40 had removed the larger B > 1 loss. The loss is the masked rows: `GatedDeltaUpdate` took
the single-pass recur32 route only for a maskless row, and only by default on G13 parts other than G13C. A masked row fell to
the two-pass snapshot scan.

## Change

- `gated_delta_prefill_recur32.comp`: a masked-out token leaves the state untouched and writes a zero output (the same contract as
  the scan routes and the composed fallback's `where`). The mask is the existing `[1, T]` scalar per-token validity.
- `primitives.cpp`: the recur32 route no longer requires `!has_mask`. By default a masked prefill takes it on every part with
  32-lane subgroups (mode 2, T >= 2). Maskless defaults are unchanged (G13 except G13C: recur32; G13C and others: coopmat chunk
  kernel). `MLX_OMARCHY_GDN_RECUR32` and `MLX_OMARCHY_NO_COOPMAT_GDN` still override.
- Tests: an fp64 partial-mask test (left padding plus every seventh token masked, non-zero initial state, T 2 to 519, Hv 16 and 32,
  masked tokens must write exactly zero) and a device dispatch-count test (a masked row is one dispatch, as a maskless row).

Behaviour change: outputs of a masked prefill on the new route differ from the old scan at the level the maskless recur32 route
already differs (subgroup reduction trees). The numerics gate is the fp64 reference tolerance of the existing correctness battery
(y 0.02 abs, state 2e-4 abs), not bit equality.

## Source and builds

| item | value |
|---|---|
| branch tip tested | `fa02100a3` (rebased onto main with no change to any GDN file; the only code that differs is unrelated `Int8Matmul` work) |
| wheel | `mlx_omarchy 0.32.4.dev202610090225+qrm.fa02100` |
| control wheel | `mlx_omarchy 0.32.4.dev202610081925+qgpr.390f8a4` (PR 40, merged) |
| model | `qwen3_5-4bit`, `model.safetensors` sha256 `b0d5de688567bf4acd5e421027acd410dabcdc255a5bd46fdbf06c75dc2e6863` |
| driver | Honeykrisp, Mesa 26.3.0-devel (git 6543eeb7df), private ICD through `VK_DRIVER_FILES`; Vulkan API 1.4.362 |
| kernel | `7.1.12-2-12.6-sep-ARCH` (aurora 12.6 prerelease) on both hosts |
| M1 (jwm1) | `Apple M1 (G13G B1)`, MacBook Pro 13-inch 2020, device tree `apple,j293` / `apple,t8103`; system firmware 27.0, OS firmware 13.5, iBoot `8422.141.2`, m1n1 `v1.6.1-omarchy.aurora14` |
| M1 Max (jw16) | `Apple M1 Max (G13C C0)`, MacBook Pro 16-inch 2021, device tree `apple,j316c` / `apple,t6001`; system firmware 26.6.2, OS firmware 13.5, iBoot `8422.141.2`, m1n1 `v1.6.1-omarchy.aurora14` |
| M2 (G14C) | not run: the host was reserved for another job |

## Before: where the padded batch loses (module profile, `pfmodsp-*-before.log`)

`pfmodsp.py` times every sublayer of the real `BatchGenerator` prefill, against the same four prompts as four single runs
(jw16, lengths 454 417 363 304, two runs; jwm1, lengths 120 110 90 70, two runs).

| host | batched / four singles | `mx.fast.gated_delta_update`, batched against the four singles | every other op type |
|---|---|---|---|
| M1 Max (G13C) | 1.33x, 1.32x | 1016 ms against 138 ms (+634%) | mlp +14%, others equal or faster |
| M1 (G13G) | 1.24x, 1.23x | 757 ms against 75 ms (+900%) | mlp +1..3%, others equal or faster |

## Route check (G13C, `pfbgenv-g13c-env-ab.log`)

Rule fixed in `pfbgenv-ticket.sh` before the run: `MLX_OMARCHY_GDN_RECUR32=2` justifies a G13C masked default iff its median ratio
is at most 1.15 and at most 0.85 of the default median. Default route 1.645 and 1.609; forced recur32 1.020 and 1.012; first tokens equal in
all four arms. The rule holds.

## After (`pfbgrm-*-after.log`, rule `rm_accept.py` fixed before the wheel was built)

`batched_s / seq_s` of the real prefill (`pfbg.py`), mirrored arms, control `qgpr` against `qrm`:

| host | lengths | control (PR 40) | this change | first tokens |
|---|---|---|---|---|
| M1 Max (G13C) | 454 417 363 304 | 1.645, 1.646 (median 1.645) | 0.991, 0.980 (median 0.985) | equal in all runs |
| M1 (G13G) | 120 110 90 70 | 1.461, 1.458 (median 1.459) | 0.966, 0.975 (median 0.970) | equal in all runs |

Rule result: ACCEPT on both hosts (at most 1.15 and at most 0.85 of the control). On the M1 the control's first token for row 0 (length
120) differed from the unpadded result in every run, 271 against 198, with a top-2 logit margin of exactly 0.0 in the unpadded pass: a tie
that the old padded scan broke the other way. With this change the padded and unpadded passes use the same reduction order and
the tokens are equal (`pfbg2.py`: verdict EQUAL, two runs on each host).

## Device tests at the tip (`device-doctest-*.log`)

One build per host from `fa02100a3`; both hosts print the same lines.

| result | M1 (G13G) | M1 Max (G13C) |
|---|---|---|
| `omarchy_gdn_decode_batch_tests` | 8 of 8 cases, 73 of 73 assertions | same |
| `[gdn_masked_prefill]` B=1 T=128 maskless, masked, B=4 masked (dispatches) | 1, 1, 4 | 1, 1, 4 |
| `[gdn_prefill_rows]` B=1 T=128, B=4 (dispatches) | 1, 4 | 1, 4 |
| `GDN recur32*` fp64 battery, partial-mask case included | 3 cases, 1178 assertions, all passed | same |

## Not covered

M2 (G14C) was not run, and the change reaches it by default (masked prefill on any 32-lane part). Its device test and the padded-batch
ratio are the open follow-up. The 4B model at larger T on the M1 was not run (device memory on the `[4, T, vocab]` logits).
