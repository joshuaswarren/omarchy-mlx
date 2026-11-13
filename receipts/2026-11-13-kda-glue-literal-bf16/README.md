# kda_glue_pre / kda_glue_post: literal `bfloat16_t` rounding (2026-11-13)

Machine: 16-inch M1 Max (G13C), Linux 7.1.12, Honeykrisp (Mesa e7631595df),
CPU-only libmlx rebuild; GPU runs through the idle-guard queue.

## Before

`recheck-before-fix.log` (26-kernel harness, main at 92f895912):
`kda_glue_pre` WRONG 13/12288 (maxdiff 0.0164835), `kda_glue_post` WRONG
53/12288 (maxdiff 0.0155039). 15 PASS / 2 WRONG / 9 REFUSED.

## Cause (measured)

`kda_pre_device_vs_emu.py` (run from the repo root, needs the device dump
`kda_glue_pre-io.npz`): the device `q` is bit-identical (0 of 12288
mismatches) to a numpy fp32 model that rounds to bf16 only at the output
store. A Metal-faithful model (round at every `bfloat16_t` local) mismatches
on 6755 of 12288. A 32-combination mask search over the rounding sites
reproduced the device only for "output store only". The translator treated
the literal `bfloat16_t` type as plain float; the template-typename path
already rounded. The same 13 and 53 counts reproduced on a 13-inch M1 (G13G)
dump. `kda_glue_post` shares the code path; that it is the same cause is
inferred from the identical fix turning it exact, not separately dumped.

## Fix and result

`overlay/mlx/backend/omarchy/custom_kernel.cpp` `round_bf16_type()`: the
typename-path rounding now also runs on the literal type; comma declarator
lists are split; the `(float)(bf16 cast)(x)` chain maps to
`_mlx_bf16_round_trip`. Tests: `test_custom_kernel_translate.cpp`, red then
green (translate tests 47/47 after the review fixes).

`recheck-after-fix.log` (libmlx rebuilt from the fix, same harness, same
dispatch via the Vulkan path): `kda_glue_pre` PASS (q, k, v bit-exact, gate
maxdiff 3.76e-07), `kda_glue_post` PASS exact. 17 PASS / 0 WRONG /
0 COMPILE-FAIL / 9 REFUSED, RC=0. The other 24 KERNEL lines match the
before log apart from timing. Repeated 3 times: each run 17 PASS / 0 WRONG
(summary lines compared only).

## Final-code proof (head of this PR, libmlx sha 19e5d2b0)

One shared translation-cache dir; old libmlx, then fixed libmlx, then fixed
again (`final-warm-cache-16inch-m1-max.txt`):

| run | 16-inch M1 Max (G13C) | 13-inch M1 (G13G) |
|---|---|---|
| old libmlx | 15 PASS / 2 WRONG (13/12288, 53/12288), 17 `.tr` entries | same counts, 17 entries |
| fixed libmlx | 17 PASS / 0 WRONG / 9 REFUSED, RC=0, 34 entries | 17 PASS / 0 WRONG / 9 REFUSED, RC=0, 34 entries |
| fixed, warm | 17 PASS, 34 entries | 17 PASS, 34 entries |

`kda_glue_pre` maxdiff 3.76e-07 and `kda_glue_post` exact on both chips. The
fixed library added 17 new cache entries instead of reusing the old ones, so a
warm cache from the old build does not serve the old translation. The
13-inch M1 logs are `recheck-before-fix-13inch-m1.log` and
`recheck-after-fix-13inch-m1.log`. The 13-inch run used the fixed libmlx
copied from the 16-inch build tree with a separate wheel for the rest of the
package, not the wheel of the 2026-11-11 baseline.

## Caveats

- Not run: kernels outside the 26 in the harness.
- Known gap (follow-up, not this change): `x = e` reassignments of a bf16 local are rounded only when the name is not also declared with another scalar type; `x += e` and other compound assignments are never rounded.
- The translation cache key includes the SHA-256 of the loaded libmlx, so a
  rebuilt library never reuses an old `.tr` entry (measured above).
