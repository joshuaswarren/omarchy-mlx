# DEC-45 RADV context - the one place to cite for subgroup-size reviews

This note connects PR #962's subgroup pin to the eGPU rows so RADV-family
reviews cite one place instead of re-deriving the history.

## The pin (PR #962, DEC-45)

`wB5/dec45-radv-pin32`: on RDNA 3/4 RADV the driver reports
`subgroup_size 64`, and every `== 32u` capability guard skipped the 32-lane
kernels (QMV 2x8w2, ResNorm, SwiGLU, FastRopeNormApple, QmmPrefillCoopmat16).
The PR adds `required_subgroup_size()` (pin 32 when
`VK_EXT_subgroup_size_control` covers compute and 32 is inside the reported
range) and `effective_subgroup_size()` (the width the shaders actually run);
all 39 capability guards in `primitives.cpp` test the effective width now.
A `rdna4-radv` profile lands in `capability_sim` for CI.

## The eGPU rows this changes (B60 / 5060 Ti on the 16-inch host)

The B60 (Intel Arc, intel_icd) and RTX 5060 Ti (NVIDIA, nvidia_icd) rows in
`docs/ANY-GPU.md` and the PT-L comparison row measure the same omarchy-mlx
Vulkan backend on non-Apple GPUs. RADV-family reviews of those rows should
cite this PR for the subgroup behavior:

- The B60/5060-Ti runs ran with the pre-#962 guards (`== 32u` against the
  reported subgroup size). On NVIDIA the reported size is 32, so those runs
  are unaffected. On AMD RADV RDNA 3/4 they would have skipped the 32-lane
  kernels entirely - the reason #962 exists.
- Any new B60/AMD RADV measurement after #962 lands uses
  `effective_subgroup_size()` semantics: pinned 32 on RDNA 3/4, reported
  size elsewhere. Old-vs-new numbers are not directly comparable on AMD.
- The `rdna4-radv` capability_sim profile reproduces the pre-#962 skip in
  CI (subgroup_size 64, no coopmat), which is the regression test for the
  guards.

## Cite this file (not the PR thread) in reviews

RADV-family review comments on the B60/5060-Ti rows, on ANY-GPU.md, and on
TESTED-HARDWARE.md should link here and to PR #962 once, then update the
row in place.
