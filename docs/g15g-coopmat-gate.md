# Cooperative matrix on G15G (M3): driver gate and qualification checklist

`VK_KHR_cooperative_matrix` on M3-class parts (G15G) depends on which Mesa
tree the driver came from. This note records where the gate sits and what a
coopmat-ON build must prove before the omarchy routes that consume it
(`QmmPrefillCoopmatF16`, `MatmulF32Coopmat`, `MatmulDirectF16`,
`MatmulDirectBF16`, `SdpaDecodeNative`) engage.

## Where the gate is

In the merged driver tree, `src/asahi/vulkan/hk_physical_device.c` advertises
`KHR_cooperative_matrix` only when `agx_chip_g17_m5(chip)` holds — G17G or
G17S (M5, AGX3C encoding). Every layer below the extension carries the same
scope:

- the `agx_nir_lower_cmat` NIR pass runs only for `agx_chip_g17_m5`;
- `cmat_muladd_agx` codegen asserts `AGX_ISA_ENCODING_AGX3C` — G15G is
  `AGX_ISA_ENCODING_AGX3B`;
- the `agx3_matrix_encode_mma8`/`gemm16` encoders return false for any
  encoding other than AGX3C;
- the 16x16 `fGeMM16x16x16` form additionally needs the kernel
  `DRM_ASAHI_FEATURE_COMPUTE_MATRIX` feature bit, which only AGX3C parts
  report.

Flipping the extension flag alone is not enough — the encoder refuses AGX3B
and the lowering pass never runs.

## The hardware fact

The 8x8 f16/f32 matrix multiply-add that backs a subgroup-scope cooperative
matrix (`f16matmad8x8`, `f32matmad8x8`) is an AGX3B ISA instruction. The
compose corpus carries real hardware captures reproduced byte-for-byte by
`agx3_matrix_encode_mma8` once the AGX3C check relaxes. Only the 16x16 GEMM
form is M5-only; it alone sets `uses_matrix16` and needs
`DRM_ASAHI_FEATURE_COMPUTE_MATRIX`. An 8x8 cooperative matrix on AGX3B is a
pure userspace capability — no kernel or firmware dependency.

An implementation exists on the `coopmat/agx3b-cmat` branch upstream of the
merged tree: it teaches the AGX3B `f32matmad8x8` encoding (same field ids as
AGX3C minus two AGX3C-only selectors, a different c-is-zero marker, and a
six-bit scoreboard-wait encoding fix) and advertises the extension for every
AGX3B part, with `ASAHI_AGX3B_COOPERATIVE_MATRIX` as an override in both
directions. Hardware evidence so far: the cooperative-matrix compute group
passes 1269/0 on G16S (M4 Pro). G15G has no hardware pass yet.

Earlier work that exposed the matrix feature through a different extension
path was reverted over failing `reduce_sum_*_changedim.*.linear`
conformance cases — that family is what a G15G run must re-prove.

## What a coopmat-ON M3 build must prove

1. The build contains the AGX3B coopmat commits: the `agx3b_cmat` arm in the
   extension gate, `agx3_matrix_encode_mma8` accepting
   `AGX_ISA_ENCODING_AGX3B`, and the scoreboard-wait encoding fix.
2. `VK_KHR_cooperative_matrix` listed, `cooperativeMatrix` feature true,
   `cooperativeMatrixSupportedStages` = compute, and an enumerated 8x8x8
   all-FLOAT32 subgroup-scope non-saturating shape — the
   `cooperative_matrix_f32_8` capability axis. The f16 shape
   (`cooperative_matrix_f16_8`) unlocks `MatmulDirectF16`; bf16 needs
   `VK_KHR_shader_bfloat16` with `shaderBFloat16CooperativeMatrix` and a
   `GL_EXT_bfloat16` build.
3. No 16x16 shapes enumerated on G15G (`max_size` stays 8); `uses_matrix16`
   never set — a kernel that sets it gets every dispatch refused, since the
   kernel reports no `COMPUTE_MATRIX` feature there.
4. `dEQP-VK.compute.*cooperative_matrix*` at 100% on real G15G silicon,
   including a clean `reduce_sum_*_changedim.*.linear` family.
5. `ASAHI_AGX3B_COOPERATIVE_MATRIX` honored in both directions.
6. Omarchy side: the device probe reports `cooperative_matrix_f32_8` with
   subgroup size 32; coopmat QMM prefill output is bit-identical to the tile
   route (the standing regression check); model decode ids match.

## When the probe fails

All coopmat routes fall back to tile/scalar kernels with a different
accumulation order and measurably slower prefill. That is correct behavior —
the routes gate on enumerated capability, not on the chip name.
