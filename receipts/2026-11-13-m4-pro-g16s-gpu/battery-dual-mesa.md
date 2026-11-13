# Dual-Mesa battery (system Mesa vs private aurora-wip Mesa)

Same binaries (build of this branch), one process per suite under `flock /tmp/m1-gpu.lock`, `timeout 1800`, capsim profiles, flash per case.

- system Mesa: 26.3.0-devel git-05ddb00f5c (Borealis)
- private Mesa: 26.3.0-devel git-623effac89 (aurora-wip, includes fd2c4a67 packHalf2x16 round-toward-zero), `VK_DRIVER_FILES` + `LD_LIBRARY_PATH` to the private prefix
- private + opt-in routes: private Mesa with `MLX_OMARCHY_QMM_VEC_TOKEN_MULTI=1 GDN_GQA_BF16A=1 GDN_PF_GQA=1 GDN_PF_WIDE=1`

Cells are passed/total doctest cases (or the exit code for units without a doctest summary).

flash-15 ("chunked causal prefill pins two score sets, not a pair per chunk") matched no case in the battery script run (0/0); it was rerun by hand with the same selector, 1 passed on all three columns.

| unit | system | private | private + opt-in |
|---|---|---|---|
| omarchy_ane_bundle_tests | 48/48 | 48/48 | 48/48 |
| omarchy_ane_tile_layout_tests | 3/3 | 3/3 | 3/3 |
| omarchy_ane_worker_gate_tests | 27/27 | 27/27 | 27/27 |
| omarchy_ane_worker_tests | 17/17 | 17/17 | 17/17 |
| omarchy_compiled_tape_tests | 14/14 | 14/14 | 14/14 |
| omarchy_complex_ops_tests | 34/34 | 34/34 | 34/34 |
| omarchy_conv_gemm_decomp_tests | 3/3 | 3/3 | 3/3 |
| omarchy_conv_tests | 15/15 | 15/15 | 15/15 |
| omarchy_copy_offset_tests | 29/29 | 29/29 | 29/29 |
| omarchy_custom_kernel_translate_tests | 44/44 | 44/44 | 44/44 |
| omarchy_device_info_tests | 13/13 | 13/13 | 13/13 |
| omarchy_distributed_tests | 9/9 | 9/9 | 9/9 |
| omarchy_eig_ops_tests | 9/9 | 9/9 | 9/9 |
| omarchy_eq_math_tests | 7/7 | 7/7 | 7/7 |
| omarchy_error_contract_tests | 3/3 | 3/3 | 3/3 |
| omarchy_exl3_decode_tests | 4/4 | 4/4 | 4/4 |
| omarchy_fast_ops_tests | 52/52 | 52/52 | 52/52 |
| omarchy_fast_regression_tests | 2/2 | 2/2 | 2/2 |
| omarchy_fft_general_tests | 14/14 | 14/14 | 14/14 |
| omarchy_fft_ops_tests | 19/19 | 19/19 | 19/19 |
| omarchy_fused_chain_tests | 42/42 | 42/42 | 42/42 |
| omarchy_gdn_bf16_state_tests | 12/12 | 12/12 | 12/12 |
| omarchy_gdn_conv_delta_tests | 7/7 | 7/7 | 7/7 |
| omarchy_gdn_decode_batch_tests | 9/9 | 9/9 | 9/9 |
| omarchy_gdn_fast_route_repeat_tests | 3/3 | 3/3 | 3/3 |
| omarchy_gdn_gqa_bf16a_tests | 10/10 | 10/10 | 10/10 |
| omarchy_gdn_legacy_policy_tests | 3/3 | 3/3 | 3/3 |
| omarchy_gdn_maskless_correctness_tests | 6/6 | 6/6 | 6/6 |
| omarchy_gdn_prefill_profile_tests | 3/3 | 3/3 | 3/3 |
| omarchy_indexing_ops_tests | 60/60 | 60/60 | 60/60 |
| omarchy_int8_matmul_tests | 5/5 | 5/5 | 5/5 |
| omarchy_kv_ops_tests | 16/16 | 16/16 | 16/16 |
| omarchy_linalg_ops_tests | 31/31 | 31/31 | 31/31 |
| omarchy_m2v_route_tests | 10/10 | 10/10 | 10/10 |
| omarchy_matmul_family_tests | 46/46 | 46/46 | 46/46 |
| omarchy_primitive_tests | 107/107 | 107/107 | 107/107 |
| omarchy_qmm_geometry_tests | rc=0 | rc=0 | rc=0 |
| omarchy_qmv_batch_tests | 4/4 | 4/4 | 4/4 |
| omarchy_reduce_ops_tests | 36/36 | 36/36 | 36/36 |
| omarchy_reduce_row_tests | 6/6 | 6/6 | 6/6 |
| omarchy_runtime_tests | 60/60 | 60/60 | 60/60 |
| omarchy_scatter_determinism_tests | 21/21 | 21/21 | 21/21 |
| omarchy_sdpa_causal_ragged_tests | 3/3 | 3/3 | 3/3 |
| omarchy_sdpa_decode_fused_tests | 7/7 | 7/7 | 7/7 |
| omarchy_sdpa_split_tests | 14/14 | 14/14 | 14/14 |
| omarchy_select_layout_tests | 13/13 | 13/13 | 13/13 |
| omarchy_shape_ops_tests | 25/25 | 25/25 | 25/25 |
| omarchy_take_bool_tests | 5/5 | 5/5 | 5/5 |
| omarchy_take_fill_tests | 9/9 | 9/9 | 9/9 |
| omarchy_take_u8_tests | 7/7 | 7/7 | 7/7 |
| omarchy_trig_reduction_tests | 4/4 | 4/4 | 4/4 |
| omarchy_vocab_tail_tests | 7/7 | 7/7 | 7/7 |
| omarchy_wrong_value_sweep_tests | 2/2 | 2/2 | 2/2 |
| capsim-m1-honeykrisp-fork | 10/10 | 10/10 | 10/10 |
| capsim-m1-stock-no-coopmat | 10/10 | 10/10 | 10/10 |
| capsim-subgroup-size-64 | 10/10 | 10/10 | 10/10 |
| capsim-small-shared-memory | 10/10 | 10/10 | 10/10 |
| capsim-no-cooperative-matrix | 10/10 | 10/10 | 10/10 |
| capsim-workgroup-256 | 10/10 | 10/10 | 10/10 |
| capsim-workgroup-128 | 10/10 | 10/10 | 10/10 |
| capsim-m1-g13-legacy | 10/10 | 10/10 | 10/10 |
| flash-1 | 1/1 | 1/1 | 1/1 |
| flash-2 | 1/1 | 1/1 | 1/1 |
| flash-3 | 1/1 | 1/1 | 1/1 |
| flash-4 | 1/1 | 1/1 | 1/1 |
| flash-5 | 1/1 | 1/1 | 1/1 |
| flash-6 | 1/1 | 1/1 | 1/1 |
| flash-7 | 1/1 | 1/1 | 1/1 |
| flash-8 | 1/1 | 1/1 | 1/1 |
| flash-9 | 1/1 | 1/1 | 1/1 |
| flash-10 | 1/1 | 1/1 | 1/1 |
| flash-11 | 1/1 | 1/1 | 1/1 |
| flash-12 | 1/1 | 1/1 | 1/1 |
| flash-13 | 1/1 | 1/1 | 1/1 |
| flash-14 | 1/1 | 1/1 | 1/1 |
| flash-15 | 1/1 | 1/1 | 1/1 |

Units: 76; units with non-zero exit: 0 (all three columns).

`omarchy_ane_runtime_tests` is not built here (needs the ANE runtime sources) and is outside the battery.

## Why the private Mesa needed an overlay change

Mesa fd2c4a67 lowers `packHalf2x16` to round-toward-zero; GLSL leaves it implementation-defined. Shaders that used `unpackHalf2x16(packHalf2x16(x))` as an f32 to f16 round-to-nearest-even idiom (fused_chain, swiglu, fast_trio, qmm_vec ROUND_STORAGE, qmm_fma output, sdpa_decode_native_p1 scratch, cast_int f32 to f16) now use `omarchy_f16_bits_rne` (shaders/omarchy_half.h, integer RNE).

Proof of the helper: the same algorithm in C against the hardware conversion (`_Float16` cast on aarch64) over all 2^32 float32 bit patterns: 0 mismatches (NaN: any quiet NaN). On the system Mesa the default and opt-in suites are unchanged (this table, system column all green with the pinned bit-equality tests).
