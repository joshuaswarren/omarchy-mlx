# Battery on three Mesa configurations

Same binaries (tests built from the code of b445c037), one process per unit under `flock /tmp/m1-gpu.lock`, `timeout 1800`, capability-simulation profiles, flash one process per case (`battery-425.sh`).

- system: Mesa 26.3.0-devel `git-05ddb00f5c` (Borealis), system ICD
- system + opt-in: the same Mesa with `MLX_OMARCHY_QMM_VEC_TOKEN_MULTI=1 MLX_OMARCHY_GDN_GQA_BF16A=1 MLX_OMARCHY_GDN_PF_GQA=1 MLX_OMARCHY_GDN_PF_WIDE=1` for every unit
- cooperative matrix: a development Mesa build based on aurora-silicon/mesa#275, `ASAHI_AGX3B_COOPERATIVE_MATRIX=1`, `VK_DRIVER_FILES` to its ICD (the device reports `cooperative_matrix_fp32_8x8x8=1`)

Cells are passed/total doctest cases (or the exit code for units without a doctest summary). No unit logs a pipeline refusal (`MESA: error`) in any column.

In the opt-in column `omarchy_gdn_gqa_bf16a_tests` fails 2 of 10: those cases assert the default kernel pick, and the process-wide `MLX_OMARCHY_GDN_PF_WIDE=1` selects the wide kernel (638) instead of the plain one (480). Its other cases, which set the switches themselves, pass.

An earlier version of this file compared the system Mesa with an older development Mesa build on f4077931; its opt-in column set the GDN switches without the `MLX_OMARCHY_` prefix, so only the token-multi switch took effect there. This table replaces it.

| unit | system | system + opt-in | cooperative matrix |
|---|---|---|---|
| omarchy_ane_bundle_tests | 55/55 | 55/55 | 55/55 |
| omarchy_ane_h16_device_tests | 23/23 | 23/23 | 23/23 |
| omarchy_ane_h16_sections_tests | 5/5 | 5/5 | 5/5 |
| omarchy_ane_tile_layout_tests | 3/3 | 3/3 | 3/3 |
| omarchy_ane_worker_gate_tests | 27/27 | 27/27 | 27/27 |
| omarchy_ane_worker_state_tests | 12/12 | 12/12 | 12/12 |
| omarchy_ane_worker_tests | 17/17 | 17/17 | 17/17 |
| omarchy_compiled_tape_tests | 14/14 | 14/14 | 14/14 |
| omarchy_complex_ops_tests | 34/34 | 34/34 | 34/34 |
| omarchy_conv_gemm_decomp_tests | 3/3 | 3/3 | 3/3 |
| omarchy_conv_tests | 15/15 | 15/15 | 15/15 |
| omarchy_copy_offset_tests | 29/29 | 29/29 | 29/29 |
| omarchy_custom_kernel_translate_tests | 52/52 | 52/52 | 52/52 |
| omarchy_dep_tracker_tests | 12/12 | 12/12 | 12/12 |
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
| omarchy_gdn_gqa_bf16a_tests | 10/10 | 8/10 (rc=1) | 10/10 |
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
| flash-16 | 1/1 | 1/1 | 1/1 |

Units: 81; units with non-zero exit: 0 / 1 / 0.

`omarchy_ane_runtime_tests` is not built here (needs the ANE runtime sources) and is outside the battery.
