// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/compute.h"

#include <stdexcept>
#include <utility>

#include "mlx/backend/omarchy/vulkan.h"

#include "arange_u32.h"
#include "arange_bf16.h"
#include "arange_f16.h"
#include "arange_f32.h"
#include "arange_i32.h"
#include "arange_i64.h"
#include "arange_u64.h"
#include "argreduce_f32.h"
#include "argreduce_bf16.h"
#include "argreduce_f16.h"
#include "argreduce_i8.h"
#include "argreduce_u8.h"
#include "argreduce_i16.h"
#include "argreduce_u16.h"
#include "argreduce_i32.h"
#include "argreduce_u32.h"
#include "argreduce_i64.h"
#include "argreduce_u64.h"
#include "cast_int_w1_w1.h"
#include "cast_int_w1_w2.h"
#include "cast_int_w1_w4.h"
#include "cast_int_w1_w8.h"
#include "cast_int_w2_w1.h"
#include "cast_int_w2_w2.h"
#include "cast_int_w2_w4.h"
#include "cast_int_w2_w8.h"
#include "cast_int_w4_w1.h"
#include "cast_int_w4_w2.h"
#include "cast_int_w4_w4.h"
#include "cast_int_w4_w8.h"
#include "cast_int_w8_w1.h"
#include "cast_int_w8_w2.h"
#include "cast_int_w8_w4.h"
#include "cast_int_w8_w8.h"
#include "elementwise_i8.h"
#include "elementwise_u8.h"
#include "elementwise_i16.h"
#include "elementwise_u16.h"
#include "elementwise_i64.h"
#include "elementwise_u64.h"
#include "cast_bf16_f16.h"
#include "cast_bool_f32.h"
#include "cast_bool_i32.h"
#include "cast_bool_f16.h"
#include "cast_bool_bf16.h"
#include "cast_bf16_f32.h"
#include "cast_f16_f32.h"
#include "cast_f16_bf16.h"
#include "cast_f32_bf16.h"
#include "cast_f32_f16.h"
#include "cast_bf16_i32.h"
#include "cast_f16_i32.h"
#include "cast_f32_i32.h"
#include "cast_u32_f32.h"
#include "cast_i32_bf16.h"
#include "cast_i32_f16.h"
#include "cast_i32_f32.h"
#include "elementwise_bf16.h"
#include "elementwise_i32.h"
#include "elementwise_u32.h"
#include "elementwise_f16.h"
#include "elementwise_f32.h"
#include "take_bf16.h"
#include "take_f16.h"
#include "take_f32.h"
#include "take_multi_bf16.h"
#include "take_multi_f16.h"
#include "take_multi_f32.h"
#include "take_multi_u32.h"
#include "take_u32.h"
#include "take_u16.h"
#include "take_i64.h"
#include "take_c64.h"
#include "take_multi_c64.h"
#include "take_multi_u16.h"
#include "take_multi_i64.h"
#include "slice_update_reduce_bf16.h"
#include "slice_update_reduce_f16.h"
#include "slice_update_reduce_f32.h"
#include "slice_update_reduce_u32.h"
#include "scatter_triple_u32.h"
#include "scatter_triple_f16.h"
#include "scatter_u8.h"
#include "scatter_i8.h"
#include "scatter_u16.h"
#include "scatter_i16.h"
#include "scatter_multi_u8.h"
#include "scatter_multi_i8.h"
#include "scatter_multi_u16.h"
#include "scatter_multi_i16.h"
#include "scatter_triple_u8.h"
#include "scatter_triple_i8.h"
#include "scatter_triple_u16.h"
#include "scatter_triple_i16.h"
#include "scatter_triple_bf16.h"
#include "scatter_bool_triple.h"
#include "gather_axis_c64.h"
#include "scatter_c64.h"
#include "scatter_axis_c64.h"
#include "scatter_fadd_triple_f32.h"
#include "scatter_fcas_triple_f32.h"
#include "scatter_general_u32.h"
#include "scatter_general_f16.h"
#include "scatter_general_bf16.h"
#include "scatter_general_bool.h"
#include "scatter_general_u8.h"
#include "scatter_general_i8.h"
#include "scatter_general_u16.h"
#include "scatter_general_i16.h"
#include "scatter_fadd_general_f32.h"
#include "scatter_fcas_general_f32.h"
#include "copy_general_bf16.h"
#include "copy_general_f16.h"
#include "slice_update_pair_f16.h"
#include "copy_general_f32.h"
#include "copy_general_u32.h"
#include "copy_general_bool.h"
#include "copy_general_u8.h"
#include "copy_general_u16.h"
#include "copy_general_u64.h"
#include "fill_bf16.h"
#include "fill_f16.h"
#include "fill_f32.h"
#include "matmul_bf16.h"
#include "compare_bf16.h"
#include "compare_f16.h"
#include "compare_f32.h"
#include "compare_complex.h"
#include "compare_i32.h"
#include "compare_i64.h"
#include "compare_u32.h"
#include "compare_i8.h"
#include "compare_u8.h"
#include "compare_i16.h"
#include "compare_u16.h"
#include "matmul_f16.h"
#include "matmul_f32.h"
#include "matmul_f32_coopmat.h"
#include "matmul_complex64.h"
#include "matmul_vec_bf16.h"
#include "matmul_vec_f16.h"
#include "matmul_vec_f32.h"
#include "matmul_vec_multi_bf16.h"
#include "select_bf16.h"
#include "select_f16.h"
#include "select_f32.h"
#include "select_i32.h"
#include "select_bool.h"
#include "select_complex64.h"
#include "reduce_bf16.h"
#include "reduce_f16.h"
#include "reduce_f32.h"
#include "hadamard_bf16.h"
#include "hadamard_f16.h"
#include "hadamard_f32.h"
#include "anyall_bf16.h"
#include "anyall_f16.h"
#include "anyall_f32.h"
#include "anyall_i32.h"
#include "anyall_u32.h"
#include "anyall_bool.h"
#include "reduce_general_bf16.h"
#include "reduce_general_f16.h"
#include "reduce_general_f32.h"
#include "reduce_general_i32.h"
#include "reduce_general_u32.h"
#include "reduce_general_bool.h"
#include "reduce_general_i8.h"
#include "reduce_general_u8.h"
#include "reduce_general_i16.h"
#include "reduce_general_u16.h"
#include "reduce_general_i64.h"
#include "reduce_general_u64.h"
#include "reduce_general_complex.h"
#include "logsumexp_bf16.h"
#include "logsumexp_f16.h"
#include "logsumexp_f32.h"
#include "softmax_bf16.h"
#include "softmax_f16.h"
#include "softmax_f32.h"
#include "sort_bf16.h"
#include "sort_f16.h"
#include "sort_f32.h"
#include "sort_c64.h"
#include "argsort_bf16.h"
#include "argsort_f16.h"
#include "argsort_f32.h"
#include "argsort_c64.h"
#include "sort_i32.h"
#include "sort_u32.h"
#include "sort_i8.h"
#include "sort_u8.h"
#include "sort_i16.h"
#include "sort_u16.h"
#include "argsort_i32.h"
#include "argsort_u32.h"
#include "argsort_i8.h"
#include "argsort_u8.h"
#include "argsort_i16.h"
#include "argsort_u16.h"
#include "sort_merge_bf16.h"
#include "sort_merge_f16.h"
#include "sort_merge_f32.h"
#include "sort_merge_i32.h"
#include "sort_merge_u32.h"
#include "argsort_merge_bf16.h"
#include "argsort_merge_f16.h"
#include "argsort_merge_f32.h"
#include "argsort_merge_c64.h"
#include "argsort_merge_i32.h"
#include "argsort_merge_u32.h"
#include "logical_or_bool.h"
#include "compare_bool.h"
#include "scan_bf16.h"
#include "fused_chain_f32.h"
#include "fused_chain_f16.h"
#include "fused_chain_bf16.h"
#include "fast_rope_bf16.h"
#include "fast_rope_f16.h"
#include "fast_rope_f32.h"
#include "fast_rope_freqs_bf16.h"
#include "fast_rope_freqs_f16.h"
#include "fast_rope_freqs_f32.h"
#include "scan_f16.h"
#include "scan_f32.h"
#include "scan_general_bf16.h"
#include "scan_general_f16.h"
#include "scan_general_f32.h"
#include "scan_general_i32.h"
#include "scan_general_u32.h"
#include "scan_general_bool.h"
#include "scan_general_i8.h"
#include "scan_general_u8.h"
#include "scan_general_i16.h"
#include "scan_general_u16.h"
#include "scan_general_i64.h"
#include "scan_general_u64.h"
#include "scan_general_complex.h"
#include "searchsorted_bf16.h"
#include "searchsorted_f16.h"
#include "searchsorted_f32.h"
#include "searchsorted_i32.h"
#include "searchsorted_u32.h"
#include "random_bits_u32.h"
#include "qmm_bf16.h"
#include "qmm_f16.h"
#include "qmm_f32.h"
#include "qmm_vec_bf16.h"
#include "qmm_vec_f16.h"
#include "qmm_vec_f32.h"
#include "qmm_vec_subgroup_bf16.h"
#include "qmm_vec_subgroup_f16.h"
#include "qmm_vec_subgroup_f32.h"
#include "qmm_vec_q4_word_bf16.h"
#include "qmm_vec_q4_word_f16.h"
#include "qmm_vec_q4_word_f32.h"
#include "qmm_vec_q4_word_subgroup_bf16.h"
#include "qmm_vec_q4_word_subgroup_f16.h"
#include "qmm_vec_q4_word_subgroup_f32.h"
#include "qmm_vec_q4_multi_bf16.h"
#include "qmm_vec_q4_multi_f16.h"
#include "qmm_vec_q4_multi_f32.h"
#include "qmm_vec_q4_multi_subgroup_bf16.h"
#include "qmm_vec_q4_multi_subgroup_f16.h"
#include "qmm_vec_q4_multi_subgroup_f32.h"
#include "sdpa_decode_native_f16.h"
#include "sdpa_decode_native_bf16.h"
#include "sdpa_decode_native_bf16_hd256.h"
#include "sdpa_decode_native_p1_f16.h"
#include "sdpa_decode_native_p2_f16.h"
#include "gated_delta_decode_bf16.h"
#include "gated_delta_prefill_bf16.h"
#include "gated_delta_prefill_coopmat_bf16.h"
#include "fast_norm_gated_bf16.h"
#include "gdn_conv_decode_bf16.h"
#include "qmm_tile_bf16.h"
#include "qmm_tile_f16.h"
#include "qmm_tile_rb_f16.h"
#include "qmm_tile_rb_precise_f16.h"
#include "qmm_coopmat_f16.h"
#include "qmm_coopmat_m16_f16.h"
#include "qmm_coopmat_bf16.h"
#include "qmm_coopmat_m16_bf16.h"
#include "qmm_coopmat_x32.h"
#include "qmm_coopmat_m16_x32.h"
#include "gated_delta_decode_bf16.h"
#include "gated_delta_prefill_bf16.h"
#include "qmm_fma_precise_f16.h"
#include "matmul_fma_bf16.h"
#include "qmm_fma_f16.h"
#include "matmul_fma_bf16.h"
#include "binary_vec_f16.h"
#include "binary_vec_bf16.h"
#include "matmul_rb_f16.h"
#include "swiglu_f16.h"
#include "swiglu_bf16.h"
#include "fast_trio_norm_f16.h"
#include "fast_trio_rope_pair_f16.h"
#include "fast_trio_swiglu_f16.h"
#include "matmul_f32_coopmat_bf16.h"
#include "qmm_tile_f32.h"
#include "dequant_f32.h"
#include "dequant_f16.h"
#include "quantize_f32.h"
#include "quantize_f16.h"
#include "dequant_bf16.h"
#include "quantize_bf16.h"
#include "quantize_fp_f32.h"
#include "quantize_fp_f16.h"
#include "quantize_fp_bf16.h"
#include "dequant_fp_f32.h"
#include "dequant_fp_f16.h"
#include "dequant_fp_bf16.h"
#include "qmm_fp_f32.h"
#include "qmm_fp_f16.h"
#include "qmm_fp_bf16.h"
#include "qmm_vec_fp_f32.h"
#include "qmm_vec_fp_f16.h"
#include "qmm_vec_fp_bf16.h"
#include "qmm_vec_subgroup_fp_f32.h"
#include "qmm_vec_subgroup_fp_f16.h"
#include "qmm_vec_subgroup_fp_bf16.h"
#include "qmm_tile_fp_f32.h"
#include "qmm_tile_fp_f16.h"
#include "qmm_tile_fp_bf16.h"
#include "conv_bf16.h"
#include "clear_u32.h"
#include "gather_axis_bf16.h"
#include "gather_axis_f16.h"
#include "gather_axis_i64.h"
#include "gather_axis_u32.h"
#include "masked_scatter_bf16.h"
#include "masked_scatter_f16.h"
#include "masked_scatter_u32.h"
#include "scatter_axis_bf16.h"
#include "scatter_axis_f16.h"
#include "scatter_axis_u32.h"
#include "scatter_bf16.h"
#include "scatter_f16.h"
#include "scatter_multi_bf16.h"
#include "scatter_multi_f16.h"
#include "scatter_bool.h"
#include "scatter_bool_multi.h"
#include "scatter_fadd_bf16.h"
#include "scatter_fadd_f16.h"
#include "scatter_fadd_f32.h"
#include "scatter_fadd_multi_f32.h"
#include "scatter_fcas_bf16.h"
#include "scatter_fcas_f16.h"
#include "scatter_fcas_f32.h"
#include "scatter_fcas_multi_f32.h"
#include "scatter_axis_bool.h"
#include "scatter_axis_fadd_bf16.h"
#include "scatter_axis_fadd_f16.h"
#include "scatter_axis_fadd_f32.h"
#include "scatter_axis_fcas_bf16.h"
#include "scatter_axis_fcas_f16.h"
#include "scatter_axis_fcas_f32.h"
#include "scatter_multi_u32.h"
#include "scatter_u32.h"
#include "conv_f16.h"
#include "conv_f32.h"
#include "block_mask_f32.h"
#include "gather_mm_f32.h"
#include "gather_mm_f16.h"
#include "gather_mm_bf16.h"
#include "segmented_mm_f32.h"
#include "segmented_mm_f16.h"
#include "segmented_mm_bf16.h"
#include "gather_qmm_f32.h"
#include "gather_qmm_f16.h"
#include "gather_qmm_bf16.h"
#include "gather_qmm_nb_f32.h"
#include "gather_qmm_nb_f16.h"
#include "gather_qmm_nb_bf16.h"
#include "gather_qmm_nb_fp_f32.h"
#include "gather_qmm_nb_fp_f16.h"
#include "gather_qmm_nb_fp_bf16.h"
#include "gather_qmm_nb_fp_hgs_f32.h"
#include "gather_qmm_nb_fp_hgs_f16.h"
#include "gather_qmm_nb_fp_hgs_bf16.h"
#include "fft_f32.h"
#include "fft_real_f32.h"
#include "fft_stage_f32.h"
#include "fast_rms_norm_f32.h"
#include "fast_rms_norm_f16.h"
#include "fast_rms_norm_bf16.h"
#include "fast_layer_norm_f32.h"
#include "fast_layer_norm_f16.h"
#include "fast_layer_norm_bf16.h"
#include "fast_rms_norm_vjp_dx_f32.h"
#include "fast_rms_norm_vjp_dx_f16.h"
#include "fast_rms_norm_vjp_dx_bf16.h"
#include "fast_layer_norm_vjp_dx_f32.h"
#include "fast_layer_norm_vjp_dx_f16.h"
#include "fast_layer_norm_vjp_dx_bf16.h"
#include "fast_rms_norm_vjp_dw_f32.h"
#include "fast_rms_norm_vjp_dw_f16.h"
#include "fast_rms_norm_vjp_dw_bf16.h"
#include "fast_layer_norm_vjp_dw_f32.h"
#include "fast_layer_norm_vjp_dw_f16.h"
#include "fast_layer_norm_vjp_dw_bf16.h"
#include "fast_rms_norm_vjp_dw_reduce_f32.h"
#include "fast_rms_norm_vjp_dw_reduce_f16.h"
#include "fast_rms_norm_vjp_dw_reduce_bf16.h"
#include "fast_cross_entropy_vjp_f32.h"
#include "fast_cross_entropy_vjp_f16.h"
#include "fast_cross_entropy_vjp_bf16.h"
#include "fast_cross_entropy_f32.h"
#include "fast_cross_entropy_f16.h"
#include "fast_cross_entropy_bf16.h"
#include "fp8_to_f32.h"
#include "fp8_to_f16.h"
#include "fp8_to_bf16.h"
#include "fp8_from_f32.h"
#include "fp8_from_f16.h"
#include "fp8_from_bf16.h"
#include "linalg_cholesky_f32.h"
#include "linalg_inverse_f32.h"
#include "linalg_lu_f32.h"
#include "linalg_qr_f32.h"
#include "linalg_eigh_f32.h"
#include "linalg_eig_f32.h"
#include "linalg_svd_f32.h"
#include "linalg_svd_finalize_f32.h"
#include "complex_elementwise.h"
#include "complex_real.h"
#include "complex_imag.h"
#include "complex_abs.h"
#include "complex_abs_as_complex.h"
#include "cast_f32_c64.h"
#include "cast_i32_c64.h"
#include "cast_u32_c64.h"
#include "cast_bool_c64.h"
#include "cast_f16_c64.h"
#include "cast_bf16_c64.h"
#include "cast_c64_f32.h"
#include "fill_c64.h"
#include "copy_general_c64.h"
#include "fill_u64.h"
#include "fill_u16.h"
#include "compare_u64.h"

namespace mlx::core::omarchy {

namespace {

using ShaderBytes = std::pair<const unsigned char*, size_t>;

ShaderBytes shader_bytes(ComputeKernel kernel) {
  using namespace shaders;
  switch (kernel) {
    case ComputeKernel::ElementwiseF32:
      return {elementwise_f32, elementwise_f32_size};
    case ComputeKernel::ElementwiseF16:
      return {elementwise_f16, elementwise_f16_size};
    case ComputeKernel::ElementwiseBF16:
      return {elementwise_bf16, elementwise_bf16_size};
    case ComputeKernel::CastF16F32:
      return {cast_f16_f32, cast_f16_f32_size};
    case ComputeKernel::CastBoolF32:
      return {cast_bool_f32, cast_bool_f32_size};
    case ComputeKernel::CastBoolI32:
      return {cast_bool_i32, cast_bool_i32_size};
    case ComputeKernel::CastBoolF16:
      return {cast_bool_f16, cast_bool_f16_size};
    case ComputeKernel::CastBoolBF16:
      return {cast_bool_bf16, cast_bool_bf16_size};
    case ComputeKernel::CastF32F16:
      return {cast_f32_f16, cast_f32_f16_size};
    case ComputeKernel::CastBF16F32:
      return {cast_bf16_f32, cast_bf16_f32_size};
    case ComputeKernel::CastF32BF16:
      return {cast_f32_bf16, cast_f32_bf16_size};
    case ComputeKernel::CastBF16F16:
      return {cast_bf16_f16, cast_bf16_f16_size};
    case ComputeKernel::CastF16BF16:
      return {cast_f16_bf16, cast_f16_bf16_size};
    case ComputeKernel::CastI32F32:
      return {cast_i32_f32, cast_i32_f32_size};
    case ComputeKernel::CastF32I32:
      return {cast_f32_i32, cast_f32_i32_size};
    case ComputeKernel::CastI32F16:
      return {cast_i32_f16, cast_i32_f16_size};
    case ComputeKernel::CastF16I32:
      return {cast_f16_i32, cast_f16_i32_size};
    case ComputeKernel::CastI32BF16:
      return {cast_i32_bf16, cast_i32_bf16_size};
    case ComputeKernel::CastBF16I32:
      return {cast_bf16_i32, cast_bf16_i32_size};
    case ComputeKernel::CastU32F32:
      return {cast_u32_f32, cast_u32_f32_size};
    case ComputeKernel::ArgReduceF32:
      return {argreduce_f32, argreduce_f32_size};
    case ComputeKernel::ArgReduceF16:
      return {argreduce_f16, argreduce_f16_size};
    case ComputeKernel::ArgReduceBF16:
      return {argreduce_bf16, argreduce_bf16_size};
    case ComputeKernel::ArgReduceI8:
      return {argreduce_i8, argreduce_i8_size};
    case ComputeKernel::ArgReduceU8:
      return {argreduce_u8, argreduce_u8_size};
    case ComputeKernel::ArgReduceI16:
      return {argreduce_i16, argreduce_i16_size};
    case ComputeKernel::ArgReduceU16:
      return {argreduce_u16, argreduce_u16_size};
    case ComputeKernel::ArgReduceI32:
      return {argreduce_i32, argreduce_i32_size};
    case ComputeKernel::ArgReduceU32:
      return {argreduce_u32, argreduce_u32_size};
    case ComputeKernel::ArgReduceI64:
      return {argreduce_i64, argreduce_i64_size};
    case ComputeKernel::ArgReduceU64:
      return {argreduce_u64, argreduce_u64_size};
    case ComputeKernel::ArangeF32:
      return {arange_f32, arange_f32_size};
    case ComputeKernel::ArangeF16:
      return {arange_f16, arange_f16_size};
    case ComputeKernel::ArangeBF16:
      return {arange_bf16, arange_bf16_size};
    case ComputeKernel::ArangeI32:
      return {arange_i32, arange_i32_size};
    case ComputeKernel::ArangeU32:
      return {arange_u32, arange_u32_size};
    case ComputeKernel::ArangeI64:
      return {arange_i64, arange_i64_size};
    case ComputeKernel::ArangeU64:
      return {arange_u64, arange_u64_size};
    case ComputeKernel::ReduceF32:
      return {reduce_f32, reduce_f32_size};
    case ComputeKernel::ReduceF16:
      return {reduce_f16, reduce_f16_size};
    case ComputeKernel::ReduceBF16:
      return {reduce_bf16, reduce_bf16_size};
    case ComputeKernel::MatmulF32:
      return {matmul_f32, matmul_f32_size};
    case ComputeKernel::MatmulF32Coopmat:
      return {matmul_f32_coopmat, matmul_f32_coopmat_size};
    case ComputeKernel::MatmulF16:
      return {matmul_f16, matmul_f16_size};
    case ComputeKernel::MatmulBF16:
      return {matmul_bf16, matmul_bf16_size};
    case ComputeKernel::FillF32:
      return {fill_f32, fill_f32_size};
    case ComputeKernel::FillF16:
      return {fill_f16, fill_f16_size};
    case ComputeKernel::FillBF16:
      return {fill_bf16, fill_bf16_size};
    case ComputeKernel::MatmulVecF32:
      return {matmul_vec_f32, matmul_vec_f32_size};
    case ComputeKernel::MatmulVecF16:
      return {matmul_vec_f16, matmul_vec_f16_size};
    case ComputeKernel::MatmulVecBF16:
      return {matmul_vec_bf16, matmul_vec_bf16_size};
    case ComputeKernel::MatmulVecMultiBF16:
      return {matmul_vec_multi_bf16, matmul_vec_multi_bf16_size};
    case ComputeKernel::SoftmaxF32:
      return {softmax_f32, softmax_f32_size};
    case ComputeKernel::SoftmaxF16:
      return {softmax_f16, softmax_f16_size};
    case ComputeKernel::SoftmaxBF16:
      return {softmax_bf16, softmax_bf16_size};
    case ComputeKernel::LogSumExpF32:
      return {logsumexp_f32, logsumexp_f32_size};
    case ComputeKernel::LogSumExpF16:
      return {logsumexp_f16, logsumexp_f16_size};
    case ComputeKernel::LogSumExpBF16:
      return {logsumexp_bf16, logsumexp_bf16_size};
    case ComputeKernel::SelectF32:
      return {select_f32, select_f32_size};
    case ComputeKernel::SelectF16:
      return {select_f16, select_f16_size};
    case ComputeKernel::SelectBF16:
      return {select_bf16, select_bf16_size};
    case ComputeKernel::SelectI32:
      return {select_i32, select_i32_size};
    case ComputeKernel::SelectBool:
      return {select_bool, select_bool_size};
    case ComputeKernel::SelectComplex64:
      return {select_complex64, select_complex64_size};
    case ComputeKernel::SelectI64:
      // int64/uint64 select rides the same two-raw-words u2 variant as
      // complex64: a select copies bits per element and both dtypes are
      // exactly two 32-bit words, so no separate SPIR-V is needed.
      return {select_complex64, select_complex64_size};
    case ComputeKernel::CompareF32:
      return {compare_f32, compare_f32_size};
    case ComputeKernel::CompareF16:
      return {compare_f16, compare_f16_size};
    case ComputeKernel::CompareBF16:
      return {compare_bf16, compare_bf16_size};
    case ComputeKernel::CompareI32:
      return {compare_i32, compare_i32_size};
    case ComputeKernel::CompareU32:
      return {compare_u32, compare_u32_size};
    case ComputeKernel::CompareI64:
      return {compare_i64, compare_i64_size};
    case ComputeKernel::CompareComplex:
      return {compare_complex, compare_complex_size};
    case ComputeKernel::LogicalOrBool:
      return {logical_or_bool, logical_or_bool_size};
    case ComputeKernel::CompareBool:
      return {compare_bool, compare_bool_size};
    case ComputeKernel::ElementwiseI32:
      return {elementwise_i32, elementwise_i32_size};
    case ComputeKernel::ElementwiseU32:
      return {elementwise_u32, elementwise_u32_size};
    case ComputeKernel::ScanF32:
      return {scan_f32, scan_f32_size};
    case ComputeKernel::ScanF16:
      return {scan_f16, scan_f16_size};
    case ComputeKernel::ScanBF16:
      return {scan_bf16, scan_bf16_size};
    case ComputeKernel::SearchSortedF32:
      return {searchsorted_f32, searchsorted_f32_size};
    case ComputeKernel::SearchSortedF16:
      return {searchsorted_f16, searchsorted_f16_size};
    case ComputeKernel::SearchSortedBF16:
      return {searchsorted_bf16, searchsorted_bf16_size};
    case ComputeKernel::SearchSortedI32:
      return {searchsorted_i32, searchsorted_i32_size};
    case ComputeKernel::SearchSortedU32:
      return {searchsorted_u32, searchsorted_u32_size};
    case ComputeKernel::ReduceGeneralF32:
      return {reduce_general_f32, reduce_general_f32_size};
    case ComputeKernel::ReduceGeneralF16:
      return {reduce_general_f16, reduce_general_f16_size};
    case ComputeKernel::ReduceGeneralBF16:
      return {reduce_general_bf16, reduce_general_bf16_size};
    case ComputeKernel::ReduceGeneralI32:
      return {reduce_general_i32, reduce_general_i32_size};
    case ComputeKernel::ReduceGeneralU32:
      return {reduce_general_u32, reduce_general_u32_size};
    case ComputeKernel::ReduceGeneralBool:
      return {reduce_general_bool, reduce_general_bool_size};
    case ComputeKernel::ReduceGeneralI8:
      return {reduce_general_i8, reduce_general_i8_size};
    case ComputeKernel::ReduceGeneralU8:
      return {reduce_general_u8, reduce_general_u8_size};
    case ComputeKernel::ReduceGeneralI16:
      return {reduce_general_i16, reduce_general_i16_size};
    case ComputeKernel::ReduceGeneralU16:
      return {reduce_general_u16, reduce_general_u16_size};
    case ComputeKernel::ReduceGeneralI64:
      return {reduce_general_i64, reduce_general_i64_size};
    case ComputeKernel::ReduceGeneralU64:
      return {reduce_general_u64, reduce_general_u64_size};
    case ComputeKernel::ReduceGeneralComplex:
      return {reduce_general_complex, reduce_general_complex_size};
    case ComputeKernel::AnyAllF32:
      return {anyall_f32, anyall_f32_size};
    case ComputeKernel::AnyAllF16:
      return {anyall_f16, anyall_f16_size};
    case ComputeKernel::AnyAllBF16:
      return {anyall_bf16, anyall_bf16_size};
    case ComputeKernel::AnyAllI32:
      return {anyall_i32, anyall_i32_size};
    case ComputeKernel::AnyAllU32:
      return {anyall_u32, anyall_u32_size};
    case ComputeKernel::AnyAllBool:
      return {anyall_bool, anyall_bool_size};
    case ComputeKernel::ScanGeneralF32:
      return {scan_general_f32, scan_general_f32_size};
    case ComputeKernel::ScanGeneralF16:
      return {scan_general_f16, scan_general_f16_size};
    case ComputeKernel::ScanGeneralBF16:
      return {scan_general_bf16, scan_general_bf16_size};
    case ComputeKernel::ScanGeneralI32:
      return {scan_general_i32, scan_general_i32_size};
    case ComputeKernel::ScanGeneralU32:
      return {scan_general_u32, scan_general_u32_size};
    case ComputeKernel::ScanGeneralBool:
      return {scan_general_bool, scan_general_bool_size};
    case ComputeKernel::ScanGeneralI8:
      return {scan_general_i8, scan_general_i8_size};
    case ComputeKernel::ScanGeneralU8:
      return {scan_general_u8, scan_general_u8_size};
    case ComputeKernel::ScanGeneralI16:
      return {scan_general_i16, scan_general_i16_size};
    case ComputeKernel::ScanGeneralU16:
      return {scan_general_u16, scan_general_u16_size};
    case ComputeKernel::ScanGeneralI64:
      return {scan_general_i64, scan_general_i64_size};
    case ComputeKernel::ScanGeneralU64:
      return {scan_general_u64, scan_general_u64_size};
    case ComputeKernel::ScanGeneralComplex:
      return {scan_general_complex, scan_general_complex_size};
    case ComputeKernel::HadamardF32:
      return {hadamard_f32, hadamard_f32_size};
    case ComputeKernel::HadamardF16:
      return {hadamard_f16, hadamard_f16_size};
    case ComputeKernel::HadamardBF16:
      return {hadamard_bf16, hadamard_bf16_size};
    case ComputeKernel::TakeF32:
      return {take_f32, take_f32_size};
    case ComputeKernel::TakeF16:
      return {take_f16, take_f16_size};
    case ComputeKernel::TakeBF16:
      return {take_bf16, take_bf16_size};
    case ComputeKernel::TakeU32:
      return {take_u32, take_u32_size};
    case ComputeKernel::TakeU16:
      return {take_u16, take_u16_size};
    case ComputeKernel::TakeI64:
      return {take_i64, take_i64_size};
    case ComputeKernel::TakeComplex64:
      return {take_c64, take_c64_size};
    case ComputeKernel::TakeMultiF32:
      return {take_multi_f32, take_multi_f32_size};
    case ComputeKernel::TakeMultiF16:
      return {take_multi_f16, take_multi_f16_size};
    case ComputeKernel::TakeMultiBF16:
      return {take_multi_bf16, take_multi_bf16_size};
    case ComputeKernel::TakeMultiU32:
      return {take_multi_u32, take_multi_u32_size};
    case ComputeKernel::TakeMultiU16:
      return {take_multi_u16, take_multi_u16_size};
    case ComputeKernel::TakeMultiI64:
      return {take_multi_i64, take_multi_i64_size};
    case ComputeKernel::TakeMultiComplex64:
      return {take_multi_c64, take_multi_c64_size};
    case ComputeKernel::SliceUpdateReduceF32:
      return {slice_update_reduce_f32, slice_update_reduce_f32_size};
    case ComputeKernel::SliceUpdateReduceF16:
      return {slice_update_reduce_f16, slice_update_reduce_f16_size};
    case ComputeKernel::SliceUpdateReduceBF16:
      return {slice_update_reduce_bf16, slice_update_reduce_bf16_size};
    case ComputeKernel::SliceUpdateReduceU32:
      return {slice_update_reduce_u32, slice_update_reduce_u32_size};
    case ComputeKernel::ScatterTripleU32:
      return {scatter_triple_u32, scatter_triple_u32_size};
    case ComputeKernel::ScatterTripleF16:
      return {scatter_triple_f16, scatter_triple_f16_size};
    case ComputeKernel::ScatterTripleBF16:
      return {scatter_triple_bf16, scatter_triple_bf16_size};
    case ComputeKernel::ScatterBoolTriple:
      return {scatter_bool_triple, scatter_bool_triple_size};
    case ComputeKernel::ScatterGeneralU32:
      return {scatter_general_u32, scatter_general_u32_size};
    case ComputeKernel::ScatterGeneralF16:
      return {scatter_general_f16, scatter_general_f16_size};
    case ComputeKernel::ScatterGeneralBF16:
      return {scatter_general_bf16, scatter_general_bf16_size};
    case ComputeKernel::ScatterGeneralBool:
      return {scatter_general_bool, scatter_general_bool_size};
    case ComputeKernel::ScatterGeneralU8:
      return {scatter_general_u8, scatter_general_u8_size};
    case ComputeKernel::ScatterGeneralI8:
      return {scatter_general_i8, scatter_general_i8_size};
    case ComputeKernel::ScatterGeneralU16:
      return {scatter_general_u16, scatter_general_u16_size};
    case ComputeKernel::ScatterGeneralI16:
      return {scatter_general_i16, scatter_general_i16_size};
    case ComputeKernel::CopyGeneralF32:
      return {copy_general_f32, copy_general_f32_size};
    case ComputeKernel::CopyGeneralF16:
      return {copy_general_f16, copy_general_f16_size};
    case ComputeKernel::SliceUpdatePairF16:
      return {slice_update_pair_f16, slice_update_pair_f16_size};
    case ComputeKernel::CopyGeneralBF16:
      return {copy_general_bf16, copy_general_bf16_size};
    case ComputeKernel::CopyGeneralU32:
      return {copy_general_u32, copy_general_u32_size};
    case ComputeKernel::CopyGeneralBool:
      return {copy_general_bool, copy_general_bool_size};
    case ComputeKernel::CopyGeneralU8:
      return {copy_general_u8, copy_general_u8_size};
    case ComputeKernel::CopyGeneralU16:
      return {copy_general_u16, copy_general_u16_size};
    case ComputeKernel::CopyGeneralU64:
      return {copy_general_u64, copy_general_u64_size};
    case ComputeKernel::ArgSortF32:
      return {argsort_f32, argsort_f32_size};
    case ComputeKernel::ArgSortF16:
      return {argsort_f16, argsort_f16_size};
    case ComputeKernel::ArgSortBF16:
      return {argsort_bf16, argsort_bf16_size};
    case ComputeKernel::ArgSortC64:
      return {argsort_c64, argsort_c64_size};
    case ComputeKernel::SortI32:
      return {sort_i32, sort_i32_size};
    case ComputeKernel::SortU32:
      return {sort_u32, sort_u32_size};
    case ComputeKernel::SortI8:
      return {sort_i8, sort_i8_size};
    case ComputeKernel::SortU8:
      return {sort_u8, sort_u8_size};
    case ComputeKernel::SortI16:
      return {sort_i16, sort_i16_size};
    case ComputeKernel::SortU16:
      return {sort_u16, sort_u16_size};
    case ComputeKernel::ArgSortI32:
      return {argsort_i32, argsort_i32_size};
    case ComputeKernel::ArgSortU32:
      return {argsort_u32, argsort_u32_size};
    case ComputeKernel::ArgSortI8:
      return {argsort_i8, argsort_i8_size};
    case ComputeKernel::ArgSortU8:
      return {argsort_u8, argsort_u8_size};
    case ComputeKernel::ArgSortI16:
      return {argsort_i16, argsort_i16_size};
    case ComputeKernel::ArgSortU16:
      return {argsort_u16, argsort_u16_size};
    case ComputeKernel::SortMergeF32:
      return {sort_merge_f32, sort_merge_f32_size};
    case ComputeKernel::SortMergeF16:
      return {sort_merge_f16, sort_merge_f16_size};
    case ComputeKernel::SortMergeBF16:
      return {sort_merge_bf16, sort_merge_bf16_size};
    case ComputeKernel::SortMergeI32:
      return {sort_merge_i32, sort_merge_i32_size};
    case ComputeKernel::SortMergeU32:
      return {sort_merge_u32, sort_merge_u32_size};
    case ComputeKernel::ArgSortMergeF32:
      return {argsort_merge_f32, argsort_merge_f32_size};
    case ComputeKernel::ArgSortMergeF16:
      return {argsort_merge_f16, argsort_merge_f16_size};
    case ComputeKernel::ArgSortMergeBF16:
      return {argsort_merge_bf16, argsort_merge_bf16_size};
    case ComputeKernel::ArgSortMergeC64:
      return {argsort_merge_c64, argsort_merge_c64_size};
    case ComputeKernel::ArgSortMergeI32:
      return {argsort_merge_i32, argsort_merge_i32_size};
    case ComputeKernel::ArgSortMergeU32:
      return {argsort_merge_u32, argsort_merge_u32_size};
    case ComputeKernel::RandomBitsU32:
      return {random_bits_u32, random_bits_u32_size};
    case ComputeKernel::QmmF32:
      return {qmm_f32, qmm_f32_size};
    case ComputeKernel::GatherAxisU32:
      return {gather_axis_u32, gather_axis_u32_size};
    case ComputeKernel::GatherAxisI64:
      return {gather_axis_i64, gather_axis_i64_size};
    case ComputeKernel::GatherAxisF16:
      return {gather_axis_f16, gather_axis_f16_size};
    case ComputeKernel::GatherAxisBF16:
      return {gather_axis_bf16, gather_axis_bf16_size};
    case ComputeKernel::GatherAxisComplex64:
      return {gather_axis_c64, gather_axis_c64_size};
    case ComputeKernel::ScatterU32:
      return {scatter_u32, scatter_u32_size};
    case ComputeKernel::ScatterF16:
      return {scatter_f16, scatter_f16_size};
    case ComputeKernel::ScatterBF16:
      return {scatter_bf16, scatter_bf16_size};
    case ComputeKernel::ScatterComplex64:
      return {scatter_c64, scatter_c64_size};
    case ComputeKernel::ScatterMultiU32:
      return {scatter_multi_u32, scatter_multi_u32_size};
    case ComputeKernel::ScatterMultiF16:
      return {scatter_multi_f16, scatter_multi_f16_size};
    case ComputeKernel::ScatterMultiBF16:
      return {scatter_multi_bf16, scatter_multi_bf16_size};
    case ComputeKernel::ScatterAxisU32:
      return {scatter_axis_u32, scatter_axis_u32_size};
    case ComputeKernel::ScatterAxisF16:
      return {scatter_axis_f16, scatter_axis_f16_size};
    case ComputeKernel::ScatterAxisBF16:
      return {scatter_axis_bf16, scatter_axis_bf16_size};
    case ComputeKernel::ScatterAxisComplex64:
      return {scatter_axis_c64, scatter_axis_c64_size};
    case ComputeKernel::MaskedScatterU32:
      return {masked_scatter_u32, masked_scatter_u32_size};
    case ComputeKernel::MaskedScatterF16:
      return {masked_scatter_f16, masked_scatter_f16_size};
    case ComputeKernel::MaskedScatterBF16:
      return {masked_scatter_bf16, masked_scatter_bf16_size};
    case ComputeKernel::ClearU32:
      return {clear_u32, clear_u32_size};
    case ComputeKernel::QmmF16:
      return {qmm_f16, qmm_f16_size};
    case ComputeKernel::QmmBF16:
      return {qmm_bf16, qmm_bf16_size};
    case ComputeKernel::DequantF32:
      return {dequant_f32, dequant_f32_size};
    case ComputeKernel::DequantF16:
      return {dequant_f16, dequant_f16_size};
    case ComputeKernel::QuantizeF32:
      return {quantize_f32, quantize_f32_size};
    case ComputeKernel::QuantizeF16:
      return {quantize_f16, quantize_f16_size};
    case ComputeKernel::QuantizeBF16:
      return {quantize_bf16, quantize_bf16_size};
    case ComputeKernel::DequantBF16:
      return {dequant_bf16, dequant_bf16_size};
    case ComputeKernel::ConvF32:
      return {conv_f32, conv_f32_size};
    case ComputeKernel::ConvF16:
      return {conv_f16, conv_f16_size};
    case ComputeKernel::ConvBF16:
      return {conv_bf16, conv_bf16_size};
    case ComputeKernel::SortF32:
      return {sort_f32, sort_f32_size};
    case ComputeKernel::SortF16:
      return {sort_f16, sort_f16_size};
    case ComputeKernel::SortBF16:
      return {sort_bf16, sort_bf16_size};
    case ComputeKernel::SortC64:
      return {sort_c64, sort_c64_size};
    case ComputeKernel::BlockMaskF32:
      return {block_mask_f32, block_mask_f32_size};
    case ComputeKernel::GatherMmF32:
      return {gather_mm_f32, gather_mm_f32_size};
    case ComputeKernel::GatherMmF16:
      return {gather_mm_f16, gather_mm_f16_size};
    case ComputeKernel::GatherMmBF16:
      return {gather_mm_bf16, gather_mm_bf16_size};
    case ComputeKernel::SegmentedMmF32:
      return {segmented_mm_f32, segmented_mm_f32_size};
    case ComputeKernel::SegmentedMmF16:
      return {segmented_mm_f16, segmented_mm_f16_size};
    case ComputeKernel::SegmentedMmBF16:
      return {segmented_mm_bf16, segmented_mm_bf16_size};
    case ComputeKernel::GatherQmmF32:
      return {gather_qmm_f32, gather_qmm_f32_size};
    case ComputeKernel::GatherQmmF16:
      return {gather_qmm_f16, gather_qmm_f16_size};
    case ComputeKernel::GatherQmmBF16:
      return {gather_qmm_bf16, gather_qmm_bf16_size};
    case ComputeKernel::GatherQmmNbF32:
      return {gather_qmm_nb_f32, gather_qmm_nb_f32_size};
    case ComputeKernel::GatherQmmNbF16:
      return {gather_qmm_nb_f16, gather_qmm_nb_f16_size};
    case ComputeKernel::FftF32:
      return {fft_f32, fft_f32_size};
    case ComputeKernel::FftRealF32:
      return {fft_real_f32, fft_real_f32_size};
    case ComputeKernel::FftStageF32:
      return {fft_stage_f32, fft_stage_f32_size};
    case ComputeKernel::GatherQmmNbBF16:
      return {gather_qmm_nb_bf16, gather_qmm_nb_bf16_size};
    case ComputeKernel::FastRmsNormF32:
      return {fast_rms_norm_f32, fast_rms_norm_f32_size};
    case ComputeKernel::FastRmsNormF16:
      return {fast_rms_norm_f16, fast_rms_norm_f16_size};
    case ComputeKernel::FastRmsNormBF16:
      return {fast_rms_norm_bf16, fast_rms_norm_bf16_size};
    case ComputeKernel::FastLayerNormF32:
      return {fast_layer_norm_f32, fast_layer_norm_f32_size};
    case ComputeKernel::FastLayerNormF16:
      return {fast_layer_norm_f16, fast_layer_norm_f16_size};
    case ComputeKernel::FastLayerNormBF16:
      return {fast_layer_norm_bf16, fast_layer_norm_bf16_size};
    case ComputeKernel::FastRmsNormVjpDxF32:
      return {fast_rms_norm_vjp_dx_f32, fast_rms_norm_vjp_dx_f32_size};
    case ComputeKernel::FastRmsNormVjpDxF16:
      return {fast_rms_norm_vjp_dx_f16, fast_rms_norm_vjp_dx_f16_size};
    case ComputeKernel::FastRmsNormVjpDxBF16:
      return {fast_rms_norm_vjp_dx_bf16, fast_rms_norm_vjp_dx_bf16_size};
    case ComputeKernel::FastLayerNormVjpDxF32:
      return {fast_layer_norm_vjp_dx_f32, fast_layer_norm_vjp_dx_f32_size};
    case ComputeKernel::FastLayerNormVjpDxF16:
      return {fast_layer_norm_vjp_dx_f16, fast_layer_norm_vjp_dx_f16_size};
    case ComputeKernel::FastLayerNormVjpDxBF16:
      return {fast_layer_norm_vjp_dx_bf16, fast_layer_norm_vjp_dx_bf16_size};
    case ComputeKernel::FastRmsNormVjpDwF32:
      return {fast_rms_norm_vjp_dw_f32, fast_rms_norm_vjp_dw_f32_size};
    case ComputeKernel::FastRmsNormVjpDwF16:
      return {fast_rms_norm_vjp_dw_f16, fast_rms_norm_vjp_dw_f16_size};
    case ComputeKernel::FastRmsNormVjpDwBF16:
      return {fast_rms_norm_vjp_dw_bf16, fast_rms_norm_vjp_dw_bf16_size};
    case ComputeKernel::FastLayerNormVjpDwF32:
      return {fast_layer_norm_vjp_dw_f32, fast_layer_norm_vjp_dw_f32_size};
    case ComputeKernel::FastLayerNormVjpDwF16:
      return {fast_layer_norm_vjp_dw_f16, fast_layer_norm_vjp_dw_f16_size};
    case ComputeKernel::FastLayerNormVjpDwBF16:
      return {fast_layer_norm_vjp_dw_bf16, fast_layer_norm_vjp_dw_bf16_size};
    case ComputeKernel::FastRmsNormVjpDwReduceF32:
      return {
          fast_rms_norm_vjp_dw_reduce_f32,
          fast_rms_norm_vjp_dw_reduce_f32_size};
    case ComputeKernel::FastRmsNormVjpDwReduceF16:
      return {
          fast_rms_norm_vjp_dw_reduce_f16,
          fast_rms_norm_vjp_dw_reduce_f16_size};
    case ComputeKernel::FastRmsNormVjpDwReduceBF16:
      return {
          fast_rms_norm_vjp_dw_reduce_bf16,
          fast_rms_norm_vjp_dw_reduce_bf16_size};
    case ComputeKernel::CrossEntropyVjpF32:
      return {fast_cross_entropy_vjp_f32, fast_cross_entropy_vjp_f32_size};
    case ComputeKernel::CrossEntropyVjpF16:
      return {fast_cross_entropy_vjp_f16, fast_cross_entropy_vjp_f16_size};
    case ComputeKernel::CrossEntropyVjpBF16:
      return {fast_cross_entropy_vjp_bf16, fast_cross_entropy_vjp_bf16_size};
    case ComputeKernel::CrossEntropyF32:
      return {fast_cross_entropy_f32, fast_cross_entropy_f32_size};
    case ComputeKernel::CrossEntropyF16:
      return {fast_cross_entropy_f16, fast_cross_entropy_f16_size};
    case ComputeKernel::FastRopeF32:
      return {fast_rope_f32, fast_rope_f32_size};
    case ComputeKernel::FastRopeF16:
      return {fast_rope_f16, fast_rope_f16_size};
    case ComputeKernel::FastRopeBF16:
      return {fast_rope_bf16, fast_rope_bf16_size};
    case ComputeKernel::FastRopeFreqsF32:
      return {fast_rope_freqs_f32, fast_rope_freqs_f32_size};
    case ComputeKernel::FastRopeFreqsF16:
      return {fast_rope_freqs_f16, fast_rope_freqs_f16_size};
    case ComputeKernel::FastRopeFreqsBF16:
      return {fast_rope_freqs_bf16, fast_rope_freqs_bf16_size};
    case ComputeKernel::CrossEntropyBF16:
      return {fast_cross_entropy_bf16, fast_cross_entropy_bf16_size};
    case ComputeKernel::Fp8ToF32:
      return {fp8_to_f32, fp8_to_f32_size};
    case ComputeKernel::Fp8ToF16:
      return {fp8_to_f16, fp8_to_f16_size};
    case ComputeKernel::Fp8ToBF16:
      return {fp8_to_bf16, fp8_to_bf16_size};
    case ComputeKernel::Fp8FromF32:
      return {fp8_from_f32, fp8_from_f32_size};
    case ComputeKernel::Fp8FromF16:
      return {fp8_from_f16, fp8_from_f16_size};
    case ComputeKernel::Fp8FromBF16:
      return {fp8_from_bf16, fp8_from_bf16_size};
    // Wave 7: linear algebra kernels.
    case ComputeKernel::LinalgCholeskyF32:
      return {linalg_cholesky_f32, linalg_cholesky_f32_size};
    case ComputeKernel::LinalgInverseF32:
      return {linalg_inverse_f32, linalg_inverse_f32_size};
    case ComputeKernel::LinalgLuF32:
      return {linalg_lu_f32, linalg_lu_f32_size};
    case ComputeKernel::LinalgQrF32:
      return {linalg_qr_f32, linalg_qr_f32_size};
    case ComputeKernel::LinalgEighF32:
      return {linalg_eigh_f32, linalg_eigh_f32_size};
    case ComputeKernel::LinalgSvdF32:
      return {linalg_svd_f32, linalg_svd_f32_size};
    case ComputeKernel::LinalgEigF32:
      return {linalg_eig_f32, linalg_eig_f32_size};
    case ComputeKernel::LinalgSvdFinalizeF32:
      return {linalg_svd_finalize_f32, linalg_svd_finalize_f32_size};
    // Complex64Transport: complex64 transport and elementwise. See
    // the enum comment in compute.h for the vec2 element layout.
    case ComputeKernel::ComplexElementwise:
      return {complex_elementwise, complex_elementwise_size};
    case ComputeKernel::ComplexReal:
      return {complex_real, complex_real_size};
    case ComputeKernel::ComplexImag:
      return {complex_imag, complex_imag_size};
    case ComputeKernel::ComplexAbs:
      return {complex_abs, complex_abs_size};
    case ComputeKernel::ComplexAbsAsComplex:
      return {complex_abs_as_complex, complex_abs_as_complex_size};
    case ComputeKernel::CastF32Complex64:
      return {cast_f32_c64, cast_f32_c64_size};
    case ComputeKernel::CastI32Complex64:
      return {cast_i32_c64, cast_i32_c64_size};
    case ComputeKernel::CastU32Complex64:
      return {cast_u32_c64, cast_u32_c64_size};
    case ComputeKernel::CastBoolComplex64:
      return {cast_bool_c64, cast_bool_c64_size};
    case ComputeKernel::CastIntW1W1:
      return {cast_int_w1_w1, cast_int_w1_w1_size};
    case ComputeKernel::CastIntW1W2:
      return {cast_int_w1_w2, cast_int_w1_w2_size};
    case ComputeKernel::CastIntW1W4:
      return {cast_int_w1_w4, cast_int_w1_w4_size};
    case ComputeKernel::CastIntW1W8:
      return {cast_int_w1_w8, cast_int_w1_w8_size};
    case ComputeKernel::CastIntW2W1:
      return {cast_int_w2_w1, cast_int_w2_w1_size};
    case ComputeKernel::CastIntW2W2:
      return {cast_int_w2_w2, cast_int_w2_w2_size};
    case ComputeKernel::CastIntW2W4:
      return {cast_int_w2_w4, cast_int_w2_w4_size};
    case ComputeKernel::CastIntW2W8:
      return {cast_int_w2_w8, cast_int_w2_w8_size};
    case ComputeKernel::CastIntW4W1:
      return {cast_int_w4_w1, cast_int_w4_w1_size};
    case ComputeKernel::CastIntW4W2:
      return {cast_int_w4_w2, cast_int_w4_w2_size};
    case ComputeKernel::CastIntW4W4:
      return {cast_int_w4_w4, cast_int_w4_w4_size};
    case ComputeKernel::CastIntW4W8:
      return {cast_int_w4_w8, cast_int_w4_w8_size};
    case ComputeKernel::CastIntW8W1:
      return {cast_int_w8_w1, cast_int_w8_w1_size};
    case ComputeKernel::CastIntW8W2:
      return {cast_int_w8_w2, cast_int_w8_w2_size};
    case ComputeKernel::CastIntW8W4:
      return {cast_int_w8_w4, cast_int_w8_w4_size};
    case ComputeKernel::CastIntW8W8:
      return {cast_int_w8_w8, cast_int_w8_w8_size};
    case ComputeKernel::CastF16Complex64:
      return {cast_f16_c64, cast_f16_c64_size};
    case ComputeKernel::CastBF16Complex64:
      return {cast_bf16_c64, cast_bf16_c64_size};
    case ComputeKernel::CastComplex64F32:
      return {cast_c64_f32, cast_c64_f32_size};
    case ComputeKernel::FillComplex64:
      return {fill_c64, fill_c64_size};
    case ComputeKernel::FillU64:
      return {fill_u64, fill_u64_size};
    case ComputeKernel::FillU16:
      return {fill_u16, fill_u16_size};
    case ComputeKernel::CompareU64:
      return {compare_u64, compare_u64_size};
    case ComputeKernel::ElementwiseI8:
      return {elementwise_i8, elementwise_i8_size};
    case ComputeKernel::ElementwiseU8:
      return {elementwise_u8, elementwise_u8_size};
    case ComputeKernel::ElementwiseI16:
      return {elementwise_i16, elementwise_i16_size};
    case ComputeKernel::ElementwiseU16:
      return {elementwise_u16, elementwise_u16_size};
    case ComputeKernel::ElementwiseI64:
      return {elementwise_i64, elementwise_i64_size};
    case ComputeKernel::ElementwiseU64:
      return {elementwise_u64, elementwise_u64_size};
    case ComputeKernel::CompareI8:
      return {compare_i8, compare_i8_size};
    case ComputeKernel::CompareU8:
      return {compare_u8, compare_u8_size};
    case ComputeKernel::CompareI16:
      return {compare_i16, compare_i16_size};
    case ComputeKernel::CompareU16:
      return {compare_u16, compare_u16_size};
    case ComputeKernel::ScatterU8:
      return {scatter_u8, scatter_u8_size};
    case ComputeKernel::ScatterI8:
      return {scatter_i8, scatter_i8_size};
    case ComputeKernel::ScatterU16:
      return {scatter_u16, scatter_u16_size};
    case ComputeKernel::ScatterI16:
      return {scatter_i16, scatter_i16_size};
    case ComputeKernel::ScatterMultiU8:
      return {scatter_multi_u8, scatter_multi_u8_size};
    case ComputeKernel::ScatterMultiI8:
      return {scatter_multi_i8, scatter_multi_i8_size};
    case ComputeKernel::ScatterMultiU16:
      return {scatter_multi_u16, scatter_multi_u16_size};
    case ComputeKernel::ScatterMultiI16:
      return {scatter_multi_i16, scatter_multi_i16_size};
    case ComputeKernel::ScatterTripleU8:
      return {scatter_triple_u8, scatter_triple_u8_size};
    case ComputeKernel::ScatterTripleI8:
      return {scatter_triple_i8, scatter_triple_i8_size};
    case ComputeKernel::ScatterTripleU16:
      return {scatter_triple_u16, scatter_triple_u16_size};
    case ComputeKernel::ScatterTripleI16:
      return {scatter_triple_i16, scatter_triple_i16_size};
    case ComputeKernel::CopyGeneralComplex64:
      return {copy_general_c64, copy_general_c64_size};
    case ComputeKernel::ScatterFAddF32:
      return {scatter_fadd_f32, scatter_fadd_f32_size};
    case ComputeKernel::ScatterFAddF16:
      return {scatter_fadd_f16, scatter_fadd_f16_size};
    case ComputeKernel::ScatterFAddBF16:
      return {scatter_fadd_bf16, scatter_fadd_bf16_size};
    case ComputeKernel::ScatterFAddMultiF32:
      return {scatter_fadd_multi_f32, scatter_fadd_multi_f32_size};
    case ComputeKernel::ScatterFAddTripleF32:
      return {scatter_fadd_triple_f32, scatter_fadd_triple_f32_size};
    case ComputeKernel::ScatterFAddGeneralF32:
      return {scatter_fadd_general_f32, scatter_fadd_general_f32_size};
    case ComputeKernel::ScatterFCasF32:
      return {scatter_fcas_f32, scatter_fcas_f32_size};
    case ComputeKernel::ScatterFCasF16:
      return {scatter_fcas_f16, scatter_fcas_f16_size};
    case ComputeKernel::ScatterFCasBF16:
      return {scatter_fcas_bf16, scatter_fcas_bf16_size};
    case ComputeKernel::ScatterFCasMultiF32:
      return {scatter_fcas_multi_f32, scatter_fcas_multi_f32_size};
    case ComputeKernel::ScatterFCasTripleF32:
      return {scatter_fcas_triple_f32, scatter_fcas_triple_f32_size};
    case ComputeKernel::ScatterFCasGeneralF32:
      return {scatter_fcas_general_f32, scatter_fcas_general_f32_size};
    case ComputeKernel::ScatterBool:
      return {scatter_bool, scatter_bool_size};
    case ComputeKernel::ScatterBoolMulti:
      return {scatter_bool_multi, scatter_bool_multi_size};
    case ComputeKernel::ScatterAxisFAddF32:
      return {scatter_axis_fadd_f32, scatter_axis_fadd_f32_size};
    case ComputeKernel::ScatterAxisFAddF16:
      return {scatter_axis_fadd_f16, scatter_axis_fadd_f16_size};
    case ComputeKernel::ScatterAxisFAddBF16:
      return {scatter_axis_fadd_bf16, scatter_axis_fadd_bf16_size};
    case ComputeKernel::ScatterAxisFCasF32:
      return {scatter_axis_fcas_f32, scatter_axis_fcas_f32_size};
    case ComputeKernel::ScatterAxisFCasF16:
      return {scatter_axis_fcas_f16, scatter_axis_fcas_f16_size};
    case ComputeKernel::ScatterAxisFCasBF16:
      return {scatter_axis_fcas_bf16, scatter_axis_fcas_bf16_size};
    case ComputeKernel::ScatterAxisBool:
      return {scatter_axis_bool, scatter_axis_bool_size};
    case ComputeKernel::QmmVecF32:
      return {qmm_vec_f32, qmm_vec_f32_size};
    case ComputeKernel::QmmVecF16:
      return {qmm_vec_f16, qmm_vec_f16_size};
    case ComputeKernel::QmmVecBF16:
      return {qmm_vec_bf16, qmm_vec_bf16_size};
    case ComputeKernel::QmmVecSubgroupF32:
      return {qmm_vec_subgroup_f32, qmm_vec_subgroup_f32_size};
    case ComputeKernel::QmmVecSubgroupF16:
      return {qmm_vec_subgroup_f16, qmm_vec_subgroup_f16_size};
    case ComputeKernel::QmmVecSubgroupBF16:
      return {qmm_vec_subgroup_bf16, qmm_vec_subgroup_bf16_size};
    case ComputeKernel::QmmVecQ4WordF32:
      return {qmm_vec_q4_word_f32, qmm_vec_q4_word_f32_size};
    case ComputeKernel::QmmVecQ4WordF16:
      return {qmm_vec_q4_word_f16, qmm_vec_q4_word_f16_size};
    case ComputeKernel::QmmVecQ4WordBF16:
      return {qmm_vec_q4_word_bf16, qmm_vec_q4_word_bf16_size};
    case ComputeKernel::QmmVecQ4WordSubgroupF32:
      return {
          qmm_vec_q4_word_subgroup_f32,
          qmm_vec_q4_word_subgroup_f32_size};
    case ComputeKernel::QmmVecQ4WordSubgroupF16:
      return {
          qmm_vec_q4_word_subgroup_f16,
          qmm_vec_q4_word_subgroup_f16_size};
    case ComputeKernel::QmmVecQ4WordSubgroupBF16:
      return {
          qmm_vec_q4_word_subgroup_bf16,
          qmm_vec_q4_word_subgroup_bf16_size};
    case ComputeKernel::QmmTileF32:
      return {qmm_tile_f32, qmm_tile_f32_size};
    case ComputeKernel::QmmTileF16:
      return {qmm_tile_f16, qmm_tile_f16_size};
    case ComputeKernel::QmmTileRbF16:
      return {qmm_tile_rb_f16, qmm_tile_rb_f16_size};
    case ComputeKernel::QmmTileRbPreciseF16:
      return {qmm_tile_rb_precise_f16, qmm_tile_rb_precise_f16_size};
    case ComputeKernel::QmmPrefillCoopmatF16:
      return {qmm_coopmat_f16, qmm_coopmat_f16_size};
    case ComputeKernel::QmmPrefillCoopmatM16F16:
      return {qmm_coopmat_m16_f16, qmm_coopmat_m16_f16_size};
    case ComputeKernel::QmmPrefillCoopmatBF16:
      return {qmm_coopmat_bf16, qmm_coopmat_bf16_size};
    case ComputeKernel::QmmPrefillCoopmatM16BF16:
      return {qmm_coopmat_m16_bf16, qmm_coopmat_m16_bf16_size};
    case ComputeKernel::QmmPrefillCoopmatBF16X32:
      return {qmm_coopmat_x32, qmm_coopmat_x32_size};
    case ComputeKernel::QmmPrefillCoopmatM16BF16X32:
      return {qmm_coopmat_m16_x32, qmm_coopmat_m16_x32_size};
    case ComputeKernel::GatedDeltaDecodeBF16:
      return {gated_delta_decode_bf16, gated_delta_decode_bf16_size};
    case ComputeKernel::GatedDeltaPrefillBF16:
      return {gated_delta_prefill_bf16, gated_delta_prefill_bf16_size};
    case ComputeKernel::GatedDeltaPrefillCoopmatBF16:
      return {
          gated_delta_prefill_coopmat_bf16,
          gated_delta_prefill_coopmat_bf16_size};
    case ComputeKernel::FastNormGatedBF16:
      return {fast_norm_gated_bf16, fast_norm_gated_bf16_size};
    case ComputeKernel::GdnConvDecodeBF16:
      return {gdn_conv_decode_bf16, gdn_conv_decode_bf16_size};
    case ComputeKernel::QmmPrefillFmaF16:
      return {qmm_fma_f16, qmm_fma_f16_size};
    case ComputeKernel::MatmulBF16Coopmat:
      return {matmul_f32_coopmat_bf16, matmul_f32_coopmat_bf16_size};
    case ComputeKernel::QmmPrefillFmaPreciseF16:
      return {qmm_fma_precise_f16, qmm_fma_precise_f16_size};
    case ComputeKernel::MatmulBf16Fma:
      return {matmul_fma_bf16, matmul_fma_bf16_size};
    case ComputeKernel::QmmTileBF16:
      return {qmm_tile_bf16, qmm_tile_bf16_size};
    case ComputeKernel::FusedChainF32:
      return {fused_chain_f32, fused_chain_f32_size};
    case ComputeKernel::FusedChainF16:
      return {fused_chain_f16, fused_chain_f16_size};
    case ComputeKernel::QuantizeFpF32:
      return {quantize_fp_f32, quantize_fp_f32_size};
    case ComputeKernel::FusedChainBF16:
      return {fused_chain_bf16, fused_chain_bf16_size};
    case ComputeKernel::BinaryVecF16:
      return {binary_vec_f16, binary_vec_f16_size};
    case ComputeKernel::BinaryVecBF16:
      return {binary_vec_bf16, binary_vec_bf16_size};
    case ComputeKernel::MatmulRbF16:
      return {matmul_rb_f16, matmul_rb_f16_size};
    case ComputeKernel::SwigluF16:
      return {swiglu_f16, swiglu_f16_size};
    case ComputeKernel::SwigluBF16:
      return {swiglu_bf16, swiglu_bf16_size};
    case ComputeKernel::FastTrioNormF16:
      return {fast_trio_norm_f16, fast_trio_norm_f16_size};
    case ComputeKernel::FastTrioRopePairF16:
      return {fast_trio_rope_pair_f16, fast_trio_rope_pair_f16_size};
    case ComputeKernel::FastTrioSwigluF16:
      return {fast_trio_swiglu_f16, fast_trio_swiglu_f16_size};
    case ComputeKernel::QmmVecQ4MultiF32:
      return {qmm_vec_q4_multi_f32, qmm_vec_q4_multi_f32_size};
    case ComputeKernel::QmmVecQ4MultiF16:
      return {qmm_vec_q4_multi_f16, qmm_vec_q4_multi_f16_size};
    case ComputeKernel::QmmVecQ4MultiBF16:
      return {qmm_vec_q4_multi_bf16, qmm_vec_q4_multi_bf16_size};
    case ComputeKernel::QmmVecQ4MultiSubgroupF32:
      return {
          qmm_vec_q4_multi_subgroup_f32, qmm_vec_q4_multi_subgroup_f32_size};
    case ComputeKernel::QmmVecQ4MultiSubgroupF16:
      return {
          qmm_vec_q4_multi_subgroup_f16, qmm_vec_q4_multi_subgroup_f16_size};
    case ComputeKernel::QmmVecQ4MultiSubgroupBF16:
      return {
          qmm_vec_q4_multi_subgroup_bf16,
          qmm_vec_q4_multi_subgroup_bf16_size};
    case ComputeKernel::SdpaDecodeNativeF16:
      return {sdpa_decode_native_f16, sdpa_decode_native_f16_size};
    case ComputeKernel::SdpaDecodeNativeBF16:
      return {sdpa_decode_native_bf16, sdpa_decode_native_bf16_size};
    case ComputeKernel::SdpaDecodeNativeBF16Hd256:
      return {sdpa_decode_native_bf16_hd256, sdpa_decode_native_bf16_hd256_size};
    case ComputeKernel::SdpaDecodeNativeTwoPassP1F16:
      return {sdpa_decode_native_p1_f16, sdpa_decode_native_p1_f16_size};
    case ComputeKernel::SdpaDecodeNativeTwoPassP2F16:
      return {sdpa_decode_native_p2_f16, sdpa_decode_native_p2_f16_size};
    case ComputeKernel::QuantizeFpF16:
      return {quantize_fp_f16, quantize_fp_f16_size};
    case ComputeKernel::QuantizeFpBF16:
      return {quantize_fp_bf16, quantize_fp_bf16_size};
    case ComputeKernel::DequantFpF32:
      return {dequant_fp_f32, dequant_fp_f32_size};
    case ComputeKernel::DequantFpF16:
      return {dequant_fp_f16, dequant_fp_f16_size};
    case ComputeKernel::DequantFpBF16:
      return {dequant_fp_bf16, dequant_fp_bf16_size};
    case ComputeKernel::QmmFpF32:
      return {qmm_fp_f32, qmm_fp_f32_size};
    case ComputeKernel::QmmFpF16:
      return {qmm_fp_f16, qmm_fp_f16_size};
    case ComputeKernel::QmmFpBF16:
      return {qmm_fp_bf16, qmm_fp_bf16_size};
    case ComputeKernel::QmmVecFpF32:
      return {qmm_vec_fp_f32, qmm_vec_fp_f32_size};
    case ComputeKernel::QmmVecFpF16:
      return {qmm_vec_fp_f16, qmm_vec_fp_f16_size};
    case ComputeKernel::QmmVecFpBF16:
      return {qmm_vec_fp_bf16, qmm_vec_fp_bf16_size};
    case ComputeKernel::QmmVecSubgroupFpF32:
      return {qmm_vec_subgroup_fp_f32, qmm_vec_subgroup_fp_f32_size};
    case ComputeKernel::QmmVecSubgroupFpF16:
      return {qmm_vec_subgroup_fp_f16, qmm_vec_subgroup_fp_f16_size};
    case ComputeKernel::QmmVecSubgroupFpBF16:
      return {qmm_vec_subgroup_fp_bf16, qmm_vec_subgroup_fp_bf16_size};
    case ComputeKernel::QmmTileFpF32:
      return {qmm_tile_fp_f32, qmm_tile_fp_f32_size};
    case ComputeKernel::QmmTileFpF16:
      return {qmm_tile_fp_f16, qmm_tile_fp_f16_size};
    case ComputeKernel::QmmTileFpBF16:
      return {qmm_tile_fp_bf16, qmm_tile_fp_bf16_size};
    case ComputeKernel::GatherQmmNbFpF32:
      return {gather_qmm_nb_fp_f32, gather_qmm_nb_fp_f32_size};
    case ComputeKernel::GatherQmmNbFpF16:
      return {gather_qmm_nb_fp_f16, gather_qmm_nb_fp_f16_size};
    case ComputeKernel::GatherQmmNbFpBF16:
      return {gather_qmm_nb_fp_bf16, gather_qmm_nb_fp_bf16_size};
    case ComputeKernel::GatherQmmNbFpHgsF32:
      return {gather_qmm_nb_fp_hgs_f32, gather_qmm_nb_fp_hgs_f32_size};
    case ComputeKernel::GatherQmmNbFpHgsF16:
      return {gather_qmm_nb_fp_hgs_f16, gather_qmm_nb_fp_hgs_f16_size};
    case ComputeKernel::GatherQmmNbFpHgsBF16:
      return {gather_qmm_nb_fp_hgs_bf16, gather_qmm_nb_fp_hgs_bf16_size};
    case ComputeKernel::MatmulComplex64:
      return {matmul_complex64, matmul_complex64_size};
    case ComputeKernel::Custom:
    case ComputeKernel::Count:
      break;
  }
  throw std::invalid_argument("[omarchy] invalid compute kernel.");
}

} // namespace

ComputeRuntime::ComputeRuntime(VkDevice device, uint32_t binding_limit)
    : device_(device), binding_limit_(binding_limit) {
  auto& dt = vk::device_table();
  if (binding_limit_ == 0 || binding_limit_ > kComputeBindingBudget) {
    throw std::invalid_argument("[omarchy] invalid compute binding budget.");
  }
  std::array<VkDescriptorSetLayoutBinding, kComputeBindingBudget> bindings{};
  for (uint32_t index = 0; index < binding_limit_; ++index) {
    bindings[index].binding = index;
    bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[index].descriptorCount = 1;
    bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo descriptor_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  descriptor_info.bindingCount = binding_limit_;
  descriptor_info.pBindings = bindings.data();
  VKX_CHECK(dt.CreateDescriptorSetLayout(
      device_, &descriptor_info, nullptr, &descriptor_layout_));

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.size = sizeof(ComputeParams);
  VkPipelineLayoutCreateInfo pipeline_info{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipeline_info.setLayoutCount = 1;
  pipeline_info.pSetLayouts = &descriptor_layout_;
  pipeline_info.pushConstantRangeCount = 1;
  pipeline_info.pPushConstantRanges = &push_range;
  try {
    VKX_CHECK(dt.CreatePipelineLayout(
        device_, &pipeline_info, nullptr, &pipeline_layout_));
  } catch (...) {
    dt.DestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
    descriptor_layout_ = VK_NULL_HANDLE;
    throw;
  }
}

ComputeRuntime::~ComputeRuntime() {
  auto& dt = vk::device_table();
  for (VkPipeline pipeline : pipelines_) {
    if (pipeline != VK_NULL_HANDLE) {
      dt.DestroyPipeline(device_, pipeline, nullptr);
    }
  }
  for (const auto& [_, pipeline] : dynamic_pipelines_) {
    dt.DestroyPipeline(device_, pipeline, nullptr);
  }
  if (pipeline_layout_ != VK_NULL_HANDLE) {
    dt.DestroyPipelineLayout(device_, pipeline_layout_, nullptr);
  }
  if (descriptor_layout_ != VK_NULL_HANDLE) {
    dt.DestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
  }
}

VkPipeline ComputeRuntime::pipeline(ComputeKernel kernel) {
  size_t index = static_cast<size_t>(kernel);
  if (index >= pipelines_.size()) {
    throw std::invalid_argument("[omarchy] invalid compute kernel.");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (pipelines_[index] == VK_NULL_HANDLE) {
    pipelines_[index] = create_pipeline(kernel);
  }
  return pipelines_[index];
}

VkPipeline ComputeRuntime::pipeline(
    const std::string& cache_key,
    std::span<const uint32_t> spirv) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto [entry, inserted] = dynamic_pipelines_.try_emplace(cache_key);
  if (inserted) {
    try {
      entry->second = create_pipeline(spirv);
    } catch (...) {
      dynamic_pipelines_.erase(entry);
      throw;
    }
  }
  return entry->second;
}

VkPipeline ComputeRuntime::create_pipeline(ComputeKernel kernel) {
  auto [bytes, size] = shader_bytes(kernel);
  if (size == 0 || size % sizeof(uint32_t) != 0) {
    throw std::runtime_error("[omarchy] embedded SPIR-V has an invalid size.");
  }
  return create_pipeline(std::span<const uint32_t>{
      reinterpret_cast<const uint32_t*>(bytes), size / sizeof(uint32_t)});
}

VkPipeline ComputeRuntime::create_pipeline(std::span<const uint32_t> spirv) {
  if (spirv.empty() || spirv.front() != 0x07230203u) {
    throw std::runtime_error("[omarchy] custom SPIR-V is invalid.");
  }
  auto& dt = vk::device_table();
  VkShaderModuleCreateInfo shader_info{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  shader_info.codeSize = spirv.size_bytes();
  shader_info.pCode = spirv.data();
  VkShaderModule shader{VK_NULL_HANDLE};
  VKX_CHECK(dt.CreateShaderModule(device_, &shader_info, nullptr, &shader));

  VkPipelineShaderStageCreateInfo stage{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = shader;
  stage.pName = "main";
  VkComputePipelineCreateInfo pipeline_info{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline_info.stage = stage;
  pipeline_info.layout = pipeline_layout_;

  VkPipeline pipeline{VK_NULL_HANDLE};
  try {
    VKX_CHECK(dt.CreateComputePipelines(
        device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline));
  } catch (...) {
    dt.DestroyShaderModule(device_, shader, nullptr);
    throw;
  }
  dt.DestroyShaderModule(device_, shader, nullptr);
  return pipeline;
}

} // namespace mlx::core::omarchy
