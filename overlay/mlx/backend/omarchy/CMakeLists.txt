# Env-gated GPU profiling harness (gpu_profiler.h). Compiled OUT by
# default: release builds carry zero profiling code. Turn on only for
# diagnostic builds (MLX_OMARCHY_GPU_PROFILE env var selects the output).
option(MLX_OMARCHY_GPU_PROFILING
       "Compile in the env-gated GPU profiling harness" OFF)
if(MLX_OMARCHY_GPU_PROFILING)
  target_compile_definitions(mlx PRIVATE MLX_OMARCHY_GPU_PROFILING)
endif()
target_compile_definitions(mlx PRIVATE MLX_OMARCHY_BACKEND)
set(MLX_OMARCHY_ANE_SOURCE_DIR "" CACHE PATH
    "Qualified joshuaswarren/omarchy-ane checkout for the private ANE runtime")
if(MLX_OMARCHY_ANE_SOURCE_DIR)
  if(NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR
            "MLX_OMARCHY_ANE_SOURCE_DIR requires BUILD_SHARED_LIBS=ON")
  endif()
  execute_process(
    COMMAND git -C "${MLX_OMARCHY_ANE_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE MLX_OMARCHY_ANE_SOURCE_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE MLX_OMARCHY_ANE_GIT_RESULT)
  execute_process(
    COMMAND git -C "${MLX_OMARCHY_ANE_SOURCE_DIR}" status --porcelain
            --untracked-files=no
    OUTPUT_VARIABLE MLX_OMARCHY_ANE_SOURCE_DIRTY
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE MLX_OMARCHY_ANE_DIRTY_RESULT)
  if(NOT MLX_OMARCHY_ANE_GIT_RESULT EQUAL 0 OR
     NOT MLX_OMARCHY_ANE_DIRTY_RESULT EQUAL 0 OR
     MLX_OMARCHY_ANE_SOURCE_DIRTY OR
     NOT MLX_OMARCHY_ANE_SOURCE_COMMIT STREQUAL
         "6fa243ac7241119a9eb229abbf8cb4dd8949f915")
    message(FATAL_ERROR
            "MLX_OMARCHY_ANE_SOURCE_DIR must be canonical omarchy-ane "
            "6fa243ac7241119a9eb229abbf8cb4dd8949f915")
  endif()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(OMARCHY_ANE_LIBDRM REQUIRED IMPORTED_TARGET libdrm)
  add_library(mlx_omarchy_libane STATIC
              "${MLX_OMARCHY_ANE_SOURCE_DIR}/libane/ane.c")
  # STRICT_BIND: libane refuses a positional channel map at load, the
  # same contract the bundle validator enforces (bundle.cpp strict
  # schema-4 checks).
  target_compile_definitions(mlx_omarchy_libane PRIVATE
                             LIBANE_CONFIG_STRICT_BIND)
  target_include_directories(
    mlx_omarchy_libane
    PUBLIC "${MLX_OMARCHY_ANE_SOURCE_DIR}/libane"
    PRIVATE "${MLX_OMARCHY_ANE_SOURCE_DIR}/ane/src/uapi/drm")
  target_link_libraries(mlx_omarchy_libane PRIVATE PkgConfig::OMARCHY_ANE_LIBDRM)
endif()

find_program(OMARCHY_GLSLC glslc)
find_program(OMARCHY_GLSLANG glslangValidator)
if(NOT OMARCHY_GLSLC AND NOT OMARCHY_GLSLANG)
  message(
    FATAL_ERROR
      "Neither glslc nor glslangValidator was found. Install a Vulkan "
      "shader compiler (Omarchy: sudo pacman -S shaderc glslang).")
endif()

function(omarchy_shader name source)
  set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
  set(spirv "${output_dir}/${name}.spv")
  set(header "${output_dir}/${name}.h")
  if(OMARCHY_GLSLC)
    add_custom_command(
      OUTPUT "${spirv}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
      COMMAND "${OMARCHY_GLSLC}" -O --target-env=vulkan1.3 ${ARGN}
              "${CMAKE_CURRENT_SOURCE_DIR}/${source}" -o "${spirv}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
      VERBATIM)
  else()
    add_custom_command(
      OUTPUT "${spirv}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
      COMMAND "${OMARCHY_GLSLANG}" -V -Os --target-env vulkan1.3 ${ARGN}
              "${CMAKE_CURRENT_SOURCE_DIR}/${source}" -o "${spirv}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
      VERBATIM)
  endif()
  add_custom_command(
    OUTPUT "${header}"
    COMMAND
      "${CMAKE_COMMAND}" -DINPUT=${spirv} -DOUTPUT=${header}
      -DNAME=${name} -P "${CMAKE_CURRENT_SOURCE_DIR}/embed_spirv.cmake"
    DEPENDS "${spirv}" "${CMAKE_CURRENT_SOURCE_DIR}/embed_spirv.cmake"
    VERBATIM)
  set(OMARCHY_SHADER_HEADERS
      ${OMARCHY_SHADER_HEADERS} "${header}"
      PARENT_SCOPE)
endfunction()

omarchy_shader(elementwise_f32 shaders/elementwise.comp)
omarchy_shader(elementwise_f16 shaders/elementwise.comp -DUSE_FP16=1)
omarchy_shader(elementwise_bf16 shaders/elementwise.comp -DUSE_BF16=1)
omarchy_shader(cast_bool_f32 shaders/cast.comp -DSOURCE_BOOL=1)
omarchy_shader(cast_bool_i32 shaders/cast.comp -DSOURCE_BOOL=1 -DDEST_I32=1)
omarchy_shader(cast_bool_f16 shaders/cast.comp -DSOURCE_BOOL=1 -DDEST_FP16=1)
omarchy_shader(cast_bool_bf16 shaders/cast.comp -DSOURCE_BOOL=1 -DDEST_BF16=1)
omarchy_shader(cast_f16_f32 shaders/cast.comp -DSOURCE_FP16=1)
omarchy_shader(cast_f32_f16 shaders/cast.comp -DDEST_FP16=1)
omarchy_shader(cast_bf16_f32 shaders/cast.comp -DSOURCE_BF16=1)
omarchy_shader(cast_f32_bf16 shaders/cast.comp -DDEST_BF16=1)
omarchy_shader(cast_bf16_f16 shaders/cast.comp -DSOURCE_BF16=1 -DDEST_FP16=1)
omarchy_shader(cast_f16_bf16 shaders/cast.comp -DSOURCE_FP16=1 -DDEST_BF16=1)
omarchy_shader(cast_i32_f32 shaders/cast.comp -DSOURCE_I32=1)
omarchy_shader(cast_f32_i32 shaders/cast.comp -DDEST_I32=1)
omarchy_shader(cast_i32_f16 shaders/cast.comp -DSOURCE_I32=1 -DDEST_FP16=1)
omarchy_shader(cast_f16_i32 shaders/cast.comp -DSOURCE_FP16=1 -DDEST_I32=1)
omarchy_shader(cast_i32_bf16 shaders/cast.comp -DSOURCE_I32=1 -DDEST_BF16=1)
omarchy_shader(cast_bf16_i32 shaders/cast.comp -DSOURCE_BF16=1 -DDEST_I32=1)
omarchy_shader(cast_u32_f32 shaders/cast.comp -DSOURCE_U32=1)
omarchy_shader(reduce_f32 shaders/reduce_suffix.comp)
omarchy_shader(reduce_f16 shaders/reduce_suffix.comp -DUSE_FP16=1)
omarchy_shader(reduce_bf16 shaders/reduce_suffix.comp -DUSE_BF16=1)
omarchy_shader(matmul_f32 shaders/matmul.comp)
omarchy_shader(matmul_f32_coopmat shaders/matmul_coopmat.comp)
omarchy_shader(matmul_f32_coopmat_bf16 shaders/matmul_coopmat_bf16.comp)
omarchy_shader(matmul_f16 shaders/matmul.comp -DUSE_FP16=1)
omarchy_shader(matmul_bf16 shaders/matmul.comp -DUSE_BF16=1)
omarchy_shader(matmul_complex64 shaders/matmul.comp -DUSE_COMPLEX64=1)
omarchy_shader(matmul_vec_f32 shaders/matmul_vec.comp)
omarchy_shader(matmul_vec_f16 shaders/matmul_vec.comp -DUSE_FP16=1)
omarchy_shader(matmul_vec_bf16 shaders/matmul_vec.comp -DUSE_BF16=1)
omarchy_shader(matmul_vec_multi_bf16 shaders/matmul_vec_multi_bf16.comp)
omarchy_shader(fill_f32 shaders/fill.comp)
omarchy_shader(fill_f16 shaders/fill.comp -DUSE_FP16=1)
omarchy_shader(fill_bf16 shaders/fill.comp -DUSE_BF16=1)
omarchy_shader(softmax_f32 shaders/softmax_suffix.comp)
omarchy_shader(softmax_f16 shaders/softmax_suffix.comp -DUSE_FP16=1)
omarchy_shader(softmax_bf16 shaders/softmax_suffix.comp -DUSE_BF16=1)
omarchy_shader(logsumexp_f32 shaders/logsumexp_suffix.comp)
omarchy_shader(logsumexp_f16 shaders/logsumexp_suffix.comp -DUSE_FP16=1)
omarchy_shader(logsumexp_bf16 shaders/logsumexp_suffix.comp -DUSE_BF16=1)
omarchy_shader(select_f32 shaders/select.comp)
omarchy_shader(select_f16 shaders/select.comp -DUSE_FP16=1)
omarchy_shader(select_bf16 shaders/select.comp -DUSE_BF16=1)
omarchy_shader(select_i32 shaders/select.comp -DUSE_I32=1)
omarchy_shader(select_bool shaders/select.comp -DUSE_BOOL=1)
omarchy_shader(select_complex64 shaders/select.comp -DUSE_U2=1)
omarchy_shader(compare_f32 shaders/compare.comp)
omarchy_shader(compare_f16 shaders/compare.comp -DUSE_FP16=1)
omarchy_shader(compare_bf16 shaders/compare.comp -DUSE_BF16=1)
omarchy_shader(compare_i32 shaders/compare.comp -DUSE_I32=1)
omarchy_shader(compare_u32 shaders/compare.comp -DUSE_U32=1)
omarchy_shader(compare_i64 shaders/compare.comp -DUSE_I64=1)
omarchy_shader(compare_i8 shaders/compare.comp -DUSE_I8=1)
omarchy_shader(compare_u8 shaders/compare.comp -DUSE_U8=1)
omarchy_shader(compare_i16 shaders/compare.comp -DUSE_I16=1)
omarchy_shader(compare_u16 shaders/compare.comp -DUSE_U16=1)
omarchy_shader(compare_u64 shaders/compare.comp -DUSE_U64=1)
omarchy_shader(compare_complex shaders/compare_complex.comp)
omarchy_shader(logical_or_bool shaders/logical_or.comp)
omarchy_shader(compare_bool shaders/compare_bool.comp)
omarchy_shader(take_f32 shaders/gather_take.comp)
omarchy_shader(take_f16 shaders/gather_take.comp -DUSE_FP16=1)
omarchy_shader(take_bf16 shaders/gather_take.comp -DUSE_BF16=1)
omarchy_shader(take_u32 shaders/gather_take.comp -DUSE_U32=1)
omarchy_shader(take_u16 shaders/gather_take.comp -DUSE_U16=1)
omarchy_shader(take_i64 shaders/gather_take.comp -DUSE_I64=1)
omarchy_shader(take_c64 shaders/gather_take.comp -DUSE_COMPLEX=1)
omarchy_shader(take_multi_f32 shaders/gather_take.comp -DUSE_MULTI_INDEX=1)
omarchy_shader(take_multi_f16 shaders/gather_take.comp -DUSE_FP16=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(take_multi_bf16 shaders/gather_take.comp -DUSE_BF16=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(take_multi_u32 shaders/gather_take.comp -DUSE_U32=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(take_multi_u16 shaders/gather_take.comp -DUSE_U16=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(take_multi_i64 shaders/gather_take.comp -DUSE_I64=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(take_multi_c64 shaders/gather_take.comp -DUSE_COMPLEX=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(slice_update_reduce_f32 shaders/slice_update_reduce.comp)
omarchy_shader(slice_update_reduce_f16 shaders/slice_update_reduce.comp -DUSE_FP16=1)
omarchy_shader(slice_update_reduce_bf16 shaders/slice_update_reduce.comp -DUSE_BF16=1)
omarchy_shader(slice_update_reduce_u32 shaders/slice_update_reduce.comp -DUSE_U32=1)
omarchy_shader(slice_update_pair_f16 shaders/slice_update_pair.comp)
omarchy_shader(scatter_triple_u32 shaders/scatter.comp -DUSE_U32=1 -DUSE_TRIPLE_INDEX=1)
omarchy_shader(scatter_triple_f16 shaders/scatter.comp -DUSE_FP16=1 -DUSE_TRIPLE_INDEX=1)
omarchy_shader(scatter_triple_bf16 shaders/scatter.comp -DUSE_BF16=1 -DUSE_TRIPLE_INDEX=1)
omarchy_shader(scatter_bool_triple shaders/scatter.comp -DUSE_BOOL=1 -DUSE_TRIPLE_INDEX=1)
omarchy_shader(scatter_general_u32 shaders/scatter.comp -DUSE_U32=1 -DUSE_GENERAL_INDEX=1)
omarchy_shader(scatter_general_f16 shaders/scatter.comp -DUSE_FP16=1 -DUSE_GENERAL_INDEX=1)
omarchy_shader(scatter_general_bf16 shaders/scatter.comp -DUSE_BF16=1 -DUSE_GENERAL_INDEX=1)
omarchy_shader(scatter_general_bool shaders/scatter.comp -DUSE_BOOL=1 -DUSE_GENERAL_INDEX=1)
omarchy_shader(scatter_general_u8 shaders/scatter.comp -DUSE_U8=1 -DUSE_GENERAL_INDEX=1)
omarchy_shader(scatter_general_i8 shaders/scatter.comp -DUSE_I8=1 -DUSE_GENERAL_INDEX=1)
omarchy_shader(scatter_general_u16 shaders/scatter.comp -DUSE_U16=1 -DUSE_GENERAL_INDEX=1)
omarchy_shader(scatter_general_i16 shaders/scatter.comp -DUSE_I16=1 -DUSE_GENERAL_INDEX=1)
omarchy_shader(scatter_triple_u8 shaders/scatter.comp -DUSE_U8=1 -DUSE_TRIPLE_INDEX=1)
omarchy_shader(scatter_triple_i8 shaders/scatter.comp -DUSE_I8=1 -DUSE_TRIPLE_INDEX=1)
omarchy_shader(scatter_triple_u16 shaders/scatter.comp -DUSE_U16=1 -DUSE_TRIPLE_INDEX=1)
omarchy_shader(scatter_triple_i16 shaders/scatter.comp -DUSE_I16=1 -DUSE_TRIPLE_INDEX=1)
omarchy_shader(copy_general_f32 shaders/copy_general.comp)
omarchy_shader(copy_general_f16 shaders/copy_general.comp -DUSE_FP16=1)
omarchy_shader(copy_general_bf16 shaders/copy_general.comp -DUSE_BF16=1)
omarchy_shader(copy_general_bool shaders/copy_general.comp -DUSE_BOOL=1)
omarchy_shader(copy_general_u32 shaders/copy_general.comp -DUSE_U32=1)
omarchy_shader(copy_general_u8 shaders/copy_general.comp -DUSE_U8=1)
omarchy_shader(copy_general_u16 shaders/copy_general.comp -DUSE_U16=1)
omarchy_shader(copy_general_u64 shaders/copy_general.comp -DUSE_U64=1)
omarchy_shader(argreduce_f32 shaders/argreduce_suffix.comp)
omarchy_shader(argreduce_f16 shaders/argreduce_suffix.comp -DUSE_FP16=1)
omarchy_shader(argreduce_bf16 shaders/argreduce_suffix.comp -DUSE_BF16=1)
omarchy_shader(argreduce_i8 shaders/argreduce_suffix.comp -DUSE_I8=1)
omarchy_shader(argreduce_u8 shaders/argreduce_suffix.comp -DUSE_U8=1)
omarchy_shader(argreduce_i16 shaders/argreduce_suffix.comp -DUSE_I16=1)
omarchy_shader(argreduce_u16 shaders/argreduce_suffix.comp -DUSE_U16=1)
omarchy_shader(argreduce_i32 shaders/argreduce_suffix.comp -DUSE_I32=1)
omarchy_shader(argreduce_u32 shaders/argreduce_suffix.comp -DUSE_U32=1)
omarchy_shader(argreduce_i64 shaders/argreduce_suffix.comp -DUSE_I64=1)
omarchy_shader(argreduce_u64 shaders/argreduce_suffix.comp -DUSE_U64=1)
omarchy_shader(arange_f32 shaders/arange.comp)
omarchy_shader(arange_f16 shaders/arange.comp -DUSE_FP16=1)
omarchy_shader(arange_bf16 shaders/arange.comp -DUSE_BF16=1)
omarchy_shader(arange_i32 shaders/arange.comp -DUSE_I32=1)
omarchy_shader(arange_u32 shaders/arange.comp -DUSE_U32=1)
omarchy_shader(arange_i64 shaders/arange.comp -DUSE_I64=1)
omarchy_shader(arange_u64 shaders/arange.comp -DUSE_U64=1)
omarchy_shader(sort_f32 shaders/sort_suffix.comp)
omarchy_shader(sort_f16 shaders/sort_suffix.comp -DUSE_FP16=1)
omarchy_shader(sort_bf16 shaders/sort_suffix.comp -DUSE_BF16=1)
omarchy_shader(sort_c64 shaders/sort_suffix.comp -DUSE_C64=1)
omarchy_shader(argsort_f32 shaders/sort_suffix.comp -DARGSORT=1)
omarchy_shader(argsort_f16 shaders/sort_suffix.comp -DUSE_FP16=1 -DARGSORT=1)
omarchy_shader(argsort_bf16 shaders/sort_suffix.comp -DUSE_BF16=1 -DARGSORT=1)
omarchy_shader(argsort_c64 shaders/sort_suffix.comp -DUSE_C64=1 -DARGSORT=1)
omarchy_shader(sort_i32 shaders/sort_suffix.comp -DUSE_I32=1)
omarchy_shader(sort_u32 shaders/sort_suffix.comp -DUSE_U32=1)
omarchy_shader(argsort_i32 shaders/sort_suffix.comp -DUSE_I32=1 -DARGSORT=1)
omarchy_shader(argsort_u32 shaders/sort_suffix.comp -DUSE_U32=1 -DARGSORT=1)
omarchy_shader(sort_i8 shaders/sort_suffix.comp -DUSE_I8=1)
omarchy_shader(sort_u8 shaders/sort_suffix.comp -DUSE_U8=1)
omarchy_shader(sort_i16 shaders/sort_suffix.comp -DUSE_I16=1)
omarchy_shader(sort_u16 shaders/sort_suffix.comp -DUSE_U16=1)
omarchy_shader(argsort_i8 shaders/sort_suffix.comp -DUSE_I8=1 -DARGSORT=1)
omarchy_shader(argsort_u8 shaders/sort_suffix.comp -DUSE_U8=1 -DARGSORT=1)
omarchy_shader(argsort_i16 shaders/sort_suffix.comp -DUSE_I16=1 -DARGSORT=1)
omarchy_shader(argsort_u16 shaders/sort_suffix.comp -DUSE_U16=1 -DARGSORT=1)
omarchy_shader(sort_merge_f32 shaders/sort_merge.comp)
omarchy_shader(sort_merge_f16 shaders/sort_merge.comp -DUSE_FP16=1)
omarchy_shader(sort_merge_bf16 shaders/sort_merge.comp -DUSE_BF16=1)
omarchy_shader(sort_merge_i32 shaders/sort_merge.comp -DUSE_I32=1)
omarchy_shader(sort_merge_u32 shaders/sort_merge.comp -DUSE_U32=1)
omarchy_shader(argsort_merge_f32 shaders/sort_merge.comp -DARGSORT=1)
omarchy_shader(argsort_merge_f16 shaders/sort_merge.comp -DUSE_FP16=1 -DARGSORT=1)
omarchy_shader(argsort_merge_bf16 shaders/sort_merge.comp -DUSE_BF16=1 -DARGSORT=1)
omarchy_shader(argsort_merge_c64 shaders/sort_merge.comp -DUSE_C64=1 -DARGSORT=1)
omarchy_shader(argsort_merge_i32 shaders/sort_merge.comp -DUSE_I32=1 -DARGSORT=1)
omarchy_shader(argsort_merge_u32 shaders/sort_merge.comp -DUSE_U32=1 -DARGSORT=1)
omarchy_shader(elementwise_i32 shaders/int_elementwise.comp)
omarchy_shader(elementwise_u32 shaders/int_elementwise.comp -DUSE_U32=1)
omarchy_shader(elementwise_i8 shaders/int_elementwise.comp -DUSE_I8=1)
omarchy_shader(elementwise_u8 shaders/int_elementwise.comp -DUSE_U8=1)
omarchy_shader(elementwise_i16 shaders/int_elementwise.comp -DUSE_I16=1)
omarchy_shader(elementwise_u16 shaders/int_elementwise.comp -DUSE_U16=1)
omarchy_shader(elementwise_i64 shaders/int_elementwise.comp -DUSE_I64=1)
omarchy_shader(elementwise_u64 shaders/int_elementwise.comp -DUSE_U64=1)
omarchy_shader(scan_f32 shaders/scan_suffix.comp)
omarchy_shader(scan_f16 shaders/scan_suffix.comp -DUSE_FP16=1)
omarchy_shader(scan_bf16 shaders/scan_suffix.comp -DUSE_BF16=1)
omarchy_shader(searchsorted_f32 shaders/searchsorted_suffix.comp)
omarchy_shader(searchsorted_f16 shaders/searchsorted_suffix.comp -DUSE_FP16=1)
omarchy_shader(searchsorted_bf16 shaders/searchsorted_suffix.comp -DUSE_BF16=1)
omarchy_shader(searchsorted_i32 shaders/searchsorted_suffix.comp -DUSE_I32=1)
omarchy_shader(searchsorted_u32 shaders/searchsorted_suffix.comp -DUSE_U32=1)
omarchy_shader(random_bits_u32 shaders/random_bits.comp)
omarchy_shader(qmm_f32 shaders/qmm.comp)
omarchy_shader(qmm_f16 shaders/qmm.comp -DUSE_FP16=1)
omarchy_shader(qmm_bf16 shaders/qmm.comp -DUSE_BF16=1)
# FuseDecodeChains: interpreted elementwise-chain kernels (one dispatch
# per fusable compiled or eager chain).
omarchy_shader(fused_chain_f32 shaders/fused_chain.comp)
omarchy_shader(fused_chain_f16 shaders/fused_chain.comp -DUSE_FP16=1)
omarchy_shader(fused_chain_bf16 shaders/fused_chain.comp -DUSE_BF16=1)
# DecodeGemv: dedicated matrix-vector kernels for the decode shape
# (lhs single row); qmm_vec fuses affine dequantization into the dot,
# the same shader compiled with USE_SUBGROUP=1 produces the
# subgroup-reduction variant of the per-column sum (see PROTOCOL in
# tools/subgroup-bench/ for the rationale).
omarchy_shader(qmm_vec_f32 shaders/qmm_vec.comp)
omarchy_shader(qmm_vec_f16 shaders/qmm_vec.comp -DUSE_FP16=1)
omarchy_shader(qmm_vec_bf16 shaders/qmm_vec.comp -DUSE_BF16=1)
omarchy_shader(qmm_vec_subgroup_f32 shaders/qmm_vec.comp -DUSE_SUBGROUP=1)
omarchy_shader(qmm_vec_subgroup_f16 shaders/qmm_vec.comp -DUSE_FP16=1 -DUSE_SUBGROUP=1)
omarchy_shader(qmm_vec_subgroup_bf16 shaders/qmm_vec.comp -DUSE_BF16=1 -DUSE_SUBGROUP=1)
# DecodeQ4Word: transposed affine 4-bit/group-64 GEMV with one packed word
# per lane load and one shared activation staging pass. It is opt-in until
# the M1 timing gate selects it.
# The packed-x subgroup variants run two rows per 32-lane slot (native
# qmv_fast keeps four rows per simdgroup): the x quads load once per block
# and feed both rows, the row loads sit independent in flight, and the
# per-row chain is unchanged, so the digests hold
# (receipts/2026-09-25-jw16-gpu-levers). Four slots keep 8 columns per
# workgroup, the dispatch contract (kGemvColumnsPerGroup).
omarchy_shader(qmm_vec_q4_word_f32 shaders/qmm_vec.comp -DQMM_VEC_Q4_WORD=1)
omarchy_shader(qmm_vec_q4_word_f16 shaders/qmm_vec.comp -DUSE_FP16=1 -DQMM_VEC_Q4_WORD=1)
omarchy_shader(qmm_vec_q4_word_bf16 shaders/qmm_vec.comp -DUSE_BF16=1 -DQMM_VEC_Q4_WORD=1)
omarchy_shader(qmm_vec_q4_word_subgroup_f32 shaders/qmm_vec.comp -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1)
omarchy_shader(qmm_vec_q4_word_subgroup_f16 shaders/qmm_vec.comp -DUSE_FP16=1 -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1 -DROWS_PER_SLOT=2 -DSLOTS_PER_GROUP=4)
omarchy_shader(qmm_vec_q4_word_subgroup_bf16 shaders/qmm_vec.comp -DUSE_BF16=1 -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1 -DROWS_PER_SLOT=2 -DSLOTS_PER_GROUP=4)
# DecodeFusion: the same Q4 word chain against up to three weights per
# dispatch with an optional fused Add epilogue (QMM_VEC_MULTI).
omarchy_shader(qmm_vec_q4_multi_f32 shaders/qmm_vec.comp -DQMM_VEC_Q4_WORD=1 -DQMM_VEC_MULTI=1)
omarchy_shader(qmm_vec_q4_multi_f16 shaders/qmm_vec.comp -DUSE_FP16=1 -DQMM_VEC_Q4_WORD=1 -DQMM_VEC_MULTI=1)
omarchy_shader(qmm_vec_q4_multi_bf16 shaders/qmm_vec.comp -DUSE_BF16=1 -DQMM_VEC_Q4_WORD=1 -DQMM_VEC_MULTI=1)
omarchy_shader(qmm_vec_q4_multi_subgroup_f32 shaders/qmm_vec.comp -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1 -DQMM_VEC_MULTI=1)
omarchy_shader(qmm_vec_q4_multi_subgroup_f16 shaders/qmm_vec.comp -DUSE_FP16=1 -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1 -DQMM_VEC_MULTI=1 -DROWS_PER_SLOT=2 -DSLOTS_PER_GROUP=4)
omarchy_shader(qmm_vec_q4_multi_subgroup_bf16 shaders/qmm_vec.comp -DUSE_BF16=1 -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1 -DQMM_VEC_MULTI=1 -DROWS_PER_SLOT=2 -DSLOTS_PER_GROUP=4)
# Greedy-argmax decode head: bounds, survivors, and flagged full path
# around the same Q4 word subgroup column (QMM_VEC_GREEDY).
omarchy_shader(qmm_vec_greedy_bf16 shaders/qmm_vec.comp -DUSE_BF16=1 -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1 -DQMM_VEC_GREEDY=1)
omarchy_shader(sdpa_decode_native_f16 shaders/sdpa_decode_native.comp)
omarchy_shader(sdpa_decode_native_bf16 shaders/sdpa_decode_native.comp -DBF16_IO=1)
omarchy_shader(sdpa_decode_native_bf16_hd256 shaders/sdpa_decode_native.comp -DBF16_IO=1 -DSDPA_DIM=256)
omarchy_shader(sdpa_decode_native_p1_f16 shaders/sdpa_decode_native_p1.comp)
omarchy_shader(sdpa_decode_native_p2_f16 shaders/sdpa_decode_native_p2.comp)
omarchy_shader(gated_delta_decode_bf16 shaders/gated_delta_decode.comp -DQKV_BF16=1)
omarchy_shader(gated_delta_prefill_bf16 shaders/gated_delta_prefill.comp)
omarchy_shader(gated_delta_prefill_coopmat_bf16 shaders/gated_delta_prefill_coopmat.comp)
# PrefillQmmTile: m-tiled quantized matmul for prefill (matrix_m > 1);
# one workgroup dequantizes each 16 x 16 weight block into shared f32
# once and applies it to a 16-row tile of x, amortizing the inline
# unpack the per-element qmm kernel repeats per output row. Dispatch is
# env-gated in primitives.cpp (MLX_OMARCHY_QMM_TILE). The f16 affine
# 4-bit/group-64 transposed candidate uses MLX_OMARCHY_QMM_TILE_RB;
# set it to 0 to retain this 16x16 tile as the native A/B baseline.
omarchy_shader(qmm_tile_f32 shaders/qmm_tile.comp)
omarchy_shader(qmm_tile_f16 shaders/qmm_tile.comp -DUSE_FP16=1)
omarchy_shader(qmm_tile_rb_f16 shaders/qmm_tile.comp -DUSE_FP16=1 -DREGISTER_BLOCK=1)
omarchy_shader(qmm_tile_rb_precise_f16 shaders/qmm_tile.comp -DUSE_FP16=1 -DREGISTER_BLOCK=1 -DPRECISE_ACCUM=1)
# QmmPrefillCoopmat: the same f16 affine 4-bit/group-64 transposed
# prefill on the 8x8x8 fp32 subgroup cooperative matrix; gated on the
# device capability in primitives.cpp, MLX_OMARCHY_NO_COOPMAT=1 forces
# the register-blocked tile.
omarchy_shader(qmm_coopmat_f16 shaders/qmm_coopmat.comp)
# Finer-M twin of the same shader for a grid too small to fill a wide
# part (primitives.cpp picks by workgroups per GPU core; measured in
# receipts/2026-09-14-qmm-occupancy-tilem/). Same per-output k chain,
# so same output ids.
omarchy_shader(qmm_coopmat_m16_f16 shaders/qmm_coopmat.comp -DTILE_ROWS=16)
# bf16-activation twins of the same shader (pinned mlx-community Qwen3.8
# 4-bit snapshots are bf16-hybrid): identical fp32 coopmat math and k
# chain, bf16 widening of x and RNE bfloat16 drain. Same per-output
# accumulation order as the f16 kernels.
omarchy_shader(qmm_coopmat_bf16 shaders/qmm_coopmat.comp -DX_BF16=1 -DOUT_BF16=1)
omarchy_shader(qmm_coopmat_m16_bf16 shaders/qmm_coopmat.comp -DX_BF16=1 -DOUT_BF16=1 -DTILE_ROWS=16)
# Direct-global-load A twins: the host widens bf16 x to f32 once (exact),
# so the shader coopMatLoads A tiles straight from the f32 buffer and the
# x_s staging disappears; same fp32 coopmat k chain, RNE bfloat16 drain.
omarchy_shader(qmm_coopmat_x32 shaders/qmm_coopmat.comp -DX_F32=1 -DOUT_BF16=1)
omarchy_shader(qmm_coopmat_m16_x32 shaders/qmm_coopmat.comp -DX_F32=1 -DOUT_BF16=1 -DTILE_ROWS=16)
omarchy_shader(qmm_fma_f16 shaders/qmm_fma.comp)
omarchy_shader(qmm_fma_precise_f16 shaders/qmm_fma.comp -DPRECISE_ACCUM=1)
omarchy_shader(matmul_fma_bf16 shaders/matmul_fma_bf16.comp)
omarchy_shader(binary_vec_f16 shaders/binary_vec.comp -DUSE_FP16=1)
omarchy_shader(binary_vec_bf16 shaders/binary_vec.comp -DUSE_BF16=1)
omarchy_shader(matmul_rb_f16 shaders/matmul_rb.comp)
omarchy_shader(swiglu_f16 shaders/swiglu.comp -DUSE_FP16=1)
omarchy_shader(swiglu_bf16 shaders/swiglu.comp -DUSE_BF16=1)
omarchy_shader(fast_trio_norm_f16 shaders/fast_trio.comp -DUSE_FP16=1 -DTRIO_NORM=1)
omarchy_shader(fast_trio_rope_pair_f16 shaders/fast_trio.comp -DUSE_FP16=1 -DTRIO_ROPE_PAIR=1)
omarchy_shader(fast_trio_swiglu_f16 shaders/fast_trio.comp -DUSE_FP16=1 -DTRIO_SWIGLU=1)
omarchy_shader(qmm_tile_bf16 shaders/qmm_tile.comp -DUSE_BF16=1)
omarchy_shader(dequant_f32 shaders/dequant.comp)
omarchy_shader(dequant_f16 shaders/dequant.comp -DUSE_FP16=1)
omarchy_shader(quantize_f32 shaders/quantize.comp)
omarchy_shader(quantize_f16 shaders/quantize.comp -DUSE_FP16=1)
omarchy_shader(quantize_bf16 shaders/quantize.comp -DUSE_BF16=1)
omarchy_shader(dequant_bf16 shaders/dequant.comp -DUSE_BF16=1)
# QuantizeMode: the non-affine quantization modes (mxfp4 / nvfp4 /
# mxfp8) reuse the affine shaders compiled with FP_MODE=1 - byte-packed
# fp4/fp8 element codes, byte scales through a uint word view, no bias.
omarchy_shader(quantize_fp_f32 shaders/quantize.comp -DFP_MODE=1)
omarchy_shader(quantize_fp_f16 shaders/quantize.comp -DFP_MODE=1 -DUSE_FP16=1)
omarchy_shader(quantize_fp_bf16 shaders/quantize.comp -DFP_MODE=1 -DUSE_BF16=1)
omarchy_shader(dequant_fp_f32 shaders/dequant.comp -DFP_MODE=1)
omarchy_shader(dequant_fp_f16 shaders/dequant.comp -DFP_MODE=1 -DUSE_FP16=1)
omarchy_shader(dequant_fp_bf16 shaders/dequant.comp -DFP_MODE=1 -DUSE_BF16=1)
omarchy_shader(qmm_fp_f32 shaders/qmm.comp -DFP_MODE=1)
omarchy_shader(qmm_fp_f16 shaders/qmm.comp -DFP_MODE=1 -DUSE_FP16=1)
omarchy_shader(qmm_fp_bf16 shaders/qmm.comp -DFP_MODE=1 -DUSE_BF16=1)
omarchy_shader(qmm_vec_fp_f32 shaders/qmm_vec.comp -DFP_MODE=1)
omarchy_shader(qmm_vec_fp_f16 shaders/qmm_vec.comp -DFP_MODE=1 -DUSE_FP16=1)
omarchy_shader(qmm_vec_fp_bf16 shaders/qmm_vec.comp -DFP_MODE=1 -DUSE_BF16=1)
omarchy_shader(qmm_vec_subgroup_fp_f32 shaders/qmm_vec.comp -DFP_MODE=1 -DUSE_SUBGROUP=1)
omarchy_shader(qmm_vec_subgroup_fp_f16 shaders/qmm_vec.comp -DFP_MODE=1 -DUSE_FP16=1 -DUSE_SUBGROUP=1)
omarchy_shader(qmm_vec_subgroup_fp_bf16 shaders/qmm_vec.comp -DFP_MODE=1 -DUSE_BF16=1 -DUSE_SUBGROUP=1)
omarchy_shader(qmm_tile_fp_f32 shaders/qmm_tile.comp -DFP_MODE=1)
omarchy_shader(qmm_tile_fp_f16 shaders/qmm_tile.comp -DFP_MODE=1 -DUSE_FP16=1)
omarchy_shader(qmm_tile_fp_bf16 shaders/qmm_tile.comp -DFP_MODE=1 -DUSE_BF16=1)
omarchy_shader(conv_f32 shaders/conv.comp)
omarchy_shader(conv_f16 shaders/conv.comp -DUSE_FP16=1)
omarchy_shader(conv_bf16 shaders/conv.comp -DUSE_BF16=1)

# Wave 5 (indexing and scatter). The u32 variants carry bitwise word
# storage, so float32, int32, and uint32 share them.
omarchy_shader(gather_axis_u32 shaders/gather_axis.comp -DUSE_U32=1)
omarchy_shader(gather_axis_i64 shaders/gather_axis.comp -DUSE_I64=1)
omarchy_shader(gather_axis_f16 shaders/gather_axis.comp -DUSE_FP16=1)
omarchy_shader(gather_axis_bf16 shaders/gather_axis.comp -DUSE_BF16=1)
omarchy_shader(gather_axis_c64 shaders/gather_axis.comp -DUSE_COMPLEX=1)
omarchy_shader(scatter_u32 shaders/scatter.comp -DUSE_U32=1)
omarchy_shader(scatter_f16 shaders/scatter.comp -DUSE_FP16=1)
omarchy_shader(scatter_u8 shaders/scatter.comp -DUSE_U8=1)
omarchy_shader(scatter_i8 shaders/scatter.comp -DUSE_I8=1)
omarchy_shader(scatter_u16 shaders/scatter.comp -DUSE_U16=1)
omarchy_shader(scatter_i16 shaders/scatter.comp -DUSE_I16=1)
omarchy_shader(scatter_bf16 shaders/scatter.comp -DUSE_BF16=1)
omarchy_shader(scatter_c64 shaders/scatter.comp -DUSE_COMPLEX=1)
omarchy_shader(scatter_multi_u32 shaders/scatter.comp -DUSE_U32=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_multi_f16 shaders/scatter.comp -DUSE_FP16=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_multi_bf16 shaders/scatter.comp -DUSE_BF16=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_axis_u32 shaders/scatter_axis.comp -DUSE_U32=1)
omarchy_shader(scatter_axis_f16 shaders/scatter_axis.comp -DUSE_FP16=1)
omarchy_shader(scatter_axis_bf16 shaders/scatter_axis.comp -DUSE_BF16=1)
omarchy_shader(scatter_axis_c64 shaders/scatter_axis.comp -DUSE_COMPLEX=1)
omarchy_shader(masked_scatter_u32 shaders/masked_scatter.comp -DUSE_U32=1)
omarchy_shader(masked_scatter_f16 shaders/masked_scatter.comp -DUSE_FP16=1)
omarchy_shader(masked_scatter_bf16 shaders/masked_scatter.comp -DUSE_BF16=1)
omarchy_shader(scatter_fadd_f32 shaders/scatter.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1)
omarchy_shader(scatter_fadd_f16 shaders/scatter.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1 -DUSE_FP16=1)
omarchy_shader(scatter_fadd_bf16 shaders/scatter.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1 -DUSE_BF16=1)
omarchy_shader(scatter_fadd_multi_f32 shaders/scatter.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_fadd_triple_f32 shaders/scatter.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1 -DUSE_TRIPLE_INDEX=1)
omarchy_shader(scatter_fadd_general_f32 shaders/scatter.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1 -DUSE_GENERAL_INDEX=1)
omarchy_shader(scatter_fcas_f32 shaders/scatter.comp -DUSE_FADD=1)
omarchy_shader(scatter_fcas_f16 shaders/scatter.comp -DUSE_FADD=1 -DUSE_FP16=1)
omarchy_shader(scatter_fcas_bf16 shaders/scatter.comp -DUSE_FADD=1 -DUSE_BF16=1)
omarchy_shader(scatter_fcas_multi_f32 shaders/scatter.comp -DUSE_FADD=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_fcas_triple_f32 shaders/scatter.comp -DUSE_FADD=1 -DUSE_TRIPLE_INDEX=1)
omarchy_shader(scatter_fcas_general_f32 shaders/scatter.comp -DUSE_FADD=1 -DUSE_GENERAL_INDEX=1)
omarchy_shader(scatter_bool shaders/scatter.comp -DUSE_BOOL=1)
omarchy_shader(scatter_bool_multi shaders/scatter.comp -DUSE_BOOL=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_axis_fadd_f32 shaders/scatter_axis.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1)
omarchy_shader(scatter_multi_u8 shaders/scatter.comp -DUSE_U8=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_multi_i8 shaders/scatter.comp -DUSE_I8=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_multi_u16 shaders/scatter.comp -DUSE_U16=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_multi_i16 shaders/scatter.comp -DUSE_I16=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_axis_fadd_f16 shaders/scatter_axis.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1 -DUSE_FP16=1)
omarchy_shader(scatter_axis_fadd_bf16 shaders/scatter_axis.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1 -DUSE_BF16=1)
omarchy_shader(scatter_axis_fcas_f32 shaders/scatter_axis.comp -DUSE_FADD=1)
omarchy_shader(scatter_axis_fcas_f16 shaders/scatter_axis.comp -DUSE_FADD=1 -DUSE_FP16=1)
omarchy_shader(scatter_axis_fcas_bf16 shaders/scatter_axis.comp -DUSE_FADD=1 -DUSE_BF16=1)
omarchy_shader(scatter_axis_bool shaders/scatter_axis.comp -DUSE_BOOL=1)
omarchy_shader(clear_u32 shaders/clear_uint.comp)
omarchy_shader(reduce_general_f32 shaders/reduce_general.comp)
omarchy_shader(reduce_general_f16 shaders/reduce_general.comp -DUSE_FP16=1)
omarchy_shader(reduce_general_bf16 shaders/reduce_general.comp -DUSE_BF16=1)
omarchy_shader(reduce_general_i32 shaders/reduce_general.comp -DUSE_I32=1)
omarchy_shader(reduce_general_u32 shaders/reduce_general.comp -DUSE_U32=1)
omarchy_shader(reduce_general_bool shaders/reduce_general.comp -DUSE_BOOL_INPUT=1 -DUSE_I32=1)
omarchy_shader(reduce_general_i8 shaders/reduce_general.comp -DUSE_I8=1)
omarchy_shader(reduce_general_u8 shaders/reduce_general.comp -DUSE_U8=1)
omarchy_shader(reduce_general_i16 shaders/reduce_general.comp -DUSE_I16=1)
omarchy_shader(reduce_general_u16 shaders/reduce_general.comp -DUSE_U16=1)
omarchy_shader(reduce_general_i64 shaders/reduce_general.comp -DUSE_I64=1)
omarchy_shader(reduce_general_u64 shaders/reduce_general.comp -DUSE_U64=1)
omarchy_shader(reduce_general_complex shaders/reduce_general.comp -DUSE_COMPLEX=1)
omarchy_shader(anyall_f32 shaders/reduce_general.comp -DBOOL_OUT=1)
omarchy_shader(anyall_f16 shaders/reduce_general.comp -DBOOL_OUT=1 -DUSE_FP16=1)
omarchy_shader(anyall_bf16 shaders/reduce_general.comp -DBOOL_OUT=1 -DUSE_BF16=1)
omarchy_shader(anyall_i32 shaders/reduce_general.comp -DBOOL_OUT=1 -DUSE_I32=1)
omarchy_shader(anyall_u32 shaders/reduce_general.comp -DBOOL_OUT=1 -DUSE_U32=1)
omarchy_shader(anyall_bool shaders/reduce_general.comp -DBOOL_OUT=1 -DUSE_BOOL_INPUT=1)
omarchy_shader(scan_general_f32 shaders/scan_general.comp)
omarchy_shader(scan_general_f16 shaders/scan_general.comp -DUSE_FP16=1)
omarchy_shader(scan_general_bf16 shaders/scan_general.comp -DUSE_BF16=1)
omarchy_shader(scan_general_i32 shaders/scan_general.comp -DUSE_I32=1)
omarchy_shader(scan_general_u32 shaders/scan_general.comp -DUSE_U32=1)
omarchy_shader(scan_general_bool shaders/scan_general.comp -DUSE_BOOL=1)
omarchy_shader(scan_general_i8 shaders/scan_general.comp -DUSE_I8=1)
omarchy_shader(scan_general_u8 shaders/scan_general.comp -DUSE_U8=1)
omarchy_shader(scan_general_i16 shaders/scan_general.comp -DUSE_I16=1)
omarchy_shader(scan_general_u16 shaders/scan_general.comp -DUSE_U16=1)
omarchy_shader(scan_general_i64 shaders/scan_general.comp -DUSE_I64=1)
omarchy_shader(scan_general_u64 shaders/scan_general.comp -DUSE_U64=1)
omarchy_shader(scan_general_complex shaders/scan_general.comp -DUSE_COMPLEX=1)
omarchy_shader(hadamard_f32 shaders/hadamard.comp)
omarchy_shader(hadamard_f16 shaders/hadamard.comp -DUSE_FP16=1)
omarchy_shader(hadamard_bf16 shaders/hadamard.comp -DUSE_BF16=1)
omarchy_shader(block_mask_f32 shaders/block_mask.comp)
omarchy_shader(gather_mm_f32 shaders/gather_mm.comp)
omarchy_shader(gather_mm_f16 shaders/gather_mm.comp -DUSE_FP16=1)
omarchy_shader(gather_mm_bf16 shaders/gather_mm.comp -DUSE_BF16=1)
omarchy_shader(segmented_mm_f32 shaders/segmented_mm.comp)
omarchy_shader(segmented_mm_f16 shaders/segmented_mm.comp -DUSE_FP16=1)
omarchy_shader(segmented_mm_bf16 shaders/segmented_mm.comp -DUSE_BF16=1)
omarchy_shader(gather_qmm_f32 shaders/gather_qmm.comp)
omarchy_shader(gather_qmm_f16 shaders/gather_qmm.comp -DUSE_FP16=1)
omarchy_shader(gather_qmm_bf16 shaders/gather_qmm.comp -DUSE_BF16=1)
omarchy_shader(gather_qmm_nb_f32 shaders/gather_qmm.comp -DNO_BIAS=1)
omarchy_shader(gather_qmm_nb_f16 shaders/gather_qmm.comp -DUSE_FP16=1 -DNO_BIAS=1)
omarchy_shader(gather_qmm_nb_bf16 shaders/gather_qmm.comp -DUSE_BF16=1 -DNO_BIAS=1)
omarchy_shader(gather_qmm_nb_fp_f32 shaders/gather_qmm.comp -DFP_MODE=1 -DNO_BIAS=1)
omarchy_shader(gather_qmm_nb_fp_f16 shaders/gather_qmm.comp -DFP_MODE=1 -DUSE_FP16=1 -DNO_BIAS=1)
omarchy_shader(gather_qmm_nb_fp_bf16 shaders/gather_qmm.comp -DFP_MODE=1 -DUSE_BF16=1 -DNO_BIAS=1)
# qqmm: the fp variants plus the nvfp4 output global-scale correction
# (binding 4 carries the float32 global_scale_w word).
omarchy_shader(gather_qmm_nb_fp_hgs_f32 shaders/gather_qmm.comp -DFP_MODE=1 -DNO_BIAS=1 -DUSE_OUT_GLOBAL_SCALE=1)
omarchy_shader(gather_qmm_nb_fp_hgs_f16 shaders/gather_qmm.comp -DFP_MODE=1 -DUSE_FP16=1 -DNO_BIAS=1 -DUSE_OUT_GLOBAL_SCALE=1)
omarchy_shader(gather_qmm_nb_fp_hgs_bf16 shaders/gather_qmm.comp -DFP_MODE=1 -DUSE_BF16=1 -DNO_BIAS=1 -DUSE_OUT_GLOBAL_SCALE=1)
omarchy_shader(fft_f32 shaders/fft_c2c.comp)
omarchy_shader(fft_real_f32 shaders/fft_real_extract.comp)
omarchy_shader(fft_stage_f32 shaders/fft_stage.comp)
# Wave 7 (linear algebra). One workgroup per batch matrix, float32 only,
# matching the upstream CPU dtype contract.
omarchy_shader(linalg_cholesky_f32 shaders/linalg_cholesky.comp)
omarchy_shader(linalg_inverse_f32 shaders/linalg_inverse.comp)
omarchy_shader(linalg_lu_f32 shaders/linalg_lu.comp)
omarchy_shader(linalg_qr_f32 shaders/linalg_qr.comp)
omarchy_shader(linalg_eigh_f32 shaders/linalg_eigh.comp)
omarchy_shader(linalg_eig_f32 shaders/linalg_eig.comp)
omarchy_shader(linalg_svd_f32 shaders/linalg_svd.comp)
omarchy_shader(linalg_svd_finalize_f32 shaders/linalg_svd_finalize.comp)
omarchy_shader(fast_rms_norm_f32 shaders/fast_norm.comp)
omarchy_shader(fast_rms_norm_f16 shaders/fast_norm.comp -DUSE_FP16=1)
omarchy_shader(fast_rms_norm_bf16 shaders/fast_norm.comp -DUSE_BF16=1)
omarchy_shader(fast_norm_gated_bf16 shaders/fast_norm_gated.comp -DUSE_BF16=1)
omarchy_shader(gdn_conv_decode_bf16 shaders/gdn_conv_decode.comp -DUSE_BF16=1)
omarchy_shader(fast_layer_norm_f32 shaders/fast_norm.comp -DLAYER_NORM=1)
omarchy_shader(fast_layer_norm_f16 shaders/fast_norm.comp -DUSE_FP16=1 -DLAYER_NORM=1)
omarchy_shader(fast_layer_norm_bf16 shaders/fast_norm.comp -DUSE_BF16=1 -DLAYER_NORM=1)
omarchy_shader(fast_rms_norm_vjp_dx_f32 shaders/fast_norm_vjp_dx.comp)
omarchy_shader(fast_rms_norm_vjp_dx_f16 shaders/fast_norm_vjp_dx.comp -DUSE_FP16=1)
omarchy_shader(fast_rms_norm_vjp_dx_bf16 shaders/fast_norm_vjp_dx.comp -DUSE_BF16=1)
omarchy_shader(fast_layer_norm_vjp_dx_f32 shaders/fast_norm_vjp_dx.comp -DLAYER_NORM=1)
omarchy_shader(fast_layer_norm_vjp_dx_f16 shaders/fast_norm_vjp_dx.comp -DUSE_FP16=1 -DLAYER_NORM=1)
omarchy_shader(fast_layer_norm_vjp_dx_bf16 shaders/fast_norm_vjp_dx.comp -DUSE_BF16=1 -DLAYER_NORM=1)
omarchy_shader(fast_rms_norm_vjp_dw_f32 shaders/fast_norm_vjp_dw.comp)
omarchy_shader(fast_rms_norm_vjp_dw_f16 shaders/fast_norm_vjp_dw.comp -DUSE_FP16=1)
omarchy_shader(fast_rms_norm_vjp_dw_bf16 shaders/fast_norm_vjp_dw.comp -DUSE_BF16=1)
omarchy_shader(fast_layer_norm_vjp_dw_f32 shaders/fast_norm_vjp_dw.comp -DLAYER_NORM=1)
omarchy_shader(fast_layer_norm_vjp_dw_f16 shaders/fast_norm_vjp_dw.comp -DUSE_FP16=1 -DLAYER_NORM=1)
omarchy_shader(fast_layer_norm_vjp_dw_bf16 shaders/fast_norm_vjp_dw.comp -DUSE_BF16=1 -DLAYER_NORM=1)
omarchy_shader(fast_rms_norm_vjp_dw_reduce_f32 shaders/fast_norm_vjp_dw_reduce.comp)
omarchy_shader(fast_rms_norm_vjp_dw_reduce_f16 shaders/fast_norm_vjp_dw_reduce.comp -DUSE_FP16=1)
omarchy_shader(fast_rms_norm_vjp_dw_reduce_bf16 shaders/fast_norm_vjp_dw_reduce.comp -DUSE_BF16=1)
omarchy_shader(fast_cross_entropy_vjp_f32 shaders/fast_cross_entropy_vjp.comp)
omarchy_shader(fast_cross_entropy_vjp_f16 shaders/fast_cross_entropy_vjp.comp -DUSE_FP16=1)
omarchy_shader(fast_cross_entropy_vjp_bf16 shaders/fast_cross_entropy_vjp.comp -DUSE_BF16=1)
omarchy_shader(fast_cross_entropy_f32 shaders/fast_cross_entropy.comp)
omarchy_shader(fast_cross_entropy_f16 shaders/fast_cross_entropy.comp -DUSE_FP16=1)
omarchy_shader(fast_cross_entropy_bf16 shaders/fast_cross_entropy.comp -DUSE_BF16=1)
omarchy_shader(fast_rope_f32 shaders/fast_rope.comp)
omarchy_shader(fast_rope_f16 shaders/fast_rope.comp -DUSE_FP16=1)
omarchy_shader(fast_rope_bf16 shaders/fast_rope.comp -DUSE_BF16=1)
omarchy_shader(fast_rope_freqs_f32 shaders/fast_rope.comp -DUSE_FREQS=1)
omarchy_shader(fast_rope_freqs_f16 shaders/fast_rope.comp -DUSE_FREQS=1 -DUSE_FP16=1)
omarchy_shader(fast_rope_freqs_bf16 shaders/fast_rope.comp -DUSE_FREQS=1 -DUSE_BF16=1)
omarchy_shader(fp8_to_f32 shaders/fp8.comp -DFP8_TO_FLOAT=1)
omarchy_shader(fp8_to_f16 shaders/fp8.comp -DFP8_TO_FLOAT=1 -DUSE_FP16=1)
omarchy_shader(fp8_to_bf16 shaders/fp8.comp -DFP8_TO_FLOAT=1 -DUSE_BF16=1)
omarchy_shader(fp8_from_f32 shaders/fp8.comp)
omarchy_shader(fp8_from_f16 shaders/fp8.comp -DUSE_FP16=1)
omarchy_shader(fp8_from_bf16 shaders/fp8.comp -DUSE_BF16=1)
# Complex64 transport (Complex64Transport). One element is a vec2
# (re, im) pair of float32 words; no 16-bit storage features are
# involved. The cast matrix mirrors the upstream static_cast rules:
# any real source promotes to (x, 0), and a complex64 source reads
# real() for a real destination.
omarchy_shader(complex_elementwise shaders/complex_elementwise.comp)
omarchy_shader(complex_real shaders/complex_extract.comp)
omarchy_shader(complex_imag shaders/complex_extract.comp -DSELECT_IMAG=1)
omarchy_shader(complex_abs shaders/complex_extract.comp -DABS_OP=1)
omarchy_shader(complex_abs_as_complex shaders/complex_extract.comp -DABS_OP=1 -DABS_AS_COMPLEX=1)
omarchy_shader(cast_f32_c64 shaders/cast.comp -DDEST_C64=1)
omarchy_shader(cast_i32_c64 shaders/cast.comp -DSOURCE_I32=1 -DDEST_C64=1)
omarchy_shader(cast_u32_c64 shaders/cast.comp -DSOURCE_U32=1 -DDEST_C64=1)
omarchy_shader(cast_bool_c64 shaders/cast.comp -DSOURCE_BOOL=1 -DDEST_C64=1)
omarchy_shader(cast_f16_c64 shaders/cast.comp -DSOURCE_FP16=1 -DDEST_C64=1)
omarchy_shader(cast_bf16_c64 shaders/cast.comp -DSOURCE_BF16=1 -DDEST_C64=1)
omarchy_shader(cast_c64_f32 shaders/cast.comp -DSOURCE_C64=1)
# Numeric casts: one blob per source/destination storage-width pair.
# Runtime dtype codes preserve integer signedness, floating conversion,
# and complex real-part projection without a per-dtype kernel matrix.
omarchy_shader(cast_int_w1_w1 shaders/cast_int.comp -DSRC_W=1 -DDST_W=1)
omarchy_shader(cast_int_w1_w2 shaders/cast_int.comp -DSRC_W=1 -DDST_W=2)
omarchy_shader(cast_int_w1_w4 shaders/cast_int.comp -DSRC_W=1 -DDST_W=4)
omarchy_shader(cast_int_w1_w8 shaders/cast_int.comp -DSRC_W=1 -DDST_W=8)
omarchy_shader(cast_int_w2_w1 shaders/cast_int.comp -DSRC_W=2 -DDST_W=1)
omarchy_shader(cast_int_w2_w2 shaders/cast_int.comp -DSRC_W=2 -DDST_W=2)
omarchy_shader(cast_int_w2_w4 shaders/cast_int.comp -DSRC_W=2 -DDST_W=4)
omarchy_shader(cast_int_w2_w8 shaders/cast_int.comp -DSRC_W=2 -DDST_W=8)
omarchy_shader(cast_int_w4_w1 shaders/cast_int.comp -DSRC_W=4 -DDST_W=1)
omarchy_shader(cast_int_w4_w2 shaders/cast_int.comp -DSRC_W=4 -DDST_W=2)
omarchy_shader(cast_int_w4_w4 shaders/cast_int.comp -DSRC_W=4 -DDST_W=4)
omarchy_shader(cast_int_w4_w8 shaders/cast_int.comp -DSRC_W=4 -DDST_W=8)
omarchy_shader(cast_int_w8_w1 shaders/cast_int.comp -DSRC_W=8 -DDST_W=1)
omarchy_shader(cast_int_w8_w2 shaders/cast_int.comp -DSRC_W=8 -DDST_W=2)
omarchy_shader(cast_int_w8_w4 shaders/cast_int.comp -DSRC_W=8 -DDST_W=4)
omarchy_shader(cast_int_w8_w8 shaders/cast_int.comp -DSRC_W=8 -DDST_W=8)
omarchy_shader(fill_c64 shaders/fill.comp -DUSE_COMPLEX64=1)
omarchy_shader(copy_general_c64 shaders/copy_general.comp -DUSE_COMPLEX64=1)
omarchy_shader(fill_u64 shaders/fill.comp -DUSE_U64=1)
omarchy_shader(fill_u16 shaders/fill.comp -DUSE_U16=1)
add_custom_target(omarchy_shaders DEPENDS ${OMARCHY_SHADER_HEADERS})
add_dependencies(mlx omarchy_shaders)

target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/compute.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/capability_sim.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device_info.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/encoder.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fused_chain.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/stream.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ane/manifest.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ane/bundle.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ane/worker.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ane/worker_libane.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/vulkan.cpp)
target_include_directories(mlx PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/shaders")
if(MLX_OMARCHY_ANE_SOURCE_DIR)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ane/runtime.cpp)
  add_executable(
    mlx-omarchy-ane-worker
    ${CMAKE_CURRENT_SOURCE_DIR}/ane/worker_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ane/runtime_worker.cpp)
  target_link_libraries(mlx-omarchy-ane-worker PRIVATE mlx mlx_omarchy_libane)
  set_target_properties(mlx-omarchy-ane-worker PROPERTIES
                        INSTALL_RPATH "$ORIGIN/../lib")
  target_compile_definitions(
    mlx PRIVATE
    MLX_OMARCHY_ANE_LIBRARY_BUILD_PATH="$<TARGET_FILE:mlx>"
    MLX_OMARCHY_ANE_WORKER_BUILD_PATH="$<TARGET_FILE:mlx-omarchy-ane-worker>")
  install(TARGETS mlx-omarchy-ane-worker
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/ane/mlx-omarchy-ane.conf
          DESTINATION lib/tmpfiles.d)
endif()


# The backend dlopens libvulkan at runtime; only headers are needed at compile
# time.
if(TARGET Vulkan::Headers)
  target_link_libraries(mlx PUBLIC Vulkan::Headers)
else()
  target_include_directories(mlx PUBLIC ${Vulkan_INCLUDE_DIRS})
endif()
