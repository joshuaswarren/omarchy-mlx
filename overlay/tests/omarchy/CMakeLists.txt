add_executable(omarchy_runtime_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_runtime.cpp)
target_link_libraries(omarchy_runtime_tests PRIVATE mlx doctest)
target_compile_options(omarchy_runtime_tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_runtime_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_runtime_tests)
add_test(NAME omarchy_runtime_tests COMMAND omarchy_runtime_tests)

add_executable(omarchy_copy_offset_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_copy_offsets.cpp)
target_link_libraries(omarchy_copy_offset_tests PRIVATE mlx doctest)
target_compile_options(omarchy_copy_offset_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_copy_offset_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_copy_offset_tests)
add_test(NAME omarchy_copy_offset_tests COMMAND omarchy_copy_offset_tests)

add_executable(omarchy_primitive_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_primitives.cpp)
target_link_libraries(omarchy_primitive_tests PRIVATE mlx doctest)
target_compile_options(omarchy_primitive_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_primitive_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_primitive_tests)
add_test(NAME omarchy_primitive_tests COMMAND omarchy_primitive_tests)

add_executable(omarchy_custom_kernel_translate_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_custom_kernel_translate.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../../mlx/backend/omarchy/custom_kernel.cpp)
target_link_libraries(omarchy_custom_kernel_translate_tests PRIVATE mlx doctest)
# The test-only translate entry lives in custom_kernel.cpp under this define;
# the target compiles that file itself so the export exists at link time.
target_include_directories(omarchy_custom_kernel_translate_tests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../mlx
                           PRIVATE ${CMAKE_BINARY_DIR}/mlx/backend/omarchy/shaders)
target_compile_definitions(omarchy_custom_kernel_translate_tests
                           PRIVATE MLX_OMARCHY_TEST_TRANSLATE)
target_compile_options(omarchy_custom_kernel_translate_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_custom_kernel_translate_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_custom_kernel_translate_tests)
add_test(NAME omarchy_custom_kernel_translate_tests
         COMMAND omarchy_custom_kernel_translate_tests)

add_executable(omarchy_ane_bundle_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/ane/test_bundle.cpp)
target_link_libraries(omarchy_ane_bundle_tests PRIVATE mlx doctest)
target_include_directories(omarchy_ane_bundle_tests
                           PRIVATE ${json_SOURCE_DIR}/single_include/nlohmann)
target_compile_options(omarchy_ane_bundle_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_ane_bundle_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_ane_bundle_tests)
add_test(NAME omarchy_ane_bundle_tests COMMAND omarchy_ane_bundle_tests)
add_executable(omarchy_ane_runtime_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/ane/test_runtime.cpp)
target_link_libraries(omarchy_ane_runtime_tests PRIVATE mlx doctest)
target_compile_options(omarchy_ane_runtime_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_ane_runtime_tests PRIVATE ${SANITIZER_LINK_FLAGS})
doctest_discover_tests(omarchy_ane_runtime_tests)
add_test(NAME omarchy_ane_runtime_tests COMMAND omarchy_ane_runtime_tests)
add_executable(omarchy_ane_worker_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/ane/test_worker.cpp)
target_link_libraries(omarchy_ane_worker_tests PRIVATE mlx doctest)
target_compile_options(omarchy_ane_worker_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_ane_worker_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_ane_worker_tests)
add_test(NAME omarchy_ane_worker_tests COMMAND omarchy_ane_worker_tests)

# Host-only ABI profile gate tests (E3.1). No /proc, /sys, libane, or
# ANE device touched; runs on any architecture so the gate's decision
# logic is covered even when the binary cannot link the worker.
add_executable(omarchy_ane_worker_gate_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/ane/test_worker_gate.cpp)
target_link_libraries(omarchy_ane_worker_gate_tests PRIVATE mlx doctest)
target_compile_options(omarchy_ane_worker_gate_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_ane_worker_gate_tests PRIVATE ${SANITIZER_LINK_FLAGS})
doctest_discover_tests(omarchy_ane_worker_gate_tests)
add_test(NAME omarchy_ane_worker_gate_tests
         COMMAND omarchy_ane_worker_gate_tests)
add_executable(omarchy_ane_tile_layout_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/ane/test_tile_layout.cpp)
target_link_libraries(omarchy_ane_tile_layout_tests PRIVATE mlx doctest)
target_compile_options(omarchy_ane_tile_layout_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_ane_tile_layout_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_ane_tile_layout_tests)
add_test(NAME omarchy_ane_tile_layout_tests
         COMMAND omarchy_ane_tile_layout_tests)


if(TARGET mlx-omarchy-ane-worker)
  add_executable(mlx-omarchy-ane-smoke
                 ${CMAKE_CURRENT_SOURCE_DIR}/ane/runtime_hardware_smoke.cpp)
  target_link_libraries(mlx-omarchy-ane-smoke PRIVATE mlx)
  add_dependencies(mlx-omarchy-ane-smoke mlx-omarchy-ane-worker)
endif()


add_executable(omarchy_kv_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_kv_ops.cpp)
target_link_libraries(omarchy_kv_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_kv_ops_tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_kv_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_kv_ops_tests)

add_test(NAME omarchy_kv_ops_tests COMMAND omarchy_kv_ops_tests)

add_executable(omarchy_error_contract_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_error_contract.cpp)
target_link_libraries(omarchy_error_contract_tests PRIVATE mlx doctest)
target_compile_options(omarchy_error_contract_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_error_contract_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_error_contract_tests)
add_test(NAME omarchy_error_contract_tests
         COMMAND omarchy_error_contract_tests)

add_executable(omarchy_shape_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_shape_ops.cpp)
target_link_libraries(omarchy_shape_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_shape_ops_tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_shape_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_shape_ops_tests)
add_test(NAME omarchy_shape_ops_tests COMMAND omarchy_shape_ops_tests)

add_executable(omarchy_reduce_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_reduce_ops.cpp)
target_link_libraries(omarchy_reduce_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_reduce_ops_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_reduce_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_reduce_ops_tests)
add_test(NAME omarchy_reduce_ops_tests COMMAND omarchy_reduce_ops_tests)

add_executable(omarchy_indexing_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_indexing_ops.cpp)
target_link_libraries(omarchy_indexing_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_indexing_ops_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_indexing_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_indexing_ops_tests)
add_test(NAME omarchy_indexing_ops_tests COMMAND omarchy_indexing_ops_tests)

add_executable(omarchy_matmul_family_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_matmul_family.cpp)
target_link_libraries(omarchy_matmul_family_tests PRIVATE mlx doctest)
target_compile_options(omarchy_matmul_family_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_matmul_family_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_matmul_family_tests)
add_test(NAME omarchy_matmul_family_tests
         COMMAND omarchy_matmul_family_tests)

add_executable(omarchy_distributed_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_ops.cpp)
target_link_libraries(omarchy_distributed_tests PRIVATE mlx doctest)
target_compile_options(omarchy_distributed_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_distributed_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_distributed_tests)
add_test(NAME omarchy_distributed_tests
         COMMAND omarchy_distributed_tests)

add_executable(omarchy_compiled_tape_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_compiled_tape.cpp)
target_link_libraries(omarchy_compiled_tape_tests PRIVATE mlx doctest)
target_compile_options(omarchy_compiled_tape_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_compiled_tape_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_compiled_tape_tests)
add_test(NAME omarchy_compiled_tape_tests
         COMMAND omarchy_compiled_tape_tests)

add_executable(omarchy_fft_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_fft_ops.cpp)
target_link_libraries(omarchy_fft_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_fft_ops_tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_fft_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_fft_ops_tests)
add_test(NAME omarchy_fft_ops_tests COMMAND omarchy_fft_ops_tests)

add_executable(omarchy_fast_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_fast_ops.cpp)
target_link_libraries(omarchy_fast_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_fast_ops_tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_fast_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_fast_ops_tests)
add_test(NAME omarchy_fast_ops_tests COMMAND omarchy_fast_ops_tests)

add_executable(omarchy_gdn_prefill_profile_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_gdn_prefill_profile.cpp)
target_link_libraries(omarchy_gdn_prefill_profile_tests PRIVATE mlx doctest)
target_compile_options(omarchy_gdn_prefill_profile_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_gdn_prefill_profile_tests
                     PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_gdn_prefill_profile_tests)
add_test(NAME omarchy_gdn_prefill_profile_tests
         COMMAND omarchy_gdn_prefill_profile_tests)

add_executable(omarchy_gdn_fast_route_repeat_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_gdn_fast_route_repeat.cpp)
target_link_libraries(omarchy_gdn_fast_route_repeat_tests PRIVATE mlx doctest)
target_compile_options(omarchy_gdn_fast_route_repeat_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_gdn_fast_route_repeat_tests
                     PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_gdn_fast_route_repeat_tests)
add_test(NAME omarchy_gdn_fast_route_repeat_tests
         COMMAND omarchy_gdn_fast_route_repeat_tests)

# GDN legacy-part dispatch: Hk==Hv decode/prefill stay fused under the
# m1-g13-legacy capability simulation (main() pins MLX_OMARCHY_CAPS_SIM).
add_executable(omarchy_gdn_legacy_policy_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_gdn_legacy_policy.cpp)
target_link_libraries(omarchy_gdn_legacy_policy_tests PRIVATE mlx doctest)
target_compile_options(omarchy_gdn_legacy_policy_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_gdn_legacy_policy_tests
                     PRIVATE ${SANITIZER_LINK_FLAGS})
doctest_discover_tests(omarchy_gdn_legacy_policy_tests)
add_test(NAME omarchy_gdn_legacy_policy_tests
         COMMAND omarchy_gdn_legacy_policy_tests)

# Batched GDN decode (B > 1): rows bit-equal to per-row B=1 decode, fused.
add_executable(omarchy_gdn_decode_batch_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_gdn_decode_batch.cpp)
target_link_libraries(omarchy_gdn_decode_batch_tests PRIVATE mlx doctest)
target_compile_options(omarchy_gdn_decode_batch_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_gdn_decode_batch_tests
                     PRIVATE ${SANITIZER_LINK_FLAGS})
doctest_discover_tests(omarchy_gdn_decode_batch_tests)
add_test(NAME omarchy_gdn_decode_batch_tests
         COMMAND omarchy_gdn_decode_batch_tests)

# Batch-shared bf16 decode GEMV (MLX_OMARCHY_QMV_BATCH, default on): per-row bit
# identity over many random datasets, plus trace-proven engagement.
add_executable(omarchy_qmv_batch_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_qmv_batch.cpp)
target_link_libraries(omarchy_qmv_batch_tests PRIVATE mlx doctest)
target_compile_options(omarchy_qmv_batch_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_qmv_batch_tests
                     PRIVATE ${SANITIZER_LINK_FLAGS})
doctest_discover_tests(omarchy_qmv_batch_tests)
add_test(NAME omarchy_qmv_batch_tests
         COMMAND omarchy_qmv_batch_tests)
add_executable(omarchy_gdn_maskless_correctness_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_gdn_maskless_correctness.cpp)
target_compile_definitions(omarchy_gdn_maskless_correctness_tests
                           PRIVATE GDN_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/gdn_coopmat")
target_link_libraries(omarchy_gdn_maskless_correctness_tests PRIVATE mlx doctest)
target_compile_options(omarchy_gdn_maskless_correctness_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_gdn_maskless_correctness_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})
doctest_discover_tests(omarchy_gdn_maskless_correctness_tests)
add_test(NAME omarchy_gdn_maskless_correctness_tests
         COMMAND omarchy_gdn_maskless_correctness_tests)

add_executable(omarchy_linalg_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_linalg_ops.cpp)
target_link_libraries(omarchy_linalg_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_linalg_ops_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_linalg_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_linalg_ops_tests)
add_test(NAME omarchy_linalg_ops_tests COMMAND omarchy_linalg_ops_tests)
add_executable(omarchy_eig_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_eig_ops.cpp)
target_link_libraries(omarchy_eig_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_eig_ops_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_eig_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_eig_ops_tests)
add_test(NAME omarchy_eig_ops_tests COMMAND omarchy_eig_ops_tests)


add_executable(omarchy_take_fill_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_take_negative_fill.cpp)
target_link_libraries(omarchy_take_fill_tests PRIVATE mlx doctest)
target_compile_options(omarchy_take_fill_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_take_fill_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_take_fill_tests)
add_test(NAME omarchy_take_fill_tests COMMAND omarchy_take_fill_tests)

add_executable(omarchy_take_bool_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_take_bool.cpp)
target_link_libraries(omarchy_take_bool_tests PRIVATE mlx doctest)
target_compile_options(omarchy_take_bool_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_take_bool_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_take_bool_tests)
add_test(NAME omarchy_take_bool_tests COMMAND omarchy_take_bool_tests)
# test_eq_math and test_sdpa_norm_regression are still owned by their
# respective authors. omarchy_topk_ops_tests is UNREGISTERED (2026-09-02,
# IndexingTopkRepair): its single case only exercised WideRowTopK's
# reverted wide-row kernel, and the kernel's shader source was lost in
# the stash sweep, so the suite had nothing real to test. The wide-row
# capability gap is pinned by the refusal test in test_indexing_ops.cpp
# ("argpartition wide rows keep the named refusal..."), which flips to a
# value test when a correct selection kernel lands. Placeholder stubs
# that assert nothing are forbidden: unregister the target instead.
# See hub IRC Main 2026-09-02 for the no-placeholder rule.

add_executable(omarchy_conv_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_conv_general.cpp)
target_link_libraries(omarchy_conv_tests PRIVATE mlx doctest)
target_compile_options(omarchy_conv_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_conv_tests PRIVATE ${SANITIZER_LINK_FLAGS})

add_executable(omarchy_conv_gemm_decomp_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_conv_gemm_decomp.cpp)
target_link_libraries(omarchy_conv_gemm_decomp_tests PRIVATE mlx doctest)
target_compile_options(omarchy_conv_gemm_decomp_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_conv_gemm_decomp_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_conv_gemm_decomp_tests)
add_test(NAME omarchy_conv_gemm_decomp_tests
         COMMAND omarchy_conv_gemm_decomp_tests)

add_executable(omarchy_trig_reduction_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_trig_reduction.cpp)
target_link_libraries(omarchy_trig_reduction_tests PRIVATE mlx doctest)
target_compile_options(omarchy_trig_reduction_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_trig_reduction_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_trig_reduction_tests)
add_test(NAME omarchy_trig_reduction_tests
         COMMAND omarchy_trig_reduction_tests)

doctest_discover_tests(omarchy_conv_tests)
add_test(NAME omarchy_conv_tests COMMAND omarchy_conv_tests)

add_executable(omarchy_complex_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_complex_ops.cpp)
target_link_libraries(omarchy_complex_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_complex_ops_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_complex_ops_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_complex_ops_tests)
add_test(NAME omarchy_complex_ops_tests
         COMMAND omarchy_complex_ops_tests)

add_executable(omarchy_select_layout_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_select_ops.cpp)
target_link_libraries(omarchy_select_layout_tests PRIVATE mlx doctest)
target_compile_options(omarchy_select_layout_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_select_layout_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_select_layout_tests)
add_test(NAME omarchy_select_layout_tests
         COMMAND omarchy_select_layout_tests)

add_executable(omarchy_fft_general_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_fft_general.cpp)
target_link_libraries(omarchy_fft_general_tests PRIVATE mlx doctest)
target_compile_options(omarchy_fft_general_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_fft_general_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_fft_general_tests)
add_test(NAME omarchy_fft_general_tests COMMAND omarchy_fft_general_tests)

add_executable(omarchy_fast_regression_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_sdpa_norm_regression.cpp)
target_link_libraries(omarchy_fast_regression_tests PRIVATE mlx doctest)
target_compile_options(omarchy_fast_regression_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_fast_regression_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_fast_regression_tests)
add_test(NAME omarchy_fast_regression_tests COMMAND omarchy_fast_regression_tests)

# F16 causal ragged SDPA: the 2026-09-11 upstream wrong values and NaN at
# qsl=127 ksl=65 (receipts/2026-09-11-f16-sdpa-causal).
add_executable(omarchy_sdpa_causal_ragged_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_sdpa_causal_ragged.cpp)
target_link_libraries(omarchy_sdpa_causal_ragged_tests PRIVATE mlx doctest)
target_compile_options(omarchy_sdpa_causal_ragged_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_sdpa_causal_ragged_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_sdpa_causal_ragged_tests)
add_test(NAME omarchy_sdpa_causal_ragged_tests
         COMMAND omarchy_sdpa_causal_ragged_tests)

# ScatterDeterminism: float scatter reductions (hardware fp32 atomicAdd
# for Sum, CAS fp32 multiply for Prod, fp32 scratch for half types),
# bool Scatter (packed-byte deterministic RMW), and non-dense layout
# materialization for indices and updates.
add_executable(omarchy_scatter_determinism_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_scatter_determinism.cpp)
target_link_libraries(omarchy_scatter_determinism_tests PRIVATE mlx doctest)
target_compile_options(omarchy_scatter_determinism_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_scatter_determinism_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_scatter_determinism_tests)
add_test(NAME omarchy_scatter_determinism_tests
         COMMAND omarchy_scatter_determinism_tests)

# FixEqualNanAndMath: W2 equal_nan flag plumbing on Equal + the bool
# comparison sextet through logical_or.comp, W5 log10 exactness
# (f32-exact powers of ten), and W8 sin/cos/tan on the common and
# full-magnitude paths.
add_executable(omarchy_eq_math_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_eq_math.cpp)
target_link_libraries(omarchy_eq_math_tests PRIVATE mlx doctest)
target_compile_options(omarchy_eq_math_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_eq_math_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_eq_math_tests)
add_test(NAME omarchy_eq_math_tests COMMAND omarchy_eq_math_tests)

# FuseDecodeChains: decode dispatch counter harness (host-side trace
# counters over a faithful Qwen2.5-0.5B decode step) and fused-chain
# compiled-versus-eager equivalence tests.
add_executable(omarchy_decode_dispatch_count
               ${CMAKE_CURRENT_SOURCE_DIR}/decode_dispatch_count.cpp)
target_link_libraries(omarchy_decode_dispatch_count PRIVATE mlx doctest)
target_compile_options(omarchy_decode_dispatch_count
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_decode_dispatch_count
                    PRIVATE ${SANITIZER_LINK_FLAGS})
add_test(NAME omarchy_decode_dispatch_count
         COMMAND omarchy_decode_dispatch_count)

# Bf16FusedDecode: the fused single-query decode SDPA stays one dispatch,
# agrees with the f32-score composition it replaced, and refuses uncovered
# decode shapes by falling through to that composition.
add_executable(omarchy_sdpa_decode_fused_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_sdpa_decode_fused.cpp)
target_link_libraries(omarchy_sdpa_decode_fused_tests PRIVATE mlx doctest)
target_compile_options(omarchy_sdpa_decode_fused_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_sdpa_decode_fused_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})
add_test(NAME omarchy_sdpa_decode_fused_tests
         COMMAND omarchy_sdpa_decode_fused_tests)

# SdpaPrefillFlashBF16Hd128: the flash-style long-L prefill route agrees
# with the f32-score composition at bf16 storage granularity, splits
# dispatches per q tile, and refuses masks/causal to the composition.
add_executable(omarchy_sdpa_prefill_flash_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_sdpa_prefill_flash.cpp)
target_link_libraries(omarchy_sdpa_prefill_flash_tests PRIVATE mlx doctest)
target_compile_options(omarchy_sdpa_prefill_flash_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_sdpa_prefill_flash_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})
add_test(NAME omarchy_sdpa_prefill_flash_tests
         COMMAND omarchy_sdpa_prefill_flash_tests)

add_executable(omarchy_int8_matmul_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_int8_matmul.cpp)
target_link_libraries(omarchy_int8_matmul_tests PRIVATE mlx doctest)
target_compile_options(omarchy_int8_matmul_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_int8_matmul_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})
add_test(NAME omarchy_int8_matmul_tests
         COMMAND omarchy_int8_matmul_tests)
# FuseDecodeChains: fused elementwise-chain compiled-versus-eager
# equivalence tests plus dispatch-collapse and bf16-refusal guards.
add_executable(omarchy_fused_chain_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_fused_chain.cpp)
target_link_libraries(omarchy_fused_chain_tests PRIVATE mlx doctest)
target_compile_options(omarchy_fused_chain_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_fused_chain_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_fused_chain_tests)
add_test(NAME omarchy_fused_chain_tests COMMAND omarchy_fused_chain_tests)

# WrongValueSweep: float-to-bool casts decide from encoded bits (driver
# denorm flush turned f32/bf16 subnormals into False), and same-mode
# convolve / padded pool keep exact boundary bytes under allocator
# churn (the scalar-fill drain the batching commit removed).
add_executable(omarchy_wrong_value_sweep_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_wrong_value_sweep.cpp)
target_link_libraries(omarchy_wrong_value_sweep_tests PRIVATE mlx doctest)
target_compile_options(omarchy_wrong_value_sweep_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_wrong_value_sweep_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_wrong_value_sweep_tests)
add_test(NAME omarchy_wrong_value_sweep_tests
         COMMAND omarchy_wrong_value_sweep_tests)

# TwoRankDistributed: the five distributed primitives value-proven on a
# real two-rank ring group over loopback. The doctest binary is ONE rank's
# process and fails its own group-size guards when run alone, so the
# registered test is the launcher script, which runs ranks 0 and 1 against
# a localhost hostfile and requires both green. No
# doctest_discover_tests here: discovery would run the binary without
# MLX_RANK/MLX_HOSTFILE and fail by design.
add_executable(omarchy_two_rank_harness
               ${CMAKE_CURRENT_SOURCE_DIR}/distributed/test_two_rank.cpp)
target_link_libraries(omarchy_two_rank_harness PRIVATE mlx doctest)
target_compile_options(omarchy_two_rank_harness
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_two_rank_harness PRIVATE ${SANITIZER_LINK_FLAGS})

add_test(NAME omarchy_two_rank_harness
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/distributed/run-two-rank.sh
                 $<TARGET_FILE:omarchy_two_rank_harness>)

# CapabilitySimulation: one binary, one MLX_OMARCHY_CAPS_SIM profile per
# invocation (test_capability_simulation.cpp). Each registered test runs
# the real kernels on whatever device the host has (llvmpipe for
# software development) under one simulated capability profile and
# asserts the intended outcome: correct result through the gate-chosen
# fallback, or the loud named refusal for a simulated axis the hardware
# cannot back. ALLOW_NON_APPLE is set here because these tests exist to
# run on non-Apple software drivers; on Omarchy hardware the flag only
# widens acceptance of the same Honeykrisp device.
add_executable(omarchy_capability_sim_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_capability_simulation.cpp)
target_link_libraries(omarchy_capability_sim_tests PRIVATE mlx doctest)
target_compile_options(omarchy_capability_sim_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_capability_sim_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

foreach(_capsim_profile IN ITEMS
            m1-honeykrisp-fork
            m1-stock-no-coopmat
            subgroup-size-64
            small-shared-memory
            no-cooperative-matrix
            m1-g13-legacy)
  add_test(NAME omarchy_capability_sim_tests/${_capsim_profile}
           COMMAND omarchy_capability_sim_tests ${_capsim_profile})
  set_tests_properties(omarchy_capability_sim_tests/${_capsim_profile}
                       PROPERTIES
                       ENVIRONMENT "MLX_OMARCHY_ALLOW_NON_APPLE=1"
                       TIMEOUT 1800)
endforeach()

# HwProbe device_info parity (matrix lane #7, rows A15/A19/B9). Tests
# the public keys oMLX (dflash, oq, memory_monitor) and TensorFold
# (kernels/device.py, threads.py, prefill_mm.py) read on real Apple
# silicon and on a dev-box llvmpipe path (MLX_OMARCHY_ALLOW_NON_APPLE=1).
add_executable(omarchy_device_info_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_device_info_real_values.cpp)
target_link_libraries(omarchy_device_info_tests PRIVATE mlx doctest)
target_compile_options(omarchy_device_info_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_device_info_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_device_info_tests)
add_test(NAME omarchy_device_info_tests COMMAND omarchy_device_info_tests)
set_tests_properties(omarchy_device_info_tests
                     PROPERTIES
                     ENVIRONMENT "MLX_OMARCHY_ALLOW_NON_APPLE=1"
                     TIMEOUT 120)
