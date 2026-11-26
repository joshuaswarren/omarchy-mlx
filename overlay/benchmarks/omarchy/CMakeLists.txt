# Standalone build for the U2 Omarchy matmul/attention spike.
#
# The spike builds OUTSIDE the main mlx CMake tree so the U2 gate never depends
# on root CMake edits:
#
# cmake -S benchmarks/omarchy -B build-omarchy/spike \ -DMLX_ROOT=$ROOT
# -DMLX_BUILD_DIR=$ROOT/build-omarchy cmake --build build-omarchy/spike
#
# It compiles glsl/*.comp to SPIR-V at build time with glslc (or
# glslangValidator) and links the already-built mlx static library, so the spike
# measures kernels through the same driver stack and reuses the same device
# classification policy as the backend.
cmake_minimum_required(VERSION 3.25)
project(omarchy_matmul_attention_spike CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT MLX_ROOT)
  message(FATAL_ERROR "Set MLX_ROOT to the prepared MLX source root.")
endif()
if(NOT MLX_BUILD_DIR)
  message(
    FATAL_ERROR
      "Set MLX_BUILD_DIR to a configured mlx-omarchy build directory with "
      "MLX_BUILD_OMARCHY=ON (tools/ci/run-omarchy-runtime.sh creates one).")
endif()

find_package(Vulkan REQUIRED)

# Kernel compiler: glslc first, glslangValidator as the fallback. Both ship with
# the Omarchy Vulkan toolchain.
find_program(OMARCHY_GLSLC glslc)
find_program(OMARCHY_GLSLANG glslangValidator)
if(NOT OMARCHY_GLSLC AND NOT OMARCHY_GLSLANG)
  message(
    FATAL_ERROR
      "Neither glslc nor glslangValidator was found. Install the Vulkan "
      "shader compiler (Omarchy: sudo pacman -S vulkan-devel) so the spike "
      "kernels in glsl/*.comp can be compiled.")
endif()

function(omarchy_spike_shader out_name src_file)
  set(out "${CMAKE_CURRENT_BINARY_DIR}/${out_name}.spv")
  if(OMARCHY_GLSLC)
    add_custom_command(
      OUTPUT "${out}"
      COMMAND "${OMARCHY_GLSLC}" -O --target-env=vulkan1.3 ${ARGN}
              "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}" -o "${out}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}"
      VERBATIM)
  else()
    add_custom_command(
      OUTPUT "${out}"
      COMMAND "${OMARCHY_GLSLANG}" -V -Os --target-env vulkan1.3 ${ARGN}
              "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}" -o "${out}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}"
      VERBATIM)
  endif()
  set_property(GLOBAL APPEND PROPERTY OMARCHY_SPIKE_SHADER_OUTPUTS "${out}")
endfunction()

omarchy_spike_shader(gemm_f16 glsl/gemm.comp -DFP16=1)
omarchy_spike_shader(gemm_f32 glsl/gemm.comp)
omarchy_spike_shader(gemv_f32 glsl/gemv_f32.comp)
omarchy_spike_shader(gemv_q4k glsl/gemv_q4k.comp)
omarchy_spike_shader(sdpa_prefill_f16 glsl/sdpa_prefill.comp -DFP16=1)
omarchy_spike_shader(sdpa_prefill_f32 glsl/sdpa_prefill.comp)
omarchy_spike_shader(sdpa_decode_split glsl/sdpa_decode_split.comp)
omarchy_spike_shader(sdpa_decode_merge glsl/sdpa_decode_merge.comp)

get_property(OMARCHY_SPIKE_SHADER_OUTPUTS GLOBAL
             PROPERTY OMARCHY_SPIKE_SHADER_OUTPUTS)

add_custom_target(omarchy_spike_shaders DEPENDS ${OMARCHY_SPIKE_SHADER_OUTPUTS})

find_library(
  MLX_LIBRARY mlx
  PATHS "${MLX_BUILD_DIR}"
  NO_DEFAULT_PATH REQUIRED)

add_executable(omarchy_matmul_attention matmul_attention.cpp)
add_dependencies(omarchy_matmul_attention omarchy_spike_shaders)
target_include_directories(omarchy_matmul_attention PRIVATE "${MLX_ROOT}")
target_link_libraries(omarchy_matmul_attention
                      PRIVATE "${MLX_LIBRARY}" Vulkan::Vulkan dl pthread)
target_compile_definitions(
  omarchy_matmul_attention
  PRIVATE OMARCHY_SPIKE_SPIRV_DIR="${CMAKE_CURRENT_BINARY_DIR}")
