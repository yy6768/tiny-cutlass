cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "Target architecture")
  endif()
  project(fused_swin_layer LANGUAGES CXX CUDA)
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(80|89)$")
  message(FATAL_ERROR "Select exactly one architecture: 80 or 89")
endif()
get_filename_component(FUSED_SWIN_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_executable(fused_swin_layer "${FUSED_SWIN_ROOT}/csrc/tests/fused_swin_layer/layer.cu")
target_include_directories(fused_swin_layer PRIVATE
  "${FUSED_SWIN_ROOT}/csrc" "${FUSED_SWIN_ROOT}/csrc/tests/common"
  "${FUSED_SWIN_ROOT}/3rdparty/cutlass/include" "${FUSED_SWIN_ROOT}/3rdparty/cutlass/tools/util/include")
target_compile_definitions(fused_swin_layer PRIVATE FUSED_SWIN_ARCH=${CMAKE_CUDA_ARCHITECTURES})
target_compile_options(fused_swin_layer PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info;--expt-relaxed-constexpr;-Xptxas=-v>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/bigobj>)
set_target_properties(fused_swin_layer PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17
  CUDA_SEPARABLE_COMPILATION OFF CUDA_RUNTIME_LIBRARY Shared
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/fused_swin_layer")
