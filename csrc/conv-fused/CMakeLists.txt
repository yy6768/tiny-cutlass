cmake_minimum_required(VERSION 3.24)

if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "89")
  message(FATAL_ERROR "conv-fused currently supports only SM89; configure with CMAKE_CUDA_ARCHITECTURES=89")
endif()

add_library(conv_fused_core STATIC
  ops/conv1x1_relu_conv1x1.cu
  device/conv1x1_relu_conv1x1.cu
  fp8/conv1x1_relu_conv1x1_relu_fp8/ops/conv1x1_relu_conv1x1_relu_fp8.cu
  fp8/conv1x1_relu_conv1x1_relu_fp8/device/conv1x1_relu_conv1x1_relu_fp8.cu
)

target_include_directories(conv_fused_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/ops"
  "${CMAKE_SOURCE_DIR}/3rdparty/cutlass/examples/13_two_tensor_op_fusion"
)

target_link_libraries(conv_fused_core PUBLIC
  tiny_cutlass::cutlass_headers
  CUDA::cudart
)

target_compile_options(conv_fused_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/bigobj>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/bigobj>
)

set_target_properties(conv_fused_core PROPERTIES
  CUDA_ARCHITECTURES 89
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF
)

add_custom_target(conv_fused DEPENDS conv_fused_core)

tiny_cutlass_add_cutlass_executable(
  conv_pool_test
  "${CMAKE_SOURCE_DIR}/csrc/tests/conv-fused/conv_pool_test.cu"
)
target_link_libraries(conv_pool_test PRIVATE conv_fused_core)

tiny_cutlass_add_cutlass_executable(
  conv_fused_relu_fp8_test
  "${CMAKE_SOURCE_DIR}/csrc/tests/conv-fused/conv_relu_fp8_test.cu"
)
target_link_libraries(conv_fused_relu_fp8_test PRIVATE conv_fused_core)

set_target_properties(
  conv_pool_test
  conv_fused_relu_fp8_test
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/conv-fused"
  CUDA_ARCHITECTURES 89
  CUDA_RUNTIME_LIBRARY Shared
)
