cmake_minimum_required(VERSION 3.24)

if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "89")
  message(FATAL_ERROR "conv-fused currently supports only SM89; configure with CMAKE_CUDA_ARCHITECTURES=89")
endif()

option(TINY_CUTLASS_BUILD_CONV_FUSED_TENSORRT_PLUGIN
       "Build the conv-fused TensorRT IPluginV3 binding" ON)

add_library(conv_fused_core STATIC
  ops/conv1x1_relu_conv1x1.cu
  ops/conv_relu_pool.cu
  device/conv1x1_relu_conv1x1.cu
  device/conv_relu_pool.cu
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

set(CONV_FUSED_TARGETS conv_fused_core)

if(TINY_CUTLASS_BUILD_CONV_FUSED_TENSORRT_PLUGIN)
  include("${CMAKE_SOURCE_DIR}/cmake/TensorRT.cmake")

  add_library(conv_fused_plugin SHARED
    binding/tensorrt/conv_relu_pool.cpp
  )

  target_include_directories(conv_fused_plugin PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )

  target_link_libraries(conv_fused_plugin PRIVATE
    conv_fused_core
    tiny_cutlass::tensorrt
    CUDA::cudart
  )

  target_compile_options(conv_fused_plugin PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/bigobj>
  )

  set_target_properties(conv_fused_plugin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/conv-fused"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/conv-fused"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/conv-fused"
  )

  tiny_cutlass_copy_tensorrt_runtime(conv_fused_plugin)

  list(APPEND CONV_FUSED_TARGETS conv_fused_plugin)

  tiny_cutlass_add_cutlass_executable(
    conv_relu_pool_plugin
    "${CMAKE_SOURCE_DIR}/csrc/tests/conv-fused/conv_relu_pool_plugin.cpp"
    binding/tensorrt/conv_relu_pool.cpp
  )
  target_include_directories(conv_relu_pool_plugin PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(conv_relu_pool_plugin PRIVATE
    conv_fused_core
    tiny_cutlass::tensorrt
    CUDA::cudart
  )
  target_compile_options(conv_relu_pool_plugin PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/bigobj>
  )
  tiny_cutlass_copy_tensorrt_runtime(conv_relu_pool_plugin)

  tiny_cutlass_add_cutlass_executable(
    conv_relu_pool_trt
    "${CMAKE_SOURCE_DIR}/csrc/tests/conv-fused/conv_relu_pool_trt.cpp"
    binding/tensorrt/conv_relu_pool.cpp
  )
  target_include_directories(conv_relu_pool_trt PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(conv_relu_pool_trt PRIVATE
    conv_fused_core
    tiny_cutlass::tensorrt
    CUDA::cudart
  )
  target_compile_options(conv_relu_pool_trt PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/bigobj>
  )
  tiny_cutlass_copy_tensorrt_runtime(conv_relu_pool_trt)

  tiny_cutlass_add_cutlass_executable(
    conv1x1_relu_conv1x1_relu_trt
    "${CMAKE_SOURCE_DIR}/csrc/tests/conv-fused/conv1x1_relu_conv1x1_relu_trt.cpp"
    binding/tensorrt/conv1x1_relu_conv1x1_relu.cpp
  )
  target_include_directories(conv1x1_relu_conv1x1_relu_trt PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(conv1x1_relu_conv1x1_relu_trt PRIVATE
    conv_fused_core
    tiny_cutlass::tensorrt
    CUDA::cudart
  )
  target_compile_options(conv1x1_relu_conv1x1_relu_trt PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/bigobj>
  )
  tiny_cutlass_copy_tensorrt_runtime(conv1x1_relu_conv1x1_relu_trt)
endif()

add_custom_target(conv_fused DEPENDS ${CONV_FUSED_TARGETS})

tiny_cutlass_add_cutlass_executable(
  conv_pool
  "${CMAKE_SOURCE_DIR}/csrc/tests/conv-fused/conv_pool.cu"
)
target_link_libraries(conv_pool PRIVATE conv_fused_core)

tiny_cutlass_add_cutlass_executable(
  conv_relu_pool
  "${CMAKE_SOURCE_DIR}/csrc/tests/conv-fused/conv_relu_pool.cu"
)
target_link_libraries(conv_relu_pool PRIVATE conv_fused_core)

tiny_cutlass_add_cutlass_executable(
  conv1x1_relu_conv1x1_relu
  "${CMAKE_SOURCE_DIR}/csrc/tests/conv-fused/conv1x1_relu_conv1x1_relu.cu"
)
target_link_libraries(conv1x1_relu_conv1x1_relu PRIVATE conv_fused_core)

set_target_properties(
  conv_pool
  conv_relu_pool
  conv1x1_relu_conv1x1_relu
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/conv-fused"
  CUDA_ARCHITECTURES 89
  CUDA_RUNTIME_LIBRARY Shared
)

if(TARGET conv_relu_pool_plugin)
  set_target_properties(
    conv_relu_pool_plugin
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/conv-fused"
    CUDA_ARCHITECTURES 89
    CUDA_RUNTIME_LIBRARY Shared
  )
endif()

if(TARGET conv_relu_pool_trt)
  set_target_properties(
    conv_relu_pool_trt
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/conv-fused"
    CUDA_ARCHITECTURES 89
    CUDA_RUNTIME_LIBRARY Shared
  )
endif()

if(TARGET conv1x1_relu_conv1x1_relu_trt)
  set_target_properties(
    conv1x1_relu_conv1x1_relu_trt
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/conv-fused"
    CUDA_ARCHITECTURES 89
    CUDA_RUNTIME_LIBRARY Shared
  )
endif()
