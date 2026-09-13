cmake_minimum_required(VERSION 3.24)

if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "89")
  message(FATAL_ERROR "conv-fused currently supports only SM89; configure with CMAKE_CUDA_ARCHITECTURES=89")
endif()

# Optional cuDNN cross-check reference for conv-fused tests. Off by default so a
# build without cuDNN installed is unaffected; turn on with
# -DTINY_CUTLASS_CONV_FUSED_USE_CUDNN=ON (requires CUDNN_PATH, see cmake/CUDNN.cmake).
option(TINY_CUTLASS_CONV_FUSED_USE_CUDNN
  "Cross-check conv-fused tests against a cuDNN reference" OFF)
if(TINY_CUTLASS_CONV_FUSED_USE_CUDNN)
  include("${CMAKE_SOURCE_DIR}/cmake/CUDNN.cmake")
endif()

function(add_conv_fused_family NAME)
  add_library(${NAME}_core STATIC ${ARGN})

  target_include_directories(${NAME}_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )

  target_link_libraries(${NAME}_core PUBLIC
    tiny_cutlass::cutlass_headers
    CUDA::cudart
  )

  target_compile_options(${NAME}_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/bigobj>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/bigobj>
  )

  set_target_properties(${NAME}_core PROPERTIES
    CUDA_ARCHITECTURES 89
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF
  )
endfunction()

add_conv_fused_family(conv1x1_dual
  conv1x1_dual/ops/conv1x1_dual.cu
  conv1x1_dual/device/conv1x1_dual.cu
)

add_conv_fused_family(conv1x1_upsample
  conv1x1_upsample/ops/conv1x1_upsample.cu
  conv1x1_upsample/device/conv1x1_upsample.cu
)

add_custom_target(conv_fused DEPENDS
  conv1x1_dual_core
  conv1x1_upsample_core
)

tiny_cutlass_add_cutlass_executable(
  conv1x1_dual
  "${CMAKE_SOURCE_DIR}/csrc/tests/conv-fused/conv1x1_dual/conv1x1_dual.cu"
)
target_link_libraries(conv1x1_dual PRIVATE conv1x1_dual_core)
target_include_directories(conv1x1_dual PRIVATE "${CMAKE_SOURCE_DIR}/csrc/tests/common")
if(TINY_CUTLASS_CONV_FUSED_USE_CUDNN)
  # The cuDNN reference is a HOST C++ TU (cudnn_frontend.h overflows nvcc's
  # cudafe++ if pulled into a .cu). Build it separately and link it in.
  add_library(conv_fused_cudnn_ref STATIC
    "${CMAKE_SOURCE_DIR}/csrc/tests/common/cudnn_conv_reference.cpp")
  target_include_directories(conv_fused_cudnn_ref PUBLIC
    "${CMAKE_SOURCE_DIR}/csrc/tests/common")
  target_link_libraries(conv_fused_cudnn_ref PUBLIC tiny_cutlass::cudnn)
  set_target_properties(conv_fused_cudnn_ref PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_link_libraries(conv1x1_dual PRIVATE conv_fused_cudnn_ref)
  target_compile_definitions(conv1x1_dual PRIVATE TINY_CUTLASS_WITH_CUDNN=1)
  tiny_cutlass_copy_cudnn_runtime(conv1x1_dual)
endif()

tiny_cutlass_add_cutlass_executable(
  conv1x1_upsample
  "${CMAKE_SOURCE_DIR}/csrc/tests/conv-fused/conv1x1_upsample/conv1x1_upsample.cu"
)
target_link_libraries(conv1x1_upsample PRIVATE conv1x1_upsample_core)

set_target_properties(
  conv1x1_dual
  conv1x1_upsample
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/conv-fused"
  CUDA_ARCHITECTURES 89
  CUDA_RUNTIME_LIBRARY Shared
)
