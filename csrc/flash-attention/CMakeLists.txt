
# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set(FLASH_ATTENTION_TEST_DIR "${PROJECT_SOURCE_DIR}/csrc/tests/flash-attention")

set(FLASH_ATTENTION_KERNEL_SOURCES
  00-naive-attention/naive_attention.cu
  01-online-softmax/online_softmax_attention.cu
  02-tiled-online-attention/tiled_online_attention.cu
)

add_library(flash_attention_kernels OBJECT
  ${FLASH_ATTENTION_KERNEL_SOURCES}
)

target_link_libraries(flash_attention_kernels PRIVATE
  tiny_cutlass::cutlass_headers
  CUDA::cudart
)

target_include_directories(flash_attention_kernels PRIVATE
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/csrc"
  "${CMAKE_CURRENT_LIST_DIR}"
)

set(FLASH_ATTENTION_NVCC_KEEP_DIR "${CMAKE_CURRENT_BINARY_DIR}/nvcc")
file(MAKE_DIRECTORY "${FLASH_ATTENTION_NVCC_KEEP_DIR}")

target_compile_options(flash_attention_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<COMPILE_LANGUAGE:CUDA>:--source-in-ptx>
  $<$<COMPILE_LANGUAGE:CUDA>:--keep>
  $<$<COMPILE_LANGUAGE:CUDA>:--keep-dir=${FLASH_ATTENTION_NVCC_KEEP_DIR}>
  $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)

function(flash_attention_add_test_executable TARGET DEFAULT_KERNEL)
  cutlass_example_add_executable(
    ${TARGET}
    "${FLASH_ATTENTION_TEST_DIR}/flash_attention_test.cpp"
    $<TARGET_OBJECTS:flash_attention_kernels>
  )

  target_include_directories(${TARGET} PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/csrc"
    "${CMAKE_CURRENT_LIST_DIR}"
    "${FLASH_ATTENTION_TEST_DIR}"
  )

  target_compile_definitions(${TARGET} PRIVATE
    FLASH_ATTENTION_DEFAULT_KERNEL="${DEFAULT_KERNEL}"
  )

  target_link_libraries(${TARGET} PRIVATE
    tiny_cutlass::cudnn
  )

  tiny_cutlass_copy_cudnn_runtime(${TARGET})
endfunction()

flash_attention_add_test_executable(flash_attention_test "all")
flash_attention_add_test_executable(flash_attention_00_naive_attention_test "00-naive")
flash_attention_add_test_executable(flash_attention_01_online_softmax_attention_test "01-online-softmax")
flash_attention_add_test_executable(flash_attention_02_tiled_online_attention_test "02-tiled-online")

set_target_properties(flash_attention_00_naive_attention_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/00-naive-attention"
)

set_target_properties(flash_attention_01_online_softmax_attention_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/01-online-softmax"
)

set_target_properties(flash_attention_02_tiled_online_attention_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/02-tiled-online-attention"
)

add_custom_target(flash_attention
  DEPENDS flash_attention_test
          flash_attention_00_naive_attention_test
          flash_attention_01_online_softmax_attention_test
          flash_attention_02_tiled_online_attention_test
)
