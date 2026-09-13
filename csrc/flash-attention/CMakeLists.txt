
# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set(FLASH_ATTENTION_TEST_DIR "${PROJECT_SOURCE_DIR}/csrc/tests/flash-attention")

set(FLASH_ATTENTION_KERNEL_SOURCES
  00-naive-attention/naive_attention.cu
  01-online-softmax/online_softmax_attention.cu
  02-split-kv/device/flash_attn.cu
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
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
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
flash_attention_add_test_executable(naive_attention "00-naive")
flash_attention_add_test_executable(online_softmax_attention "01-online-softmax")
flash_attention_add_test_executable(split_kv_attention "02-split-kv")

# CuTe warm-up GEMM. Deliberately a separate executable rather than another
# entry in the shared Kernel interface: it is not attention, so it does not
# belong in `flash_attention_test --kernel=all`. Reference is cuBLAS, which
# ships with the CUDA Toolkit, so there is no optional dependency here.
cutlass_example_add_executable(
  cute_gemm
  "${FLASH_ATTENTION_TEST_DIR}/cute_gemm_test.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/cute-gemm/simple_gemm.cu"
)

target_include_directories(cute_gemm PRIVATE
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/csrc"
  "${CMAKE_CURRENT_LIST_DIR}"
  "${FLASH_ATTENTION_TEST_DIR}"
)

target_link_libraries(cute_gemm PRIVATE CUDA::cublas)

target_compile_options(cute_gemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<COMPILE_LANGUAGE:CUDA>:--source-in-ptx>
  $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)

add_custom_target(flash_attention
  DEPENDS flash_attention_test
          naive_attention
          online_softmax_attention
          split_kv_attention
          cute_gemm
)
