
# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

cutlass_example_add_executable(
  flash_attention_01_online_softmax_test
  online_softmax_test.cu
)

cutlass_example_add_executable(
  flash_attention_01_online_softmax_attention_test
  online_softmax_attention.cu
)

set(FLASH_ATTENTION_ONLINE_SOFTMAX_NVCC_KEEP_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/nvcc")
file(MAKE_DIRECTORY "${FLASH_ATTENTION_ONLINE_SOFTMAX_NVCC_KEEP_DIR}")

target_compile_options(
  flash_attention_01_online_softmax_attention_test
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<COMPILE_LANGUAGE:CUDA>:--source-in-ptx>
  $<$<COMPILE_LANGUAGE:CUDA>:--keep>
  $<$<COMPILE_LANGUAGE:CUDA>:--keep-dir=${FLASH_ATTENTION_ONLINE_SOFTMAX_NVCC_KEEP_DIR}>
  $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)
