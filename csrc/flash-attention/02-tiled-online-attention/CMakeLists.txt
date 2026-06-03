
# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

cutlass_example_add_executable(
  flash_attention_02_tiled_online_attention_test
  tiled_online_attention.cu
)

target_include_directories(
  flash_attention_02_tiled_online_attention_test
  PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/.."
)

set(FLASH_ATTENTION_TILED_ONLINE_NVCC_KEEP_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/nvcc")
file(MAKE_DIRECTORY "${FLASH_ATTENTION_TILED_ONLINE_NVCC_KEEP_DIR}")

target_compile_options(
  flash_attention_02_tiled_online_attention_test
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<COMPILE_LANGUAGE:CUDA>:--source-in-ptx>
  $<$<COMPILE_LANGUAGE:CUDA>:--keep>
  $<$<COMPILE_LANGUAGE:CUDA>:--keep-dir=${FLASH_ATTENTION_TILED_ONLINE_NVCC_KEEP_DIR}>
  $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)
