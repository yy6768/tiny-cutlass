# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

cutlass_example_add_executable(
  swin_cutlass_test
  swin_cutlass.cu
)

target_include_directories(
  swin_cutlass_test
  PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}"
  "${CMAKE_CURRENT_LIST_DIR}/../flash-attention"
)

set(SWIN_CUTLASS_NVCC_KEEP_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/nvcc")
file(MAKE_DIRECTORY "${SWIN_CUTLASS_NVCC_KEEP_DIR}")

target_compile_options(
  swin_cutlass_test
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<COMPILE_LANGUAGE:CUDA>:--source-in-ptx>
  $<$<COMPILE_LANGUAGE:CUDA>:--keep>
  $<$<COMPILE_LANGUAGE:CUDA>:--keep-dir=${SWIN_CUTLASS_NVCC_KEEP_DIR}>
  $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)

add_custom_target(swin DEPENDS swin_cutlass_test)
