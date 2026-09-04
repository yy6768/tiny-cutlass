# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set(EVT_TEST_DIR "${PROJECT_SOURCE_DIR}/csrc/tests/evt")

tiny_cutlass_add_cutlass_executable(
  evt_gemm_broadcast
  "${EVT_TEST_DIR}/gemm_broadcast.cu"
)

set(EVT_TARGETS
    evt_gemm_broadcast)

foreach(EVT_TARGET IN LISTS EVT_TARGETS)
  target_include_directories(${EVT_TARGET} PRIVATE
    "${PROJECT_SOURCE_DIR}/csrc"
  )
  target_compile_options(${EVT_TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
    $<$<COMPILE_LANGUAGE:CUDA>:--source-in-ptx>
    $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  )
  set_target_properties(${EVT_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/evt"
    CUDA_RUNTIME_LIBRARY Shared
  )
endforeach()

add_custom_target(evt DEPENDS ${EVT_TARGETS})
