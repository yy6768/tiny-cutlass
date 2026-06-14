# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set(SWIN_TEST_DIR "${PROJECT_SOURCE_DIR}/csrc/tests/swin")

add_library(swin_device OBJECT
  swin.cu
  patch_embed.cu
)

target_link_libraries(swin_device PRIVATE
  tiny_cutlass::cutlass_headers
  CUDA::cudart
)

target_include_directories(swin_device PRIVATE
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/csrc"
  "${CMAKE_CURRENT_LIST_DIR}"
  "${CMAKE_CURRENT_LIST_DIR}/../flash-attention"
)

target_compile_options(swin_device PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<COMPILE_LANGUAGE:CUDA>:--source-in-ptx>
  $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)

cutlass_example_add_executable(
  swin_window
  "${SWIN_TEST_DIR}/window.cpp"
  $<TARGET_OBJECTS:swin_device>
)

target_include_directories(swin_window PRIVATE
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/csrc"
  "${CMAKE_CURRENT_LIST_DIR}"
  "${CMAKE_CURRENT_LIST_DIR}/../flash-attention"
  "${SWIN_TEST_DIR}"
)

target_link_libraries(swin_window PRIVATE
  tiny_cutlass::cudnn
)

tiny_cutlass_copy_cudnn_runtime(swin_window)

cutlass_example_add_executable(
  swin_patch_embed
  "${SWIN_TEST_DIR}/patch_embed.cu"
  $<TARGET_OBJECTS:swin_device>
)

target_include_directories(swin_patch_embed PRIVATE
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/csrc"
  "${CMAKE_CURRENT_LIST_DIR}"
  "${SWIN_TEST_DIR}"
)

add_custom_target(swin DEPENDS swin_window swin_patch_embed)

add_library(swin_plugin SHARED
  trt/swin_plugin.cpp
  $<TARGET_OBJECTS:swin_device>
)

target_include_directories(swin_plugin PRIVATE
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/csrc"
  "${CMAKE_CURRENT_LIST_DIR}"
  "${CMAKE_CURRENT_LIST_DIR}/../flash-attention"
  "${CMAKE_CURRENT_LIST_DIR}/trt"
)

target_link_libraries(swin_plugin PRIVATE
  tiny_cutlass::cutlass_headers
  tiny_cutlass::tensorrt
  CUDA::cudart
)

add_dependencies(swin swin_plugin)
