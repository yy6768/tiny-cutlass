# Copyright (c) 2026 tiny-cutlass contributors.
# SPDX-License-Identifier: BSD-3-Clause

set(NATTEN_TEST_DIR "${PROJECT_SOURCE_DIR}/csrc/tests/natten")

cutlass_example_add_executable(
  natten_fna_test
  "${NATTEN_TEST_DIR}/natten_fna_test.cu"
)

target_include_directories(
  natten_fna_test
  PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/include"
)

target_compile_options(
  natten_fna_test
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

if(MSVC)
  target_compile_options(
    natten_fna_test
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
  )
endif()

add_custom_target(natten DEPENDS natten_fna_test)
