
# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

cutlass_example_add_executable(
  flash_attention_00_naive_attention_test
  naive_attention.cu
)

# Keep profiling/debug metadata for Nsight Compute source/PTX/SASS correlation.
# - --generate-line-info keeps source line mappings without disabling optimization.
# - --source-in-ptx embeds CUDA source comments in generated PTX.
# - --keep preserves PTX/CUBIN/FATBIN intermediates next to this target's build files.
# - ptxas verbosity/resource usage prints registers, spills, smem, cmem, etc. at build time.
set(FLASH_ATTENTION_NAIVE_NVCC_KEEP_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/nvcc")
file(MAKE_DIRECTORY "${FLASH_ATTENTION_NAIVE_NVCC_KEEP_DIR}")

target_compile_options(
  flash_attention_00_naive_attention_test
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<COMPILE_LANGUAGE:CUDA>:--source-in-ptx>
  $<$<COMPILE_LANGUAGE:CUDA>:--keep>
  $<$<COMPILE_LANGUAGE:CUDA>:--keep-dir=${FLASH_ATTENTION_NAIVE_NVCC_KEEP_DIR}>
  $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)
