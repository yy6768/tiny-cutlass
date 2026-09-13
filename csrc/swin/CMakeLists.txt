# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Swin pipeline, rebuilt from scratch on CUTLASS 2.x. Targets are added one per
# family as each one passes reference parity; nothing is registered here before
# it has a working test, so `--target swin` always builds something verifiable.

set(SWIN_TEST_DIR "${PROJECT_SOURCE_DIR}/csrc/tests/swin")

# ---------------------------------------------------------------------------
# Host-only gate for the window index / bias / mask tables. Pure C++: these
# tables are built on the host and consumed as GatherA / ScatterD index arrays
# and as a folded bias+mask tile, so their coordinate conventions are pinned
# down here before any kernel reads them.
# ---------------------------------------------------------------------------
add_executable(swin_window_index "${SWIN_TEST_DIR}/window_index.cpp")

target_include_directories(swin_window_index PRIVATE
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/csrc"
  "${PROJECT_SOURCE_DIR}/csrc/tests/common"
)

target_compile_options(swin_window_index PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
)

# ---------------------------------------------------------------------------
# Device gate for the partition / reverse mechanism. Runs stock GemmUniversal
# with GatherA / ScatterD against an identity B so the row remap is isolated
# from any arithmetic. Step 3 folds the same index arrays into the real QKV and
# projection GEMMs.
# ---------------------------------------------------------------------------
tiny_cutlass_add_cutlass_executable(
  swin_window_gather_scatter
  "${SWIN_TEST_DIR}/window_gather_scatter.cu"
)

target_include_directories(swin_window_gather_scatter PRIVATE
  "${PROJECT_SOURCE_DIR}/csrc/tests/common"
)

target_compile_options(swin_window_gather_scatter PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/bigobj>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/bigobj>
)

# ---------------------------------------------------------------------------
# patch_embed: conv(patch x patch, stride patch) + bias + channel-axis
# LayerNorm, fused into one kernel via a forked conv kernel that drives an
# epilogue visitor. The family core is a static library so the operator's
# template instantiation is compiled once.
# ---------------------------------------------------------------------------
add_library(swin_patch_embed_core STATIC
  patch_embed/ops/patch_embed.cu
)

target_include_directories(swin_patch_embed_core PUBLIC
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/csrc"
)

target_link_libraries(swin_patch_embed_core PUBLIC
  tiny_cutlass::cutlass_headers
  CUDA::cudart
)

target_compile_options(swin_patch_embed_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/bigobj>
)

set_target_properties(swin_patch_embed_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF
  # Must match the test executables below, or cudart_static and cudart both get
  # linked in and every runtime symbol is multiply defined.
  CUDA_RUNTIME_LIBRARY Shared
)

tiny_cutlass_add_cutlass_executable(
  swin_patch_embed
  "${SWIN_TEST_DIR}/patch_embed.cu"
)

target_link_libraries(swin_patch_embed PRIVATE swin_patch_embed_core)

target_include_directories(swin_patch_embed PRIVATE
  "${PROJECT_SOURCE_DIR}/csrc/tests/common"
  "${SWIN_TEST_DIR}"
)

target_compile_options(swin_patch_embed PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/bigobj>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/bigobj>
)

# ---------------------------------------------------------------------------
# swin_mlp: LN2 -> fc1 + bias + GELU -> fc2 + bias + residual2.
# ---------------------------------------------------------------------------
add_library(swin_mlp_core STATIC
  swin_mlp/ops/swin_mlp.cu
)

target_include_directories(swin_mlp_core PUBLIC
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/csrc"
)

target_link_libraries(swin_mlp_core PUBLIC
  tiny_cutlass::cutlass_headers
  CUDA::cudart
)

target_compile_options(swin_mlp_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/bigobj>
)

set_target_properties(swin_mlp_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RUNTIME_LIBRARY Shared
)

tiny_cutlass_add_cutlass_executable(
  swin_mlp
  "${SWIN_TEST_DIR}/swin_mlp.cu"
)

target_link_libraries(swin_mlp PRIVATE swin_mlp_core)

target_include_directories(swin_mlp PRIVATE
  "${PROJECT_SOURCE_DIR}/csrc/tests/common"
  "${SWIN_TEST_DIR}"
)

target_compile_options(swin_mlp PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/bigobj>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/bigobj>
)

# ---------------------------------------------------------------------------
# patch_merging: 2x2 concat gather folded into the LayerNorm pass, then 4C -> 2C.
# ---------------------------------------------------------------------------
add_library(swin_patch_merging_core STATIC
  patch_merging/ops/patch_merging.cu
)

target_include_directories(swin_patch_merging_core PUBLIC
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/csrc"
)

target_link_libraries(swin_patch_merging_core PUBLIC
  tiny_cutlass::cutlass_headers
  CUDA::cudart
)

target_compile_options(swin_patch_merging_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/bigobj>
)

set_target_properties(swin_patch_merging_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RUNTIME_LIBRARY Shared
)

tiny_cutlass_add_cutlass_executable(
  swin_patch_merging
  "${SWIN_TEST_DIR}/patch_merging.cu"
)

target_link_libraries(swin_patch_merging PRIVATE swin_patch_merging_core)

target_include_directories(swin_patch_merging PRIVATE
  "${PROJECT_SOURCE_DIR}/csrc/tests/common"
  "${SWIN_TEST_DIR}"
)

target_compile_options(swin_patch_merging PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/bigobj>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/bigobj>
)

# Full DLSS4 window attention: one CUTLASS kernel, independently verified by
# csrc/tests/swin/verify.py against PyTorch before registration here.
add_library(swin_window_attention_core STATIC window_attention/ops/window_attention.cu)
target_include_directories(swin_window_attention_core PUBLIC
  "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/csrc")
target_link_libraries(swin_window_attention_core PUBLIC tiny_cutlass::cutlass_headers CUDA::cudart)
target_compile_options(swin_window_attention_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/bigobj>)
set_target_properties(swin_window_attention_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION OFF CUDA_RUNTIME_LIBRARY Shared)
tiny_cutlass_add_cutlass_executable(swin_window_attention "${SWIN_TEST_DIR}/window_attention.cu")
target_link_libraries(swin_window_attention PRIVATE swin_window_attention_core)
target_include_directories(swin_window_attention PRIVATE "${PROJECT_SOURCE_DIR}/csrc/tests/common")

# Full RMSNorm / attention / residual / MLP block, composed within this family.
add_library(swin_block_core STATIC window_attention/ops/swin_block.cu)
target_include_directories(swin_block_core PUBLIC "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/csrc")
target_link_libraries(swin_block_core PUBLIC tiny_cutlass::cutlass_headers CUDA::cudart)
target_compile_options(swin_block_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/bigobj>)
set_target_properties(swin_block_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF CUDA_RUNTIME_LIBRARY Shared)
tiny_cutlass_add_cutlass_executable(swin_block "${SWIN_TEST_DIR}/block.cu")
target_link_libraries(swin_block PRIVATE swin_block_core)
target_include_directories(swin_block PRIVATE "${PROJECT_SOURCE_DIR}/csrc/tests/common")
set_target_properties(swin_block PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/swin" CUDA_RUNTIME_LIBRARY Shared)

set(SWIN_TARGETS
  swin_block
  swin_window_attention
  swin_window_index
  swin_window_gather_scatter
  swin_patch_embed
  swin_mlp
  swin_patch_merging
)

if("89" IN_LIST CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.4)
  tiny_cutlass_add_cutlass_executable(swin_grouped_gemm "${SWIN_TEST_DIR}/grouped_gemm.cu")
  target_include_directories(swin_grouped_gemm PRIVATE "${PROJECT_SOURCE_DIR}/csrc/tests/common")
  target_compile_options(swin_grouped_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/bigobj>)
  set_target_properties(swin_grouped_gemm PROPERTIES CUDA_ARCHITECTURES 89 CUDA_RUNTIME_LIBRARY Shared)
  list(APPEND SWIN_TARGETS swin_grouped_gemm)
endif()

set_target_properties(${SWIN_TARGETS} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/swin"
)

set_target_properties(
  swin_window_attention
  swin_window_gather_scatter
  swin_patch_embed
  swin_mlp
  swin_patch_merging
  PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared
)

add_custom_target(swin DEPENDS ${SWIN_TARGETS})
