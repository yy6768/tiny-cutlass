cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(tiny_cutlass LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(TINY_CUTLASS_CUTLASS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass"
    CACHE PATH "CUTLASS submodule directory")

if(NOT EXISTS "${TINY_CUTLASS_CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR
    "CUTLASS headers were not found at ${TINY_CUTLASS_CUTLASS_DIR}. "
    "Initialize the submodule with: git submodule update --init --recursive")
endif()

add_library(tiny_cutlass_cutlass_headers INTERFACE)
target_include_directories(tiny_cutlass_cutlass_headers INTERFACE
  "${TINY_CUTLASS_CUTLASS_DIR}/include"
  "${TINY_CUTLASS_CUTLASS_DIR}/tools/util/include"
)
target_compile_options(tiny_cutlass_cutlass_headers INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

function(cutlass_example_add_executable NAME)
  add_executable(${NAME} ${ARGN})
  target_link_libraries(${NAME} PRIVATE
    tiny_cutlass_cutlass_headers
    CUDA::cudart
  )
  target_include_directories(${NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
  )
  set_target_properties(${NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  )
endfunction()

option(TINY_CUTLASS_BUILD_FLASH_ATTENTION "Build migrated flash-attention CUTLASS examples" ON)
option(TINY_CUTLASS_BUILD_CONV_FUSED "Build conv-fused PyTorch CUDA extension" OFF)

if(TINY_CUTLASS_BUILD_FLASH_ATTENTION)
  add_subdirectory(csrc/flash-attention)
endif()

if(TINY_CUTLASS_BUILD_CONV_FUSED)
  add_subdirectory(csrc/conv-fused)
endif()
