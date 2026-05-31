cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(tiny_cutlass LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

include(cmake/TinyCutlass.cmake)

option(TINY_CUTLASS_BUILD_FLASH_ATTENTION "Build migrated flash-attention CUTLASS examples" ON)
option(TINY_CUTLASS_BUILD_CONV_FUSED "Build conv-fused PyTorch CUDA extension" OFF)

if(TINY_CUTLASS_BUILD_FLASH_ATTENTION)
  add_subdirectory(csrc/flash-attention)
endif()

if(TINY_CUTLASS_BUILD_CONV_FUSED)
  add_subdirectory(csrc/conv-fused)
endif()
