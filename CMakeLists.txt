cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

set(TINY_CUTLASS_SUPPORTED_CUDA_ARCHITECTURES 80 89)
foreach(TINY_CUTLASS_CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(NOT TINY_CUTLASS_CUDA_ARCH IN_LIST TINY_CUTLASS_SUPPORTED_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
      "tiny-cutlass currently targets SM80 and SM89; configure with "
      "CMAKE_CUDA_ARCHITECTURES=80 or CMAKE_CUDA_ARCHITECTURES=89")
  endif()
endforeach()

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
option(TINY_CUTLASS_BUILD_CONV_FUSED "Build conv-fused CUTLASS kernels" OFF)
option(TINY_CUTLASS_BUILD_SWIN "Build Swin CUTLASS kernels and tests" ON)
option(TINY_CUTLASS_BUILD_NATTEN "Build NATTEN SM80 CUTLASS learning examples" ON)

if(TINY_CUTLASS_BUILD_FLASH_ATTENTION)
  include(cmake/CUDNN.cmake)
  add_subdirectory(csrc/flash-attention)
endif()

if(TINY_CUTLASS_BUILD_CONV_FUSED)
  add_subdirectory(csrc/conv-fused)
endif()

if(TINY_CUTLASS_BUILD_SWIN)
  include(cmake/CUDNN.cmake)
  include(cmake/TensorRT.cmake)
  add_subdirectory(csrc/swin)
endif()

if(TINY_CUTLASS_BUILD_NATTEN)
  add_subdirectory(csrc/natten)
endif()
