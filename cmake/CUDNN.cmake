include_guard(GLOBAL)

set(TINY_CUTLASS_CUDNN_PATH
    "$ENV{CUDNN_PATH}"
    CACHE PATH "cuDNN installation path")

if(NOT TINY_CUTLASS_CUDNN_PATH)
  message(FATAL_ERROR
    "CUDNN_PATH is required for flash-attention. "
    "Set CUDNN_PATH or configure with -DTINY_CUTLASS_CUDNN_PATH=<path>.")
endif()

if(NOT DEFINED TINY_CUTLASS_CUDNN_CUDA_VERSION)
  set(TINY_CUTLASS_CUDNN_CUDA_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")
  string(REGEX REPLACE "^([0-9]+\\.[0-9]+).*$" "\\1"
         TINY_CUTLASS_CUDNN_CUDA_VERSION "${TINY_CUTLASS_CUDNN_CUDA_VERSION}")
endif()

set(TINY_CUTLASS_CUDNN_FRONTEND_DIR
    "${PROJECT_SOURCE_DIR}/3rdparty/cudnn-frontend"
    CACHE PATH "cuDNN frontend source directory")

set(TINY_CUTLASS_CUDNN_INCLUDE_DIR
    "${TINY_CUTLASS_CUDNN_PATH}/include/${TINY_CUTLASS_CUDNN_CUDA_VERSION}")
set(TINY_CUTLASS_CUDNN_LIBRARY
    "${TINY_CUTLASS_CUDNN_PATH}/lib/${TINY_CUTLASS_CUDNN_CUDA_VERSION}/x64/cudnn.lib")
set(TINY_CUTLASS_CUDNN_BIN_DIR
    "${TINY_CUTLASS_CUDNN_PATH}/bin/${TINY_CUTLASS_CUDNN_CUDA_VERSION}/x64")
set(TINY_CUTLASS_CUDNN_FRONTEND_INCLUDE_DIR
    "${TINY_CUTLASS_CUDNN_FRONTEND_DIR}/include")

if(NOT EXISTS "${TINY_CUTLASS_CUDNN_INCLUDE_DIR}/cudnn.h")
  message(FATAL_ERROR
    "cuDNN header not found: ${TINY_CUTLASS_CUDNN_INCLUDE_DIR}/cudnn.h")
endif()

if(NOT EXISTS "${TINY_CUTLASS_CUDNN_LIBRARY}")
  message(FATAL_ERROR
    "cuDNN import library not found: ${TINY_CUTLASS_CUDNN_LIBRARY}")
endif()

if(NOT EXISTS "${TINY_CUTLASS_CUDNN_BIN_DIR}/cudnn64_9.dll")
  message(FATAL_ERROR
    "cuDNN runtime DLL not found: ${TINY_CUTLASS_CUDNN_BIN_DIR}/cudnn64_9.dll")
endif()

if(NOT EXISTS "${TINY_CUTLASS_CUDNN_FRONTEND_INCLUDE_DIR}/cudnn_frontend.h")
  message(FATAL_ERROR
    "cuDNN frontend header not found: "
    "${TINY_CUTLASS_CUDNN_FRONTEND_INCLUDE_DIR}/cudnn_frontend.h")
endif()

add_library(tiny_cutlass_cudnn INTERFACE)
add_library(tiny_cutlass::cudnn ALIAS tiny_cutlass_cudnn)

target_include_directories(tiny_cutlass_cudnn INTERFACE
  "${TINY_CUTLASS_CUDNN_INCLUDE_DIR}"
  "${TINY_CUTLASS_CUDNN_FRONTEND_INCLUDE_DIR}"
)

target_link_libraries(tiny_cutlass_cudnn INTERFACE
  "${TINY_CUTLASS_CUDNN_LIBRARY}"
  CUDA::cuda_driver
  CUDA::nvrtc
)
