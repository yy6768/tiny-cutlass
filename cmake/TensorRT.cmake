set(TINY_CUTLASS_TENSORRT_PATH
    "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT"
    CACHE PATH "TensorRT 10.13 installation path")

set(TINY_CUTLASS_TENSORRT_INCLUDE_DIR
    "${TINY_CUTLASS_TENSORRT_PATH}/include")
set(TINY_CUTLASS_TENSORRT_LIB_DIR
    "${TINY_CUTLASS_TENSORRT_PATH}/lib")
set(TINY_CUTLASS_TENSORRT_BIN_DIR
    "${TINY_CUTLASS_TENSORRT_PATH}/bin")

set(TINY_CUTLASS_TENSORRT_NVINFER_LIB
    "${TINY_CUTLASS_TENSORRT_LIB_DIR}/nvinfer_10.lib")
set(TINY_CUTLASS_TENSORRT_NVINFER_DLL
    "${TINY_CUTLASS_TENSORRT_LIB_DIR}/nvinfer_10.dll")
set(TINY_CUTLASS_TENSORRT_PLUGIN_DLL
    "${TINY_CUTLASS_TENSORRT_LIB_DIR}/nvinfer_plugin_10.dll")
set(TINY_CUTLASS_TENSORRT_TRTEXEC
    "${TINY_CUTLASS_TENSORRT_BIN_DIR}/trtexec.exe")
file(GLOB TINY_CUTLASS_TENSORRT_RUNTIME_DLLS
     "${TINY_CUTLASS_TENSORRT_LIB_DIR}/*.dll")

foreach(REQUIRED_TENSORRT_FILE IN ITEMS
    "${TINY_CUTLASS_TENSORRT_INCLUDE_DIR}/NvInfer.h"
    "${TINY_CUTLASS_TENSORRT_INCLUDE_DIR}/NvInferRuntime.h"
    "${TINY_CUTLASS_TENSORRT_NVINFER_LIB}"
    "${TINY_CUTLASS_TENSORRT_NVINFER_DLL}"
    "${TINY_CUTLASS_TENSORRT_PLUGIN_DLL}"
    "${TINY_CUTLASS_TENSORRT_TRTEXEC}")
  if(NOT EXISTS "${REQUIRED_TENSORRT_FILE}")
    message(FATAL_ERROR "Required TensorRT file not found: ${REQUIRED_TENSORRT_FILE}")
  endif()
endforeach()

add_library(tiny_cutlass_tensorrt INTERFACE)
add_library(tiny_cutlass::tensorrt ALIAS tiny_cutlass_tensorrt)

target_include_directories(tiny_cutlass_tensorrt INTERFACE
  "${TINY_CUTLASS_TENSORRT_INCLUDE_DIR}"
)

target_link_libraries(tiny_cutlass_tensorrt INTERFACE
  "${TINY_CUTLASS_TENSORRT_NVINFER_LIB}"
)

function(tiny_cutlass_copy_tensorrt_runtime TARGET)
  foreach(DLL_PATH IN LISTS TINY_CUTLASS_TENSORRT_RUNTIME_DLLS)
    add_custom_command(TARGET ${TARGET} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${DLL_PATH}"
              "$<TARGET_FILE_DIR:${TARGET}>"
      VERBATIM)
  endforeach()
endfunction()
