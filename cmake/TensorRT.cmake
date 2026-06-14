include_guard(GLOBAL)

set(TINY_CUTLASS_TENSORRT_PATH
    "$ENV{TENSORRT_PATH}"
    CACHE PATH "TensorRT installation path")

if(NOT TINY_CUTLASS_TENSORRT_PATH)
  set(TINY_CUTLASS_TENSORRT_PATH
      "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT"
      CACHE PATH "TensorRT installation path" FORCE)
endif()

set(TINY_CUTLASS_TENSORRT_INCLUDE_DIR
    "${TINY_CUTLASS_TENSORRT_PATH}/include")
set(TINY_CUTLASS_TENSORRT_LIBRARY_DIR
    "${TINY_CUTLASS_TENSORRT_PATH}/lib")
file(GLOB TINY_CUTLASS_TENSORRT_RUNTIME_DLLS
     "${TINY_CUTLASS_TENSORRT_LIBRARY_DIR}/*.dll")

if(NOT EXISTS "${TINY_CUTLASS_TENSORRT_INCLUDE_DIR}/NvInfer.h")
  message(FATAL_ERROR
    "TensorRT header not found: ${TINY_CUTLASS_TENSORRT_INCLUDE_DIR}/NvInfer.h")
endif()

if(NOT EXISTS "${TINY_CUTLASS_TENSORRT_LIBRARY_DIR}/nvinfer_10.lib")
  message(FATAL_ERROR
    "TensorRT import library not found: "
    "${TINY_CUTLASS_TENSORRT_LIBRARY_DIR}/nvinfer_10.lib")
endif()

add_library(tiny_cutlass_tensorrt INTERFACE)
add_library(tiny_cutlass::tensorrt ALIAS tiny_cutlass_tensorrt)

target_include_directories(tiny_cutlass_tensorrt INTERFACE
  "${TINY_CUTLASS_TENSORRT_INCLUDE_DIR}"
)

target_link_libraries(tiny_cutlass_tensorrt INTERFACE
  "${TINY_CUTLASS_TENSORRT_LIBRARY_DIR}/nvinfer_10.lib"
  "${TINY_CUTLASS_TENSORRT_LIBRARY_DIR}/nvinfer_plugin_10.lib"
  "${TINY_CUTLASS_TENSORRT_LIBRARY_DIR}/nvonnxparser_10.lib"
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
