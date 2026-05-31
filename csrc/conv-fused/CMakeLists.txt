cmake_minimum_required(VERSION 3.24)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.pyd')"
  OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures for tiny-cutlass targets")
endif()

set(CONV_FUSED_CUTLASS_DIR
    "${CMAKE_SOURCE_DIR}/3rdparty/cutlass"
    CACHE PATH "Optional CUTLASS checkout used for future conv-fused CUTLASS kernels")

if(EXISTS "${CONV_FUSED_CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(STATUS "conv-fused: found CUTLASS at ${CONV_FUSED_CUTLASS_DIR}")
else()
  message(STATUS "conv-fused: CUTLASS headers not found at ${CONV_FUSED_CUTLASS_DIR}; building correctness CUDA kernel only")
endif()

add_library(conv_fused MODULE
  conv1x1_relu_conv1x1_bilinear.cu
)

target_include_directories(conv_fused PRIVATE
  "${Python3_INCLUDE_DIRS}"
  "${TORCH_INCLUDE_DIRS}"
)

target_link_libraries(conv_fused PRIVATE
  "${TORCH_LIBRARIES}"
  "${TORCH_PYTHON_LIBRARY}"
  Python3::Module
)

target_compile_definitions(conv_fused PRIVATE
  TORCH_EXTENSION_NAME=conv_fused
  _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
)

target_compile_options(conv_fused PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/bigobj>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/bigobj>
)

set_target_properties(conv_fused PROPERTIES
  PREFIX ""
  SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
  CUDA_SEPARABLE_COMPILATION OFF
)

add_custom_target(conv_fused_check
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_conv1x1_relu_conv1x1_bilinear.py"
          --module-dir "$<TARGET_FILE_DIR:conv_fused>"
  DEPENDS conv_fused
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  USES_TERMINAL
)
