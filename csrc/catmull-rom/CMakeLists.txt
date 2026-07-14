cmake_minimum_required(VERSION 3.24)

tiny_cutlass_add_cutlass_executable(
  catmull_forward
  "${PROJECT_SOURCE_DIR}/csrc/tests/catmull-rom/forward.cu"
)

tiny_cutlass_add_cutlass_executable(
  catmull_backward
  "${PROJECT_SOURCE_DIR}/csrc/tests/catmull-rom/backward.cu"
)

tiny_cutlass_add_cutlass_executable(
  catmull_autodiff
  "${PROJECT_SOURCE_DIR}/csrc/tests/catmull-rom/autodiff.cu"
)

tiny_cutlass_add_cutlass_executable(
  catmull_dilation
  "${PROJECT_SOURCE_DIR}/csrc/tests/catmull-rom/dilation.cu"
)

set(CATMULL_ROM_TARGETS
    catmull_forward
    catmull_backward
    catmull_autodiff
    catmull_dilation)

foreach(CATMULL_ROM_TARGET IN LISTS CATMULL_ROM_TARGETS)
  target_include_directories(${CATMULL_ROM_TARGET} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_target_properties(${CATMULL_ROM_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/catmull-rom"
    CUDA_RUNTIME_LIBRARY Shared
  )
endforeach()

add_custom_target(catmull_rom DEPENDS ${CATMULL_ROM_TARGETS})
