set(NATTEN_TEST_DIR "${PROJECT_SOURCE_DIR}/csrc/tests/natten")

cutlass_example_add_executable(
  natten_fna
  "${NATTEN_TEST_DIR}/fna.cu"
)

target_include_directories(
  natten_fna
  PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/include"
  "${CMAKE_CURRENT_LIST_DIR}"
)

target_compile_options(
  natten_fna
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

if(MSVC)
  target_compile_options(
    natten_fna
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
  )
endif()

add_custom_target(natten DEPENDS natten_fna)
