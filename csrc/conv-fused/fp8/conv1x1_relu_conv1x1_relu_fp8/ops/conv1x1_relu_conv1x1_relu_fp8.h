#pragma once

#include <cuda_runtime_api.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu {

struct Problem {
  int batch = 0;
  int height = 0;
  int width = 0;
  int channels = 0;
  int hidden_channels = 0;
  int output_channels = 0;
};

struct Arguments {
  Problem problem;
  cutlass::float_e4m3_t const* input = nullptr;
  cutlass::float_e4m3_t const* weight0 = nullptr;
  cutlass::float_e4m3_t* stage0 = nullptr;
  float const* stage0_scale = nullptr;
  float const* bias0 = nullptr;
  cutlass::float_e4m3_t const* weight1 = nullptr;
  cutlass::float_e4m3_t const* bias1 = nullptr;
  cutlass::float_e4m3_t* output = nullptr;
  float stage0_alpha = 1.0f;
  float output_alpha = 1.0f;
  cudaStream_t stream = nullptr;
};

cutlass::Status conv1x1_relu_conv1x1_relu_fp8(Arguments const& args);

}  // namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu
