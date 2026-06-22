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

template <
    typename Element = cutlass::float_e4m3_t,
    typename ElementScaleBias = float,
    typename ElementCompute = float>
struct Arguments {
  Problem problem;
  Element const* input = nullptr;
  Element const* weight0 = nullptr;
  ElementScaleBias const* stage0_scale = nullptr;
  ElementScaleBias const* bias0 = nullptr;
  Element const* weight1 = nullptr;
  Element const* bias1 = nullptr;
  Element* output = nullptr;
  ElementCompute stage0_alpha = ElementCompute(1);
  ElementCompute output_alpha = ElementCompute(1);
  cudaStream_t stream = nullptr;
};

template <
    typename Element = cutlass::float_e4m3_t,
    typename ElementScaleBias = float,
    typename ElementCompute = float>
cutlass::Status conv1x1_relu_conv1x1_relu(
    Arguments<Element, ElementScaleBias, ElementCompute> const& args);

}  // namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu
