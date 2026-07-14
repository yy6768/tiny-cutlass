#pragma once

#include <cuda_runtime_api.h>

#include "cutlass/cutlass.h"

namespace tiny_cutlass::conv_fused {

struct Conv1x1ReluConv1x1Problem {
  int batch = 0;
  int height = 0;
  int width = 0;
  int channels = 0;
  int hidden_channels = 0;
  int output_channels = 0;
};

template <typename Element>
struct Conv1x1ReluConv1x1Arguments {
  Conv1x1ReluConv1x1Problem problem;
  Element const* input = nullptr;
  Element const* weight0 = nullptr;
  Element const* bias0 = nullptr;
  Element const* weight1 = nullptr;
  Element const* bias1 = nullptr;
  Element* output = nullptr;
  cudaStream_t stream = nullptr;
};

template <typename Element>
cutlass::Status conv1x1_relu_conv1x1(
    Conv1x1ReluConv1x1Arguments<Element> const& args);

}  // namespace tiny_cutlass::conv_fused
