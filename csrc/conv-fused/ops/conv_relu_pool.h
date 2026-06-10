#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/cutlass.h"

namespace tiny_cutlass::conv_fused {

struct ConvReluPoolProblem {
  int batch = 0;
  int height = 0;
  int width = 0;
  int channels = 0;
  int hidden_channels = 0;
  int output_channels = 0;
};

template <typename Element>
struct ConvReluPoolArguments {
  ConvReluPoolProblem problem;
  Element const* input = nullptr;
  Element const* weight0 = nullptr;
  Element const* bias0 = nullptr;
  Element const* weight1 = nullptr;
  Element const* bias1 = nullptr;
  Element* output = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  cudaStream_t stream = nullptr;
};

template <typename Element>
size_t conv_relu_pool_workspace_size(ConvReluPoolProblem const& problem);

template <typename Element>
cutlass::Status conv_relu_pool(ConvReluPoolArguments<Element> const& args);

}  // namespace tiny_cutlass::conv_fused
