#pragma once

/*
  Public API for the Swin MLP.

      y = x + fc2(GELU(fc1(LayerNorm(x)) + b1)) + b2

  Raw device pointers plus scalars. Workspace is caller-provided: this operator
  allocates nothing, so a stage can reuse one pair of buffers across blocks.
*/

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/cutlass.h"

namespace tiny_cutlass::swin::swin_mlp {

template <typename Element, typename ElementCompute = float>
struct SwinMlpArguments {
  int rows = 0;
  int channels = 0;
  int hidden = 0;

  Element const* input = nullptr;            // [rows, channels], also residual2
  ElementCompute const* gamma = nullptr;     // LN2 scale, [channels]
  ElementCompute const* beta = nullptr;      // LN2 shift, [channels]
  Element const* fc1_weight = nullptr;       // [hidden, channels] row-major
  ElementCompute const* fc1_bias = nullptr;  // [hidden]
  Element const* fc2_weight = nullptr;       // [channels, hidden] row-major
  ElementCompute const* fc2_bias = nullptr;  // [channels]
  Element* output = nullptr;                 // [rows, channels]

  // Caller-owned scratch, sized by the helpers below.
  Element* normalized = nullptr;             // [rows, channels]
  Element* hidden_buffer = nullptr;          // [rows, hidden]

  ElementCompute epsilon = ElementCompute(1e-5f);
  cudaStream_t stream = nullptr;
};

template <typename Element, typename ElementCompute = float>
cutlass::Status swin_mlp(SwinMlpArguments<Element, ElementCompute> const& args);

template <typename Element, typename ElementCompute = float>
cutlass::Status swin_mlp_can_implement(
    SwinMlpArguments<Element, ElementCompute> const& args);

/// Bytes required for the `normalized` scratch buffer.
template <typename Element, typename ElementCompute = float>
size_t swin_mlp_normalized_size(int rows, int channels);

/// Bytes required for the `hidden_buffer` scratch buffer.
template <typename Element, typename ElementCompute = float>
size_t swin_mlp_hidden_size(int rows, int hidden);

}  // namespace tiny_cutlass::swin::swin_mlp
