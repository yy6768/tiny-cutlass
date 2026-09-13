#pragma once

// Declaration-only header for the cuDNN b2b conv reference.
//
// The implementation lives in cudnn_conv_reference.cpp and is compiled as HOST
// C++ (NOT through nvcc). cudnn_frontend.h is enormous and blows nvcc's
// cudafe++ front-end stack when pulled into a .cu translation unit, so we keep
// it out of any .cu. A .cu test includes THIS header (which never touches
// cudnn_frontend) and links against the compiled .cpp.
//
// Only relevant when TINY_CUTLASS_WITH_CUDNN is defined.

#include <cstdint>
#include <string>

#include <cuda_runtime_api.h>

namespace tiny_cutlass::testing {

// Problem shape for the two chained 1x1 convolutions (NHWC).
struct ConvDualProblem {
  int batch = 0;
  int height = 0;
  int width = 0;
  int channels = 0;         // C0: input channels
  int hidden_channels = 0;  // K0 == C1: intermediate channels
  int output_channels = 0;  // K1: output channels
};

// Device pointers for a cuDNN reference run. Pointers are raw device memory
// holding half-precision (16-bit) values in packed NHWC / (K,C) layout.
struct ConvDualTensorsHalf {
  void const* input = nullptr;    // (N, H, W, C0)
  void const* weight0 = nullptr;  // (K0, C0) 1x1 filter
  void const* bias0 = nullptr;    // (K0)
  void const* weight1 = nullptr;  // (K1, K0) 1x1 filter
  void const* bias1 = nullptr;    // (K1)
  void* output = nullptr;         // (N, H, W, K1)
};

// Runs conv0 -> +bias0 -> ReLU -> conv1 -> +bias1 as one cuDNN graph
// (HALF io / FLOAT compute). Returns:
//   cudaSuccess          - reference ran, `output` populated
//   cudaErrorNotSupported- backend cannot support this graph (caller may skip)
//   other                - hard failure, see error_message
cudaError_t run_cudnn_conv_dual_reference_half(
    ConvDualProblem const& problem,
    ConvDualTensorsHalf const& tensors,
    cudaStream_t stream,
    std::string& error_message);

}  // namespace tiny_cutlass::testing
