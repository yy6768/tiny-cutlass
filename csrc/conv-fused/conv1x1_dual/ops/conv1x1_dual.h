#pragma once

#include <cuda_runtime_api.h>

#include "cutlass/cutlass.h"

namespace tiny_cutlass::conv_fused {

// Public residency selector, mirrored from the kernel-level enum so callers
// (tests, benchmarks) do not need to include the CUTLASS-heavy kernel headers.
enum class Residency {
  kRF,    // stage-0 accumulator stays in registers
  kSmem,  // stage-0 accumulator staged through shared memory
};

struct Conv1x1DualProblem {
  int batch = 0;
  int height = 0;
  int width = 0;
  int channels = 0;
  int hidden_channels = 0;
  int output_channels = 0;
};

template <typename Element>
struct Conv1x1DualArguments {
  Conv1x1DualProblem problem;
  Element const* input = nullptr;
  Element const* weight0 = nullptr;
  Element const* bias0 = nullptr;
  Element const* weight1 = nullptr;
  Element const* bias1 = nullptr;
  Element* output = nullptr;
  Residency residency = Residency::kRF;
  cudaStream_t stream = nullptr;
};

template <typename Element>
cutlass::Status conv1x1_dual(
    Conv1x1DualArguments<Element> const& args);

}  // namespace tiny_cutlass::conv_fused
