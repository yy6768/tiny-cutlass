#pragma once

#include <cuda_runtime_api.h>

#include "cutlass/cutlass.h"

namespace tiny_cutlass::conv_fused::conv1x1_upsample {

// conv1x1 -> nearest-neighbor UpsampleH x UpsampleW upsample.
//
// height/width are the LOW (pre-upsample) spatial extent -- the resolution the
// conv1x1 actually runs at. The output is written at the upsampled resolution
// (UpsampleH * height, UpsampleW * width): each low-res output pixel is
// replicated into its UpsampleH x UpsampleW block (nearest-neighbor). Because
// a 1x1 conv is pointwise in space, computing the conv at low resolution and
// broadcasting is exact, not an approximation -- see
// conv1x1_upsample/epilogue/predicated_tile_iterator_upsample.h.
struct Conv1x1UpsampleProblem {
  int batch = 0;
  int height = 0;   // low-res height
  int width = 0;    // low-res width
  int channels = 0;
  int output_channels = 0;
};

template <typename Element, int UpsampleH = 2, int UpsampleW = 2>
struct Conv1x1UpsampleArguments {
  Conv1x1UpsampleProblem problem;
  Element const* input = nullptr;
  Element const* weight = nullptr;
  Element const* bias = nullptr;
  Element* output = nullptr;
  cudaStream_t stream = nullptr;
};

template <typename Element, int UpsampleH = 2, int UpsampleW = 2>
cutlass::Status conv1x1_upsample(
    Conv1x1UpsampleArguments<Element, UpsampleH, UpsampleW> const& args);

}  // namespace tiny_cutlass::conv_fused::conv1x1_upsample
