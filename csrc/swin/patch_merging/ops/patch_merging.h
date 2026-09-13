#pragma once

/*
  Public API for PatchMerging, the Swin stage transition.

      [B, H, W, C] -> concat 2x2 -> LayerNorm(4C) -> linear 4C -> 2C

  Raw device pointers plus a problem descriptor. The gather table comes from
  swin/window_index.h's build_patch_merging_index(); scratch is caller-provided.
*/

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/cutlass.h"

#include "swin/swin_problem.h"

namespace tiny_cutlass::swin::patch_merging {

template <typename Element, typename ElementCompute = float>
struct PatchMergingArguments {
  PatchMergingProblem problem;

  Element const* input = nullptr;          // [B*H*W, C]
  int const* gather = nullptr;             // [4, num_out_tokens]
  ElementCompute const* gamma = nullptr;   // [4C]
  ElementCompute const* beta = nullptr;    // [4C]
  Element const* weight = nullptr;         // [2C, 4C] row-major
  ElementCompute const* bias = nullptr;    // [2C], may be null
  Element* output = nullptr;               // [num_out_tokens, 2C]

  Element* normalized = nullptr;           // scratch [num_out_tokens, 4C]

  ElementCompute epsilon = ElementCompute(1e-5f);
  cudaStream_t stream = nullptr;
};

template <typename Element, typename ElementCompute = float>
cutlass::Status patch_merging(
    PatchMergingArguments<Element, ElementCompute> const& args);

template <typename Element, typename ElementCompute = float>
cutlass::Status patch_merging_can_implement(
    PatchMergingArguments<Element, ElementCompute> const& args);

/// Bytes required for the `normalized` scratch buffer.
template <typename Element, typename ElementCompute = float>
size_t patch_merging_normalized_size(PatchMergingProblem const& problem);

}  // namespace tiny_cutlass::swin::patch_merging
