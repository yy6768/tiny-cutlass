#pragma once

/*
  Public API for PatchEmbed.

      out[token, c] = gamma[c] * norm(conv(x)[token, c] + bias[c]) + beta[c]

  where the normalization is over the embed_dim axis of each token, and the conv
  is a patch x patch kernel with stride == patch (non-overlapping patchify).

  Raw device pointers, a problem descriptor, and a stream. No CUTLASS types in
  this header, so callers (tests, benchmarks) do not pay for the template
  instantiation.
*/

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/cutlass.h"

#include "swin/swin_problem.h"

namespace tiny_cutlass::swin::patch_embed {

template <typename Element, typename ElementCompute = float>
struct PatchEmbedArguments {
  PatchEmbedProblem problem;
  Element const* input = nullptr;          // NHWC, C = in_channels_padded
  Element const* filter = nullptr;         // KRSC, C = in_channels_padded
  ElementCompute const* bias = nullptr;    // [embed_dim], pre-norm
  ElementCompute const* gamma = nullptr;   // [embed_dim], post-norm scale
  ElementCompute const* beta = nullptr;    // [embed_dim], post-norm shift
  Element* output = nullptr;               // [num_tokens, embed_dim] row-major
  ElementCompute epsilon = ElementCompute(1e-5f);
  cudaStream_t stream = nullptr;
};

/// Runs the fused conv + bias + LayerNorm in a single kernel launch.
template <typename Element, typename ElementCompute = float>
cutlass::Status patch_embed(
    PatchEmbedArguments<Element, ElementCompute> const& args);

/// Reports whether this build supports the requested problem, without running.
template <typename Element, typename ElementCompute = float>
cutlass::Status patch_embed_can_implement(
    PatchEmbedArguments<Element, ElementCompute> const& args);

/// Shared-memory footprint of the fused kernel, for budget reporting.
template <typename Element, typename ElementCompute = float>
size_t patch_embed_shared_storage_size();

}  // namespace tiny_cutlass::swin::patch_embed
