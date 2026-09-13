#pragma once

#include <cuda_runtime_api.h>
#include "cutlass/cutlass.h"
#include "swin/window_attention/problem.h"

namespace tiny_cutlass::swin::window_attention {

template <typename Element>
struct WindowAttentionArguments {
  WindowAttentionProblem problem;
  Element const* input = nullptr;          // [B,H,W,C], contiguous
  Element const* rms_weight = nullptr;     // [C], mandatory RMSNorm scale
  Element const* qkv_weight = nullptr;     // [G, projected_channels, C]
  Element const* position_bias = nullptr;  // preexpanded 2D [G*16,16], shared by all windows
  Element const* output_weight = nullptr;  // [C,G*V]
  Element const* output_bias = nullptr;    // [C], optional zero bias
  int const* gather = nullptr;             // [window_rows], host builder -> device
  int const* scatter = nullptr;            // [window_rows], -1 for reflect halo
  Element* output = nullptr;               // [B,H,W,C]; must not alias inputs
};

// No allocations, H2D copies or intermediate global workspace. Caller owns
// all buffers and immutable index tables until stream completion.
template <typename Element>
cutlass::Status window_attention_can_implement(WindowAttentionArguments<Element> const& args);

template <typename Element>
cutlass::Status window_attention(WindowAttentionArguments<Element> const& args,
                                 cudaStream_t stream = nullptr);

}  // namespace tiny_cutlass::swin::window_attention
