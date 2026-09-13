#pragma once
#include "swin/window_attention/ops/window_attention.h"

namespace tiny_cutlass::swin::window_attention {

template <typename Element>
struct SwinBlockArguments : WindowAttentionArguments<Element> {
  int mlp_ratio = 4;
  float mlp_rms_epsilon = 0.0009765625f;
  Element const* mlp_rms_weight = nullptr;  // [C], independent gamma
  Element const* fc1_weight = nullptr;     // [C*ratio,C]
  Element const* fc1_bias = nullptr;       // [C*ratio]
  Element const* fc2_weight = nullptr;     // [C,C*ratio]
  Element const* fc2_bias = nullptr;       // [C]
};

template <typename Element>
cutlass::Status swin_block_can_implement(SwinBlockArguments<Element> const& args);
template <typename Element>
cutlass::Status swin_block(SwinBlockArguments<Element> const& args, cudaStream_t stream = nullptr);

}  // namespace tiny_cutlass::swin::window_attention
