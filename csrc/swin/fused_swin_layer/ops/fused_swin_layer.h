#pragma once
#include <cstdint>

namespace tiny_cutlass::swin::fused_swin_layer {

struct Problem {
  int batch = 1, height = 4, width = 4, channels = 32;
  int groups = 1, qk_channels = 32, value_channels = 32;
  int window_size = 4, shift_h = 0, shift_w = 0, mlp_ratio = 4;
  bool separate_qk = false, quant_fp8 = false;
  float rms_epsilon = 0.0009765625f, mlp_rms_epsilon = 0.0009765625f;
  float attn_drop = 0, proj_drop = 0;
};

template <typename Element>
struct Arguments {
  Problem problem;
  Element const* input = nullptr;
  Element const* rms_weight = nullptr;
  Element const* qkv_weight = nullptr;
  Element const* position_bias = nullptr;
  Element const* output_weight = nullptr;
  Element const* output_bias = nullptr;
  Element const* mlp_rms_weight = nullptr;
  Element const* fc1_weight = nullptr;
  Element const* fc1_bias = nullptr;
  Element const* fc2_weight = nullptr;
  Element const* fc2_bias = nullptr;
  int const* gather = nullptr;
  int const* scatter = nullptr;
  Element* output = nullptr;
};

}  // namespace tiny_cutlass::swin::fused_swin_layer
