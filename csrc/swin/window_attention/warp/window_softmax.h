#pragma once

#include "cutlass/cutlass.h"
#include <math_constants.h>

namespace tiny_cutlass::swin::window_attention::warp {

// Four lanes own a complete 16-token row, two rows per lane group.
template <typename Element>
struct WindowSoftmax {
  CUTLASS_DEVICE
  void operator()(float* scores, Element* probabilities, int lane) const {
    for (int row = lane / 4; row < 16; row += 8) {
      float values[4];
      float maximum = -CUDART_INF_F;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        values[i] = scores[row * 32 + (lane % 4) + 4 * i];
        maximum = fmaxf(maximum, values[i]);
      }
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1, 4));
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2, 4));
      float sum = 0;
      #pragma unroll
      for (int i = 0; i < 4; ++i) { values[i] = expf(values[i] - maximum); sum += values[i]; }
      sum += __shfl_xor_sync(0xffffffff, sum, 1, 4);
      sum += __shfl_xor_sync(0xffffffff, sum, 2, 4);
      #pragma unroll
      for (int i = 0; i < 4; ++i)
        probabilities[row * 16 + (lane % 4) + 4 * i] = Element(values[i] / sum);
    }
  }
};

}  // namespace tiny_cutlass::swin::window_attention::warp
