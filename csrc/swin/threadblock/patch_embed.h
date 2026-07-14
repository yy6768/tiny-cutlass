/*
  Threadblock-level Swin PatchEmbed glue kernels.
*/

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

namespace tiny_cutlass {
namespace swin {
namespace threadblock {

template <typename Element_, int kThreads_ = 256>
struct ImagePadChannels {
  using Element = Element_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kNumThreads = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    Element const* input = nullptr;
    Element* output = nullptr;
    int64_t elements = 0;
    int32_t channels = 0;
    int32_t channels_padded = 0;
    int32_t height = 0;
    int32_t width = 0;

    CUTLASS_HOST dim3 getBlocksGrid() const {
      return dim3((elements + kThreads - 1) / kThreads, 1, 1);
    }
    CUTLASS_HOST dim3 getThreadsGrid() const {
      return dim3(kThreads, 1, 1);
    }
  };

  static CUTLASS_DEVICE void run(Params const& p) {
    int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= p.elements) {
      return;
    }
    int c = int(idx % p.channels_padded);
    if (c < p.channels) {
      int x = int((idx / p.channels_padded) % p.width);
      int y = int((idx / (int64_t(p.channels_padded) * p.width)) % p.height);
      int b = int(idx / (int64_t(p.channels_padded) * p.height * p.width));
      int64_t input_idx =
          ((int64_t(b) * p.height + y) * p.width + x) * p.channels + c;
      p.output[idx] = p.input[input_idx];
    } else {
      p.output[idx] = Element(0);
    }
  }
};

template <typename Element_, int kThreads_ = 256>
struct FilterOihwToKrscPadded {
  using Element = Element_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kNumThreads = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    Element const* input = nullptr;
    Element* output = nullptr;
    int64_t elements = 0;
    int32_t output_channels = 0;
    int32_t input_channels = 0;
    int32_t input_channels_padded = 0;
    int32_t filter_h = 0;
    int32_t filter_w = 0;

    CUTLASS_HOST dim3 getBlocksGrid() const {
      return dim3((elements + kThreads - 1) / kThreads, 1, 1);
    }
    CUTLASS_HOST dim3 getThreadsGrid() const {
      return dim3(kThreads, 1, 1);
    }
  };

  static CUTLASS_DEVICE void run(Params const& p) {
    int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= p.elements) {
      return;
    }
    int c = int(idx % p.input_channels_padded);
    int s = int((idx / p.input_channels_padded) % p.filter_w);
    int r = int((idx / (int64_t(p.input_channels_padded) * p.filter_w)) % p.filter_h);
    int k = int(idx / (int64_t(p.input_channels_padded) * p.filter_w * p.filter_h));
    if (c < p.input_channels) {
      int64_t input_idx =
          ((int64_t(k) * p.input_channels + c) * p.filter_h + r) * p.filter_w + s;
      p.output[idx] = p.input[input_idx];
    } else {
      p.output[idx] = Element(0);
    }
  }
};

} // namespace threadblock
} // namespace swin
} // namespace tiny_cutlass
