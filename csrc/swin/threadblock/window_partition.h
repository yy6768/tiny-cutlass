/*
  Threadblock-level Swin window partition and reverse kernels.
*/

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

#include "../warp/window_mapping.h"

namespace tiny_cutlass {
namespace swin {
namespace threadblock {

template <typename Element_, int kThreads_ = 256>
struct WindowPartition {
  using Element = Element_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kNumThreads = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    Element const* input = nullptr;
    Element* output = nullptr;

    int32_t batch = 0;
    int32_t height = 0;
    int32_t width = 0;
    int32_t channels = 0;
    int32_t shift_size = 0;
    int32_t window_size = 0;

    CUTLASS_HOST dim3 getBlocksGrid() const {
      return dim3(batch * height * width, 1, 1);
    }
    CUTLASS_HOST dim3 getThreadsGrid() const {
      return dim3(kThreads, 1, 1);
    }
  };

  static CUTLASS_DEVICE void run(Params const& p) {
    int64_t pixel = blockIdx.x;
    warp::WindowMapping mapping =
        warp::window_mapping(pixel, p.height, p.width, p.window_size, p.shift_size);

    for (int c = threadIdx.x; c < p.channels; c += blockDim.x) {
      p.output[mapping.window_token * p.channels + c] =
          p.input[mapping.image_token * p.channels + c];
    }
  }
};

template <typename Element_, int kThreads_ = 256>
struct WindowReverse {
  using Element = Element_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kNumThreads = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    Element const* input = nullptr;
    Element* output = nullptr;

    int32_t batch = 0;
    int32_t height = 0;
    int32_t width = 0;
    int32_t channels = 0;
    int32_t shift_size = 0;
    int32_t window_size = 0;

    CUTLASS_HOST dim3 getBlocksGrid() const {
      return dim3(batch * height * width, 1, 1);
    }
    CUTLASS_HOST dim3 getThreadsGrid() const {
      return dim3(kThreads, 1, 1);
    }
  };

  static CUTLASS_DEVICE void run(Params const& p) {
    int64_t pixel = blockIdx.x;
    warp::WindowMapping mapping =
        warp::window_mapping(pixel, p.height, p.width, p.window_size, p.shift_size);

    for (int c = threadIdx.x; c < p.channels; c += blockDim.x) {
      p.output[mapping.image_token * p.channels + c] =
          p.input[mapping.window_token * p.channels + c];
    }
  }
};

} // namespace threadblock
} // namespace swin
} // namespace tiny_cutlass
