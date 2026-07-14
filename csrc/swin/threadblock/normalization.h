/*
  Threadblock-level Swin normalization kernels.
*/

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

#include "../warp/window_mapping.h"

namespace tiny_cutlass {
namespace swin {
namespace threadblock {

template <typename Element_, int kThreads_ = 256>
struct AddBiasLayerNorm {
  using Element = Element_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kNumThreads = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    Element const* input = nullptr;
    Element const* bias = nullptr;
    Element const* gamma = nullptr;
    Element const* beta = nullptr;
    Element* output = nullptr;

    int32_t tokens = 0;
    int32_t channels = 0;
    float epsilon = 1.0e-5f;

    CUTLASS_HOST dim3 getBlocksGrid() const {
      return dim3(tokens, 1, 1);
    }
    CUTLASS_HOST dim3 getThreadsGrid() const {
      return dim3(kThreads, 1, 1);
    }
  };

  static CUTLASS_DEVICE void run(Params const& p) {
    extern __shared__ float shared[];
    float* sums = shared;
    float* squares = shared + kThreads;

    int token = int(blockIdx.x);
    float sum = 0.0f;
    float square_sum = 0.0f;

    for (int c = threadIdx.x; c < p.channels; c += blockDim.x) {
      float value = float(p.input[int64_t(token) * p.channels + c]);
      if (p.bias != nullptr) {
        value += float(p.bias[c]);
      }
      sum += value;
      square_sum += value * value;
    }

    sums[threadIdx.x] = sum;
    squares[threadIdx.x] = square_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        sums[threadIdx.x] += sums[threadIdx.x + stride];
        squares[threadIdx.x] += squares[threadIdx.x + stride];
      }
      __syncthreads();
    }

    float mean = sums[0] / float(p.channels);
    float variance = squares[0] / float(p.channels) - mean * mean;
    float inv_std = rsqrtf(variance + p.epsilon);

    for (int c = threadIdx.x; c < p.channels; c += blockDim.x) {
      float value = float(p.input[int64_t(token) * p.channels + c]);
      if (p.bias != nullptr) {
        value += float(p.bias[c]);
      }
      float normalized = (value - mean) * inv_std;
      float scale = p.gamma != nullptr ? float(p.gamma[c]) : 1.0f;
      float shift = p.beta != nullptr ? float(p.beta[c]) : 0.0f;
      p.output[int64_t(token) * p.channels + c] =
          Element(normalized * scale + shift);
    }
  }
};

// Fuses norm1 + cyclic shift + window partition into one kernel (FT's
// invokeLayernormShiftPartition). One block per image pixel: it LayerNorms the
// channel vector of the source pixel, then scatters the result to its shifted
// window-token slot. LayerNorm statistics use the raw (pre-norm) input so the
// residual shortcut stays the un-normalized activation.
template <typename Element_, int kThreads_ = 256>
struct LayerNormShiftPartition {
  using Element = Element_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kNumThreads = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    Element const* input = nullptr;   // [B, H, W, C] NHWC
    Element const* gamma = nullptr;   // [C]
    Element const* beta = nullptr;    // [C]
    Element* output = nullptr;        // [BW, L, C] window-partitioned

    int32_t batch = 0;
    int32_t height = 0;
    int32_t width = 0;
    int32_t channels = 0;
    int32_t shift_size = 0;
    int32_t window_size = 0;
    float epsilon = 1.0e-5f;

    CUTLASS_HOST dim3 getBlocksGrid() const {
      return dim3(batch * height * width, 1, 1);
    }
    CUTLASS_HOST dim3 getThreadsGrid() const {
      return dim3(kThreads, 1, 1);
    }
  };

  static CUTLASS_DEVICE void run(Params const& p) {
    extern __shared__ float shared[];
    float* sums = shared;
    float* squares = shared + kThreads;

    int64_t pixel = blockIdx.x;
    warp::WindowMapping mapping = warp::window_mapping(
        pixel, p.height, p.width, p.window_size, p.shift_size);
    int64_t in_base = mapping.image_token * p.channels;
    int64_t out_base = mapping.window_token * p.channels;

    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int c = threadIdx.x; c < p.channels; c += blockDim.x) {
      float value = float(p.input[in_base + c]);
      sum += value;
      square_sum += value * value;
    }

    sums[threadIdx.x] = sum;
    squares[threadIdx.x] = square_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        sums[threadIdx.x] += sums[threadIdx.x + stride];
        squares[threadIdx.x] += squares[threadIdx.x + stride];
      }
      __syncthreads();
    }

    float mean = sums[0] / float(p.channels);
    float variance = squares[0] / float(p.channels) - mean * mean;
    float inv_std = rsqrtf(variance + p.epsilon);

    for (int c = threadIdx.x; c < p.channels; c += blockDim.x) {
      float value = float(p.input[in_base + c]);
      float normalized = (value - mean) * inv_std;
      float scale = p.gamma != nullptr ? float(p.gamma[c]) : 1.0f;
      float shift = p.beta != nullptr ? float(p.beta[c]) : 0.0f;
      p.output[out_base + c] = Element(normalized * scale + shift);
    }
  }
};

// Fuses window reverse + attention residual (residual 1) + norm2 into one
// kernel (FT's invokeGeneralAddBiasResidualPreLayerNorm, minus the bias which
// the projection epilogue already applied). One block per image pixel:
//   residual = shortcut + reverse(projected)
//   normed2  = norm2(residual)
// It gathers the projected window token back to image order, adds the shortcut,
// writes the residual, then LayerNorms it (norm2) for the MLP input.
template <typename Element_, int kThreads_ = 256>
struct ReverseAddResidualLayerNorm {
  using Element = Element_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kNumThreads = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    Element const* projected = nullptr;  // [BW, L, C] attn output projection
    Element const* shortcut = nullptr;   // [B, H, W, C] block input
    Element const* gamma = nullptr;      // norm2 scale [C]
    Element const* beta = nullptr;       // norm2 shift [C]
    Element* residual = nullptr;         // [B, H, W, C] shortcut + attn
    Element* normed = nullptr;           // [B, H, W, C] norm2(residual)

    int32_t batch = 0;
    int32_t height = 0;
    int32_t width = 0;
    int32_t channels = 0;
    int32_t shift_size = 0;
    int32_t window_size = 0;
    float epsilon = 1.0e-5f;

    CUTLASS_HOST dim3 getBlocksGrid() const {
      return dim3(batch * height * width, 1, 1);
    }
    CUTLASS_HOST dim3 getThreadsGrid() const {
      return dim3(kThreads, 1, 1);
    }
  };

  static CUTLASS_DEVICE void run(Params const& p) {
    extern __shared__ float shared[];
    float* sums = shared;
    float* squares = shared + kThreads;

    int64_t pixel = blockIdx.x;
    warp::WindowMapping mapping = warp::window_mapping(
        pixel, p.height, p.width, p.window_size, p.shift_size);
    int64_t image_base = mapping.image_token * p.channels;
    int64_t window_base = mapping.window_token * p.channels;

    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int c = threadIdx.x; c < p.channels; c += blockDim.x) {
      float value = float(p.shortcut[image_base + c]) +
          float(p.projected[window_base + c]);
      p.residual[image_base + c] = Element(value);
      sum += value;
      square_sum += value * value;
    }

    sums[threadIdx.x] = sum;
    squares[threadIdx.x] = square_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        sums[threadIdx.x] += sums[threadIdx.x + stride];
        squares[threadIdx.x] += squares[threadIdx.x + stride];
      }
      __syncthreads();
    }

    float mean = sums[0] / float(p.channels);
    float variance = squares[0] / float(p.channels) - mean * mean;
    float inv_std = rsqrtf(variance + p.epsilon);

    for (int c = threadIdx.x; c < p.channels; c += blockDim.x) {
      float value = float(p.residual[image_base + c]);
      float normalized = (value - mean) * inv_std;
      float scale = p.gamma != nullptr ? float(p.gamma[c]) : 1.0f;
      float shift = p.beta != nullptr ? float(p.beta[c]) : 0.0f;
      p.normed[image_base + c] = Element(normalized * scale + shift);
    }
  }
};

} // namespace threadblock
} // namespace swin
} // namespace tiny_cutlass
