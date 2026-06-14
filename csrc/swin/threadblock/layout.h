/*
  Threadblock-level Swin layout and vector kernels.
*/

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

#include "../warp/layout.h"

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

template <typename Element_, int kThreads_ = 256>
struct PatchMerge {
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

    CUTLASS_HOST dim3 getBlocksGrid() const {
      return dim3(batch * (height / 2) * (width / 2), 1, 1);
    }
    CUTLASS_HOST dim3 getThreadsGrid() const {
      return dim3(kThreads, 1, 1);
    }
  };

  static CUTLASS_DEVICE void run(Params const& p) {
    int64_t pixel = blockIdx.x;
    int out_h = p.height / 2;
    int out_w = p.width / 2;
    int b = int(pixel / (int64_t(out_h) * out_w));
    int oh = int((pixel / out_w) % out_h);
    int ow = int(pixel % out_w);
    int64_t output_base =
        (int64_t(b) * out_h * out_w + oh * out_w + ow) * (4 * p.channels);

    for (int col = threadIdx.x; col < 4 * p.channels; col += blockDim.x) {
      int64_t input_idx =
          warp::patch_merge_input_token(pixel, p.height, p.width, p.channels, col);
      p.output[output_base + col] = p.input[input_idx];
    }
  }
};

template <typename Element_, int kThreads_ = 256>
struct AddQkvBiasSplit {
  using Element = Element_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kNumThreads = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    Element const* qkv = nullptr;
    Element const* bias = nullptr;
    Element* q = nullptr;
    Element* k = nullptr;
    Element* v = nullptr;

    int32_t elements = 0;
    int32_t channels = 0;

    CUTLASS_HOST dim3 getBlocksGrid() const {
      return dim3((elements + kThreads - 1) / kThreads, 1, 1);
    }
    CUTLASS_HOST dim3 getThreadsGrid() const {
      return dim3(kThreads, 1, 1);
    }
  };

  static CUTLASS_DEVICE void run(Params const& p) {
    int idx = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= p.elements) {
      return;
    }
    int c = idx % p.channels;
    int row = idx / p.channels;
    int64_t base = int64_t(row) * (3 * p.channels);
    Element qv = p.qkv[base + c];
    Element kv = p.qkv[base + p.channels + c];
    Element vv = p.qkv[base + 2 * p.channels + c];
    if (p.bias != nullptr) {
      qv = Element(float(qv) + float(p.bias[c]));
      kv = Element(float(kv) + float(p.bias[p.channels + c]));
      vv = Element(float(vv) + float(p.bias[2 * p.channels + c]));
    }
    p.q[idx] = qv;
    p.k[idx] = kv;
    p.v[idx] = vv;
  }
};

template <typename Element_, int kThreads_ = 256>
struct AddBias {
  using Element = Element_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kNumThreads = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    Element* output = nullptr;
    Element const* bias = nullptr;
    int32_t elements = 0;
    int32_t channels = 0;

    CUTLASS_HOST dim3 getBlocksGrid() const {
      return dim3((elements + kThreads - 1) / kThreads, 1, 1);
    }
    CUTLASS_HOST dim3 getThreadsGrid() const {
      return dim3(kThreads, 1, 1);
    }
  };

  static CUTLASS_DEVICE void run(Params const& p) {
    if (p.bias == nullptr) {
      return;
    }
    int idx = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= p.elements) {
      return;
    }
    int c = idx % p.channels;
    p.output[idx] = Element(float(p.output[idx]) + float(p.bias[c]));
  }
};

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
