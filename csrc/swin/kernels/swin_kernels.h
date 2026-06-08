/*
  Tiny Swin layout kernels.

  The public Swin path mirrors FasterTransformer at the model level:

    WindowPartition -> QKV projection -> WindowAttention -> output projection
    -> WindowReverse

  This header only keeps the Swin-specific glue around the CUTLASS GEMMs and
  attention core. QK, softmax and PV are not exposed as standalone interfaces.
*/

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

namespace tiny_cutlass {
namespace swin {

////////////////////////////////////////////////////////////////////////////////
// Window partition
////////////////////////////////////////////////////////////////////////////////

template <typename Element_, int kThreads_ = 256>
struct WindowPartitionKernel {
  using Element = Element_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kNumThreads = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    Element const* input = nullptr; // [B, H, W, C]
    Element* output = nullptr;      // [B * nW, window_len, C]

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
    int x = int(pixel % p.width);
    int y = int((pixel / p.width) % p.height);
    int b = int(pixel / (int64_t(p.height) * p.width));

    int shifted_y = p.shift_size != 0
        ? ((y - p.shift_size + p.height) % p.height)
        : y;
    int shifted_x = p.shift_size != 0
        ? ((x - p.shift_size + p.width) % p.width)
        : x;

    int windows_per_row = p.width / p.window_size;
    int window_y = shifted_y / p.window_size;
    int window_x = shifted_x / p.window_size;
    int window_idx = window_y * windows_per_row + window_x;
    int idx_in_window = (shifted_y % p.window_size) * p.window_size
                      + (shifted_x % p.window_size);

    int64_t tokens_per_batch = int64_t(p.height) * p.width;
    int64_t input_token = int64_t(b) * p.height * p.width + y * p.width + x;
    int64_t output_token = int64_t(b) * tokens_per_batch
                         + int64_t(window_idx) * p.window_size * p.window_size
                         + idx_in_window;

    for (int c = threadIdx.x; c < p.channels; c += blockDim.x) {
      p.output[output_token * p.channels + c] =
          p.input[input_token * p.channels + c];
    }
  }
};

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::kNumThreads, Kernel::kMinBlocksPerSm)
    swin_window_partition_kernel(typename Kernel::Params p) {
  Kernel::run(p);
}

////////////////////////////////////////////////////////////////////////////////
// Window reverse
////////////////////////////////////////////////////////////////////////////////

template <typename Element_, int kThreads_ = 256>
struct WindowReverseKernel {
  using Element = Element_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kNumThreads = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    Element const* input = nullptr; // [B * nW, window_len, C]
    Element* output = nullptr;      // [B, H, W, C]

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
    int x = int(pixel % p.width);
    int y = int((pixel / p.width) % p.height);
    int b = int(pixel / (int64_t(p.height) * p.width));

    int shifted_y = p.shift_size != 0
        ? ((y - p.shift_size + p.height) % p.height)
        : y;
    int shifted_x = p.shift_size != 0
        ? ((x - p.shift_size + p.width) % p.width)
        : x;

    int windows_per_row = p.width / p.window_size;
    int window_y = shifted_y / p.window_size;
    int window_x = shifted_x / p.window_size;
    int window_idx = window_y * windows_per_row + window_x;
    int idx_in_window = (shifted_y % p.window_size) * p.window_size
                      + (shifted_x % p.window_size);

    int64_t tokens_per_batch = int64_t(p.height) * p.width;
    int64_t input_token = int64_t(b) * tokens_per_batch
                        + int64_t(window_idx) * p.window_size * p.window_size
                        + idx_in_window;
    int64_t output_token = int64_t(b) * p.height * p.width + y * p.width + x;

    for (int c = threadIdx.x; c < p.channels; c += blockDim.x) {
      p.output[output_token * p.channels + c] =
          p.input[input_token * p.channels + c];
    }
  }
};

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::kNumThreads, Kernel::kMinBlocksPerSm)
    swin_window_reverse_kernel(typename Kernel::Params p) {
  Kernel::run(p);
}

////////////////////////////////////////////////////////////////////////////////
// Patch merge gather
////////////////////////////////////////////////////////////////////////////////

template <typename Element_, int kThreads_ = 256>
struct PatchMergeKernel {
  using Element = Element_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kNumThreads = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    Element const* input = nullptr; // [B, H, W, C]
    Element* output = nullptr;      // [B, H/2, W/2, 4C]

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
    int out_h = p.height / 2;
    int out_w = p.width / 2;
    int64_t pixel = blockIdx.x;
    int ow = int(pixel % out_w);
    int oh = int((pixel / out_w) % out_h);
    int b = int(pixel / (int64_t(out_h) * out_w));

    int64_t output_base =
        (int64_t(b) * out_h * out_w + oh * out_w + ow) * (4 * p.channels);

    for (int col = threadIdx.x; col < 4 * p.channels; col += blockDim.x) {
      int part = col / p.channels;
      int c = col - part * p.channels;
      int offset_w = part / 2;
      int offset_h = part % 2;
      int y = 2 * oh + offset_h;
      int x = 2 * ow + offset_w;
      int64_t input_idx =
          ((int64_t(b) * p.height * p.width + y * p.width + x) * p.channels) + c;
      p.output[output_base + col] = p.input[input_idx];
    }
  }
};

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::kNumThreads, Kernel::kMinBlocksPerSm)
    swin_patch_merge_kernel(typename Kernel::Params p) {
  Kernel::run(p);
}

////////////////////////////////////////////////////////////////////////////////
// QKV bias + split
////////////////////////////////////////////////////////////////////////////////

template <typename Element_, int kThreads_ = 256>
struct AddQkvBiasSplitKernel {
  using Element = Element_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kNumThreads = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    Element const* qkv = nullptr;  // [rows, 3C]
    Element const* bias = nullptr; // [3C], nullable
    Element* q = nullptr;          // [BW, L, heads, D]
    Element* k = nullptr;          // [BW, L, heads, D]
    Element* v = nullptr;          // [BW, L, heads, D]

    int32_t elements = 0; // rows * C
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

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::kNumThreads, Kernel::kMinBlocksPerSm)
    swin_add_qkv_bias_split_kernel(typename Kernel::Params p) {
  Kernel::run(p);
}

////////////////////////////////////////////////////////////////////////////////
// Add output projection bias
////////////////////////////////////////////////////////////////////////////////

template <typename Element_, int kThreads_ = 256>
struct AddBiasKernel {
  using Element = Element_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kNumThreads = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    Element* output = nullptr;     // [rows, C]
    Element const* bias = nullptr; // [C], nullable
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

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::kNumThreads, Kernel::kMinBlocksPerSm)
    swin_add_bias_kernel(typename Kernel::Params p) {
  Kernel::run(p);
}

} // namespace swin
} // namespace tiny_cutlass
