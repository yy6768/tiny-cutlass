/*
  Threadblock-level Swin attention glue kernels.
*/

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

namespace tiny_cutlass {
namespace swin {
namespace threadblock {

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

// Fuses fc1 bias + GELU (FT's invokeAddBiasGeluV2). Uses the exact (erf) GELU
// to match PyTorch nn.GELU default. Works in-place on the fc1 output.
template <typename Element_, int kThreads_ = 256>
struct AddBiasGelu {
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
    int idx = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= p.elements) {
      return;
    }
    int c = idx % p.channels;
    float value = float(p.output[idx]);
    if (p.bias != nullptr) {
      value += float(p.bias[c]);
    }
    // Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))).
    float gelu = value * 0.5f * (1.0f + erff(value * 0.7071067811865476f));
    p.output[idx] = Element(gelu);
  }
};

// Fuses fc2 bias + MLP residual (residual 2, FT's invokeAddBiasResidual).
// out = residual + (fc2_out + bias). The residual source is the attention
// residual buffer produced by ReverseAddResidualLayerNorm.
template <typename Element_, int kThreads_ = 256>
struct AddBiasResidual {
  using Element = Element_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kNumThreads = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    Element const* input = nullptr;     // fc2 output [rows, C]
    Element const* bias = nullptr;      // [C]
    Element const* residual = nullptr;  // [rows, C]
    Element* output = nullptr;          // [rows, C]
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
    float value = float(p.input[idx]);
    if (p.bias != nullptr) {
      value += float(p.bias[c]);
    }
    value += float(p.residual[idx]);
    p.output[idx] = Element(value);
  }
};

} // namespace threadblock
} // namespace swin
} // namespace tiny_cutlass
