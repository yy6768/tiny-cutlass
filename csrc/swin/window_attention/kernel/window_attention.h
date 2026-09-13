#pragma once

#include "cutlass/cutlass.h"

namespace tiny_cutlass::swin::window_attention::kernel {

template <typename Mma_>
struct WindowAttention {
  using Mma = Mma_;
  using Params = typename Mma::Params;
  using SharedStorage = typename Mma::SharedStorage;
  static constexpr int kThreadCount = Mma::kWarps * 32;

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared) {
    Mma{}(params, shared, int(blockIdx.x));
  }
};

}  // namespace tiny_cutlass::swin::window_attention::kernel
