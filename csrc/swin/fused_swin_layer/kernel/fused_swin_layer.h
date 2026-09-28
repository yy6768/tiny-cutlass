#pragma once
#include "swin/fused_swin_layer/ops/fused_swin_layer.h"

namespace tiny_cutlass::swin::fused_swin_layer::kernel {

template <typename Mma_, typename Epilogue_>
struct FusedSwinLayer {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using Params = Arguments<typename Mma::Element>;
  using SharedStorage = typename Mma::SharedStorage;
  static int const kThreadCount = 32;

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared) {
    Mma{}(params, shared, int(blockIdx.x));
  }
};

}  // namespace tiny_cutlass::swin::fused_swin_layer::kernel
