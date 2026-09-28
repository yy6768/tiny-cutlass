#pragma once

#include "cutlass/cutlass.h"

namespace tiny_cutlass::swin::encoder::kernel {

template <typename Mma_>
struct Encoder {
  using Mma = Mma_;
  using Params = typename Mma::Params;
  using SharedStorage = typename Mma::SharedStorage;
  static int const kThreadCount = 32;

  CUTLASS_DEVICE
  void operator()(Params const& args, SharedStorage& shared) const {
    int tile = int(blockIdx.x);
    if (tile >= args.problem.tiles) return;
    Mma{}(args, shared, tile);
  }
};

}  // namespace tiny_cutlass::swin::encoder::kernel
