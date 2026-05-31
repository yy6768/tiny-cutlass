#pragma once

#include "cutlass/gemm/gemm.h"

namespace tiny_cutlass::conv_fused::warp {

struct Sm80RfResident {
  using WarpShape0 = cutlass::gemm::GemmShape<32, 64, 32>;
  using WarpShape1 = cutlass::gemm::GemmShape<32, 128, 32>;
};

}  // namespace tiny_cutlass::conv_fused::warp
