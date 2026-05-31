#pragma once

#include "cutlass/gemm/gemm.h"

namespace tiny_cutlass::conv_fused::threadblock {

struct Sm80RfResident {
  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 128, 32>;
};

}  // namespace tiny_cutlass::conv_fused::threadblock
