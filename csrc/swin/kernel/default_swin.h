/*
  CUTLASS-style Swin kernel policy factory.
*/

#pragma once

#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

namespace tiny_cutlass {
namespace swin {
namespace kernel {

template <typename Policy_>
struct Swin {
  using Policy = Policy_;
  using Element = typename Policy::Element;
};

template <
    typename ArchTag_,
    typename Element_,
    typename ThreadblockShape_ = cutlass::gemm::GemmShape<128, 64, 64>,
    typename WarpShape_ = cutlass::gemm::GemmShape<64, 32, 64>,
    typename InstructionShape_ = cutlass::gemm::GemmShape<16, 8, 16>,
    typename ElementAccumulator_ = float,
    typename ElementCompute_ = float,
    typename MatrixLayout_ = cutlass::layout::RowMajor,
    int kThreads_ = 256,
    int kAttentionQueriesPerBlock_ = 64,
    int kAttentionKeysPerBlock_ = 64,
    int kAttentionMaxHeadDim_ = 64>
struct DefaultSwin {
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using MatrixLayout = MatrixLayout_;

  static constexpr int kThreads = kThreads_;
  static constexpr int kAttentionQueriesPerBlock = kAttentionQueriesPerBlock_;
  static constexpr int kAttentionKeysPerBlock = kAttentionKeysPerBlock_;
  static constexpr int kAttentionMaxHeadDim = kAttentionMaxHeadDim_;
  static constexpr bool kAttentionSupportsBias = true;
  static constexpr bool kAttentionIsCausal = false;
  static constexpr bool kAttentionIsAligned = true;

  using Kernel = Swin<DefaultSwin>;
};

} // namespace kernel
} // namespace swin
} // namespace tiny_cutlass
