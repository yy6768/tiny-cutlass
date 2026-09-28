#pragma once
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "swin/fused_swin_layer/threadblock/mma.h"

namespace tiny_cutlass::swin::fused_swin_layer::threadblock {

template <typename ArchTag, typename Element, typename LayoutB,
          typename ThreadblockShape, typename WarpShape, bool SharedB>
struct DefaultMma {
  using Default = cutlass::gemm::threadblock::DefaultMma<
      Element, cutlass::layout::RowMajor, 8, Element, LayoutB, 8,
      float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, ArchTag,
      ThreadblockShape, WarpShape, cutlass::gemm::GemmShape<16, 8, 16>,
      2, cutlass::arch::OpMultiplyAdd>;
  using MmaCore = typename Default::MmaCore;
  using IteratorA = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      Element, cutlass::layout::RowMajor, 1, typename MmaCore::IteratorThreadMapA>;
  using IteratorB = typename cutlass::platform::conditional<SharedB,
      cutlass::transform::threadblock::RegularTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          Element, LayoutB, 0, typename MmaCore::IteratorThreadMapB>,
      typename Default::IteratorB>::type;
  using ThreadblockMma = Mma<ThreadblockShape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, typename MmaCore::MmaPolicy>;
};

}  // namespace tiny_cutlass::swin::fused_swin_layer::threadblock
