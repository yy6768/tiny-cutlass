#pragma once

#include "cutlass/array.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"

// WarpShape describes one warp's logical output tile. CUTLASS owns the
// ldmatrix iterators, operand transform and m16n8k16 instruction scheduling.
template <class WarpShape, class Element, class LayoutA, class LayoutB>
struct FlashAttnWarpMma {
  using Mma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, cutlass::gemm::GemmShape<16, 8, 16>,
      Element, LayoutA, Element, LayoutB,
      float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>::Type;
  using Fragment = typename Mma::FragmentC;
  using IteratorA = typename Mma::IteratorA;
  using IteratorB = typename Mma::IteratorB;
  static constexpr int kRows = WarpShape::kM / 16 * 2;
  using FragmentRow = cutlass::Array<float, kRows>;

  // MmaTensorOp uses column-major ordering of the m16n8 instruction tiles
  // (AccumulatorsInRowMajor=false), even though the logical output is row-major.
  CUTLASS_HOST_DEVICE
  static int row_slot(int i) { return ((i / 4) % (WarpShape::kM / 16)) * 2 + (i % 4) / 2; }
  CUTLASS_HOST_DEVICE
  static int row(int slot, int lane) { return (slot / 2) * 16 + (slot % 2) * 8 + lane / 4; }
  CUTLASS_HOST_DEVICE
  static int column(int i, int lane) { return (i / 4 / (WarpShape::kM / 16)) * 8 + (lane % 4) * 2 + i % 2; }

  CUTLASS_DEVICE
  void operator()(IteratorA a, IteratorB b, int iterations, Fragment& accum) const {
    Mma mma;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < iterations; ++k) {
      typename Mma::FragmentA frag_a;
      typename Mma::FragmentB frag_b;
      typename Mma::TransformedFragmentA operand_a;
      typename Mma::TransformedFragmentB operand_b;
      a.load(frag_a);
      b.load(frag_b);
      ++a;
      ++b;
      mma.transform(operand_a, operand_b, frag_a, frag_b);
      mma(accum, operand_a, operand_b, accum);
    }
  }
};
