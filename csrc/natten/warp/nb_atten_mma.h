#pragma once

#include <cutlass/array.h>
#include <cutlass/gemm/warp/default_mma_tensor_op.h>

namespace tiny_cutlass { namespace natten {

// CUTLASS's canonical warp TensorOp and shared-memory iterators. The fragment
// coordinate helpers below describe the m16n8k16 accumulator arrangement.
template <class Shape_, class Element_, class LayoutA_, class LayoutB_>
struct NeighborhoodWarpMma {
  using Shape = Shape_;
  using Element = Element_;
  using Mma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, cutlass::gemm::GemmShape<16, 8, 16>, Element, LayoutA_,
      Element, LayoutB_, float, cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd>::Type;
  using Fragment = typename Mma::FragmentC;
  using FragmentA = typename Mma::FragmentA;
  using FragmentB = typename Mma::FragmentB;
  using OperandA = typename Mma::TransformedFragmentA;
  using OperandB = typename Mma::TransformedFragmentB;
  using IteratorA = typename Mma::IteratorA;
  using IteratorB = typename Mma::IteratorB;
  using IteratorC = typename Mma::IteratorC;
  using FragmentRow = cutlass::Array<float, Shape::kM / 8>;

  CUTLASS_HOST_DEVICE static int row_slot(int i) {
    return ((i / 4) % (Shape::kM / 16)) * 2 + (i % 4) / 2;
  }
  CUTLASS_HOST_DEVICE static int row(int slot, int lane) {
    return (slot / 2) * 16 + (slot % 2) * 8 + lane / 4;
  }
  CUTLASS_HOST_DEVICE static int column(int i, int lane) {
    return (i / 4 / (Shape::kM / 16)) * 8 + (lane % 4) * 2 + i % 2;
  }

  CUTLASS_DEVICE void operator()(IteratorA a, IteratorB b,
                                 int iterations, Fragment& accum) const {
    Mma mma;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < iterations; ++k) {
      FragmentA frag_a; FragmentB frag_b;
      OperandA operand_a; OperandB operand_b;
      a.load(frag_a); b.load(frag_b);
      ++a; ++b;
      mma.transform(operand_a, operand_b, frag_a, frag_b);
      mma(accum, operand_a, operand_b, accum);
    }
  }
};

}} // namespace tiny_cutlass::natten
