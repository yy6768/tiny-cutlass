#pragma once

#include <type_traits>
#include "cutlass/arch/arch.h"
#include "cutlass/half.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "swin/encoder/threadblock/encoder.h"
#include "swin/encoder/kernel/encoder.h"

namespace tiny_cutlass::swin::encoder::kernel {

template <typename ArchTag_, typename Element_,
          typename ThreadblockShape_ = cutlass::gemm::GemmShape<16, 32, 32>,
          typename WarpShape_ = cutlass::gemm::GemmShape<16, 32, 32>>
struct DefaultEncoder {
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  static_assert(std::is_same_v<ArchTag, cutlass::arch::Sm80> ||
                std::is_same_v<ArchTag, cutlass::arch::Sm89>, "unsupported architecture");
  static_assert(std::is_same_v<Element, cutlass::half_t>, "the PTX contract requires half");
  static_assert(std::is_same_v<ThreadblockShape, WarpShape> && WarpShape::kM == 16 &&
                WarpShape::kN == 32 && WarpShape::kK == 32, "unsupported tile policy");
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, 32>;
  using WarpMma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, cutlass::gemm::GemmShape<16, 8, 8>, Element, LayoutA,
      Element, LayoutB, Element, cutlass::layout::RowMajor>::Type;
  using Mma = threadblock::Encoder<WarpMma>;
  using CutlassKernel = Encoder<Mma>;
};

}  // namespace tiny_cutlass::swin::encoder::kernel
