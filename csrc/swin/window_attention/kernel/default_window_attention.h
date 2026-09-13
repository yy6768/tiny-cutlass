#pragma once

#include <type_traits>
#include "cutlass/arch/arch.h"
#include "cutlass/half.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "swin/window_attention/threadblock/window_attention_mma.h"
#include "swin/window_attention/kernel/window_attention.h"

namespace tiny_cutlass::swin::window_attention::kernel {

template <typename ArchTag_, typename Element_,
          typename ThreadblockShape_ = cutlass::gemm::GemmShape<16, 128, 32>,
          typename WarpShape_ = cutlass::gemm::GemmShape<16, 32, 32>,
          typename Layout_ = cutlass::layout::RowMajor,
          int MaxQk = 64, int MaxValue = 64>
struct DefaultWindowAttention {
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  static_assert(std::is_same_v<ArchTag, cutlass::arch::Sm80> ||
                std::is_same_v<ArchTag, cutlass::arch::Sm89>, "unsupported architecture policy");
  static_assert(std::is_same_v<Element, cutlass::half_t>, "unsupported element policy");
  static_assert(std::is_same_v<Layout_, cutlass::layout::RowMajor>, "contiguous token rows required");
  static_assert(WarpShape::kM == 16 && WarpShape::kN == 32 && WarpShape::kK == 32,
                "this window implementation requires a 16x32x32 warp tile");
  static_assert(ThreadblockShape::kM == 16 && ThreadblockShape::kK == 32 &&
                ThreadblockShape::kN % WarpShape::kN == 0, "unsupported CTA tile");
  static constexpr int kWarps = ThreadblockShape::kN / WarpShape::kN;
  static_assert(kWarps >= 1 && kWarps <= 8, "unsupported warp count");
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, 32>;
  using WarpMma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, cutlass::gemm::GemmShape<16, 8, 16>,
      Element, LayoutA, Element, LayoutB, float, cutlass::layout::RowMajor>::Type;
  using Mma = threadblock::WindowAttentionMma<WarpMma, kWarps, MaxQk, MaxValue>;
  using CutlassKernel = WindowAttention<Mma>;
};

}  // namespace tiny_cutlass::swin::window_attention::kernel
