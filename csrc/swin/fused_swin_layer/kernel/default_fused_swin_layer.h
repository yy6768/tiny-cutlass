#pragma once
#include "cutlass/gemm/kernel/default_gemm.h"
#include "swin/fused_swin_layer/threadblock/default_mma.h"
#include "swin/fused_swin_layer/threadblock/fused_swin_layer.h"
#include "swin/fused_swin_layer/epilogue/default_epilogue.h"
#include "swin/fused_swin_layer/kernel/fused_swin_layer.h"

namespace tiny_cutlass::swin::fused_swin_layer::kernel {

template <typename ArchTag_, typename Element_,
          typename ThreadblockShape = cutlass::gemm::GemmShape<16, 32, 32>,
          typename WarpShape = cutlass::gemm::GemmShape<16, 32, 32>>
struct DefaultFusedSwinLayer {
  using ArchTag = ArchTag_;
  using Element = Element_;
  static_assert(cutlass::platform::is_same<ArchTag, cutlass::arch::Sm80>::value ||
                cutlass::platform::is_same<ArchTag, cutlass::arch::Sm89>::value, "unsupported architecture");
  static_assert(cutlass::platform::is_same<Element, cutlass::half_t>::value, "unsupported element type");
  static_assert(ThreadblockShape::kM == 16 && ThreadblockShape::kN == 32 && ThreadblockShape::kK == 32 &&
                WarpShape::kM == 16 && WarpShape::kN == 32 && WarpShape::kK == 32, "unsupported tile");
  using Mma0 = typename threadblock::DefaultMma<ArchTag, Element, cutlass::layout::ColumnMajor,
      ThreadblockShape, WarpShape, false>::ThreadblockMma;
  using Mma1 = typename threadblock::DefaultMma<ArchTag, Element, cutlass::layout::ColumnMajor,
      ThreadblockShape, WarpShape, true>::ThreadblockMma;
  using Mma2 = typename threadblock::DefaultMma<ArchTag, Element, cutlass::layout::RowMajor,
      ThreadblockShape, WarpShape, true>::ThreadblockMma;
  using Epilogue = typename epilogue::DefaultEpilogue<ThreadblockShape, typename Mma0::Operator>::Epilogue;
  using Mma = threadblock::FusedSwinLayer<Mma0, Mma1, Mma2, Epilogue>;
  using CutlassKernel = FusedSwinLayer<Mma, Epilogue>;
};

}  // namespace tiny_cutlass::swin::fused_swin_layer::kernel
