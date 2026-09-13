#pragma once

/*
  Policy factory for PatchMerging, the Swin stage transition:

      [B, H, W, C] -> concat 2x2 neighbourhood -> [B, H/2, W/2, 4C]
                   -> LayerNorm(4C) -> linear 4C -> 2C

  ---------------------------------------------------------------------------
  TWO LAUNCHES, NOT THREE
  ---------------------------------------------------------------------------
  Note the norm comes BEFORE the projection here -- the opposite of PatchEmbed,
  where it follows the conv. As argued in
  swin/layernorm/kernel/token_layernorm.h, a LayerNorm preceding a GEMM has to
  be its own pass, because it reduces along what is the GEMM's K axis.

  Since that pass is unavoidable, the 2x2 concat rides along inside it: the
  LayerNorm's read path takes a segmented gather, so it assembles each 4C row
  from four input rows on the fly. The [out_tokens, 4C] concatenated tensor is
  therefore NEVER materialized -- no extra kernel, no extra HBM round trip. This
  is the same "fold data movement into an existing pass" trick that turns window
  partition into GatherA on the QKV GEMM.

  So:
    launch 1  gather 2x2 + LayerNorm(4C)   -> [out_tokens, 4C]
    launch 2  linear 4C -> 2C (+ optional bias in the epilogue)

  Official concat order is
      cat([x[0::2, 0::2], x[1::2, 0::2], x[0::2, 1::2], x[1::2, 1::2]], -1)
  i.e. segment k corresponds to (di, dj) = (0,0), (1,0), (0,1), (1,1); the table
  is built by build_patch_merging_index() and verified by swin_window_index.
*/

#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

#include "swin/layernorm/device/token_layernorm.h"

namespace tiny_cutlass::swin::patch_merging::kernel {

template <
    typename ArchTag_,
    typename Element_,
    typename ElementAccumulator_ = float,
    typename ElementCompute_ = float,
    typename ThreadblockShape_ = cutlass::gemm::GemmShape<128, 128, 32>,
    typename WarpShape_ = cutlass::gemm::GemmShape<64, 64, 32>,
    typename InstructionShape_ = cutlass::gemm::GemmShape<16, 8, 16>,
    int Stages = 3>
struct DefaultPatchMerging {
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  static int const kStages = Stages;
  static int const kAlignment = 8;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;  // weight is [2C, 4C] row-major
  using LayoutC = cutlass::layout::RowMajor;

  /// Gather + LayerNorm in one pass. num_segments = 4 for the 2x2 concat.
  using LayerNorm = layernorm::device::TokenLayerNorm<Element, ElementCompute>;

  static int const kEpilogueElementsPerAccess =
      128 / cutlass::sizeof_bits<Element>::value;

  /// Projection epilogue: optional per-column bias, no activation, no residual
  /// (PatchMerging has no skip connection -- its output shape differs from its
  /// input shape).
  using ReductionEpilogueOp =
      cutlass::epilogue::thread::LinearCombinationBiasElementwise<
          Element,
          ElementAccumulator,
          ElementCompute,
          Element,
          Element,
          kEpilogueElementsPerAccess,
          cutlass::epilogue::thread::Identity<ElementCompute>,
          cutlass::plus<ElementCompute>,
          /*StoreT=*/false,
          ElementCompute>;

  using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using Reduction = cutlass::gemm::device::GemmUniversalWithBroadcast<
      Element, LayoutA,
      Element, LayoutB,
      Element, LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      ArchTag,
      ThreadblockShape, WarpShape, InstructionShape,
      ReductionEpilogueOp,
      Swizzle,
      kStages,
      kAlignment, kAlignment>;
};

}  // namespace tiny_cutlass::swin::patch_merging::kernel
