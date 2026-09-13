#pragma once

/*
  Policy factory for PatchEmbed:

      out = LayerNorm_channel(conv(x, patch x patch, stride patch) + bias)

  ---------------------------------------------------------------------------
  WHY THE TILE SHAPE IS PART OF THE CONTRACT
  ---------------------------------------------------------------------------
  The single-kernel LayerNorm only works when a whole output row (all channels)
  is owned by ONE threadblock, i.e. ThreadblockShape::kN >= embed_dim so that
  grid.n() == 1. If N were tiled, each CTA would see a slice of the row and the
  channel-axis statistics would be partial -- which is exactly why CUTLASS
  example 37 needs three launches.

  can_implement() in the device layer enforces kN >= channels at runtime; the
  visitor additionally static_asserts Iterations::kColumn == 1, which is the
  in-kernel consequence of the same choice.

  Default TB 128x128x32 / warp 64x64x32 measured for fp16 output:

      kThreads 128, kElementsPerAccess 8, Iterations::kColumn 1,
      kThreadsPerRow 16

  ---------------------------------------------------------------------------
  WHY THE ACTIVATION CHANNEL COUNT IS PADDED
  ---------------------------------------------------------------------------
  A TensorOp fp16 mainloop needs an 8-element-wide load along C, and this family
  takes no SIMT fallback, so the 3-channel image is padded to 8 before the conv
  runs. The padding columns are zero and the filter's matching taps are zero, so
  they contribute nothing to the dot product.
*/

#include "cutlass/arch/mma.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/tensor.h"

#include "swin/patch_embed/epilogue/layernorm_visitor.h"
#include "swin/patch_embed/kernel/implicit_gemm_convolution_with_visitor.h"

namespace tiny_cutlass::swin::patch_embed::kernel {

template <
    typename ArchTag_,
    typename Element_,
    typename ElementAccumulator_ = float,
    typename ElementCompute_ = float,
    typename ThreadblockShape_ = cutlass::gemm::GemmShape<128, 128, 32>,
    typename WarpShape_ = cutlass::gemm::GemmShape<64, 64, 32>,
    typename InstructionShape_ = cutlass::gemm::GemmShape<16, 8, 16>,
    int Stages = 3>
struct DefaultPatchEmbed {
  using ArchTag = ArchTag_;
  using ElementA = Element_;
  using ElementB = Element_;
  using ElementC = Element_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  static int const kStages = Stages;

  using LayoutActivation = cutlass::layout::TensorNHWC;
  using LayoutFilter = cutlass::layout::TensorNHWC;
  using LayoutOutput = cutlass::layout::TensorNHWC;

  static int const kAlignment = 8;

  // Placeholder output op: only its element types and vector width are used, to
  // let DefaultConv2dFprop build the standard epilogue whose iterators and
  // shared-load machinery the visitor epilogue then reuses. The functor itself
  // is never invoked -- the visitor replaces it.
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute>;

  using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  /// Stock conv kernel, used only as a source of Mma + epilogue building blocks.
  using BaseKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
      ElementA, LayoutActivation,
      ElementB, LayoutFilter,
      ElementC, LayoutOutput,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      ArchTag,
      ThreadblockShape, WarpShape, InstructionShape,
      EpilogueOutputOp,
      Swizzle,
      kStages,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized,
      cutlass::conv::StrideSupport::kStrided,
      kAlignment, kAlignment>::Kernel;

  using Mma = typename BaseKernel::Mma;
  using BaseEpilogue = typename BaseKernel::Epilogue;
  using OutputTileIterator = typename BaseEpilogue::OutputTileIterator;
  using AccumulatorTile =
      typename BaseEpilogue::AccumulatorFragmentIterator::AccumulatorTile;

  /// bias + channel-axis LayerNorm, computed entirely inside the epilogue.
  using Visitor = epilogue::EpilogueVisitorLayerNorm<
      ThreadblockShape,
      BaseKernel::kThreadCount,
      OutputTileIterator,
      AccumulatorTile,
      ElementAccumulator,
      ElementCompute>;

  /// Rebuild the epilogue around the visitor, reusing every iterator and the
  /// shared-load path from the stock conv epilogue.
  using Epilogue = typename cutlass::epilogue::threadblock::
      EpilogueWithVisitorFromExistingEpilogue<Visitor, BaseEpilogue>::Epilogue;

  using CutlassKernel = ImplicitGemmConvolutionWithVisitor<
      Mma,
      Epilogue,
      Swizzle,
      cutlass::conv::Operator::kFprop,
      cutlass::conv::Conv2dProblemSize>;
};

}  // namespace tiny_cutlass::swin::patch_embed::kernel
