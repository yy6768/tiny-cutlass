#pragma once

/*
  Fixes the NHWC layout and dtype choices for the conv1x1 -> nearest-neighbor
  upsample family on top of the generic DefaultConv2dUpsampleFprop factory
  (mirrors conv1x1_dual/kernel/conv1x1_dual.h's DefaultConv1x1Dual
  for the single-stage upsample case).

  The single conv1x1 stage uses a LinearCombination epilogue with beta = 1 so
  the source tensor C carries a per-channel bias: the host binds C to the bias
  vector with zero spatial strides, so every output pixel reads bias[k]. The
  upsample broadcast happens in the output iterator's store path, not here.
*/

#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/tensor.h"

#include "conv1x1_upsample/kernel/default_conv2d_upsample_fprop.h"

namespace tiny_cutlass::conv_fused::conv1x1_upsample::kernel {

template <
    typename ArchTag_,
    typename Element_,
    int UpsampleH_ = 2,
    int UpsampleW_ = 2,
    typename ThreadblockShape_ = cutlass::gemm::GemmShape<128, 128, 64>,
    typename WarpShape_ = cutlass::gemm::GemmShape<64, 64, 64>>
struct DefaultConv1x1Upsample {
  using ArchTag = ArchTag_;
  using ElementA = Element_;
  using ElementB = Element_;
  using ElementC = Element_;
  using ElementAccumulator = Element_;
  using ElementCompute = Element_;

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  static int const kUpsampleH = UpsampleH_;
  static int const kUpsampleW = UpsampleW_;

  // beta = 1 path pulls a per-channel bias out of the source tensor C.
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::Default>;

  using CutlassKernel = typename kernel::DefaultConv2dUpsampleFprop<
      ElementA,
      cutlass::layout::TensorNHWC,
      ElementB,
      cutlass::layout::TensorNHWC,
      ElementC,
      cutlass::layout::TensorNHWC,
      ElementAccumulator,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      3,
      kUpsampleH,
      kUpsampleW,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
};

}  // namespace tiny_cutlass::conv_fused::conv1x1_upsample::kernel
