/*
  CUTLASS-style Swin PatchEmbed policy factory.
*/

#pragma once

#include "cutlass/arch/mma.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"

namespace tiny_cutlass {
namespace swin {
namespace kernel {

template <typename Policy_>
struct PatchEmbed {
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
    int kThreads_ = 256>
struct DefaultPatchEmbed {
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using LayoutActivation = cutlass::layout::TensorNHWC;
  using LayoutFilter = cutlass::layout::TensorNHWC;
  using LayoutOutput = cutlass::layout::TensorNHWC;
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
      Element,
      128 / cutlass::sizeof_bits<Element>::value,
      ElementAccumulator,
      ElementCompute>;

  static constexpr int kThreads = kThreads_;
  static constexpr cutlass::conv::Operator kConvolutionalOperator =
      cutlass::conv::Operator::kFprop;
  static constexpr cutlass::conv::IteratorAlgorithm kIteratorAlgorithm =
      cutlass::conv::IteratorAlgorithm::kOptimized;
  static constexpr cutlass::conv::StrideSupport kStrideSupport =
      cutlass::conv::StrideSupport::kStrided;
  static constexpr int kStages = 3;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
      Element,
      LayoutActivation,
      Element,
      LayoutFilter,
      Element,
      LayoutOutput,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      kStages,
      cutlass::arch::OpMultiplyAdd,
      kIteratorAlgorithm,
      kStrideSupport>::Kernel;

  using Kernel = PatchEmbed<DefaultPatchEmbed>;
};

} // namespace kernel
} // namespace swin
} // namespace tiny_cutlass
