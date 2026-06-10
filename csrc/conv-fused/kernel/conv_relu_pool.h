#pragma once

#include "cutlass/arch/mma.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/functional.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/reduction/device/tensor_reduce_affine_strided.h"

namespace tiny_cutlass::conv_fused::kernel {

template <
    typename ArchTag_,
    typename Element_,
    typename ThreadblockShape_ = cutlass::gemm::GemmShape<128, 64, 32>,
    typename WarpShape_ = cutlass::gemm::GemmShape<64, 32, 32>>
struct DefaultConvRelu {
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationRelu<
      Element,
      128 / cutlass::sizeof_bits<Element>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

  using CutlassKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
      Element,
      cutlass::layout::TensorNHWC,
      Element,
      cutlass::layout::TensorNHWC,
      Element,
      cutlass::layout::TensorNHWC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      3,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized,
      cutlass::conv::StrideSupport::kUnity>::Kernel;

  using Operation =
      cutlass::conv::device::ImplicitGemmConvolution<CutlassKernel>;
};

template <
    typename Element_,
    typename ElementCompute_ = float,
    int VectorLength_ = 128 / cutlass::sizeof_bits<Element_>::value>
struct DefaultPool {
  using Element = Element_;
  using ElementCompute = ElementCompute_;
  using ReductionOp = cutlass::maximum<ElementCompute>;

  using Operation = cutlass::reduction::device::TensorReductionAffineStrided<
      6,
      4,
      Element,
      Element,
      ReductionOp,
      VectorLength_,
      ElementCompute>;
};

}  // namespace tiny_cutlass::conv_fused::kernel
