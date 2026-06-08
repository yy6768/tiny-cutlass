#pragma once

#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/tensor.h"

#include "device/b2b_implicit_gemm_convolution.h"
#include "kernel/default_b2b_conv2d_fprop_sm80.h"

#include "threads/epilogue_ops.h"

namespace tiny_cutlass::conv_fused::kernel {

template <
    typename ArchTag_,
    typename Element_,
    typename ThreadblockShape0_ = cutlass::gemm::GemmShape<64, 64, 32>,
    typename ThreadblockShape1_ = cutlass::gemm::GemmShape<64, 128, 32>,
    typename WarpShape0_ = cutlass::gemm::GemmShape<32, 64, 32>,
    typename WarpShape1_ = cutlass::gemm::GemmShape<32, 128, 32>>
struct DefaultConv1x1ReluConv1x1 {
  using ArchTag = ArchTag_;
  using ElementA = Element_;
  using ElementB = Element_;
  using ElementC = Element_;
  using ElementAccumulator = Element_;
  using ElementCompute = Element_;

  using ThreadblockShape0 = ThreadblockShape0_;
  using ThreadblockShape1 = ThreadblockShape1_;
  using WarpShape0 = WarpShape0_;
  using WarpShape1 = WarpShape1_;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using EpilogueOutputOp0 =
      threads::Conv0Relu<
          ElementC,
          ElementAccumulator,
          ElementCompute,
          cutlass::gemm::GemmShape<16, 8, 16>::kM * cutlass::gemm::GemmShape<16, 8, 16>::kN / 32>;
  using EpilogueOutputOp1 = threads::Conv1Linear<
      ElementC,
      ElementAccumulator,
      ElementCompute,
      128 / cutlass::sizeof_bits<ElementC>::value>;

  using CutlassKernel = typename cutlass::conv::kernel::DefaultB2bConv2dFprop<
      ElementA,
      cutlass::layout::TensorNHWC,
      ElementB,
      cutlass::layout::TensorNHWC,
      ElementC,
      cutlass::layout::TensorNHWC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      ArchTag,
      ThreadblockShape0,
      ThreadblockShape1,
      WarpShape0,
      WarpShape1,
      InstructionShape,
      EpilogueOutputOp0,
      EpilogueOutputOp1,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      3,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
};

}  // namespace tiny_cutlass::conv_fused::kernel
