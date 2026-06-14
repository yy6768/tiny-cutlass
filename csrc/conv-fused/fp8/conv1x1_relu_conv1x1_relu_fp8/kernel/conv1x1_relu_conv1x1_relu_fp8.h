#pragma once

#include "cutlass/arch/mma.h"
#include "cutlass/arch/mma_sm89.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"

#include "kernel/default_b2b_conv2d_fprop.h"
#include "kernel/default_b2b_conv2d_fprop_smem_accumulator_sm80.h"

#include "fp8/conv1x1_relu_conv1x1_relu_fp8/threads/conv1x1_relu_conv1x1_relu_fp8_epilogue_ops.h"

namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu::kernel {

template <
    typename ArchTag_,
    typename ElementA_ = cutlass::float_e4m3_t,
    typename ElementB_ = ElementA_,
    typename ElementC_ = ElementA_,
    typename ElementAccumulator_ = float,
    typename ElementCompute_ = float,
    typename ThreadblockShape0_ = cutlass::gemm::GemmShape<64, 64, 64>,
    typename ThreadblockShape1_ = cutlass::gemm::GemmShape<64, 64, 64>,
    typename WarpShape0_ = cutlass::gemm::GemmShape<32, 64, 64>,
    typename WarpShape1_ = cutlass::gemm::GemmShape<32, 64, 64>>
struct DefaultConv1x1ReluConv1x1ReluFp8 {
  using ArchTag = ArchTag_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  using ThreadblockShape0 = ThreadblockShape0_;
  using ThreadblockShape1 = ThreadblockShape1_;
  using WarpShape0 = WarpShape0_;
  using WarpShape1 = WarpShape1_;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  using EpilogueOutputOp0 = threads::Stage0ReluQuantize<
      ElementC,
      ElementAccumulator,
      ElementCompute,
      InstructionShape::kM * InstructionShape::kN / 32>;
  using EpilogueOutputOp1 = threads::Stage1ReluQuantize<
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
      cutlass::conv::IteratorAlgorithm::kOptimized,
      true>::Kernel;
};

}  // namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu::kernel
