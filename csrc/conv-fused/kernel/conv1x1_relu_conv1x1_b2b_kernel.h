#pragma once

#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/tensor.h"

#include "device/b2b_implicit_gemm_convolution.h"
#include "kernel/default_b2b_conv2d_fprop_sm80.h"

#include "threadblock/conv1x1_relu_conv1x1_threadblock_shapes.h"
#include "threads/epilogue_ops.h"
#include "warp/conv1x1_relu_conv1x1_warp_shapes.h"

namespace tiny_cutlass::conv_fused::kernel {

template <typename Element>
struct Conv1x1ReluConv1x1B2bSm80 {
  using ElementA = Element;
  using ElementB = Element;
  using ElementC = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using ThreadblockShape0 =
      typename threadblock::Sm80RfResident::ThreadblockShape0;
  using ThreadblockShape1 =
      typename threadblock::Sm80RfResident::ThreadblockShape1;
  using WarpShape0 = typename warp::Sm80RfResident::WarpShape0;
  using WarpShape1 = typename warp::Sm80RfResident::WarpShape1;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using EpilogueOutputOp0 =
      threads::Conv0Relu<ElementC, ElementAccumulator, ElementCompute>;
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
      cutlass::arch::Sm80,
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
