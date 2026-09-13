#pragma once

/*
  RF-residency variant of the fused conv2d fprop factory (Sm80, NHWC,
  optimized iterators only).

  Stage-0 warp accumulator stays in the register file and is streamed straight
  into stage-1 MMA through a warp fragment iterator (FragmentIteratorA1). No
  shared-memory round trip for the intermediate activation.

  This residency requires warp_N1 == threadblock_N1: each warp must own the full
  stage-1 filter kBlock in registers, so stage-1 N is bounded by the register
  budget. This family's RF policy caps N1 at 128; wider N1 problems use the
  SMEM variant in
  default_b2b_conv2d_fprop_smem_accumulator_sm80.h.

  Trimmed from CUTLASS example 13's default_b2b_conv2d_fprop_sm80.h to the single
  specialization this teaching family exercises: OpClassTensorOp + kOptimized +
  non-interleaved NHWC + SmemAccumulator = false. The dropped specializations
  (analytic iterators, NCxHWx interleaved layouts) are never instantiated here.
*/

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d.h"

#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_optimized.h"

#include "cutlass/transform/threadblock/predicated_vector_access_iterator.h"
#include "cutlass/transform/threadblock/vector_iterator.h"
#include "cutlass/transform/warp/vector_fragment_iterator.h"

#include "cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h"

#include "conv1x1_dual/kernel/default_b2b_conv2d_fprop.h"
#include "conv1x1_dual/kernel/b2b_implicit_gemm_convolution.h"
#include "conv1x1_dual/threadblock/b2b_implicit_gemm_multistage.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Conv2dFprop, Optimized iterators, multistage pipeline, register-file
/// residency (SmemAccumulator = false, the default).
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape0,
  typename ThreadblockShape1,
  typename WarpShape0,
  typename WarpShape1,
  typename InstructionShape,
  typename EpilogueOutputOp0,
  typename EpilogueOutputOp1,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag
>
struct DefaultB2bConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape0,
  ThreadblockShape1,
  WarpShape0,
  WarpShape1,
  InstructionShape,
  EpilogueOutputOp0,
  EpilogueOutputOp1,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized
> {

  // Define the core components from GEMM
  using MmaCore0 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape0, WarpShape0, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, MathOperatorTag>;
  using MmaCore1 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape1, WarpShape1, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA0 = typename MmaCore0::IteratorThreadMapA;
  using IteratorA0 =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape0::kM, ThreadblockShape0::kK>,
      ElementA, LayoutA,
      ThreadMapA0
    >;

  using SmemIteratorA0 = typename MmaCore0::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB0 = typename MmaCore0::IteratorThreadMapB;
  using IteratorB0 =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape0::kK, ThreadblockShape0::kN>,
      ElementB, LayoutB,
      ThreadMapB0
    >;

  using SmemIteratorB0 = typename MmaCore0::SmemIteratorB;

  // Use fragment iterator for A operand: stage-1's A tiles come straight from
  // stage-0's warp accumulator in the register file, with output_op_0 (bias0 +
  // ReLU) applied as the fragment is read.
  using AccumulatorLayout = cutlass::layout::ColumnMajor;
  using FragmentIteratorA1 =
      cutlass::gemm::warp::MmaTensorOpFragmentIterator<
          cutlass::MatrixShape<MmaCore1::WarpShape::kM, MmaCore1::InstructionShape::kK>, //warp shape
          cutlass::MatrixShape<MmaCore0::WarpShape::kM, MmaCore0::WarpShape::kN>, //accumulator shape
          MmaCore1::Shape::kK, //kBlocksColumn
          ElementAccumulator, ElementA, AccumulatorLayout, InstructionShape, EpilogueOutputOp0>;

  /// Define iterators over tiles from scale/bias vectors
  using ElementScaleBias = typename EpilogueOutputOp0::ElementCompute;
  using LayoutScaleBias = layout::RowMajor; //vector layout doesn't really matter
  static int const kElementsPerAccess = 2;
  using IteratorAccumulatorScaleBias =
    cutlass::transform::threadblock::VectorIterator<
      cutlass::transform::threadblock::PredicatedVectorAccessIterator<
          cutlass::MatrixShape<ThreadblockShape0::kM, ThreadblockShape0::kN>,
          cutlass::MatrixShape<WarpShape1::kM, WarpShape1::kK>,
          ElementScaleBias, LayoutScaleBias, kElementsPerAccess>
    >;

  // Warp-level iterators to load scale and bias vectors
  using FragmentIteratorA1ScaleBias = cutlass::transform::warp::VectorFragmentIterator<
      MatrixShape<1, IteratorAccumulatorScaleBias::Fragment::kElements>, ElementScaleBias,
      LayoutScaleBias, InstructionShape, kElementsPerAccess>;

  // Define iterators over tiles from the B operand
  using ThreadMapB1 = typename MmaCore1::IteratorThreadMapB;
  using IteratorB1 =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape1::kK, ThreadblockShape1::kN>,
      ElementB, LayoutB,
      ThreadMapB1
    >;

  using SmemIteratorB1 = typename MmaCore1::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp1 = typename MmaCore1::MmaTensorOp;
  using MmaPolicy0 = typename MmaCore0::MmaPolicy;
  using MmaPolicy1 = typename MmaCore1::MmaPolicy;

  // Define the Mma
  using B2bMma = threadblock::B2bImplicitGemmMultistage<
    ThreadblockShape0,
    IteratorA0,
    SmemIteratorA0,
    arch::CacheOperation::Always,
    IteratorB0,
    SmemIteratorB0,
    arch::CacheOperation::Global,
    ThreadblockShape1,
    FragmentIteratorA1,
    IteratorAccumulatorScaleBias,
    FragmentIteratorA1ScaleBias,
    IteratorB1,
    SmemIteratorB1,
    arch::CacheOperation::Global,
    EpilogueOutputOp0,
    MmaPolicy0,
    MmaPolicy1,
    Stages
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape1,
    WarpMmaTensorOp1,
    1,
    EpilogueOutputOp1,
    EpilogueOutputOp1::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::B2bImplicitGemmConvolution<
    B2bMma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
