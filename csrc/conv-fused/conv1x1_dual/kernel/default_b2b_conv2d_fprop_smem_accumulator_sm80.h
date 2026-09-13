#pragma once

/*
  SMEM-residency variant of the fused conv2d fprop factory (Sm80, NHWC,
  optimized iterators only).

  This is the sibling of default_b2b_conv2d_fprop_sm80.h. The only structural
  difference is what carries the stage-0 accumulator into stage 1:

    RF   variant: stage-0 warp accumulator stays in the register file and is
                  streamed straight into stage-1 MMA via a fragment iterator.
                  Requires warp_N1 == threadblock_N1 (each warp owns the full
                  stage-1 filter kBlock), so N1 is capped by register budget.

    SMEM variant: stage-0 accumulator is written to shared memory
                  (SmemIteratorD0), then re-loaded per warp tile via
                  WarpIteratorA1. This relaxes the warp_N1 == threadblock_N1
                  constraint, letting stage-1 N grow (e.g. 256) at the cost of
                  one extra shared-memory round trip.

  Trimmed from CUTLASS example 13's
  default_b2b_conv2d_fprop_smem_accumulator_sm80.h to the single specialization
  this teaching family exercises: OpClassTensorOp + kOptimized + non-interleaved
  NHWC + SmemAccumulator = true.
*/

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d.h"

#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_optimized.h"

#include "cutlass/transform/threadblock/predicated_vector_access_iterator.h"
#include "cutlass/transform/threadblock/vector_iterator.h"
#include "cutlass/transform/warp/vector_fragment_iterator.h"

#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"

#include "conv1x1_dual/kernel/default_b2b_conv2d_fprop.h"
#include "conv1x1_dual/kernel/b2b_implicit_gemm_convolution.h"
#include "conv1x1_dual/threadblock/b2b_implicit_gemm_multistage_smem_accumulator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Conv2dFprop, Optimized iterators, multistage pipeline, accumulator staged in
/// shared memory (SmemAccumulator = true).
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
  IteratorAlgorithm::kOptimized,
  true
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

  /// Define iterators over tiles from scale/bias vectors
  using ElementScaleBias = typename EpilogueOutputOp0::ElementCompute;
  using LayoutScaleBias = layout::RowMajor; //vector layout doesn't really matter
  static int const kElementsPerAccess = 2;
  using IteratorAccumulatorScaleBias =
    cutlass::transform::threadblock::VectorIterator<
      cutlass::transform::threadblock::PredicatedVectorAccessIterator<
          cutlass::MatrixShape<ThreadblockShape0::kM, ThreadblockShape0::kN>,
          cutlass::MatrixShape<WarpShape0::kM, WarpShape0::kN>,
          ElementScaleBias, LayoutScaleBias, kElementsPerAccess>
    >;

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
  using WarpMmaTensorOp0 = typename MmaCore0::MmaTensorOp;
  using WarpMmaTensorOp1 = typename MmaCore1::MmaTensorOp;
  using MmaPolicy0 = typename MmaCore0::MmaPolicy;
  using MmaPolicy1 = typename MmaCore1::MmaPolicy;

  // Use fragment iterator for the accumulator
  using SmemAccumulatorLayout = cutlass::layout::RowMajor;
  using FragmentIteratorAccumulator = cutlass::epilogue::warp::FragmentIteratorTensorOp<
          WarpShape0, InstructionShape,
          ElementAccumulator,
          typename WarpMmaTensorOp0::Policy::Operator::FragmentC,
          SmemAccumulatorLayout
        >;

  // Store Accumulator tiles to Shared Memory
  using SmemIteratorD0 =
      cutlass::epilogue::warp::TileIteratorTensorOp<
          WarpShape0,
          InstructionShape,
          ElementC,
          SmemAccumulatorLayout
        >;

  static int const kThreadCount = 32;
  // load warp tile from Shared Memory accumulator
  using WarpIteratorA1 = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIterator<
    MatrixShape<WarpShape1::kM, InstructionShape::kK>, cutlass::gemm::Operand::kA,
    ElementA, SmemAccumulatorLayout,
    MatrixShape<InstructionShape::kM, InstructionShape::kK>,
    WarpMmaTensorOp1::Policy::OpDelta::kRow, kThreadCount>;

  // Define the Mma
  using B2bMma = threadblock::B2bImplicitGemmMultistageSmemAccumulator<
    ThreadblockShape0,
    IteratorA0,
    SmemIteratorA0,
    arch::CacheOperation::Always,
    IteratorB0,
    SmemIteratorB0,
    arch::CacheOperation::Global,
    IteratorAccumulatorScaleBias,
    FragmentIteratorAccumulator,
    SmemIteratorD0,
    ThreadblockShape1,
    WarpIteratorA1,
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
