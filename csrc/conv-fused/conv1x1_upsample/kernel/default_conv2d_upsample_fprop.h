#pragma once

/*
  Generic CUTLASS-style kernel policy factory for conv1x1 -> nearest-neighbor
  UpsampleH x UpsampleW upsample.

  The conv itself is an ordinary implicit-GEMM conv1x1 computed at the LOW
  (pre-upsample) resolution, so this factory reuses the stock
  cutlass::conv::kernel::DefaultConv2dFprop to build the threadblock mainloop
  (Mma) and every epilogue component EXCEPT the output tile iterator. The
  output iterator is swapped for conv1x1_upsample's
  PredicatedTileIteratorUpsample, which broadcasts each low-res output pixel
  into its UpsampleH x UpsampleW high-res block during the epilogue store (see
  conv1x1_upsample/epilogue/predicated_tile_iterator_upsample.h). The resulting
  Epilogue is composed with the stock ImplicitGemmConvolution kernel entry, so
  the host side can drive it through cutlass::conv::device::
  ImplicitGemmConvolution without a bespoke device operator.
*/

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/epilogue.h"

#include "conv1x1_upsample/epilogue/predicated_tile_iterator_upsample.h"

namespace tiny_cutlass::conv_fused::conv1x1_upsample::kernel {

template <
    typename ElementA_,
    typename LayoutA_,
    typename ElementB_,
    typename LayoutB_,
    typename ElementC_,
    typename LayoutC_,
    typename ElementAccumulator_,
    typename ArchTag_,
    typename ThreadblockShape_,
    typename WarpShape_,
    typename InstructionShape_,
    typename EpilogueOutputOp_,
    typename ThreadblockSwizzle_,
    int Stages_,
    int UpsampleH_,
    int UpsampleW_,
    typename MathOperatorTag_ = cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm IteratorAlgorithm_ =
        cutlass::conv::IteratorAlgorithm::kOptimized>
struct DefaultConv2dUpsampleFprop {
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using EpilogueOutputOp = EpilogueOutputOp_;

  static int const kUpsampleH = UpsampleH_;
  static int const kUpsampleW = UpsampleW_;

  // Reuse the stock conv fprop factory to build the threadblock mainloop and
  // the default epilogue components. StrideSupport::kUnity is correct because
  // the conv runs at the low resolution with unit output stride between
  // adjacent low-res pixels; the upsample scatter is layered on afterward by
  // swapping the output iterator only.
  using DefaultConv = cutlass::conv::kernel::DefaultConv2dFprop<
      ElementA_,
      LayoutA_,
      ElementB_,
      LayoutB_,
      ElementC_,
      LayoutC_,
      ElementAccumulator_,
      cutlass::arch::OpClassTensorOp,
      ArchTag_,
      ThreadblockShape_,
      WarpShape_,
      InstructionShape_,
      EpilogueOutputOp_,
      ThreadblockSwizzle_,
      Stages_,
      MathOperatorTag_,
      IteratorAlgorithm_,
      cutlass::conv::StrideSupport::kUnity>;

  using Mma = typename DefaultConv::Mma;
  using ThreadblockShape = ThreadblockShape_;
  using WarpMmaTensorOp = typename Mma::Operator;

  static int const kPartitionsK = ThreadblockShape_::kK / WarpShape_::kK;

  // Harvest every default epilogue component (thread map, accumulator fragment
  // iterator, shared-memory warp/load iterators, padding) so only the output
  // tile iterator differs from the stock conv epilogue.
  using DefaultEpilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
      ThreadblockShape_,
      WarpMmaTensorOp,
      kPartitionsK,
      EpilogueOutputOp_,
      EpilogueOutputOp_::kCount>;

  // The upsample output iterator, driven by the same thread map the stock
  // conv epilogue uses.
  using OutputTileIterator = epilogue::PredicatedTileIteratorUpsample<
      typename DefaultEpilogue::OutputTileThreadMap,
      ElementC_,
      kUpsampleH,
      kUpsampleW>;

  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
      ThreadblockShape_,
      WarpMmaTensorOp,
      kPartitionsK,
      OutputTileIterator,
      typename DefaultEpilogue::AccumulatorFragmentIterator,
      typename DefaultEpilogue::WarpTileIterator,
      typename DefaultEpilogue::SharedLoadIterator,
      EpilogueOutputOp_,
      typename DefaultEpilogue::Padding,
      DefaultEpilogue::kFragmentsPerIteration>;

  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
      Mma,
      Epilogue,
      ThreadblockSwizzle_,
      cutlass::conv::Operator::kFprop>;
};

}  // namespace tiny_cutlass::conv_fused::conv1x1_upsample::kernel
