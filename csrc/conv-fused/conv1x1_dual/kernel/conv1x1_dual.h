#pragma once

/*
  Teaching wrapper over CUTLASS example 13's DefaultB2bConv2dFprop factory.

  The upstream DefaultB2bConv2dFprop family spans analytic/optimized iterators
  and non-interleaved/interleaved layouts. This local copy keeps the two paths
  this teaching family instantiates: OpClassTensorOp + kOptimized + NHWC, with
  either RF or SMEM intermediate residency. The wrapper fixes the 3-stage
  multistage pipeline and 16x8x16 MMA while keeping arch, element type, shapes,
  and residency template-driven.

      Residency::kRF    stage-0 accumulator stays in the register file
      Residency::kSmem  stage-0 accumulator is staged through shared memory

  The residency choice flips the SmemAccumulator bool of the underlying factory
  and, because the two paths have different tile-shape constraints, selects a
  different default stage-1 tile:

      RF   : threadblock_N1 == warp_N1 (each warp owns the full stage-1 filter
             kBlock in registers). N1 is capped by the register budget, so the
             default caps stage-1 N at 128.
      SMEM : that constraint is relaxed (warp_N1 < threadblock_N1), so stage-1 N
             can grow. The default uses N1 = 256 to show the regime where RF
             would spill.

  This is the crux of the series: RF removes the intermediate shared-memory
  round trip, while SMEM permits a wider stage-1 N tile. Which path is faster
  is deliberately left to a verified benchmark/profile comparison.
*/

#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/tensor.h"

#include "conv1x1_dual/kernel/default_b2b_conv2d_fprop_sm80.h"
#include "conv1x1_dual/kernel/default_b2b_conv2d_fprop_smem_accumulator_sm80.h"
#include "conv1x1_dual/threads/epilogue_ops.h"

namespace tiny_cutlass::conv_fused::kernel {

/// Which memory space carries the stage-0 accumulator into stage 1.
enum class Residency {
  kRF,    // register-file resident (SmemAccumulator = false)
  kSmem,  // shared-memory staged    (SmemAccumulator = true)
};

namespace detail {

// Per-residency defaults. Both stages share ThreadblockShape::kM = 64 and
// K = 32; only stage-1 N and the warp shapes differ, because that is exactly
// what the RF vs SMEM constraint governs.
template <Residency Kind>
struct DualShapeDefaults;

// RF: warp_N1 == threadblock_N1 == 128 (the register-resident constraint).
template <>
struct DualShapeDefaults<Residency::kRF> {
  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape0 = cutlass::gemm::GemmShape<32, 64, 32>;
  using WarpShape1 = cutlass::gemm::GemmShape<32, 128, 32>;
};

// SMEM: warp_N1 (64) < threadblock_N1 (256); the smem stage relaxes the
// constraint so stage-1 N can grow past what fits in registers.
template <>
struct DualShapeDefaults<Residency::kSmem> {
  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 256, 32>;
  using WarpShape0 = cutlass::gemm::GemmShape<32, 32, 32>;
  using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 32>;
};

}  // namespace detail

template <
    typename ArchTag_,
    typename Element_,
    Residency ResidencyKind = Residency::kRF,
    typename ThreadblockShape0_ =
        typename detail::DualShapeDefaults<ResidencyKind>::ThreadblockShape0,
    typename ThreadblockShape1_ =
        typename detail::DualShapeDefaults<ResidencyKind>::ThreadblockShape1,
    typename WarpShape0_ =
        typename detail::DualShapeDefaults<ResidencyKind>::WarpShape0,
    typename WarpShape1_ =
        typename detail::DualShapeDefaults<ResidencyKind>::WarpShape1>
struct DefaultConv1x1Dual {
  using ArchTag = ArchTag_;
  using ElementA = Element_;
  using ElementB = Element_;
  using ElementC = Element_;
  using ElementAccumulator = Element_;
  using ElementCompute = Element_;

  static Residency const kResidency = ResidencyKind;
  static bool const kSmemAccumulator = (ResidencyKind == Residency::kSmem);

  using ThreadblockShape0 = ThreadblockShape0_;
  using ThreadblockShape1 = ThreadblockShape1_;
  using WarpShape0 = WarpShape0_;
  using WarpShape1 = WarpShape1_;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  // Stage 0: bias0 + ReLU folded into the fused mainloop (alpha-only scaling).
  using EpilogueOutputOp0 = threads::Conv0Relu<
      ElementC,
      ElementAccumulator,
      ElementCompute,
      InstructionShape::kM * InstructionShape::kN / 32>;
  // Stage 1: plain linear combination, bias1 via source tensor C (beta = 1).
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
      cutlass::conv::IteratorAlgorithm::kOptimized,
      kSmemAccumulator>::Kernel;
};

}  // namespace tiny_cutlass::conv_fused::kernel
