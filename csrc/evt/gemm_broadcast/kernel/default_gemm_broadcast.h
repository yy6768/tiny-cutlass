/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief CUTLASS-style factory for a GEMM whose epilogue is a real Epilogue Visitor Tree.

    Ported from CUTLASS example 47
    (examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk_broadcast.cu)
    to study the 2x EVT path end to end. The math is a residual block:

        D = ((acc + bias_row) + C1) + C2

    where `acc = A . B`, `bias_row` is a length-N vector broadcast along M, and C1/C2 are
    full MxN residual tensors. Every term is applied inside ONE GEMM epilogue by the visitor
    tree below -- no separate elementwise kernels, no intermediate global buffers.

    ------------------------------------------------------------------------------------------------
    THE VISITOR TREE
    ------------------------------------------------------------------------------------------------
    Written as a tree (node op first, then children):

        EVTStore = Sm80EVT<AuxStore(D),
                     Sm80EVT<Compute(plus),                        // + C2
                       Sm80EVT<Compute(plus),                      // + C1
                         Sm80EVT<Compute(plus),                    // acc + bias
                           AccFetch,                               //   the A.B accumulator
                           RowBroadcast(bias)>,                    //   length-N vector, M-broadcast
                         AuxLoad(C1)>,                             //   full MxN
                       AuxLoad(C2)>>                               //   full MxN

    Two things this file is meant to teach:

      1. `StrideMNL` is where the tensor's SHAPE lives, not just its stride. The bias uses
         `Stride<_0, _1, int32_t>` -- a compile-time 0 in the M mode makes the M term vanish
         from address arithmetic, which IS the broadcast. C1/C2/D use
         `Stride<int64_t, _1, int64_t>` (row stride, contiguous columns, batch stride).

      2. All four global-touching nodes share ONE `OutputTileThreadMap`. The thread map decides
         how a CTA's output tile is partitioned across threads, so nodes can only line up on the
         same `frg_idx` if they agree on it.

    ------------------------------------------------------------------------------------------------
    STREAMK
    ------------------------------------------------------------------------------------------------
    The swizzle is a template parameter. `DefaultGemmWithVisitor::SelectBase` SFINAEs on
    `ThreadblockSwizzle::StreamkFeature`, so passing `ThreadblockSwizzleStreamK` selects
    `GemmWithEpilogueVisitorStreamk` while a plain swizzle selects `GemmWithEpilogueVisitor`.
    The visitor tree is identical either way -- StreamK's fixup path replays the same callback
    sequence through `EpilogueWithVisitorCallbacks::reduce()`. Both are instantiated by the test
    so the two can be compared directly.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

#include "cute/tensor.hpp"

namespace tiny_cutlass {
namespace evt {
namespace kernel {

// ---------------------------------------------------------------------------
// DefaultGemmBroadcast
//
//   D = ((A.B + bias_row) + C1) + C2, fused in a single Sm80EVT epilogue.
//
//   ThreadblockSwizzle selects the data-parallel vs StreamK kernel; everything
//   else is a normal CUTLASS TensorOp GEMM policy. Instantiable on SM80/86/89.
// ---------------------------------------------------------------------------
template <
    typename ArchTag_,
    typename Element_,
    typename ElementAccumulator_ = float,
    typename ElementCompute_ = float,
    typename LayoutA_ = cutlass::layout::RowMajor,
    typename LayoutB_ = cutlass::layout::RowMajor,
    typename LayoutC_ = cutlass::layout::RowMajor,
    typename ThreadblockShape_ = cutlass::gemm::GemmShape<128, 128, 32>,
    typename WarpShape_ = cutlass::gemm::GemmShape<64, 64, 32>,
    typename InstructionShape_ = cutlass::gemm::GemmShape<16, 8, 16>,
    typename ThreadblockSwizzle_ =
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    int Stages_ = 4,
    int EpilogueStages_ = 1>
struct DefaultGemmBroadcast {
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  using LayoutA = LayoutA_;
  using LayoutB = LayoutB_;
  using LayoutC = LayoutC_;

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr int kStages = Stages_;
  static constexpr int kEpilogueStages = EpilogueStages_;

  // 128-bit accesses for every operand. The test only runs problem shapes whose
  // M/N/K are multiples of this, so no predication-driven alignment drop is needed.
  static constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<Element>::value;
  static constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<Element>::value;
  static constexpr int kAlignmentC = 128 / cutlass::sizeof_bits<Element>::value;

  // -- EVT epilogue ----------------------------------------------------------
  // One thread map shared by bias / C1 / C2 / D. Lives in
  // epilogue/threadblock/fusion/visitor_2x.hpp (pulled in via visitors.hpp) --
  // NOT in the similarly named threadblock/output_tile_thread_map.h.
  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape, WarpShape, Element, kAlignmentC, kEpilogueStages>;

  // Leaf: the raw A.B accumulator.
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  // Leaf: length-N bias vector broadcast along M. The compile-time _0 in the M
  // mode is the broadcast; mode 2 is the batch stride.
  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, Element,
      cute::Stride<cute::_0, cute::_1, int32_t>>;

  // Leaves: two full MxN residual tensors.
  using ResidualC1 = cutlass::epilogue::threadblock::VisitorAuxLoad<
      OutputTileThreadMap, Element,
      cute::Stride<int64_t, cute::_1, int64_t>>;

  using ResidualC2 = cutlass::epilogue::threadblock::VisitorAuxLoad<
      OutputTileThreadMap, Element,
      cute::Stride<int64_t, cute::_1, int64_t>>;

  // Three identical binary adds. ComputeFn is a `template <class> class` slot,
  // so only single-template-parameter functors fit (cutlass::plus is fine here;
  // multiply_add would need the `homogeneous_` wrapper).
  using ComputeAddBias = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using ComputeAddC1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  // Last compute converts down to the output element type.
  using ComputeAddC2 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, Element, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTAddBias = cutlass::epilogue::threadblock::Sm80EVT<
      ComputeAddBias, Accum, Bias>;

  using EVTAddC1 = cutlass::epilogue::threadblock::Sm80EVT<
      ComputeAddC1, EVTAddBias, ResidualC1>;

  using EVTAddC2 = cutlass::epilogue::threadblock::Sm80EVT<
      ComputeAddC2, EVTAddC1, ResidualC2>;

  // Root: store D. AuxStore is a pass-through node (visit returns its input and
  // the actual global_store happens in end_step), which is what lets a tree
  // carry several outputs on one path.
  using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, Element, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t>>;

  using EVTStore = cutlass::epilogue::threadblock::Sm80EVT<Store, EVTAddC2>;

  // -- GEMM kernel / device adapter ------------------------------------------
  // ElementC/AlignmentC still feed the discarded LinearCombination that
  // DefaultGemmWithVisitor uses internally to derive Mma + the default epilogue
  // descriptor, so they must match the narrowest element in the tree.
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      Element, LayoutA, cutlass::ComplexTransform::kNone, kAlignmentA,
      Element, LayoutB, cutlass::ComplexTransform::kNone, kAlignmentB,
      Element, LayoutC, kAlignmentC,
      ElementAccumulator,
      ElementCompute,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EVTStore,
      ThreadblockSwizzle,
      kStages,
      cutlass::arch::OpMultiplyAdd,
      kEpilogueStages>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

} // namespace kernel
} // namespace evt
} // namespace tiny_cutlass
