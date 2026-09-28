#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/array.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h"
#include "../../02-split-kv/warp/flash_attn_mma.h"
#include "cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "../threadblock/flash_attn_mma.h"
#include "../epilogue/flash_attn_epilogue.h"
#include "flash_attn.h"

// DefaultXxx is a policy factory, matching the assembly boundary in example 13.
template <class ArchTag_, class Element_, class ThreadblockShape_,
          class WarpShapeQK_, class WarpShapePV_,
          class Layout_ = cutlass::layout::RowMajor>
struct DefaultFlashAttnSplitQ {
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShapeQK = WarpShapeQK_;
  using WarpShapePV = WarpShapePV_;
  using Layout = Layout_;
  static constexpr int kBr = ThreadblockShape::kM;
  static constexpr int kBc = ThreadblockShape::kN;
  static constexpr int kHeadDim = ThreadblockShape::kK;
  static constexpr int kWarpsM = kBr / WarpShapeQK::kM;
  static constexpr int kThreadCount = kWarpsM * 32;
  static constexpr int kHeadDimV = kHeadDim;
  static constexpr int kStorageBc = kBc;
  static constexpr int kPitchQ = kHeadDim;
  static constexpr int kPitchV = WarpShapePV::kN;
  static_assert(kBr % WarpShapeQK::kM == 0 &&
      WarpShapePV::kM == WarpShapeQK::kM, "QK/PV must own identical Q rows");
  static_assert(WarpShapeQK::kN == kBc && WarpShapePV::kN >= kHeadDimV,
      "Each warp owns all KV columns and output channels; no split-K reduction");
  static_assert(WarpShapeQK::kK == 16 && WarpShapePV::kK == 16 &&
      kBc % 16 == 0 && kHeadDim % 16 == 0, "Complete TensorOp instruction groups required");
  static_assert(cutlass::platform::is_same<Element, cutlass::half_t>::value &&
      cutlass::platform::is_same<Layout, cutlass::layout::RowMajor>::value,
      "This policy supports half row-major operands only");
  // QK keeps the standard canonical TensorOp operand iterators used by 02.
  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutP = LayoutQ;
  // Example 13 composes the stock MmaCore shared layout, ThreadMap and iterator.
  // Only V uses this optimized path; Q/K retain their current canonical layout.
  // Stages=3 selects SM80 access-iterator traits; this Mma allocates one V tile,
  // not the complete three-stage DefaultMmaCore pipeline.
  using MmaCorePV = cutlass::gemm::threadblock::DefaultMmaCore<
      cutlass::gemm::GemmShape<kBr, kPitchV, kBc>,
      cutlass::gemm::GemmShape<WarpShapePV::kM, kPitchV, kBc>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      Element, cutlass::layout::RowMajor, Element, cutlass::layout::RowMajor,
      float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, 3>;
  using LayoutV = typename MmaCorePV::SmemLayoutB;

  // Assemble stock global/shared access iterators here, as in example 13's
  // DefaultB2bMma. All operands use vector accesses; V uses MmaCore's swizzle.
  static constexpr int kCopyThreadsQ = 32;
  static constexpr int kCopyThreadsK = 32;
  static constexpr int kCopyThreadsV = MmaCorePV::kThreads;
  static_assert(kCopyThreadsV > 0 && kCopyThreadsV <= kThreadCount,
      "The CTA must contain every thread in V's copy map");
  using ThreadMapQ = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<kPitchQ, kBr>, kCopyThreadsQ,
      cutlass::layout::PitchLinearShape<4, 8>, 8>;
  using ThreadMapK = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<kPitchQ, kStorageBc>, kCopyThreadsK,
      cutlass::layout::PitchLinearShape<4, 8>, 8>;
  using ThreadMapV = typename MmaCorePV::IteratorThreadMapB;

  // Global tile advancement uses logical Bc; the thread maps cover physical storage.
  using IteratorQ = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      cutlass::layout::PitchLinearShape<kPitchQ, kBr>, Element const,
      cutlass::layout::PitchLinear, 1, ThreadMapQ, cutlass::Array<Element, 8>>;
  using IteratorK = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      cutlass::layout::PitchLinearShape<kPitchQ, kBc>, Element const,
      cutlass::layout::PitchLinear, 1, ThreadMapK, cutlass::Array<Element, 8>>;
  using IteratorV = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      cutlass::layout::PitchLinearShape<kPitchV, kBc>, Element const,
      cutlass::layout::PitchLinear, 1, ThreadMapV, cutlass::Array<Element, 8>>;
  using SmemIteratorQ = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::layout::PitchLinearShape<kPitchQ, kBr>, Element,
      cutlass::layout::PitchLinear, 1, ThreadMapQ>;
  using SmemIteratorK = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::layout::PitchLinearShape<kPitchQ, kStorageBc>, Element,
      cutlass::layout::PitchLinear, 1, ThreadMapK>;
  using SmemIteratorV = typename MmaCorePV::SmemIteratorB;

  using WarpQK = FlashAttnWarpMma<WarpShapeQK, Element, LayoutQ, LayoutK>;
  using WarpPV = FlashAttnWarpMma<WarpShapePV, Element, LayoutP, LayoutV>;
  // Example 13 default_b2b_mma.h: accumulator -> next GEMM A, no shared P.
  using ProbabilityOutputOp = cutlass::epilogue::thread::LinearCombination<
      Element, 4, float, float, cutlass::epilogue::thread::ScaleType::Nothing>;
  using FragmentIteratorP = cutlass::gemm::warp::MmaTensorOpFragmentIterator<
      cutlass::MatrixShape<WarpShapePV::kM, 16>,
      cutlass::MatrixShape<WarpShapeQK::kM, WarpShapeQK::kN>, kBc,
      float, Element, cutlass::layout::ColumnMajor,
      cutlass::gemm::GemmShape<16, 8, 16>, ProbabilityOutputOp>;
  using Mma = FlashAttnMmaSplitQ<DefaultFlashAttnSplitQ>;
  using Epilogue = FlashAttnEpilogueSplitQ<DefaultFlashAttnSplitQ>;
  using Kernel = FlashAttnKernelSplitQ<Mma, Epilogue, ArchTag>;
};
