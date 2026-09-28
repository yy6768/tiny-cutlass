#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h"
#include "../../02-split-kv/warp/flash_attn_mma.h"
#include "../epilogue/flash_attn_backward_epilogue.h"
#include "../threadblock/flash_attn_backward_mma.h"
#include "flash_attn_backward.h"

// Example 13-style policy factory. Shapes and physical channel padding are
// supplied by the device instantiation; logical D remains the public extent.
template <class ArchTag_, class Element_, class ThreadblockShape_,
          class WarpShapeScore_, class WarpShapeGrad_, class WarpShapeDq_,
          class Layout_ = cutlass::layout::RowMajor>
struct DefaultFlashAttnBackward {
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShapeScore = WarpShapeScore_;
  using WarpShapeGrad = WarpShapeGrad_;
  using WarpShapeDq = WarpShapeDq_;
  using Layout = Layout_;
  static constexpr int kBc = ThreadblockShape::kM;
  static constexpr int kBr = ThreadblockShape::kN;
  static constexpr int kHeadDim = ThreadblockShape::kK;
  static constexpr int kStorageDim = WarpShapeGrad::kN;
  static constexpr int kThreadCount = 128;
  static constexpr int kCanonicalCopyThreads = 32;
  static_assert(cutlass::platform::is_same<Element, cutlass::half_t>::value &&
      cutlass::platform::is_same<Layout, cutlass::layout::RowMajor>::value,
      "Backward policy supports half row-major tensors only");
  static_assert(kBr == 64 && kBc == 64 && WarpShapeScore::kM == 16 &&
      WarpShapeScore::kN == kBr && WarpShapeGrad::kM == 16,
      "Backward KV ownership requires four 16-row warps");
  static_assert(WarpShapeDq::kM == 32 && WarpShapeDq::kN * 2 == kStorageDim &&
      WarpShapeDq::kN >= 32 && kStorageDim >= kHeadDim,
      "dQ uses two Q-row groups and two channel groups with congruous tile offsets");
  static_assert(kHeadDim % 32 == 0 && kStorageDim % 64 == 0 &&
      WarpShapeScore::kK == 16 && WarpShapeGrad::kK == 16 && WarpShapeDq::kK == 16,
      "Complete TensorOp instruction groups are required");

  using CanonicalThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<kHeadDim, kBr>, kCanonicalCopyThreads,
      cutlass::layout::PitchLinearShape<4, 8>, 8>;
  using CanonicalIterator = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      cutlass::layout::PitchLinearShape<kHeadDim, kBr>, Element const,
      cutlass::layout::PitchLinear, 1, CanonicalThreadMap, cutlass::Array<Element, 8>>;
  using CanonicalSmemIterator = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::layout::PitchLinearShape<kHeadDim, kBr>, Element,
      cutlass::layout::PitchLinear, 1, CanonicalThreadMap>;

  // As in example 13 DefaultB2bMma: use the core's matching ThreadMap,
  // TensorOp shared layout and access iterator. Only one tile is allocated.
  using MmaCoreGrad = cutlass::gemm::threadblock::DefaultMmaCore<
      cutlass::gemm::GemmShape<kBc, kStorageDim, kBr>,
      cutlass::gemm::GemmShape<16, kStorageDim, kBr>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      Element, cutlass::layout::RowMajor, Element, cutlass::layout::RowMajor,
      float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, 3>;
  using SwizzleLayout = typename MmaCoreGrad::SmemLayoutB;
  using SwizzleThreadMap = typename MmaCoreGrad::IteratorThreadMapB;
  using SwizzleSmemIterator = typename MmaCoreGrad::SmemIteratorB;
  using SwizzleIterator = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      cutlass::layout::PitchLinearShape<kStorageDim, kBr>, Element const,
      cutlass::layout::PitchLinear, 1, SwizzleThreadMap, cutlass::Array<Element, 8>>;
  using ScoreLayout = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<16, 64>;
  using TransposedScoreLayout = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<16, 64>;
  using WarpScore = FlashAttnWarpMma<WarpShapeScore, Element,
      cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;
  using WarpGrad = FlashAttnWarpMma<WarpShapeGrad, Element,
      cutlass::layout::RowMajor, SwizzleLayout>;
  using WarpDq = FlashAttnWarpMma<WarpShapeDq, Element,
      TransposedScoreLayout, SwizzleLayout>;
  using ProbabilityOutputOp = cutlass::epilogue::thread::LinearCombination<
      Element, 4, float, float, cutlass::epilogue::thread::ScaleType::Nothing>;
  using FragmentIterator = cutlass::gemm::warp::MmaTensorOpFragmentIterator<
      cutlass::MatrixShape<WarpShapeGrad::kM, 16>,
      cutlass::MatrixShape<WarpShapeScore::kM, WarpShapeScore::kN>, kBr,
      float, Element, cutlass::layout::ColumnMajor,
      cutlass::gemm::GemmShape<16, 8, 16>, ProbabilityOutputOp>;
  using Epilogue = FlashAttnBackwardEpilogue<WarpGrad, Element>;
  using AtomicEpilogue = FlashAttnBackwardAtomicEpilogue<WarpDq>;
  using Mma = FlashAttnBackwardMma<DefaultFlashAttnBackward>;
  using Kernel = FlashAttnBackwardKernel<Mma, Epilogue, ArchTag>;
};
