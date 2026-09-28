#pragma once

#include <cutlass/arch/arch.h>
#include <type_traits>
#include <cutlass/array.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor_op_multiplicand_sm80.h>
#include <cutlass/transform/pitch_linear_thread_map.h>
#include <cutlass/transform/threadblock/predicated_tile_access_iterator.h>
#include <cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h>
#include "../warp/nb_atten_mma.h"
#include "../epilogue/nb_atten_epilogue.h"
#include "../threadblock/nb_atten_mma.h"
#include "nb_atten.h"

namespace tiny_cutlass { namespace natten {

// CUTLASS 2.x assembly boundary, analogous to example 13's DefaultB2bMma.
// Physical D and Dv are 64; global iterators predicate runtime 8..64 dims.
// One 64-row query CTA scans only the union of its neighborhood key windows.
template <class ArchTag_, class Element_, class ThreadblockShape_,
          class WarpShapeQK_, class WarpShapePV_,
          class Layout_ = cutlass::layout::RowMajor>
struct DefaultNBAtten {
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShapeQK = WarpShapeQK_;
  using WarpShapePV = WarpShapePV_;
  using Layout = Layout_;
  static constexpr int kBr = ThreadblockShape::kM;
  static constexpr int kBc = ThreadblockShape::kN;
  static constexpr int kPhysicalDim = ThreadblockShape::kK;
  static constexpr int kWarps = kBr / WarpShapeQK::kM;
  static constexpr int kThreads = kWarps * 32;
  static_assert(std::is_same<ArchTag, cutlass::arch::Sm80>::value &&
      std::is_same<Element, cutlass::half_t>::value &&
      std::is_same<Layout, cutlass::layout::RowMajor>::value,
      "Current TensorOp policy supports Sm80, FP16 and row-major");
  static_assert(kBr == 64 && kBc == 64 && kPhysicalDim == 64 &&
      WarpShapeQK::kM == 32 && WarpShapeQK::kN == 64 && WarpShapeQK::kK == 16 &&
      WarpShapePV::kM == 32 && WarpShapePV::kN == 64 && WarpShapePV::kK == 16,
      "Current neighborhood tile is 64x64 with two Q-row warps");
  using LayoutQ = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 64>;
  using LayoutK = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, 64>;
  using LayoutV = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<16, 64>;
  using TileShape = cutlass::layout::PitchLinearShape<64, 64>;
  using CopyMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
      TileShape, kThreads, cutlass::layout::PitchLinearShape<8, 4>, 8>;
  using IteratorQ = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      TileShape, Element const, cutlass::layout::PitchLinear, 1, CopyMap,
      cutlass::Array<Element, 8>>;
  using IteratorK = IteratorQ;
  using IteratorV = IteratorQ;
  using SmemIteratorQ = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<64, 64>, Element, LayoutQ, 0, CopyMap>;
  using SmemIteratorK = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<64, 64>, Element, LayoutK, 1, CopyMap>;
  using SmemIteratorV = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<64, 64>, Element, LayoutV, 0, CopyMap>;
  using WarpQK = NeighborhoodWarpMma<WarpShapeQK, Element, LayoutQ, LayoutK>;
  using WarpPV = NeighborhoodWarpMma<WarpShapePV, Element,
      cutlass::layout::RowMajor, LayoutV>;
  using ProbabilityOutputOp = cutlass::epilogue::thread::LinearCombination<
      Element, 4, float, float, cutlass::epilogue::thread::ScaleType::Nothing>;
  using ProbabilityIterator = cutlass::gemm::warp::MmaTensorOpFragmentIterator<
      cutlass::MatrixShape<WarpShapePV::kM, 16>,
      cutlass::MatrixShape<WarpShapeQK::kM, WarpShapeQK::kN>, 64,
      float, Element, cutlass::layout::ColumnMajor,
      cutlass::gemm::GemmShape<16, 8, 16>, ProbabilityOutputOp>;
  using Mma = NeighborhoodMma<DefaultNBAtten>;
  using Epilogue = NeighborhoodEpilogue<DefaultNBAtten>;
  using Kernel = NeighborhoodKernel<Mma, Epilogue, ArchTag>;
};

}} // namespace tiny_cutlass::natten
