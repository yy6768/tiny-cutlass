#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/array.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"
#include "../warp/flash_attn_mma.h"
#include "../threadblock/flash_attn_mma.h"
#include "../epilogue/flash_attn_epilogue.h"
#include "flash_attn.h"

// DefaultXxx is a policy factory, matching the assembly boundary in example 13.
template <class ArchTag_, class Element_, class ThreadblockShape_,
          class WarpShapeQK_, class WarpShapePV_,
          class Layout_ = cutlass::layout::RowMajor>
struct DefaultFlashAttn {
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ThreadblockShape0 = ThreadblockShape_;
  using WarpShapeQK = WarpShapeQK_;
  using WarpShapePV = WarpShapePV_;
  using Layout = Layout_;
  static constexpr int kBr = ThreadblockShape0::kM;
  static constexpr int kBc = ThreadblockShape0::kN;
  static constexpr int kHeadDim = ThreadblockShape0::kK;
  static constexpr int kWarpsM = kBr / WarpShapeQK::kM;
  // The stock raked copy map requires its strided warp count to divide the
  // physical tile. Br=48 has three compute warps but needs four copy warps.
  static_assert(kWarpsM == 2 || kWarpsM == 3 || kWarpsM == 4,
      "Supported Q tiles require two to four row warps");
  static constexpr int kCopyWarps = kWarpsM == 3 ? 4 : kWarpsM;
  static constexpr int kThreadCount = kCopyWarps * 32;
  static constexpr int kHeadDimV = kHeadDim;
  // Logical Bc remains the paper's tile size. Only the instruction footprint
  // is padded (Bc=48 -> 64); copy predicates and softmax masks exclude it.
  static constexpr int kStorageBc = WarpShapeQK::kN;
  // QK: [Br, D] x [D, Bc]; PV: [Br, Bc] x [Bc, Dv].
  // The physical PV reduction footprint includes the padded KV columns.
  using ThreadblockShape1 = cutlass::gemm::GemmShape<kBr, kHeadDimV, kStorageBc>;
  static constexpr int kPitchQ = kHeadDim;
  static constexpr int kPitchV = kHeadDimV == 96 ? 128 : kHeadDimV;
  static_assert(kBr % WarpShapeQK::kM == 0 && WarpShapePV::kM == WarpShapeQK::kM,
      "Each warp owns a disjoint group of Q rows");
  static_assert(WarpShapePV::kN == kHeadDimV,
      "PV warps must cover all output channels for their Q rows");
  static_assert(kStorageBc >= kBc,
      "QK warps must cover every logical KV column");
  static_assert(WarpShapeQK::kK == 16 && WarpShapePV::kK == 16,
      "Canonical operand iterators advance one m16n8k16 reduction group");
  static_assert(kStorageBc % 16 == 0 && kHeadDim % 16 == 0,
      "TensorOp reduction extents must cover complete m16n8k16 instructions");
  // SM80 TensorOp swizzles are bounded by a 128-byte shared cache line.
  // The Q/K map must issue a complete crosswise group per contiguous step.
  // D=96 uses a 32-element group; D=128 uses a 64-element group.
  static constexpr int kCrosswiseQK = kHeadDim == 96 ? 32 : (kHeadDim < 64 ? kHeadDim : 64);
  static constexpr int kWarpContiguousQK = kCrosswiseQK / 8;
  using LayoutQ = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, kCrosswiseQK>;
  using LayoutK = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, kCrosswiseQK>;
  // Match DefaultMmaCore's SM80 row-major B operand layout directly. Its A
  // operand would impose an unnecessary K<=64 restriction on wide KV tiles.
  static constexpr int kCrosswiseV = kPitchV < 64 ? kPitchV : 64;
  static constexpr int kWarpContiguousV = kPitchV / 8 < 8 ? kPitchV / 8 : 8;
  using LayoutV = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<16, kCrosswiseV>;
  static_assert(kBr % ((32 / kWarpContiguousQK) * kCopyWarps) == 0 &&
                kStorageBc % ((32 / kWarpContiguousQK) * kCopyWarps) == 0 &&
                kStorageBc % ((32 / kWarpContiguousV) * kCopyWarps) == 0,
      "Raked copy map must cover every physical shared-memory row");
  static_assert(sizeof(Element) * (kBr * kHeadDim + kStorageBc * kHeadDim +
                                   kStorageBc * kPitchV) <= 48 * 1024,
      "Q/K/V shared storage must fit the 48 KiB static limit");

  // Assemble stock global/shared access iterators here, as in example 13's
  // DefaultB2bMma. All three operands use vectorized stock copy iterators.
  static constexpr int kCopyThreadsQ = kThreadCount;
  static constexpr int kCopyThreadsK = kThreadCount;
  static constexpr int kCopyThreadsV = kThreadCount;
  static_assert(kCopyThreadsQ == kThreadCount && kCopyThreadsV == kThreadCount,
      "MmaCore copy maps must cover exactly the CTA threads");
  using ThreadMapQ = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<kPitchQ, kBr>, kCopyThreadsQ,
      cutlass::layout::PitchLinearShape<kWarpContiguousQK, 32 / kWarpContiguousQK>, 8>;
  using ThreadMapK = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<kPitchQ, kStorageBc>, kCopyThreadsK,
      cutlass::layout::PitchLinearShape<kWarpContiguousQK, 32 / kWarpContiguousQK>, 8>;
  using ThreadMapV = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<kPitchV, kStorageBc>, kCopyThreadsV,
      cutlass::layout::PitchLinearShape<kWarpContiguousV, 32 / kWarpContiguousV>, 8>;

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
      cutlass::MatrixShape<kBr, kHeadDim>, Element, LayoutQ, 0, ThreadMapQ>;
  using SmemIteratorK = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<kHeadDim, kStorageBc>, Element, LayoutK, 1, ThreadMapK>;
  using SmemIteratorV = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<kStorageBc, kPitchV>, Element, LayoutV, 0, ThreadMapV>;

  using WarpQK = FlashAttnWarpMma<WarpShapeQK, Element, LayoutQ, LayoutK>;
  using WarpPV = FlashAttnWarpMma<WarpShapePV, Element, cutlass::layout::RowMajor, LayoutV>;
  using ProbabilityOutputOp = cutlass::epilogue::thread::LinearCombination<
      Element, 4, float, float, cutlass::epilogue::thread::ScaleType::Nothing>;
  using FragmentIteratorP = cutlass::gemm::warp::MmaTensorOpFragmentIterator<
      cutlass::MatrixShape<WarpShapePV::kM, 16>,
      cutlass::MatrixShape<WarpShapeQK::kM, WarpShapeQK::kN>, kStorageBc,
      float, Element, cutlass::layout::ColumnMajor,
      cutlass::gemm::GemmShape<16, 8, 16>, ProbabilityOutputOp>;
  using Mma = FlashAttnMma<DefaultFlashAttn>;
  using Epilogue = FlashAttnEpilogue<DefaultFlashAttn>;
  using Kernel = FlashAttnKernel<Mma, Epilogue, ArchTag>;
};
