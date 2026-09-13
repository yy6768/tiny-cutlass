#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "../warp/flash_attn_mma.h"
#include "../threadblock/tile_loader.h"
#include "../threadblock/flash_attn_mma.h"
#include "../epilogue/flash_attn_epilogue.h"
#include "flash_attn.h"

// DefaultXxx is a policy factory, matching the assembly boundary in example 13.
template <class ArchTag_, class Element_, class ThreadblockShape_,
          class WarpShapeQK_, class WarpShapePV_, int Stages_,
          class Layout_ = cutlass::layout::RowMajor>
struct DefaultFlashAttn {
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShapeQK = WarpShapeQK_;
  using WarpShapePV = WarpShapePV_;
  using Layout = Layout_;
  static constexpr int kStages = Stages_;
  static constexpr int kBr = ThreadblockShape::kM;
  static constexpr int kBc = ThreadblockShape::kN;
  static constexpr int kHeadDim = ThreadblockShape::kK;
  static constexpr int kWarpsM = kBr / WarpShapeQK::kM;
  static constexpr int kWarpsN = kBc / WarpShapeQK::kN;
  static constexpr int kThreadCount = kWarpsM * kWarpsN * 32;
  static constexpr int kHeadDimV = kHeadDim;
  static constexpr int kPitchQ = ((kHeadDim + 63) / 64) * 64;
  // Stock congruous B iterator supports its pointer permutation at N=32,
  // but add_tile_offset() does not implement that permutation for N=16.
  // Keep all four PV column warps and predicate the padded output channels.
  static constexpr int kPitchV = ((kHeadDimV + 127) / 128) * 128;
  using LayoutQ = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 64>;
  using LayoutK = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, 64>;
  using LayoutP = LayoutQ;
  using LayoutV = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<16, 64>;
  using LoaderQ = FlashAttnTileLoader<Element, LayoutQ, kBr, kPitchQ, kThreadCount>;
  using LoaderK = FlashAttnTileLoader<Element, LayoutK, kBc, kPitchQ, kThreadCount, true>;
  using LoaderV = FlashAttnTileLoader<Element, LayoutV, kBc, kPitchV, kThreadCount>;
  using WarpQK = FlashAttnWarpMma<WarpShapeQK, Element, LayoutQ, LayoutK>;
  using WarpPV = FlashAttnWarpMma<WarpShapePV, Element, LayoutP, LayoutV>;
  using Mma = FlashAttnMma<DefaultFlashAttn>;
  using Epilogue = FlashAttnEpilogue<DefaultFlashAttn>;
  using Kernel = FlashAttnKernel<Mma, Epilogue, ArchTag>;
};
