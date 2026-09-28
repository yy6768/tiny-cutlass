#pragma once

#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/numeric_conversion.h"

// Every (KV,Q) iteration normalizes O in Mma. Intermediate rounds store FP32
// through the stock MMA accumulator iterator; the final round converts to FP16.
template <class Policy>
struct FlashAttnEpilogue {
  using Element = typename Policy::Element;
  using Warp = typename Policy::WarpPV;
  using Mma = typename Warp::Mma;
  using FragmentO = typename Warp::Fragment;
  using StateTileIterator = typename Warp::IteratorC;
  using FragmentIterator = cutlass::epilogue::warp::FragmentIteratorTensorOp<
      typename Mma::Shape, typename Mma::InstructionShape, float,
      typename Mma::Policy::Operator::FragmentC, cutlass::layout::RowMajor>;
  using Fragment = typename FragmentIterator::Fragment;
  using OutputTileIterator = cutlass::epilogue::warp::TileIteratorTensorOpCanonical<
      typename Mma::Shape, typename Mma::InstructionShape, Element, cutlass::layout::RowMajor>;
  using OutputConverter = cutlass::NumericArrayConverter<Element, float, Fragment::kElements>;
  static constexpr int kWarpColumns = Policy::WarpShapePV::kN;

  CUTLASS_DEVICE
  void store_intermediate(FragmentO const& accum, StateTileIterator state) const {
    state.store(accum);
  }

  CUTLASS_DEVICE
  void operator()(FragmentO const& accum, OutputTileIterator output) const {
    FragmentIterator fragments(accum);
    OutputConverter convert;
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < FragmentIterator::kIterations; ++r) {
      Fragment result;
      fragments.load(result);
      output.store(convert(result));
      ++fragments;
      ++output;
    }
  }
};
