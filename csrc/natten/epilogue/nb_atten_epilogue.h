#pragma once

#include <cutlass/epilogue/warp/fragment_iterator_tensor_op.h>
#include <cutlass/epilogue/warp/tile_iterator_tensor_op.h>
#include <cutlass/numeric_conversion.h>

namespace tiny_cutlass { namespace natten {

// The online mainloop has already normalized O. This layer converts FP32 to
// the output type using CUTLASS iterators and stores the optional row LSE.
template <class Policy>
struct NeighborhoodEpilogue {
  using Element = typename Policy::Element;
  using Warp = typename Policy::WarpPV;
  using WarpQK = typename Policy::WarpQK;
  using Mma = typename Warp::Mma;
  using FragmentO = typename Warp::Fragment;
  using FragmentRow = typename WarpQK::FragmentRow;
  using FragmentIterator = cutlass::epilogue::warp::FragmentIteratorTensorOp<
      typename Mma::Shape, typename Mma::InstructionShape, float,
      typename Mma::Policy::Operator::FragmentC, cutlass::layout::RowMajor>;
  using Fragment = typename FragmentIterator::Fragment;
  using OutputTileIterator = cutlass::epilogue::warp::TileIteratorTensorOpCanonical<
      typename Mma::Shape, typename Mma::InstructionShape, Element,
      cutlass::layout::RowMajor>;
  using Converter = cutlass::NumericArrayConverter<Element, float, Fragment::kElements>;

  CUTLASS_DEVICE void operator()(FragmentO const& accum, OutputTileIterator output,
      FragmentRow const& maximum, FragmentRow const& denominator,
      float* lse, int query_base, int length, int lane) const {
    FragmentIterator fragments(accum);
    Converter convert;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentIterator::kIterations; ++i) {
      Fragment part;
      fragments.load(part);
      output.store(convert(part));
      ++fragments;
      ++output;
    }
    if (lse && lane % 4 == 0) {
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < FragmentRow::kElements; ++r) {
        int row = query_base + WarpQK::row(r, lane);
        if (row < length) lse[row] = maximum[r] + logf(denominator[r]);
      }
    }
  }
};

}} // namespace tiny_cutlass::natten
