#pragma once

#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/functional.h"

// dK/dV have one CTA owner. Stock fragment and output iterators handle the
// TensorOp accumulator layout and logical-channel/tail predicates.
template <class Warp_, class Element_>
struct FlashAttnBackwardEpilogue {
  using Warp = Warp_;
  using Element = Element_;
  using Mma = typename Warp::Mma;
  using Accumulator = typename Warp::Fragment;
  using FragmentIterator = cutlass::epilogue::warp::FragmentIteratorTensorOp<
      typename Mma::Shape, typename Mma::InstructionShape, float,
      typename Mma::Policy::Operator::FragmentC, cutlass::layout::RowMajor>;
  using Fragment = typename FragmentIterator::Fragment;
  using OutputTileIterator = cutlass::epilogue::warp::TileIteratorTensorOpCanonical<
      typename Mma::Shape, typename Mma::InstructionShape, Element,
      cutlass::layout::RowMajor>;
  using Converter = cutlass::NumericArrayConverter<Element, float, Fragment::kElements>;

  CUTLASS_DEVICE
  void operator()(Accumulator const& accum, OutputTileIterator output) const {
    FragmentIterator fragments(accum);
    Converter convert;
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < FragmentIterator::kIterations; ++r) {
      Fragment value;
      fragments.load(value);
      output.store(convert(value));
      ++fragments;
      ++output;
    }
  }
};

// Attention-specific dQ accumulation across KV CTAs (FA2 Algorithm 2).
// Each warp owns complete output elements; no reduction is split across warps.
// The stock canonical output iterator has no atomic store operation, so this
// epilogue uses the already-audited m16n8 accumulator coordinate mapping.
template <class Warp_>
struct FlashAttnBackwardAtomicEpilogue {
  using Warp = Warp_;
  using Accumulator = typename Warp::Fragment;

  CUTLASS_DEVICE
  void operator()(Accumulator const& accum, float* output, int64_t stride,
                  int valid_rows, int valid_columns, int lane) const {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Accumulator::kElements; ++i) {
      int row = Warp::row(Warp::row_slot(i), lane);
      int column = Warp::column(i, lane);
      if (row < valid_rows && column < valid_columns)
        cutlass::atomic_add<float>{}(output + int64_t(row) * stride + column, accum[i]);
    }
  }
};
