#pragma once

#include "../../02-split-kv/epilogue/flash_attn_epilogue.h"

// Reuse the standard accumulator/output iterators from 02. Saving natural-log
// LSE is optional attention training state, not part of CUTLASS GEMM epilogues.
template <class Policy>
struct FlashAttnEpilogueSplitQ : FlashAttnEpilogue<Policy> {
  using Base = FlashAttnEpilogue<Policy>;
  using FragmentO = typename Base::FragmentO;
  using FragmentIterator = typename Base::FragmentIterator;
  using Fragment = typename Base::Fragment;
  using OutputTileIterator = typename Base::OutputTileIterator;
  using OutputConverter = typename Base::OutputConverter;
  using WarpQK = typename Policy::WarpQK;
  using FragmentL = typename WarpQK::FragmentRow;
  static_assert(FragmentIterator::kIterations == FragmentL::kElements,
      "Each epilogue slice must correspond to one TensorOp row slot");

  // Split-Q retains an unnormalized numerator throughout its KV scan.
  // Its final normalization belongs here; 02 now normalizes O on every tile.
  CUTLASS_DEVICE
  void operator()(FragmentO const& accum, FragmentL const& denominator,
                  OutputTileIterator output) const {
    FragmentIterator fragments(accum);
    OutputConverter convert;
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < FragmentIterator::kIterations; ++r) {
      Fragment result;
      fragments.load(result);
      float inverse_sum = 1.f / denominator[r];
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < Fragment::kElements; ++e)
        result[e] *= inverse_sum;
      output.store(convert(result));
      ++fragments;
      ++output;
    }
  }

  CUTLASS_DEVICE
  void store_logsumexp(FragmentL const& logsumexp, float* output,
                      int valid_rows, int lane) const {
    if (output && lane % 4 == 0) {
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < FragmentL::kElements; ++r) {
        int row = WarpQK::row(r, lane);
        if (row < valid_rows) output[row] = logsumexp[r];
      }
    }
  }
};
