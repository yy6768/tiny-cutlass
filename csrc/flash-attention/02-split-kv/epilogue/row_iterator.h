#pragma once

#include <math_constants.h>

// Attention has one statistic per Q row, shared by all column warps.
template <class WarpMma>
class FlashAttnRowIterator {
 public:
  using Fragment = typename WarpMma::FragmentRow;

  CUTLASS_DEVICE
  FlashAttnRowIterator(float* maximum, float* sum, int valid_rows, int warp, int lane)
      : maximum_(maximum), sum_(sum), valid_rows_(valid_rows), lane_(lane),
        writer_(warp == 0 && lane % 4 == 0) {}

  CUTLASS_DEVICE
  void load(Fragment& maximum, Fragment& sum) const {
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < Fragment::kElements; ++r) {
      int row = WarpMma::row(r, lane_);
      maximum[r] = row < valid_rows_ ? maximum_[row] : -CUDART_INF_F;
      sum[r] = row < valid_rows_ ? sum_[row] : 0.f;
    }
  }

  CUTLASS_DEVICE
  void store(Fragment const& maximum, Fragment const& sum) const {
    if (!writer_) return;
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < Fragment::kElements; ++r) {
      int row = WarpMma::row(r, lane_);
      if (row < valid_rows_) {
        maximum_[row] = maximum[r];
        sum_[row] = sum[r];
      }
    }
  }

 private:
  float* maximum_;
  float* sum_;
  int valid_rows_, lane_;
  bool writer_;
};
