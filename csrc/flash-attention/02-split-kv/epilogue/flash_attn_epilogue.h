#pragma once

#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_ref.h"

template <class Policy>
struct FlashAttnEpilogue {
  using Element = typename Policy::Element;
  using Warp = typename Policy::WarpPV;
  using FragmentO = typename Warp::Fragment;
  using FragmentL = typename Warp::FragmentRow;
  using TensorRefO = cutlass::TensorRef<Element, cutlass::layout::RowMajor>;

  // No shared-memory scratch: this is not an artificial union with mainloop storage.
  CUTLASS_DEVICE
  void operator()(FragmentO const& accum, FragmentL const& denominator,
                  TensorRefO output, int valid_rows, int warp, int lane) const {
    int warp_m = warp % Policy::kWarpsM;
    int warp_n = warp / Policy::kWarpsM;
    cutlass::NumericConverter<Element, float> convert;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentO::kElements; ++i) {
      int r = Warp::row_slot(i);
      int row = warp_m * Policy::WarpShapePV::kM + Warp::row(r, lane);
      int col = warp_n * Policy::WarpShapePV::kN + Warp::column(i, lane);
      if (row < valid_rows && col < Policy::kHeadDimV)
        output.at({row, col}) = convert(accum[i] / denominator[r]);
    }
  }
};
