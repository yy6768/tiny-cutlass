#pragma once

#include <math_constants.h>
#include "natten/neighborhood.h"

namespace tiny_cutlass { namespace natten {

// Attention-specific row masking and online normalization. CUTLASS GEMM
// supplies the score fragments but does not define neighborhood softmax.
template <class Warp_>
struct NeighborhoodSoftmax {
  using Warp = Warp_;
  using FragmentScore = typename Warp::Fragment;
  using FragmentRow = typename Warp::FragmentRow;

  CUTLASS_DEVICE static float max_row(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1));
    return fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2));
  }
  CUTLASS_DEVICE static float sum_row(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 1);
    return x + __shfl_xor_sync(0xffffffff, x, 2);
  }

  CUTLASS_DEVICE void operator()(FragmentScore& score, FragmentRow& maximum,
      FragmentRow& denominator, FragmentRow& old_weight, FragmentRow& beta,
      int query_base, int key_base, int length, int window, int lane,
      float scale) const {
    FragmentRow tile_max, tile_sum;
    tile_sum.clear();
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < FragmentRow::kElements; ++r) tile_max[r] = -CUDART_INF_F;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentScore::kElements; ++i) {
      int r = Warp::row_slot(i);
      int query = query_base + Warp::row(r, lane);
      int key = key_base + Warp::column(i, lane);
      int begin = query < length ? neighborhood_start(query, length, window) : 0;
      bool valid = query < length && key < length && key >= begin && key < begin + window;
      score[i] = valid ? score[i] * scale : -CUDART_INF_F;
      tile_max[r] = fmaxf(tile_max[r], score[i]);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < FragmentRow::kElements; ++r) tile_max[r] = max_row(tile_max[r]);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentScore::kElements; ++i) {
      int r = Warp::row_slot(i);
      float probability = isfinite(tile_max[r]) ? __expf(score[i] - tile_max[r]) : 0.0f;
      score[i] = probability;
      tile_sum[r] += probability;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < FragmentRow::kElements; ++r) {
      tile_sum[r] = sum_row(tile_sum[r]);
      float next = fmaxf(maximum[r], tile_max[r]);
      float alpha = isfinite(maximum[r]) ? __expf(maximum[r] - next) : 0.0f;
      beta[r] = isfinite(tile_max[r]) ? __expf(tile_max[r] - next) : 0.0f;
      old_weight[r] = alpha * denominator[r];
      denominator[r] = old_weight[r] + beta[r] * tile_sum[r];
      maximum[r] = next;
    }
  }
};

}} // namespace tiny_cutlass::natten
