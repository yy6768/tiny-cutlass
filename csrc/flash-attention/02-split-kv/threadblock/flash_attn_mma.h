#pragma once

#include <cmath>
#include <math_constants.h>
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/array.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/numeric_conversion.h"

// The policy supplies both stock warp MMAs and the global/shared iterators.
// This class owns attention-specific dependencies between the two GEMMs.
template <class Policy>
class FlashAttnMma {
 public:
  using Element = typename Policy::Element;
  using WarpQK = typename Policy::WarpQK;
  using WarpPV = typename Policy::WarpPV;
  using FragmentS = typename WarpQK::Fragment;
  using FragmentO = typename WarpPV::Fragment;
  using FragmentL = typename WarpQK::FragmentRow;
  using IteratorQ = typename Policy::LoaderQ::Iterator;
  using IteratorK = typename Policy::LoaderK::Iterator;
  using IteratorV = typename Policy::LoaderV::Iterator;
  static constexpr int kBr = Policy::kBr;
  static constexpr int kBc = Policy::kBc;
  static constexpr int kHeadDim = Policy::kHeadDim;
  static constexpr int kHeadDimV = Policy::kHeadDimV;
  static constexpr int kThreadCount = Policy::kThreadCount;

  struct SharedStorage {
    alignas(128) Element q[kBr * Policy::kPitchQ];
    alignas(128) Element k[Policy::kStages][kBc * Policy::kPitchQ];
    alignas(128) Element v[kBc * Policy::kPitchV];
    alignas(128) Element p[kBr * kBc];
    // Each KV warp contributes one statistic for every row that it owns.
    alignas(128) float row_max[Policy::kWarpsN][kBr];
    alignas(128) float row_sum[Policy::kWarpsN][kBr];
  };

 private:
  SharedStorage& storage_;
  int thread_, lane_, warp_m_, warp_n_;

  CUTLASS_DEVICE
  static float reduce_max(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1));
    return fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2));
  }
  CUTLASS_DEVICE
  static float reduce_sum(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 1);
    return x + __shfl_xor_sync(0xffffffff, x, 2);
  }

 public:
  CUTLASS_DEVICE
  FlashAttnMma(SharedStorage& storage, int thread, int warp, int lane)
      : storage_(storage), thread_(thread), lane_(lane),
        warp_m_(warp % Policy::kWarpsM), warp_n_(warp / Policy::kWarpsM) {}

  CUTLASS_DEVICE
  void operator()(int kv_tile_iterations, int first_kv_rows,
                  FragmentO& output, FragmentL& denominator,
                  IteratorQ iterator_q, IteratorK iterator_k, IteratorV iterator_v,
                  float scale) {
    using LayoutQ = typename Policy::LayoutQ;
    using LayoutK = typename Policy::LayoutK;
    using LayoutP = typename Policy::LayoutP;
    using LayoutV = typename Policy::LayoutV;
    FragmentL running_max;
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < FragmentL::kElements; ++r) running_max[r] = -CUDART_INF_F;

    // Q is persistent. Only K is staged; V keeps one buffer like LeetCUDA.
    Policy::LoaderQ::copy(iterator_q, storage_.q, thread_);
    cutlass::arch::cp_async_fence();

    if (Policy::kStages == 2) {
      Policy::LoaderK::copy(iterator_k, storage_.k[0], thread_);
      iterator_k.add_tile_offset({0, 1});
      cutlass::arch::cp_async_fence();
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();
    }

    int read_stage = 0;
    for (int tile = 0; tile < kv_tile_iterations; ++tile) {
      // Stock access iterators visit the residue first, then full KV tiles.
      // QK's padded score columns must use the same extent as the copies.
      int valid_kv = tile == 0 ? first_kv_rows : kBc;
      if (Policy::kStages == 1) {
        Policy::LoaderK::copy(iterator_k, storage_.k[0], thread_);
        iterator_k.add_tile_offset({0, 1});
        cutlass::arch::cp_async_fence();
      }
      Policy::LoaderV::copy(iterator_v, storage_.v, thread_);
      iterator_v.add_tile_offset({0, 1});
      cutlass::arch::cp_async_fence();

      bool has_next = tile + 1 < kv_tile_iterations;
      if (Policy::kStages == 2 && has_next) {
        int write_stage = read_stage ^ 1;
        Policy::LoaderK::copy(iterator_k, storage_.k[write_stage], thread_);
        iterator_k.add_tile_offset({0, 1});
        cutlass::arch::cp_async_fence();
      }
      if (Policy::kStages == 1) {
        cutlass::arch::cp_async_wait<1>();
        __syncthreads(); // Q and current K are visible; V may still be in flight.
      }

      typename WarpQK::IteratorA qa({storage_.q, LayoutQ(Policy::kPitchQ)}, lane_);
      typename WarpQK::IteratorB kb({storage_.k[read_stage], LayoutK(Policy::kPitchQ)}, lane_);
      qa.add_tile_offset({warp_m_, 0});
      kb.add_tile_offset({0, warp_n_});
      FragmentS score;
      score.clear();
      WarpQK{}(qa, kb, kHeadDim / 16, score);

      FragmentL maximum, sum, alpha;
      sum.clear();
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < FragmentL::kElements; ++r) maximum[r] = -CUDART_INF_F;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentS::kElements; ++i) {
        int col = warp_n_ * Policy::WarpShapeQK::kN + WarpQK::column(i, lane_);
        score[i] = col < valid_kv ? score[i] * scale : -CUDART_INF_F;
        int r = WarpQK::row_slot(i);
        maximum[r] = fmaxf(maximum[r], score[i]);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < FragmentL::kElements; ++r) {
        maximum[r] = reduce_max(maximum[r]);
        int row = warp_m_ * Policy::WarpShapeQK::kM + WarpQK::row(r, lane_);
        if (lane_ % 4 == 0) storage_.row_max[warp_n_][row] = maximum[r];
      }
      __syncthreads(); // QK warps have only 1/4 of each row: exchange max.

      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < FragmentL::kElements; ++r) {
        int row = warp_m_ * Policy::WarpShapeQK::kM + WarpQK::row(r, lane_);
        float m = running_max[r];
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Policy::kWarpsN; ++n) m = fmaxf(m, storage_.row_max[n][row]);
        alpha[r] = __expf(running_max[r] - m);
        running_max[r] = m;
      }
      LayoutP layout_p(kBc);
      cutlass::NumericConverter<Element, float> convert;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentS::kElements; ++i) {
        int r = WarpQK::row_slot(i);
        int row = warp_m_ * Policy::WarpShapeQK::kM + WarpQK::row(r, lane_);
        int col = warp_n_ * Policy::WarpShapeQK::kN + WarpQK::column(i, lane_);
        float p = __expf(score[i] - running_max[r]);
        sum[r] += p;
        storage_.p[layout_p({row, col})] = convert(p);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < FragmentL::kElements; ++r) {
        sum[r] = reduce_sum(sum[r]);
        int row = warp_m_ * Policy::WarpShapeQK::kM + WarpQK::row(r, lane_);
        if (lane_ % 4 == 0) storage_.row_sum[warp_n_][row] = sum[r];
      }
      __syncthreads(); // P is readable by every PV warp; exchange row sum.

      if (Policy::kStages == 2 && has_next)
        cutlass::arch::cp_async_wait<1>(); // Drain V, leave next K outstanding.
      else
        cutlass::arch::cp_async_wait<0>();
      __syncthreads(); // cp.async wait is per thread; make V visible CTA-wide.

      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < FragmentL::kElements; ++r) {
        int row = warp_m_ * Policy::WarpShapeQK::kM + WarpQK::row(r, lane_);
        float l = 0.f;
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Policy::kWarpsN; ++n) l += storage_.row_sum[n][row];
        denominator[r] = alpha[r] * denominator[r] + l;
      }
      // Warp N now indexes output channels. It no longer partitions softmax K.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentO::kElements; ++i) output[i] *= alpha[WarpPV::row_slot(i)];
      typename WarpPV::IteratorA pa({storage_.p, LayoutP(kBc)}, lane_);
      typename WarpPV::IteratorB vb({storage_.v, LayoutV(Policy::kPitchV)}, lane_);
      pa.add_tile_offset({warp_m_, 0});
      vb.add_tile_offset({0, warp_n_});
      WarpPV{}(pa, vb, kBc / 16, output);
      __syncthreads(); // Release P/V and reduction scratch before the next tile.

      if (Policy::kStages == 2) {
        cutlass::arch::cp_async_wait<0>();
        __syncthreads(); // The prefetched K stage is now visible to every warp.
        if (has_next) read_stage ^= 1;
      }
    }
  }
};
