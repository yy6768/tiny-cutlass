#pragma once

#include <math_constants.h>
#include <cutlass/arch/memory_sm80.h>
#include <cutlass/array.h>
#include "../thread/nb_atten_softmax.h"

namespace tiny_cutlass { namespace natten {

// Owns the complete Q/K/V mainloop, CUTLASS iterator advancement, shared
// storage reuse and CTA barriers. Copying follows example 13's
// B2bMmaMultistage::copy_tiles_and_advance_0 responsibility boundary.
template <class Policy>
class NeighborhoodMma {
 public:
  using Element = typename Policy::Element;
  using LayoutQ = typename Policy::LayoutQ;
  using LayoutK = typename Policy::LayoutK;
  using LayoutV = typename Policy::LayoutV;
  using WarpQK = typename Policy::WarpQK;
  using WarpPV = typename Policy::WarpPV;
  using FragmentS = typename WarpQK::Fragment;
  using FragmentO = typename WarpPV::Fragment;
  using FragmentRow = typename WarpQK::FragmentRow;
  using IteratorQ = typename Policy::IteratorQ;
  using IteratorK = typename Policy::IteratorK;
  using IteratorV = typename Policy::IteratorV;
  using SmemIteratorQ = typename Policy::SmemIteratorQ;
  using SmemIteratorK = typename Policy::SmemIteratorK;
  using SmemIteratorV = typename Policy::SmemIteratorV;
  using ThreadMap = typename IteratorQ::ThreadMap;
  using AccessType = typename IteratorQ::AccessType;
  using WarpIteratorQ = typename WarpQK::IteratorA;
  using WarpIteratorK = typename WarpQK::IteratorB;
  using WarpIteratorV = typename WarpPV::IteratorB;
  using ProbabilityIterator = typename Policy::ProbabilityIterator;
  using ProbabilityOutputOp = typename Policy::ProbabilityOutputOp;
  using FragmentP = typename WarpPV::FragmentA;
  using FragmentV = typename WarpPV::FragmentB;
  using OperandP = typename WarpPV::OperandA;
  using OperandV = typename WarpPV::OperandB;
  using Epilogue = typename Policy::Epilogue;
  using OutputTileIterator = typename Epilogue::OutputTileIterator;
  using Softmax = NeighborhoodSoftmax<WarpQK>;
  static constexpr int kBr = Policy::kBr;
  static constexpr int kBc = Policy::kBc;
  static constexpr int kPhysicalDim = Policy::kPhysicalDim;
  static constexpr int kThreads = Policy::kThreads;

  struct SharedStorage {
    alignas(16) Element q[kBr * kPhysicalDim];
    alignas(16) Element k[kBc * kPhysicalDim];
    alignas(16) Element v[kBc * kPhysicalDim];
  };

 private:
  SharedStorage& storage_;
  int thread_, warp_, lane_;
  SmemIteratorQ smem_q_;
  SmemIteratorK smem_k_;
  SmemIteratorV smem_v_;

  CUTLASS_DEVICE void copy_query(IteratorQ source, int rows) {
    source.set_iteration_index(0);
    smem_q_.set_iteration_index(0);
    int first = ThreadMap::initial_offset(thread_).strided();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ThreadMap::Iterations::kCount; ++i) {
      int row = first + (i / ThreadMap::Iterations::kContiguous) * ThreadMap::Delta::kStrided;
      cutlass::arch::cp_async_zfill<sizeof(AccessType), cutlass::arch::CacheOperation::Global>(
          smem_q_.get(), source.get(), source.valid() && row < rows);
      ++source; ++smem_q_;
    }
  }

  CUTLASS_DEVICE void copy_key_value(IteratorK key, IteratorV value, int rows) {
    key.set_iteration_index(0); value.set_iteration_index(0);
    smem_k_.set_iteration_index(0); smem_v_.set_iteration_index(0);
    int first = ThreadMap::initial_offset(thread_).strided();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ThreadMap::Iterations::kCount; ++i) {
      int row = first + (i / ThreadMap::Iterations::kContiguous) * ThreadMap::Delta::kStrided;
      cutlass::arch::cp_async_zfill<sizeof(AccessType), cutlass::arch::CacheOperation::Global>(
          smem_k_.get(), key.get(), key.valid() && row < rows);
      cutlass::arch::cp_async_zfill<sizeof(AccessType), cutlass::arch::CacheOperation::Global>(
          smem_v_.get(), value.get(), value.valid() && row < rows);
      ++key; ++value; ++smem_k_; ++smem_v_;
    }
  }

 public:
  CUTLASS_DEVICE NeighborhoodMma(SharedStorage& storage, int thread, int warp, int lane)
      : storage_(storage), thread_(thread), warp_(warp), lane_(lane),
        smem_q_({storage.q, LayoutQ(kPhysicalDim)}, thread),
        smem_k_({storage.k, LayoutK(kPhysicalDim)}, thread),
        smem_v_({storage.v, LayoutV(kPhysicalDim)}, thread) {}

  CUTLASS_DEVICE void operator()(int query_base, int key_begin,
      int key_tile_count, int last_key_rows,
      int length, int window, float scale, IteratorQ query, IteratorK key,
      IteratorV value, OutputTileIterator output_tile, float* lse,
      Epilogue const& epilogue) {
    int valid_q = length - query_base < kBr ? length - query_base : kBr;
    copy_query(query, valid_q);
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    FragmentO output; output.clear();
    FragmentRow maximum, denominator;
    denominator.clear();
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < FragmentRow::kElements; ++r) maximum[r] = -CUDART_INF_F;
    ProbabilityOutputOp convert(typename ProbabilityOutputOp::Params{});
    Softmax softmax;

    for (int j = 0; j < key_tile_count; ++j) {
      int key_base = key_begin + j * kBc;
      int valid_kv = j + 1 == key_tile_count ? last_key_rows : kBc;
      copy_key_value(key, value, valid_kv);
      cutlass::arch::cp_async_fence();
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();

      WarpIteratorQ qa({storage_.q, LayoutQ(kPhysicalDim)}, lane_);
      WarpIteratorK kb({storage_.k, LayoutK(kPhysicalDim)}, lane_);
      qa.add_tile_offset({warp_, 0});
      FragmentS score; score.clear();
      WarpQK qk;
      qk(qa, kb, kPhysicalDim / 16, score);

      FragmentRow old_weight, beta;
      softmax(score, maximum, denominator, old_weight, beta,
              query_base + warp_ * WarpQK::Shape::kM, key_base,
              length, window, lane_, scale);

      ProbabilityIterator pa(score);
      WarpIteratorV vb({storage_.v, LayoutV(kPhysicalDim)}, lane_);
      FragmentO partial; partial.clear();
      typename WarpPV::Mma pv;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kBc / 16; ++i) {
        FragmentP p; FragmentV v;
        OperandP op_p; OperandV op_v;
        pa.load(p, convert); vb.load(v);
        ++pa; ++vb;
        pv.transform(op_p, op_v, p, v);
        pv(partial, op_p, op_v, partial);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentO::kElements; ++i) {
        int r = WarpPV::row_slot(i);
        output[i] = denominator[r] > 0.0f
            ? (old_weight[r] * output[i] + beta[r] * partial[i]) / denominator[r]
            : 0.0f;
      }
      __syncthreads(); // all warps release K/V before the next tile copy
      if (j + 1 < key_tile_count) {
        key.add_tile_offset({0, 1});
        value.add_tile_offset({0, 1});
      }
    }
    epilogue(output, output_tile, maximum, denominator, lse,
             query_base + warp_ * WarpQK::Shape::kM, length, lane_);
  }
};

}} // namespace tiny_cutlass::natten
