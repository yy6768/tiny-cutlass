#pragma once

#include <math_constants.h>
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/numeric_conversion.h"

// Custom attention backward mainloop, following example 13's ownership:
// this class owns operand movement, shared lifetimes, iterator progression,
// all five TensorOp products and attention-specific pointwise operations.
template <class Policy>
class FlashAttnBackwardMma {
 public:
  using Element = typename Policy::Element;
  using CanonicalIterator = typename Policy::CanonicalIterator;
  using CanonicalSmemIterator = typename Policy::CanonicalSmemIterator;
  using CanonicalThreadMap = typename Policy::CanonicalThreadMap;
  using CanonicalAccess = typename CanonicalIterator::AccessType;
  using SwizzleIterator = typename Policy::SwizzleIterator;
  using SwizzleSmemIterator = typename Policy::SwizzleSmemIterator;
  using SwizzleThreadMap = typename Policy::SwizzleThreadMap;
  using SwizzleAccess = typename SwizzleIterator::AccessType;
  using SwizzleLayout = typename Policy::SwizzleLayout;
  using ScoreLayout = typename Policy::ScoreLayout;
  using TransposedScoreLayout = typename Policy::TransposedScoreLayout;
  using WarpScore = typename Policy::WarpScore;
  using WarpGrad = typename Policy::WarpGrad;
  using WarpDq = typename Policy::WarpDq;
  using WarpShapeDq = typename Policy::WarpShapeDq;
  using ScoreIteratorA = typename WarpScore::IteratorA;
  using ScoreIteratorB = typename WarpScore::IteratorB;
  using GradIteratorB = typename WarpGrad::IteratorB;
  using DqIteratorA = typename WarpDq::IteratorA;
  using DqIteratorB = typename WarpDq::IteratorB;
  using FragmentScore = typename WarpScore::Fragment;
  using FragmentGrad = typename WarpGrad::Fragment;
  using FragmentDq = typename WarpDq::Fragment;
  using GradMma = typename WarpGrad::Mma;
  using GradFragmentA = typename GradMma::FragmentA;
  using GradFragmentB = typename GradMma::FragmentB;
  using GradTransformedA = typename GradMma::TransformedFragmentA;
  using GradTransformedB = typename GradMma::TransformedFragmentB;
  using FragmentIterator = typename Policy::FragmentIterator;
  using ProbabilityOutputOp = typename Policy::ProbabilityOutputOp;
  using ProbabilityOutputParams = typename ProbabilityOutputOp::Params;
  using AtomicEpilogue = typename Policy::AtomicEpilogue;
  using Converter = cutlass::NumericConverter<Element, float>;
  static constexpr int kBr = Policy::kBr;
  static constexpr int kBc = Policy::kBc;
  static constexpr int kHeadDim = Policy::kHeadDim;
  static constexpr int kStorageDim = Policy::kStorageDim;
  static constexpr int kThreadCount = Policy::kThreadCount;
  static constexpr int kCanonicalCopyThreads = Policy::kCanonicalCopyThreads;

  struct SharedStorage {
    alignas(16) Element key[kBc * kHeadDim];
    alignas(16) Element value[kBc * kHeadDim];
    alignas(16) Element key_swizzle[kBc * kStorageDim];
    union QueryStorage {
      alignas(16) Element canonical[kBr * kHeadDim];
      alignas(16) Element swizzle[kBr * kStorageDim];
    } query;
    union GradOutputStorage {
      alignas(16) Element canonical[kBr * kHeadDim];
      alignas(16) Element swizzle[kBr * kStorageDim];
    } grad_output;
    alignas(16) Element grad_score[kBc * kBr];
    alignas(16) float logsumexp[kBr];
    alignas(16) float delta[kBr];
  };

 private:
  SharedStorage& storage_;
  int thread_, warp_, lane_;

  CUTLASS_DEVICE
  void copy_canonical(CanonicalIterator iterator, Element* destination, int valid_rows) {
    if (thread_ >= kCanonicalCopyThreads) return;
    CanonicalSmemIterator smem(
        {destination, cutlass::layout::PitchLinear(kHeadDim)}, thread_);
    iterator.set_iteration_index(0);
    int first_row = CanonicalThreadMap::initial_offset(thread_).strided();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < CanonicalThreadMap::Iterations::kCount; ++i) {
      int row = first_row + (i / CanonicalThreadMap::Iterations::kContiguous) *
          CanonicalThreadMap::Delta::kStrided;
      cutlass::arch::cp_async_zfill<sizeof(CanonicalAccess), cutlass::arch::CacheOperation::Global>(
          smem.get(), iterator.get(), iterator.valid() && row < valid_rows);
      ++iterator;
      ++smem;
    }
  }

  CUTLASS_DEVICE
  void copy_swizzle(SwizzleIterator iterator, Element* destination, int valid_rows) {
    SwizzleSmemIterator smem({destination, SwizzleLayout(kStorageDim)}, thread_);
    iterator.set_iteration_index(0);
    int first_row = SwizzleThreadMap::initial_offset(thread_).strided();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < SwizzleThreadMap::Iterations::kCount; ++i) {
      int row = first_row + (i / SwizzleThreadMap::Iterations::kContiguous) *
          SwizzleThreadMap::Delta::kStrided;
      cutlass::arch::cp_async_zfill<sizeof(SwizzleAccess), cutlass::arch::CacheOperation::Global>(
          smem.get(), iterator.get(), iterator.valid() && row < valid_rows);
      ++iterator;
      ++smem;
    }
  }

  CUTLASS_DEVICE
  void accumulate(FragmentScore const& source, Element* operand_b,
                  FragmentGrad& accum) {
    FragmentIterator iterator_a(source);
    GradIteratorB iterator_b({operand_b, SwizzleLayout(kStorageDim)}, lane_);
    ProbabilityOutputOp convert(ProbabilityOutputParams{});
    GradMma mma;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kBr / 16; ++k) {
      GradFragmentA a;
      GradFragmentB b;
      GradTransformedA transformed_a;
      GradTransformedB transformed_b;
      iterator_a.load(a, convert);
      iterator_b.load(b);
      ++iterator_a;
      ++iterator_b;
      mma.transform(transformed_a, transformed_b, a, b);
      mma(accum, transformed_a, transformed_b, accum);
    }
  }

 public:
  CUTLASS_DEVICE
  FlashAttnBackwardMma(SharedStorage& storage, int thread, int warp, int lane)
      : storage_(storage), thread_(thread), warp_(warp), lane_(lane) {}

  CUTLASS_DEVICE
  void operator()(int query_iterations, int last_query_rows, int valid_keys,
                  FragmentGrad& grad_key, FragmentGrad& grad_value,
                  CanonicalIterator query, CanonicalIterator key,
                  CanonicalIterator value, CanonicalIterator grad_output,
                  SwizzleIterator query_swizzle, SwizzleIterator key_swizzle,
                  SwizzleIterator grad_output_swizzle, float scale,
                  float const* logsumexp, float const* delta,
                  float* grad_query, int64_t query_stride) {
    copy_canonical(key, storage_.key, valid_keys);
    copy_canonical(value, storage_.value, valid_keys);
    copy_swizzle(key_swizzle, storage_.key_swizzle, valid_keys);
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    CUTLASS_PRAGMA_UNROLL
    for (int tile = 0; tile < query_iterations; ++tile) {
      int valid_queries = tile + 1 == query_iterations ? last_query_rows : kBr;
      int query_start = tile * kBr;
      copy_canonical(query, storage_.query.canonical, valid_queries);
      copy_canonical(grad_output, storage_.grad_output.canonical, valid_queries);
      if (thread_ < kBr) {
        storage_.logsumexp[thread_] = thread_ < valid_queries ?
            logsumexp[query_start + thread_] : CUDART_INF_F;
        storage_.delta[thread_] = thread_ < valid_queries ?
            delta[query_start + thread_] : 0.f;
      }
      cutlass::arch::cp_async_fence();
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();

      // S^T=KQ^T and dP^T=VdO^T: each warp owns sixteen complete KV rows.
      ScoreIteratorA key_a({storage_.key, cutlass::layout::RowMajor(kHeadDim)}, lane_);
      ScoreIteratorB query_b(
          {storage_.query.canonical, cutlass::layout::ColumnMajor(kHeadDim)}, lane_);
      key_a.add_tile_offset({warp_, 0});
      FragmentScore probability;
      probability.clear();
      WarpScore{}(key_a, query_b, kHeadDim / 16, probability);

      ScoreIteratorA value_a({storage_.value, cutlass::layout::RowMajor(kHeadDim)}, lane_);
      ScoreIteratorB grad_output_b(
          {storage_.grad_output.canonical, cutlass::layout::ColumnMajor(kHeadDim)}, lane_);
      value_a.add_tile_offset({warp_, 0});
      FragmentScore grad_probability;
      grad_probability.clear();
      WarpScore{}(value_a, grad_output_b, kHeadDim / 16, grad_probability);

      // Probability reconstruction is attention-specific, not a GEMM epilogue.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentScore::kElements; ++i) {
        int q = WarpScore::column(i, lane_);
        int k = warp_ * 16 + WarpScore::row(WarpScore::row_slot(i), lane_);
        probability[i] = q < valid_queries && k < valid_keys ?
            __expf(probability[i] * scale - storage_.logsumexp[q]) : 0.f;
      }

      // All canonical Q/dO readers finish before the same storage is reused.
      __syncthreads();
      copy_swizzle(query_swizzle, storage_.query.swizzle, valid_queries);
      copy_swizzle(grad_output_swizzle, storage_.grad_output.swizzle, valid_queries);
      cutlass::arch::cp_async_fence();
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();

      // dV=P^T dO; register P stays FP32 until dS is formed below.
      accumulate(probability, storage_.grad_output.swizzle, grad_value);
      Converter convert;
      ScoreLayout score_layout(kBr);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentScore::kElements; ++i) {
        int q = WarpScore::column(i, lane_);
        int k = warp_ * 16 + WarpScore::row(WarpScore::row_slot(i), lane_);
        float ds = probability[i] * (grad_probability[i] - storage_.delta[q]) * scale;
        probability[i] = ds;
        storage_.grad_score[score_layout({k, q})] = convert(ds);
      }
      // dK=dS^T Q uses the same official register-to-A fragment iterator.
      accumulate(probability, storage_.query.swizzle, grad_key);
      __syncthreads();

      // The same swizzled addresses represent dS^T[k,q] and dS[q,k].
      // M32/N>=32 offsets are required by the stock congruous warp iterators.
      int warp_q = warp_ / 2;
      int warp_n = warp_ % 2;
      DqIteratorA ds_a({storage_.grad_score, TransposedScoreLayout(kBr)}, lane_);
      DqIteratorB key_b({storage_.key_swizzle, SwizzleLayout(kStorageDim)}, lane_);
      ds_a.add_tile_offset({warp_q, 0});
      key_b.add_tile_offset({0, warp_n});
      FragmentDq dq;
      dq.clear();
      WarpDq{}(ds_a, key_b, kBc / 16, dq);
      int row_offset = warp_q * WarpShapeDq::kM;
      int column_offset = warp_n * WarpShapeDq::kN;
      AtomicEpilogue{}(dq,
          grad_query + int64_t(query_start + row_offset) * query_stride + column_offset,
          query_stride, max(0, min(WarpShapeDq::kM, valid_queries - row_offset)),
          max(0, min(WarpShapeDq::kN, kHeadDim - column_offset)), lane_);
      __syncthreads();

      if (tile + 1 < query_iterations) {
        query.add_tile_offset({0, 1});
        grad_output.add_tile_offset({0, 1});
        query_swizzle.add_tile_offset({0, 1});
        grad_output_swizzle.add_tile_offset({0, 1});
      }
    }
  }
};
