#pragma once

#include <math_constants.h>
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

// One fixed Q tile scans every KV tile. This layer owns operand movement,
// online softmax, numerator accumulation and shared-memory synchronization.
template <class Policy>
class FlashAttnMmaSplitQ {
 public:
  using Element = typename Policy::Element;
  using LayoutQ = typename Policy::LayoutQ;
  using LayoutK = typename Policy::LayoutK;
  using LayoutP = typename Policy::LayoutP;
  using LayoutV = typename Policy::LayoutV;
  using WarpQK = typename Policy::WarpQK;
  using WarpPV = typename Policy::WarpPV;
  using WarpShapeQK = typename Policy::WarpShapeQK;
  using FragmentS = typename WarpQK::Fragment;
  using FragmentO = typename WarpPV::Fragment;
  using FragmentL = typename WarpQK::FragmentRow;
  using IteratorQ = typename Policy::IteratorQ;
  using IteratorK = typename Policy::IteratorK;
  using IteratorV = typename Policy::IteratorV;
  using SmemIteratorQ = typename Policy::SmemIteratorQ;
  using SmemIteratorK = typename Policy::SmemIteratorK;
  using SmemIteratorV = typename Policy::SmemIteratorV;
  using ThreadMapQ = typename IteratorQ::ThreadMap;
  using ThreadMapK = typename IteratorK::ThreadMap;
  using ThreadMapV = typename IteratorV::ThreadMap;
  using AccessTypeQ = typename IteratorQ::AccessType;
  using AccessTypeK = typename IteratorK::AccessType;
  using AccessTypeV = typename IteratorV::AccessType;
  using WarpIteratorQ = typename WarpQK::IteratorA;
  using WarpIteratorK = typename WarpQK::IteratorB;
  using WarpIteratorV = typename WarpPV::IteratorB;
  using FragmentIteratorP = typename Policy::FragmentIteratorP;
  using ProbabilityOutputOp = typename Policy::ProbabilityOutputOp;
  using ProbabilityOutputParams = typename ProbabilityOutputOp::Params;
  using MmaPV = typename WarpPV::Mma;
  using FragmentP = typename MmaPV::FragmentA;
  using FragmentV = typename MmaPV::FragmentB;
  using TransformedFragmentP = typename MmaPV::TransformedFragmentA;
  using TransformedFragmentV = typename MmaPV::TransformedFragmentB;
  static constexpr int kBr = Policy::kBr;
  static constexpr int kBc = Policy::kBc;
  static constexpr int kHeadDim = Policy::kHeadDim;
  static constexpr int kHeadDimV = Policy::kHeadDimV;
  static constexpr int kThreadCount = Policy::kThreadCount;
  static constexpr int kQkIterations = kHeadDim / 16;
  static constexpr int kPvIterations = Policy::kStorageBc / 16;
  static constexpr int kCopyThreadsQ = Policy::kCopyThreadsQ;
  static constexpr int kCopyThreadsK = Policy::kCopyThreadsK;
  static constexpr int kCopyThreadsV = Policy::kCopyThreadsV;

  struct SharedStorage {
    alignas(16) Element k[Policy::kStorageBc * kHeadDim];
    alignas(16) Element v[Policy::kStorageBc * Policy::kPitchV];
    // Q survives the complete KV scan. P remains in accumulator registers.
    alignas(16) Element q[kBr * kHeadDim];
  };

 private:
  SharedStorage& storage_;
  int thread_, warp_, lane_;
  SmemIteratorQ smem_iterator_q_;
  SmemIteratorK smem_iterator_k_;
  SmemIteratorV smem_iterator_v_;

  // Same organization as example 13's B2bMmaMultistage::copy_tiles_and_advance_0:
  // Mma owns the shared iterators and issues copies through stock iterator APIs.
  // Global iterators are local copies; the KV mainloop advances the tile origins.
  CUTLASS_DEVICE
  void copy_query(IteratorQ iterator_q) {
    if (thread_ < kCopyThreadsQ) {
      iterator_q.set_iteration_index(0);
      smem_iterator_q_.set_iteration_index(0);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < ThreadMapQ::Iterations::kCount; ++i) {
        cutlass::arch::cp_async_zfill<sizeof(AccessTypeQ), cutlass::arch::CacheOperation::Global>(
            smem_iterator_q_.get(), iterator_q.get(), iterator_q.valid());
        ++iterator_q;
        ++smem_iterator_q_;
      }
    }
  }

  CUTLASS_DEVICE
  void copy_tiles(IteratorK iterator_k, IteratorV iterator_v, int valid_kv) {
    if (thread_ < kCopyThreadsK) {
      iterator_k.set_iteration_index(0);
      smem_iterator_k_.set_iteration_index(0);
      int first_row = ThreadMapK::initial_offset(thread_).strided();
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < ThreadMapK::Iterations::kCount; ++i) {
        int row = first_row + (i / ThreadMapK::Iterations::kContiguous) * ThreadMapK::Delta::kStrided;
        // Zero the physical padding beyond logical Bc, including steady-state tiles.
        cutlass::arch::cp_async_zfill<sizeof(AccessTypeK), cutlass::arch::CacheOperation::Global>(
            smem_iterator_k_.get(), iterator_k.get(), iterator_k.valid() && row < valid_kv);
        ++iterator_k;
        ++smem_iterator_k_;
      }
    }
    if (thread_ < kCopyThreadsV) {
      iterator_v.set_iteration_index(0);
      smem_iterator_v_.set_iteration_index(0);
      int first_row = ThreadMapV::initial_offset(thread_).strided();
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < ThreadMapV::Iterations::kCount; ++i) {
        int row = first_row + (i / ThreadMapV::Iterations::kContiguous) * ThreadMapV::Delta::kStrided;
        cutlass::arch::cp_async_zfill<sizeof(AccessTypeV), cutlass::arch::CacheOperation::Global>(
            smem_iterator_v_.get(), iterator_v.get(), iterator_v.valid() && row < valid_kv);
        ++iterator_v;
        ++smem_iterator_v_;
      }
    }
  }

  CUTLASS_DEVICE
  static float reduce_max(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1));
    return fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2));
  }
  CUTLASS_DEVICE
  static float reduce_sum(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 2);
    return x + __shfl_xor_sync(0xffffffff, x, 1);
  }

 public:
  CUTLASS_DEVICE
  FlashAttnMmaSplitQ(SharedStorage& storage, int thread, int warp, int lane)
      : storage_(storage), thread_(thread), warp_(warp), lane_(lane),
        smem_iterator_q_({storage.q, cutlass::layout::PitchLinear(kHeadDim)},
                         thread % kCopyThreadsQ),
        smem_iterator_k_({storage.k, cutlass::layout::PitchLinear(kHeadDim)},
                         thread % kCopyThreadsK),
        smem_iterator_v_({storage.v, LayoutV(Policy::kPitchV)},
                         thread % kCopyThreadsV) {}

  // output and denominator start at zero; maximum is private to the mainloop.
  CUTLASS_DEVICE
  void operator()(int kv_tile_iterations, int last_kv_rows,
                  FragmentO& output, FragmentL& denominator,
                  IteratorQ iterator_q, IteratorK iterator_k, IteratorV iterator_v,
                  float scale, FragmentL& logsumexp, bool save_logsumexp) {
    FragmentL maximum;
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < FragmentL::kElements; ++r)
      maximum[r] = -CUDART_INF_F;

    copy_query(iterator_q);
    // Query copy is committed with the first K tile below.
    ProbabilityOutputOp convert(ProbabilityOutputParams{});
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kv_tile_iterations; ++j) { // Fixed Q tile scans KV tiles
      int valid_kv = j + 1 == kv_tile_iterations ? last_kv_rows : kBc;

      copy_tiles(iterator_k, iterator_v, valid_kv);
      cutlass::arch::cp_async_fence();
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();

      // QK: the warp layer owns the D/16 instruction loop.
      WarpIteratorQ qa({storage_.q, LayoutQ(kHeadDim)}, lane_);
      WarpIteratorK kb({storage_.k, LayoutK(kHeadDim)}, lane_);
      qa.add_tile_offset({warp_, 0});
      FragmentS score;
      score.clear();
      WarpQK mma_qk{};
      mma_qk(qa, kb, kQkIterations, score);

      // Attention-specific online softmax; CUTLASS GEMM does not supply this step.
      // FA2 softmax.h:153-172 keeps row sums lane-local until normalization.
      FragmentL tile_max, alpha;
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < FragmentL::kElements; ++r)
        tile_max[r] = -CUDART_INF_F;
      // The maximum stays in natural-log score units, including finite
      // negative and zero scale. Mask after scaling so padding remains -inf.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentS::kElements; ++i) {
        int col = WarpQK::column(i, lane_);
        score[i] = col < valid_kv ? score[i] * scale : -CUDART_INF_F;
        int r = WarpQK::row_slot(i);
        tile_max[r] = fmaxf(tile_max[r], score[i]);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < FragmentL::kElements; ++r) {
        float new_max = fmaxf(maximum[r], reduce_max(tile_max[r]));
        alpha[r] = __expf(maximum[r] - new_max);
        maximum[r] = new_max;
        denominator[r] *= alpha[r];
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentS::kElements; ++i) {
        int r = WarpQK::row_slot(i);
        float probability = __expf(score[i] - maximum[r]);
        denominator[r] += probability;
        score[i] = probability;
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentO::kElements; ++i)
        output[i] *= alpha[WarpPV::row_slot(i)];

      // PV adds the current contribution to the rescaled FP32 numerator.
      // Example 13's accumulator fragment iterator delivers register P as A1.
      FragmentIteratorP pa(score);
      WarpIteratorV vb({storage_.v, LayoutV(Policy::kPitchV)}, lane_);
      MmaPV mma_pv;
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < kPvIterations; ++k) {
        FragmentP probability;
        FragmentV value;
        TransformedFragmentP operand_p;
        TransformedFragmentV operand_v;
        pa.load(probability, convert);
        vb.load(value);
        ++pa;
        ++vb;
        mma_pv.transform(operand_p, operand_v, probability, value);
        mma_pv(output, operand_p, operand_v, output);
      }
      __syncthreads(); // All K/V readers finish before the next tile copy.

      if (j + 1 < kv_tile_iterations) {
        iterator_k.add_tile_offset({0, 1});
        iterator_v.add_tile_offset({0, 1});
      }
    }
    // Every lane accumulated only its own columns. Combine them once after
    // the KV scan, when the epilogue first needs the complete row denominator.
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < FragmentL::kElements; ++r)
      denominator[r] = reduce_sum(denominator[r]);
    if (save_logsumexp) {
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < FragmentL::kElements; ++r)
        logsumexp[r] = maximum[r] + __logf(denominator[r]);
    }
  }
};
