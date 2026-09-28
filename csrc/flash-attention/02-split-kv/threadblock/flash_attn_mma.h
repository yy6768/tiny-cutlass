#pragma once

#include <math_constants.h>
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/array.h"
#include "cutlass/matrix_coord.h"
#include "../epilogue/row_iterator.h"

// FA1: KV is the outer loop, Q is the inner loop. One CTA owns a batch/head.
// This layer owns both traversals, iterator advancement and synchronization.
template <class Policy>
class FlashAttnMma {
 public:
  using Element = typename Policy::Element;
  using LayoutQ = typename Policy::LayoutQ;
  using LayoutK = typename Policy::LayoutK;
  using LayoutV = typename Policy::LayoutV;
  using WarpQK = typename Policy::WarpQK;
  using WarpPV = typename Policy::WarpPV;
  using WarpShapeQK = typename Policy::WarpShapeQK;
  using ThreadblockShape0 = typename Policy::ThreadblockShape0;
  using ThreadblockShape1 = typename Policy::ThreadblockShape1;
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
  using Epilogue = typename Policy::Epilogue;
  using StateTileIterator = typename Epilogue::StateTileIterator;
  using OutputTileIterator = typename Epilogue::OutputTileIterator;
  using RowIterator = FlashAttnRowIterator<WarpQK>;
  static constexpr int kBr = Policy::kBr;
  static constexpr int kBc = Policy::kBc;
  static constexpr int kHeadDim = Policy::kHeadDim;
  static constexpr int kHeadDimV = Policy::kHeadDimV;
  static constexpr int kThreadCount = Policy::kThreadCount;
  static constexpr int kComputeWarps = Policy::kWarpsM;
  static constexpr int kQkIterations = kHeadDim / 16;
  static constexpr int kPvIterations = Policy::kStorageBc / 16;
  static constexpr int kCopyThreadsQ = Policy::kCopyThreadsQ;
  static constexpr int kCopyThreadsK = Policy::kCopyThreadsK;
  static constexpr int kCopyThreadsV = Policy::kCopyThreadsV;
  static_assert(kBr % OutputTileIterator::Shape::kRow == 0,
      "Output iterator slices must cover a complete Q tile");

  struct SharedStorage {
    alignas(16) Element k[Policy::kStorageBc * kHeadDim];
    alignas(16) Element v[Policy::kStorageBc * Policy::kPitchV];
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
  // Global iterators are local copies; the mainloop advances the tile origins.
  CUTLASS_DEVICE
  void copy_query(IteratorQ iterator_q, int valid_q) {
    if (thread_ < kCopyThreadsQ) {
      iterator_q.set_iteration_index(0);
      smem_iterator_q_.set_iteration_index(0);
      int first_row = ThreadMapQ::initial_offset(thread_).strided();
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < ThreadMapQ::Iterations::kCount; ++i) {
        int row = first_row + (i / ThreadMapQ::Iterations::kContiguous) * ThreadMapQ::Delta::kStrided;
        cutlass::arch::cp_async_zfill<sizeof(AccessTypeQ), cutlass::arch::CacheOperation::Global>(
            smem_iterator_q_.get(), iterator_q.get(), iterator_q.valid() && row < valid_q);
        ++iterator_q;
        ++smem_iterator_q_;
      }
    }
  }

  CUTLASS_DEVICE
  void copy_key_value(IteratorK iterator_k, IteratorV iterator_v, int valid_kv) {
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
        // V remains logically row-major; the SM80 congruous layout swizzles banks.
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
    x += __shfl_xor_sync(0xffffffff, x, 1);
    return x + __shfl_xor_sync(0xffffffff, x, 2);
  }

 public:
  CUTLASS_DEVICE
  FlashAttnMma(SharedStorage& storage, int thread, int warp, int lane)
      : storage_(storage), thread_(thread), warp_(warp), lane_(lane),
        smem_iterator_q_({storage.q, LayoutQ(kHeadDim)},
                         thread % kCopyThreadsQ),
        smem_iterator_k_({storage.k, LayoutK(kHeadDim)},
                         thread % kCopyThreadsK),
        smem_iterator_v_({storage.v, LayoutV(Policy::kPitchV)},
                         thread % kCopyThreadsV) {}

  // Match official FA1 device_1xN_loop -> device_1xN_ (num_splits=1).
  // Keep explicit m/l (Algorithm 1), rather than the official LSE encoding.
  // Algorithm 1 line 2: first KV round initializes O_i=0, l_i=0, m_i=-inf in
  // registers. Later rounds reload their FP32 state from global workspace.
  CUTLASS_DEVICE
  void operator()(SharedStorage &storage, ) {
    // Swizzle block
    ProbabilityOutputOp convert(ProbabilityOutputParams{});
    for (int j = 0; j < kv_tile_iterations; ++j) {
      // Matrix coordinates are (sequence, channel). The iterator itself uses
      // PitchLinear (channel, sequence), hence the {0, tile_index} conversion.
      cutlass::MatrixCoord tb_offset_kv{j * kBc, 0};
      IteratorK iterator_k = iterator_k_begin;
      IteratorV iterator_v = iterator_v_begin;
      if (tb_offset_kv.row()) {
        iterator_k.add_tile_offset({0, tb_offset_kv.row() / kBc});
        iterator_v.add_tile_offset({0, tb_offset_kv.row() / kBc});
      }
      int remaining_kv = seq_length_kv - tb_offset_kv.row();
      int valid_kv = remaining_kv < kBc ? remaining_kv : kBc;
      bool is_first = j == 0;
      bool is_last = j + 1 == kv_tile_iterations;
      copy_key_value(iterator_k, iterator_v, valid_kv);
      cutlass::arch::cp_async_fence();
      cutlass::arch::cp_async_wait<0>();
      __syncthreads(); // K/V stay resident throughout the inner Q scan.

      StateTileIterator state = state_begin;
      OutputTileIterator destination = output_begin;
      for (int q = 0; q < q_tile_iterations; ++q) {
        cutlass::MatrixCoord tb_offset_q{q * kBr, 0};
        IteratorQ iterator_q = iterator_q_begin;
        if (tb_offset_q.row())
          iterator_q.add_tile_offset({0, tb_offset_q.row() / kBr});
        int remaining_q = seq_length - tb_offset_q.row();
        int valid_q = remaining_q < kBr ? remaining_q : kBr;
        copy_query(iterator_q, valid_q);
        cutlass::arch::cp_async_fence();
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();
        if (warp_ < kComputeWarps) {
          RowIterator rows(row_maximum + tb_offset_q.row() + warp_ * WarpShapeQK::kM,
                           row_denominator + tb_offset_q.row() + warp_ * WarpShapeQK::kM,
                           WarpShapeQK::kM, 0, lane_);
          FragmentO output;
          FragmentL maximum, denominator;
          if (is_first) {
            output.clear();
            denominator.clear();
            CUTLASS_PRAGMA_UNROLL
            for (int r = 0; r < FragmentL::kElements; ++r)
              maximum[r] = -CUDART_INF_F;
          } else {
            state.load(output);
            rows.load(maximum, denominator);
          }
          // QK: the warp layer owns the D/16 instruction loop.
          WarpIteratorQ qa({storage_.q, LayoutQ(kHeadDim)}, lane_);
          WarpIteratorK kb({storage_.k, LayoutK(kHeadDim)}, lane_);
          qa.add_tile_offset({warp_, 0});
          FragmentS score;
          score.clear();
          WarpQK mma_qk{};
          mma_qk(qa, kb, kQkIterations, score);

          // Attention-specific online softmax; CUTLASS GEMM does not supply
          // this step.
          FragmentL tile_max, tile_sum; // tilde m_ij, tilde l_ij
          CUTLASS_PRAGMA_UNROLL
          for (int r = 0; r < FragmentL::kElements; ++r)
            tile_max[r] = -CUDART_INF_F;
          tile_sum.clear();

          // Each warp owns all KV columns of its Q rows; no cross-warp
          // reduction.
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < FragmentS::kElements; ++i) {
            int col = WarpQK::column(i, lane_);
            score[i] = col < valid_kv ? score[i] * scale : -CUDART_INF_F;
            int r = WarpQK::row_slot(i);
            tile_max[r] = fmaxf(tile_max[r], score[i]);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int r = 0; r < FragmentL::kElements; ++r) {
            tile_max[r] = reduce_max(tile_max[r]);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < FragmentS::kElements; ++i) {
            int r = WarpQK::row_slot(i);
            // Algorithm 1, line 10: tilde P uses the current tile's maximum.
            // Invalid KV columns have score=-inf, so their probability is zero.
            float probability = __expf(score[i] - tile_max[r]);
            tile_sum[r] += probability;
            score[i] = probability;
          }
          CUTLASS_PRAGMA_UNROLL
          for (int r = 0; r < FragmentL::kElements; ++r) {
            tile_sum[r] = reduce_sum(tile_sum[r]);
          }

          // Algorithm 1, line 11. Keep alpha*l_old for the old normalized O;
          // beta scales the current tile's tilde P V contribution.
          FragmentL previous_weight, beta;
          CUTLASS_PRAGMA_UNROLL
          for (int r = 0; r < FragmentL::kElements; ++r) {
            float new_max = fmaxf(maximum[r], tile_max[r]);
            float alpha = __expf(maximum[r] - new_max);
            beta[r] = __expf(tile_max[r] - new_max);
            previous_weight[r] = alpha * denominator[r];
            denominator[r] = previous_weight[r] + beta[r] * tile_sum[r];
            maximum[r] = new_max;
          }

          // Compute tilde P V separately from the previous normalized output.
          FragmentIteratorP pa(score);
          WarpIteratorV vb({storage_.v, LayoutV(Policy::kPitchV)}, lane_);
          FragmentO tile_output;
          tile_output.clear();
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
            mma_pv(tile_output, operand_p, operand_v, tile_output);
          }

          // Algorithm 1, line 12: O_new = (alpha*l_old*O_old + beta*tilde P
          // V)/l_new. Normalize after PV on every tile; neither P nor the final
          // epilogue divides by l.
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < FragmentO::kElements; ++i) {
            int r = WarpPV::row_slot(i);
            output[i] =
                (previous_weight[r] * output[i] + beta[r] * tile_output[i]) /
                denominator[r];
          }
          if (is_last) {
            epilogue(output, destination);
          } else {
            epilogue.store_intermediate(output, state);
          }
          rows.store(maximum, denominator);
        }
        // Release Q/K/V; also order global state stores before a later
        // KV round reloads them. All threads execute the same Q/KV loop bounds.
        __syncthreads();

        if (q + 1 < q_tile_iterations) {
          state.add_tile_offset({kBr / WarpShapeQK::kM, 0});
          destination.add_tile_offset(
              {kBr / OutputTileIterator::Shape::kRow, 0});
        }
      } // Q: every tile sees the same resident K_j/V_j.

    }
  }
};
