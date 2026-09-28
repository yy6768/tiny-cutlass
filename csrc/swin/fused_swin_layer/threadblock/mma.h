#pragma once
#include "cutlass/gemm/threadblock/mma_base.h"

namespace tiny_cutlass::swin::fused_swin_layer::threadblock {

// A synchronous stage for operands whose source can already be shared memory.
// Same iterator/warp contracts as CUTLASS MmaPipelined; no address remapping here.
template <typename Shape, typename IteratorA_, typename SmemIteratorA,
          typename IteratorB_, typename SmemIteratorB, typename Policy>
class Mma : public cutlass::gemm::threadblock::MmaBase<Shape, Policy, 1> {
 public:
  using Base = cutlass::gemm::threadblock::MmaBase<Shape, Policy, 1>;
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using Operator = typename Base::Operator;
  using FragmentC = typename Operator::FragmentC;
  using SharedStorage = typename Base::SharedStorage;
 private:
  SharedStorage& shared_;
  int lane_;
  SmemIteratorA smem_iterator_A_;
  SmemIteratorB smem_iterator_B_;
 public:
  CUTLASS_DEVICE
  Mma(SharedStorage& shared, int thread_idx, int warp_idx, int lane_idx)
      : Base(shared, thread_idx, warp_idx, lane_idx), shared_(shared), lane_(lane_idx),
        smem_iterator_A_(shared.operand_A_ref(), thread_idx),
        smem_iterator_B_(shared.operand_B_ref(), thread_idx) {
    static_assert(Base::WarpCount::kCount == 1, "one warp owns one window tile");
  }

  CUTLASS_DEVICE
  void operator()(int gemm_k_iterations, FragmentC& accum, IteratorA iterator_A,
                  IteratorB iterator_B, FragmentC const& src_accum) {
    accum = src_accum;
    Operator warp_mma;
    for (int tile = 0; tile < gemm_k_iterations; ++tile) {
      typename IteratorA::Fragment a; a.clear();
      typename IteratorB::Fragment b; b.clear();
      iterator_A.load(a); iterator_B.load(b);
      smem_iterator_A_.store(a); smem_iterator_B_.store(b);
      __syncthreads();
      typename Operator::IteratorA warp_A(shared_.operand_A_ref(), lane_);
      typename Operator::IteratorB warp_B(shared_.operand_B_ref(), lane_);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < Base::kWarpGemmIterations; ++k) {
        typename Operator::FragmentA wa;
        typename Operator::FragmentB wb;
        warp_A.set_kgroup_index(k); warp_B.set_kgroup_index(k);
        warp_A.load(wa); warp_B.load(wb);
        warp_mma(accum, wa, wb, accum);
        ++warp_A; ++warp_B;
      }
      __syncthreads();
      if (tile + 1 < gemm_k_iterations) { ++iterator_A; ++iterator_B; }
    }
  }
};

}  // namespace tiny_cutlass::swin::fused_swin_layer::threadblock
