/*
  Naive multi-head attention kernels — three-stage, no fusion.

  Computes  O = softmax(Q @ K^T * scale) @ V  by launching three independent
  CUTLASS-based kernels, each driven by its own threadblock loop. The full
  attention matrix `P` is materialized in global memory between stages.

  Stages
  ------
    MM0      : P = Q @ K^T * scale       (CUTLASS threadblock GEMM)
    Softmax  : P[i,:] <- softmax(P[i,:]) (raw kernel, in-place per row)
    MM1      : O = P @ V                 (CUTLASS threadblock GEMM)

  Layout (BMHK)
  -------------
    Q : [B, Sq, H, d ]  RowMajor   q_strideM = H*d    q_strideH = d
    K : [B, Sk, H, d ]  RowMajor   k_strideM = H*d    k_strideH = d
    V : [B, Sk, H, dv]  RowMajor   v_strideM = H*dv   v_strideH = dv
    O : [B, Sq, H, dv]  RowMajor   o_strideM = H*dv   o_strideH = dv
    P : [B, H, Sq, Sk]  RowMajor   p_strideM = Sk     p_strideH = Sq*Sk

  Notes / scope
  -------------
    - dtype: P is materialized in scalar_t (matches Q/K/V/O). softmax casts
      to float internally for numerical stability; storage stays scalar_t.
    - The following features from the fused reference (kernel_forward.h in
      example 41) are intentionally REMOVED in this naive baseline:
        * dropout
        * causal / custom masks
        * attention bias
        * logsumexp output
        * variable sequence lengths (seqstart_q/k, seqlen_k)
      They will be reintroduced incrementally in later milestones.
    - Layout assumes a single contiguous BMHK tensor; non-default strides
      via cu_seqlens are not supported.
*/

#pragma once

#include <cmath>
#include <cinttypes>

#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/platform/platform.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "../gemm_kernel_utils.h"

using namespace gemm_kernel_utils;

namespace {

template <typename scalar_t, typename Arch>
constexpr int getWarpsPerSmFw() {
  return (Arch::kMinComputeCapability >= 80 &&
          !cutlass::platform::is_same<scalar_t, float>::value)
      ? 16
      : 12;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// AttentionMMKernel
//
// Generic CUTLASS GEMM skeleton used by both MM0 and MM1.
// The caller selects A/B/C layouts and the problem dimensions at
// instantiation time; the execution flow stays identical:
//
//   IteratorA / IteratorB -> threadblock MMA -> epilogue writeback
//
// MM0 instantiation:
//   A = Q, B = K (logical ColumnMajor / K^T), C = P, alpha = scale
//
// MM1 instantiation:
//   A = P, B = V (RowMajor), C = O, alpha = 1
////////////////////////////////////////////////////////////////////////////////
template <
    typename scalar_t_,
    // eg cutlass::arch::Sm80
    typename ArchTag,
    // ?
    bool isAligned_,
    int kQueriesPerBlock_,
    int kKeysPerBlock_,
    typename LayoutA_,
    typename LayoutB_,
    typename LayoutC_>
struct AttentionMMKernel {
  using scalar_t = scalar_t_;
  using accum_t  = float;
  using output_t = scalar_t; // P is stored in scalar_t

  static constexpr bool kIsAligned        = isAligned_;
  static constexpr int  kQueriesPerBlock  = kQueriesPerBlock_;
  static constexpr int  kKeysPerBlock     = kKeysPerBlock_;
  static constexpr bool kIsHalf           = cutlass::sizeof_bits<scalar_t>::value == 16;
  static constexpr int  kWarpSize         = 32;
  static_assert(kQueriesPerBlock % 32 == 0, "");
  static_assert(kKeysPerBlock    % 32 == 0, "");
  static constexpr int  kNumWarpsPerBlock = kQueriesPerBlock * kKeysPerBlock / (32 * 32);
  static constexpr int  kNumThreads       = kWarpSize * kNumWarpsPerBlock;
  static constexpr int  kMinBlocksPerSm   = getWarpsPerSmFw<scalar_t, ArchTag>() / kNumWarpsPerBlock;

  // ---- CUTLASS GEMM types ------------------------------------------------
  using GemmType    = DefaultGemmType<ArchTag, scalar_t>;
  using OpClass     = typename GemmType::OpClass;
  using DefaultConfig = typename cutlass::gemm::device::DefaultGemmConfiguration<
      OpClass, ArchTag, scalar_t, scalar_t, output_t, accum_t>;

  static constexpr int kAlignmentA = kIsAligned ? DefaultConfig::kAlignmentA
                                                : GemmType::kMinimumAlignment;
  static constexpr int kAlignmentB = kIsAligned ? DefaultConfig::kAlignmentB
                                                : GemmType::kMinimumAlignment;

  using ThreadblockShape = cutlass::gemm::GemmShape<
      kQueriesPerBlock, kKeysPerBlock, GemmType::ThreadK>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
  using InstructionShape = typename GemmType::InstructionShape;

  static constexpr int kStages =
      (ArchTag::kMinComputeCapability >= 80 && kIsHalf) ? 4 : DefaultConfig::kStages;

  // P = Q @ K^T : A=Q (RowMajor), B=K^T (treated as ColumnMajor over K), C=P (RowMajor)
  using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
      scalar_t, LayoutA_, kAlignmentA,
      scalar_t, LayoutB_, kAlignmentB,
      output_t, LayoutC_,
      accum_t,
      OpClass, ArchTag,
      ThreadblockShape, WarpShape, InstructionShape,
      typename DefaultConfig::EpilogueOutputOp,
      void, // ThreadblockSwizzle (unused — we drive the loop manually)
      kStages,
      false, // SplitKSerial
      typename GemmType::Operator>;

  using Mma             = typename DefaultGemm::Mma;
  using Epilogue        = typename DefaultGemm::Epilogue;
  using EpilogueOutputOp = typename DefaultConfig::EpilogueOutputOp;

  // ---- Params ------------------------------------------------------------
  struct Params {
    scalar_t* a_ptr = nullptr; // [B, M, H, K]
    scalar_t* b_ptr = nullptr; // [B, N, H, K] or [B, K, H, N] depending on layout
    output_t* c_ptr = nullptr; // [B, H, M, N] / [B, M, H, N]

    accum_t alpha = 0.0f;

    int32_t problem_m = 0;
    int32_t problem_n = 0;
    int32_t problem_k = 0;

    int32_t a_strideM = 0;
    int32_t b_strideM = 0;
    int32_t c_strideM = 0;
    int32_t a_strideH = 0;
    int32_t b_strideH = 0;
    int32_t c_strideH = 0;
    int64_t a_strideB = 0;
    int64_t b_strideB = 0;
    int64_t c_strideB = 0;

    int32_t num_heads   = 0;
    int32_t num_batches = 0;

    // Move pointers to the (batch, head, query-block) this CTA owns.
    CUTLASS_DEVICE bool advance_to_block() {
      auto batch_id    = blockIdx.z;
      auto head_id     = blockIdx.y;
      auto query_start = blockIdx.x * kQueriesPerBlock;

      if (query_start >= problem_m) return false;

      a_ptr += batch_id * a_strideB + query_start * a_strideM + head_id * a_strideH;
      b_ptr += batch_id * b_strideB + head_id * b_strideH;
      c_ptr += batch_id * c_strideB + query_start * c_strideM + head_id * c_strideH;

      problem_m -= query_start;
      return true;
    }

    CUTLASS_HOST dim3 getBlocksGrid() const {
      return dim3(
          cutlass::ceil_div(problem_m, kQueriesPerBlock),
          num_heads,
          num_batches);
    }
    CUTLASS_HOST dim3 getThreadsGrid() const {
      return dim3(kWarpSize, kNumWarpsPerBlock, 1);
    }
  };

  // ---- Shared storage (mma <-> epilogue share smem) ----------------------
  union SharedStorage {
    typename Mma::SharedStorage      mm0;
    typename Epilogue::SharedStorage epilogue;
  };

  // ---- Helpers -----------------------------------------------------------
  static CUTLASS_DEVICE int8_t  lane_id()    { return threadIdx.x; }
  static CUTLASS_DEVICE int8_t  warp_id()    { return threadIdx.y; }
  static CUTLASS_DEVICE int16_t thread_id()  { return threadIdx.x + threadIdx.y * blockDim.x; }

  // ---- Main body ---------------------------------------------------------
  static CUTLASS_DEVICE void attention_kernel(Params& p) {
    extern __shared__ char smem_buffer[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);

    auto my_warp_id = warp_id();
    auto my_lane_id = lane_id();
    auto my_thread_id = thread_id();

    int32_t problem_size_m = cutlass::fast_min(
        (int32_t)kQueriesPerBlock, p.problem_m);
    int32_t problem_size_k = p.problem_k;

    if constexpr (cutlass::platform::is_same<LayoutB_, cutlass::layout::ColumnMajor>::value) {
      for (int32_t iter_key_start = 0; iter_key_start < p.problem_n;
           iter_key_start += kKeysPerBlock) {
        int32_t problem_size_n = cutlass::fast_min(
            (int32_t)kKeysPerBlock, p.problem_n - iter_key_start);

        __syncthreads();

        // -- Operand iterators ------------------------------------------------
        typename Mma::IteratorA iterator_A(
            typename Mma::IteratorA::Params(
                LayoutA_(p.a_strideM)),
            p.a_ptr,
            {problem_size_m, problem_size_k},
            my_thread_id,
            {0, 0});

        typename Mma::IteratorB iterator_B(
            typename Mma::IteratorB::Params(
                LayoutB_(p.b_strideM)),
            p.b_ptr + int64_t(iter_key_start) * p.b_strideM,
            {problem_size_k, problem_size_n},
            my_thread_id,
            {0, 0});

        // -- Threadblock-scoped MMA -------------------------------------------
        Mma mma(shared_storage.mm0, my_thread_id, my_warp_id, my_lane_id);

        typename Mma::FragmentC accum;
        accum.clear();

        int gemm_k_iterations =
            (problem_size_k + Mma::Shape::kK - 1) / Mma::Shape::kK;
        mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
        __syncthreads();

        // -- Epilogue: write tile to P with alpha=scale, beta=0 ---------------
        typename Epilogue::OutputTileIterator iterator_D(
            typename Epilogue::OutputTileIterator::Params(p.c_strideM),
            p.c_ptr,
            {problem_size_m, p.problem_n},
            my_thread_id,
            {0, iter_key_start});

        EpilogueOutputOp output_op({p.alpha, accum_t(0)});
        Epilogue epilogue(shared_storage.epilogue, my_thread_id, my_warp_id, my_lane_id);
        epilogue(output_op, iterator_D, accum, iterator_D);
      }
    } else {
      int32_t nBlockN = cutlass::ceil_div(
          (int32_t)p.problem_n, (int32_t)Mma::Shape::kN);

      for (int32_t blockN = 0; blockN < nBlockN; ++blockN) {
        int32_t problem_size_n = cutlass::fast_min(
            (int32_t)Mma::Shape::kN,
            p.problem_n - blockN * (int32_t)Mma::Shape::kN);

        __syncthreads();

        // -- Operand iterators ------------------------------------------------
        typename Mma::IteratorA iterator_A(
            typename Mma::IteratorA::Params(
                LayoutA_(p.a_strideM)),
            p.a_ptr,
            {problem_size_m, problem_size_k},
            my_thread_id,
            {0, 0});

        typename Mma::IteratorB iterator_B(
            typename Mma::IteratorB::Params(
                LayoutB_(p.b_strideM)),
            p.b_ptr,
            {problem_size_k, problem_size_n},
            my_thread_id,
            {0, blockN * (int)Mma::Shape::kN});

        // -- Threadblock-scoped MMA -------------------------------------------
        Mma mma(shared_storage.mm0, my_thread_id, my_warp_id, my_lane_id);

        typename Mma::FragmentC accum;
        accum.clear();

        int gemm_k_iterations =
            (problem_size_k + Mma::Shape::kK - 1) / Mma::Shape::kK;
        mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
        __syncthreads();

        // -- Epilogue: write tile to P with alpha=scale, beta=0 ---------------
        typename Epilogue::OutputTileIterator iterator_D(
            typename Epilogue::OutputTileIterator::Params(p.c_strideM),
            p.c_ptr,
            {problem_size_m, p.problem_n},
            my_thread_id,
            {0, blockN * (int)Mma::Shape::kN});

        EpilogueOutputOp output_op({p.alpha, accum_t(0)});
        Epilogue epilogue(shared_storage.epilogue, my_thread_id, my_warp_id, my_lane_id);
        epilogue(output_op, iterator_D, accum, iterator_D);
      }
    }
  }
};

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_mm0_kernel(typename AK::Params p) {
  if (!p.advance_to_block()) return;
  AK::attention_kernel(p);
}

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_mm1_kernel(typename AK::Params p) {
  if (!p.advance_to_block()) return;
  AK::attention_kernel(p);
}

////////////////////////////////////////////////////////////////////////////////
// Softmax — row-wise safe softmax, in-place on P
//
// Grid : (num_queries, 1, num_batches * num_heads)
// Block: (kThreads, 1, 1)
// Smem : kNumWarps * sizeof(float)
//
// Three passes; each cross-warp reduction goes warp-shuffle -> smem -> warp-shuffle:
//   (1) row_max = max_j P[i, j]
//   (2) P[i, j] <- exp(P[i, j] - row_max);  row_sum = sum_j P[i, j]
//   (3) P[i, j] <- P[i, j] / row_sum
////////////////////////////////////////////////////////////////////////////////
template <
    typename scalar_t_,
    int kThreads_ = 256>
struct AttentionSoftmaxKernel {
  using scalar_t = scalar_t_;
  using accum_t = float; // reduction always done in fp32 for stability

  static constexpr int kWarpSize = 32;
  static constexpr int kThreads  = kThreads_;
  static constexpr int kNumWarps = kThreads / kWarpSize;
  static_assert(kThreads % kWarpSize == 0, "");

  static constexpr int kNumThreads     = kThreads;
  static constexpr int kMinBlocksPerSm = 1;

  struct Params {
    scalar_t* p_ptr       = nullptr; // [B, H, Sq, Sk]
    int32_t   num_queries = 0;
    int32_t   num_keys    = 0;
    int32_t   num_heads   = 0;
    int32_t   num_batches = 0;

    int32_t   p_strideM = 0;          // = num_keys
    int64_t   p_strideH = 0;          // = num_queries * num_keys
    int64_t   p_strideB = 0;          // = num_heads * num_queries * num_keys

    CUTLASS_DEVICE bool advance_to_block() {
      auto bh      = blockIdx.z;
      auto head_id  = bh % num_heads;
      auto batch_id = bh / num_heads;
      auto row_id   = blockIdx.x;
      if (row_id >= num_queries) return false;
      p_ptr += batch_id * p_strideB + head_id * p_strideH + row_id * p_strideM;
      return true;
    }

    CUTLASS_HOST dim3 getBlocksGrid() const {
      return dim3(num_queries, 1, num_batches * num_heads);
    }
    CUTLASS_HOST dim3 getThreadsGrid() const {
      return dim3(kThreads, 1, 1);
    }
    CUTLASS_HOST int getSmemBytes() const {
      return kNumWarps * sizeof(accum_t);
    }
  };

  static CUTLASS_DEVICE accum_t warp_reduce_max(accum_t v) {
    CUTLASS_PRAGMA_UNROLL
    for (int o = kWarpSize / 2; o > 0; o >>= 1) {
      v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    }
    return v;
  }
  static CUTLASS_DEVICE accum_t warp_reduce_sum(accum_t v) {
    CUTLASS_PRAGMA_UNROLL
    for (int o = kWarpSize / 2; o > 0; o >>= 1) {
      v += __shfl_xor_sync(0xffffffff, v, o);
    }
    return v;
  }

  static CUTLASS_DEVICE void attention_kernel(Params& p) {
    extern __shared__ char smem_buffer[];
    accum_t* warp_storage = reinterpret_cast<accum_t*>(smem_buffer);

    int tid     = threadIdx.x;
    int lane_id = tid & (kWarpSize - 1);
    int warp_id = tid / kWarpSize;
    scalar_t* row_ptr = p.p_ptr;

    // ---------- Pass 1: row max ----------
    accum_t mi = -cutlass::platform::numeric_limits<accum_t>::infinity();
    for (int j = tid; j < p.num_keys; j += kThreads) {
      mi = fmaxf(mi, accum_t(row_ptr[j]));
    }
    mi = warp_reduce_max(mi);
    if (lane_id == 0) warp_storage[warp_id] = mi;
    __syncthreads();
    if (warp_id == 0) {
      mi = (lane_id < kNumWarps) ? warp_storage[lane_id]
          : -cutlass::platform::numeric_limits<accum_t>::infinity();
      mi = warp_reduce_max(mi);
      if (lane_id == 0) warp_storage[0] = mi;
    }
    __syncthreads();
    accum_t row_max = warp_storage[0];

    // ---------- Pass 2: exp(x - max) and row sum ----------
    accum_t di = 0.0f;
    for (int j = tid; j < p.num_keys; j += kThreads) {
      accum_t e = __expf(accum_t(row_ptr[j]) - row_max);
      row_ptr[j] = scalar_t(e);
      di += e;
    }
    di = warp_reduce_sum(di);
    if (lane_id == 0) warp_storage[warp_id] = di;
    __syncthreads();
    if (warp_id == 0) {
      di = (lane_id < kNumWarps) ? warp_storage[lane_id] : 0.0f;
      di = warp_reduce_sum(di);
      if (lane_id == 0) warp_storage[0] = di;
    }
    __syncthreads();
    accum_t inv_sum = 1.0f / warp_storage[0];

    // ---------- Pass 3: normalize ----------
    for (int j = tid; j < p.num_keys; j += kThreads) {
      row_ptr[j] = scalar_t(accum_t(row_ptr[j]) * inv_sum);
    }
  }
};

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_softmax_kernel(typename AK::Params p) {
  if (!p.advance_to_block()) return;
  AK::attention_kernel(p);
}
