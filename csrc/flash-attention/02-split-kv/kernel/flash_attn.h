#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_ref.h"
#include "../../flash_attention.h"

// Official FA1 loop structure with num_splits=1: one CTA owns one batch/head.
// Kernel constructs initial iterators; Mma owns the complete KV -> Q traversal.
template <class Mma_, class Epilogue_, class ArchTag_>
struct FlashAttnKernel {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ArchTag = ArchTag_;
  using Element = typename Mma::Element;
  using Layout = cutlass::layout::RowMajor;
  using TensorRefQ = cutlass::TensorRef<Element const, Layout>;
  using TensorRefK = cutlass::TensorRef<Element const, Layout>;
  using TensorRefV = cutlass::TensorRef<Element const, Layout>;
  using TensorRefO = cutlass::TensorRef<Element, Layout>;
  using IteratorQ = typename Mma::IteratorQ;
  using IteratorK = typename Mma::IteratorK;
  using IteratorV = typename Mma::IteratorV;
  using IteratorParamsQ = typename IteratorQ::Params;
  using IteratorParamsK = typename IteratorK::Params;
  using IteratorParamsV = typename IteratorV::Params;
  using StateTileIterator = typename Epilogue::StateTileIterator;
  using OutputTileIterator = typename Epilogue::OutputTileIterator;
  using ThreadblockShape0 = typename Mma::ThreadblockShape0;
  using ThreadblockShape1 = typename Mma::ThreadblockShape1;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using SharedStorage = typename Mma::SharedStorage;
  static constexpr int kThreadCount = Mma::kThreadCount;
  static constexpr int kBr = Mma::kBr;
  static constexpr int kBc = Mma::kBc;
  static constexpr int kHeadDim = Mma::kHeadDim;
  static constexpr int kHeadDimV = Mma::kHeadDimV;

  // Host-side capability check, as in cutlass::gemm::kernel::Gemm.
  // Input alignment follows the vectorized global iterators; the canonical
  // output iterator writes individual elements.
  static cutlass::Status can_implement(Problem const& p, Tensors const& t) {
    if (p.batch_size <= 0 || p.head_number <= 0 ||
        p.seq_length <= 0 || p.seq_length_kv <= 0 ||
        p.head_size != kHeadDim || p.head_size_v != kHeadDimV ||
        !std::isfinite(p.scale))
      return cutlass::Status::kErrorInvalidProblem;

    int64_t padded_q = (int64_t(p.seq_length - 1) / kBr + 1) * kBr;
    int64_t padded_kv = (int64_t(p.seq_length_kv - 1) / kBc + 1) * kBc;
    if (padded_q > std::numeric_limits<int>::max() ||
        padded_kv > std::numeric_limits<int>::max() ||
        int64_t(p.head_number) * kHeadDim > std::numeric_limits<int>::max() ||
        int64_t(p.head_number) * kHeadDimV > std::numeric_limits<int>::max())
      return cutlass::Status::kErrorInvalidProblem;

    if (!t.query || !t.key || !t.value || !t.output)
      return cutlass::Status::kErrorInvalidProblem;
    if ((reinterpret_cast<uintptr_t>(t.query) % sizeof(typename IteratorQ::AccessType)) ||
        (reinterpret_cast<uintptr_t>(t.key) % sizeof(typename IteratorK::AccessType)) ||
        (reinterpret_cast<uintptr_t>(t.value) % sizeof(typename IteratorV::AccessType)) ||
        (reinterpret_cast<uintptr_t>(t.output) % sizeof(typename OutputTileIterator::Element)))
      return cutlass::Status::kErrorMisalignedOperand;
    return cutlass::Status::kSuccess;
  }

  struct Params {
    Problem problem;
    TensorRefQ ref_q;
    TensorRefK ref_k;
    TensorRefV ref_v;
    TensorRefO ref_o;
    cutlass::gemm::GemmCoord problem_size_0{};
    cutlass::gemm::GemmCoord problem_size_1{};
    cutlass::gemm::GemmCoord grid_tiled_shape{};
    int swizzle_log_tile = 0;
    IteratorParamsQ params_q;
    IteratorParamsK params_k;
    IteratorParamsV params_v;
    int q_tile_count = 0;
    int padded_q_rows = 0;
    float* partial_output = nullptr;
    float* maximum = nullptr;
    float* denominator = nullptr;

    Params() = default;
    Params(Problem const& p, cutlass::gemm::GemmCoord const& grid_shape,
           TensorRefQ ref_q_, TensorRefK ref_k_, TensorRefV ref_v_,
           TensorRefO ref_o_, void* workspace)
        : problem(p), ref_q(ref_q_), ref_k(ref_k_), ref_v(ref_v_), ref_o(ref_o_),
          problem_size_0(p.seq_length, p.seq_length_kv, p.head_size),
          problem_size_1(p.seq_length, p.head_size_v, p.seq_length_kv),
          grid_tiled_shape(grid_shape),
          swizzle_log_tile(ThreadblockSwizzle::get_log_tile(grid_shape)),
          params_q(cutlass::layout::PitchLinear(ref_q.stride(0))),
          params_k(cutlass::layout::PitchLinear(ref_k.stride(0))),
          params_v(cutlass::layout::PitchLinear(ref_v.stride(0))),
          q_tile_count((p.seq_length - 1) / kBr + 1),
          padded_q_rows(q_tile_count * kBr) {
      int64_t rows = int64_t(p.batch_size) * p.head_number * padded_q_rows;
      partial_output = static_cast<float*>(workspace);
      maximum = partial_output + rows * kHeadDimV;
      denominator = maximum + rows;
    }
  };

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& storage) {
    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);
    //  Early quit
    if (threadblock_tile_offset.m() >= params.grid_tiled_shape.m() ||
        threadblock_tile_offset.n() >= params.grid_tiled_shape.n() ||
        threadblock_tile_offset.k() >= params.grid_tiled_shape.k()) return;

    auto const& p = params.problem;
    // The CTA owns one (batch, head); KV and Q tile indices stay in Mma.
    int batch = threadblock_tile_offset.m();
    int head = threadblock_tile_offset.n();

    // TensorRefs describe the physical BSHD pitch. Move each view to this
    // batch/head; its logical matrix is now [sequence, head dimension].
    TensorRefQ ref_q = params.ref_q;
    TensorRefK ref_k = params.ref_k;
    TensorRefV ref_v = params.ref_v;
    TensorRefO ref_o = params.ref_o;
    ref_q.add_pointer_offset(int64_t(batch) * p.seq_length * ref_q.stride(0));
    ref_k.add_pointer_offset(int64_t(batch) * p.seq_length_kv * ref_k.stride(0));
    ref_v.add_pointer_offset(int64_t(batch) * p.seq_length_kv * ref_v.stride(0));
    ref_o.add_pointer_offset(int64_t(batch) * p.seq_length * ref_o.stride(0));
    ref_q.add_coord_offset({0, head * kHeadDim});
    ref_k.add_coord_offset({0, head * kHeadDim});
    ref_v.add_coord_offset({0, head * kHeadDimV});
    ref_o.add_coord_offset({0, head * kHeadDimV});

    // Padded iterator extents make every advance a full logical tile. Mma's
    // copy predicates separately mask real tails, so no padded input is read.
    int kv_tile_iterations = (p.seq_length_kv - 1) / kBc + 1;
    int padded_kv_rows = kv_tile_iterations * kBc;
    int thread_idx = int(threadIdx.x);
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx % 32;

    // Construct global iterators once. PitchLinear coordinates are (D, sequence).
    IteratorQ iterator_q(params.params_q, ref_q.data(),
        {p.head_size, params.padded_q_rows}, thread_idx);
    IteratorK iterator_k(params.params_k, ref_k.data(),
        {p.head_size, padded_kv_rows}, thread_idx);
    IteratorV iterator_v(params.params_v, ref_v.data(),
        {p.head_size_v, padded_kv_rows}, thread_idx);

    // Padded FP32 state permits the stock accumulator iterator's complete-tile
    // load/store. The final output iterator predicates the true sequence tail.
    int warp_row = warp_idx < Mma::kComputeWarps
        ? warp_idx * Mma::WarpShapeQK::kM : 0;
    int64_t state_row = (int64_t(batch) * p.head_number + head) * params.padded_q_rows;
    StateTileIterator state(
        {params.partial_output + (state_row + warp_row) * kHeadDimV,
         cutlass::layout::RowMajor(kHeadDimV)}, lane_idx);
    ref_o.add_coord_offset({warp_row, 0});
    OutputTileIterator output(
        ref_o,
        {p.seq_length - warp_row, Epilogue::kWarpColumns}, lane_idx);
    Mma mma(storage, thread_idx, warp_idx, lane_idx);
    Epilogue epilogue;
    mma(kv_tile_iterations, p.seq_length_kv, params.q_tile_count, p.seq_length,
        iterator_q, iterator_k, iterator_v, state, output,
        params.maximum + state_row, params.denominator + state_row, p.scale, epilogue);
  }
};
