#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "../../flash_attention.h"

// cutlass::Kernel uses extern shared memory. This entry preserves its operator
// contract while enforcing this experiment's static, <=48 KiB allocation.
template <class Operator>
__global__ void flash_attn_entry(typename Operator::Params params) {
  __shared__ typename Operator::SharedStorage storage;
  Operator op;
  op(params, storage);
}

template <class Kernel_>
class FlashAttn {
 public:
  using Kernel = Kernel_;
  using Params = typename Kernel::Params;
  using SharedStorage = typename Kernel::SharedStorage;
  using ThreadblockShape0 = typename Kernel::ThreadblockShape0;
  using ThreadblockShape1 = typename Kernel::ThreadblockShape1;
  using ThreadblockSwizzle = typename Kernel::ThreadblockSwizzle;
  struct Arguments {
    Problem problem;
    Tensors tensors;
  };
  static constexpr std::size_t kSharedMemoryBudgetBytes = 48 * 1024;
  static constexpr std::size_t kSharedStorageBytes = sizeof(SharedStorage); // Q/K/V only
  static_assert(kSharedStorageBytes <= kSharedMemoryBudgetBytes,
      "The selected attention policy exceeds the static shared-memory budget");
  static_assert(ThreadblockShape0::kM == ThreadblockShape1::kM &&
                ThreadblockShape0::kN <= ThreadblockShape1::kK &&
                ThreadblockShape1::kN == Kernel::kHeadDimV,
      "QK and PV tile shapes must share Q rows and the KV reduction axis");

  static std::size_t get_workspace_size(Arguments const& args) {
    auto const& p = args.problem;
    if (p.batch_size <= 0 || p.head_number <= 0 || p.seq_length <= 0 ||
        p.head_size_v != Kernel::kHeadDimV) return 0;
    std::size_t rows_per_head =
        (std::size_t(p.seq_length - 1) / Kernel::kBr + 1) * Kernel::kBr;
    std::size_t heads = std::size_t(p.batch_size) * p.head_number;
    if (heads > std::numeric_limits<std::size_t>::max() / rows_per_head)
      return 0;
    std::size_t state_rows = heads * rows_per_head;
    std::size_t elements_per_row = std::size_t(Kernel::kHeadDimV) + 2;
    if (state_rows > std::numeric_limits<std::size_t>::max() /
                         elements_per_row / sizeof(float)) return 0;
    return state_rows * elements_per_row * sizeof(float);
  }

  static cutlass::Status can_implement(Arguments const& args) {
    return Kernel::can_implement(args.problem, args.tensors);
  }

  cutlass::Status initialize(Arguments const& args, Workspace workspace) {
    params_ = Params{};
    cutlass::Status status = can_implement(args);
    if (status != cutlass::Status::kSuccess) return status;

    std::size_t required = get_workspace_size(args);
    if (!required || workspace.bytes < required)
      return cutlass::Status::kErrorInvalidProblem;
    if (!workspace.data) return cutlass::Status::kErrorWorkspaceNull;
    if (reinterpret_cast<uintptr_t>(workspace.data) % 16)
      return cutlass::Status::kErrorMisalignedOperand;

    // FA1's KV-outer/Q-inner scan stays within one CTA per batch/head.
    // The QK and PV GEMM shapes describe the work *inside* that CTA; unlike
    // example 13, tiling the full QK problem would change CTA ownership.
    cutlass::gemm::GemmCoord grid_shape(args.problem.batch_size,
                                       args.problem.head_number, 1);
    using Layout = typename Kernel::Layout;
    auto const& p = args.problem;
    auto const& t = args.tensors;
    params_ = Params(p, grid_shape,
        typename Kernel::TensorRefQ(t.query, Layout(int64_t(p.head_number) * p.head_size)),
        typename Kernel::TensorRefK(t.key, Layout(int64_t(p.head_number) * p.head_size)),
        typename Kernel::TensorRefV(t.value, Layout(int64_t(p.head_number) * p.head_size_v)),
        typename Kernel::TensorRefO(t.output, Layout(int64_t(p.head_number) * p.head_size_v)),
        workspace.data);
    return cutlass::Status::kSuccess;
  }

  cutlass::Status run(cudaStream_t stream = nullptr) const {
    if (params_.grid_tiled_shape.m() <= 0 || params_.grid_tiled_shape.n() <= 0)
      return cutlass::Status::kErrorInvalidProblem;
    dim3 grid = ThreadblockSwizzle::get_grid_shape(params_.grid_tiled_shape);
    flash_attn_entry<Kernel><<<grid, Kernel::kThreadCount, 0, stream>>>(params_);
    return cudaGetLastError() == cudaSuccess
        ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
  }

  cutlass::Status operator()(cudaStream_t stream = nullptr) const { return run(stream); }

  cutlass::Status operator()(Arguments const& args, Workspace workspace,
                             cudaStream_t stream = nullptr) {
    cutlass::Status status = initialize(args, workspace);
    return status == cutlass::Status::kSuccess ? run(stream) : status;
  }

 private:
  Params params_{};
};
