#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

#include "catmull-rom/kernel/backward.h"

namespace tiny_cutlass::catmull_rom::device {

// CUTLASS-style host-side operator for the Catmull-Rom reprojection backward
// pass. Same Arguments / can_implement / initialize / run / operator()
// contract as the forward op. run() zeroes grad_history before launching the
// scatter kernel (the kernel accumulates into it with atomicAdd).
//
// Gradients are produced in ElementAccum (fp32 by default): grad_history is
// [N,H,W,C], grad_motion is [N,H,W,2]. The caller casts to the training dtype.
template <
    typename ArchTag = cutlass::arch::Sm80,
    typename Element = cutlass::half_t,
    typename ElementMotion = float,
    typename ElementAccum = float,
    int ThreadsPerBlock = 128>
class CatmullRomReprojectBackward {
 public:
  using KernelPolicy = kernel::DefaultCatmullRomReprojectBackward<
      ArchTag, Element, ElementMotion, ElementAccum, ThreadsPerBlock>;
  using Params = typename KernelPolicy::Params;

  struct Arguments {
    Element const* history = nullptr;        // [N, H, W, C]
    ElementMotion const* motion = nullptr;   // [N, H, W, 2]
    Element const* grad_output = nullptr;    // [N, H, W, C]
    ElementAccum* grad_history = nullptr;    // [N, H, W, C]
    ElementAccum* grad_motion = nullptr;     // [N, H, W, 2]
    int N = 0;
    int H = 0;
    int W = 0;
    int C = 0;
  };

 private:
  Params params_;

 public:
  CatmullRomReprojectBackward() = default;

  static cutlass::Status can_implement(Arguments const& args) {
    if (args.N <= 0 || args.H <= 0 || args.W <= 0 || args.C <= 0) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (!args.history || !args.motion || !args.grad_output || !args.grad_history ||
        !args.grad_motion) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    return cutlass::Status::kSuccess;
  }

  static size_t get_workspace_size(Arguments const& /*args*/) { return 0; }

  cutlass::Status initialize(
      Arguments const& args,
      void* /*workspace*/ = nullptr,
      cudaStream_t /*stream*/ = nullptr) {
    cutlass::Status status = can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    params_.history = args.history;
    params_.motion = args.motion;
    params_.grad_output = args.grad_output;
    params_.grad_history = args.grad_history;
    params_.grad_motion = args.grad_motion;
    params_.N = args.N;
    params_.H = args.H;
    params_.W = args.W;
    params_.C = args.C;
    return cutlass::Status::kSuccess;
  }

  cutlass::Status run(cudaStream_t stream = nullptr) {
    using ElementAccumT = typename Params::ElementAccum;
    int64_t const grad_history_count =
        static_cast<int64_t>(params_.N) * params_.H * params_.W * params_.C;

    // grad_history is accumulated via atomicAdd, so it must start at zero.
    cudaError_t error = cudaMemsetAsync(
        params_.grad_history, 0,
        sizeof(ElementAccumT) * static_cast<size_t>(grad_history_count), stream);
    if (error != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    KernelPolicy::invoke(params_, stream);
    error = cudaGetLastError();
    return error == cudaSuccess ? cutlass::Status::kSuccess
                                : cutlass::Status::kErrorInternal;
  }

  cutlass::Status operator()(cudaStream_t stream = nullptr) { return run(stream); }

  cutlass::Status operator()(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr) {
    cutlass::Status status = initialize(args, workspace, stream);
    if (status == cutlass::Status::kSuccess) {
      status = run(stream);
    }
    return status;
  }
};

}  // namespace tiny_cutlass::catmull_rom::device

