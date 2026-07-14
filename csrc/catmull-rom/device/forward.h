#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

#include "catmull-rom/kernel/forward.h"

namespace tiny_cutlass::catmull_rom::device {

// CUTLASS-style host-side operator for Catmull-Rom reprojection.
//
// This op is a gather/resampling kernel, not an implicit GEMM, so there is no
// DefaultXxx MMA factory to wrap -- the algorithm lives in the kernel layer
// (catmull-rom/kernel/forward.h). The device layer keeps the
// same contract every operator in this project exposes: Arguments,
// can_implement, initialize, run, operator(). It builds the kernel Params from
// the raw pointers and launches through the kernel-layer policy factory.
template <
    typename ArchTag = cutlass::arch::Sm80,
    typename Element = cutlass::half_t,
    typename ElementMotion = float,
    int ThreadsPerBlock = 128>
class CatmullRomReproject {
 public:
  using KernelPolicy =
      kernel::DefaultCatmullRomReproject<ArchTag, Element, ElementMotion, ThreadsPerBlock>;
  using Params = typename KernelPolicy::Params;

  struct Arguments {
    Element const* history = nullptr;       // [N, H, W, C]
    ElementMotion const* motion = nullptr;  // [N, H, W, 2] (dx, dy) in pixels
    Element* output = nullptr;              // [N, H, W, C]
    int N = 0;
    int H = 0;
    int W = 0;
    int C = 0;
  };

 private:
  Params params_;

 public:
  CatmullRomReproject() = default;

  static cutlass::Status can_implement(Arguments const& args) {
    if (args.N <= 0 || args.H <= 0 || args.W <= 0 || args.C <= 0) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (!args.history || !args.motion || !args.output) {
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
    params_.output = args.output;
    params_.N = args.N;
    params_.H = args.H;
    params_.W = args.W;
    params_.C = args.C;
    return cutlass::Status::kSuccess;
  }

  cutlass::Status run(cudaStream_t stream = nullptr) {
    KernelPolicy::invoke(params_, stream);
    cudaError_t error = cudaGetLastError();
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

