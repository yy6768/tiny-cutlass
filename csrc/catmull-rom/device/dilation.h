#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

#include "catmull-rom/kernel/dilation.h"

namespace tiny_cutlass::catmull_rom::device {

// CUTLASS-style host-side operator for depth-based motion dilation. Same
// Arguments / can_implement / initialize / run / operator() contract as the
// reprojection ops. See catmull-rom/kernel/dilation.h for the algorithm.
template <
    typename ArchTag = cutlass::arch::Sm80,
    typename ElementDepth = float,
    typename ElementMotion = float,
    bool NearerIsGreater = true,
    int ThreadsPerBlock = 256>
class MotionDilation {
 public:
  using KernelPolicy = kernel::DefaultMotionDilation<
      ArchTag, ElementDepth, ElementMotion, NearerIsGreater, ThreadsPerBlock>;
  using Params = typename KernelPolicy::Params;

  struct Arguments {
    ElementDepth const* depth = nullptr;    // [N, H, W]
    ElementMotion const* motion = nullptr;  // [N, H, W, 2]
    ElementMotion* output = nullptr;        // [N, H, W, 2]
    int N = 0;
    int H = 0;
    int W = 0;
  };

 private:
  Params params_;

 public:
  MotionDilation() = default;

  static cutlass::Status can_implement(Arguments const& args) {
    if (args.N <= 0 || args.H <= 0 || args.W <= 0) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (!args.depth || !args.motion || !args.output) {
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
    params_.depth = args.depth;
    params_.motion = args.motion;
    params_.output = args.output;
    params_.N = args.N;
    params_.H = args.H;
    params_.W = args.W;
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

// CUTLASS-style host-side operator for the fused depth-dilation + Catmull-Rom
// reprojection. Same Arguments / can_implement / initialize / run / operator()
// contract as the other ops. See
// catmull-rom/kernel/dilation.h for the algorithm.
template <
    typename ArchTag = cutlass::arch::Sm80,
    typename Element = cutlass::half_t,
    typename ElementDepth = float,
    typename ElementMotion = float,
    bool NearerIsGreater = true,
    int ThreadsPerBlock = 128>
class DilatedCatmullReproject {
 public:
  using KernelPolicy = kernel::DefaultDilatedCatmullReproject<
      ArchTag, Element, ElementDepth, ElementMotion, NearerIsGreater, ThreadsPerBlock>;
  using Params = typename KernelPolicy::Params;

  struct Arguments {
    Element const* history = nullptr;       // [N, H, W, C]
    ElementDepth const* depth = nullptr;    // [N, H, W]
    ElementMotion const* motion = nullptr;  // [N, H, W, 2]
    Element* output = nullptr;              // [N, H, W, C]
    int N = 0;
    int H = 0;
    int W = 0;
    int C = 0;
  };

 private:
  Params params_;

 public:
  DilatedCatmullReproject() = default;

  static cutlass::Status can_implement(Arguments const& args) {
    if (args.N <= 0 || args.H <= 0 || args.W <= 0 || args.C <= 0) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (!args.history || !args.depth || !args.motion || !args.output) {
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
    params_.depth = args.depth;
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

