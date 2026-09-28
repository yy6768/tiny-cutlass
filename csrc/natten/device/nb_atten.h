#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include "natten/nb_atten.h"

namespace tiny_cutlass { namespace natten {

// CUTLASS 2.x device operator: Arguments are the public launch contract;
// Kernel::Params is the precomputed device contract. run() alone launches.
template <class Kernel_>
class NBAtten {
 public:
  using Kernel = Kernel_;
  using Element = typename Kernel::Element;
  using Params = typename Kernel::Params;
  using SharedStorage = typename Kernel::SharedStorage;

  struct Arguments {
    Element const* query = nullptr;
    Element const* key = nullptr;
    Element const* value = nullptr;
    Element* output = nullptr;
    float* logsumexp = nullptr; // optional
    NeighborhoodProblem problem;
  };

  static size_t get_workspace_size(NeighborhoodProblem const&) { return 0; }

  static size_t get_lse_size(NeighborhoodProblem const& p) {
    if (p.batch_size <= 0 || p.heads <= 0 || p.length <= 0) return 0;
    return size_t(p.batch_size) * p.heads *
        ((int64_t(p.length) + 31) / 32 * 32);
  }

  static cutlass::Status can_implement(Arguments const& a) {
    auto const& p = a.problem;
    if (p.batch_size <= 0 || p.batch_size > 65535 ||
        p.heads <= 0 || p.heads > 65535 || p.length <= 0 ||
        p.length > (1 << 20) || p.kernel_size <= 0 ||
        p.kernel_size > p.length || p.stride != 1 || p.dilation != 1 ||
        p.head_dim <= 0 || p.head_dim > 64 || p.head_dim % 8 ||
        p.head_dim_value <= 0 || p.head_dim_value > 64 ||
        p.head_dim_value % 8 || !std::isfinite(p.scale) ||
        int64_t(p.batch_size) * p.heads * ((p.length + 31) / 32 * 32) >
            std::numeric_limits<int32_t>::max())
      return cutlass::Status::kErrorNotSupported;
    if (!a.query || !a.key || !a.value || !a.output)
      return cutlass::Status::kErrorInvalidProblem;
    if (reinterpret_cast<uintptr_t>(a.query) % 16 ||
        reinterpret_cast<uintptr_t>(a.key) % 16 ||
        reinterpret_cast<uintptr_t>(a.value) % 16 ||
        reinterpret_cast<uintptr_t>(a.output) % 16 ||
        reinterpret_cast<uintptr_t>(a.logsumexp) % alignof(float))
      return cutlass::Status::kErrorMisalignedOperand;
    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess)
      return cutlass::Status::kErrorInternal;
    int cc = properties.major * 10 + properties.minor;
    if (cc != 80 && cc != 89) return cutlass::Status::kErrorArchMismatch;
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes, cutlass::Kernel<Kernel>) != cudaSuccess) {
      cudaGetLastError();
      return cutlass::Status::kErrorArchMismatch;
    }
    if (sizeof(SharedStorage) > properties.sharedMemPerBlock)
      return cutlass::Status::kErrorNotSupported;
    return cutlass::Status::kSuccess;
  }

  cutlass::Status initialize(Arguments const& args) {
    ready_ = false;
    auto status = can_implement(args);
    if (status != cutlass::Status::kSuccess) return status;
    params_ = Params(args.problem, args.query, args.key, args.value,
                     args.output, args.logsumexp);
    ready_ = true;
    return cutlass::Status::kSuccess;
  }

  cutlass::Status run(cudaStream_t stream = nullptr) const {
    if (!ready_) return cutlass::Status::kErrorInvalidProblem;
    dim3 grid((params_.problem.length + Kernel::kBr - 1) / Kernel::kBr,
              params_.problem.heads, params_.problem.batch_size);
    cutlass::Kernel<Kernel><<<grid, Kernel::kThreads, sizeof(SharedStorage), stream>>>(params_);
    return cudaGetLastError() == cudaSuccess ? cutlass::Status::kSuccess
                                            : cutlass::Status::kErrorInternal;
  }

  cutlass::Status operator()(Arguments const& args, cudaStream_t stream = nullptr) {
    auto status = initialize(args);
    return status == cutlass::Status::kSuccess ? run(stream) : status;
  }

 private:
  Params params_{};
  bool ready_ = false;
};

}} // namespace tiny_cutlass::natten
