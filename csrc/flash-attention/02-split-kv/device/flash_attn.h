#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include "cutlass/device_kernel.h"
#include "../../flash_attention.h"

template <class Kernel_>
class FlashAttn {
 public:
  using Kernel = Kernel_;
  using Arguments = typename Kernel::Params;

  static cudaError_t can_implement(Arguments const& args) {
    auto const& p = args.problem;
    auto const& t = args.tensors;
    if (p.batch_size <= 0 || p.batch_size > 65535 || p.head_number <= 0 ||
        p.head_number > 65535 || p.seq_length <= 0 || p.seq_length_kv <= 0 ||
        p.head_size != Kernel::kHeadDim || p.head_size_v != Kernel::kHeadDimV ||
        !std::isfinite(p.scale)) return cudaErrorInvalidValue;
    if (!t.query || !t.key || !t.value || !t.output) return cudaErrorInvalidValue;
    if ((reinterpret_cast<uintptr_t>(t.query) % 16) ||
        (reinterpret_cast<uintptr_t>(t.key) % 16) ||
        (reinterpret_cast<uintptr_t>(t.value) % 16) ||
        (reinterpret_cast<uintptr_t>(t.output) % 16)) return cudaErrorInvalidValue;
    return cudaSuccess;
  }

  cudaError_t initialize(Arguments const& args) {
    cudaError_t err = can_implement(args);
    if (err != cudaSuccess) return err;
    int device = 0;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;
    cudaDeviceProp props{};
    err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess) return err;
    if (props.major * 10 + props.minor != Kernel::ArchTag::kMinComputeCapability)
      return cudaErrorNotSupported;
    if (sizeof(typename Kernel::SharedStorage) > props.sharedMemPerBlockOptin)
      return cudaErrorNotSupported;
    if (sizeof(typename Kernel::SharedStorage) > 48 * 1024) {
      err = cudaFuncSetAttribute(cutlass::Kernel<Kernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(typename Kernel::SharedStorage)));
      if (err != cudaSuccess) return err;
    }
    params_ = args;
    return cudaSuccess;
  }

  cudaError_t run(cudaStream_t stream) const {
    auto const& p = params_.problem;
    dim3 grid((p.seq_length - 1) / Kernel::kBr + 1, p.head_number, p.batch_size);
    cutlass::Kernel<Kernel><<<grid, Kernel::kThreadCount, sizeof(typename Kernel::SharedStorage), stream>>>(params_);
    return cudaGetLastError();
  }

 private:
  Arguments params_{};
};
