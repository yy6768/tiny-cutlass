#pragma once

#include <cstdint>
#include "cutlass/device_kernel.h"
#include "swin/window_attention/kernel/default_window_attention.h"

namespace tiny_cutlass::swin::window_attention::device {

template <typename Policy>
struct WindowAttention {
  using Kernel = typename Policy::CutlassKernel;
  using Arguments = typename Kernel::Params;

  static size_t shared_size(Arguments const& a) {
    return sizeof(typename Kernel::SharedStorage) + 16 * (a.problem.hidden_channels() + a.problem.channels) * sizeof(typename Policy::Element);
  }

  static cutlass::Status can_implement(Arguments const& a) {
    if (!a.problem.valid() || a.problem.qk_channels > Policy::Mma::kMaxQk ||
        a.problem.value_channels > Policy::Mma::kMaxValue) return cutlass::Status::kErrorNotSupported;
    if (!a.input || !a.rms_weight || !a.qkv_weight || !a.position_bias || !a.output_weight ||
        !a.gather || !a.scatter || !a.output || a.output == a.input ||
        a.output == a.rms_weight || a.output == a.qkv_weight || a.output == a.output_weight ||
        a.output == a.position_bias || a.output == a.output_bias)
      return cutlass::Status::kErrorInvalidProblem;
    typename Policy::Element const* pointers[] = {
        a.input, a.rms_weight, a.qkv_weight, a.position_bias, a.output_weight, a.output_bias, a.output};
    for (auto ptr : pointers)
      if (reinterpret_cast<uintptr_t>(ptr) % alignof(typename Policy::Element))
        return cutlass::Status::kErrorMisalignedOperand;
    if (reinterpret_cast<uintptr_t>(a.gather) % alignof(int) ||
        reinterpret_cast<uintptr_t>(a.scatter) % alignof(int)) return cutlass::Status::kErrorMisalignedOperand;
    int device;
    cudaDeviceProp properties;
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess)
      return cutlass::Status::kErrorInternal;
    int capability = properties.major * 10 + properties.minor;
    if ((capability != 80 && capability != 86 && capability != 89) ||
        capability < Policy::ArchTag::kMinComputeCapability ||
        shared_size(a) > properties.sharedMemPerBlockOptin)
      return cutlass::Status::kErrorNotSupported;
    // The active device may satisfy the policy but lack an image in this build
    // (for example an SM89-only binary loaded on SM80).
    cudaFuncAttributes attributes;
    if (cudaFuncGetAttributes(&attributes, cutlass::Kernel<Kernel>) != cudaSuccess)
      return cutlass::Status::kErrorNotSupported;
    return cutlass::Status::kSuccess;
  }

  cutlass::Status operator()(Arguments const& a, cudaStream_t stream) const {
    auto status = can_implement(a);
    if (status != cutlass::Status::kSuccess) return status;
    size_t bytes = shared_size(a);
    if (bytes > 48 * 1024 && cudaFuncSetAttribute(cutlass::Kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)) != cudaSuccess)
      return cutlass::Status::kErrorInternal;
    cutlass::Kernel<Kernel><<<a.problem.windows(), Kernel::kThreadCount, bytes, stream>>>(a);
    return cudaGetLastError() == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
  }
};

}  // namespace tiny_cutlass::swin::window_attention::device
