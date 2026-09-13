#pragma once
#include "swin/window_attention/device/window_attention.h"
#include "swin/window_attention/kernel/default_swin_block.h"

namespace tiny_cutlass::swin::window_attention::device {

template <typename Policy>
struct SwinBlock {
  using Kernel = typename Policy::CutlassKernel;
  using Arguments = typename Kernel::Params;
  static cutlass::Status can_implement(Arguments const& a) {
    if (a.problem.channels != Policy::kChannels || a.problem.groups != 1 ||
        a.problem.qk_channels != Policy::kChannels || a.problem.value_channels != Policy::kChannels ||
        a.mlp_ratio != Policy::kRatio || !(a.mlp_rms_epsilon > 0 && a.mlp_rms_epsilon <= std::numeric_limits<float>::max()))
      return cutlass::Status::kErrorNotSupported;
    if constexpr (Policy::Mma::AttentionMma::kAccess == 8) {
      typename Policy::Element const* vector_pointers[] = {a.input,a.output,a.rms_weight,a.qkv_weight,a.position_bias,
          a.output_weight,a.output_bias,a.mlp_rms_weight,a.fc1_weight,a.fc1_bias,a.fc2_weight,a.fc2_bias};
      for (auto ptr : vector_pointers)
        if (reinterpret_cast<uintptr_t>(ptr) % 16) return cutlass::Status::kErrorMisalignedOperand;
    }
    auto status = WindowAttention<typename Policy::Attention>::can_implement(a);
    if (status != cutlass::Status::kSuccess) return status;
    typename Policy::Element const* pointers[] = {a.mlp_rms_weight, a.fc1_weight, a.fc1_bias, a.fc2_weight, a.fc2_bias};
    for (auto ptr : pointers) {
      if (!ptr || ptr == a.output) return cutlass::Status::kErrorInvalidProblem;
      if (reinterpret_cast<uintptr_t>(ptr) % alignof(typename Policy::Element))
        return cutlass::Status::kErrorMisalignedOperand;
    }
    int device; cudaDeviceProp properties;
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess)
      return cutlass::Status::kErrorInternal;
    if (sizeof(typename Kernel::SharedStorage) > properties.sharedMemPerBlockOptin)
      return cutlass::Status::kErrorNotSupported;
    cudaFuncAttributes attributes;
    return cudaFuncGetAttributes(&attributes, cutlass::Kernel<Kernel>) == cudaSuccess
        ? cutlass::Status::kSuccess : cutlass::Status::kErrorNotSupported;
  }
  cutlass::Status operator()(Arguments const& a, cudaStream_t stream) const {
    auto status = can_implement(a);
    if (status != cutlass::Status::kSuccess) return status;
    constexpr size_t bytes = sizeof(typename Kernel::SharedStorage);
    if constexpr (bytes > 48 * 1024) {
      if (cudaFuncSetAttribute(cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)) != cudaSuccess)
        return cutlass::Status::kErrorInternal;
    }
    cutlass::Kernel<Kernel><<<a.problem.windows(), Kernel::kThreadCount, bytes, stream>>>(a);
    return cudaGetLastError() == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
  }
};

}  // namespace tiny_cutlass::swin::window_attention::device
