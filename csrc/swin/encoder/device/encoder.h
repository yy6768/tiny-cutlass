#pragma once

#include <cuda_runtime.h>
#include "cutlass/device_kernel.h"
#include "swin/encoder/kernel/default_encoder.h"

namespace tiny_cutlass::swin::encoder::device {

template <typename Policy>
class Encoder {
 public:
  using Element = typename Policy::Element;
  using Kernel = typename Policy::CutlassKernel;
  using Params = typename Kernel::Params;
  using SharedStorage = typename Kernel::SharedStorage;

  static cutlass::Status can_implement(Params const& a) {
    auto const& p = a.problem;
    if (p.tiles < 1 || p.channels != 32 || p.query_tokens != 64 ||
        p.key_tokens != 96 || p.heads != 2 || p.hidden_channels != 128)
      return cutlass::Status::kErrorNotSupported;
    auto valid = [](void const* ptr) { return ptr && (reinterpret_cast<uintptr_t>(ptr) % 16 == 0); };
    if (!valid(a.input) || !valid(a.output) || !valid(a.merged) || !valid(a.head))
      return cutlass::Status::kErrorInvalidProblem;
    auto const& w = a.weights;
    for (int i = 0; i < 2; ++i)
      if (!valid(w.input_weight[i]) || !valid(w.input_bias[i]) || !valid(w.norm_weight[i]) ||
          !valid(w.qk_weight[i]) || !valid(w.value_weight[i]) || !valid(w.position_bias[i]) ||
          !valid(w.projection_weight[i])) return cutlass::Status::kErrorInvalidProblem;
    for (int i = 0; i < 4; ++i)
      if (!valid(w.expand_weight[i]) || !valid(w.expand_bias[i]) || !valid(w.contract_weight[i]))
        return cutlass::Status::kErrorInvalidProblem;
    if (!valid(w.projection_bias) || !valid(w.contract_bias) || !valid(w.merge_weight) ||
        !valid(w.merge_bias) || !valid(w.head_weight) || !valid(w.head_bias))
      return cutlass::Status::kErrorInvalidProblem;
    int device;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess)
      return cutlass::Status::kErrorInternal;
    if ((prop.major * 10 + prop.minor != 80 && prop.major * 10 + prop.minor != 89) ||
        int(sizeof(SharedStorage)) > prop.sharedMemPerBlockOptin)
      return cutlass::Status::kErrorNotSupported;
    return cutlass::Status::kSuccess;
  }

  static size_t get_workspace_size(Params const&) { return 0; }

  cutlass::Status initialize(Params const& args, void* = nullptr, cudaStream_t = nullptr) {
    auto status = can_implement(args);
    if (status != cutlass::Status::kSuccess) return status;
    if (cudaFuncSetAttribute(cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             int(sizeof(SharedStorage))) != cudaSuccess)
      return cutlass::Status::kErrorInternal;
    params_ = args;
    initialized_ = true;
    return cutlass::Status::kSuccess;
  }

  cutlass::Status run(cudaStream_t stream = nullptr) const {
    if (!initialized_) return cutlass::Status::kErrorInvalidProblem;
    cutlass::Kernel<Kernel><<<params_.problem.tiles, Kernel::kThreadCount, sizeof(SharedStorage), stream>>>(params_);
    return cudaGetLastError() == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
  }

  cutlass::Status operator()(cudaStream_t stream = nullptr) const { return run(stream); }

 private:
  Params params_{};
  bool initialized_ = false;
};

}  // namespace tiny_cutlass::swin::encoder::device
