#pragma once

#include <cmath>
#include <climits>
#include <cstdint>
#include "../../flash_attention.h"

#include "cutlass/device_kernel.h"

template <class Kernel_>
class FlashAttnSplitQ {
 public:
  using Kernel = Kernel_;
  using Params = typename Kernel::Params;
  using SharedStorage = typename Kernel::SharedStorage;
  struct Arguments {
    Problem problem;
    Tensors tensors;
  };
  static constexpr std::size_t kSharedMemoryBytes = 48 * 1024;
  static_assert(sizeof(SharedStorage) <= kSharedMemoryBytes,
      "The selected attention policy exceeds the 48 KiB shared-memory budget");

  static std::size_t get_workspace_size(Arguments const&) { return 0; }

  static cudaError_t can_implement(Arguments const& args) {
    auto const& p = args.problem;
    auto const& t = args.tensors;
    if (p.batch_size <= 0 || p.batch_size > 65535 || p.head_number <= 0 ||
        p.head_number > 65535 || p.seq_length <= 0 || p.seq_length_kv <= 0 ||
        p.seq_length_kv > INT_MAX - (Kernel::kBc - 1) ||
        p.head_size != Kernel::kHeadDim || p.head_size_v != Kernel::kHeadDimV ||
        !std::isfinite(p.scale)) return cudaErrorInvalidValue;
    if (!t.query || !t.key || !t.value || !t.output) return cudaErrorInvalidValue;
    if ((reinterpret_cast<uintptr_t>(t.query) % 16) ||
        (reinterpret_cast<uintptr_t>(t.key) % 16) ||
        (reinterpret_cast<uintptr_t>(t.value) % 16) ||
        (reinterpret_cast<uintptr_t>(t.output) % 16) ||
        (t.logsumexp && reinterpret_cast<uintptr_t>(t.logsumexp) % alignof(float)))
      return cudaErrorInvalidValue;
    return cudaSuccess;
  }

  cudaError_t initialize(Arguments const& args, Workspace) {
    initialized_ = false;
    cudaError_t err = can_implement(args);
    if (err != cudaSuccess) return err;
    int device = 0;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;
    cudaDeviceProp props{};
    err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess) return err;
    if (props.major * 10 + props.minor != Kernel::ArchTag::kMinComputeCapability ||
        sizeof(SharedStorage) > props.sharedMemPerBlock)
      return cudaErrorNotSupported;
    params_ = Params(args.problem, args.tensors);
    initialized_ = true;
    return cudaSuccess;
  }

  cudaError_t run(cudaStream_t stream = nullptr) const {
    if (!initialized_) return cudaErrorInvalidValue;
    auto const& p = params_.problem;
    dim3 grid(params_.q_tile_count, p.head_number, p.batch_size);
    cutlass::Kernel<Kernel><<<grid, Kernel::kThreadCount, sizeof(SharedStorage), stream>>>(params_);
    return cudaGetLastError();
  }

  cudaError_t operator()(cudaStream_t stream = nullptr) const { return run(stream); }

  cudaError_t operator()(Arguments const& args, Workspace workspace, cudaStream_t stream = nullptr) {
    cudaError_t err = initialize(args, workspace);
    return err == cudaSuccess ? run(stream) : err;
  }

 private:
  Params params_{};
  bool initialized_ = false;
};
