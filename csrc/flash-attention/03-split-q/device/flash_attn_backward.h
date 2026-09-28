#pragma once

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <limits>
#include "cutlass/device_kernel.h"
#include "../kernel/flash_attn_backward_prepare.h"

template <class Kernel_>
class FlashAttnBackward {
 public:
  using Kernel = Kernel_;
  using Mma = typename Kernel::Mma;
  using Element = typename Mma::Element;
  using Params = typename Kernel::Params;
  using SharedStorage = typename Kernel::SharedStorage;
  using DeltaKernel = FlashAttnBackwardDelta<Element, 8, 8>;
  using DeltaParams = typename DeltaKernel::Params;
  using ConvertKernel = FlashAttnBackwardConvert<Element>;
  using ConvertParams = typename ConvertKernel::Params;
  struct Arguments {
    Problem problem;
    BackwardTensors tensors;
  };

  static bool valid_problem(Problem const& p) {
    return p.batch_size > 0 && p.batch_size <= 65535 &&
        p.head_number > 0 && p.head_number <= 65535 &&
        p.seq_length > 0 && p.seq_length <= INT_MAX - (Kernel::kBr - 1) &&
        p.seq_length_kv > 0 && p.head_size == Kernel::kHeadDim &&
        p.head_size_v == Kernel::kHeadDim && std::isfinite(p.scale);
  }

  static std::size_t delta_bytes(Problem const& p) {
    uint64_t rows = uint64_t(p.batch_size) * p.head_number * p.seq_length;
    return std::size_t((rows * sizeof(float) + 15) & ~uint64_t(15));
  }

  static std::size_t get_workspace_size(Problem const& p) {
    if (!valid_problem(p)) return 0;
    uint64_t rows = uint64_t(p.batch_size) * p.head_number * p.seq_length;
    uint64_t limit = uint64_t((std::numeric_limits<std::size_t>::max)());
    uint64_t signed_limit = uint64_t((std::numeric_limits<int64_t>::max)());
    if (rows > (limit - 15) / (sizeof(float) * uint64_t(p.head_size + 1)) ||
        rows > signed_limit / p.head_size) return 0;
    uint64_t key_rows = uint64_t(p.batch_size) * p.head_number * p.seq_length_kv;
    if (key_rows > signed_limit / p.head_size ||
        key_rows > limit / (sizeof(Element) * uint64_t(p.head_size))) return 0;
    return delta_bytes(p) + std::size_t(rows * p.head_size * sizeof(float));
  }

  static cudaError_t can_implement(Arguments const& args, Workspace workspace) {
    std::size_t required = get_workspace_size(args.problem);
    if (!required) return cudaErrorInvalidValue;
    auto const& t = args.tensors;
    void const* pointers[] = {t.query, t.key, t.value, t.output, t.grad_output,
                            t.grad_query, t.grad_key, t.grad_value};
    for (void const* pointer : pointers)
      if (!pointer || reinterpret_cast<uintptr_t>(pointer) % 16)
        return cudaErrorInvalidValue;
    if (!t.logsumexp || reinterpret_cast<uintptr_t>(t.logsumexp) % alignof(float))
      return cudaErrorInvalidValue;
    if (!workspace.data || workspace.bytes < required ||
        reinterpret_cast<uintptr_t>(workspace.data) % 16) return cudaErrorInvalidValue;
    if (t.grad_query == t.grad_key || t.grad_query == t.grad_value ||
        t.grad_key == t.grad_value) return cudaErrorInvalidValue;
    return cudaSuccess;
  }

  cudaError_t initialize(Arguments const& args, Workspace workspace) {
    initialized_ = false;
    cudaError_t err = can_implement(args, workspace);
    if (err != cudaSuccess) return err;
    int device = 0;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;
    cudaDeviceProp props{};
    err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess) return err;
    if (props.major * 10 + props.minor != Kernel::ArchTag::kMinComputeCapability ||
        sizeof(SharedStorage) > std::size_t(props.sharedMemPerBlockOptin))
      return cudaErrorNotSupported;
    err = cudaFuncSetAttribute(cutlass::Kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(SharedStorage)));
    if (err != cudaSuccess) return err;

    auto const& p = args.problem;
    auto const& t = args.tensors;
    auto* delta = static_cast<float*>(workspace.data);
    grad_query_accum_ = reinterpret_cast<float*>(
        static_cast<unsigned char*>(workspace.data) + delta_bytes(p));
    int64_t rows = int64_t(p.batch_size) * p.head_number * p.seq_length;
    int64_t elements = rows * p.head_size;
    params_ = Params(p, t, delta, grad_query_accum_);
    delta_params_ = DeltaParams{p, t.output, t.grad_output, delta, rows};
    convert_params_ = ConvertParams{grad_query_accum_, t.grad_query, elements};
    initialized_ = true;
    return cudaSuccess;
  }

  cudaError_t run(cudaStream_t stream = nullptr) const {
    if (!initialized_) return cudaErrorInvalidValue;
    auto const& p = params_.problem;
    int64_t rows = delta_params_.rows;
    int64_t elements = convert_params_.elements;
    cudaError_t err = cudaMemsetAsync(
        grad_query_accum_, 0, std::size_t(elements) * sizeof(float), stream);
    if (err != cudaSuccess) return err;

    unsigned delta_blocks = unsigned((std::min)(int64_t(65535),
        (rows - 1) / DeltaKernel::kRowsPerBlock + 1));
    cutlass::Kernel<DeltaKernel><<<delta_blocks, DeltaKernel::kThreadCount, 0, stream>>>(delta_params_);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    dim3 grid((p.seq_length_kv - 1) / Kernel::kBc + 1, p.head_number, p.batch_size);
    cutlass::Kernel<Kernel><<<grid, Kernel::kThreadCount, sizeof(SharedStorage), stream>>>(params_);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    unsigned convert_blocks = unsigned((std::min)(int64_t(65535),
        (elements - 1) / ConvertKernel::kThreadCount + 1));
    cutlass::Kernel<ConvertKernel><<<convert_blocks, ConvertKernel::kThreadCount, 0, stream>>>(convert_params_);
    return cudaGetLastError();
  }

  cudaError_t operator()(cudaStream_t stream = nullptr) const { return run(stream); }

  cudaError_t operator()(Arguments const& args, Workspace workspace, cudaStream_t stream = nullptr) {
    cudaError_t err = initialize(args, workspace);
    return err == cudaSuccess ? run(stream) : err;
  }

 private:
  Params params_{};
  DeltaParams delta_params_{};
  ConvertParams convert_params_{};
  float* grad_query_accum_ = nullptr;
  bool initialized_ = false;
};
