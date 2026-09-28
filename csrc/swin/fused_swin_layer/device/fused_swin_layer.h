#pragma once
#include "cutlass/device_kernel.h"
#include "swin/fused_swin_layer/kernel/default_fused_swin_layer.h"
#include <limits>

namespace tiny_cutlass::swin::fused_swin_layer::device {

template <typename Policy>
class FusedSwinLayer {
 public:
  using Kernel = typename Policy::CutlassKernel;
  using Arguments = typename Kernel::Params;
  using Element = typename Policy::Element;
 private:
  Arguments params_{};
  int grid_ = 0;
 public:
  static cutlass::Status can_implement(Arguments const& args) {
    auto const& p = args.problem;
    if (p.batch <= 0 || p.batch > 65535 || p.height <= 0 || p.width <= 0 ||
        p.height > 1048576 || p.width > 1048576 || p.channels != 32 || p.groups != 1 ||
        p.qk_channels != 32 || p.value_channels != 32 || p.window_size != 4 || p.mlp_ratio != 4 ||
        p.shift_h < 0 || p.shift_h > 3 || p.shift_w < 0 || p.shift_w > 3 ||
        p.attn_drop != 0 || p.proj_drop != 0 || p.quant_fp8 ||
        !(p.rms_epsilon > 0 && p.rms_epsilon <= std::numeric_limits<float>::max()) ||
        !(p.mlp_rms_epsilon > 0 && p.mlp_rms_epsilon <= std::numeric_limits<float>::max()))
      return cutlass::Status::kErrorNotSupported;
    int64_t rows = int64_t(p.batch) * ((p.height + p.shift_h + 3) / 4 * 4) *
                   ((p.width + p.shift_w + 3) / 4 * 4);
    if (rows > std::numeric_limits<int>::max()) return cutlass::Status::kErrorNotSupported;
    if (!args.output || !args.gather || !args.scatter) return cutlass::Status::kErrorInvalidProblem;
    if (reinterpret_cast<uintptr_t>(args.output) % 16 ||
        reinterpret_cast<uintptr_t>(args.gather) % alignof(int) ||
        reinterpret_cast<uintptr_t>(args.scatter) % alignof(int))
      return cutlass::Status::kErrorMisalignedOperand;
    Element const* pointers[] = {args.input, args.rms_weight, args.qkv_weight, args.position_bias,
        args.output_weight, args.mlp_rms_weight, args.fc1_weight, args.fc1_bias,
        args.fc2_weight, args.fc2_bias};
    size_t counts[] = {size_t(p.batch) * p.height * p.width * 32, 32,
        size_t(p.separate_qk ? 3 : 2) * 32 * 32, 256, 32 * 32, 32, 128 * 32, 128, 32 * 128, 32};
    uintptr_t begin = reinterpret_cast<uintptr_t>(args.output);
    uintptr_t end = begin + counts[0] * sizeof(Element);
    for (int i = 0; i < 10; ++i) {
      uintptr_t address = reinterpret_cast<uintptr_t>(pointers[i]);
      if (!address) return cutlass::Status::kErrorInvalidProblem;
      if (address % 16) return cutlass::Status::kErrorMisalignedOperand;
      if (address < end && address + counts[i] * sizeof(Element) > begin)
        return cutlass::Status::kErrorInvalidProblem;
    }
    if (args.output_bias) {
      uintptr_t address = reinterpret_cast<uintptr_t>(args.output_bias);
      if (address % 16) return cutlass::Status::kErrorMisalignedOperand;
      if (address < end && address + 32 * sizeof(Element) > begin)
        return cutlass::Status::kErrorInvalidProblem;
    }
    int const* indices[] = {args.gather, args.scatter};
    for (int const* pointer : indices) {
      uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
      if (address < end && address + size_t(rows) * sizeof(int) > begin)
        return cutlass::Status::kErrorInvalidProblem;
    }
    int device;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess)
      return cutlass::Status::kErrorInternal;
    if (properties.major * 10 + properties.minor != Policy::ArchTag::kMinComputeCapability ||
        sizeof(typename Kernel::SharedStorage) > properties.sharedMemPerBlockOptin)
      return cutlass::Status::kErrorNotSupported;
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes, cutlass::Kernel<Kernel>) != cudaSuccess)
      return cutlass::Status::kErrorNotSupported;
    return cutlass::Status::kSuccess;
  }

  static size_t get_workspace_size(Arguments const&) { return 0; }

  cutlass::Status initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    grid_ = 0;
    auto status = can_implement(args);
    if (status != cutlass::Status::kSuccess) return status;
    constexpr size_t bytes = sizeof(typename Kernel::SharedStorage);
    if constexpr (bytes > 48 * 1024) {
      if (cudaFuncSetAttribute(cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)) != cudaSuccess)
        return cutlass::Status::kErrorInternal;
    }
    params_ = args;
    auto const& p = args.problem;
    grid_ = int(int64_t(p.batch) * ((p.height + p.shift_h + 3) / 4) * ((p.width + p.shift_w + 3) / 4));
    return cutlass::Status::kSuccess;
  }

  cutlass::Status run(cudaStream_t stream = nullptr) const {
    if (!grid_) return cutlass::Status::kErrorInvalidProblem;
    cutlass::Kernel<Kernel><<<grid_, Kernel::kThreadCount, sizeof(typename Kernel::SharedStorage), stream>>>(params_);
    return cudaGetLastError() == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
  }

  cutlass::Status operator()(Arguments const& args, cudaStream_t stream = nullptr) {
    auto status = initialize(args, nullptr, stream);
    return status == cutlass::Status::kSuccess ? run(stream) : status;
  }
};

}  // namespace tiny_cutlass::swin::fused_swin_layer::device
