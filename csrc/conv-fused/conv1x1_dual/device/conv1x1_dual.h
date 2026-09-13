#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/arch/arch.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/half.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_ref.h"

#include "conv1x1_dual/device/b2b_implicit_gemm_convolution.h"
#include "conv1x1_dual/kernel/conv1x1_dual.h"

namespace tiny_cutlass::conv_fused::device {

// Device-level driver for the fused conv1x1 -> ReLU -> conv1x1 family.
//
// The residency knob (RF vs SMEM) is a compile-time template parameter: it
// selects a different underlying kernel through DefaultConv1x1Dual, but the
// host-facing Arguments and call sequence are identical for both.
template <
    typename ArchTag = cutlass::arch::Sm80,
    typename Element = cutlass::half_t,
    kernel::Residency ResidencyKind = kernel::Residency::kRF>
class Conv1x1Dual {
 public:
  static kernel::Residency const kResidency = ResidencyKind;

  using KernelConfig = kernel::DefaultConv1x1Dual<ArchTag, Element, ResidencyKind>;
  using Operation = cutlass::conv::device::B2bImplicitGemmConvolution<
      typename KernelConfig::CutlassKernel>;
  using CutlassArguments = typename Operation::Arguments;

  struct Arguments {
    cutlass::conv::Conv2dProblemSize problem_size_0;
    cutlass::conv::Conv2dProblemSize problem_size_1;
    Element const* input = nullptr;
    Element const* weight0 = nullptr;
    Element const* bias0 = nullptr;
    Element const* weight1 = nullptr;
    Element const* bias1 = nullptr;
    Element* output = nullptr;
  };

 private:
  Operation operation_;
  CutlassArguments cutlass_args_;

 public:
  Conv1x1Dual() = default;

  static cutlass::Status can_implement(Arguments const& args);

  static size_t get_workspace_size(Arguments const& args);

  cutlass::Status initialize(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr);

  cutlass::Status run(cudaStream_t stream = nullptr);

  cutlass::Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

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

}  // namespace tiny_cutlass::conv_fused::device
