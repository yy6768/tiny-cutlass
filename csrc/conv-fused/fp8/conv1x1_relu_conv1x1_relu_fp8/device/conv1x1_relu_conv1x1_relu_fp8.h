#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/arch/arch.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/numeric_types.h"

#include "device/b2b_implicit_gemm_convolution.h"
#include "fp8/conv1x1_relu_conv1x1_relu_fp8/kernel/conv1x1_relu_conv1x1_relu_fp8.h"

namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu::device {

template <
    typename ArchTag = cutlass::arch::Sm89,
    typename ElementA = cutlass::float_e4m3_t,
    typename ElementB = ElementA,
    typename ElementC = ElementA,
    typename ElementScaleBias = float,
    typename ElementAccumulator = float,
    typename ElementCompute = float>
class Conv1x1ReluConv1x1Relu {
 public:
  using KernelConfig = kernel::DefaultConv1x1ReluConv1x1Relu<
      ArchTag,
      ElementA,
      ElementB,
      ElementC,
      ElementAccumulator,
      ElementCompute>;
  using Operation = cutlass::conv::device::B2bImplicitGemmConvolution<
      typename KernelConfig::CutlassKernel>;
  using CutlassArguments = typename Operation::Arguments;

  struct Arguments {
    cutlass::conv::Conv2dProblemSize problem_size_0;
    cutlass::conv::Conv2dProblemSize problem_size_1;
    ElementA const* input = nullptr;
    ElementB const* weight0 = nullptr;
    ElementScaleBias const* stage0_scale = nullptr;
    ElementScaleBias const* bias0 = nullptr;
    ElementB const* weight1 = nullptr;
    ElementC const* bias1 = nullptr;
    ElementC* output = nullptr;
    ElementCompute stage0_alpha = ElementCompute(1);
    ElementCompute output_alpha = ElementCompute(1);
  };

 private:
  Operation operation_;
  CutlassArguments cutlass_args_;

 public:
  Conv1x1ReluConv1x1Relu() = default;

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

}  // namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu::device
