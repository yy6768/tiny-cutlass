#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/arch/arch.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/half.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_ref.h"

#include "device/b2b_implicit_gemm_convolution.h"
#include "kernel/conv1x1_relu_conv1x1.h"

namespace tiny_cutlass::conv_fused::device {

template <
    typename ArchTag = cutlass::arch::Sm80,
    typename Element = cutlass::half_t>
class Conv1x1ReluConv1x1 {
 public:
  using KernelConfig = kernel::DefaultConv1x1ReluConv1x1<ArchTag, Element>;
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
  Conv1x1ReluConv1x1() = default;

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
