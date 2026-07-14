#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/arch/arch.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/half.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_ref.h"

#include "conv1x1_upsample/kernel/conv1x1_upsample.h"

namespace tiny_cutlass::conv_fused::conv1x1_upsample::device {

// CUTLASS-style device operator for conv1x1 -> nearest-neighbor UpsampleH x
// UpsampleW upsample. This family needs no bespoke device operator: the conv
// is an ordinary implicit-GEMM conv1x1 at the LOW
// (pre-upsample) resolution, and the entire upsample lives in the swapped
// epilogue output iterator (conv1x1_upsample/epilogue/
// predicated_tile_iterator_upsample.h). So this simply wraps the stock
// cutlass::conv::device::ImplicitGemmConvolution over the conv1x1_upsample
// kernel.
//
// problem_size here is expressed at the LOW resolution (P, Q). The caller
// binds the output TensorRef with HIGH-resolution packed NHWC strides
// (UpsampleH*P, UpsampleW*Q), which is exactly what ConvOutputIteratorParameter
// forwards to the output iterator: low-res divmods (from problem_size) plus
// high-res strides (from ref_D).
template <
    typename ArchTag = cutlass::arch::Sm80,
    typename Element = cutlass::half_t,
    int UpsampleH = 2,
    int UpsampleW = 2>
class Conv1x1Upsample {
 public:
  using KernelConfig = kernel::DefaultConv1x1Upsample<ArchTag, Element, UpsampleH, UpsampleW>;
  using Operation = cutlass::conv::device::ImplicitGemmConvolution<
      typename KernelConfig::CutlassKernel>;
  using CutlassArguments = typename Operation::Arguments;

  static int const kUpsampleH = UpsampleH;
  static int const kUpsampleW = UpsampleW;

  struct Arguments {
    // Conv2dProblemSize at the LOW (pre-upsample) resolution.
    cutlass::conv::Conv2dProblemSize problem_size;
    Element const* input = nullptr;
    Element const* weight = nullptr;
    Element const* bias = nullptr;
    Element* output = nullptr;
  };

 private:
  Operation operation_;
  CutlassArguments cutlass_args_;

 public:
  Conv1x1Upsample() = default;

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

}  // namespace tiny_cutlass::conv_fused::conv1x1_upsample::device
