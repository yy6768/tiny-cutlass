#include "fp8/conv1x1_relu_conv1x1_relu_fp8/device/conv1x1_relu_conv1x1_relu_fp8.h"

#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_ref.h"

#include "device/implicit_gemm_convolution_fusion.h"
#include "fp8/conv1x1_relu_conv1x1_relu_fp8/kernel/conv1x1_relu_conv1x1_relu_fp8.h"

namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu::device {

template <
    typename ArchTag,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename ElementScaleBias,
    typename ElementAccumulator,
    typename ElementCompute>
cutlass::Status run_conv1x1_relu_conv1x1_relu_fp8(
    cutlass::conv::Conv2dProblemSize const& problem_size_0,
    cutlass::conv::Conv2dProblemSize const& problem_size_1,
    ElementA const* input_nhwc,
    ElementB const* weight0_krsc,
    ElementC* stage0_output_nhwc,
    ElementScaleBias const* stage0_scale,
    ElementScaleBias const* bias0,
    ElementB const* weight1_krsc,
    ElementC const* bias1,
    ElementC* output_nhwc,
    ElementCompute stage0_alpha,
    ElementCompute output_alpha,
    cudaStream_t stream) {
  using KernelConfig =
      kernel::DefaultConv1x1ReluConv1x1ReluFp8<
          ArchTag,
          ElementA,
          ElementB,
          ElementC,
          ElementAccumulator,
          ElementCompute>;
  using Conv =
      tiny_cutlass::conv_fused::device::ImplicitGemmConvolutionFusion<
          typename KernelConfig::CutlassKernel>;
  using TensorRefA = cutlass::TensorRef<ElementA, cutlass::layout::TensorNHWC>;
  using TensorRefB = cutlass::TensorRef<ElementB, cutlass::layout::TensorNHWC>;
  using TensorRefC = cutlass::TensorRef<ElementC, cutlass::layout::TensorNHWC>;
  using ScaleBiasRef =
      cutlass::TensorRef<ElementScaleBias, cutlass::layout::RowMajor>;

  ElementA* mutable_input = const_cast<ElementA*>(input_nhwc);
  ElementB* mutable_weight0 = const_cast<ElementB*>(weight0_krsc);
  ElementScaleBias* mutable_stage0_scale =
      const_cast<ElementScaleBias*>(stage0_scale);
  ElementScaleBias* mutable_bias0 = const_cast<ElementScaleBias*>(bias0);
  ElementB* mutable_weight1 = const_cast<ElementB*>(weight1_krsc);
  ElementC* mutable_bias1 = const_cast<ElementC*>(bias1);

  typename Conv::Arguments args(
      problem_size_0,
      problem_size_1,
      TensorRefA{mutable_input, cutlass::layout::TensorNHWC::packed(problem_size_0.activation_extent())},
      TensorRefB{mutable_weight0, cutlass::layout::TensorNHWC::packed(problem_size_0.filter_extent())},
      TensorRefC{stage0_output_nhwc, cutlass::layout::TensorNHWC::packed(problem_size_0.output_extent())},
      ScaleBiasRef{mutable_stage0_scale, cutlass::layout::RowMajor(problem_size_0.K)},
      ScaleBiasRef{mutable_bias0, cutlass::layout::RowMajor(problem_size_0.K)},
      TensorRefB{mutable_weight1, cutlass::layout::TensorNHWC::packed(problem_size_1.filter_extent())},
      TensorRefC{mutable_bias1, cutlass::layout::TensorNHWC(0, 0, 0)},
      TensorRefC{output_nhwc, cutlass::layout::TensorNHWC::packed(problem_size_1.output_extent())},
      {stage0_alpha, 0.0f},
      {output_alpha, 1.0f},
      cutlass::conv::SplitKMode::kSerial);

  cutlass::Status status = Conv::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  Conv op;
  return op(args, nullptr, stream);
}

template cutlass::Status run_conv1x1_relu_conv1x1_relu_fp8<
    cutlass::arch::Sm89,
    cutlass::float_e4m3_t,
    cutlass::float_e4m3_t,
    cutlass::float_e4m3_t,
    float,
    float,
    float>(
    cutlass::conv::Conv2dProblemSize const&,
    cutlass::conv::Conv2dProblemSize const&,
    cutlass::float_e4m3_t const*,
    cutlass::float_e4m3_t const*,
    cutlass::float_e4m3_t*,
    float const*,
    float const*,
    cutlass::float_e4m3_t const*,
    cutlass::float_e4m3_t const*,
    cutlass::float_e4m3_t*,
    float,
    float,
    cudaStream_t);

}  // namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu::device
