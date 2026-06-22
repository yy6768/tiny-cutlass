#include "fp8/conv1x1_relu_conv1x1_relu_fp8/device/conv1x1_relu_conv1x1_relu_fp8.h"

#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_ref.h"

namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu::device {
namespace {

bool is_conv1x1(
    cutlass::conv::Conv2dProblemSize const& problem_size) {
  return problem_size.R == 1 && problem_size.S == 1;
}

template <
    typename ArchTag,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename ElementScaleBias,
    typename ElementAccumulator,
    typename ElementCompute>
typename Conv1x1ReluConv1x1Relu<
    ArchTag,
    ElementA,
    ElementB,
    ElementC,
    ElementScaleBias,
    ElementAccumulator,
    ElementCompute>::CutlassArguments
make_cutlass_arguments(
    typename Conv1x1ReluConv1x1Relu<
        ArchTag,
        ElementA,
        ElementB,
        ElementC,
        ElementScaleBias,
        ElementAccumulator,
        ElementCompute>::Arguments const& arguments) {
  using Conv = Conv1x1ReluConv1x1Relu<
      ArchTag,
      ElementA,
      ElementB,
      ElementC,
      ElementScaleBias,
      ElementAccumulator,
      ElementCompute>;
  using TensorRefA = cutlass::TensorRef<ElementA, cutlass::layout::TensorNHWC>;
  using TensorRefB = cutlass::TensorRef<ElementB, cutlass::layout::TensorNHWC>;
  using TensorRefC = cutlass::TensorRef<ElementC, cutlass::layout::TensorNHWC>;
  using ScaleBiasRef =
      cutlass::TensorRef<ElementScaleBias, cutlass::layout::RowMajor>;

  ElementA* mutable_input = const_cast<ElementA*>(arguments.input);
  ElementB* mutable_weight0 = const_cast<ElementB*>(arguments.weight0);
  ElementScaleBias* mutable_stage0_scale =
      const_cast<ElementScaleBias*>(arguments.stage0_scale);
  ElementScaleBias* mutable_bias0 = const_cast<ElementScaleBias*>(arguments.bias0);
  ElementB* mutable_weight1 = const_cast<ElementB*>(arguments.weight1);
  ElementC* mutable_bias1 = const_cast<ElementC*>(arguments.bias1);

  typename Conv::CutlassArguments cutlass_args(
      arguments.problem_size_0,
      arguments.problem_size_1,
      TensorRefA{
          mutable_input,
          cutlass::layout::TensorNHWC::packed(
              arguments.problem_size_0.activation_extent())},
      TensorRefB{
          mutable_weight0,
          cutlass::layout::TensorNHWC::packed(
              arguments.problem_size_0.filter_extent())},
      TensorRefC{
          nullptr,
          cutlass::layout::TensorNHWC::packed(
              arguments.problem_size_0.output_extent())},
      ScaleBiasRef{
          mutable_stage0_scale,
          cutlass::layout::RowMajor(arguments.problem_size_0.K)},
      ScaleBiasRef{
          mutable_bias0,
          cutlass::layout::RowMajor(arguments.problem_size_0.K)},
      TensorRefB{
          mutable_weight1,
          cutlass::layout::TensorNHWC::packed(
              arguments.problem_size_1.filter_extent())},
      TensorRefC{mutable_bias1, cutlass::layout::TensorNHWC(0, 0, 0)},
      TensorRefC{
          arguments.output,
          cutlass::layout::TensorNHWC::packed(
              arguments.problem_size_1.output_extent())},
      {arguments.stage0_alpha, 0.0f},
      {arguments.output_alpha, 1.0f},
      cutlass::conv::SplitKMode::kSerial);
  return cutlass_args;
}

}  // namespace

template <
    typename ArchTag,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename ElementScaleBias,
    typename ElementAccumulator,
    typename ElementCompute>
cutlass::Status Conv1x1ReluConv1x1Relu<
    ArchTag,
    ElementA,
    ElementB,
    ElementC,
    ElementScaleBias,
    ElementAccumulator,
    ElementCompute>::can_implement(Arguments const& args) {
  if (!is_conv1x1(args.problem_size_0) ||
      !is_conv1x1(args.problem_size_1)) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  CutlassArguments cutlass_args =
      make_cutlass_arguments<
          ArchTag,
          ElementA,
          ElementB,
          ElementC,
          ElementScaleBias,
          ElementAccumulator,
          ElementCompute>(args);
  return Operation::can_implement(cutlass_args);
}

template <
    typename ArchTag,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename ElementScaleBias,
    typename ElementAccumulator,
    typename ElementCompute>
size_t Conv1x1ReluConv1x1Relu<
    ArchTag,
    ElementA,
    ElementB,
    ElementC,
    ElementScaleBias,
    ElementAccumulator,
    ElementCompute>::get_workspace_size(Arguments const& args) {
  if (can_implement(args) != cutlass::Status::kSuccess) {
    return 0;
  }

  CutlassArguments cutlass_args =
      make_cutlass_arguments<
          ArchTag,
          ElementA,
          ElementB,
          ElementC,
          ElementScaleBias,
          ElementAccumulator,
          ElementCompute>(args);
  return Operation::get_workspace_size(cutlass_args);
}

template <
    typename ArchTag,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename ElementScaleBias,
    typename ElementAccumulator,
    typename ElementCompute>
cutlass::Status Conv1x1ReluConv1x1Relu<
    ArchTag,
    ElementA,
    ElementB,
    ElementC,
    ElementScaleBias,
    ElementAccumulator,
    ElementCompute>::initialize(
    Arguments const& args,
    void* workspace,
    cudaStream_t stream) {
  if (!args.input || !args.weight0 || !args.stage0_scale || !args.bias0 ||
      !args.weight1 || !args.bias1 || !args.output) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  cutlass::Status status = can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  cutlass_args_ =
      make_cutlass_arguments<
          ArchTag,
          ElementA,
          ElementB,
          ElementC,
          ElementScaleBias,
          ElementAccumulator,
          ElementCompute>(args);
  return operation_.initialize(cutlass_args_, workspace, stream);
}

template <
    typename ArchTag,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename ElementScaleBias,
    typename ElementAccumulator,
    typename ElementCompute>
cutlass::Status Conv1x1ReluConv1x1Relu<
    ArchTag,
    ElementA,
    ElementB,
    ElementC,
    ElementScaleBias,
    ElementAccumulator,
    ElementCompute>::run(cudaStream_t stream) {
  return operation_.run(stream);
}

template class Conv1x1ReluConv1x1Relu<
    cutlass::arch::Sm89,
    cutlass::float_e4m3_t,
    cutlass::float_e4m3_t,
    cutlass::float_e4m3_t,
    float,
    float,
    float>;

}  // namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu::device
