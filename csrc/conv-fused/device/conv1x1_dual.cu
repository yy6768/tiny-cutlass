#include "device/conv1x1_relu_conv1x1.h"

#include "cutlass/half.h"

namespace tiny_cutlass::conv_fused::device {
namespace {

bool is_conv1x1(
    cutlass::conv::Conv2dProblemSize const& problem_size) {
  return problem_size.R == 1 && problem_size.S == 1;
}

template <typename ArchTag, typename Element>
typename Conv1x1ReluConv1x1<ArchTag, Element>::CutlassArguments
make_cutlass_arguments(
    typename Conv1x1ReluConv1x1<ArchTag, Element>::Arguments const& arguments) {
  using Conv = Conv1x1ReluConv1x1<ArchTag, Element>;
  using TensorRef = cutlass::TensorRef<Element, cutlass::layout::TensorNHWC>;
  using VectorRef = cutlass::TensorRef<Element, cutlass::layout::RowMajor>;

  Element* mutable_input = const_cast<Element*>(arguments.input);
  Element* mutable_weight0 = const_cast<Element*>(arguments.weight0);
  Element* mutable_bias0 = const_cast<Element*>(arguments.bias0);
  Element* mutable_weight1 = const_cast<Element*>(arguments.weight1);
  Element* mutable_bias1 = const_cast<Element*>(arguments.bias1);

  typename Conv::CutlassArguments cutlass_args(
      arguments.problem_size_0,
      arguments.problem_size_1,
      TensorRef{
          mutable_input,
          cutlass::layout::TensorNHWC::packed(
              arguments.problem_size_0.activation_extent())},
      TensorRef{
          mutable_weight0,
          cutlass::layout::TensorNHWC::packed(
              arguments.problem_size_0.filter_extent())},
      TensorRef{
          nullptr,
          cutlass::layout::TensorNHWC::packed(
              arguments.problem_size_0.output_extent())},
      VectorRef{nullptr, cutlass::layout::RowMajor(arguments.problem_size_0.K)},
      VectorRef{
          mutable_bias0,
          cutlass::layout::RowMajor(arguments.problem_size_0.K)},
      TensorRef{
          mutable_weight1,
          cutlass::layout::TensorNHWC::packed(
              arguments.problem_size_1.filter_extent())},
      TensorRef{mutable_bias1, cutlass::layout::TensorNHWC(0, 0, 0)},
      TensorRef{
          arguments.output,
          cutlass::layout::TensorNHWC::packed(
              arguments.problem_size_1.output_extent())},
      {Element(1), Element(0)},
      {Element(1), Element(1)},
      cutlass::conv::SplitKMode::kSerial);
  return cutlass_args;
}

}  // namespace

template <typename ArchTag, typename Element>
cutlass::Status Conv1x1ReluConv1x1<ArchTag, Element>::can_implement(
    Arguments const& args) {
  if (!is_conv1x1(args.problem_size_0) ||
      !is_conv1x1(args.problem_size_1)) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  CutlassArguments cutlass_args =
      make_cutlass_arguments<ArchTag, Element>(args);
  return Operation::can_implement(cutlass_args);
}

template <typename ArchTag, typename Element>
size_t Conv1x1ReluConv1x1<ArchTag, Element>::get_workspace_size(
    Arguments const& args) {
  if (can_implement(args) != cutlass::Status::kSuccess) {
    return 0;
  }

  CutlassArguments cutlass_args =
      make_cutlass_arguments<ArchTag, Element>(args);
  return Operation::get_workspace_size(cutlass_args);
}

template <typename ArchTag, typename Element>
cutlass::Status Conv1x1ReluConv1x1<ArchTag, Element>::initialize(
    Arguments const& args,
    void* workspace,
    cudaStream_t stream) {
  if (!args.input || !args.weight0 || !args.bias0 || !args.weight1 ||
      !args.bias1 || !args.output) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  cutlass::Status status = can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  cutlass_args_ = make_cutlass_arguments<ArchTag, Element>(args);
  return operation_.initialize(cutlass_args_, workspace, stream);
}

template <typename ArchTag, typename Element>
cutlass::Status Conv1x1ReluConv1x1<ArchTag, Element>::run(
    cudaStream_t stream) {
  return operation_.run(stream);
}

template class Conv1x1ReluConv1x1<cutlass::arch::Sm80, cutlass::half_t>;

}  // namespace tiny_cutlass::conv_fused::device
