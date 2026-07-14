#include "conv1x1_upsample/device/conv1x1_upsample.h"

#include "cutlass/half.h"

namespace tiny_cutlass::conv_fused::conv1x1_upsample::device {
namespace {

bool is_conv1x1(cutlass::conv::Conv2dProblemSize const& problem_size) {
  return problem_size.R == 1 && problem_size.S == 1;
}

// High-resolution output extent: the conv runs at the low resolution
// (problem_size.P, problem_size.Q), and each output pixel is broadcast into a
// UpsampleH x UpsampleW block, so the destination tensor is
// (N, UpsampleH*P, UpsampleW*Q, K).
template <int UpsampleH, int UpsampleW>
cutlass::Tensor4DCoord upsampled_output_extent(
    cutlass::conv::Conv2dProblemSize const& problem_size) {
  return cutlass::Tensor4DCoord(
      problem_size.N,
      UpsampleH * problem_size.P,
      UpsampleW * problem_size.Q,
      problem_size.K);
}

template <typename ArchTag, typename Element, int UpsampleH, int UpsampleW>
typename Conv1x1Upsample<ArchTag, Element, UpsampleH, UpsampleW>::CutlassArguments
make_cutlass_arguments(
    typename Conv1x1Upsample<ArchTag, Element, UpsampleH, UpsampleW>::Arguments const& arguments) {
  using Conv = Conv1x1Upsample<ArchTag, Element, UpsampleH, UpsampleW>;
  using TensorRef = cutlass::TensorRef<Element, cutlass::layout::TensorNHWC>;

  Element* mutable_input = const_cast<Element*>(arguments.input);
  Element* mutable_weight = const_cast<Element*>(arguments.weight);
  Element* mutable_bias = const_cast<Element*>(arguments.bias);

  typename Conv::CutlassArguments cutlass_args(
      arguments.problem_size,
      // A: activation at low resolution, packed NHWC.
      TensorRef{
          mutable_input,
          cutlass::layout::TensorNHWC::packed(arguments.problem_size.activation_extent())},
      // B: 1x1 filter (K, 1, 1, C), packed.
      TensorRef{
          mutable_weight,
          cutlass::layout::TensorNHWC::packed(arguments.problem_size.filter_extent())},
      // C: per-channel bias broadcast -- zero spatial strides so every (p, q)
      // reads bias[k]. Loaded through the (unmodified) low-res coordinate
      // mapping of the upsample output iterator.
      TensorRef{mutable_bias, cutlass::layout::TensorNHWC(0, 0, 0)},
      // D: destination at the HIGH (upsampled) resolution, packed NHWC. Its
      // strides are what the output iterator scatters into; problem_size stays
      // low-res so the row->(n, p, q) divmods decompose against low-res P, Q.
      TensorRef{
          arguments.output,
          cutlass::layout::TensorNHWC::packed(
              upsampled_output_extent<UpsampleH, UpsampleW>(arguments.problem_size))},
      {Element(1), Element(1)},
      cutlass::conv::SplitKMode::kSerial);
  return cutlass_args;
}

}  // namespace

template <typename ArchTag, typename Element, int UpsampleH, int UpsampleW>
cutlass::Status Conv1x1Upsample<ArchTag, Element, UpsampleH, UpsampleW>::can_implement(
    Arguments const& args) {
  if (!is_conv1x1(args.problem_size)) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  CutlassArguments cutlass_args =
      make_cutlass_arguments<ArchTag, Element, UpsampleH, UpsampleW>(args);
  return Operation::can_implement(cutlass_args);
}

template <typename ArchTag, typename Element, int UpsampleH, int UpsampleW>
size_t Conv1x1Upsample<ArchTag, Element, UpsampleH, UpsampleW>::get_workspace_size(
    Arguments const& args) {
  if (can_implement(args) != cutlass::Status::kSuccess) {
    return 0;
  }

  CutlassArguments cutlass_args =
      make_cutlass_arguments<ArchTag, Element, UpsampleH, UpsampleW>(args);
  return Operation::get_workspace_size(cutlass_args);
}

template <typename ArchTag, typename Element, int UpsampleH, int UpsampleW>
cutlass::Status Conv1x1Upsample<ArchTag, Element, UpsampleH, UpsampleW>::initialize(
    Arguments const& args,
    void* workspace,
    cudaStream_t stream) {
  if (!args.input || !args.weight || !args.bias || !args.output) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  cutlass::Status status = can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  cutlass_args_ = make_cutlass_arguments<ArchTag, Element, UpsampleH, UpsampleW>(args);
  return operation_.initialize(cutlass_args_, workspace, stream);
}

template <typename ArchTag, typename Element, int UpsampleH, int UpsampleW>
cutlass::Status Conv1x1Upsample<ArchTag, Element, UpsampleH, UpsampleW>::run(
    cudaStream_t stream) {
  return operation_.run(stream);
}

template class Conv1x1Upsample<cutlass::arch::Sm80, cutlass::half_t, 2, 2>;

}  // namespace tiny_cutlass::conv_fused::conv1x1_upsample::device
