#include "conv1x1_dual/device/conv1x1_dual.h"

#include "cutlass/half.h"

namespace tiny_cutlass::conv_fused::device {
namespace {

bool is_conv1x1(
    cutlass::conv::Conv2dProblemSize const& problem_size) {
  return problem_size.R == 1 && problem_size.S == 1;
}

// Wraps a (possibly const) raw pointer as a packed NHWC TensorRef. CUTLASS has
// no single helper that builds a whole conv Arguments struct, but it does give
// us the pieces: TensorNHWC::packed() for the layout and make_TensorRef() for
// the ref. This just folds in the const_cast the C API forces on us.
template <typename Element>
cutlass::TensorRef<Element, cutlass::layout::TensorNHWC>
packed_nhwc_ref(Element const* ptr, cutlass::Tensor4DCoord const& extent) {
  return cutlass::make_TensorRef(
      const_cast<Element*>(ptr),
      cutlass::layout::TensorNHWC::packed(extent));
}

template <typename ArchTag, typename Element, kernel::Residency ResidencyKind>
typename Conv1x1Dual<ArchTag, Element, ResidencyKind>::CutlassArguments
make_cutlass_arguments(
    typename Conv1x1Dual<ArchTag, Element, ResidencyKind>::Arguments const& arguments) {
  using Conv = Conv1x1Dual<ArchTag, Element, ResidencyKind>;
  using VectorRef = cutlass::TensorRef<Element, cutlass::layout::RowMajor>;

  auto const& p0 = arguments.problem_size_0;
  auto const& p1 = arguments.problem_size_1;

  return typename Conv::CutlassArguments(
      p0,
      p1,
      // Stage-0 GEMM operands.
      packed_nhwc_ref(arguments.input, p0.activation_extent()),   // A0
      packed_nhwc_ref(arguments.weight0, p0.filter_extent()),     // B0
      packed_nhwc_ref<Element>(nullptr, p0.output_extent()),      // C0 (unused)
      // Stage-0 accumulator scale/bias vectors (RowMajor, length K0).
      VectorRef{nullptr, cutlass::layout::RowMajor(p0.K)},        // Scale0 (none)
      VectorRef{const_cast<Element*>(arguments.bias0),
                cutlass::layout::RowMajor(p0.K)},                 // Bias0
      // Stage-1 operands.
      packed_nhwc_ref(arguments.weight1, p1.filter_extent()),     // B1
      // bias1 rides in through source tensor C1 with zero spatial strides, so
      // every output pixel reads bias1[k] (beta1 = 1 below applies it).
      cutlass::make_TensorRef(const_cast<Element*>(arguments.bias1),
                              cutlass::layout::TensorNHWC(0, 0, 0)),  // C1
      packed_nhwc_ref(arguments.output, p1.output_extent()),      // D1
      {Element(1), Element(0)},  // stage-0 epilogue: alpha0=1, beta0=0 (bias via Bias0)
      {Element(1), Element(1)},  // stage-1 epilogue: alpha1=1, beta1=1 (bias via C1)
      cutlass::conv::SplitKMode::kSerial);
}

}  // namespace

template <typename ArchTag, typename Element, kernel::Residency ResidencyKind>
cutlass::Status Conv1x1Dual<ArchTag, Element, ResidencyKind>::can_implement(
    Arguments const& args) {
  if (!is_conv1x1(args.problem_size_0) ||
      !is_conv1x1(args.problem_size_1)) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  return Operation::can_implement(
      make_cutlass_arguments<ArchTag, Element, ResidencyKind>(args));
}

template <typename ArchTag, typename Element, kernel::Residency ResidencyKind>
size_t Conv1x1Dual<ArchTag, Element, ResidencyKind>::get_workspace_size(
    Arguments const& args) {
  if (can_implement(args) != cutlass::Status::kSuccess) {
    return 0;
  }

  return Operation::get_workspace_size(
      make_cutlass_arguments<ArchTag, Element, ResidencyKind>(args));
}

template <typename ArchTag, typename Element, kernel::Residency ResidencyKind>
cutlass::Status Conv1x1Dual<ArchTag, Element, ResidencyKind>::initialize(
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

  cutlass_args_ = make_cutlass_arguments<ArchTag, Element, ResidencyKind>(args);
  return operation_.initialize(cutlass_args_, workspace, stream);
}

template <typename ArchTag, typename Element, kernel::Residency ResidencyKind>
cutlass::Status Conv1x1Dual<ArchTag, Element, ResidencyKind>::run(
    cudaStream_t stream) {
  return operation_.run(stream);
}

// Explicit instantiations: both residencies for fp16.
template class Conv1x1Dual<
    cutlass::arch::Sm80, cutlass::half_t, kernel::Residency::kRF>;
template class Conv1x1Dual<
    cutlass::arch::Sm80, cutlass::half_t, kernel::Residency::kSmem>;

}  // namespace tiny_cutlass::conv_fused::device
