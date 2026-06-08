#include "device/conv1x1_relu_conv1x1.h"

#include "cutlass/half.h"

#include "kernel/conv1x1_relu_conv1x1.h"

namespace tiny_cutlass::conv_fused::device {
namespace {

template <typename ArchTag, typename Element>
cutlass::Status run_impl(
    cutlass::conv::Conv2dProblemSize const& problem_size_0,
    cutlass::conv::Conv2dProblemSize const& problem_size_1,
    Element const* input,
    Element const* weight0,
    Element* stage0,
    Element const* bias0,
    Element const* weight1,
    Element const* bias1,
    Element* output,
    cudaStream_t stream) {
  using KernelConfig =
      kernel::DefaultConv1x1ReluConv1x1<ArchTag, Element>;
  using Conv = cutlass::conv::device::B2bImplicitGemmConvolution<
      typename KernelConfig::CutlassKernel>;
  using TensorRef = cutlass::TensorRef<Element, cutlass::layout::TensorNHWC>;
  using VectorRef = cutlass::TensorRef<Element, cutlass::layout::RowMajor>;

  Element* mutable_input = const_cast<Element*>(input);
  Element* mutable_weight0 = const_cast<Element*>(weight0);
  Element* mutable_bias0 = const_cast<Element*>(bias0);
  Element* mutable_weight1 = const_cast<Element*>(weight1);
  Element* mutable_bias1 = const_cast<Element*>(bias1);

  typename Conv::Arguments args(
      problem_size_0,
      problem_size_1,
      TensorRef{mutable_input, cutlass::layout::TensorNHWC::packed(problem_size_0.activation_extent())},
      TensorRef{mutable_weight0, cutlass::layout::TensorNHWC::packed(problem_size_0.filter_extent())},
      TensorRef{stage0, cutlass::layout::TensorNHWC::packed(problem_size_0.output_extent())},
      VectorRef{nullptr, cutlass::layout::RowMajor(problem_size_0.K)},
      VectorRef{mutable_bias0, cutlass::layout::RowMajor(problem_size_0.K)},
      TensorRef{mutable_weight1, cutlass::layout::TensorNHWC::packed(problem_size_1.filter_extent())},
      TensorRef{mutable_bias1, cutlass::layout::TensorNHWC(0, 0, 0)},
      TensorRef{output, cutlass::layout::TensorNHWC::packed(problem_size_1.output_extent())},
      {Element(1), Element(0)},
      {Element(1), Element(1)},
      cutlass::conv::SplitKMode::kSerial);

  cutlass::Status status = Conv::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  Conv op;
  return op(args, nullptr, stream);
}

}  // namespace

template <typename ArchTag, typename Element>
cutlass::Status run_conv1x1_relu_conv1x1(
    cutlass::conv::Conv2dProblemSize const& problem_size_0,
    cutlass::conv::Conv2dProblemSize const& problem_size_1,
    Element const* input,
    Element const* weight0,
    Element* stage0,
    Element const* bias0,
    Element const* weight1,
    Element const* bias1,
    Element* output,
    cudaStream_t stream) {
  return run_impl<ArchTag, Element>(
      problem_size_0,
      problem_size_1,
      input,
      weight0,
      stage0,
      bias0,
      weight1,
      bias1,
      output,
      stream);
}

template cutlass::Status run_conv1x1_relu_conv1x1<cutlass::arch::Sm80, cutlass::half_t>(
    cutlass::conv::Conv2dProblemSize const&,
    cutlass::conv::Conv2dProblemSize const&,
    cutlass::half_t const*,
    cutlass::half_t const*,
    cutlass::half_t*,
    cutlass::half_t const*,
    cutlass::half_t const*,
    cutlass::half_t const*,
    cutlass::half_t*,
    cudaStream_t);

}  // namespace tiny_cutlass::conv_fused::device
