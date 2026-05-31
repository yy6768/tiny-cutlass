#include "device/conv1x1_relu_conv1x1_b2b.h"

#include "cutlass/half.h"

#include "kernel/conv1x1_relu_conv1x1_b2b_kernel.h"

namespace tiny_cutlass::conv_fused::device {
namespace {

template <typename Element>
cutlass::TensorRef<Element, cutlass::layout::TensorNHWC> packed_nhwc(
    Element* ptr,
    cutlass::Tensor4DCoord extent) {
  return {ptr, cutlass::layout::TensorNHWC::packed(extent)};
}

template <typename Element>
cutlass::TensorRef<Element const, cutlass::layout::TensorNHWC> packed_nhwc(
    Element const* ptr,
    cutlass::Tensor4DCoord extent) {
  return {ptr, cutlass::layout::TensorNHWC::packed(extent)};
}

template <typename Element>
cutlass::TensorRef<Element, cutlass::layout::TensorNHWC> broadcast_channel_ref(
    Element* ptr) {
  return {ptr, cutlass::layout::TensorNHWC(0, 0, 0)};
}

template <typename Element>
cutlass::TensorRef<Element, cutlass::layout::RowMajor> vector_ref(Element* ptr) {
  return {ptr, cutlass::layout::RowMajor(0)};
}

template <typename Element>
cutlass::TensorRef<Element, cutlass::layout::RowMajor> vector_ref(Element* ptr, int64_t stride) {
  return {ptr, cutlass::layout::RowMajor(stride)};
}

template <typename Element>
cutlass::Status run_impl(
    cutlass::conv::Conv2dProblemSize const& problem_size_0,
    cutlass::conv::Conv2dProblemSize const& problem_size_1,
    Element const* input_nhwc,
    Element const* weight0_krsc,
    Element const* bias0,
    Element const* weight1_krsc,
    Element const* bias1,
    Element* output_nhwc,
    cudaStream_t stream) {
  using KernelConfig = kernel::Conv1x1ReluConv1x1B2bSm80<Element>;
  using B2bConv = cutlass::conv::device::B2bImplicitGemmConvolution<
      typename KernelConfig::CutlassKernel>;

  Element* mutable_bias0 = const_cast<Element*>(bias0);
  Element* mutable_bias1 = const_cast<Element*>(bias1);
  at::Tensor c0_source = at::empty(
      {problem_size_0.N, problem_size_0.P, problem_size_0.Q, problem_size_0.K},
      at::TensorOptions().device(at::kCUDA).dtype(cutlass::platform::is_same<Element, float>::value
            ? at::kFloat
            : at::kHalf)
          .memory_format(at::MemoryFormat::ChannelsLast));

  typename B2bConv::Arguments args(
      problem_size_0,
      problem_size_1,
      packed_nhwc(input_nhwc, problem_size_0.activation_extent()),
      packed_nhwc(weight0_krsc, problem_size_0.filter_extent()),
      packed_nhwc(c0_source.data_ptr<Element>(), problem_size_0.output_extent()),
      vector_ref<Element>(nullptr, problem_size_0.K),
      vector_ref(mutable_bias0, problem_size_0.K),
      packed_nhwc(weight1_krsc, problem_size_1.filter_extent()),
      broadcast_channel_ref(mutable_bias1),
      packed_nhwc(output_nhwc, problem_size_1.output_extent()),
      {Element(1), Element(0)},
      {Element(1), Element(1)},
      cutlass::conv::SplitKMode::kSerial);

  cutlass::Status status = B2bConv::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  B2bConv op;
  return op(args, nullptr, stream);
}

}  // namespace

template <typename Element>
cutlass::Status run_conv1x1_relu_conv1x1_b2b_sm80(
    cutlass::conv::Conv2dProblemSize const& problem_size_0,
    cutlass::conv::Conv2dProblemSize const& problem_size_1,
    Element const* input_nhwc,
    Element const* weight0_krsc,
    Element const* bias0,
    Element const* weight1_krsc,
    Element const* bias1,
    Element* output_nhwc,
    cudaStream_t stream) {
  return run_impl(
      problem_size_0,
      problem_size_1,
      input_nhwc,
      weight0_krsc,
      bias0,
      weight1_krsc,
      bias1,
      output_nhwc,
      stream);
}

template cutlass::Status run_conv1x1_relu_conv1x1_b2b_sm80<float>(
    cutlass::conv::Conv2dProblemSize const&,
    cutlass::conv::Conv2dProblemSize const&,
    float const*,
    float const*,
    float const*,
    float const*,
    float const*,
    float*,
    cudaStream_t);

template cutlass::Status run_conv1x1_relu_conv1x1_b2b_sm80<cutlass::half_t>(
    cutlass::conv::Conv2dProblemSize const&,
    cutlass::conv::Conv2dProblemSize const&,
    cutlass::half_t const*,
    cutlass::half_t const*,
    cutlass::half_t const*,
    cutlass::half_t const*,
    cutlass::half_t const*,
    cutlass::half_t*,
    cudaStream_t);

}  // namespace tiny_cutlass::conv_fused::device
