#include "device/conv_relu_pool.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/half.h"
#include "cutlass/reduction/device/tensor_reduce_affine_strided.h"

#include "kernel/conv_relu_pool.h"

namespace tiny_cutlass::conv_fused::device {
namespace {

constexpr size_t kWorkspaceAlignment = 256;

size_t align_up(size_t value, size_t alignment = kWorkspaceAlignment) {
  return (value + alignment - 1) / alignment * alignment;
}

cutlass::conv::Conv2dProblemSize make_problem(
    int batch,
    int height,
    int width,
    int channels,
    int filters,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w) {
  return cutlass::conv::Conv2dProblemSize(
      cutlass::Tensor4DCoord(batch, height, width, channels),
      cutlass::Tensor4DCoord(filters, kernel_h, kernel_w, channels),
      cutlass::Tensor4DCoord(pad_h, pad_h, pad_w, pad_w),
      cutlass::MatrixCoord(1, 1),
      cutlass::MatrixCoord(1, 1),
      cutlass::Tensor4DCoord(batch, height, width, filters),
      cutlass::conv::Mode::kCrossCorrelation,
      1,
      1);
}

template <typename Element>
size_t tensor_bytes(int batch, int height, int width, int channels) {
  return sizeof(Element) *
         size_t(batch) *
         size_t(height) *
         size_t(width) *
         size_t(channels);
}

template <typename Element>
using PoolOperation = cutlass::reduction::device::TensorReductionAffineStrided<
    6,
    4,
    Element,
    Element,
    typename kernel::DefaultPool<Element>::ReductionOp,
    kernel::DefaultPool<Element>::kVectorLength,
    typename kernel::DefaultPool<Element>::ElementCompute>;

template <typename Element>
PoolOperation<Element> make_pool(
    int batch,
    int output_h,
    int output_w,
    int channels) {
  cutlass::Coord<6> extent;
  extent[0] = batch;
  extent[1] = output_h;
  extent[2] = output_w;
  extent[3] = 2;
  extent[4] = 2;
  extent[5] = channels;
  return PoolOperation<Element>(extent);
}

template <typename ArchTag, typename Element>
using ConvReluOperation = cutlass::conv::device::ImplicitGemmConvolution<
    typename kernel::DefaultConvRelu<ArchTag, Element>::CutlassKernel>;

template <typename ArchTag, typename Element>
typename ConvReluOperation<ArchTag, Element>::Arguments
make_conv_args(
    cutlass::conv::Conv2dProblemSize const& problem_size,
    Element const* input,
    Element const* weight,
    Element const* bias,
    Element* output) {
  using Conv = ConvReluOperation<ArchTag, Element>;
  using Layout = cutlass::layout::TensorNHWC;
  using TensorRef = cutlass::TensorRef<Element, Layout>;

  return typename Conv::Arguments{
      problem_size,
      TensorRef{
          const_cast<Element*>(input),
          Layout::packed(problem_size.activation_extent())},
      TensorRef{
          const_cast<Element*>(weight),
          Layout::packed(problem_size.filter_extent())},
      TensorRef{const_cast<Element*>(bias), Layout::Stride(0)},
      TensorRef{output, Layout::packed(problem_size.output_extent())},
      {1.0f}};
}

template <typename ArchTag, typename Element>
size_t conv_workspace_size(cutlass::conv::Conv2dProblemSize const& problem) {
  using Conv = ConvReluOperation<ArchTag, Element>;
  auto args = make_conv_args<ArchTag, Element>(
      problem,
      static_cast<Element const*>(nullptr),
      static_cast<Element const*>(nullptr),
      static_cast<Element const*>(nullptr),
      static_cast<Element*>(nullptr));
  return Conv::get_workspace_size(args);
}

template <typename Element>
size_t pool_workspace_size(int batch, int output_h, int output_w, int channels) {
  auto pool = make_pool<Element>(batch, output_h, output_w, channels);
  return pool.good() ? size_t(pool.workspace_size()) : 0;
}

template <typename ArchTag, typename Element>
typename ConvReluPool<ArchTag, Element>::WorkspaceLayout make_workspace_layout(
    int batch,
    int height,
    int width,
    int channels,
    int hidden_channels,
    int output_channels) {
  typename ConvReluPool<ArchTag, Element>::WorkspaceLayout layout;

  int const pool0_h = height / 2;
  int const pool0_w = width / 2;
  int const pool1_h = pool0_h / 2;
  int const pool1_w = pool0_w / 2;

  auto const conv0_problem = make_problem(
      batch, height, width, channels, hidden_channels, 3, 3, 1, 1);
  auto const conv1_problem = make_problem(
      batch, pool0_h, pool0_w, hidden_channels, output_channels, 1, 1, 0, 0);

  size_t offset = 0;
  layout.stage0_offset = offset;
  offset += align_up(tensor_bytes<Element>(
      batch, height, width, hidden_channels));

  layout.stage1_offset = offset;
  offset += align_up(tensor_bytes<Element>(
      batch, pool0_h, pool0_w, hidden_channels));

  layout.stage2_offset = offset;
  offset += align_up(tensor_bytes<Element>(
      batch, pool0_h, pool0_w, output_channels));

  layout.scratch_offset = offset;
  layout.scratch_bytes = std::max({
      conv_workspace_size<ArchTag, Element>(conv0_problem),
      conv_workspace_size<ArchTag, Element>(conv1_problem),
      pool_workspace_size<Element>(batch, pool0_h, pool0_w, hidden_channels),
      pool_workspace_size<Element>(batch, pool1_h, pool1_w, output_channels)});
  offset += align_up(layout.scratch_bytes);

  layout.total_bytes = offset;
  return layout;
}

template <typename ArchTag, typename Element>
cutlass::Status run_conv(
    cutlass::conv::Conv2dProblemSize const& problem,
    Element const* input,
    Element const* weight,
    Element const* bias,
    Element* output,
    void* workspace,
    cudaStream_t stream) {
  using Conv = ConvReluOperation<ArchTag, Element>;

  auto args =
      make_conv_args<ArchTag, Element>(problem, input, weight, bias, output);

  cutlass::Status status = Conv::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  Conv op;
  return op(args, workspace, stream);
}

template <typename Element>
cutlass::Status run_pool(
    int batch,
    int input_h,
    int input_w,
    int channels,
    Element const* input,
    Element* output,
    void* workspace,
    cudaStream_t stream) {
  int const output_h = input_h / 2;
  int const output_w = input_w / 2;
  auto pool = make_pool<Element>(batch, output_h, output_w, channels);
  if (!pool.good()) {
    return pool.status;
  }

  int64_t dst_stride[3] = {
      int64_t(output_h) * output_w * channels,
      int64_t(output_w) * channels,
      channels};
  int64_t src_stride[5] = {
      int64_t(input_h) * input_w * channels,
      int64_t(2) * input_w * channels,
      int64_t(2) * channels,
      int64_t(input_w) * channels,
      channels};

  return pool(
      output,
      dst_stride,
      input,
      src_stride,
      workspace,
      -std::numeric_limits<float>::infinity(),
      typename kernel::DefaultPool<Element>::ReductionOp(),
      stream);
}

}  // namespace

template <typename ArchTag, typename Element>
cutlass::Status ConvReluPool<ArchTag, Element>::can_implement(
    Arguments const& args) {
  if (args.batch <= 0 || args.height <= 0 || args.width <= 0 ||
      args.channels <= 0 || args.hidden_channels <= 0 ||
      args.output_channels <= 0 || (args.height % 4) != 0 ||
      (args.width % 4) != 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  if ((args.channels % 8) != 0 || (args.hidden_channels % 8) != 0 ||
      (args.output_channels % 8) != 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  return cutlass::Status::kSuccess;
}

template <typename ArchTag, typename Element>
size_t ConvReluPool<ArchTag, Element>::get_workspace_size(
    Arguments const& args) {
  if (can_implement(args) != cutlass::Status::kSuccess) {
    return 0;
  }

  return make_workspace_layout<ArchTag, Element>(
             args.batch,
             args.height,
             args.width,
             args.channels,
             args.hidden_channels,
             args.output_channels)
      .total_bytes;
}

template <typename ArchTag, typename Element>
cutlass::Status ConvReluPool<ArchTag, Element>::initialize(
    Arguments const& args,
    cudaStream_t stream) {
  (void)stream;

  cutlass::Status status = can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  auto const layout = make_workspace_layout<ArchTag, Element>(
      args.batch,
      args.height,
      args.width,
      args.channels,
      args.hidden_channels,
      args.output_channels);
  if (!args.workspace || args.workspace_bytes < layout.total_bytes) {
    return cutlass::Status::kErrorWorkspaceNull;
  }

  auto* bytes = reinterpret_cast<uint8_t*>(args.workspace);

  params_.args = args;
  params_.layout = layout;
  params_.stage0 = reinterpret_cast<Element*>(bytes + layout.stage0_offset);
  params_.stage1 = reinterpret_cast<Element*>(bytes + layout.stage1_offset);
  params_.stage2 = reinterpret_cast<Element*>(bytes + layout.stage2_offset);
  params_.scratch =
      layout.scratch_bytes ? bytes + layout.scratch_offset : nullptr;
  params_.pool0_h = args.height / 2;
  params_.pool0_w = args.width / 2;
  params_.conv0_problem = make_problem(
      args.batch,
      args.height,
      args.width,
      args.channels,
      args.hidden_channels,
      3,
      3,
      1,
      1);
  params_.conv1_problem = make_problem(
      args.batch,
      params_.pool0_h,
      params_.pool0_w,
      args.hidden_channels,
      args.output_channels,
      1,
      1,
      0,
      0);

  return cutlass::Status::kSuccess;
}

template <typename ArchTag, typename Element>
cutlass::Status ConvReluPool<ArchTag, Element>::run(cudaStream_t stream) {
  auto const& args = params_.args;

  cutlass::Status status = run_conv<ArchTag, Element>(
      params_.conv0_problem,
      args.input,
      args.weight0,
      args.bias0,
      params_.stage0,
      params_.scratch,
      stream);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  status = run_pool<Element>(
      args.batch,
      args.height,
      args.width,
      args.hidden_channels,
      params_.stage0,
      params_.stage1,
      params_.scratch,
      stream);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  status = run_conv<ArchTag, Element>(
      params_.conv1_problem,
      params_.stage1,
      args.weight1,
      args.bias1,
      params_.stage2,
      params_.scratch,
      stream);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  return run_pool<Element>(
      args.batch,
      params_.pool0_h,
      params_.pool0_w,
      args.output_channels,
      params_.stage2,
      args.output,
      params_.scratch,
      stream);
}

template <typename ArchTag, typename Element>
size_t conv_relu_pool_workspace_size(
    int batch,
    int height,
    int width,
    int channels,
    int hidden_channels,
    int output_channels) {
  typename ConvReluPool<ArchTag, Element>::Arguments args;
  args.batch = batch;
  args.height = height;
  args.width = width;
  args.channels = channels;
  args.hidden_channels = hidden_channels;
  args.output_channels = output_channels;
  return ConvReluPool<ArchTag, Element>::get_workspace_size(args);
}

template <typename ArchTag, typename Element>
cutlass::Status run_conv_relu_pool(
    int batch,
    int height,
    int width,
    int channels,
    int hidden_channels,
    int output_channels,
    Element const* input,
    Element const* weight0,
    Element const* bias0,
    Element const* weight1,
    Element const* bias1,
    Element* output,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
  typename ConvReluPool<ArchTag, Element>::Arguments args;
  args.batch = batch;
  args.height = height;
  args.width = width;
  args.channels = channels;
  args.hidden_channels = hidden_channels;
  args.output_channels = output_channels;
  args.input = input;
  args.weight0 = weight0;
  args.bias0 = bias0;
  args.weight1 = weight1;
  args.bias1 = bias1;
  args.output = output;
  args.workspace = workspace;
  args.workspace_bytes = workspace_bytes;

  ConvReluPool<ArchTag, Element> op;
  return op(args, stream);
}

template class ConvReluPool<cutlass::arch::Sm89, cutlass::half_t>;

template size_t conv_relu_pool_workspace_size<cutlass::arch::Sm89, cutlass::half_t>(
    int,
    int,
    int,
    int,
    int,
    int);

template cutlass::Status run_conv_relu_pool<cutlass::arch::Sm89, cutlass::half_t>(
    int,
    int,
    int,
    int,
    int,
    int,
    cutlass::half_t const*,
    cutlass::half_t const*,
    cutlass::half_t const*,
    cutlass::half_t const*,
    cutlass::half_t const*,
    cutlass::half_t*,
    void*,
    size_t,
    cudaStream_t);

}  // namespace tiny_cutlass::conv_fused::device
