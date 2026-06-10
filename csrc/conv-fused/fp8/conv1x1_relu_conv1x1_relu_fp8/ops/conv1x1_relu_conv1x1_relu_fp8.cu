#include "fp8/conv1x1_relu_conv1x1_relu_fp8/ops/conv1x1_relu_conv1x1_relu_fp8.h"

#include "cutlass/conv/conv2d_problem_size.h"

#include "fp8/conv1x1_relu_conv1x1_relu_fp8/device/conv1x1_relu_conv1x1_relu_fp8.h"

namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu {
namespace {

cutlass::conv::Conv2dProblemSize make_problem(
    int batch,
    int height,
    int width,
    int channels,
    int filters) {
  return cutlass::conv::Conv2dProblemSize(
      cutlass::Tensor4DCoord(batch, height, width, channels),
      cutlass::Tensor4DCoord(filters, 1, 1, channels),
      cutlass::Tensor4DCoord(0, 0, 0, 0),
      cutlass::MatrixCoord(1, 1),
      cutlass::MatrixCoord(1, 1),
      cutlass::Tensor4DCoord(batch, height, width, filters),
      cutlass::conv::Mode::kCrossCorrelation,
      1,
      1);
}

template <typename Element, typename ElementScaleBias, typename ElementCompute>
cutlass::Status validate(
    Arguments<Element, ElementScaleBias, ElementCompute> const& args) {
  auto const& p = args.problem;
  if (p.batch <= 0 || p.height <= 0 || p.width <= 0 ||
      p.channels <= 0 || p.hidden_channels <= 0 || p.output_channels <= 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  if ((p.channels % 16) != 0 || (p.hidden_channels % 16) != 0 ||
      (p.output_channels % 16) != 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  if (!args.input || !args.weight0 || !args.stage0 || !args.stage0_scale ||
      !args.bias0 || !args.weight1 || !args.bias1 || !args.output) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  if (!(args.stage0_alpha > 0.0f) || !(args.output_alpha > 0.0f)) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  return cutlass::Status::kSuccess;
}

}  // namespace

template <typename Element, typename ElementScaleBias, typename ElementCompute>
cutlass::Status conv1x1_relu_conv1x1_relu(
    Arguments<Element, ElementScaleBias, ElementCompute> const& args) {
  cutlass::Status status =
      validate<Element, ElementScaleBias, ElementCompute>(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  auto const& p = args.problem;
  auto problem0 = make_problem(
      p.batch,
      p.height,
      p.width,
      p.channels,
      p.hidden_channels);
  auto problem1 = make_problem(
      p.batch,
      p.height,
      p.width,
      p.hidden_channels,
      p.output_channels);

  return device::run_conv1x1_relu_conv1x1_relu_fp8<
      cutlass::arch::Sm89,
      Element,
      Element,
      Element,
      ElementScaleBias,
      float,
      ElementCompute>(
      problem0,
      problem1,
      args.input,
      args.weight0,
      args.stage0,
      args.stage0_scale,
      args.bias0,
      args.weight1,
      args.bias1,
      args.output,
      args.stage0_alpha,
      args.output_alpha,
      args.stream);
}

cutlass::Status conv1x1_relu_conv1x1_relu_fp8(E4m3Arguments const& args) {
  return conv1x1_relu_conv1x1_relu<
      cutlass::float_e4m3_t,
      float,
      float>(args);
}

template cutlass::Status conv1x1_relu_conv1x1_relu<
    cutlass::float_e4m3_t,
    float,
    float>(E4m3Arguments const&);

}  // namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu
