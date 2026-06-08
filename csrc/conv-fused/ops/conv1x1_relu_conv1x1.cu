#include "ops/conv1x1_relu_conv1x1.h"

#include "device/conv1x1_relu_conv1x1.h"

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/half.h"

namespace tiny_cutlass::conv_fused {
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

template <typename Element>
cutlass::Status validate(Conv1x1ReluConv1x1Arguments<Element> const& args) {
  auto const& p = args.problem;
  if (p.batch <= 0 || p.height <= 0 || p.width <= 0 ||
      p.channels <= 0 || p.hidden_channels <= 0 || p.output_channels <= 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  if (!args.input || !args.weight0 || !args.stage0 || !args.bias0 ||
      !args.weight1 || !args.bias1 || !args.output) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  return cutlass::Status::kSuccess;
}

}  // namespace

template <typename Element>
cutlass::Status conv1x1_relu_conv1x1(
    Conv1x1ReluConv1x1Arguments<Element> const& args) {
  cutlass::Status status = validate(args);
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

  return device::run_conv1x1_relu_conv1x1(
      problem0,
      problem1,
      args.input,
      args.weight0,
      args.stage0,
      args.bias0,
      args.weight1,
      args.bias1,
      args.output,
      args.stream);
}

template cutlass::Status conv1x1_relu_conv1x1<cutlass::half_t>(
    Conv1x1ReluConv1x1Arguments<cutlass::half_t> const&);

}  // namespace tiny_cutlass::conv_fused
