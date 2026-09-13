#include "conv1x1_dual/ops/conv1x1_dual.h"

#include "cutlass/arch/arch.h"
#include "conv1x1_dual/device/conv1x1_dual.h"

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
cutlass::Status validate(Conv1x1DualArguments<Element> const& args) {
  auto const& p = args.problem;
  if (p.batch <= 0 || p.height <= 0 || p.width <= 0 ||
      p.channels <= 0 || p.hidden_channels <= 0 || p.output_channels <= 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  if (!args.input || !args.weight0 || !args.bias0 || !args.weight1 ||
      !args.bias1 || !args.output) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  return cutlass::Status::kSuccess;
}

// Runs the fused op for a fixed residency. The residency is a compile-time
// template parameter of the device operator, so we dispatch on the runtime
// enum in conv1x1_dual() and forward to the matching instantiation here.
template <typename Element, kernel::Residency ResidencyKind>
cutlass::Status run_with_residency(
    Conv1x1DualArguments<Element> const& args,
    cutlass::conv::Conv2dProblemSize const& problem0,
    cutlass::conv::Conv2dProblemSize const& problem1) {
  using Operation = device::Conv1x1Dual<
      cutlass::arch::Sm80, Element, ResidencyKind>;

  typename Operation::Arguments device_args;
  device_args.problem_size_0 = problem0;
  device_args.problem_size_1 = problem1;
  device_args.input = args.input;
  device_args.weight0 = args.weight0;
  device_args.bias0 = args.bias0;
  device_args.weight1 = args.weight1;
  device_args.bias1 = args.bias1;
  device_args.output = args.output;

  Operation op;
  return op(device_args, nullptr, args.stream);
}

}  // namespace

template <typename Element>
cutlass::Status conv1x1_dual(
    Conv1x1DualArguments<Element> const& args) {
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

  // Map the public runtime residency enum to the compile-time template
  // instantiation of the device operator.
  switch (args.residency) {
    case Residency::kSmem:
      return run_with_residency<Element, kernel::Residency::kSmem>(
          args, problem0, problem1);
    case Residency::kRF:
    default:
      return run_with_residency<Element, kernel::Residency::kRF>(
          args, problem0, problem1);
  }
}

template cutlass::Status conv1x1_dual<cutlass::half_t>(
    Conv1x1DualArguments<cutlass::half_t> const&);

}  // namespace tiny_cutlass::conv_fused
