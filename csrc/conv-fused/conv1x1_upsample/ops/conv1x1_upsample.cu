#include "conv1x1_upsample/ops/conv1x1_upsample.h"

#include "cutlass/arch/arch.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/half.h"

#include "conv1x1_upsample/device/conv1x1_upsample.h"

namespace tiny_cutlass::conv_fused::conv1x1_upsample {
namespace {

// Ordinary conv1x1 problem at the LOW resolution: 1x1 filter, unit stride,
// zero padding. The upsample is layered on entirely in the epilogue output
// iterator (via a high-res output TensorRef), so problem_size stays low-res.
cutlass::conv::Conv2dProblemSize make_problem(
    int batch, int height, int width, int channels, int filters) {
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

template <typename Element, int UpsampleH, int UpsampleW>
cutlass::Status validate(
    Conv1x1UpsampleArguments<Element, UpsampleH, UpsampleW> const& args) {
  auto const& p = args.problem;
  if (p.batch <= 0 || p.height <= 0 || p.width <= 0 || p.channels <= 0 ||
      p.output_channels <= 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  if (!args.input || !args.weight || !args.bias || !args.output) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  return cutlass::Status::kSuccess;
}

}  // namespace

template <typename Element, int UpsampleH, int UpsampleW>
cutlass::Status conv1x1_upsample(
    Conv1x1UpsampleArguments<Element, UpsampleH, UpsampleW> const& args) {
  cutlass::Status status = validate(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  auto const& p = args.problem;
  auto problem = make_problem(p.batch, p.height, p.width, p.channels, p.output_channels);

  using Operation = device::Conv1x1Upsample<cutlass::arch::Sm80, Element, UpsampleH, UpsampleW>;

  typename Operation::Arguments device_args;
  device_args.problem_size = problem;
  device_args.input = args.input;
  device_args.weight = args.weight;
  device_args.bias = args.bias;
  device_args.output = args.output;

  Operation op;
  return op(device_args, nullptr, args.stream);
}

template cutlass::Status conv1x1_upsample<cutlass::half_t, 2, 2>(
    Conv1x1UpsampleArguments<cutlass::half_t, 2, 2> const&);

}  // namespace tiny_cutlass::conv_fused::conv1x1_upsample
