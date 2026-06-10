#include "ops/conv_relu_pool.h"

#include "cutlass/half.h"

#include "device/conv_relu_pool.h"

namespace tiny_cutlass::conv_fused {
namespace {

template <typename Element>
cutlass::Status validate(ConvReluPoolArguments<Element> const& args) {
  auto const& p = args.problem;
  if (p.batch <= 0 || p.height <= 0 || p.width <= 0 ||
      p.channels <= 0 || p.hidden_channels <= 0 || p.output_channels <= 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  if ((p.height % 4) != 0 || (p.width % 4) != 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  if (!args.input || !args.weight0 || !args.bias0 ||
      !args.weight1 || !args.bias1 || !args.output) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  return cutlass::Status::kSuccess;
}

}  // namespace

template <typename Element>
cutlass::Status conv_relu_pool(ConvReluPoolArguments<Element> const& args) {
  cutlass::Status status = validate(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  auto const& p = args.problem;
  return device::run_conv_relu_pool(
      p.batch,
      p.height,
      p.width,
      p.channels,
      p.hidden_channels,
      p.output_channels,
      args.input,
      args.weight0,
      args.bias0,
      args.weight1,
      args.bias1,
      args.output,
      args.workspace,
      args.workspace_bytes,
      args.stream);
}

template <typename Element>
size_t conv_relu_pool_workspace_size(ConvReluPoolProblem const& problem) {
  return device::conv_relu_pool_workspace_size<cutlass::arch::Sm89, Element>(
      problem.batch,
      problem.height,
      problem.width,
      problem.channels,
      problem.hidden_channels,
      problem.output_channels);
}

template cutlass::Status conv_relu_pool<cutlass::half_t>(
    ConvReluPoolArguments<cutlass::half_t> const&);

template size_t conv_relu_pool_workspace_size<cutlass::half_t>(
    ConvReluPoolProblem const&);

}  // namespace tiny_cutlass::conv_fused
