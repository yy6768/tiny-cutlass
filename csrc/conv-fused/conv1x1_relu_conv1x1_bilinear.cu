#include "conv_fused.h"

#include "conv_fused_utils.h"
#include "device/conv1x1_relu_conv1x1_b2b.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/conv/conv2d_problem_size.h>
#include <cutlass/half.h>

#include <optional>

namespace tiny_cutlass::conv_fused {
namespace {

cutlass::conv::Conv2dProblemSize make_1x1_problem(
    int n,
    int h,
    int w,
    int c,
    int k) {
  return cutlass::conv::Conv2dProblemSize(
      cutlass::Tensor4DCoord(n, h, w, c),
      cutlass::Tensor4DCoord(k, 1, 1, c),
      cutlass::Tensor4DCoord(0, 0, 0, 0),
      cutlass::MatrixCoord(1, 1),
      cutlass::MatrixCoord(1, 1),
      cutlass::Tensor4DCoord(n, h, w, k),
      cutlass::conv::Mode::kCrossCorrelation,
      1,
      1);
}

void check_cutlass_conv_inputs(
    const at::Tensor& input_nhwc,
    const at::Tensor& weight0_krsc,
    const at::Tensor& bias0,
    const at::Tensor& weight1_krsc,
    const at::Tensor& bias1) {
  check_cuda_4d(input_nhwc, "input_nhwc");
  check_cuda_4d(weight0_krsc, "weight0_krsc");
  check_cuda_4d(weight1_krsc, "weight1_krsc");

  TORCH_CHECK(input_nhwc.scalar_type() == weight0_krsc.scalar_type(), "input and weight0 dtype mismatch");
  TORCH_CHECK(input_nhwc.scalar_type() == weight1_krsc.scalar_type(), "input and weight1 dtype mismatch");
  TORCH_CHECK(input_nhwc.is_contiguous(at::MemoryFormat::ChannelsLast), "input_nhwc must be channels-last contiguous");
  TORCH_CHECK(weight0_krsc.is_contiguous(), "weight0_krsc must be contiguous");
  TORCH_CHECK(weight1_krsc.is_contiguous(), "weight1_krsc must be contiguous");

  TORCH_CHECK(weight0_krsc.size(1) == 1 && weight0_krsc.size(2) == 1, "weight0_krsc must be KRSC 1x1");
  TORCH_CHECK(weight1_krsc.size(1) == 1 && weight1_krsc.size(2) == 1, "weight1_krsc must be KRSC 1x1");
  TORCH_CHECK(input_nhwc.size(3) == weight0_krsc.size(3), "weight0 input channel mismatch");
  TORCH_CHECK(weight1_krsc.size(3) == weight0_krsc.size(0), "weight1 input channel must match weight0 output channel");

  check_bias(bias0, "bias0", weight0_krsc.size(0), input_nhwc.scalar_type());
  check_bias(bias1, "bias1", weight1_krsc.size(0), input_nhwc.scalar_type());
}

void check_cutlass_status(cutlass::Status status) {
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS conv-fused b2b implicit-GEMM failed with status ",
      static_cast<int>(status));
}

}  // namespace

at::Tensor conv1x1_relu_conv1x1_cutlass(
    at::Tensor input_nhwc,
    at::Tensor weight0_krsc,
    std::optional<at::Tensor> bias0,
    at::Tensor weight1_krsc,
    std::optional<at::Tensor> bias1) {
  TORCH_CHECK(bias0.has_value() && bias0->defined(), "bias0 must be materialized before calling CUTLASS path");
  TORCH_CHECK(bias1.has_value() && bias1->defined(), "bias1 must be materialized before calling CUTLASS path");
  check_cutlass_conv_inputs(input_nhwc, weight0_krsc, *bias0, weight1_krsc, *bias1);

  c10::cuda::CUDAGuard device_guard(input_nhwc.device());

  int n = checked_int64_to_int(input_nhwc.size(0), "batch");
  int h = checked_int64_to_int(input_nhwc.size(1), "height");
  int w = checked_int64_to_int(input_nhwc.size(2), "width");
  int c = checked_int64_to_int(input_nhwc.size(3), "input channels");
  int hidden = checked_int64_to_int(weight0_krsc.size(0), "hidden channels");
  int out_channels = checked_int64_to_int(weight1_krsc.size(0), "output channels");

  at::Tensor output_nhwc = at::empty(
      {input_nhwc.size(0), input_nhwc.size(1), input_nhwc.size(2), weight1_krsc.size(0)},
      input_nhwc.options().memory_format(at::MemoryFormat::ChannelsLast));

  if (output_nhwc.numel() == 0) {
    return output_nhwc;
  }

  auto problem0 = make_1x1_problem(n, h, w, c, hidden);
  auto problem1 = make_1x1_problem(n, h, w, hidden, out_channels);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (input_nhwc.scalar_type() == at::ScalarType::Float) {
    cutlass::Status status = device::run_conv1x1_relu_conv1x1_b2b_sm80<float>(
        problem0,
        problem1,
        input_nhwc.data_ptr<float>(),
        weight0_krsc.data_ptr<float>(),
        bias0->data_ptr<float>(),
        weight1_krsc.data_ptr<float>(),
        bias1->data_ptr<float>(),
        output_nhwc.data_ptr<float>(),
        stream);
    check_cutlass_status(status);
  } else if (input_nhwc.scalar_type() == at::ScalarType::Half) {
    cutlass::Status status = device::run_conv1x1_relu_conv1x1_b2b_sm80<cutlass::half_t>(
        problem0,
        problem1,
        reinterpret_cast<cutlass::half_t const*>(input_nhwc.data_ptr<at::Half>()),
        reinterpret_cast<cutlass::half_t const*>(weight0_krsc.data_ptr<at::Half>()),
        reinterpret_cast<cutlass::half_t const*>(bias0->data_ptr<at::Half>()),
        reinterpret_cast<cutlass::half_t const*>(weight1_krsc.data_ptr<at::Half>()),
        reinterpret_cast<cutlass::half_t const*>(bias1->data_ptr<at::Half>()),
        reinterpret_cast<cutlass::half_t*>(output_nhwc.data_ptr<at::Half>()),
        stream);
    check_cutlass_status(status);
  } else {
    TORCH_CHECK(false, "conv-fused CUTLASS path currently supports float32 and float16");
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output_nhwc;
}

}  // namespace tiny_cutlass::conv_fused
