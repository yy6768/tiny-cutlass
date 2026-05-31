#include "conv_fused.h"

#include "conv_fused_utils.h"

#include <ATen/ops/avg_pool2d.h>
#include <ATen/ops/max_pool2d.h>
#include <ATen/ops/upsample_bilinear2d.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/stl.h>

#include <array>
#include <optional>
#include <vector>

namespace tiny_cutlass::conv_fused {
namespace {

void check_conv_chain_inputs(
    const at::Tensor& input,
    const at::Tensor& weight0,
    const std::optional<at::Tensor>& bias0,
    const at::Tensor& weight1,
    const std::optional<at::Tensor>& bias1) {
  check_cuda_4d(input, "input");
  check_cuda_4d(weight0, "weight0");
  check_cuda_4d(weight1, "weight1");

  TORCH_CHECK(input.scalar_type() == weight0.scalar_type(), "input and weight0 dtype mismatch");
  TORCH_CHECK(input.scalar_type() == weight1.scalar_type(), "input and weight1 dtype mismatch");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float ||
          input.scalar_type() == at::ScalarType::Half,
      "conv-fused CUTLASS path currently supports float32 and float16");

  TORCH_CHECK(input.size(1) == weight0.size(1), "weight0 input channel mismatch");
  TORCH_CHECK(weight0.size(2) == 1 && weight0.size(3) == 1, "weight0 must be a 1x1 OIHW kernel");
  TORCH_CHECK(weight1.size(1) == weight0.size(0), "weight1 input channel must match weight0 output channel");
  TORCH_CHECK(weight1.size(2) == 1 && weight1.size(3) == 1, "weight1 must be a 1x1 OIHW kernel");
  TORCH_CHECK(input.size(2) > 0 && input.size(3) > 0, "input spatial dimensions must be positive");

  if (auto normalized = normalize_optional_tensor(bias0)) {
    check_bias(*normalized, "bias0", weight0.size(0), input.scalar_type());
  }
  if (auto normalized = normalize_optional_tensor(bias1)) {
    check_bias(*normalized, "bias1", weight1.size(0), input.scalar_type());
  }
}

at::Tensor nchw_to_nhwc_contiguous(const at::Tensor& input) {
  return input.contiguous(at::MemoryFormat::ChannelsLast);
}

at::Tensor oihw_to_krsc_contiguous(const at::Tensor& weight) {
  return weight.permute({0, 2, 3, 1}).contiguous();
}

at::Tensor nhwc_to_nchw_contiguous(const at::Tensor& input) {
  return input.permute({0, 3, 1, 2}).contiguous();
}

std::array<at::Tensor, 5> prepare_conv_chain(
    const at::Tensor& input,
    const at::Tensor& weight0,
    const std::optional<at::Tensor>& bias0,
    const at::Tensor& weight1,
    const std::optional<at::Tensor>& bias1) {
  at::Tensor input_nhwc = nchw_to_nhwc_contiguous(input);
  at::Tensor weight0_krsc = oihw_to_krsc_contiguous(weight0);
  at::Tensor weight1_krsc = oihw_to_krsc_contiguous(weight1);

  at::Tensor bias0_c = normalize_optional_tensor(bias0)
      ? bias0->contiguous()
      : at::zeros({weight0.size(0)}, input.options());
  at::Tensor bias1_c = normalize_optional_tensor(bias1)
      ? bias1->contiguous()
      : at::zeros({weight1.size(0)}, input.options());

  return {input_nhwc, weight0_krsc, bias0_c, weight1_krsc, bias1_c};
}

at::Tensor run_conv_chain_nchw(
    const at::Tensor& input,
    const at::Tensor& weight0,
    const std::optional<at::Tensor>& bias0,
    const at::Tensor& weight1,
    const std::optional<at::Tensor>& bias1) {
  check_conv_chain_inputs(input, weight0, bias0, weight1, bias1);
  c10::cuda::CUDAGuard device_guard(input.device());

  auto prepared = prepare_conv_chain(input, weight0, bias0, weight1, bias1);
  at::Tensor output_nhwc = conv1x1_relu_conv1x1_cutlass(
      prepared[0],
      prepared[1],
      prepared[2],
      prepared[3],
      prepared[4]);

  return nhwc_to_nchw_contiguous(output_nhwc);
}

}  // namespace

at::Tensor conv1x1_relu_conv1x1_bilinear(
    const at::Tensor& input,
    const at::Tensor& weight0,
    const std::optional<at::Tensor>& bias0,
    const at::Tensor& weight1,
    const std::optional<at::Tensor>& bias1,
    std::vector<int64_t> output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TORCH_CHECK(output_size.size() == 2, "output_size must contain [height, width]");
  TORCH_CHECK(output_size[0] > 0 && output_size[1] > 0, "output_size entries must be positive");

  at::Tensor conv = run_conv_chain_nchw(input, weight0, bias0, weight1, bias1);
  return at::upsample_bilinear2d(conv, output_size, align_corners, scales_h, scales_w);
}

at::Tensor conv1x1_relu_conv1x1_avg_pool2d(
    const at::Tensor& input,
    const at::Tensor& weight0,
    const std::optional<at::Tensor>& bias0,
    const at::Tensor& weight1,
    const std::optional<at::Tensor>& bias1,
    std::vector<int64_t> kernel_size,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0, "divisor_override must be non-zero");

  auto kernel = expand_2d(kernel_size, "kernel_size");
  auto stride_2d = expand_2d_or(stride, "stride", kernel);
  auto padding_2d = expand_2d_or(padding, "padding", {0, 0});
  check_pool_args(kernel, stride_2d, padding_2d, {1, 1}, true);

  at::Tensor conv = run_conv_chain_nchw(input, weight0, bias0, weight1, bias1);
  return at::avg_pool2d(
      conv,
      kernel,
      stride_2d,
      padding_2d,
      ceil_mode,
      count_include_pad,
      divisor_override);
}

at::Tensor conv1x1_relu_conv1x1_max_pool2d(
    const at::Tensor& input,
    const at::Tensor& weight0,
    const std::optional<at::Tensor>& bias0,
    const at::Tensor& weight1,
    const std::optional<at::Tensor>& bias1,
    std::vector<int64_t> kernel_size,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    bool ceil_mode) {
  auto kernel = expand_2d(kernel_size, "kernel_size");
  auto stride_2d = expand_2d_or(stride, "stride", kernel);
  auto padding_2d = expand_2d_or(padding, "padding", {0, 0});
  auto dilation_2d = expand_2d_or(dilation, "dilation", {1, 1});
  check_pool_args(kernel, stride_2d, padding_2d, dilation_2d, false);

  at::Tensor conv = run_conv_chain_nchw(input, weight0, bias0, weight1, bias1);
  return at::max_pool2d(conv, kernel, stride_2d, padding_2d, dilation_2d, ceil_mode);
}

}  // namespace tiny_cutlass::conv_fused

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace fused = tiny_cutlass::conv_fused;

  m.doc() = "CUTLASS b2b implicit-GEMM conv1x1 -> relu -> conv1x1 with ATen spatial tails";
  m.def(
      "conv1x1_relu_conv1x1_bilinear",
      &fused::conv1x1_relu_conv1x1_bilinear,
      pybind11::arg("input"),
      pybind11::arg("weight0"),
      pybind11::arg("bias0") = std::nullopt,
      pybind11::arg("weight1"),
      pybind11::arg("bias1") = std::nullopt,
      pybind11::arg("output_size"),
      pybind11::arg("align_corners") = false,
      pybind11::arg("scales_h") = std::nullopt,
      pybind11::arg("scales_w") = std::nullopt);
  m.def(
      "conv1x1_relu_conv1x1_avg_pool2d",
      &fused::conv1x1_relu_conv1x1_avg_pool2d,
      pybind11::arg("input"),
      pybind11::arg("weight0"),
      pybind11::arg("bias0") = std::nullopt,
      pybind11::arg("weight1"),
      pybind11::arg("bias1") = std::nullopt,
      pybind11::arg("kernel_size"),
      pybind11::arg("stride") = std::vector<int64_t>{},
      pybind11::arg("padding") = std::vector<int64_t>{0, 0},
      pybind11::arg("ceil_mode") = false,
      pybind11::arg("count_include_pad") = true,
      pybind11::arg("divisor_override") = std::nullopt);
  m.def(
      "conv1x1_relu_conv1x1_max_pool2d",
      &fused::conv1x1_relu_conv1x1_max_pool2d,
      pybind11::arg("input"),
      pybind11::arg("weight0"),
      pybind11::arg("bias0") = std::nullopt,
      pybind11::arg("weight1"),
      pybind11::arg("bias1") = std::nullopt,
      pybind11::arg("kernel_size"),
      pybind11::arg("stride") = std::vector<int64_t>{},
      pybind11::arg("padding") = std::vector<int64_t>{0, 0},
      pybind11::arg("dilation") = std::vector<int64_t>{1, 1},
      pybind11::arg("ceil_mode") = false);
}
