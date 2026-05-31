#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace tiny_cutlass::conv_fused {

at::Tensor conv1x1_relu_conv1x1_cutlass(
    at::Tensor input_nhwc,
    at::Tensor weight0_krsc,
    std::optional<at::Tensor> bias0,
    at::Tensor weight1_krsc,
    std::optional<at::Tensor> bias1);

at::Tensor conv1x1_relu_conv1x1_bilinear(
    const at::Tensor& input,
    const at::Tensor& weight0,
    const std::optional<at::Tensor>& bias0,
    const at::Tensor& weight1,
    const std::optional<at::Tensor>& bias1,
    std::vector<int64_t> output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w);

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
    std::optional<int64_t> divisor_override);

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
    bool ceil_mode);

}  // namespace tiny_cutlass::conv_fused
