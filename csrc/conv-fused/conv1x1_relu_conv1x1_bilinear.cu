#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/stl.h>

namespace {

constexpr int kThreadsPerBlock = 256;

template <typename acc_t>
__host__ __forceinline__ acc_t compute_scales_value(
    std::optional<double> scale,
    int64_t input_size,
    int64_t output_size) {
  return (scale.has_value() && scale.value() > 0.0)
      ? static_cast<acc_t>(1.0 / scale.value())
      : static_cast<acc_t>(input_size) / static_cast<acc_t>(output_size);
}

template <typename acc_t>
__host__ __forceinline__ acc_t area_pixel_compute_scale(
    int64_t input_size,
    int64_t output_size,
    bool align_corners,
    std::optional<double> scale) {
  if (align_corners) {
    return output_size > 1
        ? static_cast<acc_t>(input_size - 1) / static_cast<acc_t>(output_size - 1)
        : static_cast<acc_t>(0);
  }
  return compute_scales_value<acc_t>(scale, input_size, output_size);
}

template <typename acc_t>
__device__ __forceinline__ acc_t area_pixel_compute_source_index(
    acc_t scale,
    int dst_index,
    bool align_corners) {
  if (align_corners) {
    return scale * static_cast<acc_t>(dst_index);
  }

  acc_t src_idx = scale * (static_cast<acc_t>(dst_index) + static_cast<acc_t>(0.5)) -
      static_cast<acc_t>(0.5);
  return src_idx < static_cast<acc_t>(0) ? static_cast<acc_t>(0) : src_idx;
}

__device__ __forceinline__ void compute_source_index_and_lambda(
    int& input_index0,
    int& input_index1,
    float& lambda0,
    float& lambda1,
    float scale,
    int output_index,
    int input_size,
    int output_size,
    bool align_corners) {
  if (output_size == input_size) {
    input_index0 = output_index;
    input_index1 = output_index;
    lambda0 = 1.0f;
    lambda1 = 0.0f;
    return;
  }

  float real_input_index =
      area_pixel_compute_source_index<float>(scale, output_index, align_corners);
  input_index0 = min(static_cast<int>(floorf(real_input_index)), input_size - 1);
  lambda1 = fminf(fmaxf(real_input_index - static_cast<float>(input_index0), 0.0f), 1.0f);
  input_index1 = input_index0 + (input_index0 < input_size - 1 ? 1 : 0);
  lambda0 = 1.0f - lambda1;
}

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr, int64_t offset) {
  return static_cast<float>(ptr[offset]);
}

template <typename scalar_t>
__device__ __forceinline__ float conv1x1_relu_conv1x1_at(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight0,
    const scalar_t* __restrict__ bias0,
    bool has_bias0,
    const scalar_t* __restrict__ weight1,
    const scalar_t* __restrict__ bias1,
    bool has_bias1,
    int n,
    int h,
    int w,
    int co,
    int channels_in,
    int channels_hidden,
    int height,
    int width) {
  float acc1 = has_bias1 ? load_as_float(bias1, co) : 0.0f;

  for (int ch = 0; ch < channels_hidden; ++ch) {
    float hidden = has_bias0 ? load_as_float(bias0, ch) : 0.0f;

    for (int ci = 0; ci < channels_in; ++ci) {
      int64_t input_offset =
          (((static_cast<int64_t>(n) * channels_in + ci) * height + h) * width + w);
      int64_t w0_offset = static_cast<int64_t>(ch) * channels_in + ci;
      hidden += load_as_float(input, input_offset) * load_as_float(weight0, w0_offset);
    }

    hidden = hidden > 0.0f ? hidden : 0.0f;
    int64_t w1_offset = static_cast<int64_t>(co) * channels_hidden + ch;
    acc1 += hidden * load_as_float(weight1, w1_offset);
  }

  return acc1;
}

template <typename scalar_t>
__global__ void conv1x1_relu_conv1x1_bilinear_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight0,
    const scalar_t* __restrict__ bias0,
    bool has_bias0,
    const scalar_t* __restrict__ weight1,
    const scalar_t* __restrict__ bias1,
    bool has_bias1,
    scalar_t* __restrict__ output,
    int batch,
    int channels_in,
    int height,
    int width,
    int channels_hidden,
    int channels_out,
    int output_height,
    int output_width,
    float height_scale,
    float width_scale,
    bool align_corners,
    int64_t total_elements) {
  int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear_idx >= total_elements) {
    return;
  }

  int ow = static_cast<int>(linear_idx % output_width);
  int oh = static_cast<int>((linear_idx / output_width) % output_height);
  int co = static_cast<int>((linear_idx / (static_cast<int64_t>(output_width) * output_height)) % channels_out);
  int n = static_cast<int>(linear_idx / (static_cast<int64_t>(output_width) * output_height * channels_out));

  int ih0 = 0;
  int ih1 = 0;
  int iw0 = 0;
  int iw1 = 0;
  float h0lambda = 0.0f;
  float h1lambda = 0.0f;
  float w0lambda = 0.0f;
  float w1lambda = 0.0f;

  compute_source_index_and_lambda(
      ih0, ih1, h0lambda, h1lambda,
      height_scale, oh, height, output_height, align_corners);
  compute_source_index_and_lambda(
      iw0, iw1, w0lambda, w1lambda,
      width_scale, ow, width, output_width, align_corners);

  float v00 = conv1x1_relu_conv1x1_at(
      input, weight0, bias0, has_bias0, weight1, bias1, has_bias1,
      n, ih0, iw0, co, channels_in, channels_hidden, height, width);
  float v01 = conv1x1_relu_conv1x1_at(
      input, weight0, bias0, has_bias0, weight1, bias1, has_bias1,
      n, ih0, iw1, co, channels_in, channels_hidden, height, width);
  float v10 = conv1x1_relu_conv1x1_at(
      input, weight0, bias0, has_bias0, weight1, bias1, has_bias1,
      n, ih1, iw0, co, channels_in, channels_hidden, height, width);
  float v11 = conv1x1_relu_conv1x1_at(
      input, weight0, bias0, has_bias0, weight1, bias1, has_bias1,
      n, ih1, iw1, co, channels_in, channels_hidden, height, width);

  float top = h0lambda * (w0lambda * v00 + w1lambda * v01);
  float bottom = h1lambda * (w0lambda * v10 + w1lambda * v11);
  output[linear_idx] = static_cast<scalar_t>(top + bottom);
}

void check_cuda_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.dim() == 4, name, " must be a 4D NCHW/OIHW tensor");
}

void check_bias(const at::Tensor& bias, const char* name, int64_t expected_size) {
  TORCH_CHECK(bias.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(bias.dim() == 1, name, " must be a 1D tensor");
  TORCH_CHECK(bias.size(0) == expected_size, name, " length mismatch");
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
  check_cuda_tensor(input, "input");
  check_cuda_tensor(weight0, "weight0");
  check_cuda_tensor(weight1, "weight1");

  TORCH_CHECK(input.scalar_type() == weight0.scalar_type(), "input and weight0 dtype mismatch");
  TORCH_CHECK(input.scalar_type() == weight1.scalar_type(), "input and weight1 dtype mismatch");
  TORCH_CHECK(input.size(1) == weight0.size(1), "weight0 input channel mismatch");
  TORCH_CHECK(weight0.size(2) == 1 && weight0.size(3) == 1, "weight0 must be a 1x1 OIHW kernel");
  TORCH_CHECK(weight1.size(1) == weight0.size(0), "weight1 input channel must match weight0 output channel");
  TORCH_CHECK(weight1.size(2) == 1 && weight1.size(3) == 1, "weight1 must be a 1x1 OIHW kernel");
  TORCH_CHECK(output_size.size() == 2, "output_size must contain [height, width]");
  TORCH_CHECK(output_size[0] > 0 && output_size[1] > 0, "output_size entries must be positive");
  TORCH_CHECK(input.size(2) > 0 && input.size(3) > 0, "input spatial dimensions must be positive");

  bool has_bias0 = bias0.has_value() && bias0->defined();
  bool has_bias1 = bias1.has_value() && bias1->defined();

  if (has_bias0) {
    check_bias(*bias0, "bias0", weight0.size(0));
    TORCH_CHECK(bias0->scalar_type() == input.scalar_type(), "bias0 dtype mismatch");
  }
  if (has_bias1) {
    check_bias(*bias1, "bias1", weight1.size(0));
    TORCH_CHECK(bias1->scalar_type() == input.scalar_type(), "bias1 dtype mismatch");
  }

  c10::cuda::CUDAGuard device_guard(input.device());

  at::Tensor input_c = input.contiguous();
  at::Tensor weight0_c = weight0.contiguous();
  at::Tensor weight1_c = weight1.contiguous();
  at::Tensor bias0_c = has_bias0 ? bias0->contiguous() : at::Tensor();
  at::Tensor bias1_c = has_bias1 ? bias1->contiguous() : at::Tensor();

  int64_t batch = input_c.size(0);
  int64_t channels_in = input_c.size(1);
  int64_t height = input_c.size(2);
  int64_t width = input_c.size(3);
  int64_t channels_hidden = weight0_c.size(0);
  int64_t channels_out = weight1_c.size(0);
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  TORCH_CHECK(batch <= std::numeric_limits<int>::max(), "batch is too large");
  TORCH_CHECK(channels_in <= std::numeric_limits<int>::max(), "input channels are too large");
  TORCH_CHECK(channels_hidden <= std::numeric_limits<int>::max(), "hidden channels are too large");
  TORCH_CHECK(channels_out <= std::numeric_limits<int>::max(), "output channels are too large");
  TORCH_CHECK(height <= std::numeric_limits<int>::max() && width <= std::numeric_limits<int>::max(), "input spatial size is too large");
  TORCH_CHECK(output_height <= std::numeric_limits<int>::max() && output_width <= std::numeric_limits<int>::max(), "output spatial size is too large");

  at::Tensor output = at::empty(
      {batch, channels_out, output_height, output_width},
      input_c.options());

  int64_t total_elements = output.numel();
  if (total_elements == 0) {
    return output;
  }

  float height_scale = area_pixel_compute_scale<float>(height, output_height, align_corners, scales_h);
  float width_scale = area_pixel_compute_scale<float>(width, output_width, align_corners, scales_w);

  dim3 block(kThreadsPerBlock);
  dim3 grid(static_cast<unsigned int>((total_elements + kThreadsPerBlock - 1) / kThreadsPerBlock));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_c.scalar_type(), "conv1x1_relu_conv1x1_bilinear", [&] {
    const scalar_t* bias0_ptr = has_bias0 ? bias0_c.data_ptr<scalar_t>() : nullptr;
    const scalar_t* bias1_ptr = has_bias1 ? bias1_c.data_ptr<scalar_t>() : nullptr;
    conv1x1_relu_conv1x1_bilinear_kernel<scalar_t><<<grid, block, 0, stream>>>(
        input_c.data_ptr<scalar_t>(),
        weight0_c.data_ptr<scalar_t>(),
        bias0_ptr,
        has_bias0,
        weight1_c.data_ptr<scalar_t>(),
        bias1_ptr,
        has_bias1,
        output.data_ptr<scalar_t>(),
        static_cast<int>(batch),
        static_cast<int>(channels_in),
        static_cast<int>(height),
        static_cast<int>(width),
        static_cast<int>(channels_hidden),
        static_cast<int>(channels_out),
        static_cast<int>(output_height),
        static_cast<int>(output_width),
        height_scale,
        width_scale,
        align_corners,
        total_elements);
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Fused conv1x1 -> relu -> conv1x1 -> bilinear CUDA operator";
  m.def(
      "conv1x1_relu_conv1x1_bilinear",
      &conv1x1_relu_conv1x1_bilinear,
      pybind11::arg("input"),
      pybind11::arg("weight0"),
      pybind11::arg("bias0") = std::nullopt,
      pybind11::arg("weight1"),
      pybind11::arg("bias1") = std::nullopt,
      pybind11::arg("output_size"),
      pybind11::arg("align_corners") = false,
      pybind11::arg("scales_h") = std::nullopt,
      pybind11::arg("scales_w") = std::nullopt);
}
