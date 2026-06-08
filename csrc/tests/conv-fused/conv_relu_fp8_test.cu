#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "fp8/conv1x1_relu_conv1x1_relu_fp8/ops/conv1x1_relu_conv1x1_relu_fp8.h"
#include "cutlass/numeric_types.h"

namespace {

using Element = cutlass::float_e4m3_t;
namespace fp8 = tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu;

struct Case {
  std::string name;
  int batch;
  int height;
  int width;
  int channels;
  int hidden;
  int output_channels;
  float stage0_scale;
  float output_scale;
};

template <typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t count) {
    reset(count);
  }

  ~DeviceBuffer() {
    if (ptr_) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;

  bool reset(size_t count) {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    count_ = count;
    if (!count_) {
      return true;
    }
    cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * count_);
    if (error != cudaSuccess) {
      std::cerr << "cudaMalloc failed: " << cudaGetErrorString(error) << "\n";
      return false;
    }
    return true;
  }

  bool copy_from_host(std::vector<T> const& host) {
    cudaError_t error = cudaMemcpy(ptr_, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(error) << "\n";
      return false;
    }
    return true;
  }

  bool copy_to_host(std::vector<T>& host) const {
    cudaError_t error = cudaMemcpy(host.data(), ptr_, sizeof(T) * host.size(), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(error) << "\n";
      return false;
    }
    return true;
  }

  T* get() const {
    return ptr_;
  }

private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

int64_t nhwc_index(int n, int h, int w, int c, int height, int width, int channels) {
  return ((int64_t(n) * height + h) * width + w) * channels + c;
}

int64_t krsc_index(int k, int c, int channels) {
  return int64_t(k) * channels + c;
}

float value_at(int index, float scale, float phase) {
  float x = std::sin(float(index + 1) * 0.117f + phase);
  float y = std::cos(float(index + 5) * 0.061f - phase);
  return scale * (0.65f * x + 0.35f * y);
}

void fill_tensor(std::vector<Element>& tensor, float scale, float phase) {
  for (size_t i = 0; i < tensor.size(); ++i) {
    tensor[i] = Element(value_at(int(i), scale, phase));
  }
}

std::vector<float> reference(
    Case const& c,
    std::vector<Element> const& input,
    std::vector<Element> const& weight0,
    std::vector<Element> const& weight1) {
  float output_alpha = c.output_scale / c.stage0_scale;
  std::vector<Element> stage0(int64_t(c.batch) * c.height * c.width * c.hidden);
  std::vector<float> output(int64_t(c.batch) * c.height * c.width * c.output_channels, 0.0f);

  for (int n = 0; n < c.batch; ++n) {
    for (int h = 0; h < c.height; ++h) {
      for (int w = 0; w < c.width; ++w) {
        for (int k = 0; k < c.hidden; ++k) {
          float acc = 0.0f;
          for (int ci = 0; ci < c.channels; ++ci) {
            acc += float(input[nhwc_index(n, h, w, ci, c.height, c.width, c.channels)]) *
                   float(weight0[krsc_index(k, ci, c.channels)]);
          }
          stage0[nhwc_index(n, h, w, k, c.height, c.width, c.hidden)] =
              Element(std::max(0.0f, c.stage0_scale * acc));
        }

        for (int o = 0; o < c.output_channels; ++o) {
          float acc = 0.0f;
          for (int k = 0; k < c.hidden; ++k) {
            acc += float(stage0[nhwc_index(n, h, w, k, c.height, c.width, c.hidden)]) *
                   float(weight1[krsc_index(o, k, c.hidden)]);
          }
          Element quantized = Element(std::max(0.0f, output_alpha * acc));
          output[nhwc_index(n, h, w, o, c.height, c.width, c.output_channels)] =
              float(quantized);
        }
      }
    }
  }

  return output;
}

bool run_case(Case const& c) {
  int64_t input_count = int64_t(c.batch) * c.height * c.width * c.channels;
  int64_t stage0_count = int64_t(c.batch) * c.height * c.width * c.hidden;
  int64_t output_count = int64_t(c.batch) * c.height * c.width * c.output_channels;
  int64_t weight0_count = int64_t(c.hidden) * c.channels;
  int64_t weight1_count = int64_t(c.output_channels) * c.hidden;

  std::vector<Element> input(input_count);
  std::vector<Element> weight0(weight0_count);
  std::vector<Element> weight1(weight1_count);
  std::vector<Element> stage0(stage0_count);
  std::vector<Element> output(output_count);
  std::vector<float> stage0_scale(c.hidden, c.stage0_scale);
  std::vector<float> bias0(c.hidden, 0.0f);
  std::vector<Element> bias1(c.output_channels, Element(0.0f));

  fill_tensor(input, 0.25f, 0.13f);
  fill_tensor(weight0, 0.20f, 0.29f);
  fill_tensor(weight1, 0.18f, 0.47f);

  DeviceBuffer<Element> d_input(input.size());
  DeviceBuffer<Element> d_weight0(weight0.size());
  DeviceBuffer<Element> d_weight1(weight1.size());
  DeviceBuffer<Element> d_bias1(bias1.size());
  DeviceBuffer<Element> d_stage0(stage0.size());
  DeviceBuffer<Element> d_output(output.size());
  DeviceBuffer<float> d_stage0_scale(stage0_scale.size());
  DeviceBuffer<float> d_bias0(bias0.size());

  if (!d_input.get() || !d_weight0.get() || !d_weight1.get() || !d_bias1.get() ||
      !d_stage0.get() || !d_output.get() || !d_stage0_scale.get() || !d_bias0.get()) {
    return false;
  }

  if (!d_input.copy_from_host(input) || !d_weight0.copy_from_host(weight0) ||
      !d_weight1.copy_from_host(weight1) || !d_bias1.copy_from_host(bias1) ||
      !d_stage0_scale.copy_from_host(stage0_scale) || !d_bias0.copy_from_host(bias0)) {
    return false;
  }

  fp8::Arguments args;
  args.problem = fp8::Problem{
      c.batch,
      c.height,
      c.width,
      c.channels,
      c.hidden,
      c.output_channels};
  args.input = d_input.get();
  args.weight0 = d_weight0.get();
  args.stage0 = d_stage0.get();
  args.stage0_scale = d_stage0_scale.get();
  args.bias0 = d_bias0.get();
  args.weight1 = d_weight1.get();
  args.bias1 = d_bias1.get();
  args.output = d_output.get();
  args.stage0_alpha = c.stage0_scale;
  args.output_alpha = c.output_scale / c.stage0_scale;

  cutlass::Status status = fp8::conv1x1_relu_conv1x1_relu_fp8(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "case " << c.name << " failed can_implement/run with status "
              << static_cast<int>(status) << "\n";
    return false;
  }

  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cerr << "case " << c.name << " failed synchronize: "
              << cudaGetErrorString(error) << "\n";
    return false;
  }

  if (!d_output.copy_to_host(output)) {
    return false;
  }

  std::vector<float> expected = reference(c, input, weight0, weight1);
  float max_abs = 0.0f;
  for (size_t i = 0; i < output.size(); ++i) {
    float diff = std::abs(float(output[i]) - expected[i]);
    max_abs = std::max(max_abs, diff);
    if (diff > 1.0f) {
      std::cerr << "case " << c.name << " mismatch at " << i
                << " actual=" << float(output[i])
                << " expected=" << expected[i]
                << " diff=" << diff << "\n";
      return false;
    }
  }

  std::cout << "pass " << c.name << " shape=(" << c.batch << ","
            << c.height << "," << c.width << "," << c.channels
            << ") hidden=" << c.hidden << " out=" << c.output_channels
            << " stage0_scale=" << c.stage0_scale
            << " output_scale=" << c.output_scale
            << " max_abs=" << max_abs << "\n";
  return true;
}

}  // namespace

int main() {
  std::vector<Case> cases = {
      {"fp8_square", 1, 4, 4, 16, 16, 16, 1.0f, 1.0f},
      {"fp8_rect", 2, 3, 5, 32, 32, 16, 0.5f, 0.75f},
      {"fp8_wide", 1, 2, 3, 64, 32, 32, 1.25f, 0.5f},
  };

  for (auto const& c : cases) {
    if (!run_case(c)) {
      return 1;
    }
  }

  return 0;
}
