#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "ops/conv1x1_relu_conv1x1.h"
#include "cutlass/half.h"

namespace {

using Element = cutlass::half_t;
namespace conv = tiny_cutlass::conv_fused;

struct Case {
  std::string name;
  int batch;
  int height;
  int width;
  int channels;
  int hidden;
  int output_channels;
  bool use_bias;
  cutlass::Status expected_status = cutlass::Status::kSuccess;
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
  float x = std::sin(float(index + 1) * 0.173f + phase);
  float y = std::cos(float(index + 3) * 0.071f - phase);
  return scale * (0.7f * x + 0.3f * y);
}

void fill_tensor(std::vector<Element>& tensor, float scale, float phase) {
  for (size_t i = 0; i < tensor.size(); ++i) {
    tensor[i] = Element(value_at(int(i), scale, phase));
  }
}

void fill_bias(std::vector<Element>& tensor, bool enabled, float scale, float phase) {
  for (size_t i = 0; i < tensor.size(); ++i) {
    tensor[i] = enabled ? Element(value_at(int(i), scale, phase)) : Element(0.0f);
  }
}

std::vector<float> reference(
    Case const& c,
    std::vector<Element> const& input,
    std::vector<Element> const& weight0,
    std::vector<Element> const& bias0,
    std::vector<Element> const& weight1,
    std::vector<Element> const& bias1) {
  std::vector<float> stage0(int64_t(c.batch) * c.height * c.width * c.hidden, 0.0f);
  std::vector<float> output(int64_t(c.batch) * c.height * c.width * c.output_channels, 0.0f);

  for (int n = 0; n < c.batch; ++n) {
    for (int h = 0; h < c.height; ++h) {
      for (int w = 0; w < c.width; ++w) {
        for (int k = 0; k < c.hidden; ++k) {
          float acc = float(bias0[k]);
          for (int ci = 0; ci < c.channels; ++ci) {
            acc += float(input[nhwc_index(n, h, w, ci, c.height, c.width, c.channels)]) *
                   float(weight0[krsc_index(k, ci, c.channels)]);
          }
          acc = std::max(acc, 0.0f);
          stage0[nhwc_index(n, h, w, k, c.height, c.width, c.hidden)] = acc;
        }

        for (int o = 0; o < c.output_channels; ++o) {
          float acc = float(bias1[o]);
          for (int k = 0; k < c.hidden; ++k) {
            acc += stage0[nhwc_index(n, h, w, k, c.height, c.width, c.hidden)] *
                   float(weight1[krsc_index(o, k, c.hidden)]);
          }
          output[nhwc_index(n, h, w, o, c.height, c.width, c.output_channels)] = acc;
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
  std::vector<Element> bias0(c.hidden);
  std::vector<Element> bias1(c.output_channels);
  std::vector<Element> stage0(stage0_count);
  std::vector<Element> output(output_count);

  fill_tensor(input, 0.12f, 0.11f);
  fill_tensor(weight0, 0.09f, 0.23f);
  fill_tensor(weight1, 0.08f, 0.37f);
  fill_bias(bias0, c.use_bias, 0.04f, 0.41f);
  fill_bias(bias1, c.use_bias, 0.03f, 0.53f);

  DeviceBuffer<Element> d_input(input.size());
  DeviceBuffer<Element> d_weight0(weight0.size());
  DeviceBuffer<Element> d_weight1(weight1.size());
  DeviceBuffer<Element> d_bias0(bias0.size());
  DeviceBuffer<Element> d_bias1(bias1.size());
  DeviceBuffer<Element> d_stage0(stage0.size());
  DeviceBuffer<Element> d_output(output.size());

  if (!d_input.get() || !d_weight0.get() || !d_weight1.get() || !d_bias0.get() ||
      !d_bias1.get() || !d_stage0.get() || !d_output.get()) {
    return false;
  }

  if (!d_input.copy_from_host(input) || !d_weight0.copy_from_host(weight0) ||
      !d_weight1.copy_from_host(weight1) || !d_bias0.copy_from_host(bias0) ||
      !d_bias1.copy_from_host(bias1)) {
    return false;
  }

  conv::Conv1x1ReluConv1x1Arguments<Element> args;
  args.problem = conv::Conv1x1ReluConv1x1Problem{
      c.batch,
      c.height,
      c.width,
      c.channels,
      c.hidden,
      c.output_channels};
  args.input = d_input.get();
  args.weight0 = d_weight0.get();
  args.stage0 = d_stage0.get();
  args.bias0 = d_bias0.get();
  args.weight1 = d_weight1.get();
  args.bias1 = d_bias1.get();
  args.output = d_output.get();

  cutlass::Status status = conv::conv1x1_relu_conv1x1(args);
  if (status != c.expected_status) {
    std::cerr << "case " << c.name << " returned "
              << cutlassGetStatusString(status) << ", expected "
              << cutlassGetStatusString(c.expected_status) << "\n";
    return false;
  }

  if (status != cutlass::Status::kSuccess) {
    std::cout << "pass " << c.name << " rejected with "
              << cutlassGetStatusString(status) << "\n";
    return true;
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

  std::vector<float> expected = reference(c, input, weight0, bias0, weight1, bias1);
  float max_abs = 0.0f;
  for (size_t i = 0; i < output.size(); ++i) {
    float diff = std::abs(float(output[i]) - expected[i]);
    max_abs = std::max(max_abs, diff);
    if (diff > 0.08f) {
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
            << " bias=" << c.use_bias << " max_abs=" << max_abs << "\n";
  return true;
}

}  // namespace

int main() {
  std::vector<Case> cases = {
      {"reject_unaligned", 1, 5, 7, 3, 4, 2, true,
       cutlass::Status::kErrorInvalidProblem},
      {"aligned_min", 1, 4, 4, 8, 8, 8, true},
      {"aligned_rect", 2, 3, 5, 16, 32, 16, false},
      {"aligned_wide", 1, 2, 7, 32, 16, 24, true},
      {"aligned_out64", 1, 2, 3, 8, 16, 64, true},
  };

  for (auto const& c : cases) {
    if (!run_case(c)) {
      return 1;
    }
  }

  return 0;
}
