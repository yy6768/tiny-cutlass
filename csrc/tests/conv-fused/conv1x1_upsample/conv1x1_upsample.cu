/*
  Correctness test for conv1x1 -> nearest-neighbor 2x2 upsample.

  A problem-size sweep runs through the raw-pointer core API
  (conv1x1_upsample/ops/conv1x1_upsample.h), each checked against an independent
  host reference that computes conv1x1 at the low resolution and then
  nearest-neighbor upsamples by replicating each output pixel into its 2x2
  block (no shared code with the CUTLASS kernel path).
*/

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "conv1x1_upsample/ops/conv1x1_upsample.h"
#include "cutlass/half.h"

namespace {

using DefaultElement = cutlass::half_t;
namespace conv1x1_upsample = tiny_cutlass::conv_fused::conv1x1_upsample;

static int const kUpsampleH = 2;
static int const kUpsampleW = 2;

struct Case {
  std::string name;
  int batch;
  int height;   // low-res height
  int width;    // low-res width
  int channels;
  int output_channels;
  bool use_bias;
  cutlass::Status expected_status = cutlass::Status::kSuccess;
};

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t count) { reset(count); }

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

  T* get() const { return ptr_; }

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

template <typename Element>
void fill_tensor(std::vector<Element>& tensor, float scale, float phase) {
  for (size_t i = 0; i < tensor.size(); ++i) {
    tensor[i] = Element(value_at(int(i), scale, phase));
  }
}

template <typename Element>
void fill_bias(std::vector<Element>& tensor, bool enabled, float scale, float phase) {
  for (size_t i = 0; i < tensor.size(); ++i) {
    tensor[i] = enabled ? Element(value_at(int(i), scale, phase)) : Element(0.0f);
  }
}

// Independent host reference: conv1x1 (bias, no activation) at the low
// resolution, then nearest-neighbor 2x2 upsample by replicating each output
// pixel into its 2x2 block. Deliberately does not reuse the upsample-scatter
// trick from the CUTLASS kernel path.
template <typename Element>
std::vector<float> reference(
    Case const& c,
    std::vector<Element> const& input,
    std::vector<Element> const& weight,
    std::vector<Element> const& bias) {
  int hi_h = c.height * kUpsampleH;
  int hi_w = c.width * kUpsampleW;

  std::vector<float> low(int64_t(c.batch) * c.height * c.width * c.output_channels, 0.0f);
  std::vector<float> output(int64_t(c.batch) * hi_h * hi_w * c.output_channels, 0.0f);

  for (int n = 0; n < c.batch; ++n) {
    for (int p = 0; p < c.height; ++p) {
      for (int q = 0; q < c.width; ++q) {
        for (int o = 0; o < c.output_channels; ++o) {
          float acc = float(bias[o]);
          for (int ci = 0; ci < c.channels; ++ci) {
            acc += float(input[nhwc_index(n, p, q, ci, c.height, c.width, c.channels)]) *
                   float(weight[krsc_index(o, ci, c.channels)]);
          }
          low[nhwc_index(n, p, q, o, c.height, c.width, c.output_channels)] = acc;
        }
      }
    }

    for (int p = 0; p < c.height; ++p) {
      for (int q = 0; q < c.width; ++q) {
        for (int o = 0; o < c.output_channels; ++o) {
          float v = low[nhwc_index(n, p, q, o, c.height, c.width, c.output_channels)];
          for (int dh = 0; dh < kUpsampleH; ++dh) {
            for (int dw = 0; dw < kUpsampleW; ++dw) {
              output[nhwc_index(
                  n, kUpsampleH * p + dh, kUpsampleW * q + dw, o, hi_h, hi_w, c.output_channels)] = v;
            }
          }
        }
      }
    }
  }

  return output;
}

template <typename Element>
bool run_case(Case const& c) {
  int hi_h = c.height * kUpsampleH;
  int hi_w = c.width * kUpsampleW;

  int64_t input_count = int64_t(c.batch) * c.height * c.width * c.channels;
  int64_t output_count = int64_t(c.batch) * hi_h * hi_w * c.output_channels;
  int64_t weight_count = int64_t(c.output_channels) * c.channels;

  std::vector<Element> input(input_count);
  std::vector<Element> weight(weight_count);
  std::vector<Element> bias(c.output_channels);
  std::vector<Element> output(output_count);

  fill_tensor<Element>(input, 0.12f, 0.11f);
  fill_tensor<Element>(weight, 0.09f, 0.23f);
  fill_bias<Element>(bias, c.use_bias, 0.04f, 0.41f);

  DeviceBuffer<Element> d_input(input.size());
  DeviceBuffer<Element> d_weight(weight.size());
  DeviceBuffer<Element> d_bias(bias.size());
  DeviceBuffer<Element> d_output(output.size());

  if (!d_input.get() || !d_weight.get() || !d_bias.get() || !d_output.get()) {
    return false;
  }

  if (!d_input.copy_from_host(input) || !d_weight.copy_from_host(weight) ||
      !d_bias.copy_from_host(bias)) {
    return false;
  }

  conv1x1_upsample::Conv1x1UpsampleArguments<Element, kUpsampleH, kUpsampleW> args;
  args.problem = conv1x1_upsample::Conv1x1UpsampleProblem{
      c.batch, c.height, c.width, c.channels, c.output_channels};
  args.input = d_input.get();
  args.weight = d_weight.get();
  args.bias = d_bias.get();
  args.output = d_output.get();

  cutlass::Status status =
      conv1x1_upsample::conv1x1_upsample<Element, kUpsampleH, kUpsampleW>(args);
  if (status != c.expected_status) {
    std::cerr << "case " << c.name << " returned " << cutlassGetStatusString(status)
              << ", expected " << cutlassGetStatusString(c.expected_status) << "\n";
    return false;
  }

  if (status != cutlass::Status::kSuccess) {
    std::cout << "pass " << c.name << " rejected with " << cutlassGetStatusString(status) << "\n";
    return true;
  }

  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cerr << "case " << c.name << " failed synchronize: " << cudaGetErrorString(error) << "\n";
    return false;
  }

  if (!d_output.copy_to_host(output)) {
    return false;
  }

  std::vector<float> expected = reference<Element>(c, input, weight, bias);
  float max_abs = 0.0f;
  for (size_t i = 0; i < output.size(); ++i) {
    float diff = std::abs(float(output[i]) - expected[i]);
    max_abs = std::max(max_abs, diff);
    if (diff > 0.08f) {
      std::cerr << "case " << c.name << " mismatch at " << i << " actual=" << float(output[i])
                << " expected=" << expected[i] << " diff=" << diff << "\n";
      return false;
    }
  }

  std::cout << "pass " << c.name << " low_shape=(" << c.batch << "," << c.height << "," << c.width
            << "," << c.channels << ") out=" << c.output_channels
            << " up=(" << kUpsampleH << "," << kUpsampleW << ")"
            << " bias=" << c.use_bias << " max_abs=" << max_abs << "\n";
  return true;
}

template <typename Element>
bool run_all() {
  std::vector<Case> cases = {
      {"aligned_min", 1, 4, 4, 8, 8, true},
      {"aligned_rect", 2, 3, 5, 16, 16, false},
      {"aligned_wide", 1, 2, 7, 32, 24, true},
      {"aligned_out64", 1, 3, 3, 8, 64, true},
      {"aligned_deep", 2, 4, 4, 64, 32, true},
  };

  for (auto const& c : cases) {
    if (!run_case<Element>(c)) {
      return false;
    }
  }

  return true;
}

}  // namespace

int main() {
  return run_all<DefaultElement>() ? 0 : 1;
}
