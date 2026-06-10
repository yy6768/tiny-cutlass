#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "cutlass/half.h"
#include "ops/conv_relu_pool.h"

namespace {

using DefaultElement = cutlass::half_t;
namespace conv = tiny_cutlass::conv_fused;

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * count);
    if (error != cudaSuccess) {
      std::cerr << "cudaMalloc failed: " << cudaGetErrorString(error) << "\n";
      ptr_ = nullptr;
      count_ = 0;
    }
  }

  ~DeviceBuffer() {
    if (ptr_) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;

  T* get() const {
    return ptr_;
  }

  size_t count() const {
    return count_;
  }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

bool cuda_ok(cudaError_t error, char const* what) {
  if (error != cudaSuccess) {
    std::cerr << what << " failed: " << cudaGetErrorString(error) << "\n";
    return false;
  }
  return true;
}

template <typename T>
bool copy_to_device(DeviceBuffer<T> const& dst, std::vector<T> const& src) {
  return cuda_ok(
      cudaMemcpy(dst.get(), src.data(), sizeof(T) * src.size(), cudaMemcpyHostToDevice),
      "cudaMemcpy host-to-device");
}

template <typename T>
bool copy_to_host(std::vector<T>& dst, DeviceBuffer<T> const& src) {
  return cuda_ok(
      cudaMemcpy(dst.data(), src.get(), sizeof(T) * dst.size(), cudaMemcpyDeviceToHost),
      "cudaMemcpy device-to-host");
}

size_t tensor_count(int batch, int height, int width, int channels) {
  return size_t(batch) * height * width * channels;
}

int offset4(int n, int h, int w, int c, int height, int width, int channels) {
  return ((n * height + h) * width + w) * channels + c;
}

template <typename Element>
std::vector<Element> random_tensor(size_t count, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
  std::vector<Element> tensor(count);
  for (auto& value : tensor) {
    value = Element(dist(rng));
  }
  return tensor;
}

template <typename Element>
std::vector<Element> reference(conv::ConvReluPoolProblem const& p,
                               std::vector<Element> const& input,
                               std::vector<Element> const& weight0,
                               std::vector<Element> const& bias0,
                               std::vector<Element> const& weight1,
                               std::vector<Element> const& bias1) {
  int const h0 = p.height;
  int const w0 = p.width;
  int const h1 = h0 / 2;
  int const w1 = w0 / 2;
  int const h2 = h1 / 2;
  int const w2 = w1 / 2;

  std::vector<Element> stage0(tensor_count(p.batch, h0, w0, p.hidden_channels));
  std::vector<Element> stage1(tensor_count(p.batch, h1, w1, p.hidden_channels));
  std::vector<Element> stage2(tensor_count(p.batch, h1, w1, p.output_channels));
  std::vector<Element> output(tensor_count(p.batch, h2, w2, p.output_channels));

  for (int n = 0; n < p.batch; ++n) {
    for (int h = 0; h < h0; ++h) {
      for (int w = 0; w < w0; ++w) {
        for (int k = 0; k < p.hidden_channels; ++k) {
          float acc = float(bias0[k]);
          for (int r = 0; r < 3; ++r) {
            int ih = h + r - 1;
            if (ih < 0 || ih >= h0) {
              continue;
            }
            for (int s = 0; s < 3; ++s) {
              int iw = w + s - 1;
              if (iw < 0 || iw >= w0) {
                continue;
              }
              for (int c = 0; c < p.channels; ++c) {
                int input_idx = offset4(n, ih, iw, c, h0, w0, p.channels);
                int weight_idx = ((k * 3 + r) * 3 + s) * p.channels + c;
                acc += float(input[input_idx]) * float(weight0[weight_idx]);
              }
            }
          }
          stage0[offset4(n, h, w, k, h0, w0, p.hidden_channels)] =
              Element(acc > 0.0f ? acc : 0.0f);
        }
      }
    }
  }

  for (int n = 0; n < p.batch; ++n) {
    for (int h = 0; h < h1; ++h) {
      for (int w = 0; w < w1; ++w) {
        for (int c = 0; c < p.hidden_channels; ++c) {
          float value = -std::numeric_limits<float>::infinity();
          for (int r = 0; r < 2; ++r) {
            for (int s = 0; s < 2; ++s) {
              int src_idx = offset4(n, 2 * h + r, 2 * w + s, c, h0, w0, p.hidden_channels);
              value = std::max(value, float(stage0[src_idx]));
            }
          }
          stage1[offset4(n, h, w, c, h1, w1, p.hidden_channels)] = Element(value);
        }
      }
    }
  }

  for (int n = 0; n < p.batch; ++n) {
    for (int h = 0; h < h1; ++h) {
      for (int w = 0; w < w1; ++w) {
        for (int k = 0; k < p.output_channels; ++k) {
          float acc = float(bias1[k]);
          for (int c = 0; c < p.hidden_channels; ++c) {
            int input_idx = offset4(n, h, w, c, h1, w1, p.hidden_channels);
            int weight_idx = k * p.hidden_channels + c;
            acc += float(stage1[input_idx]) * float(weight1[weight_idx]);
          }
          stage2[offset4(n, h, w, k, h1, w1, p.output_channels)] =
              Element(acc > 0.0f ? acc : 0.0f);
        }
      }
    }
  }

  for (int n = 0; n < p.batch; ++n) {
    for (int h = 0; h < h2; ++h) {
      for (int w = 0; w < w2; ++w) {
        for (int c = 0; c < p.output_channels; ++c) {
          float value = -std::numeric_limits<float>::infinity();
          for (int r = 0; r < 2; ++r) {
            for (int s = 0; s < 2; ++s) {
              int src_idx = offset4(n, 2 * h + r, 2 * w + s, c, h1, w1, p.output_channels);
              value = std::max(value, float(stage2[src_idx]));
            }
          }
          output[offset4(n, h, w, c, h2, w2, p.output_channels)] = Element(value);
        }
      }
    }
  }

  return output;
}

template <typename Element>
bool expect_status(
    char const* name,
    conv::ConvReluPoolArguments<Element> const& args,
    cutlass::Status expected) {
  cutlass::Status status = conv::conv_relu_pool(args);
  if (status != expected) {
    std::cerr << "case " << name << " returned "
              << cutlassGetStatusString(status) << ", expected "
              << cutlassGetStatusString(expected) << "\n";
    return false;
  }
  std::cout << "pass " << name << " returned "
            << cutlassGetStatusString(status) << "\n";
  return true;
}

template <typename Element>
bool run_small_correctness() {
  conv::ConvReluPoolProblem problem{1, 8, 8, 8, 8, 8};
  auto input = random_tensor<Element>(tensor_count(1, 8, 8, 8), 11);
  auto weight0 = random_tensor<Element>(size_t(8) * 3 * 3 * 8, 12);
  auto bias0 = random_tensor<Element>(8, 13);
  auto weight1 = random_tensor<Element>(size_t(8) * 1 * 1 * 8, 14);
  auto bias1 = random_tensor<Element>(8, 15);
  auto expected = reference<Element>(problem, input, weight0, bias0, weight1, bias1);

  DeviceBuffer<Element> d_input(input.size());
  DeviceBuffer<Element> d_weight0(weight0.size());
  DeviceBuffer<Element> d_bias0(bias0.size());
  DeviceBuffer<Element> d_weight1(weight1.size());
  DeviceBuffer<Element> d_bias1(bias1.size());
  DeviceBuffer<Element> d_output(expected.size());

  size_t workspace_bytes = conv::conv_relu_pool_workspace_size<Element>(problem);
  DeviceBuffer<uint8_t> d_workspace(workspace_bytes);
  if (!d_input.get() || !d_weight0.get() || !d_bias0.get() || !d_weight1.get() ||
      !d_bias1.get() || !d_output.get() || !d_workspace.get()) {
    return false;
  }

  if (!copy_to_device(d_input, input) ||
      !copy_to_device(d_weight0, weight0) ||
      !copy_to_device(d_bias0, bias0) ||
      !copy_to_device(d_weight1, weight1) ||
      !copy_to_device(d_bias1, bias1)) {
    return false;
  }

  conv::ConvReluPoolArguments<Element> args;
  args.problem = problem;
  args.input = d_input.get();
  args.weight0 = d_weight0.get();
  args.bias0 = d_bias0.get();
  args.weight1 = d_weight1.get();
  args.bias1 = d_bias1.get();
  args.output = d_output.get();
  args.workspace = d_workspace.get();
  args.workspace_bytes = workspace_bytes;

  cutlass::Status status = conv::conv_relu_pool(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "small correctness returned "
              << cutlassGetStatusString(status) << "\n";
    return false;
  }
  if (!cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    return false;
  }

  std::vector<Element> actual(expected.size());
  if (!copy_to_host(actual, d_output)) {
    return false;
  }

  float max_abs = 0.0f;
  for (size_t i = 0; i < actual.size(); ++i) {
    max_abs = std::max(max_abs, std::fabs(float(actual[i]) - float(expected[i])));
  }

  if (max_abs > 0.02f) {
    std::cerr << "small correctness max_abs=" << max_abs << "\n";
    return false;
  }

  std::cout << "pass small_correctness max_abs=" << max_abs << "\n";

  auto missing_workspace = args;
  missing_workspace.workspace = nullptr;
  if (!expect_status<Element>(
          "reject_missing_workspace",
          missing_workspace,
          cutlass::Status::kErrorWorkspaceNull)) {
    return false;
  }

  auto bad_shape = args;
  bad_shape.problem.width = 7;
  if (!expect_status<Element>(
          "reject_non_pool_aligned_shape",
          bad_shape,
          cutlass::Status::kErrorInvalidProblem)) {
    return false;
  }

  auto null_input = args;
  null_input.input = nullptr;
  if (!expect_status<Element>(
          "reject_null_input",
          null_input,
          cutlass::Status::kErrorInvalidProblem)) {
    return false;
  }

  return true;
}

}  // namespace

int main() {
  return run_small_correctness<DefaultElement>() ? 0 : 1;
}
