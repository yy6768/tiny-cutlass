#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/arch/mma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"

#include "../../swin/device/patch_embed.h"
#include "../../swin/kernel/default_patch_embed.h"
#include "../../swin/swin_problem.h"

namespace tiny_cutlass {
namespace swin {
namespace {

using Kernel = typename kernel::DefaultPatchEmbed<
    cutlass::arch::Sm80,
    cutlass::half_t>::Kernel;
using Op = device::PatchEmbed<Kernel>;
using Element = typename Op::Element;

struct Options {
  bool error = false;
  int batch_size = 1;
  int image_size = 224;
  int in_channels = 3;
  int input_channels_padded = 8;
  int embed_dim = 96;
  int patch_size = 4;
  int iterations = 20;
  int seed = 2026;
  float epsilon = 1.0e-5f;
  float abs_tolerance = 2.5e-2f;
  float rel_tolerance = 2.5e-2f;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("batch_size", batch_size, batch_size);
    cmd.get_cmd_line_argument("image_size", image_size, image_size);
    cmd.get_cmd_line_argument("in_channels", in_channels, in_channels);
    cmd.get_cmd_line_argument(
        "input_channels_padded", input_channels_padded, input_channels_padded);
    cmd.get_cmd_line_argument("embed_dim", embed_dim, embed_dim);
    cmd.get_cmd_line_argument("patch_size", patch_size, patch_size);
    cmd.get_cmd_line_argument("iterations", iterations, iterations);
    cmd.get_cmd_line_argument("seed", seed, seed);
    cmd.get_cmd_line_argument("epsilon", epsilon, epsilon);
    cmd.get_cmd_line_argument("abs_tolerance", abs_tolerance, abs_tolerance);
    cmd.get_cmd_line_argument("rel_tolerance", rel_tolerance, rel_tolerance);
    error = batch_size <= 0 || image_size <= 0 || in_channels <= 0 ||
        input_channels_padded <= 0 || embed_dim <= 0 || patch_size <= 0 ||
        iterations <= 0 || epsilon <= 0.0f;
  }

  PatchEmbedProblem problem() const {
    PatchEmbedProblem p;
    p.batch_size = batch_size;
    p.image_size = image_size;
    p.in_channels = in_channels;
    p.input_channels_padded = input_channels_padded;
    p.embed_dim = embed_dim;
    p.patch_size = patch_size;
    p.layernorm_eps = epsilon;
    return p;
  }
};

void fill_random(std::vector<Element>& data, int seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (Element& value : data) {
    value = Element(dist(rng));
  }
}

void patch_embed_reference(
    PatchEmbedProblem const& p,
    std::vector<Element> const& input,
    std::vector<Element> const& kernel,
    std::vector<Element> const& bias,
    std::vector<Element> const& gamma,
    std::vector<Element> const& beta,
    std::vector<Element>& output) {
  int out = patch_embed_output_size(p);
  output.assign(patch_embed_output_elements(p), Element(0));

  std::vector<float> token(p.embed_dim);
  for (int b = 0; b < p.batch_size; ++b) {
    for (int oy = 0; oy < out; ++oy) {
      for (int ox = 0; ox < out; ++ox) {
        for (int k = 0; k < p.embed_dim; ++k) {
          float acc = float(bias[k]);
          for (int c = 0; c < p.in_channels; ++c) {
            for (int r = 0; r < p.patch_size; ++r) {
              for (int s = 0; s < p.patch_size; ++s) {
                int iy = oy * p.patch_size + r;
                int ix = ox * p.patch_size + s;
                int64_t input_idx =
                    ((int64_t(b) * p.image_size + iy) * p.image_size + ix) *
                        p.in_channels +
                    c;
                int64_t kernel_idx =
                    ((int64_t(k) * p.in_channels + c) * p.patch_size + r) *
                        p.patch_size +
                    s;
                acc += float(input[input_idx]) * float(kernel[kernel_idx]);
              }
            }
          }
          token[k] = acc;
        }

        float sum = 0.0f;
        float square_sum = 0.0f;
        for (float value : token) {
          sum += value;
          square_sum += value * value;
        }
        float mean = sum / float(p.embed_dim);
        float variance = square_sum / float(p.embed_dim) - mean * mean;
        float inv_std = 1.0f / std::sqrt(variance + p.layernorm_eps);

        int64_t token_idx = (int64_t(b) * out * out + oy * out + ox) * p.embed_dim;
        for (int k = 0; k < p.embed_dim; ++k) {
          float value =
              (token[k] - mean) * inv_std * float(gamma[k]) + float(beta[k]);
          output[token_idx + k] = Element(value);
        }
      }
    }
  }
}

bool compare(
    std::vector<Element> const& actual,
    std::vector<Element> const& expected,
    float abs_tolerance,
    float rel_tolerance) {
  double abs_sum = 0.0;
  float max_abs = 0.0f;
  int64_t max_index = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    float a = float(actual[i]);
    float e = float(expected[i]);
    float diff = std::fabs(a - e);
    float rel = diff / (std::fabs(e) + 1.0e-5f);
    abs_sum += double(diff);
    if (diff > max_abs) {
      max_abs = diff;
      max_index = int64_t(i);
    }
    if (!std::isfinite(a) || (diff > abs_tolerance && rel > rel_tolerance)) {
      std::cerr << "Mismatch at " << i << ": actual=" << a
                << " expected=" << e << " diff=" << diff
                << " rel=" << rel << "\n";
      return false;
    }
  }
  std::cout << "    MAE     : " << (abs_sum / double(actual.size())) << "\n"
            << "    Max abs : " << max_abs << " at index " << max_index << "\n";
  return true;
}

int run(Options const& options) {
  PatchEmbedProblem problem = options.problem();
  char const* reason = nullptr;
  if (!Op::can_implement(problem, &reason)) {
    std::cerr << "Unsupported PatchEmbed problem: " << reason << "\n";
    return -1;
  }

  std::vector<Element> host_input(patch_embed_input_elements(problem));
  std::vector<Element> host_kernel(patch_embed_kernel_elements(problem));
  std::vector<Element> host_bias(problem.embed_dim);
  std::vector<Element> host_gamma(problem.embed_dim);
  std::vector<Element> host_beta(problem.embed_dim);
  fill_random(host_input, options.seed, -1.0f, 1.0f);
  fill_random(host_kernel, options.seed + 1, -0.08f, 0.08f);
  fill_random(host_bias, options.seed + 2, -0.05f, 0.05f);
  fill_random(host_gamma, options.seed + 3, 0.8f, 1.2f);
  fill_random(host_beta, options.seed + 4, -0.05f, 0.05f);

  int64_t workspace_elements =
      patch_embed_output_elements(problem) +
      patch_embed_input_padded_elements(problem) +
      patch_embed_kernel_padded_elements(problem);
  cutlass::DeviceAllocation<Element> input(patch_embed_input_elements(problem));
  cutlass::DeviceAllocation<Element> kernel(patch_embed_kernel_elements(problem));
  cutlass::DeviceAllocation<Element> bias(problem.embed_dim);
  cutlass::DeviceAllocation<Element> gamma(problem.embed_dim);
  cutlass::DeviceAllocation<Element> beta(problem.embed_dim);
  cutlass::DeviceAllocation<Element> workspace(workspace_elements);
  cutlass::DeviceAllocation<Element> output(patch_embed_output_elements(problem));

  cutlass::device_memory::copy_to_device(input.get(), host_input.data(), host_input.size());
  cutlass::device_memory::copy_to_device(kernel.get(), host_kernel.data(), host_kernel.size());
  cutlass::device_memory::copy_to_device(bias.get(), host_bias.data(), host_bias.size());
  cutlass::device_memory::copy_to_device(gamma.get(), host_gamma.data(), host_gamma.size());
  cutlass::device_memory::copy_to_device(beta.get(), host_beta.data(), host_beta.size());
  cudaMemset(workspace.get(), 0, workspace_elements * sizeof(Element));
  cudaMemset(output.get(), 0, patch_embed_output_elements(problem) * sizeof(Element));

  Op::Tensors tensors;
  tensors.input = input.get();
  tensors.kernel = kernel.get();
  tensors.bias = bias.get();
  tensors.gamma = gamma.get();
  tensors.beta = beta.get();
  tensors.conv_output = workspace.get();
  tensors.output = output.get();

  cutlass::Status status = Op::run(problem, tensors, nullptr);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "PatchEmbed run failed: "
              << cutlassGetStatusString(status) << "\n";
    return -1;
  }
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "PatchEmbed sync failed: " << cudaGetErrorString(err) << "\n";
    return -1;
  }

  std::vector<Element> host_output(patch_embed_output_elements(problem));
  std::vector<Element> host_reference;
  cutlass::device_memory::copy_to_host(
      host_output.data(), output.get(), host_output.size());
  patch_embed_reference(
      problem,
      host_input,
      host_kernel,
      host_bias,
      host_gamma,
      host_beta,
      host_reference);

  if (!compare(host_output, host_reference, options.abs_tolerance, options.rel_tolerance)) {
    std::cout << "\nFailed\n";
    return -1;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < options.iterations; ++i) {
    status = Op::run(problem, tensors, nullptr);
    if (status != cutlass::Status::kSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return -1;
    }
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double runtime_ms = double(elapsed_ms) / double(options.iterations);
  int out = patch_embed_output_size(problem);
  double flops = 2.0 * problem.batch_size * out * out * problem.embed_dim *
      problem.in_channels * problem.patch_size * problem.patch_size;
  double gflops = flops / 1.0e6 / runtime_ms;

  std::cout << "\nSwin PatchEmbed path:\n"
            << "====================================================\n"
            << "    {B, image, patch, Cin, Cin_pad, embed} = {"
            << problem.batch_size << ", " << problem.image_size << ", "
            << problem.patch_size << ", " << problem.in_channels << ", "
            << problem.input_channels_padded << ", " << problem.embed_dim << "}\n"
            << "    Path: NHWC input -> channel pad -> CUTLASS Conv2d -> BiasLayerNorm -> NHWC output\n"
            << "    Runtime: " << runtime_ms << " ms\n"
            << "    GFLOPs : " << gflops << "\n\nPassed\n";

  return 0;
}

} // namespace
} // namespace swin
} // namespace tiny_cutlass

int main(int argc, char const** args) {
  using namespace tiny_cutlass::swin;

  cudaDeviceProp props;
  cudaError_t err = cudaGetDeviceProperties(&props, 0);
  if (err != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties: " << cudaGetErrorString(err) << "\n";
    return -1;
  }
  std::cout << "Device: " << props.name << " (SM" << props.major << props.minor << ")\n";

  Options options;
  options.parse(argc, args);
  if (options.error) {
    std::cerr << "Invalid options.\n";
    return -1;
  }
  return run(options);
}
