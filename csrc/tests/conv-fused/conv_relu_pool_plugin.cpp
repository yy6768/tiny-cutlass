#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "NvInfer.h"
#include "binding/tensorrt/conv_relu_pool.h"
#include "cutlass/half.h"
#include "ops/conv_relu_pool.h"

namespace {

using Element = cutlass::half_t;
namespace conv = tiny_cutlass::conv_fused;
namespace trt_plugin = tiny_cutlass::conv_fused::binding::tensorrt::conv_relu_pool;

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, char const* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << msg << "\n";
    }
  }
};

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

size_t tensor_count(int batch, int height, int width, int channels) {
  return size_t(batch) * height * width * channels;
}

std::vector<Element> random_tensor(size_t count, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
  std::vector<Element> tensor(count);
  for (auto& value : tensor) {
    value = Element(dist(rng));
  }
  return tensor;
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

bool build_and_run_trt(conv::ConvReluPoolProblem const& problem,
                       DeviceBuffer<Element> const& input,
                       DeviceBuffer<Element> const& weight0,
                       DeviceBuffer<Element> const& bias0,
                       DeviceBuffer<Element> const& weight1,
                       DeviceBuffer<Element> const& bias1,
                       DeviceBuffer<Element> const& output) {
  Logger logger;
  trt_plugin::Creator creator;
  if (!getPluginRegistry()->registerCreator(creator, "tiny_cutlass")) {
    std::cerr << "TensorRT registerCreator failed\n";
    return false;
  }

  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    std::cerr << "createInferBuilder failed\n";
    return false;
  }

  auto flags = uint32_t(1)
      << uint32_t(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
  std::unique_ptr<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(flags));
  std::unique_ptr<nvinfer1::IBuilderConfig> config(
      builder->createBuilderConfig());
  if (!network || !config) {
    std::cerr << "TensorRT network/config creation failed\n";
    return false;
  }
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, size_t(1) << 30);

  nvinfer1::ITensor* tensors[] = {
      network->addInput(
          "input", nvinfer1::DataType::kHALF,
          nvinfer1::Dims4{problem.batch, problem.height, problem.width, problem.channels}),
      network->addInput(
          "weight0", nvinfer1::DataType::kHALF,
          nvinfer1::Dims4{problem.hidden_channels, 3, 3, problem.channels}),
      network->addInput(
          "bias0", nvinfer1::DataType::kHALF,
          nvinfer1::Dims{1, {problem.hidden_channels}}),
      network->addInput(
          "weight1", nvinfer1::DataType::kHALF,
          nvinfer1::Dims4{problem.output_channels, 1, 1, problem.hidden_channels}),
      network->addInput(
          "bias1", nvinfer1::DataType::kHALF,
          nvinfer1::Dims{1, {problem.output_channels}}),
  };
  for (auto* tensor : tensors) {
    if (!tensor) {
      std::cerr << "TensorRT addInput failed\n";
      return false;
    }
  }

  trt_plugin::Plugin plugin;
  nvinfer1::IPluginV3Layer* layer =
      network->addPluginV3(tensors, 5, nullptr, 0, plugin);
  if (!layer || !layer->getOutput(0)) {
    std::cerr << "TensorRT addPluginV3 failed\n";
    return false;
  }
  layer->getOutput(0)->setName("output");
  network->markOutput(*layer->getOutput(0));

  std::unique_ptr<nvinfer1::IHostMemory> plan(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    std::cerr << "TensorRT buildSerializedNetwork failed\n";
    return false;
  }

  std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
  if (!runtime) {
    std::cerr << "createInferRuntime failed\n";
    return false;
  }
  std::unique_ptr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(plan->data(), plan->size()));
  if (!engine) {
    std::cerr << "deserializeCudaEngine failed\n";
    return false;
  }
  std::unique_ptr<nvinfer1::IExecutionContext> context(
      engine->createExecutionContext());
  if (!context) {
    std::cerr << "createExecutionContext failed\n";
    return false;
  }

  if (!context->setInputTensorAddress("input", input.get()) ||
      !context->setInputTensorAddress("weight0", weight0.get()) ||
      !context->setInputTensorAddress("bias0", bias0.get()) ||
      !context->setInputTensorAddress("weight1", weight1.get()) ||
      !context->setInputTensorAddress("bias1", bias1.get()) ||
      !context->setOutputTensorAddress("output", output.get())) {
    std::cerr << "TensorRT setTensorAddress failed\n";
    return false;
  }

  cudaStream_t stream = nullptr;
  if (!cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate")) {
    return false;
  }
  bool const enqueued = context->enqueueV3(stream);
  bool const synced = cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  cudaStreamDestroy(stream);
  if (!enqueued) {
    std::cerr << "TensorRT enqueueV3 failed\n";
  }
  return enqueued && synced;
}

bool run_direct_core(conv::ConvReluPoolProblem const& problem,
                     DeviceBuffer<Element> const& input,
                     DeviceBuffer<Element> const& weight0,
                     DeviceBuffer<Element> const& bias0,
                     DeviceBuffer<Element> const& weight1,
                     DeviceBuffer<Element> const& bias1,
                     DeviceBuffer<Element> const& output) {
  size_t workspace_bytes = conv::conv_relu_pool_workspace_size<Element>(problem);
  DeviceBuffer<uint8_t> workspace(workspace_bytes);
  if (!workspace.get()) {
    return false;
  }

  conv::ConvReluPoolArguments<Element> args;
  args.problem = problem;
  args.input = input.get();
  args.weight0 = weight0.get();
  args.bias0 = bias0.get();
  args.weight1 = weight1.get();
  args.bias1 = bias1.get();
  args.output = output.get();
  args.workspace = workspace.get();
  args.workspace_bytes = workspace_bytes;

  cutlass::Status status = conv::conv_relu_pool(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "direct core returned " << cutlassGetStatusString(status) << "\n";
    return false;
  }
  return cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

}  // namespace

int main() {
  conv::ConvReluPoolProblem problem{1, 8, 8, 8, 8, 8};

  auto input = random_tensor(tensor_count(1, 8, 8, 8), 21);
  auto weight0 = random_tensor(size_t(8) * 3 * 3 * 8, 22);
  auto bias0 = random_tensor(8, 23);
  auto weight1 = random_tensor(size_t(8) * 1 * 1 * 8, 24);
  auto bias1 = random_tensor(8, 25);
  size_t const output_count = tensor_count(1, 2, 2, 8);

  DeviceBuffer<Element> d_input(input.size());
  DeviceBuffer<Element> d_weight0(weight0.size());
  DeviceBuffer<Element> d_bias0(bias0.size());
  DeviceBuffer<Element> d_weight1(weight1.size());
  DeviceBuffer<Element> d_bias1(bias1.size());
  DeviceBuffer<Element> d_trt(output_count);
  DeviceBuffer<Element> d_direct(output_count);

  if (!d_input.get() || !d_weight0.get() || !d_bias0.get() || !d_weight1.get() ||
      !d_bias1.get() || !d_trt.get() || !d_direct.get()) {
    return 1;
  }

  if (!copy_to_device(d_input, input) ||
      !copy_to_device(d_weight0, weight0) ||
      !copy_to_device(d_bias0, bias0) ||
      !copy_to_device(d_weight1, weight1) ||
      !copy_to_device(d_bias1, bias1)) {
    return 1;
  }

  if (!run_direct_core(problem, d_input, d_weight0, d_bias0, d_weight1, d_bias1, d_direct)) {
    return 1;
  }
  if (!build_and_run_trt(problem, d_input, d_weight0, d_bias0, d_weight1, d_bias1, d_trt)) {
    return 1;
  }

  std::vector<Element> trt_output(output_count);
  std::vector<Element> direct_output(output_count);
  if (!copy_to_host(trt_output, d_trt) || !copy_to_host(direct_output, d_direct)) {
    return 1;
  }

  float max_abs = 0.0f;
  for (size_t i = 0; i < output_count; ++i) {
    max_abs = std::max(
        max_abs,
        std::fabs(float(trt_output[i]) - float(direct_output[i])));
  }

  if (max_abs > 0.0f) {
    std::cerr << "plugin output mismatch max_abs=" << max_abs << "\n";
    return 1;
  }

  std::cout << "pass plugin_small max_abs=" << max_abs << "\n";
  return 0;
}
