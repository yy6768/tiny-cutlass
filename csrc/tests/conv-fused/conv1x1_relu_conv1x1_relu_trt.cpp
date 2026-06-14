#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "binding/tensorrt/conv1x1_relu_conv1x1_relu.h"
#include "cutlass/half.h"

namespace {

using OutputElement = cutlass::half_t;
namespace trt_plugin =
    tiny_cutlass::conv_fused::binding::tensorrt::conv1x1_relu_conv1x1_relu;

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
    cudaError_t error =
        cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * count);
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

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

struct Options {
  std::filesystem::path artifact_dir = "build/conv-fused/fp8-reference";
  std::filesystem::path output_bin;
  std::filesystem::path metrics_json;
  int batch = 1;
  int height = 32;
  int width = 32;
  int channels = 16;
  int hidden_channels = 32;
  int output_channels = 16;
  int warmup = 20;
  int iterations = 100;
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

std::filesystem::path artifact_path(
    Options const& options,
    char const* filename) {
  return options.artifact_dir / filename;
}

template <typename T>
bool read_binary(
    std::filesystem::path const& path,
    std::vector<T>& values,
    size_t count) {
  values.resize(count);
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    std::cerr << "failed to open " << path << "\n";
    return false;
  }
  input.read(
      reinterpret_cast<char*>(values.data()),
      static_cast<std::streamsize>(sizeof(T) * count));
  if (input.gcount() != static_cast<std::streamsize>(sizeof(T) * count)) {
    std::cerr << "short read from " << path << "\n";
    return false;
  }
  return true;
}

template <typename T>
bool write_binary(std::filesystem::path const& path, std::vector<T> const& values) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    std::cerr << "failed to open " << path << " for writing\n";
    return false;
  }
  output.write(
      reinterpret_cast<char const*>(values.data()),
      static_cast<std::streamsize>(sizeof(T) * values.size()));
  return bool(output);
}

bool read_scalar(
    std::filesystem::path const& path,
    float& value) {
  std::vector<float> values;
  if (!read_binary(path, values, 1)) {
    return false;
  }
  value = values[0];
  return true;
}

template <typename T>
bool copy_to_device(DeviceBuffer<T> const& dst, std::vector<T> const& src) {
  return cuda_ok(
      cudaMemcpy(
          dst.get(),
          src.data(),
          sizeof(T) * src.size(),
          cudaMemcpyHostToDevice),
      "cudaMemcpy host-to-device");
}

template <typename T>
bool copy_to_host(std::vector<T>& dst, DeviceBuffer<T> const& src) {
  return cuda_ok(
      cudaMemcpy(
          dst.data(),
          src.get(),
          sizeof(T) * dst.size(),
          cudaMemcpyDeviceToHost),
      "cudaMemcpy device-to-host");
}

bool parse_int(
    int argc,
    char** argv,
    int& index,
    char const* flag,
    int& value) {
  if (std::string(argv[index]) != flag) {
    return false;
  }
  if (index + 1 >= argc) {
    std::cerr << "missing value for " << flag << "\n";
    return false;
  }
  value = std::stoi(argv[++index]);
  return true;
}

bool parse_path(
    int argc,
    char** argv,
    int& index,
    char const* flag,
    std::filesystem::path& value) {
  if (std::string(argv[index]) != flag) {
    return false;
  }
  if (index + 1 >= argc) {
    std::cerr << "missing value for " << flag << "\n";
    return false;
  }
  value = argv[++index];
  return true;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string const arg = argv[i];
    if (parse_path(argc, argv, i, "--artifact-dir", options.artifact_dir) ||
        parse_path(argc, argv, i, "--output-bin", options.output_bin) ||
        parse_path(argc, argv, i, "--metrics", options.metrics_json) ||
        parse_int(argc, argv, i, "--batch", options.batch) ||
        parse_int(argc, argv, i, "--height", options.height) ||
        parse_int(argc, argv, i, "--width", options.width) ||
        parse_int(argc, argv, i, "--channels", options.channels) ||
        parse_int(argc, argv, i, "--hidden-channels", options.hidden_channels) ||
        parse_int(argc, argv, i, "--output-channels", options.output_channels) ||
        parse_int(argc, argv, i, "--warmup", options.warmup) ||
        parse_int(argc, argv, i, "--iterations", options.iterations)) {
      continue;
    }
    std::cerr << "unknown argument: " << arg << "\n";
    std::exit(2);
  }

  if (options.output_bin.empty()) {
    options.output_bin = options.artifact_dir / "ours_output.bin";
  }
  if (options.metrics_json.empty()) {
    options.metrics_json = options.artifact_dir / "ours_times.json";
  }
  return options;
}

nvinfer1::Weights fp8_weights(std::vector<uint8_t> const& values) {
  return nvinfer1::Weights{
      nvinfer1::DataType::kFP8,
      values.data(),
      static_cast<int64_t>(values.size())};
}

nvinfer1::Weights float_weights(std::vector<float> const& values) {
  return nvinfer1::Weights{
      nvinfer1::DataType::kFLOAT,
      values.data(),
      static_cast<int64_t>(values.size())};
}

bool build_engine(
    Logger& logger,
    Options const& options,
    std::vector<uint8_t> const& weight0,
    std::vector<float> const& stage0_scale,
    std::vector<float> const& bias0,
    std::vector<uint8_t> const& weight1,
    std::vector<uint8_t> const& bias1,
    float output_scale_inv,
    float output_alpha,
    std::unique_ptr<nvinfer1::ICudaEngine>& engine) {
  trt_plugin::Creator creator;
  getPluginRegistry()->registerCreator(creator, "tiny_cutlass");

  std::unique_ptr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(logger));
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
  config->setMemoryPoolLimit(
      nvinfer1::MemoryPoolType::kWORKSPACE,
      size_t(1) << 30);

  nvinfer1::ITensor* input = network->addInput(
      "input_e4m3",
      nvinfer1::DataType::kFP8,
      nvinfer1::Dims4{
          options.batch,
          options.height,
          options.width,
          options.channels});
  nvinfer1::IConstantLayer* weight0_layer = network->addConstant(
      nvinfer1::Dims4{
          options.hidden_channels,
          1,
          1,
          options.channels},
      fp8_weights(weight0));
  nvinfer1::IConstantLayer* stage0_scale_layer = network->addConstant(
      nvinfer1::Dims{1, {options.hidden_channels}},
      float_weights(stage0_scale));
  nvinfer1::IConstantLayer* bias0_layer = network->addConstant(
      nvinfer1::Dims{1, {options.hidden_channels}},
      float_weights(bias0));
  nvinfer1::IConstantLayer* weight1_layer = network->addConstant(
      nvinfer1::Dims4{
          options.output_channels,
          1,
          1,
          options.hidden_channels},
      fp8_weights(weight1));
  nvinfer1::IConstantLayer* bias1_layer = network->addConstant(
      nvinfer1::Dims{1, {options.output_channels}},
      fp8_weights(bias1));
  std::vector<float> output_scale_values{output_scale_inv};
  nvinfer1::IConstantLayer* output_scale_layer = network->addConstant(
      nvinfer1::Dims{1, {1}},
      float_weights(output_scale_values));
  if (!input || !weight0_layer || !stage0_scale_layer || !bias0_layer ||
      !weight1_layer || !bias1_layer || !output_scale_layer) {
    std::cerr << "TensorRT input/constant creation failed\n";
    return false;
  }

  nvinfer1::ITensor* plugin_inputs[] = {
      input,
      weight0_layer->getOutput(0),
      stage0_scale_layer->getOutput(0),
      bias0_layer->getOutput(0),
      weight1_layer->getOutput(0),
      bias1_layer->getOutput(0)};
  trt_plugin::Plugin plugin(output_alpha);
  nvinfer1::IPluginV3Layer* plugin_layer =
      network->addPluginV3(plugin_inputs, 6, nullptr, 0, plugin);
  if (!plugin_layer || !plugin_layer->getOutput(0)) {
    std::cerr << "TensorRT addPluginV3 failed\n";
    return false;
  }

  nvinfer1::IDequantizeLayer* dequantize_layer = network->addDequantize(
      *plugin_layer->getOutput(0),
      *output_scale_layer->getOutput(0),
      nvinfer1::DataType::kHALF);
  if (!dequantize_layer || !dequantize_layer->getOutput(0)) {
    std::cerr << "TensorRT addDequantize failed\n";
    return false;
  }
  dequantize_layer->getOutput(0)->setName("output_nhwc");
  network->markOutput(*dequantize_layer->getOutput(0));

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
  engine.reset(runtime->deserializeCudaEngine(plan->data(), plan->size()));
  if (!engine) {
    std::cerr << "deserializeCudaEngine failed\n";
    return false;
  }
  return true;
}

bool run_plugin(
    Options const& options,
    nvinfer1::ICudaEngine& engine,
    DeviceBuffer<uint8_t> const& input,
    DeviceBuffer<OutputElement> const& output,
    float& average_ms) {
  std::unique_ptr<nvinfer1::IExecutionContext> context(
      engine.createExecutionContext());
  if (!context) {
    std::cerr << "createExecutionContext failed\n";
    return false;
  }
  if (!context->setInputTensorAddress("input_e4m3", input.get()) ||
      !context->setOutputTensorAddress("output_nhwc", output.get())) {
    std::cerr << "TensorRT setTensorAddress failed\n";
    return false;
  }

  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (!cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate") ||
      !cuda_ok(cudaEventCreate(&start), "cudaEventCreate") ||
      !cuda_ok(cudaEventCreate(&stop), "cudaEventCreate")) {
    return false;
  }

  bool ok = true;
  for (int i = 0; i < options.warmup; ++i) {
    ok = context->enqueueV3(stream) && ok;
  }
  ok = cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize") && ok;

  if (ok) {
    ok = cuda_ok(cudaEventRecord(start, stream), "cudaEventRecord start");
  }
  for (int i = 0; ok && i < options.iterations; ++i) {
    ok = context->enqueueV3(stream);
  }
  if (ok) {
    ok = cuda_ok(cudaEventRecord(stop, stream), "cudaEventRecord stop") &&
        cuda_ok(cudaEventSynchronize(stop), "cudaEventSynchronize");
  }

  float elapsed_ms = 0.0f;
  if (ok) {
    ok = cuda_ok(cudaEventElapsedTime(&elapsed_ms, start, stop),
                 "cudaEventElapsedTime");
  }
  average_ms = options.iterations > 0 ? elapsed_ms / options.iterations : 0.0f;

  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaStreamDestroy(stream);
  return ok;
}

std::string json_path(std::filesystem::path const& path) {
  std::string text = path.generic_string();
  std::string escaped;
  escaped.reserve(text.size());
  for (char ch : text) {
    if (ch == '\\' || ch == '"') {
      escaped.push_back('\\');
    }
    escaped.push_back(ch);
  }
  return escaped;
}

bool write_metrics(
    Options const& options,
    float average_ms,
    std::filesystem::path const& output_bin) {
  std::ofstream metrics(options.metrics_json);
  if (!metrics) {
    std::cerr << "failed to open " << options.metrics_json << " for writing\n";
    return false;
  }
  metrics << "{\n";
  metrics << "  \"name\": \"ours_plugin\",\n";
  metrics << "  \"operator\": \"conv1x1_relu_conv1x1_relu\",\n";
  metrics << "  \"quantization\": \"e4m3\",\n";
  metrics << "  \"input_shape\": ["
          << options.batch << ", "
          << options.height << ", "
          << options.width << ", "
          << options.channels << "],\n";
  metrics << "  \"output_shape\": ["
          << options.batch << ", "
          << options.height << ", "
          << options.width << ", "
          << options.output_channels << "],\n";
  metrics << "  \"warmup\": " << options.warmup << ",\n";
  metrics << "  \"iterations\": " << options.iterations << ",\n";
  metrics << "  \"average_ms\": " << std::setprecision(8) << average_ms << ",\n";
  metrics << "  \"output_bin\": \"" << json_path(output_bin) << "\"\n";
  metrics << "}\n";
  return bool(metrics);
}

bool run(Options const& options) {
  std::vector<uint8_t> input;
  std::vector<uint8_t> weight0;
  std::vector<float> stage0_scale;
  std::vector<float> bias0;
  std::vector<uint8_t> weight1;
  std::vector<uint8_t> bias1;
  float output_scale_inv = 0.0f;
  float output_alpha = 0.0f;

  if (!read_binary(
          artifact_path(options, "input_e4m3.bin"),
          input,
          tensor_count(
              options.batch,
              options.height,
              options.width,
              options.channels)) ||
      !read_binary(
          artifact_path(options, "conv0_weight_e4m3_krsc.bin"),
          weight0,
          size_t(options.hidden_channels) * options.channels) ||
      !read_binary(
          artifact_path(options, "stage0_scale.bin"),
          stage0_scale,
          options.hidden_channels) ||
      !read_binary(
          artifact_path(options, "bias0.bin"),
          bias0,
          options.hidden_channels) ||
      !read_binary(
          artifact_path(options, "conv1_weight_e4m3_krsc.bin"),
          weight1,
          size_t(options.output_channels) * options.hidden_channels) ||
      !read_binary(
          artifact_path(options, "bias1_e4m3.bin"),
          bias1,
          options.output_channels) ||
      !read_scalar(
          artifact_path(options, "output_scale_inv.bin"),
          output_scale_inv) ||
      !read_scalar(
          artifact_path(options, "output_alpha.bin"),
          output_alpha)) {
    return false;
  }

  size_t const output_count = tensor_count(
      options.batch,
      options.height,
      options.width,
      options.output_channels);
  DeviceBuffer<uint8_t> d_input(input.size());
  DeviceBuffer<OutputElement> d_output(output_count);
  if (!d_input.get() || !d_output.get()) {
    return false;
  }
  if (!copy_to_device(d_input, input)) {
    return false;
  }

  Logger logger;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  if (!build_engine(
          logger,
          options,
          weight0,
          stage0_scale,
          bias0,
          weight1,
          bias1,
          output_scale_inv,
          output_alpha,
          engine)) {
    return false;
  }

  float average_ms = 0.0f;
  if (!run_plugin(options, *engine, d_input, d_output, average_ms)) {
    return false;
  }

  std::vector<OutputElement> output(output_count);
  if (!copy_to_host(output, d_output)) {
    return false;
  }
  if (!write_binary(options.output_bin, output) ||
      !write_metrics(options, average_ms, options.output_bin)) {
    return false;
  }

  std::cout << "pass ours_plugin output=" << options.output_bin
            << " average_ms=" << average_ms
            << " iterations=" << options.iterations << "\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options = parse_options(argc, argv);
  return run(options) ? 0 : 1;
}
