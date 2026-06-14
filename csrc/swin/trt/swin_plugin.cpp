#include "swin_plugin.h"

#include <algorithm>
#include <cstring>

#include "cutlass/arch/mma.h"
#include "cutlass/numeric_types.h"
#include "swin.h"

namespace tiny_cutlass {
namespace swin {
namespace trt {
namespace {

constexpr char kPluginName[] = "tiny_cutlass_swin";
constexpr char kPluginVersion[] = "1";
constexpr int32_t kInputCount = 6;
nvinfer1::PluginField const kPluginFields[] = {
    {"batch_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
    {"image_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
    {"window_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
    {"shift_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
    {"head_number", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
    {"head_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
    {"scale", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
};
constexpr int32_t kPluginFieldCount = 7;

template <typename T>
T read_field(nvinfer1::PluginField const& field, T fallback) {
  if (field.data == nullptr) {
    return fallback;
  }
  T value{};
  std::memcpy(&value, field.data, sizeof(T));
  return value;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

std::size_t workspace_bytes(
    SwinPluginConfig const& config) {
  SwinProblem problem;
  problem.batch_size = config.batch_size;
  problem.image_size = config.image_size;
  problem.window_size = config.window_size;
  problem.shift_size = config.shift_size;
  problem.head_number = config.head_number;
  problem.head_size = config.head_size;
  problem.scale = config.scale;

  std::size_t bytes = 0;
  auto add = [&](std::int64_t elements) {
    bytes = align_up(bytes, 256);
    bytes += static_cast<std::size_t>(elements) * sizeof(cutlass::half_t);
  };
  add(swin_window_elements(problem));
  add(swin_qkv_elements(problem));
  add(swin_window_elements(problem));
  add(swin_window_elements(problem));
  add(swin_window_elements(problem));
  add(swin_window_elements(problem));
  add(swin_window_elements(problem));
  return align_up(bytes, 256);
}

cutlass::half_t* take_workspace(char*& ptr, std::int64_t elements) {
  auto address = reinterpret_cast<std::uintptr_t>(ptr);
  address = align_up(address, 256);
  ptr = reinterpret_cast<char*>(address + static_cast<std::size_t>(elements) * sizeof(cutlass::half_t));
  return reinterpret_cast<cutlass::half_t*>(address);
}

SwinProblem make_problem(SwinPluginConfig const& config) {
  SwinProblem problem;
  problem.batch_size = config.batch_size;
  problem.image_size = config.image_size;
  problem.window_size = config.window_size;
  problem.shift_size = config.shift_size;
  problem.head_number = config.head_number;
  problem.head_size = config.head_size;
  problem.scale = config.scale;
  return problem;
}

bool is_nhwc_activation(
    nvinfer1::Dims const& dims,
    SwinPluginConfig const& config) {
  int channels = config.head_number * config.head_size;
  return dims.nbDims == 4 &&
      dims.d[0] == config.batch_size &&
      dims.d[1] == config.image_size &&
      dims.d[2] == config.image_size &&
      dims.d[3] == channels;
}

} // namespace

SwinPlugin::SwinPlugin(char const* name, SwinPluginConfig const& config)
    : name_(name ? name : kPluginName), config_(config) {}

SwinPlugin::SwinPlugin(char const* name, void const* serial_data, std::size_t serial_length)
    : name_(name ? name : kPluginName) {
  if (serial_data != nullptr && serial_length == sizeof(SwinPluginConfig)) {
    std::memcpy(&config_, serial_data, sizeof(SwinPluginConfig));
  }
}

nvinfer1::IPluginV2DynamicExt* SwinPlugin::clone() const noexcept {
  auto* plugin = new SwinPlugin(name_.c_str(), config_);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

nvinfer1::DimsExprs SwinPlugin::getOutputDimensions(
    int32_t output_index,
    nvinfer1::DimsExprs const* inputs,
    int32_t nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) noexcept {
  (void)expr_builder;
  if (output_index != 0 || nb_inputs < 1) {
    return nvinfer1::DimsExprs{};
  }
  return inputs[0];
}

bool SwinPlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::PluginTensorDesc const* in_out,
    int32_t nb_inputs,
    int32_t nb_outputs) noexcept {
  if (nb_inputs != kInputCount || nb_outputs != 1 || pos < 0 || pos >= nb_inputs + nb_outputs) {
    return false;
  }
  if (in_out[pos].format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }
  if (pos == 0) {
    return in_out[pos].type == nvinfer1::DataType::kHALF;
  }
  if (pos == nb_inputs) {
    return in_out[pos].type == nvinfer1::DataType::kHALF;
  }
  return in_out[pos].type == nvinfer1::DataType::kHALF;
}

void SwinPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in,
    int32_t nb_inputs,
    nvinfer1::DynamicPluginTensorDesc const* out,
    int32_t nb_outputs) noexcept {
  (void)in;
  (void)nb_inputs;
  (void)out;
  (void)nb_outputs;
}

std::size_t SwinPlugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const* inputs,
    int32_t nb_inputs,
    nvinfer1::PluginTensorDesc const* outputs,
    int32_t nb_outputs) const noexcept {
  (void)inputs;
  (void)nb_inputs;
  (void)outputs;
  (void)nb_outputs;
  return workspace_bytes(config_);
}

int32_t SwinPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* input_desc,
    nvinfer1::PluginTensorDesc const* output_desc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  (void)input_desc;
  (void)output_desc;
  if (inputs == nullptr || outputs == nullptr || workspace == nullptr) {
    return 1;
  }

  SwinProblem problem = make_problem(config_);
  char* workspace_ptr = static_cast<char*>(workspace);

  if (input_desc[0].type != nvinfer1::DataType::kHALF ||
      output_desc[0].type != nvinfer1::DataType::kHALF ||
      !is_nhwc_activation(input_desc[0].dims, config_) ||
      !is_nhwc_activation(output_desc[0].dims, config_)) {
    return 1;
  }

  SwinTensors<cutlass::half_t> tensors;
  tensors.input = static_cast<cutlass::half_t const*>(inputs[0]);
  tensors.qkv_weight = static_cast<cutlass::half_t const*>(inputs[1]);
  tensors.qkv_bias = static_cast<cutlass::half_t const*>(inputs[2]);
  tensors.output_weight = static_cast<cutlass::half_t const*>(inputs[3]);
  tensors.output_bias = static_cast<cutlass::half_t const*>(inputs[4]);
  tensors.attention_bias = static_cast<cutlass::half_t const*>(inputs[5]);
  tensors.windows = take_workspace(workspace_ptr, swin_window_elements(problem));
  tensors.qkv = take_workspace(workspace_ptr, swin_qkv_elements(problem));
  tensors.query = take_workspace(workspace_ptr, swin_window_elements(problem));
  tensors.key = take_workspace(workspace_ptr, swin_window_elements(problem));
  tensors.value = take_workspace(workspace_ptr, swin_window_elements(problem));
  tensors.attention_output = take_workspace(workspace_ptr, swin_window_elements(problem));
  tensors.projected = take_workspace(workspace_ptr, swin_window_elements(problem));
  tensors.output = static_cast<cutlass::half_t*>(outputs[0]);
  tensors.patch_merged = nullptr;

  using Kernel = typename kernel::DefaultSwin<
      cutlass::arch::Sm80,
      cutlass::half_t>::Kernel;
  cudaError_t err = device::Swin<Kernel>::run(problem, tensors, stream);
  if (err != cudaSuccess) {
    return 1;
  }
  return 0;
}

nvinfer1::DataType SwinPlugin::getOutputDataType(
    int32_t index,
    nvinfer1::DataType const* input_types,
    int32_t nb_inputs) const noexcept {
  (void)index;
  (void)input_types;
  (void)nb_inputs;
  return nvinfer1::DataType::kHALF;
}

char const* SwinPlugin::getPluginType() const noexcept {
  return kPluginName;
}

char const* SwinPlugin::getPluginVersion() const noexcept {
  return kPluginVersion;
}

int32_t SwinPlugin::getNbOutputs() const noexcept {
  return 1;
}

void SwinPlugin::configureWithFormat(
    nvinfer1::Dims const* input_dims,
    int32_t nb_inputs,
    nvinfer1::Dims const* output_dims,
    int32_t nb_outputs,
    nvinfer1::DataType type,
    nvinfer1::PluginFormat format,
    int32_t max_batch_size) noexcept {
  (void)input_dims;
  (void)nb_inputs;
  (void)output_dims;
  (void)nb_outputs;
  (void)type;
  (void)format;
  (void)max_batch_size;
}

int32_t SwinPlugin::initialize() noexcept {
  return 0;
}

void SwinPlugin::terminate() noexcept {}

std::size_t SwinPlugin::getSerializationSize() const noexcept {
  return sizeof(SwinPluginConfig);
}

void SwinPlugin::serialize(void* buffer) const noexcept {
  std::memcpy(buffer, &config_, sizeof(SwinPluginConfig));
}

void SwinPlugin::destroy() noexcept {
  delete this;
}

void SwinPlugin::setPluginNamespace(char const* plugin_namespace) noexcept {
  namespace_ = plugin_namespace ? plugin_namespace : "";
}

char const* SwinPlugin::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

SwinPluginCreator::SwinPluginCreator() {
  fields_.nbFields = kPluginFieldCount;
  fields_.fields = kPluginFields;
}

char const* SwinPluginCreator::getPluginName() const noexcept {
  return kPluginName;
}

char const* SwinPluginCreator::getPluginVersion() const noexcept {
  return kPluginVersion;
}

nvinfer1::PluginFieldCollection const* SwinPluginCreator::getFieldNames() noexcept {
  return &fields_;
}

nvinfer1::IPluginV2* SwinPluginCreator::createPlugin(
    char const* name,
    nvinfer1::PluginFieldCollection const* fc) noexcept {
  SwinPluginConfig config;
  if (fc != nullptr) {
    for (int32_t i = 0; i < fc->nbFields; ++i) {
      nvinfer1::PluginField const& field = fc->fields[i];
      if (std::strcmp(field.name, "batch_size") == 0) {
        config.batch_size = read_field<int32_t>(field, config.batch_size);
      } else if (std::strcmp(field.name, "image_size") == 0) {
        config.image_size = read_field<int32_t>(field, config.image_size);
      } else if (std::strcmp(field.name, "window_size") == 0) {
        config.window_size = read_field<int32_t>(field, config.window_size);
      } else if (std::strcmp(field.name, "shift_size") == 0) {
        config.shift_size = read_field<int32_t>(field, config.shift_size);
      } else if (std::strcmp(field.name, "head_number") == 0) {
        config.head_number = read_field<int32_t>(field, config.head_number);
      } else if (std::strcmp(field.name, "head_size") == 0) {
        config.head_size = read_field<int32_t>(field, config.head_size);
      } else if (std::strcmp(field.name, "scale") == 0) {
        config.scale = read_field<float>(field, config.scale);
      }
    }
  }
  return new SwinPlugin(name, config);
}

nvinfer1::IPluginV2* SwinPluginCreator::deserializePlugin(
    char const* name,
    void const* serial_data,
    std::size_t serial_length) noexcept {
  return new SwinPlugin(name, serial_data, serial_length);
}

void SwinPluginCreator::setPluginNamespace(char const* plugin_namespace) noexcept {
  namespace_ = plugin_namespace ? plugin_namespace : "";
}

char const* SwinPluginCreator::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

} // namespace trt
} // namespace swin
} // namespace tiny_cutlass

namespace {
class SwinPluginRegistrarCreator final : public tiny_cutlass::swin::trt::SwinPluginCreator {};
}

REGISTER_TENSORRT_PLUGIN(SwinPluginRegistrarCreator);
