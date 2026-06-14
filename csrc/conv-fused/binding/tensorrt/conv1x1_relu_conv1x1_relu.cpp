#include "binding/tensorrt/conv1x1_relu_conv1x1_relu.h"

#include <cstdint>
#include <cstring>

#include "cutlass/numeric_types.h"
#include "fp8/conv1x1_relu_conv1x1_relu_fp8/ops/conv1x1_relu_conv1x1_relu_fp8.h"

namespace tiny_cutlass::conv_fused::binding::tensorrt::conv1x1_relu_conv1x1_relu {
namespace {

constexpr char const* kPluginName = "conv1x1_relu_conv1x1_relu";
constexpr char const* kPluginVersion = "1";
constexpr char const* kPluginNamespace = "tiny_cutlass";
constexpr char const* kOutputAlphaField = "output_alpha";
constexpr int32_t kNbInputs = 6;
constexpr int32_t kNbOutputs = 1;

using CoreElement = cutlass::float_e4m3_t;
namespace core = tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu;

bool is_known_dim(int32_t value) {
  return value >= 0;
}

bool dim_is(nvinfer1::Dims const& dims, int32_t index, int32_t expected) {
  return !is_known_dim(dims.d[index]) || dims.d[index] == expected;
}

int32_t resolved_dim(
    nvinfer1::DynamicPluginTensorDesc const& desc,
    int32_t index) {
  int32_t const value = desc.desc.dims.d[index];
  return is_known_dim(value) ? value : desc.max.d[index];
}

bool type_matches(int32_t pos, nvinfer1::DataType type) {
  if (pos == 0 || pos == 1 || pos == 4 || pos == 5 || pos == kNbInputs) {
    return type == nvinfer1::DataType::kFP8;
  }
  if (pos == 2 || pos == 3) {
    return type == nvinfer1::DataType::kFLOAT;
  }
  return false;
}

core::Problem make_problem(nvinfer1::DynamicPluginTensorDesc const* inputs) {
  return core::Problem{
      resolved_dim(inputs[0], 0),
      resolved_dim(inputs[0], 1),
      resolved_dim(inputs[0], 2),
      resolved_dim(inputs[0], 3),
      resolved_dim(inputs[1], 0),
      resolved_dim(inputs[4], 0),
  };
}

size_t stage0_workspace_bytes(core::Problem const& problem) {
  return static_cast<size_t>(problem.batch) *
         static_cast<size_t>(problem.height) *
         static_cast<size_t>(problem.width) *
         static_cast<size_t>(problem.hidden_channels) *
         sizeof(CoreElement);
}

bool validate_build_desc(
    nvinfer1::DynamicPluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out,
    int32_t nbOutputs) {
  if (nbInputs != kNbInputs || nbOutputs != kNbOutputs) {
    return false;
  }
  for (int32_t i = 0; i < nbInputs + nbOutputs; ++i) {
    nvinfer1::DynamicPluginTensorDesc const& desc =
        i < nbInputs ? in[i] : out[i - nbInputs];
    if (desc.desc.format != nvinfer1::TensorFormat::kLINEAR ||
        !type_matches(i, desc.desc.type)) {
      return false;
    }
  }

  nvinfer1::Dims const& input = in[0].desc.dims;
  nvinfer1::Dims const& weight0 = in[1].desc.dims;
  nvinfer1::Dims const& stage0_scale = in[2].desc.dims;
  nvinfer1::Dims const& bias0 = in[3].desc.dims;
  nvinfer1::Dims const& weight1 = in[4].desc.dims;
  nvinfer1::Dims const& bias1 = in[5].desc.dims;
  nvinfer1::Dims const& output = out[0].desc.dims;

  if (input.nbDims != 4 || weight0.nbDims != 4 ||
      stage0_scale.nbDims != 1 || bias0.nbDims != 1 ||
      weight1.nbDims != 4 || bias1.nbDims != 1 || output.nbDims != 4) {
    return false;
  }
  if (!dim_is(weight0, 1, 1) || !dim_is(weight0, 2, 1) ||
      !dim_is(weight1, 1, 1) || !dim_is(weight1, 2, 1)) {
    return false;
  }
  if (is_known_dim(input.d[3]) && is_known_dim(weight0.d[3]) &&
      input.d[3] != weight0.d[3]) {
    return false;
  }
  if (is_known_dim(weight0.d[0]) && is_known_dim(stage0_scale.d[0]) &&
      weight0.d[0] != stage0_scale.d[0]) {
    return false;
  }
  if (is_known_dim(weight0.d[0]) && is_known_dim(bias0.d[0]) &&
      weight0.d[0] != bias0.d[0]) {
    return false;
  }
  if (is_known_dim(weight0.d[0]) && is_known_dim(weight1.d[3]) &&
      weight0.d[0] != weight1.d[3]) {
    return false;
  }
  if (is_known_dim(weight1.d[0]) && is_known_dim(bias1.d[0]) &&
      weight1.d[0] != bias1.d[0]) {
    return false;
  }
  if (is_known_dim(input.d[0]) && is_known_dim(output.d[0]) &&
      input.d[0] != output.d[0]) {
    return false;
  }
  if (is_known_dim(input.d[1]) && is_known_dim(output.d[1]) &&
      input.d[1] != output.d[1]) {
    return false;
  }
  if (is_known_dim(input.d[2]) && is_known_dim(output.d[2]) &&
      input.d[2] != output.d[2]) {
    return false;
  }
  if (is_known_dim(weight1.d[0]) && is_known_dim(output.d[3]) &&
      weight1.d[0] != output.d[3]) {
    return false;
  }
  return true;
}

float parse_output_alpha(nvinfer1::PluginFieldCollection const* fields) {
  if (!fields || !fields->fields) {
    return 1.0f;
  }
  for (int32_t i = 0; i < fields->nbFields; ++i) {
    nvinfer1::PluginField const& field = fields->fields[i];
    if (field.name && std::strcmp(field.name, kOutputAlphaField) == 0 &&
        field.type == nvinfer1::PluginFieldType::kFLOAT32 &&
        field.length == 1 && field.data) {
      return *static_cast<float const*>(field.data);
    }
  }
  return 1.0f;
}

}  // namespace

Plugin::Plugin(float output_alpha) : output_alpha_(output_alpha) {
  init_fields_to_serialize();
}

Plugin::Plugin(Plugin const& other) : output_alpha_(other.output_alpha_) {
  init_fields_to_serialize();
}

void Plugin::init_fields_to_serialize() {
  fields_to_serialize_storage_.clear();
  fields_to_serialize_storage_.emplace_back(
      kOutputAlphaField,
      &output_alpha_,
      nvinfer1::PluginFieldType::kFLOAT32,
      1);
  fields_to_serialize_.nbFields =
      static_cast<int32_t>(fields_to_serialize_storage_.size());
  fields_to_serialize_.fields = fields_to_serialize_storage_.data();
}

nvinfer1::IPluginCapability* Plugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept {
  if (type == nvinfer1::PluginCapabilityType::kBUILD) {
    return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
  }
  if (type == nvinfer1::PluginCapabilityType::kRUNTIME) {
    return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
  }
  if (type == nvinfer1::PluginCapabilityType::kCORE) {
    return static_cast<nvinfer1::IPluginV3OneCore*>(this);
  }
  return nullptr;
}

nvinfer1::IPluginV3* Plugin::clone() noexcept {
  return new Plugin(*this);
}

char const* Plugin::getPluginName() const noexcept {
  return kPluginName;
}

char const* Plugin::getPluginVersion() const noexcept {
  return kPluginVersion;
}

char const* Plugin::getPluginNamespace() const noexcept {
  return kPluginNamespace;
}

int32_t Plugin::getNbOutputs() const noexcept {
  return kNbOutputs;
}

int32_t Plugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
  return validate_build_desc(in, nbInputs, out, nbOutputs) &&
                 output_alpha_ > 0.0f
             ? 0
             : -1;
}

bool Plugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  if (nbInputs != kNbInputs || nbOutputs != kNbOutputs ||
      pos < 0 || pos >= nbInputs + nbOutputs) {
    return false;
  }
  return inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
         type_matches(pos, inOut[pos].desc.type);
}

int32_t Plugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
  if (nbInputs != kNbInputs || nbOutputs != kNbOutputs ||
      !inputTypes || !outputTypes) {
    return -1;
  }
  for (int32_t i = 0; i < nbInputs; ++i) {
    if (!type_matches(i, inputTypes[i])) {
      return -1;
    }
  }
  outputTypes[0] = nvinfer1::DataType::kFP8;
  return 0;
}

int32_t Plugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::DimsExprs const*,
    int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs,
    nvinfer1::IExprBuilder&) noexcept {
  if (nbInputs != kNbInputs || nbShapeInputs != 0 || nbOutputs != kNbOutputs ||
      inputs[0].nbDims != 4 || inputs[4].nbDims != 4) {
    return -1;
  }

  outputs[0].nbDims = 4;
  outputs[0].d[0] = inputs[0].d[0];
  outputs[0].d[1] = inputs[0].d[1];
  outputs[0].d[2] = inputs[0].d[2];
  outputs[0].d[3] = inputs[4].d[0];
  return 0;
}

size_t Plugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept {
  if (!validate_build_desc(inputs, nbInputs, outputs, nbOutputs)) {
    return 0;
  }
  return stage0_workspace_bytes(make_problem(inputs));
}

char const* Plugin::getTimingCacheID() noexcept {
  return "conv1x1_relu_conv1x1_relu_v1";
}

char const* Plugin::getMetadataString() noexcept {
  return "conv1x1_relu_conv1x1_relu";
}

int32_t Plugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
  if (!in || !out || nbInputs != kNbInputs || nbOutputs != kNbOutputs) {
    return -1;
  }
  return in[0].dims.nbDims == 4 && in[1].dims.nbDims == 4 &&
                 in[4].dims.nbDims == 4 && out[0].dims.nbDims == 4
             ? 0
             : -1;
}

int32_t Plugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const*,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  if (!inputDesc || !inputs || !outputs || !workspace ||
      inputDesc[0].dims.nbDims != 4 || inputDesc[1].dims.nbDims != 4 ||
      inputDesc[4].dims.nbDims != 4) {
    return -1;
  }

  core::Arguments<CoreElement> args;
  args.problem = core::Problem{
      static_cast<int>(inputDesc[0].dims.d[0]),
      static_cast<int>(inputDesc[0].dims.d[1]),
      static_cast<int>(inputDesc[0].dims.d[2]),
      static_cast<int>(inputDesc[0].dims.d[3]),
      static_cast<int>(inputDesc[1].dims.d[0]),
      static_cast<int>(inputDesc[4].dims.d[0]),
  };
  args.input = reinterpret_cast<CoreElement const*>(inputs[0]);
  args.weight0 = reinterpret_cast<CoreElement const*>(inputs[1]);
  args.stage0 = reinterpret_cast<CoreElement*>(workspace);
  args.stage0_scale = reinterpret_cast<float const*>(inputs[2]);
  args.bias0 = reinterpret_cast<float const*>(inputs[3]);
  args.weight1 = reinterpret_cast<CoreElement const*>(inputs[4]);
  args.bias1 = reinterpret_cast<CoreElement const*>(inputs[5]);
  args.output = reinterpret_cast<CoreElement*>(outputs[0]);
  args.stage0_alpha = 1.0f;
  args.output_alpha = output_alpha_;
  args.stream = stream;

  cutlass::Status status = core::conv1x1_relu_conv1x1_relu<CoreElement>(args);
  return status == cutlass::Status::kSuccess ? 0 : -1;
}

nvinfer1::IPluginV3* Plugin::attachToContext(
    nvinfer1::IPluginResourceContext*) noexcept {
  return clone();
}

nvinfer1::PluginFieldCollection const* Plugin::getFieldsToSerialize() noexcept {
  return &fields_to_serialize_;
}

Creator::Creator() {
  fields_.emplace_back(
      kOutputAlphaField,
      nullptr,
      nvinfer1::PluginFieldType::kFLOAT32,
      1);
  field_collection_.nbFields = static_cast<int32_t>(fields_.size());
  field_collection_.fields = fields_.data();
}

char const* Creator::getPluginName() const noexcept {
  return kPluginName;
}

char const* Creator::getPluginVersion() const noexcept {
  return kPluginVersion;
}

char const* Creator::getPluginNamespace() const noexcept {
  return kPluginNamespace;
}

nvinfer1::PluginFieldCollection const* Creator::getFieldNames() noexcept {
  return &field_collection_;
}

nvinfer1::IPluginV3* Creator::createPlugin(
    char const*,
    nvinfer1::PluginFieldCollection const* fields,
    nvinfer1::TensorRTPhase) noexcept {
  return new Plugin(parse_output_alpha(fields));
}

}  // namespace tiny_cutlass::conv_fused::binding::tensorrt::conv1x1_relu_conv1x1_relu

namespace {

tiny_cutlass::conv_fused::binding::tensorrt::conv1x1_relu_conv1x1_relu::Creator&
creator() {
  static tiny_cutlass::conv_fused::binding::tensorrt::conv1x1_relu_conv1x1_relu::
      Creator instance;
  return instance;
}

}  // namespace

extern "C" __declspec(dllexport)
nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators) {
  static nvinfer1::IPluginCreatorInterface* creators[] = {&creator()};
  nbCreators = 1;
  return creators;
}
