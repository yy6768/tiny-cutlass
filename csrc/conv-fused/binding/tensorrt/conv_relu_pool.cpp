#include "binding/tensorrt/conv_relu_pool.h"

#include <cstdint>
#include <cstring>
#include <memory>

#include "cutlass/half.h"
#include "ops/conv_relu_pool.h"

namespace tiny_cutlass::conv_fused::binding::tensorrt::conv_relu_pool {
namespace {

constexpr char const* kPluginName = "conv_relu_pool";
constexpr char const* kPluginVersion = "1";
constexpr char const* kPluginNamespace = "tiny_cutlass";
constexpr int32_t kNbInputs = 5;
constexpr int32_t kNbOutputs = 1;

bool is_half_linear(nvinfer1::DynamicPluginTensorDesc const& desc) {
  return desc.desc.type == nvinfer1::DataType::kHALF &&
         desc.desc.format == nvinfer1::TensorFormat::kLINEAR;
}

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

tiny_cutlass::conv_fused::ConvReluPoolProblem make_problem(
    nvinfer1::DynamicPluginTensorDesc const* inputs) {
  return tiny_cutlass::conv_fused::ConvReluPoolProblem{
      resolved_dim(inputs[0], 0),
      resolved_dim(inputs[0], 1),
      resolved_dim(inputs[0], 2),
      resolved_dim(inputs[0], 3),
      resolved_dim(inputs[1], 0),
      resolved_dim(inputs[3], 0),
  };
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
    if (!is_half_linear(desc)) {
      return false;
    }
  }

  nvinfer1::Dims const& input = in[0].desc.dims;
  nvinfer1::Dims const& weight0 = in[1].desc.dims;
  nvinfer1::Dims const& bias0 = in[2].desc.dims;
  nvinfer1::Dims const& weight1 = in[3].desc.dims;
  nvinfer1::Dims const& bias1 = in[4].desc.dims;
  nvinfer1::Dims const& output = out[0].desc.dims;

  if (input.nbDims != 4 || weight0.nbDims != 4 || bias0.nbDims != 1 ||
      weight1.nbDims != 4 || bias1.nbDims != 1 || output.nbDims != 4) {
    return false;
  }

  if (!dim_is(weight0, 1, 3) || !dim_is(weight0, 2, 3)) {
    return false;
  }
  if (!dim_is(weight1, 1, 1) || !dim_is(weight1, 2, 1)) {
    return false;
  }
  if (is_known_dim(input.d[3]) && is_known_dim(weight0.d[3]) &&
      input.d[3] != weight0.d[3]) {
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
  return true;
}

}  // namespace

Plugin::Plugin() {
  init_fields_to_serialize();
}

Plugin::Plugin(Plugin const& other)
    : fields_to_serialize_(other.fields_to_serialize_) {
  init_fields_to_serialize();
}

void Plugin::init_fields_to_serialize() {
  fields_to_serialize_.nbFields = 0;
  fields_to_serialize_.fields = nullptr;
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
  return validate_build_desc(in, nbInputs, out, nbOutputs) ? 0 : -1;
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
  return is_half_linear(inOut[pos]);
}

int32_t Plugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
  if (nbInputs != kNbInputs || nbOutputs != kNbOutputs ||
      inputTypes[0] != nvinfer1::DataType::kHALF) {
    return -1;
  }
  outputTypes[0] = nvinfer1::DataType::kHALF;
  return 0;
}

int32_t Plugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::DimsExprs const*,
    int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  if (nbInputs != kNbInputs || nbShapeInputs != 0 || nbOutputs != kNbOutputs ||
      inputs[0].nbDims != 4 || inputs[3].nbDims != 4) {
    return -1;
  }

  outputs[0].nbDims = 4;
  outputs[0].d[0] = inputs[0].d[0];
  outputs[0].d[1] = exprBuilder.operation(
      nvinfer1::DimensionOperation::kFLOOR_DIV,
      *inputs[0].d[1],
      *exprBuilder.constant(4));
  outputs[0].d[2] = exprBuilder.operation(
      nvinfer1::DimensionOperation::kFLOOR_DIV,
      *inputs[0].d[2],
      *exprBuilder.constant(4));
  outputs[0].d[3] = inputs[3].d[0];
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
  return tiny_cutlass::conv_fused::conv_relu_pool_workspace_size<cutlass::half_t>(
      make_problem(inputs));
}

char const* Plugin::getTimingCacheID() noexcept {
  return "conv_relu_pool_v1";
}

char const* Plugin::getMetadataString() noexcept {
  return "conv3x3_relu_pool_conv1x1_relu_pool";
}

int32_t Plugin::onShapeChange(
    nvinfer1::PluginTensorDesc const*,
    int32_t nbInputs,
    nvinfer1::PluginTensorDesc const*,
    int32_t nbOutputs) noexcept {
  return (nbInputs == kNbInputs && nbOutputs == kNbOutputs) ? 0 : -1;
}

int32_t Plugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const*,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  using Element = cutlass::half_t;
  if (!inputDesc || !inputs || !outputs || inputDesc[0].dims.nbDims != 4 ||
      inputDesc[1].dims.nbDims != 4 || inputDesc[3].dims.nbDims != 4) {
    return -1;
  }

  tiny_cutlass::conv_fused::ConvReluPoolArguments<Element> args;
  args.problem = tiny_cutlass::conv_fused::ConvReluPoolProblem{
      static_cast<int>(inputDesc[0].dims.d[0]),
      static_cast<int>(inputDesc[0].dims.d[1]),
      static_cast<int>(inputDesc[0].dims.d[2]),
      static_cast<int>(inputDesc[0].dims.d[3]),
      static_cast<int>(inputDesc[1].dims.d[0]),
      static_cast<int>(inputDesc[3].dims.d[0]),
  };
  args.input = reinterpret_cast<Element const*>(inputs[0]);
  args.weight0 = reinterpret_cast<Element const*>(inputs[1]);
  args.bias0 = reinterpret_cast<Element const*>(inputs[2]);
  args.weight1 = reinterpret_cast<Element const*>(inputs[3]);
  args.bias1 = reinterpret_cast<Element const*>(inputs[4]);
  args.output = reinterpret_cast<Element*>(outputs[0]);
  args.workspace = workspace;
  args.workspace_bytes =
      tiny_cutlass::conv_fused::conv_relu_pool_workspace_size<Element>(
          args.problem);
  args.stream = stream;

  cutlass::Status status = tiny_cutlass::conv_fused::conv_relu_pool(args);
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
  field_collection_.nbFields = 0;
  field_collection_.fields = nullptr;
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
    nvinfer1::PluginFieldCollection const*,
    nvinfer1::TensorRTPhase) noexcept {
  return new Plugin();
}

}  // namespace tiny_cutlass::conv_fused::binding::tensorrt::conv_relu_pool

namespace {

nvinfer1::ILoggerFinder* g_logger_finder = nullptr;

tiny_cutlass::conv_fused::binding::tensorrt::conv_relu_pool::Creator&
creator() {
  static tiny_cutlass::conv_fused::binding::tensorrt::conv_relu_pool::Creator
      instance;
  return instance;
}

}  // namespace

extern "C" __declspec(dllexport) void setLoggerFinder(
    nvinfer1::ILoggerFinder* finder) {
  g_logger_finder = finder;
}

extern "C" __declspec(dllexport)
nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators) {
  static nvinfer1::IPluginCreatorInterface* creators[] = {&creator()};
  nbCreators = 1;
  return creators;
}
