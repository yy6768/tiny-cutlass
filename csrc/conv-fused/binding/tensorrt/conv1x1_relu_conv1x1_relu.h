#pragma once

#include <vector>

#include "NvInfer.h"

namespace tiny_cutlass::conv_fused::binding::tensorrt::conv1x1_relu_conv1x1_relu {

class Plugin final
    : public nvinfer1::IPluginV3,
      public nvinfer1::IPluginV3OneCore,
      public nvinfer1::IPluginV3OneBuild,
      public nvinfer1::IPluginV3OneRuntime {
 public:
  explicit Plugin(float output_alpha = 1.0f);
  Plugin(Plugin const& other);

  nvinfer1::IPluginCapability* getCapabilityInterface(
      nvinfer1::PluginCapabilityType type) noexcept override;
  nvinfer1::IPluginV3* clone() noexcept override;

  char const* getPluginName() const noexcept override;
  char const* getPluginVersion() const noexcept override;
  char const* getPluginNamespace() const noexcept override;

  int32_t getNbOutputs() const noexcept override;
  int32_t configurePlugin(
      nvinfer1::DynamicPluginTensorDesc const* in,
      int32_t nbInputs,
      nvinfer1::DynamicPluginTensorDesc const* out,
      int32_t nbOutputs) noexcept override;
  bool supportsFormatCombination(
      int32_t pos,
      nvinfer1::DynamicPluginTensorDesc const* inOut,
      int32_t nbInputs,
      int32_t nbOutputs) noexcept override;
  int32_t getOutputDataTypes(
      nvinfer1::DataType* outputTypes,
      int32_t nbOutputs,
      nvinfer1::DataType const* inputTypes,
      int32_t nbInputs) const noexcept override;
  int32_t getOutputShapes(
      nvinfer1::DimsExprs const* inputs,
      int32_t nbInputs,
      nvinfer1::DimsExprs const* shapeInputs,
      int32_t nbShapeInputs,
      nvinfer1::DimsExprs* outputs,
      int32_t nbOutputs,
      nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  size_t getWorkspaceSize(
      nvinfer1::DynamicPluginTensorDesc const* inputs,
      int32_t nbInputs,
      nvinfer1::DynamicPluginTensorDesc const* outputs,
      int32_t nbOutputs) const noexcept override;
  char const* getTimingCacheID() noexcept override;
  char const* getMetadataString() noexcept override;

  int32_t onShapeChange(
      nvinfer1::PluginTensorDesc const* in,
      int32_t nbInputs,
      nvinfer1::PluginTensorDesc const* out,
      int32_t nbOutputs) noexcept override;
  int32_t enqueue(
      nvinfer1::PluginTensorDesc const* inputDesc,
      nvinfer1::PluginTensorDesc const* outputDesc,
      void const* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override;
  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext* context) noexcept override;
  nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

 private:
  void init_fields_to_serialize();

  float output_alpha_ = 1.0f;
  nvinfer1::PluginFieldCollection fields_to_serialize_{};
  std::vector<nvinfer1::PluginField> fields_to_serialize_storage_{};
};

class Creator final : public nvinfer1::IPluginCreatorV3One {
 public:
  Creator();

  char const* getPluginName() const noexcept override;
  char const* getPluginVersion() const noexcept override;
  char const* getPluginNamespace() const noexcept override;
  nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
  nvinfer1::IPluginV3* createPlugin(
      char const* name,
      nvinfer1::PluginFieldCollection const* fields,
      nvinfer1::TensorRTPhase phase) noexcept override;

 private:
  nvinfer1::PluginFieldCollection field_collection_{};
  std::vector<nvinfer1::PluginField> fields_{};
};

}  // namespace tiny_cutlass::conv_fused::binding::tensorrt::conv1x1_relu_conv1x1_relu
