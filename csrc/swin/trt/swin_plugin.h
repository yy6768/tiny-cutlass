#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <NvInfer.h>
#include <NvInferRuntimePlugin.h>

namespace tiny_cutlass {
namespace swin {
namespace trt {

struct SwinPluginConfig {
  int32_t batch_size = 1;
  int32_t image_size = 14;
  int32_t window_size = 7;
  int32_t shift_size = 3;
  int32_t head_number = 3;
  int32_t head_size = 32;
  float scale = 0.176776695f;
};

class SwinPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  explicit SwinPlugin(char const* name, SwinPluginConfig const& config);
  SwinPlugin(char const* name, void const* serial_data, std::size_t serial_length);

  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
      int32_t output_index,
      nvinfer1::DimsExprs const* inputs,
      int32_t nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept override;
  bool supportsFormatCombination(
      int32_t pos,
      nvinfer1::PluginTensorDesc const* in_out,
      int32_t nb_inputs,
      int32_t nb_outputs) noexcept override;
  void configurePlugin(
      nvinfer1::DynamicPluginTensorDesc const* in,
      int32_t nb_inputs,
      nvinfer1::DynamicPluginTensorDesc const* out,
      int32_t nb_outputs) noexcept override;
  std::size_t getWorkspaceSize(
      nvinfer1::PluginTensorDesc const* inputs,
      int32_t nb_inputs,
      nvinfer1::PluginTensorDesc const* outputs,
      int32_t nb_outputs) const noexcept override;
  int32_t enqueue(
      nvinfer1::PluginTensorDesc const* input_desc,
      nvinfer1::PluginTensorDesc const* output_desc,
      void const* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override;

  nvinfer1::DataType getOutputDataType(
      int32_t index,
      nvinfer1::DataType const* input_types,
      int32_t nb_inputs) const noexcept override;

  char const* getPluginType() const noexcept override;
  char const* getPluginVersion() const noexcept override;
  int32_t getNbOutputs() const noexcept override;
  void configureWithFormat(
      nvinfer1::Dims const* input_dims,
      int32_t nb_inputs,
      nvinfer1::Dims const* output_dims,
      int32_t nb_outputs,
      nvinfer1::DataType type,
      nvinfer1::PluginFormat format,
      int32_t max_batch_size) noexcept override;
  int32_t initialize() noexcept override;
  void terminate() noexcept override;
  std::size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(char const* plugin_namespace) noexcept override;
  char const* getPluginNamespace() const noexcept override;

 private:
  std::string name_;
  std::string namespace_;
  SwinPluginConfig config_;
};

class SwinPluginCreator : public nvinfer1::IPluginCreator {
 public:
  SwinPluginCreator();

  char const* getPluginName() const noexcept override;
  char const* getPluginVersion() const noexcept override;
  nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
  nvinfer1::IPluginV2* createPlugin(
      char const* name,
      nvinfer1::PluginFieldCollection const* fc) noexcept override;
  nvinfer1::IPluginV2* deserializePlugin(
      char const* name,
      void const* serial_data,
      std::size_t serial_length) noexcept override;
  void setPluginNamespace(char const* plugin_namespace) noexcept override;
  char const* getPluginNamespace() const noexcept override;

 private:
  std::string namespace_;
  nvinfer1::PluginFieldCollection fields_;
};

} // namespace trt
} // namespace swin
} // namespace tiny_cutlass
