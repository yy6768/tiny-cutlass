#pragma once

#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace tiny_cutlass::conv_fused {

inline void check_cuda_4d(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(!tensor.is_sparse(), name, " must be dense");
  TORCH_CHECK(tensor.dim() == 4, name, " must be 4D");
}

inline void check_bias(
    const at::Tensor& bias,
    const char* name,
    int64_t expected_size,
    at::ScalarType dtype) {
  TORCH_CHECK(bias.defined(), name, " must be defined");
  TORCH_CHECK(bias.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(!bias.is_sparse(), name, " must be dense");
  TORCH_CHECK(bias.dim() == 1, name, " must be 1D");
  TORCH_CHECK(bias.size(0) == expected_size, name, " length mismatch");
  TORCH_CHECK(bias.scalar_type() == dtype, name, " dtype mismatch");
}

inline std::array<int64_t, 2> expand_2d(
    const std::vector<int64_t>& values,
    const char* name) {
  TORCH_CHECK(values.size() == 1 || values.size() == 2, name, " must contain one or two values");
  if (values.size() == 1) {
    return {values[0], values[0]};
  }
  return {values[0], values[1]};
}

inline std::array<int64_t, 2> expand_2d_or(
    const std::vector<int64_t>& values,
    const char* name,
    std::array<int64_t, 2> default_value) {
  return values.empty() ? default_value : expand_2d(values, name);
}

inline void check_pool_args(
    std::array<int64_t, 2> kernel_size,
    std::array<int64_t, 2> stride,
    std::array<int64_t, 2> padding,
    std::array<int64_t, 2> dilation,
    bool is_avg_pool) {
  for (int dim = 0; dim < 2; ++dim) {
    TORCH_CHECK(kernel_size[dim] > 0, "kernel_size entries must be positive");
    TORCH_CHECK(stride[dim] > 0, "stride entries must be positive");
    TORCH_CHECK(padding[dim] >= 0, "padding entries must be non-negative");
    TORCH_CHECK(dilation[dim] > 0, "dilation entries must be positive");
    if (is_avg_pool) {
      TORCH_CHECK(dilation[dim] == 1, "avg_pool2d does not support dilation");
    }
    TORCH_CHECK(
        padding[dim] * 2 <= kernel_size[dim],
        "pool2d padding must be at most half of kernel_size");
  }
}

inline int checked_int64_to_int(int64_t value, const char* name) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), name, " is out of int range");
  return static_cast<int>(value);
}

inline std::optional<at::Tensor> normalize_optional_tensor(
    const std::optional<at::Tensor>& tensor) {
  if (tensor.has_value() && tensor->defined()) {
    return tensor;
  }
  return std::nullopt;
}

}  // namespace tiny_cutlass::conv_fused
