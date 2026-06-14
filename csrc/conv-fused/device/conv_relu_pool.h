#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/arch/arch.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

namespace tiny_cutlass::conv_fused::device {

template <
    typename ArchTag = cutlass::arch::Sm89,
    typename Element = cutlass::half_t>
class ConvReluPool {
 public:
  struct Arguments {
    int batch = 0;
    int height = 0;
    int width = 0;
    int channels = 0;
    int hidden_channels = 0;
    int output_channels = 0;
    Element const* input = nullptr;
    Element const* weight0 = nullptr;
    Element const* bias0 = nullptr;
    Element const* weight1 = nullptr;
    Element const* bias1 = nullptr;
    Element* output = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
  };

  struct WorkspaceLayout {
    size_t stage0_offset = 0;
    size_t stage1_offset = 0;
    size_t stage2_offset = 0;
    size_t scratch_offset = 0;
    size_t scratch_bytes = 0;
    size_t total_bytes = 0;
  };

  struct Params {
    Arguments args;
    WorkspaceLayout layout;
    cutlass::conv::Conv2dProblemSize conv0_problem;
    cutlass::conv::Conv2dProblemSize conv1_problem;
    Element* stage0 = nullptr;
    Element* stage1 = nullptr;
    Element* stage2 = nullptr;
    void* scratch = nullptr;
    int pool0_h = 0;
    int pool0_w = 0;
  };

 private:
  Params params_;

 public:
  ConvReluPool() = default;

  static cutlass::Status can_implement(Arguments const& args);

  static size_t get_workspace_size(Arguments const& args);

  cutlass::Status initialize(
      Arguments const& args,
      cudaStream_t stream = nullptr);

  cutlass::Status run(cudaStream_t stream = nullptr);

  cutlass::Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  cutlass::Status operator()(
      Arguments const& args,
      cudaStream_t stream = nullptr) {
    cutlass::Status status = initialize(args, stream);
    if (status == cutlass::Status::kSuccess) {
      status = run(stream);
    }
    return status;
  }
};

template <
    typename ArchTag = cutlass::arch::Sm89,
    typename Element = cutlass::half_t>
size_t conv_relu_pool_workspace_size(
    int batch,
    int height,
    int width,
    int channels,
    int hidden_channels,
    int output_channels);

template <
    typename ArchTag = cutlass::arch::Sm89,
    typename Element = cutlass::half_t>
cutlass::Status run_conv_relu_pool(
    int batch,
    int height,
    int width,
    int channels,
    int hidden_channels,
    int output_channels,
    Element const* input,
    Element const* weight0,
    Element const* bias0,
    Element const* weight1,
    Element const* bias1,
    Element* output,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

}  // namespace tiny_cutlass::conv_fused::device
