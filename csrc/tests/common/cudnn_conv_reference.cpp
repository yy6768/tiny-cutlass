// cuDNN b2b conv reference implementation. Compiled as HOST C++ (see the
// header for why this must not go through nvcc).
//
// The reference computes conv0 -> +bias0 -> ReLU -> conv1 -> +bias1 as TWO
// separate 1x1-conv cuDNN graphs staged through a scratch buffer. cuDNN cannot
// express the two convolutions as one fused graph (mmaNodeCount > 1 is
// unsupported on SM89) -- that back-to-back mainloop fusion is exactly the
// CUTLASS-only capability under test. So the reference reproduces the math, not
// the fusion, which is all a numerical cross-check needs.

#include "cudnn_conv_reference.h"

#include <memory>
#include <unordered_map>
#include <vector>

#include <cudnn.h>
#include <cudnn_frontend.h>

namespace tiny_cutlass::testing {
namespace {

namespace fe = cudnn_frontend;

struct CudnnHandle {
  cudnnHandle_t handle = nullptr;
  CudnnHandle() { cudnnCreate(&handle); }
  ~CudnnHandle() {
    if (handle) {
      cudnnDestroy(handle);
    }
  }
};

// NHWC strides for a cudnn tensor whose dims are (n, c, h, w).
std::vector<int64_t> nhwc_strides(int64_t c, int64_t h, int64_t w) {
  return {c * h * w, 1, c * w, c};
}

// Builds and executes one 1x1 conv graph: conv_fprop(x, w) -> +bias -> [ReLU],
// writing packed-NHWC half output to `out`. Each operand gets a distinct uid so
// the variant pack can bind device pointers by role.
cudaError_t run_single_conv1x1(
    cudnnHandle_t handle,
    int64_t n, int64_t h, int64_t w, int64_t cin, int64_t cout,
    void const* x, void const* filter, void const* bias, bool relu,
    void* out, std::string& error_message) {
  enum Uid : int64_t { kX = 1, kW = 2, kB = 3, kY = 4 };

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(fe::DataType_t::HALF)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

  // Tensor-attribute boilerplate, once.
  auto tensor = [&](char const* name, int64_t uid,
                    std::vector<int64_t> dim, std::vector<int64_t> stride) {
    return graph->tensor(fe::graph::Tensor_attributes()
                             .set_name(name)
                             .set_uid(uid)
                             .set_dim(std::move(dim))
                             .set_stride(std::move(stride)));
  };
  auto add = [&](auto in, auto operand, char const* name, fe::PointwiseMode_t mode) {
    return graph->pointwise(
        in, operand,
        fe::graph::Pointwise_attributes().set_name(name).set_mode(mode));
  };

  auto xt = tensor("x", kX, {n, cin, h, w}, nhwc_strides(cin, h, w));
  auto wt = tensor("w", kW, {cout, cin, 1, 1}, {cin, 1, 1, 1});
  auto bt = tensor("b", kB, {1, cout, 1, 1}, {cout, 1, 1, 1});

  auto yc = graph->conv_fprop(
      xt, wt,
      fe::graph::Conv_fprop_attributes()
          .set_name("conv").set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1}));
  auto yb = add(yc, bt, "bias", fe::PointwiseMode_t::ADD);
  auto y = relu ? graph->pointwise(
                      yb, fe::graph::Pointwise_attributes().set_name("relu").set_mode(
                              fe::PointwiseMode_t::RELU_FWD))
                : yb;
  y->set_output(true).set_dim({n, cout, h, w}).set_stride(nhwc_strides(cout, h, w)).set_uid(kY);

  if (auto s = graph->build(handle, {fe::HeurMode_t::A}); !s.is_good()) {
    error_message = s.get_message();
    return cudaErrorNotSupported;
  }

  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {
      {kX, const_cast<void*>(x)},
      {kW, const_cast<void*>(filter)},
      {kB, const_cast<void*>(bias)},
      {kY, out},
  };

  int64_t workspace_size = 0;
  if (auto s = graph->get_workspace_size(workspace_size); !s.is_good()) {
    error_message = s.get_message();
    return cudaErrorUnknown;
  }

  void* workspace = nullptr;
  if (workspace_size > 0) {
    if (cudaError_t e = cudaMalloc(&workspace, size_t(workspace_size)); e != cudaSuccess) {
      error_message = cudaGetErrorString(e);
      return e;
    }
  }

  auto exec = graph->execute(handle, pack, workspace);
  if (workspace) {
    cudaFree(workspace);
  }
  if (!exec.is_good()) {
    error_message = exec.get_message();
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

}  // namespace

cudaError_t run_cudnn_conv_dual_reference_half(
    ConvDualProblem const& p,
    ConvDualTensorsHalf const& tensors,
    cudaStream_t stream,
    std::string& error_message) {
  int64_t const n = p.batch, h = p.height, w = p.width;
  int64_t const c0 = p.channels, k0 = p.hidden_channels, k1 = p.output_channels;

  CudnnHandle handle;
  if (!handle.handle) {
    error_message = "cudnnCreate failed.";
    return cudaErrorUnknown;
  }
  if (auto s = cudnnSetStream(handle.handle, stream); s != CUDNN_STATUS_SUCCESS) {
    error_message = cudnnGetErrorString(s);
    return cudaErrorUnknown;
  }

  // Scratch holds relu(conv0(x) + bias0): (N, H, W, K0) packed half.
  void* scratch = nullptr;
  size_t scratch_bytes = size_t(n) * h * w * k0 * sizeof(uint16_t);
  if (cudaError_t e = cudaMalloc(&scratch, scratch_bytes); e != cudaSuccess) {
    error_message = cudaGetErrorString(e);
    return e;
  }

  cudaError_t status = run_single_conv1x1(
      handle.handle, n, h, w, c0, k0,
      tensors.input, tensors.weight0, tensors.bias0, /*relu=*/true,
      scratch, error_message);
  if (status == cudaSuccess) {
    status = run_single_conv1x1(
        handle.handle, n, h, w, k0, k1,
        scratch, tensors.weight1, tensors.bias1, /*relu=*/false,
        tensors.output, error_message);
  }

  cudaFree(scratch);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaGetLastError();
}

}  // namespace tiny_cutlass::testing
