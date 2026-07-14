#include "reference.h"

#include <cstdint>
#include <memory>
#include <unordered_map>

#include <cudnn.h>
#include <cudnn_frontend.h>

namespace tiny_cutlass {
namespace swin {
namespace {

struct CudnnHandle {
  cudnnHandle_t handle = nullptr;

  CudnnHandle() {
    cudnnCreate(&handle);
  }

  ~CudnnHandle() {
    if (handle) {
      cudnnDestroy(handle);
    }
  }
};

} // namespace

cudaError_t run_cudnn_swin_attention_reference(
    SwinAttentionProblem const& problem,
    cutlass::half_t const* query,
    cutlass::half_t const* key,
    cutlass::half_t const* value,
    cutlass::half_t const* attention_bias,
    cutlass::half_t* output,
    cudaStream_t stream,
    std::string& error_message) {
  namespace fe = cudnn_frontend;

  static constexpr int64_t kQUid = 1;
  static constexpr int64_t kKUid = 2;
  static constexpr int64_t kVUid = 3;
  static constexpr int64_t kBiasUid = 4;
  static constexpr int64_t kOUid = 5;

  if (cudnnGetVersion() < 8903) {
    error_message = "cuDNN SDPA reference requires cuDNN 8.9.3 or newer.";
    return cudaErrorNotSupported;
  }

  int64_t const bw = swin_batched_windows(problem);
  int64_t const h = problem.head_number;
  int64_t const l = swin_window_len(problem);
  int64_t const lp = swin_window_len_padded(problem);
  int64_t const d = problem.head_size;
  int64_t const c = swin_channels(problem);

  CudnnHandle handle;
  if (!handle.handle) {
    error_message = "cudnnCreate failed.";
    return cudaErrorUnknown;
  }

  cudnnStatus_t status = cudnnSetStream(handle.handle, stream);
  if (status != CUDNN_STATUS_SUCCESS) {
    error_message = cudnnGetErrorString(status);
    return cudaErrorUnknown;
  }

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(fe::DataType_t::HALF)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

  auto q = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("Q")
                             .set_uid(kQUid)
                             .set_dim({bw, h, l, d})
                             .set_stride({l * c, d, c, 1}));
  auto k = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("K")
                             .set_uid(kKUid)
                             .set_dim({bw, h, l, d})
                             .set_stride({l * c, d, c, 1}));
  auto v = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("V")
                             .set_uid(kVUid)
                             .set_dim({bw, h, l, d})
                             .set_stride({l * c, d, c, 1}));
  auto bias = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Bias")
                                .set_uid(kBiasUid)
                                .set_dim({bw, h, l, l})
                                .set_stride({h * l * lp, l * lp, lp, 1}));

  auto options = fe::graph::SDPA_attributes()
                     .set_name("swin_window_attention_reference")
                     .set_generate_stats(false)
                     .set_attn_scale(problem.scale)
                     .set_bias(bias);
  auto [o, stats] = graph->sdpa(q, k, v, options);
  o->set_output(true)
      .set_dim({bw, h, l, d})
      .set_stride({l * c, d, c, 1})
      .set_uid(kOUid);
  (void)stats;

  auto build_status = graph->build(handle.handle, {fe::HeurMode_t::A});
  if (!build_status.is_good()) {
    error_message = build_status.get_message();
    return cudaErrorNotSupported;
  }

  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> tensor_map = {
      {kQUid, const_cast<cutlass::half_t*>(query)},
      {kKUid, const_cast<cutlass::half_t*>(key)},
      {kVUid, const_cast<cutlass::half_t*>(value)},
      {kBiasUid, const_cast<cutlass::half_t*>(attention_bias)},
      {kOUid, output},
  };

  int64_t workspace_size = 0;
  auto workspace_status = graph->get_workspace_size(workspace_size);
  if (!workspace_status.is_good()) {
    error_message = workspace_status.get_message();
    return cudaErrorUnknown;
  }

  void* workspace = nullptr;
  if (workspace_size > 0) {
    cudaError_t err = cudaMalloc(&workspace, std::size_t(workspace_size));
    if (err != cudaSuccess) {
      error_message = cudaGetErrorString(err);
      return err;
    }
  }

  auto execute_status = graph->execute(handle.handle, tensor_map, workspace);
  if (workspace) {
    cudaFree(workspace);
  }
  if (!execute_status.is_good()) {
    error_message = execute_status.get_message();
    return cudaErrorUnknown;
  }
  return cudaGetLastError();
}

} // namespace swin
} // namespace tiny_cutlass
