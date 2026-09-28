#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include "../../flash-attention/flash_attention.h"

inline cudaError_t run_cudnn_reference(
    Problem const& problem,
    Tensors const& tensors,
    cudaStream_t stream,
    std::string& error_message) {
  namespace fe = cudnn_frontend;

  static constexpr int64_t kQUid = 1;
  static constexpr int64_t kKUid = 2;
  static constexpr int64_t kVUid = 3;
  static constexpr int64_t kOUid = 4;
  static constexpr int64_t kStatsUid = 5;

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

  auto create_graph = [](Problem const& p, bool generate_stats) {
    int64_t const b = p.batch_size;
    int64_t const h = p.head_number;
    int64_t const sq = p.seq_length;
    int64_t const sk = p.seq_length_kv;
    int64_t const d = p.head_size;
    int64_t const dv = p.head_size_v;

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(kQUid)
                               .set_dim({b, h, sq, d})
                               .set_stride({sq * h * d, d, h * d, 1}));

    auto k = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(kKUid)
                               .set_dim({b, h, sk, d})
                               .set_stride({sk * h * d, d, h * d, 1}));

    auto v = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(kVUid)
                               .set_dim({b, h, sk, dv})
                               .set_stride({sk * h * dv, dv, h * dv, 1}));

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("flash_attention_reference")
                            .set_generate_stats(generate_stats)
                            .set_attn_scale(p.scale);

    auto [o, stats] = graph->sdpa(q, k, v, sdpa_options);
    o->set_output(true)
        .set_dim({b, h, sq, dv})
        .set_stride({sq * h * dv, dv, h * dv, 1})
        .set_uid(kOUid);

    if (generate_stats) {
      stats->set_output(true)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({b, h, sq, 1})
          .set_stride({h * sq, sq, 1, 1})
          .set_uid(kStatsUid);
    }
    return graph;
  };

  if (cudnnGetVersion() < 8903) {
    error_message = "cuDNN SDPA reference requires cuDNN 8.9.3 or newer.";
    return cudaErrorNotSupported;
  }

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

  auto graph = create_graph(problem, tensors.logsumexp != nullptr);
  auto build_status = graph->build(handle.handle, {fe::HeurMode_t::A});
  if (!build_status.is_good()) {
    error_message = build_status.get_message();
    return cudaErrorNotSupported;
  }

  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> tensor_map = {
      {kQUid, const_cast<Element*>(tensors.query)},
      {kKUid, const_cast<Element*>(tensors.key)},
      {kVUid, const_cast<Element*>(tensors.value)},
      {kOUid, tensors.output},
  };
  if (tensors.logsumexp) tensor_map.emplace(kStatsUid, tensors.logsumexp);

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

// Training backward follows samples/cpp/sdpa/fp16_bwd.cpp. O and natural-log
// Stats must come from the independent cuDNN training forward for final parity.
inline cudaError_t run_cudnn_backward_reference(
    Problem const& p, BackwardTensors const& t, cudaStream_t stream,
    std::string& error_message) {
  namespace fe = cudnn_frontend;
  struct CudnnHandle {
    cudnnHandle_t handle = nullptr;
    CudnnHandle() { cudnnCreate(&handle); }
    ~CudnnHandle() { if (handle) cudnnDestroy(handle); }
  } handle;
  if (!handle.handle) {
    error_message = "cudnnCreate failed.";
    return cudaErrorUnknown;
  }
  cudnnStatus_t status = cudnnSetStream(handle.handle, stream);
  if (status != CUDNN_STATUS_SUCCESS) {
    error_message = cudnnGetErrorString(status);
    return cudaErrorUnknown;
  }

  int64_t b = p.batch_size, h = p.head_number, sq = p.seq_length;
  int64_t sk = p.seq_length_kv, d = p.head_size, dv = p.head_size_v;
  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(fe::DataType_t::HALF)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  auto input = [&](char const* name, int64_t uid, int64_t sequence, int64_t dim) {
    return graph->tensor(fe::graph::Tensor_attributes().set_name(name).set_uid(uid)
        .set_dim({b, h, sequence, dim})
        .set_stride({sequence * h * dim, dim, h * dim, 1}));
  };
  auto q = input("Q", 1, sq, d);
  auto k = input("K", 2, sk, d);
  auto v = input("V", 3, sk, dv);
  auto o = input("O", 4, sq, dv);
  auto grad_o = input("dO", 6, sq, dv);
  auto stats = graph->tensor(fe::graph::Tensor_attributes().set_name("Stats").set_uid(5)
      .set_dim({b, h, sq, 1}).set_stride({h * sq, sq, 1, 1})
      .set_data_type(fe::DataType_t::FLOAT));
  auto options = fe::graph::SDPA_backward_attributes()
      .set_name("flash_attention_backward_reference")
      .set_attn_scale(p.scale).set_deterministic_algorithm(true);
  auto [grad_q, grad_k, grad_v] = graph->sdpa_backward(q, k, v, o, grad_o, stats, options);
  grad_q->set_output(true).set_uid(7).set_dim({b, h, sq, d})
      .set_stride({sq * h * d, d, h * d, 1});
  grad_k->set_output(true).set_uid(8).set_dim({b, h, sk, d})
      .set_stride({sk * h * d, d, h * d, 1});
  grad_v->set_output(true).set_uid(9).set_dim({b, h, sk, dv})
      .set_stride({sk * h * dv, dv, h * dv, 1});
  auto built = graph->build(handle.handle, {fe::HeurMode_t::A});
  if (!built.is_good()) {
    error_message = built.get_message();
    return cudaErrorNotSupported;
  }
  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> tensor_map = {
      {1, const_cast<Element*>(t.query)}, {2, const_cast<Element*>(t.key)},
      {3, const_cast<Element*>(t.value)}, {4, const_cast<Element*>(t.output)},
      {5, const_cast<float*>(t.logsumexp)}, {6, const_cast<Element*>(t.grad_output)},
      {7, t.grad_query}, {8, t.grad_key}, {9, t.grad_value}};
  int64_t workspace_size = 0;
  auto size_status = graph->get_workspace_size(workspace_size);
  if (!size_status.is_good()) {
    error_message = size_status.get_message();
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
  auto executed = graph->execute(handle.handle, tensor_map, workspace);
  if (workspace) cudaFree(workspace);
  if (!executed.is_good()) {
    error_message = executed.get_message();
    return cudaErrorUnknown;
  }
  return cudaGetLastError();
}
