/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief Kernel entry for 02-tiled-online.

    The shared test harness owns allocation, cuDNN reference verification, and
    timing. This file only launches the 02 fused device path:

      tiled MM0 -> online softmax -> tiled MM1
*/

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "../flash_attention.h"
#include "kernel_forward.h"

namespace {

using scalar_t = Element;

template <
    typename ArchTag_,
    int kQueriesPerBlock_,
    int kKeysPerBlock_,
    int kMaxK_>
struct TiledOnlineAttentionConfig {
  using ArchTag = ArchTag_;

  static constexpr bool kIsAligned = true;
  static constexpr int kQueriesPerBlock = kQueriesPerBlock_;
  static constexpr int kKeysPerBlock = kKeysPerBlock_;
  static constexpr int kMaxK = kMaxK_;

  using Attention = AttentionKernel<
      scalar_t,
      ArchTag,
      kIsAligned,
      kQueriesPerBlock,
      kKeysPerBlock,
      kMaxK,
      false,
      false>;
};

template <typename Attention>
std::size_t workspace_bytes_for(Problem const& problem) {
  if constexpr (Attention::kNeedsOutputAccumulatorBuffer) {
    return total_output_elements(problem) * sizeof(typename Attention::output_accum_t);
  }
  return 0;
}

template <typename Attention>
typename Attention::Params make_params(
    Problem const& problem,
    Tensors const& tensors,
    typename Attention::output_accum_t* output_accum_ptr) {
  int B = problem.batch_size;
  int Sq = problem.seq_length;
  int Sk = problem.seq_length_kv;
  int H = problem.head_number;
  int d = problem.head_size;
  int dv = problem.head_size_v;

  int32_t q_strideH = d;
  int32_t k_strideH = d;
  int32_t v_strideH = dv;
  int32_t q_strideM = H * d;
  int32_t k_strideM = H * d;
  int32_t v_strideM = H * dv;
  int32_t o_strideM = H * dv;
  int64_t q_strideB = int64_t(Sq) * q_strideM;
  int64_t k_strideB = int64_t(Sk) * k_strideM;
  int64_t v_strideB = int64_t(Sk) * v_strideM;

  typename Attention::Params params;
  params.query_ptr = const_cast<scalar_t*>(tensors.query);
  params.key_ptr = const_cast<scalar_t*>(tensors.key);
  params.value_ptr = const_cast<scalar_t*>(tensors.value);
  params.output_ptr = tensors.output;
  params.output_accum_ptr = output_accum_ptr;
  params.logsumexp_ptr = nullptr;
  params.scale = problem.scale;
  params.num_queries = Sq;
  params.num_keys = Sk;
  params.head_dim = d;
  params.head_dim_value = dv;
  params.num_heads = H;
  params.num_batches = B;
  params.q_strideM = q_strideM;
  params.k_strideM = k_strideM;
  params.v_strideM = v_strideM;
  params.o_strideM = o_strideM;
  params.q_strideH = q_strideH;
  params.k_strideH = k_strideH;
  params.v_strideH = v_strideH;
  params.q_strideB = q_strideB;
  params.k_strideB = k_strideB;
  params.v_strideB = v_strideB;
  return params;
}

template <typename Attention>
cudaError_t launch_attention(
    Problem const& problem,
    Tensors const& tensors,
    Workspace workspace,
    cudaStream_t stream) {
  using OutputAccum = typename Attention::output_accum_t;

  OutputAccum* output_accum_ptr = nullptr;
  std::size_t required_workspace = workspace_bytes_for<Attention>(problem);
  if constexpr (Attention::kNeedsOutputAccumulatorBuffer) {
    if (!workspace.data || workspace.bytes < required_workspace) {
      return cudaErrorInvalidValue;
    }
    output_accum_ptr = reinterpret_cast<OutputAccum*>(workspace.data);
  }

  typename Attention::Params params =
      make_params<Attention>(problem, tensors, output_accum_ptr);

  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = int(sizeof(typename Attention::SharedStorage));
  if (smem_bytes > 0xc000) {
    cudaError_t err = cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) {
      return err;
    }
  }
  if (!Attention::check_supported(params)) {
    return cudaErrorInvalidValue;
  }

  kernel_fn<<<params.getBlocksGrid(), params.getThreadsGrid(), smem_bytes, stream>>>(params);
  return cudaGetLastError();
}

template <typename Config>
std::size_t workspace_bytes_config(Problem const& problem) {
  return workspace_bytes_for<typename Config::Attention>(problem);
}

template <typename Config>
cudaError_t launch_config(
    Problem const& problem,
    Tensors const& tensors,
    Workspace workspace,
    cudaStream_t stream) {
  return launch_attention<typename Config::Attention>(
      problem, tensors, workspace, stream);
}

std::size_t workspace_bytes(Problem const& problem) {
  int max_head_size = std::max(problem.head_size, problem.head_size_v);

  if (max_head_size > 64) {
    if (max_head_size <= 128) {
      using Config = TiledOnlineAttentionConfig<cutlass::arch::Sm80, 32, 128, 128>;
      return workspace_bytes_config<Config>(problem);
    }
    if (max_head_size <= 256) {
      using Config = TiledOnlineAttentionConfig<cutlass::arch::Sm80, 64, 64, 65536>;
      return workspace_bytes_config<Config>(problem);
    }
    using Config = TiledOnlineAttentionConfig<cutlass::arch::Sm80, 32, 128, 65536>;
    return workspace_bytes_config<Config>(problem);
  }

  using Config = TiledOnlineAttentionConfig<cutlass::arch::Sm80, 64, 64, 64>;
  return workspace_bytes_config<Config>(problem);
}

bool can_run(Problem const& problem, std::string& reason) {
  if (problem.head_size <= 0 || problem.head_size_v <= 0 ||
      problem.seq_length <= 0 || problem.seq_length_kv <= 0 ||
      problem.head_number <= 0 || problem.batch_size <= 0) {
    reason = "all problem dimensions must be positive";
    return false;
  }
  if (std::max(problem.head_size, problem.head_size_v) > 65536) {
    reason = "head dimensions exceed the largest 02 tile policy";
    return false;
  }
  return true;
}

cudaError_t run(
    Problem const& problem,
    Tensors const& tensors,
    Workspace workspace,
    cudaStream_t stream) {
  int max_head_size = std::max(problem.head_size, problem.head_size_v);

  if (max_head_size > 64) {
    if (max_head_size <= 128) {
      using Config = TiledOnlineAttentionConfig<cutlass::arch::Sm80, 32, 128, 128>;
      return launch_config<Config>(problem, tensors, workspace, stream);
    }
    if (max_head_size <= 256) {
      using Config = TiledOnlineAttentionConfig<cutlass::arch::Sm80, 64, 64, 65536>;
      return launch_config<Config>(problem, tensors, workspace, stream);
    }
    using Config = TiledOnlineAttentionConfig<cutlass::arch::Sm80, 32, 128, 65536>;
    return launch_config<Config>(problem, tensors, workspace, stream);
  }

  using Config = TiledOnlineAttentionConfig<cutlass::arch::Sm80, 64, 64, 64>;
  return launch_config<Config>(problem, tensors, workspace, stream);
}

Kernel const kKernel = {
    "02-tiled-online",
    "Tiled Online Attention (fused MMA)",
    workspace_bytes,
    can_run,
    run,
};

} // namespace

Kernel const& kernel_02_tiled_online() {
  return kKernel;
}
