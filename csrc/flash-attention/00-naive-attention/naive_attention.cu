/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief Kernel entry for 00-naive-attention.

    The shared test harness owns allocation, cuDNN reference verification, and
    timing. This file only launches the 00 three-stage device path:

      MM0 -> safe softmax -> MM1
*/

#include <algorithm>
#include <cstdint>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "../flash_attention.h"
#include "kernel_forward.h"

namespace {

using scalar_t = Element;
using accum_t = float;

template <typename ArchTag_, int kQueriesPerBlock_, int kKeysPerBlock_>
struct NaiveAttentionConfig {
  using ArchTag = ArchTag_;

  static constexpr bool kIsAligned = true;
  static constexpr int kQueriesPerBlock = kQueriesPerBlock_;
  static constexpr int kKeysPerBlock = kKeysPerBlock_;

  using MM0 = AttentionMMKernel<
      scalar_t,
      ArchTag,
      kIsAligned,
      kQueriesPerBlock,
      kKeysPerBlock,
      cutlass::layout::RowMajor,
      cutlass::layout::ColumnMajor,
      cutlass::layout::RowMajor>;

  using Softmax = AttentionSoftmaxKernel<scalar_t, 256>;

  using MM1 = AttentionMMKernel<
      scalar_t,
      ArchTag,
      kIsAligned,
      kQueriesPerBlock,
      kKeysPerBlock,
      cutlass::layout::RowMajor,
      cutlass::layout::RowMajor,
      cutlass::layout::RowMajor>;
};

template <typename Config>
cudaError_t launch_config(
    Problem const& problem,
    Tensors const& tensors,
    Workspace workspace,
    cudaStream_t stream) {
  using MM0 = typename Config::MM0;
  using Softmax = typename Config::Softmax;
  using MM1 = typename Config::MM1;

  int B = problem.batch_size;
  int Sq = problem.seq_length;
  int Sk = problem.seq_length_kv;
  int H = problem.head_number;
  int d = problem.head_size;
  int dv = problem.head_size_v;

  std::size_t required_workspace = total_probability_elements(problem) * sizeof(scalar_t);
  if (!workspace.data || workspace.bytes < required_workspace) {
    return cudaErrorInvalidValue;
  }

  auto* block_p = reinterpret_cast<scalar_t*>(workspace.data);

  cudaError_t err = cudaMemsetAsync(block_p, 0, required_workspace, stream);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaMemsetAsync(tensors.output, 0, total_output_elements(problem) * sizeof(scalar_t), stream);
  if (err != cudaSuccess) {
    return err;
  }

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
  int64_t o_strideB = int64_t(Sq) * o_strideM;

  int32_t p_strideM = Sk;
  int64_t p_strideH = int64_t(Sq) * Sk;
  int64_t p_strideB = int64_t(H) * p_strideH;

  {
    typename MM0::Params p;
    p.a_ptr = const_cast<scalar_t*>(tensors.query);
    p.b_ptr = const_cast<scalar_t*>(tensors.key);
    p.c_ptr = block_p;
    p.alpha = problem.scale;
    p.problem_m = Sq;
    p.problem_n = Sk;
    p.problem_k = d;
    p.num_heads = H;
    p.num_batches = B;
    p.a_strideM = q_strideM; p.b_strideM = k_strideM; p.c_strideM = p_strideM;
    p.a_strideH = q_strideH; p.b_strideH = k_strideH; p.c_strideH = p_strideH;
    p.a_strideB = q_strideB; p.b_strideB = k_strideB; p.c_strideB = p_strideB;

    auto kernel_fn = attention_mm0_kernel<MM0>;
    int smem_bytes = int(sizeof(typename MM0::SharedStorage));
    if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);
  }

  {
    typename Softmax::Params p;
    p.p_ptr = block_p;
    p.num_queries = Sq;
    p.num_keys = Sk;
    p.num_heads = H;
    p.num_batches = B;
    p.p_strideM = p_strideM;
    p.p_strideH = p_strideH;
    p.p_strideB = p_strideB;

    auto kernel_fn = attention_softmax_kernel<Softmax>;
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), p.getSmemBytes(), stream>>>(p);
  }

  {
    typename MM1::Params p;
    p.a_ptr = block_p;
    p.b_ptr = const_cast<scalar_t*>(tensors.value);
    p.c_ptr = tensors.output;
    p.alpha = accum_t(1);
    p.problem_m = Sq;
    p.problem_n = dv;
    p.problem_k = Sk;
    p.num_heads = H;
    p.num_batches = B;
    p.a_strideM = p_strideM; p.b_strideM = v_strideM; p.c_strideM = o_strideM;
    p.a_strideH = p_strideH; p.b_strideH = v_strideH; p.c_strideH = dv;
    p.a_strideB = p_strideB; p.b_strideB = v_strideB; p.c_strideB = o_strideB;

    auto kernel_fn = attention_mm1_kernel<MM1>;
    int smem_bytes = int(sizeof(typename MM1::SharedStorage));
    if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);
  }

  return cudaGetLastError();
}

std::size_t workspace_bytes(Problem const& problem) {
  return total_probability_elements(problem) * sizeof(scalar_t);
}

bool can_run(Problem const& problem, std::string& reason) {
  if (problem.head_size <= 0 || problem.head_size_v <= 0 ||
      problem.seq_length <= 0 || problem.seq_length_kv <= 0 ||
      problem.head_number <= 0 || problem.batch_size <= 0) {
    reason = "all problem dimensions must be positive";
    return false;
  }
  return true;
}

cudaError_t run(
    Problem const& problem,
    Tensors const& tensors,
    Workspace workspace,
    cudaStream_t stream) {
  if (problem.head_size_v > 64) {
    using Config = NaiveAttentionConfig<cutlass::arch::Sm80, 32, 128>;
    return launch_config<Config>(problem, tensors, workspace, stream);
  }

  using Config = NaiveAttentionConfig<cutlass::arch::Sm80, 64, 64>;
  return launch_config<Config>(problem, tensors, workspace, stream);
}

Kernel const kKernel = {
    "00-naive",
    "Naive Attention (3-stage, safe softmax)",
    workspace_bytes,
    can_run,
    run,
};

} // namespace

Kernel const& kernel_00_naive() {
  return kKernel;
}
