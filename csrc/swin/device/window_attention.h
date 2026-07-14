/*
  Device-level fused Swin window attention facade.
*/

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "cutlass/arch/mma.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

#include "device/launch.h"
#include "kernel/window_attention.h"
#include "swin_problem.h"
#include "threadblock/attention_glue.h"

namespace tiny_cutlass {
namespace swin {
namespace device {
namespace detail {

template <typename KernelConfig>
cutlass::Status launch_projection(
    int rows,
    int k,
    int n,
    typename KernelConfig::Element const* input,
    typename KernelConfig::Element const* weight,
    typename KernelConfig::Element* output,
    cudaStream_t stream) {
  using Element = typename KernelConfig::Element;
  using Gemm = cutlass::gemm::device::Gemm<
      Element,
      typename KernelConfig::MatrixLayout,
      Element,
      typename KernelConfig::MatrixLayout,
      Element,
      typename KernelConfig::MatrixLayout,
      typename KernelConfig::ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      typename KernelConfig::ArchTag,
      typename KernelConfig::ThreadblockShape,
      typename KernelConfig::WarpShape,
      typename KernelConfig::InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          Element,
          128 / cutlass::sizeof_bits<Element>::value,
          typename KernelConfig::ElementAccumulator,
          typename KernelConfig::ElementCompute>>;

  typename Gemm::Arguments args(
      {rows, n, k},
      {input, k},
      {weight, n},
      {output, n},
      {output, n},
      {typename KernelConfig::ElementCompute(1), typename KernelConfig::ElementCompute(0)});

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  Gemm op;
  return op(args, nullptr, stream);
}

} // namespace detail

template <typename KernelConfig_>
class WindowAttentionCore {
 public:
  using KernelConfig = KernelConfig_;
  using Element = typename KernelConfig::Element;
  using Core = kernel::WindowAttentionCore<KernelConfig>;
  using Tensors = WindowAttentionTensors<Element>;

  static cudaError_t run(
      SwinAttentionProblem const& problem,
      Tensors const& tensors,
      cudaStream_t stream) {
    typename Core::Params params;
    initialize_params(problem, tensors, params);

    constexpr auto kernel_fn = kernel::window_attention_core_kernel<Core>;
    int smem_bytes = int(sizeof(typename Core::SharedStorage));
    if (smem_bytes > 0xc000) {
      cudaError_t err = cudaFuncSetAttribute(
          kernel_fn,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_bytes);
      if (err != cudaSuccess) {
        return err;
      }
    }

    if (!Core::check_supported(params)) {
      return cudaErrorInvalidValue;
    }

    kernel_fn<<<params.getBlocksGrid(), params.getThreadsGrid(), smem_bytes, stream>>>(
        params);
    return cudaGetLastError();
  }

 private:
  static void initialize_params(
      SwinAttentionProblem const& problem,
      Tensors const& tensors,
      typename Core::Params& params) {
    int bw = swin_batched_windows(problem);
    int l = swin_window_len(problem);
    int lp = swin_window_len_padded(problem);
    int c = swin_channels(problem);

    params.query_ptr = const_cast<Element*>(tensors.query);
    params.key_ptr = const_cast<Element*>(tensors.key);
    params.value_ptr = const_cast<Element*>(tensors.value);
    params.attn_bias_ptr = const_cast<Element*>(tensors.attention_bias);
    params.output_ptr = tensors.attention_output;
    params.output_accum_ptr = nullptr;
    params.logsumexp_ptr = nullptr;
    params.scale = problem.scale;
    params.num_queries = l;
    params.num_keys = l;
    params.head_dim = problem.head_size;
    params.head_dim_value = problem.head_size;
    params.num_heads = problem.head_number;
    params.num_batches = bw;
    params.q_strideM = c;
    params.k_strideM = c;
    params.v_strideM = c;
    params.bias_strideM = lp;
    params.o_strideM = c;
    params.q_strideH = problem.head_size;
    params.k_strideH = problem.head_size;
    params.v_strideH = problem.head_size;
    params.bias_strideH = int64_t(l) * lp;
    params.q_strideB = int64_t(l) * c;
    params.k_strideB = int64_t(l) * c;
    params.v_strideB = int64_t(l) * c;
    params.bias_strideB = int64_t(problem.head_number) * l * lp;
    params.custom_mask_type = Core::kNoCustomMask;
  }
};

template <typename KernelConfig_>
class WindowAttention {
 public:
  using KernelConfig = KernelConfig_;
  using Element = typename KernelConfig::Element;
  using Tensors = WindowAttentionTensors<Element>;

  static cudaError_t run(
      SwinAttentionProblem const& problem,
      Tensors const& tensors,
      cudaStream_t stream) {
    cutlass::Status status = detail::launch_projection<KernelConfig>(
        swin_rows(problem),
        swin_channels(problem),
        3 * swin_channels(problem),
        tensors.windows,
        tensors.qkv_weight,
        tensors.qkv,
        stream);
    if (status != cutlass::Status::kSuccess) {
      return cudaErrorInvalidValue;
    }

    {
      using Threadblock = threadblock::AddQkvBiasSplit<Element, KernelConfig::kThreads>;
      typename Threadblock::Params params;
      params.qkv = tensors.qkv;
      params.bias = tensors.qkv_bias;
      params.q = tensors.query;
      params.k = tensors.key;
      params.v = tensors.value;
      params.elements = swin_rows(problem) * swin_channels(problem);
      params.channels = swin_channels(problem);
      cudaError_t err = detail::launch_threadblock<Threadblock>(params, stream);
      if (err != cudaSuccess) {
        return err;
      }
    }

    cudaError_t err = WindowAttentionCore<KernelConfig>::run(problem, tensors, stream);
    if (err != cudaSuccess) {
      return err;
    }

    status = detail::launch_projection<KernelConfig>(
        swin_rows(problem),
        swin_channels(problem),
        swin_channels(problem),
        tensors.attention_output,
        tensors.output_weight,
        tensors.projected,
        stream);
    if (status != cutlass::Status::kSuccess) {
      return cudaErrorInvalidValue;
    }

    {
      using Threadblock = threadblock::AddBias<Element, KernelConfig::kThreads>;
      typename Threadblock::Params params;
      params.output = tensors.projected;
      params.bias = tensors.output_bias;
      params.elements = swin_rows(problem) * swin_channels(problem);
      params.channels = swin_channels(problem);
      err = detail::launch_threadblock<Threadblock>(params, stream);
      if (err != cudaSuccess) {
        return err;
      }
    }

    return cudaSuccess;
  }
};

} // namespace device
} // namespace swin
} // namespace tiny_cutlass
