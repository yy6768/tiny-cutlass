#include "swin.h"

#include <type_traits>

#include <cuda_runtime.h>

#include "cutlass/arch/mma.h"

#include "device/swin_block.h"
#include "device/window_attention.h"
#include "device/window_partition.h"
#include "kernel/default_swin_attention.h"
#include "kernel/default_swin_block.h"
#include "kernel/swin.h"
#include "threadblock/attention_glue.h"
#include "threadblock/normalization.h"

namespace tiny_cutlass {
namespace swin {
namespace {

template <typename ArchTag, typename Element>
bool supported_type() {
  return std::is_same<ArchTag, cutlass::arch::Sm80>::value &&
      std::is_same<Element, cutlass::half_t>::value;
}

template <typename ArchTag, typename Element>
cutlass::Status validate_problem(SwinAttentionProblem const& problem) {
  if (!supported_type<ArchTag, Element>()) {
    return cutlass::Status::kErrorNotSupported;
  }
  if (problem.batch_size <= 0 || problem.image_size <= 0 ||
      problem.window_size <= 0 || problem.head_number <= 0 ||
      problem.head_size <= 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }
  if ((problem.image_size % problem.window_size) != 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }
  if (problem.shift_size < 0 || problem.shift_size >= problem.window_size) {
    return cutlass::Status::kErrorInvalidProblem;
  }
  if ((swin_channels(problem) % 8) != 0 || (problem.head_size % 8) != 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }
  if (swin_window_len(problem) > 64) {
    return cutlass::Status::kErrorInvalidProblem;
  }
  if (problem.scale <= 0.0f) {
    return cutlass::Status::kErrorInvalidProblem;
  }
  return cutlass::Status::kSuccess;
}

cutlass::Status from_cuda(cudaError_t status) {
  return status == cudaSuccess
      ? cutlass::Status::kSuccess
      : cutlass::Status::kErrorInternal;
}

} // namespace

namespace device {

template <typename ArchTag, typename Element>
cutlass::Status SwinAttention<ArchTag, Element>::can_implement(
    SwinAttentionProblem const& problem) {
  return validate_problem<ArchTag, Element>(problem);
}

template <typename ArchTag, typename Element>
cutlass::Status SwinAttention<ArchTag, Element>::run(
    SwinAttentionProblem const& problem,
    Tensors const& tensors,
    cudaStream_t stream) {
  using KernelConfig = kernel::DefaultSwinAttention<ArchTag, Element>;
  cutlass::Status status = can_implement(problem);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }
  auto const& attention = tensors.attention;
  if (tensors.input == nullptr || attention.qkv_weight == nullptr ||
      attention.output_weight == nullptr || attention.attention_bias == nullptr ||
      attention.windows == nullptr || attention.qkv == nullptr ||
      attention.query == nullptr || attention.key == nullptr ||
      attention.value == nullptr || attention.attention_output == nullptr ||
      attention.projected == nullptr || tensors.output == nullptr) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  cudaError_t err = detail::launch_window_partition<Element, KernelConfig::kThreads>(
      problem, tensors, stream);
  if (err != cudaSuccess) {
    return from_cuda(err);
  }

  err = WindowAttention<KernelConfig>::run(problem, attention, stream);
  if (err != cudaSuccess) {
    return from_cuda(err);
  }

  err = detail::launch_window_reverse<Element, KernelConfig::kThreads>(
      problem, tensors, stream);
  if (err != cudaSuccess) {
    return from_cuda(err);
  }

  return cutlass::Status::kSuccess;
}

template class SwinAttention<cutlass::arch::Sm80, cutlass::half_t>;

// ---------------------------------------------------------------------------
// Full fused SwinBlock (norm1 + attention + residual1 + norm2 + MLP + residual2)
// ---------------------------------------------------------------------------

template <typename ArchTag, typename Element>
cutlass::Status SwinBlock<ArchTag, Element>::can_implement(
    SwinBlockProblem const& problem) {
  cutlass::Status status = validate_problem<ArchTag, Element>(problem);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }
  if (problem.mlp_ratio <= 0 || problem.layernorm_eps <= 0.0f) {
    return cutlass::Status::kErrorInvalidProblem;
  }
  return cutlass::Status::kSuccess;
}

template <typename ArchTag, typename Element>
cutlass::Status SwinBlock<ArchTag, Element>::run(
    SwinBlockProblem const& problem,
    Tensors const& tensors,
    cudaStream_t stream) {
  using KernelConfig = kernel::DefaultSwinBlock<ArchTag, Element>;
  cutlass::Status status = can_implement(problem);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }
  auto const& attention = tensors.attention;
  if (tensors.input == nullptr || attention.qkv_weight == nullptr ||
      attention.output_weight == nullptr || attention.attention_bias == nullptr ||
      attention.windows == nullptr || attention.qkv == nullptr ||
      attention.query == nullptr || attention.key == nullptr ||
      attention.value == nullptr || attention.attention_output == nullptr ||
      attention.projected == nullptr || tensors.output == nullptr ||
      tensors.gamma1 == nullptr || tensors.gamma2 == nullptr ||
      tensors.fc1_weight == nullptr || tensors.fc1_bias == nullptr ||
      tensors.fc2_weight == nullptr || tensors.fc2_bias == nullptr ||
      tensors.residual == nullptr || tensors.normed2 == nullptr ||
      tensors.mlp_hidden == nullptr || tensors.mlp_output == nullptr) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  int c = swin_channels(problem);
  int image_tokens =
      problem.batch_size * problem.image_size * problem.image_size;
  int mlp_hidden = swin_mlp_hidden(problem);
  int smem_bytes = int(2 * KernelConfig::kThreads * sizeof(float));

  // --- Launch 1: LayerNormShiftPartition (norm1 + shift + partition) ---
  {
    using Threadblock =
        threadblock::LayerNormShiftPartition<Element, KernelConfig::kThreads>;
    typename Threadblock::Params params;
    params.input = tensors.input;
    params.gamma = tensors.gamma1;
    params.beta = tensors.beta1;
    params.output = attention.windows;
    params.batch = problem.batch_size;
    params.height = problem.image_size;
    params.width = problem.image_size;
    params.channels = c;
    params.shift_size = problem.shift_size;
    params.window_size = problem.window_size;
    params.epsilon = problem.layernorm_eps;

    kernel::swin_threadblock_kernel<Threadblock>
        <<<params.getBlocksGrid(), params.getThreadsGrid(), smem_bytes, stream>>>(
            params);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return from_cuda(err);
    }
  }

  // --- Launches 2-5: WindowAttention (QKV GEMM, bias/split, attn, proj) ---
  {
    cudaError_t err = WindowAttention<typename KernelConfig::AttentionConfig>::run(
        problem, attention, stream);
    if (err != cudaSuccess) {
      return from_cuda(err);
    }
  }

  // --- Launch 6: ReverseAddResidualLayerNorm (reverse + res1 + norm2) ---
  {
    using Threadblock =
        threadblock::ReverseAddResidualLayerNorm<Element, KernelConfig::kThreads>;
    typename Threadblock::Params params;
    params.projected = attention.projected;
    params.shortcut = tensors.input;
    params.gamma = tensors.gamma2;
    params.beta = tensors.beta2;
    params.residual = tensors.residual;
    params.normed = tensors.normed2;
    params.batch = problem.batch_size;
    params.height = problem.image_size;
    params.width = problem.image_size;
    params.channels = c;
    params.shift_size = problem.shift_size;
    params.window_size = problem.window_size;
    params.epsilon = problem.layernorm_eps;

    kernel::swin_threadblock_kernel<Threadblock>
        <<<params.getBlocksGrid(), params.getThreadsGrid(), smem_bytes, stream>>>(
            params);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return from_cuda(err);
    }
  }

  // --- Launch 7: fc1 GEMM (C -> mlp_hidden) ---
  {
    status = detail::launch_projection<KernelConfig>(
        image_tokens,
        c,
        mlp_hidden,
        tensors.normed2,
        tensors.fc1_weight,
        tensors.mlp_hidden,
        stream);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
  }

  // --- Launch 8: AddBiasGelu ---
  {
    using Threadblock = threadblock::AddBiasGelu<Element, KernelConfig::kThreads>;
    typename Threadblock::Params params;
    params.output = tensors.mlp_hidden;
    params.bias = tensors.fc1_bias;
    params.elements = image_tokens * mlp_hidden;
    params.channels = mlp_hidden;
    cudaError_t err = detail::launch_threadblock<Threadblock>(params, stream);
    if (err != cudaSuccess) {
      return from_cuda(err);
    }
  }

  // --- Launch 9: fc2 GEMM (mlp_hidden -> C) ---
  {
    status = detail::launch_projection<KernelConfig>(
        image_tokens,
        mlp_hidden,
        c,
        tensors.mlp_hidden,
        tensors.fc2_weight,
        tensors.mlp_output,
        stream);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
  }

  // --- Launch 10: AddBiasResidual (fc2 bias + residual 2) ---
  {
    using Threadblock = threadblock::AddBiasResidual<Element, KernelConfig::kThreads>;
    typename Threadblock::Params params;
    params.input = tensors.mlp_output;
    params.bias = tensors.fc2_bias;
    params.residual = tensors.residual;
    params.output = tensors.output;
    params.elements = image_tokens * c;
    params.channels = c;
    cudaError_t err = detail::launch_threadblock<Threadblock>(params, stream);
    if (err != cudaSuccess) {
      return from_cuda(err);
    }
  }

  return cutlass::Status::kSuccess;
}

template class SwinBlock<cutlass::arch::Sm80, cutlass::half_t>;

} // namespace device

} // namespace swin
} // namespace tiny_cutlass
