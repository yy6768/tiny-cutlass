#include "swin.h"

#include <type_traits>

#include <cuda_runtime.h>

#include "cutlass/arch/mma.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

#include "../flash-attention/02-tiled-online-attention/kernel_forward.h"
#include "kernel/swin.h"
#include "threadblock/layout.h"

namespace tiny_cutlass {
namespace swin {
namespace {

template <typename Threadblock>
cudaError_t launch_threadblock(
    typename Threadblock::Params const& params,
    cudaStream_t stream,
    int shared_storage_bytes = 0) {
  kernel::swin_threadblock_kernel<Threadblock>
      <<<params.getBlocksGrid(),
         params.getThreadsGrid(),
         shared_storage_bytes,
         stream>>>(params);
  return cudaGetLastError();
}

template <typename Kernel>
cutlass::Status launch_projection(
    int rows,
    int k,
    int n,
    typename Kernel::Element const* input,
    typename Kernel::Element const* weight,
    typename Kernel::Element* output,
    cudaStream_t stream) {
  using Policy = typename Kernel::Policy;
  using Element = typename Policy::Element;
  using Gemm = cutlass::gemm::device::Gemm<
      Element,
      typename Policy::MatrixLayout,
      Element,
      typename Policy::MatrixLayout,
      Element,
      typename Policy::MatrixLayout,
      typename Policy::ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      typename Policy::ArchTag,
      typename Policy::ThreadblockShape,
      typename Policy::WarpShape,
      typename Policy::InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          Element,
          128 / cutlass::sizeof_bits<Element>::value,
          typename Policy::ElementAccumulator,
          typename Policy::ElementCompute>>;

  typename Gemm::Arguments args(
      {rows, n, k},
      {input, k},
      {weight, n},
      {output, n},
      {output, n},
      {typename Policy::ElementCompute(1), typename Policy::ElementCompute(0)});

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  Gemm op;
  return op(args, nullptr, stream);
}

template <typename Kernel>
cudaError_t launch_attention(
    SwinProblem const& problem,
    SwinTensors<typename Kernel::Element> const& tensors,
    cudaStream_t stream) {
  using Policy = typename Kernel::Policy;
  using Element = typename Policy::Element;
  using Attention = AttentionKernel<
      Element,
      typename Policy::ArchTag,
      Policy::kAttentionIsAligned,
      Policy::kAttentionQueriesPerBlock,
      Policy::kAttentionKeysPerBlock,
      Policy::kAttentionMaxHeadDim,
      false,
      Policy::kAttentionSupportsBias>;

  int bw = swin_batched_windows(problem);
  int l = swin_window_len(problem);
  int lp = swin_window_len_padded(problem);
  int c = swin_channels(problem);

  typename Attention::Params params;
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
  params.custom_mask_type = Attention::NoCustomMask;

  constexpr auto kernel_fn = kernel::swin_attention_kernel<Attention>;
  int smem_bytes = int(sizeof(typename Attention::SharedStorage));
  if (smem_bytes > 0xc000) {
    cudaError_t err = cudaFuncSetAttribute(
        kernel_fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes);
    if (err != cudaSuccess) {
      return err;
    }
  }

  if (!Attention::check_supported(params)) {
    return cudaErrorInvalidValue;
  }

  kernel_fn<<<params.getBlocksGrid(), params.getThreadsGrid(), smem_bytes, stream>>>(
      params);
  return cudaGetLastError();
}

template <typename Kernel>
bool supported_type() {
  using Policy = typename Kernel::Policy;
  return std::is_same<typename Policy::ArchTag, cutlass::arch::Sm80>::value &&
      std::is_same<typename Policy::Element, cutlass::half_t>::value;
}

template <typename Kernel>
char const* validate_problem(SwinProblem const& problem) {
  if (!supported_type<Kernel>()) {
    return "only the explicitly instantiated TensorOp policy is available";
  }
  if (problem.batch_size <= 0 || problem.image_size <= 0 ||
      problem.window_size <= 0 || problem.head_number <= 0 ||
      problem.head_size <= 0) {
    return "all dimensions must be positive";
  }
  if ((problem.image_size % problem.window_size) != 0) {
    return "image_size must be divisible by window_size";
  }
  if (problem.shift_size < 0 || problem.shift_size >= problem.window_size) {
    return "shift_size must satisfy 0 <= shift_size < window_size";
  }
  if ((swin_channels(problem) % 8) != 0 || (problem.head_size % 8) != 0) {
    return "TensorOp path requires channels and head_size to be multiples of 8";
  }
  if (swin_window_len(problem) > 64) {
    return "WindowAttention path currently supports window_len <= 64";
  }
  if (problem.scale <= 0.0f) {
    return "attention scale must be positive";
  }
  return nullptr;
}

} // namespace

namespace device {

template <typename Kernel>
bool Swin<Kernel>::can_implement(
    SwinProblem const& problem,
    char const** reason) {
  char const* local_reason = validate_problem<Kernel>(problem);
  if (reason != nullptr) {
    *reason = local_reason;
  }
  return local_reason == nullptr;
}

template <typename Kernel>
cudaError_t Swin<Kernel>::run(
    SwinProblem const& problem,
    Tensors const& tensors,
    cudaStream_t stream) {
  char const* reason = nullptr;
  if (!can_implement(problem, &reason)) {
    (void)reason;
    return cudaErrorInvalidValue;
  }
  if (tensors.input == nullptr || tensors.qkv_weight == nullptr ||
      tensors.output_weight == nullptr || tensors.attention_bias == nullptr ||
      tensors.windows == nullptr || tensors.qkv == nullptr ||
      tensors.query == nullptr || tensors.key == nullptr ||
      tensors.value == nullptr || tensors.attention_output == nullptr ||
      tensors.projected == nullptr || tensors.output == nullptr) {
    return cudaErrorInvalidValue;
  }

  using Element = typename Kernel::Element;

  {
    using Threadblock = threadblock::WindowPartition<Element, 256>;
    typename Threadblock::Params params;
    params.input = tensors.input;
    params.output = tensors.windows;
    params.batch = problem.batch_size;
    params.height = problem.image_size;
    params.width = problem.image_size;
    params.channels = swin_channels(problem);
    params.shift_size = problem.shift_size;
    params.window_size = problem.window_size;
    cudaError_t err = launch_threadblock<Threadblock>(params, stream);
    if (err != cudaSuccess) {
      return err;
    }
  }

  cutlass::Status status = launch_projection<Kernel>(
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
    using Threadblock = threadblock::AddQkvBiasSplit<Element, 256>;
    typename Threadblock::Params params;
    params.qkv = tensors.qkv;
    params.bias = tensors.qkv_bias;
    params.q = tensors.query;
    params.k = tensors.key;
    params.v = tensors.value;
    params.elements = swin_rows(problem) * swin_channels(problem);
    params.channels = swin_channels(problem);
    cudaError_t err = launch_threadblock<Threadblock>(params, stream);
    if (err != cudaSuccess) {
      return err;
    }
  }

  cudaError_t err = launch_attention<Kernel>(problem, tensors, stream);
  if (err != cudaSuccess) {
    return err;
  }

  status = launch_projection<Kernel>(
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
    using Threadblock = threadblock::AddBias<Element, 256>;
    typename Threadblock::Params params;
    params.output = tensors.projected;
    params.bias = tensors.output_bias;
    params.elements = swin_rows(problem) * swin_channels(problem);
    params.channels = swin_channels(problem);
    err = launch_threadblock<Threadblock>(params, stream);
    if (err != cudaSuccess) {
      return err;
    }
  }

  {
    using Threadblock = threadblock::WindowReverse<Element, 256>;
    typename Threadblock::Params params;
    params.input = tensors.projected;
    params.output = tensors.output;
    params.batch = problem.batch_size;
    params.height = problem.image_size;
    params.width = problem.image_size;
    params.channels = swin_channels(problem);
    params.shift_size = problem.shift_size;
    params.window_size = problem.window_size;
    err = launch_threadblock<Threadblock>(params, stream);
    if (err != cudaSuccess) {
      return err;
    }
  }

  if (tensors.patch_merged != nullptr) {
    if ((problem.image_size % 2) != 0) {
      return cudaErrorInvalidValue;
    }
    using Threadblock = threadblock::PatchMerge<Element, 256>;
    typename Threadblock::Params params;
    params.input = tensors.input;
    params.output = tensors.patch_merged;
    params.batch = problem.batch_size;
    params.height = problem.image_size;
    params.width = problem.image_size;
    params.channels = swin_channels(problem);
    err = launch_threadblock<Threadblock>(params, stream);
    if (err != cudaSuccess) {
      return err;
    }
  }

  return cudaSuccess;
}

template class Swin<typename kernel::DefaultSwin<
    cutlass::arch::Sm80,
    cutlass::half_t>::Kernel>;

} // namespace device

} // namespace swin
} // namespace tiny_cutlass
