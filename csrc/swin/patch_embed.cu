#include "device/patch_embed.h"

#include <type_traits>

#include <cuda_runtime.h>

#include "cutlass/arch/mma.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_ref.h"

#include "kernel/default_patch_embed.h"
#include "kernel/swin.h"
#include "threadblock/layout.h"

namespace tiny_cutlass {
namespace swin {
namespace device {
namespace {

template <typename Element>
cudaError_t launch_pad_activation(
    PatchEmbedProblem const& problem,
    Element const* input,
    Element* output,
    cudaStream_t stream) {
  using Threadblock = threadblock::ImagePadChannels<Element, 256>;
  typename Threadblock::Params params;
  params.input = input;
  params.output = output;
  params.elements = patch_embed_input_padded_elements(problem);
  params.channels = problem.in_channels;
  params.channels_padded = problem.input_channels_padded;
  params.height = problem.image_size;
  params.width = problem.image_size;

  kernel::swin_threadblock_kernel<Threadblock>
      <<<params.getBlocksGrid(), params.getThreadsGrid(), 0, stream>>>(params);
  return cudaGetLastError();
}

template <typename Element>
cudaError_t launch_pad_filter(
    PatchEmbedProblem const& problem,
    Element const* input,
    Element* output,
    cudaStream_t stream) {
  using Threadblock = threadblock::FilterOihwToKrscPadded<Element, 256>;
  typename Threadblock::Params params;
  params.input = input;
  params.output = output;
  params.elements = patch_embed_kernel_padded_elements(problem);
  params.output_channels = problem.embed_dim;
  params.input_channels = problem.in_channels;
  params.input_channels_padded = problem.input_channels_padded;
  params.filter_h = problem.patch_size;
  params.filter_w = problem.patch_size;

  kernel::swin_threadblock_kernel<Threadblock>
      <<<params.getBlocksGrid(), params.getThreadsGrid(), 0, stream>>>(params);
  return cudaGetLastError();
}

template <typename Element>
cudaError_t launch_layernorm(
    PatchEmbedProblem const& problem,
    Element const* input,
    Element const* bias,
    Element const* gamma,
    Element const* beta,
    Element* output,
    cudaStream_t stream) {
  using Threadblock = threadblock::AddBiasLayerNorm<Element, 256>;
  typename Threadblock::Params params;
  params.input = input;
  params.bias = bias;
  params.gamma = gamma;
  params.beta = beta;
  params.output = output;
  params.tokens = problem.batch_size * patch_embed_output_size(problem) *
      patch_embed_output_size(problem);
  params.channels = problem.embed_dim;
  params.epsilon = problem.layernorm_eps;

  int smem_bytes = int(2 * Threadblock::kThreads * sizeof(float));
  kernel::swin_threadblock_kernel<Threadblock>
      <<<params.getBlocksGrid(), params.getThreadsGrid(), smem_bytes, stream>>>(
          params);
  return cudaGetLastError();
}

cutlass::conv::Conv2dProblemSize make_conv_problem(
    PatchEmbedProblem const& problem) {
  int out = patch_embed_output_size(problem);
  return cutlass::conv::Conv2dProblemSize(
      cutlass::Tensor4DCoord(
          problem.batch_size,
          problem.image_size,
          problem.image_size,
          problem.input_channels_padded),
      cutlass::Tensor4DCoord(
          problem.embed_dim,
          problem.patch_size,
          problem.patch_size,
          problem.input_channels_padded),
      cutlass::Tensor4DCoord(0, 0, 0, 0),
      cutlass::MatrixCoord(problem.patch_size, problem.patch_size),
      cutlass::MatrixCoord(1, 1),
      cutlass::Tensor4DCoord(problem.batch_size, out, out, problem.embed_dim),
      cutlass::conv::Mode::kCrossCorrelation,
      1,
      1);
}

template <typename Kernel>
cutlass::Status run_conv(
    PatchEmbedProblem const& problem,
    typename Kernel::Element const* activation,
    typename Kernel::Element const* filter,
    typename Kernel::Element* output,
    cudaStream_t stream) {
  using Element = typename Kernel::Element;
  using Policy = typename Kernel::Policy;
  using Conv = cutlass::conv::device::ImplicitGemmConvolution<
      typename Policy::Conv2dFpropKernel>;
  using TensorRef = cutlass::TensorRef<Element, cutlass::layout::TensorNHWC>;

  auto conv_problem = make_conv_problem(problem);
  typename Conv::Arguments args(
      conv_problem,
      TensorRef{
          const_cast<Element*>(activation),
          cutlass::layout::TensorNHWC::packed(conv_problem.activation_extent())},
      TensorRef{
          const_cast<Element*>(filter),
          cutlass::layout::TensorNHWC::packed(conv_problem.filter_extent())},
      TensorRef{output, cutlass::layout::TensorNHWC::packed(conv_problem.output_extent())},
      TensorRef{output, cutlass::layout::TensorNHWC::packed(conv_problem.output_extent())},
      {typename Policy::ElementCompute(1), typename Policy::ElementCompute(0)});

  cutlass::Status status = Conv::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  Conv op;
  return op(args, nullptr, stream);
}

template <typename Kernel>
bool supported_type() {
  using Policy = typename Kernel::Policy;
  return std::is_same<typename Policy::ArchTag, cutlass::arch::Sm80>::value &&
      std::is_same<typename Policy::Element, cutlass::half_t>::value;
}

template <typename Kernel>
char const* validate_problem(PatchEmbedProblem const& problem) {
  if (!supported_type<Kernel>()) {
    return "only the explicitly instantiated TensorOp policy is available";
  }
  if (problem.batch_size <= 0 || problem.image_size <= 0 ||
      problem.in_channels <= 0 || problem.input_channels_padded <= 0 ||
      problem.embed_dim <= 0 || problem.patch_size <= 0 ||
      problem.layernorm_eps <= 0.0f) {
    return "all dimensions and layernorm epsilon must be positive";
  }
  if ((problem.image_size % problem.patch_size) != 0) {
    return "image_size must be divisible by patch_size";
  }
  if (problem.input_channels_padded < problem.in_channels) {
    return "input_channels_padded must be >= in_channels";
  }
  if ((problem.input_channels_padded % 8) != 0 || (problem.embed_dim % 8) != 0) {
    return "TensorOp NHWC conv requires padded input channels and embed_dim to be multiples of 8";
  }
  return nullptr;
}

} // namespace

template <typename Kernel>
bool PatchEmbed<Kernel>::can_implement(
    PatchEmbedProblem const& problem,
    char const** reason) {
  char const* local_reason = validate_problem<Kernel>(problem);
  if (reason != nullptr) {
    *reason = local_reason;
  }
  return local_reason == nullptr;
}

template <typename Kernel>
cutlass::Status PatchEmbed<Kernel>::run(
    PatchEmbedProblem const& problem,
    Tensors const& tensors,
    cudaStream_t stream) {
  char const* reason = nullptr;
  if (!can_implement(problem, &reason)) {
    (void)reason;
    return cutlass::Status::kErrorInvalidProblem;
  }
  if (tensors.input == nullptr || tensors.kernel == nullptr ||
      tensors.conv_output == nullptr || tensors.output == nullptr) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  Element* activation = tensors.conv_output + patch_embed_output_elements(problem);
  Element* filter = activation + patch_embed_input_padded_elements(problem);

  cudaError_t err =
      launch_pad_activation(problem, tensors.input, activation, stream);
  if (err != cudaSuccess) {
    return cutlass::Status::kErrorInternal;
  }

  err = launch_pad_filter(problem, tensors.kernel, filter, stream);
  if (err != cudaSuccess) {
    return cutlass::Status::kErrorInternal;
  }

  cutlass::Status status =
      run_conv<Kernel>(problem, activation, filter, tensors.conv_output, stream);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  err = launch_layernorm(
      problem,
      tensors.conv_output,
      tensors.bias,
      tensors.gamma,
      tensors.beta,
      tensors.output,
      stream);
  return err == cudaSuccess ? cutlass::Status::kSuccess
                            : cutlass::Status::kErrorInternal;
}

template class PatchEmbed<typename kernel::DefaultPatchEmbed<
    cutlass::arch::Sm80,
    cutlass::half_t>::Kernel>;

} // namespace device
} // namespace swin
} // namespace tiny_cutlass
