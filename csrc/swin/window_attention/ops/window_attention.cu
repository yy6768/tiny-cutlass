#include "swin/window_attention/device/window_attention.h"

namespace tiny_cutlass::swin::window_attention {

template <typename Element>
cutlass::Status window_attention_can_implement(WindowAttentionArguments<Element> const& args) {
  if (args.problem.qk_channels <= 32 && args.problem.value_channels <= 32) {
    using Policy = kernel::DefaultWindowAttention<cutlass::arch::Sm80, Element,
        cutlass::gemm::GemmShape<16, 128, 32>, cutlass::gemm::GemmShape<16, 32, 32>,
        cutlass::layout::RowMajor, 32, 32>;
    return device::WindowAttention<Policy>::can_implement(args);
  }
  using Policy = kernel::DefaultWindowAttention<cutlass::arch::Sm80, Element>;
  return device::WindowAttention<Policy>::can_implement(args);
}

template <typename Element>
cutlass::Status window_attention(WindowAttentionArguments<Element> const& args, cudaStream_t stream) {
  if (args.problem.qk_channels <= 32 && args.problem.value_channels <= 32) {
    using Policy = kernel::DefaultWindowAttention<cutlass::arch::Sm80, Element,
        cutlass::gemm::GemmShape<16, 128, 32>, cutlass::gemm::GemmShape<16, 32, 32>,
        cutlass::layout::RowMajor, 32, 32>;
    return device::WindowAttention<Policy>{}(args, stream);
  }
  using Policy = kernel::DefaultWindowAttention<cutlass::arch::Sm80, Element>;
  return device::WindowAttention<Policy>{}(args, stream);
}

template cutlass::Status window_attention_can_implement<cutlass::half_t>(WindowAttentionArguments<cutlass::half_t> const&);
template cutlass::Status window_attention<cutlass::half_t>(WindowAttentionArguments<cutlass::half_t> const&, cudaStream_t);

}  // namespace tiny_cutlass::swin::window_attention
