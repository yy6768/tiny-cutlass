#include "swin/window_attention/device/swin_block.h"

namespace tiny_cutlass::swin::window_attention {

template <typename Element>
cutlass::Status swin_block_can_implement(SwinBlockArguments<Element> const& args) {
  using Policy = kernel::DefaultSwinBlock<cutlass::arch::Sm80, Element, 32, 4,
      cutlass::gemm::GemmShape<16, 32, 32>, true, 8>;
  return device::SwinBlock<Policy>::can_implement(args);
}
template <typename Element>
cutlass::Status swin_block(SwinBlockArguments<Element> const& args, cudaStream_t stream) {
  using Policy = kernel::DefaultSwinBlock<cutlass::arch::Sm80, Element, 32, 4,
      cutlass::gemm::GemmShape<16, 32, 32>, true, 8>;
  return device::SwinBlock<Policy>{}(args, stream);
}
template cutlass::Status swin_block_can_implement<cutlass::half_t>(SwinBlockArguments<cutlass::half_t> const&);
template cutlass::Status swin_block<cutlass::half_t>(SwinBlockArguments<cutlass::half_t> const&, cudaStream_t);

}  // namespace tiny_cutlass::swin::window_attention
