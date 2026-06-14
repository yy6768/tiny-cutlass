/*
  Kernel-level launch wrappers for Swin threadblock stages.
*/

#pragma once

namespace tiny_cutlass {
namespace swin {
namespace kernel {

template <typename Threadblock>
__global__ void __launch_bounds__(Threadblock::kNumThreads, Threadblock::kMinBlocksPerSm)
    swin_threadblock_kernel(typename Threadblock::Params params) {
  Threadblock::run(params);
}

template <typename Attention>
__global__ void __launch_bounds__(Attention::kNumThreads, Attention::kMinBlocksPerSm)
    swin_attention_kernel(typename Attention::Params params) {
  if (!params.advance_to_block()) {
    return;
  }
  Attention::attention_kernel(params);
}

} // namespace kernel
} // namespace swin
} // namespace tiny_cutlass
