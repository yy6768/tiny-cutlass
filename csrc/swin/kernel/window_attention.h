/*
  Kernel-level fused Swin window attention wrapper.
*/

#pragma once

#include "../flash-attention/02-tiled-online-attention/kernel_forward.h"

namespace tiny_cutlass {
namespace swin {
namespace kernel {

template <typename KernelConfig_>
struct WindowAttentionCore {
  using KernelConfig = KernelConfig_;
  using Element = typename KernelConfig::Element;
  using Attention = AttentionKernel<
      Element,
      typename KernelConfig::ArchTag,
      KernelConfig::kAttentionIsAligned,
      KernelConfig::kAttentionQueriesPerBlock,
      KernelConfig::kAttentionKeysPerBlock,
      KernelConfig::kAttentionMaxHeadDim,
      false,
      KernelConfig::kAttentionSupportsBias>;
  using Params = typename Attention::Params;
  using SharedStorage = typename Attention::SharedStorage;

  static constexpr int kNumThreads = Attention::kNumThreads;
  static constexpr int kMinBlocksPerSm = Attention::kMinBlocksPerSm;
  static constexpr int kNoCustomMask = Attention::NoCustomMask;

  static bool check_supported(Params const& params) {
    return Attention::check_supported(params);
  }

  static void CUTLASS_DEVICE run(Params& params) {
    if (!params.advance_to_block()) {
      return;
    }
    Attention::attention_kernel(params);
  }
};

template <typename AttentionCore>
__global__ void __launch_bounds__(
    AttentionCore::kNumThreads,
    AttentionCore::kMinBlocksPerSm)
    window_attention_core_kernel(typename AttentionCore::Params params) {
  AttentionCore::run(params);
}

} // namespace kernel
} // namespace swin
} // namespace tiny_cutlass
