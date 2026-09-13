#pragma once
#include "swin/window_attention/kernel/default_window_attention.h"
#include "swin/window_attention/threadblock/swin_block_mma.h"

namespace tiny_cutlass::swin::window_attention::kernel {

template <typename ArchTag_, typename Element_, int Channels = 32, int Ratio = 4,
          typename ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 32>,
          bool StaticAttention = false, int Access = 1>
struct DefaultSwinBlock {
  using ArchTag = ArchTag_;
  using Element = Element_;
  using Attention = DefaultWindowAttention<ArchTag, Element, ThreadblockShape,
      cutlass::gemm::GemmShape<16, 32, 32>, cutlass::layout::RowMajor, Channels, Channels>;
  using AttentionMma = threadblock::WindowAttentionMma<typename Attention::WarpMma, Attention::kWarps, Channels, Channels, Access>;
  using Mma = threadblock::SwinBlockMma<AttentionMma, Channels, Ratio, StaticAttention>;
  using CutlassKernel = WindowAttention<Mma>;
  static constexpr int kChannels = Channels, kRatio = Ratio;
};

}  // namespace tiny_cutlass::swin::window_attention::kernel
