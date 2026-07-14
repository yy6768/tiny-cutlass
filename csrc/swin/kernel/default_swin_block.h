/*
  CUTLASS-style factory for a complete Swin block.
*/

#pragma once

#include "kernel/default_swin_attention.h"

namespace tiny_cutlass {
namespace swin {
namespace kernel {

template <
    typename ArchTag_,
    typename Element_,
    typename AttentionConfig_ = DefaultSwinAttention<ArchTag_, Element_>>
struct DefaultSwinBlock {
  using ArchTag = ArchTag_;
  using Element = Element_;
  using AttentionConfig = AttentionConfig_;
  using ThreadblockShape = typename AttentionConfig::ThreadblockShape;
  using WarpShape = typename AttentionConfig::WarpShape;
  using InstructionShape = typename AttentionConfig::InstructionShape;
  using ElementAccumulator = typename AttentionConfig::ElementAccumulator;
  using ElementCompute = typename AttentionConfig::ElementCompute;
  using MatrixLayout = typename AttentionConfig::MatrixLayout;

  static constexpr int kThreads = AttentionConfig::kThreads;
};

} // namespace kernel
} // namespace swin
} // namespace tiny_cutlass
