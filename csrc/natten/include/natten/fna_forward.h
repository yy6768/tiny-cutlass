/***************************************************************************************************
 * Copyright (c) 2026 tiny-cutlass contributors.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <cutlass/arch/arch.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/numeric_types.h>

namespace tiny_cutlass {
namespace natten {

template <
    int Rank,
    typename CausalMask,
    typename Element,
    typename ArchTag,
    typename ThreadblockShape,
    bool IsAligned = true>
struct DefaultFnaForwardPolicy {
  static constexpr int kRank = Rank;
  static constexpr bool kIsAligned = IsAligned;

  using CausalMaskTag = CausalMask;
  using ElementInput = Element;
  using ElementOutput = Element;
  using ElementAccumulator = float;
  using Arch = ArchTag;
  using Threadblock = ThreadblockShape;
};

template <bool IsCausal>
struct FnaCausalMask {
  static constexpr bool kIsCausal = IsCausal;
};

struct FnaForwardProblem {
  int batch_size = 2;
  int length = 128;
  int heads = 2;
  int head_dim = 32;
  int head_dim_value = 32;
  int kernel_size = 33;
  int stride = 1;
  int dilation = 1;
  int query_tile_size = 64;
  int key_tile_size = 64;
  float scale = 1.0f;
};

} // namespace natten
} // namespace tiny_cutlass
