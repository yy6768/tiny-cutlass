// Copyright (c) 2022 - 2026 Ali Hassani.
// SPDX-License-Identifier: MIT
// Adapted from NATTEN na_utils.cuh at 92750c3cf837652d58b6091e5ebd37dcad46e753.
// See ../../LICENSE-NATTEN for the full license.
#pragma once

#include <cutlass/cutlass.h>

namespace tiny_cutlass {
namespace natten {

// Non-causal, stride=dilation=1: shift the complete window at either edge.
// For even windows the extra neighbor is on the left, as in NATTEN.
CUTLASS_HOST_DEVICE inline int neighborhood_start(int query, int length, int window) {
  int start = query - window / 2;
  start = start < 0 ? 0 : start;
  return start > length - window ? length - window : start;
}

} // namespace natten
} // namespace tiny_cutlass
