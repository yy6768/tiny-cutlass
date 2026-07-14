#pragma once

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"

namespace tiny_cutlass::conv_fused::threads {

// Family aliases for CUTLASS thread-level epilogues.
// This file only chooses the output-op flavor used by the two fused conv stages.
template <typename Element, typename Accumulator, typename Compute, int Count>
using Conv0Relu = cutlass::epilogue::thread::LinearCombinationRelu<
    Element,
    Count,
    Accumulator,
    Compute,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

template <typename Element, typename Accumulator, typename Compute, int Count>
using Conv1Linear = cutlass::epilogue::thread::LinearCombination<
    Element,
    Count,
    Accumulator,
    Compute,
    cutlass::epilogue::thread::ScaleType::Default>;

}  // namespace tiny_cutlass::conv_fused::threads
