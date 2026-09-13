#pragma once

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"

namespace tiny_cutlass::conv_fused::threads {

// Family aliases for CUTLASS thread-level epilogues.
// This file only chooses the output-op flavor used by the two fused conv stages.
//
// Stage 0 (the fused, block-resident conv): OnlyAlphaScaling means beta is
// dropped, because the fused kernel folds bias0 in through its scale/bias
// iterator rather than through the source tensor C.
template <typename Element, typename Accumulator, typename Compute, int Count>
using Conv0Relu = cutlass::epilogue::thread::LinearCombinationRelu<
    Element,
    Count,
    Accumulator,
    Compute,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

// Stage 1 (the final, memory-resident conv): plain linear combination, no
// activation. Default scaling keeps beta so bias1 is pulled out of the source
// tensor C with beta = 1 (see device layer binding C to the bias vector).
template <typename Element, typename Accumulator, typename Compute, int Count>
using Conv1Linear = cutlass::epilogue::thread::LinearCombination<
    Element,
    Count,
    Accumulator,
    Compute,
    cutlass::epilogue::thread::ScaleType::Default>;

}  // namespace tiny_cutlass::conv_fused::threads
