#pragma once

#include "cutlass/epilogue/thread/linear_combination_relu.h"

namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu::threads {

// Stage0 runs in the A1 fragment loader between the two implicit-GEMMs:
// scale * accumulator + bias -> ReLU -> e4m3 fragment for the second conv.
template <typename ElementOutput, typename ElementAccumulator, typename ElementCompute, int Count>
using Stage0ReluQuantize = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,
    Count,
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaPerChannelScaling>;

// Stage1 is the final output op. In example 13's fused conv path, bias1 is exposed
// through the epilogue source iterator C1, so NoBetaScaling means
// alpha * accumulator + source -> ReLU -> e4m3 output.
template <typename ElementOutput, typename ElementAccumulator, typename ElementCompute, int Count>
using Stage1ReluQuantize = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,
    Count,
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

}  // namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu::threads
