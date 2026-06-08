#pragma once

#include <cuda_runtime_api.h>

#include "cutlass/arch/arch.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/numeric_types.h"

namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu::device {

template <
    typename ArchTag = cutlass::arch::Sm89,
    typename ElementA = cutlass::float_e4m3_t,
    typename ElementB = ElementA,
    typename ElementC = ElementA,
    typename ElementScaleBias = float,
    typename ElementAccumulator = float,
    typename ElementCompute = float>
cutlass::Status run_conv1x1_relu_conv1x1_relu_fp8(
    cutlass::conv::Conv2dProblemSize const& problem_size_0,
    cutlass::conv::Conv2dProblemSize const& problem_size_1,
    ElementA const* input_nhwc,
    ElementB const* weight0_krsc,
    ElementC* stage0_output_nhwc,
    ElementScaleBias const* stage0_scale,
    ElementScaleBias const* bias0,
    ElementB const* weight1_krsc,
    ElementC const* bias1,
    ElementC* output_nhwc,
    ElementCompute stage0_alpha,
    ElementCompute output_alpha,
    cudaStream_t stream);

}  // namespace tiny_cutlass::conv_fused::fp8::conv1x1_relu_conv1x1_relu::device
