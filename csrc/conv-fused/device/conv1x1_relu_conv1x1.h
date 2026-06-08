#pragma once

#include <cuda_runtime_api.h>

#include "cutlass/arch/arch.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/half.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_ref.h"

namespace tiny_cutlass::conv_fused::device {

template <typename ArchTag = cutlass::arch::Sm80, typename Element = cutlass::half_t>
cutlass::Status run_conv1x1_relu_conv1x1(
    cutlass::conv::Conv2dProblemSize const& problem_size_0,
    cutlass::conv::Conv2dProblemSize const& problem_size_1,
    Element const* input,
    Element const* weight0,
    Element* stage0,
    Element const* bias0,
    Element const* weight1,
    Element const* bias1,
    Element* output,
    cudaStream_t stream);

}  // namespace tiny_cutlass::conv_fused::device
