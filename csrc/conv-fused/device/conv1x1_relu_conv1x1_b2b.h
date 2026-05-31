#pragma once

#include <cuda_runtime_api.h>

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_ref.h"

namespace tiny_cutlass::conv_fused::device {

template <typename Element>
cutlass::Status run_conv1x1_relu_conv1x1_b2b_sm80(
    cutlass::conv::Conv2dProblemSize const& problem_size_0,
    cutlass::conv::Conv2dProblemSize const& problem_size_1,
    Element const* input_nhwc,
    Element const* weight0_krsc,
    Element const* bias0,
    Element const* weight1_krsc,
    Element const* bias1,
    Element* output_nhwc,
    cudaStream_t stream);

}  // namespace tiny_cutlass::conv_fused::device
