#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

namespace tiny_cutlass::conv_fused::device {

template <typename ArchTag = cutlass::arch::Sm89, typename Element = cutlass::half_t>
size_t conv_relu_pool_workspace_size(
    int batch,
    int height,
    int width,
    int channels,
    int hidden_channels,
    int output_channels);

template <typename ArchTag = cutlass::arch::Sm89, typename Element = cutlass::half_t>
cutlass::Status run_conv_relu_pool(
    int batch,
    int height,
    int width,
    int channels,
    int hidden_channels,
    int output_channels,
    Element const* input,
    Element const* weight0,
    Element const* bias0,
    Element const* weight1,
    Element const* bias1,
    Element* output,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

}  // namespace tiny_cutlass::conv_fused::device
