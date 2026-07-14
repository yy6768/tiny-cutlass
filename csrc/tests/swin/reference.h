/*
  Swin attention correctness reference implemented with cuDNN frontend SDPA.
*/

#pragma once

#include <string>

#include <cuda_runtime.h>
#include "cutlass/half.h"

#include "../../swin/swin.h"

namespace tiny_cutlass {
namespace swin {

cudaError_t run_cudnn_swin_attention_reference(
    SwinAttentionProblem const& problem,
    cutlass::half_t const* query,
    cutlass::half_t const* key,
    cutlass::half_t const* value,
    cutlass::half_t const* attention_bias,
    cutlass::half_t* output,
    cudaStream_t stream,
    std::string& error_message);

} // namespace swin
} // namespace tiny_cutlass
