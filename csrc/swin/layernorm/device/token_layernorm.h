#pragma once

/*
  Device-level driver for the pre-GEMM token LayerNorm.

  Shared by swin_mlp (LN2) and patch_merging (norm before the projection). See
  kernel/token_layernorm.h for why a LayerNorm that precedes a GEMM cannot be
  folded into that GEMM.
*/

#include <cstdint>

#include <cuda_runtime_api.h>

#include "cutlass/cutlass.h"
#include "cutlass/half.h"

#include "swin/layernorm/kernel/token_layernorm.h"

namespace tiny_cutlass::swin::layernorm::device {

template <
    typename Element_ = cutlass::half_t,
    typename ElementCompute_ = float,
    int ElementsPerAccess = 8,
    int WarpsPerBlock = 4>
class TokenLayerNorm {
 public:
  using Element = Element_;
  using ElementCompute = ElementCompute_;

  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kWarpsPerBlock = WarpsPerBlock;
  static int const kThreadsPerBlock = 32 * kWarpsPerBlock;

  using Params = kernel::TokenLayerNormParams<Element, ElementCompute>;

  struct Arguments {
    Element const* input = nullptr;
    Element* output = nullptr;
    ElementCompute const* gamma = nullptr;
    ElementCompute const* beta = nullptr;
    int rows = 0;
    int cols = 0;  // total normalized width
    // Row strides in ELEMENTS. Zero means "packed": cols for the plain case, or
    // segment_cols when a segmented gather is configured (the input rows are
    // narrower than the normalized width).
    int64_t input_row_stride = 0;
    int64_t output_row_stride = 0;
    ElementCompute epsilon = ElementCompute(1e-5f);

    // Optional segmented gather: [num_segments, rows] row indices, where
    // segment k fills channels [k * cols/num_segments, ...). Used by
    // PatchMerging to fold its 2x2 concat into this pass.
    int const* gather = nullptr;
    int num_segments = 1;
  };

  static cutlass::Status can_implement(Arguments const& args) {
    if (args.input == nullptr || args.output == nullptr) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (args.rows <= 0 || args.cols <= 0) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (args.num_segments <= 0 || args.cols % args.num_segments != 0) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (args.num_segments > 1 && args.gather == nullptr) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    // Vectorized access only; a tail loop is deliberately not implemented
    // because every Swin channel count in this pipeline is a multiple of 8.
    // Each SEGMENT must be vector-aligned too, since a gather jumps rows at
    // every segment boundary.
    if (args.cols % kElementsPerAccess != 0) {
      return cutlass::Status::kErrorNotSupported;
    }
    if ((args.cols / args.num_segments) % kElementsPerAccess != 0) {
      return cutlass::Status::kErrorNotSupported;
    }
    return cutlass::Status::kSuccess;
  }

  static cutlass::Status run(
      Arguments const& args, cudaStream_t stream = nullptr) {
    cutlass::Status status = can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    Params params;
    params.input = args.input;
    params.output = args.output;
    params.gamma = args.gamma;
    params.beta = args.beta;
    params.rows = args.rows;
    params.cols = args.cols;
    params.gather = args.gather;
    params.num_segments = args.num_segments;
    params.segment_cols = args.cols / args.num_segments;
    // Packed default: a gathered input's rows are one SEGMENT wide, not the full
    // normalized width.
    params.input_row_stride = args.input_row_stride != 0
                                  ? args.input_row_stride
                                  : int64_t(params.segment_cols);
    params.output_row_stride =
        args.output_row_stride != 0 ? args.output_row_stride : int64_t(args.cols);
    params.epsilon = args.epsilon;

    dim3 const grid((args.rows + kWarpsPerBlock - 1) / kWarpsPerBlock, 1, 1);
    dim3 const block(kThreadsPerBlock, 1, 1);

    kernel::token_layernorm_kernel<Element, ElementCompute, kElementsPerAccess,
                                   kWarpsPerBlock>
        <<<grid, block, 0, stream>>>(params);

    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? cutlass::Status::kSuccess
                                : cutlass::Status::kErrorInternal;
  }

  cutlass::Status operator()(
      Arguments const& args, cudaStream_t stream = nullptr) const {
    return run(args, stream);
  }
};

}  // namespace tiny_cutlass::swin::layernorm::device
