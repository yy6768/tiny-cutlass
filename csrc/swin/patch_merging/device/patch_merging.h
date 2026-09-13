#pragma once

/*
  Device-level driver for PatchMerging.

  Two launches: (gather 2x2 + LayerNorm(4C)), then the 4C -> 2C projection.
  See kernel/default_patch_merging.h for why the concat is free and why the
  LayerNorm cannot fold into the projection.
*/

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

#include "swin/patch_merging/kernel/default_patch_merging.h"
#include "swin/swin_problem.h"

namespace tiny_cutlass::swin::patch_merging::device {

template <
    typename ArchTag_ = cutlass::arch::Sm80,
    typename Element_ = cutlass::half_t,
    typename ElementCompute_ = float>
class PatchMerging {
 public:
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ElementCompute = ElementCompute_;

  using KernelConfig =
      kernel::DefaultPatchMerging<ArchTag, Element, float, ElementCompute>;
  using LayerNorm = typename KernelConfig::LayerNorm;
  using Reduction = typename KernelConfig::Reduction;

  static int const kSegments = 4;  // the 2x2 neighbourhood

  struct Arguments {
    PatchMergingProblem problem{};

    Element const* input = nullptr;   // [B*H*W, C] row-major
    // [4, num_out_tokens] gather table from build_patch_merging_index().
    int const* gather = nullptr;
    ElementCompute const* gamma = nullptr;  // [4C]
    ElementCompute const* beta = nullptr;   // [4C]
    Element const* weight = nullptr;        // [2C, 4C] row-major
    ElementCompute const* bias = nullptr;   // [2C], may be null
    Element* output = nullptr;              // [num_out_tokens, 2C]

    // Caller-owned scratch for the normalized concatenated tokens,
    // [num_out_tokens, 4C].
    Element* normalized = nullptr;

    ElementCompute epsilon = ElementCompute(1e-5f);
  };

  static cutlass::Status can_implement(Arguments const& args) {
    PatchMergingProblem const& problem = args.problem;

    if (args.input == nullptr || args.output == nullptr ||
        args.weight == nullptr || args.normalized == nullptr ||
        args.gather == nullptr) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (problem.batch <= 0 || problem.height <= 0 || problem.width <= 0 ||
        problem.channels <= 0) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    // A 2x2 merge needs both spatial extents even.
    if (problem.height % 2 != 0 || problem.width % 2 != 0) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    // Every segment is a separate gathered row, so each must be vector-aligned.
    if (problem.channels % KernelConfig::kAlignment != 0) {
      return cutlass::Status::kErrorNotSupported;
    }
    return cutlass::Status::kSuccess;
  }

  /// Bytes required for the `normalized` scratch buffer.
  static size_t normalized_size(PatchMergingProblem const& problem) {
    return size_t(problem.num_out_tokens()) *
           size_t(problem.concat_channels()) * sizeof(Element);
  }

  cutlass::Status run(Arguments const& args, cudaStream_t stream = nullptr) {
    cutlass::Status status = can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    PatchMergingProblem const& problem = args.problem;
    int const rows = problem.num_out_tokens();
    int const concat_channels = problem.concat_channels();  // 4C
    int const out_channels = problem.out_channels();        // 2C

    //
    // 1. Gather the 2x2 neighbourhood and LayerNorm over 4C, in one pass.
    //
    typename LayerNorm::Arguments ln_args;
    ln_args.input = args.input;
    ln_args.output = args.normalized;
    ln_args.gamma = args.gamma;
    ln_args.beta = args.beta;
    ln_args.rows = rows;
    ln_args.cols = concat_channels;
    ln_args.gather = args.gather;
    ln_args.num_segments = kSegments;
    // Input rows are C wide (one segment); output rows are 4C wide.
    ln_args.input_row_stride = int64_t(problem.channels);
    ln_args.output_row_stride = int64_t(concat_channels);
    ln_args.epsilon = args.epsilon;

    status = LayerNorm::run(ln_args, stream);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    //
    // 2. Projection 4C -> 2C, with the bias folded into the epilogue.
    //
    typename Reduction::Arguments reduction_args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        cutlass::gemm::GemmCoord{rows, out_channels, concat_channels},
        /*batch_count=*/1,
        {ElementCompute(1.0f), ElementCompute(0.0f)},
        args.normalized,
        args.weight,
        /*ptr_C=*/nullptr,   // no residual: the output shape differs from input
        args.output,
        /*ptr_Vector=*/const_cast<ElementCompute*>(args.bias),
        /*ptr_Tensor=*/nullptr,
        /*batch_stride_A=*/int64_t(0),
        /*batch_stride_B=*/int64_t(0),
        /*batch_stride_C=*/int64_t(0),
        /*batch_stride_D=*/int64_t(0),
        /*batch_stride_Vector=*/int64_t(0),
        /*batch_stride_Tensor=*/int64_t(0),
        /*lda=*/concat_channels,
        /*ldb=*/concat_channels,  // column-major view of [2C, 4C] row-major
        /*ldc=*/0,
        /*ldd=*/out_channels,
        /*ldr=*/0,   // broadcast bias
        /*ldt=*/0);

    Reduction reduction;
    status = reduction.can_implement(reduction_args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    status = reduction.initialize(reduction_args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    return reduction(stream);
  }

  cutlass::Status operator()(
      Arguments const& args, cudaStream_t stream = nullptr) {
    return run(args, stream);
  }
};

}  // namespace tiny_cutlass::swin::patch_merging::device
