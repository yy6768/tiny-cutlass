#pragma once

/*
  Device-level driver for PatchEmbed.

  Owns the explicit supportability checks. Everything this operator refuses is
  refused loudly (kErrorNotSupported / kErrorInvalidProblem) -- there is no SIMT
  or raw-CUDA fallback, per the workspace rules.
*/

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/arch/arch.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/half.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_ref.h"

#include "swin/patch_embed/kernel/default_patch_embed.h"
#include "swin/swin_problem.h"

namespace tiny_cutlass::swin::patch_embed::device {

template <
    typename ArchTag_ = cutlass::arch::Sm80,
    typename Element_ = cutlass::half_t,
    typename ElementCompute_ = float>
class PatchEmbed {
 public:
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ElementCompute = ElementCompute_;

  using KernelConfig = kernel::DefaultPatchEmbed<ArchTag, Element, float, ElementCompute>;
  using CutlassKernel = typename KernelConfig::CutlassKernel;
  using ThreadblockShape = typename KernelConfig::ThreadblockShape;
  using Visitor = typename KernelConfig::Visitor;

  using LayoutActivation = typename KernelConfig::LayoutActivation;
  using LayoutFilter = typename KernelConfig::LayoutFilter;
  // NHWC over the TOKEN grid, not RowMajor: the conv output iterator decomposes
  // a GEMM-M row back into (n, p, q). For a packed buffer this is the same
  // memory as a [num_tokens, embed_dim] row-major tensor.
  using LayoutOutput = typename KernelConfig::LayoutOutput;

  /// Host-facing arguments: raw device pointers plus a problem descriptor.
  ///
  /// `input` is the CHANNEL-PADDED activation, NHWC with C =
  /// problem.in_channels_padded. `filter` is KRSC with the same padded C. The
  /// output is [num_tokens, embed_dim] row-major, which is also the token layout
  /// the rest of the pipeline consumes.
  struct Arguments {
    PatchEmbedProblem problem{};
    Element const* input = nullptr;
    Element const* filter = nullptr;
    ElementCompute const* bias = nullptr;   // [embed_dim], added pre-norm
    ElementCompute const* gamma = nullptr;  // [embed_dim], post-norm scale
    ElementCompute const* beta = nullptr;   // [embed_dim], post-norm shift
    Element* output = nullptr;
    ElementCompute epsilon = ElementCompute(1e-5f);
  };

  static cutlass::conv::Conv2dProblemSize to_conv_problem(
      PatchEmbedProblem const& problem) {
    int const patch = problem.patch_size;
    return cutlass::conv::Conv2dProblemSize(
        cutlass::Tensor4DCoord(
            problem.batch, problem.image_size, problem.image_size,
            problem.in_channels_padded),
        cutlass::Tensor4DCoord(
            problem.embed_dim, patch, patch, problem.in_channels_padded),
        cutlass::Tensor4DCoord(0, 0, 0, 0),   // no padding: patches tile exactly
        cutlass::MatrixCoord(patch, patch),   // stride == patch
        cutlass::MatrixCoord(1, 1),           // dilation
        cutlass::conv::Mode::kCrossCorrelation,
        /*split_k_slices=*/1);
  }

  static cutlass::Status can_implement(Arguments const& args) {
    PatchEmbedProblem const& problem = args.problem;

    if (args.input == nullptr || args.filter == nullptr || args.output == nullptr) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (problem.batch <= 0 || problem.image_size <= 0 || problem.patch_size <= 0 ||
        problem.embed_dim <= 0) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    // Non-overlapping patches: this is what makes the im2col expansion 1.0 and
    // the halo empty.
    if (problem.image_size % problem.patch_size != 0) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (problem.in_channels > problem.in_channels_padded) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    // The whole channel axis must land in one N-tile, otherwise the epilogue's
    // row statistics would be partial. See kernel/default_patch_embed.h.
    if (problem.embed_dim > ThreadblockShape::kN) {
      return cutlass::Status::kErrorNotSupported;
    }
    // A lane is either wholly in or wholly out of bounds only when the channel
    // count is a whole number of vector accesses; partial-lane masking is not
    // implemented.
    if (problem.embed_dim % Visitor::kElementsPerAccess != 0) {
      return cutlass::Status::kErrorNotSupported;
    }
    // TensorOp fp16 needs an 8-wide C load and there is no SIMT fallback.
    if (problem.in_channels_padded % KernelConfig::kAlignment != 0) {
      return cutlass::Status::kErrorNotSupported;
    }

    return cutlass::Status::kSuccess;
  }

  static size_t get_workspace_size(Arguments const&) { return 0; }

  cutlass::Status initialize(
      Arguments const& args,
      void* /*workspace*/ = nullptr,
      cudaStream_t /*stream*/ = nullptr) {
    cutlass::Status status = can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    PatchEmbedProblem const& problem = args.problem;
    cutlass::conv::Conv2dProblemSize const conv_problem = to_conv_problem(problem);

    // The output is [num_tokens, embed_dim]. As an NHWC tensor over the TOKEN
    // grid (P x Q tokens, embed_dim channels) that is the same packed buffer,
    // and it is the form the conv output iterator needs: it decomposes a GEMM-M
    // row back into (n, p, q) using the token-grid extents.
    int const P = problem.tokens_per_side();
    typename CutlassKernel::TensorRefC ref_D(
        args.output,
        LayoutOutput::packed({problem.batch, P, P, problem.embed_dim}));

    typename CutlassKernel::Arguments kernel_args(
        conv_problem,
        typename CutlassKernel::TensorRefA(
            const_cast<Element*>(args.input),
            LayoutActivation::packed({problem.batch, problem.image_size,
                                      problem.image_size,
                                      problem.in_channels_padded})),
        typename CutlassKernel::TensorRefB(
            const_cast<Element*>(args.filter),
            LayoutFilter::packed({problem.embed_dim, problem.patch_size,
                                  problem.patch_size,
                                  problem.in_channels_padded})),
        typename Visitor::Arguments(
            typename Visitor::OutputTileIterator::Params(
                CutlassKernel::ConvOutputIteratorParameter::layout(ref_D),
                cutlass::conv::implicit_gemm_tensor_c_extent(
                    cutlass::conv::Operator::kFprop, conv_problem)),
            args.output,
            args.bias,
            args.gamma,
            args.beta,
            args.epsilon));

    params_ = typename CutlassKernel::Params(kernel_args);
    return cutlass::Status::kSuccess;
  }

  cutlass::Status run(cudaStream_t stream = nullptr) {
    ThreadblockSwizzleGrid grid = grid_shape();
    dim3 const block(CutlassKernel::kThreadCount, 1, 1);
    int const smem_size = int(sizeof(typename CutlassKernel::SharedStorage));

    // Opt in to the larger dynamic shared-memory budget when the tile needs it.
    if (smem_size > (48 << 10)) {
      cudaError_t error = cudaFuncSetAttribute(
          cutlass::Kernel<CutlassKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (error != cudaSuccess) {
        return cutlass::Status::kErrorInternal;
      }
    }

    cutlass::Kernel<CutlassKernel>
        <<<grid.grid, block, smem_size, stream>>>(params_);

    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? cutlass::Status::kSuccess
                                : cutlass::Status::kErrorInternal;
  }

  cutlass::Status operator()(cudaStream_t stream = nullptr) { return run(stream); }

  cutlass::Status operator()(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr) {
    cutlass::Status status = initialize(args, workspace, stream);
    if (status == cutlass::Status::kSuccess) {
      status = run(stream);
    }
    return status;
  }

  /// Shared-memory footprint of the fused kernel, for budget reporting.
  static size_t shared_storage_size() {
    return sizeof(typename CutlassKernel::SharedStorage);
  }

 private:
  struct ThreadblockSwizzleGrid {
    dim3 grid;
  };

  ThreadblockSwizzleGrid grid_shape() const {
    typename KernelConfig::Swizzle swizzle;
    return {swizzle.get_grid_shape(params_.grid_tiled_shape)};
  }

  typename CutlassKernel::Params params_{};
};

}  // namespace tiny_cutlass::swin::patch_embed::device
