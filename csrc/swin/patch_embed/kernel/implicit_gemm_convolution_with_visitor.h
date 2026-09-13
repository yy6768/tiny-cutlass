#pragma once

/*
  Implicit-GEMM convolution kernel that drives an EPILOGUE VISITOR.

  ---------------------------------------------------------------------------
  WHY THIS FORK EXISTS
  ---------------------------------------------------------------------------
  Verified against CUTLASS 4.5.2: the epilogue-visitor mechanism only reaches
  the GEMM kernel layer. Nothing under include/cutlass/conv/ references
  EpilogueWithVisitor at all, and the stock conv kernel calls its epilogue with
  the plain output-op signature

      cutlass/conv/kernel/implicit_gemm_convolution.h:425
          epilogue(output_op, iterator_D, accumulators, iterator_C);

  A plain EpilogueOutputOp is a per-element thread functor: no cross-lane hook,
  no begin_row/end_row, no second pass. A LayerNorm along the channel axis needs
  a reduction across lanes, so it cannot be expressed that way. An
  EpilogueWithVisitor instead calls

      epilogue(visitor, accumulators);

  and drives visitor.begin_row / visit / end_row per output row.

  So this file is a near-verbatim copy of the stock ImplicitGemmConvolution with
  exactly three changes:

    1. Epilogue is constructed and invoked in visitor form.
    2. Params/Arguments carry the visitor's Arguments instead of an
       EpilogueOutputOp::Params, and there is no ref_C / ptr_C -- the visitor
       owns its own output iterator and any auxiliary tensors.
    3. Split-K is removed rather than adapted. Under split-K a CTA holds only a
       PARTIAL accumulator when the epilogue runs, so channel-axis statistics
       taken there would be statistics of a partial sum. Rather than silently
       computing a wrong normalization, the whole path is gone: no semaphore, no
       SplitKMode, and the device layer rejects split_k_slices != 1.

  Forking a conv kernel to change its epilogue contract is the established
  pattern in this repo -- see
  csrc/conv-fused/conv1x1_dual/kernel/b2b_implicit_gemm_convolution.h.

  Everything else -- iterator construction, the mainloop call, the swizzle, the
  grid mapping -- is upstream's, unchanged.
*/

#include "cutlass/cutlass.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/epilogue/threadblock/output_iterator_parameter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

namespace tiny_cutlass::swin::patch_embed::kernel {

/// Implicit GEMM convolution whose epilogue is an EpilogueWithVisitor.
///
/// Mma_       : threadblock-scoped MMA (same as upstream)
/// Epilogue_  : an EpilogueWithVisitor instantiation
/// ConvOperator: fprop / dgrad / wgrad -- this family only instantiates fprop
template <
    typename Mma_,
    typename Epilogue_,
    typename ThreadblockSwizzle_,
    cutlass::conv::Operator ConvOperator,
    typename ConvProblemSize_ = cutlass::conv::Conv2dProblemSize>
struct ImplicitGemmConvolutionWithVisitor {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using Visitor = typename Epilogue::Visitor;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static cutlass::conv::Operator const kConvolutionalOperator = ConvOperator;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;

  // The visitor owns the output tensor, so C's element type comes from its
  // output iterator rather than from an output op.
  using ElementC = typename Visitor::ElementOutput;
  using LayoutC = LayoutA;

  using ElementAccumulator = typename Visitor::ElementAccumulator;
  using ElementCompute = typename Visitor::ElementCompute;

  using WarpMmaOperator = typename Mma::Policy::Operator;
  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename ArchMmaOperator::Operator;
  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;

  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename WarpMmaOperator::Shape;
  using InstructionShape = typename ArchMmaOperator::Shape;

  static int const kStages = Mma::kStages;
  static cutlass::conv::IteratorAlgorithm const kIteratorAlgorithm =
      Mma::IteratorA::kIteratorAlgorithm;
  static cutlass::conv::StrideSupport const kStrideSupport =
      Mma::IteratorA::kStrideSupport;

  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using TensorRefA = typename Mma::IteratorA::TensorRef;
  using TensorRefB = typename Mma::IteratorB::TensorRef;
  using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;

  static_assert(
      Mma::IteratorA::kConvDim == Mma::IteratorB::kConvDim,
      "Convolution on different dimensions is not supported");
  static int const kConvDim = Mma::IteratorA::kConvDim;

  using ConvProblemSize = ConvProblemSize_;
  static cutlass::conv::GroupMode const kGroupMode =
      cutlass::conv::GroupMode::kNone;

  static int const kTensorCStrideIdx = 0;  // fprop

  using ConvOutputIteratorParameter =
      cutlass::epilogue::threadblock::ConvOutputIteratorParameter<
          LayoutC,
          typename Visitor::OutputTileIterator::Layout,
          TensorRefC,
          ConvOperator,
          ConvProblemSize>;

  /// Host-facing arguments. No ref_C / ref_D: the visitor holds its own tensors.
  struct Arguments {
    ConvProblemSize problem_size{};
    TensorRefA ref_A{};
    TensorRefB ref_B{};
    typename Visitor::Arguments visitor{};

    CUTLASS_HOST_DEVICE
    Arguments() {}

    CUTLASS_HOST_DEVICE
    Arguments(
        ConvProblemSize const& problem_size_,
        TensorRefA const& ref_A_,
        TensorRefB const& ref_B_,
        typename Visitor::Arguments const& visitor_)
        : problem_size(problem_size_),
          ref_A(ref_A_),
          ref_B(ref_B_),
          visitor(visitor_) {}
  };

  /// Device-side parameters.
  struct Params {
    ConvProblemSize problem_size{};
    cutlass::gemm::GemmCoord grid_tiled_shape{};
    cutlass::gemm::GemmCoord implicit_gemm_problem_size{};
    int swizzle_log_tile = 0;

    int gemm_k_iterations = 0;
    int gemm_k_iterations_per_channel = 0;

    typename Mma::IteratorA::Params iterator_A{};
    typename Mma::IteratorA::Element const* ptr_A = nullptr;
    typename Mma::IteratorB::Params iterator_B{};
    typename Mma::IteratorB::Element const* ptr_B = nullptr;

    typename Visitor::Params visitor{};

    CUTLASS_HOST_DEVICE
    Params() {}

    CUTLASS_HOST_DEVICE
    explicit Params(Arguments const& args)
        : problem_size(args.problem_size),
          implicit_gemm_problem_size(cutlass::conv::implicit_gemm_problem_size(
              kConvolutionalOperator, args.problem_size)),
          iterator_A(
              Mma::IteratorA::getParams(args.problem_size, args.ref_A.layout())),
          ptr_A(args.ref_A.data()),
          iterator_B(args.problem_size, args.ref_B.layout()),
          ptr_B(args.ref_B.data()),
          visitor(args.visitor) {
      gemm_k_iterations = cutlass::conv::implicit_gemm_k_iterations(
          kConvolutionalOperator,
          ThreadblockShape::kK,
          args.problem_size,
          kIteratorAlgorithm,
          kGroupMode,
          ThreadblockShape::kN);

      gemm_k_iterations_per_channel =
          cutlass::conv::implicit_gemm_k_iterations_per_channel(
              kConvolutionalOperator, args.problem_size, kIteratorAlgorithm);

      ThreadblockSwizzle threadblock_swizzle;

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
          implicit_gemm_problem_size,
          {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
          args.problem_size.split_k_slices);

      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  /// Shared storage. The visitor's storage joins the union because it is live
  /// only during the epilogue, exactly like the epilogue's own storage.
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    struct {
      typename Epilogue::SharedStorage epilogue;
      typename Visitor::SharedStorage visitor;
    } epilogue;
  };

  CUTLASS_HOST_DEVICE
  ImplicitGemmConvolutionWithVisitor() {}

  /// Executes one implicit GEMM with a visitor epilogue.
  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared_storage) {
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if the CTA is out of range.
    if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m() ||
        params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {
      return;
    }

    int thread_idx = threadIdx.x;
    int iterator_A_column_offset = threadblock_tile_idx.k() * Mma::Shape::kK;

    typename Mma::IteratorA iterator_A(
        params.iterator_A,
        params.problem_size,
        params.ptr_A,
        thread_idx,
        cutlass::MatrixCoord(
            threadblock_tile_idx.m() * Mma::Shape::kM, iterator_A_column_offset));

    typename Mma::IteratorB iterator_B(
        params.iterator_B,
        params.problem_size,
        params.ptr_B,
        thread_idx,
        cutlass::MatrixCoord(
            threadblock_tile_idx.k() * Mma::Shape::kK,
            threadblock_tile_idx.n() * Mma::Shape::kN));

    // Broadcast the warp id computed by lane 0 so dependent code is
    // warp-uniform.
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop (unchanged from upstream)
    //

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;
    accumulators.clear();

    mma(params.gemm_k_iterations,
        accumulators,
        iterator_A,
        iterator_B,
        accumulators,
        params.gemm_k_iterations_per_channel);

    //
    // Epilogue: visitor form
    //

    // No semaphore and no split-K bookkeeping. The accumulator here is the
    // COMPLETE dot product for this output tile, which is what makes the
    // channel-axis reduction in the visitor meaningful.
    threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    cutlass::MatrixCoord threadblock_offset(
        threadblock_tile_idx.m() * Mma::Shape::kM,
        threadblock_tile_idx.n() * Mma::Shape::kN);

    Visitor visitor(
        params.visitor,
        shared_storage.epilogue.visitor,
        ConvOutputIteratorParameter::extent(params.problem_size),
        thread_idx,
        warp_idx,
        lane_idx,
        threadblock_offset);

    Epilogue epilogue(
        shared_storage.epilogue.epilogue, thread_idx, warp_idx, lane_idx);

    epilogue(visitor, accumulators);
  }
};

}  // namespace tiny_cutlass::swin::patch_embed::kernel
