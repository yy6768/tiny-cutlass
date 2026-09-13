#pragma once

/*
  Epilogue visitor that finishes PatchEmbed inside the conv kernel:

      out[m, :] = gamma * (conv(x)[m, :] + bias - mean_m) / sqrt(var_m + eps)
                  + beta

  where the mean / variance are taken along the channel (N / GEMM-column) axis
  of row m. Both statistics are produced and consumed inside a single kernel --
  no partial-reduction buffer, no finalize kernel.

  ---------------------------------------------------------------------------
  WHY A SINGLE KERNEL WORKS HERE (and does not in CUTLASS example 37)
  ---------------------------------------------------------------------------
  Example 37 splits its layernorm into three launches because in general a
  row of the output tile is spread across several threadblocks along N, so no
  single CTA can see a whole row.

  PatchEmbed escapes that: choose ThreadblockShape::kN >= embed_dim and the
  whole channel axis lands in ONE N-tile, so grid.n() == 1 and every row is
  owned end to end by one CTA. Measured geometry for the shipping config
  (TB 128x128x32, warps 64x64x32, fp16, 8-element access):

      kThreads             128
      kElementsPerAccess     8
      Iterations::kColumn    1     <-- the load-bearing one
      Iterations::kRow       2
      kThreadsPerRow        16     (128 columns / 8 per access)

  Iterations::kColumn == 1 means a thread's entire contribution to a row
  arrives in a SINGLE visit() call. So after the 16-lane butterfly

      for (i = 8; i > 0; i >>= 1) v += __shfl_xor_sync(-1u, v, i);

  every lane in the row group holds the COMPLETE sum and sum-of-squares for
  that row. Nothing is left to reduce, which is why this visitor can normalize
  in place rather than spilling partials.

  The static_asserts below enforce exactly that precondition, so a future tile
  shape that breaks it fails to compile instead of silently normalizing by a
  partial sum.

  ---------------------------------------------------------------------------
  WHY THE MASKED LANES ARE BENIGN AT embed_dim = 96
  ---------------------------------------------------------------------------
  N-tile 128 with 8 elements per access gives 16 lanes per row; 96 / 8 == 12
  exactly, so lanes 0-11 are fully in bounds and lanes 12-15 are fully out of
  bounds. There is no partially-valid lane, so masking is just "zero the whole
  fragment's contribution" -- done by clearing both partial sums under
  column_guard before the butterfly. A channel count that is not a multiple of
  kElementsPerAccess would need per-element masking; can_implement rejects it.

  ---------------------------------------------------------------------------
  ORDERING (this is where a layernorm fusion usually goes wrong)
  ---------------------------------------------------------------------------
  bias is added BEFORE the statistics are taken -- it is part of the conv
  output being normalized, not an afterthought. gamma/beta are applied AFTER
  normalizing. The variance is the BIASED one (divide by N, not N-1) and is
  accumulated in fp32, matching both PyTorch's LayerNorm and the host
  reference.

  Two passes over the row would be the textbook way to get mean then variance;
  instead this accumulates sum and sum-of-squares in one pass and uses
  var = E[x^2] - E[x]^2. That is algebraically identical and, in fp32 over 96
  well-scaled values, numerically indistinguishable at the tolerance this
  family verifies against.
*/

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/fast_math.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"

namespace tiny_cutlass::swin::patch_embed::epilogue {

/// Epilogue visitor computing bias + channel-axis LayerNorm in one pass.
///
/// Satisfies: EpilogueFusedVisitorConcept (see
/// cutlass/epilogue/threadblock/epilogue_with_visitor.h)
template <
    typename ThreadblockShape_,
    int ThreadCount,
    typename OutputTileIterator_,
    typename AccumulatorTile_,
    typename ElementAccumulator_,
    typename ElementCompute_ = float>
class EpilogueVisitorLayerNorm {
 public:
  using ThreadblockShape = ThreadblockShape_;
  using OutputTileIterator = OutputTileIterator_;
  using AccumulatorTile = AccumulatorTile_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  using ElementOutput = typename OutputTileIterator::Element;
  using LayoutOutput = cutlass::layout::RowMajor;
  using TensorRefD = cutlass::TensorRef<ElementOutput, LayoutOutput>;

  static int const kThreadCount = ThreadCount;
  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  using ThreadMap = typename OutputTileIterator::ThreadMap;
  static int const kThreads = ThreadMap::kThreads;
  static int const kRowIterations = ThreadMap::Iterations::kRow;
  static int const kColumnIterations = ThreadMap::Iterations::kColumn;

  // Lanes cooperating on one output row, and the butterfly's first stride.
  static int const kThreadsPerRow =
      ThreadMap::Detail::RowArrangement::Detail::kShapeWidth;
  static int const kHalfThreadsPerRow = kThreadsPerRow >> 1;

  // The single-kernel-LayerNorm precondition, enforced at compile time: a
  // thread must see its whole row contribution in one visit() so the butterfly
  // produces COMPLETE statistics. See the file header.
  static_assert(
      kColumnIterations == 1,
      "EpilogueVisitorLayerNorm requires ThreadMap::Iterations::kColumn == 1 so "
      "that one visit() carries a thread's entire row contribution; pick a tile "
      "shape with ThreadblockShape::kN >= channels");
  static_assert(
      kThreadsPerRow > 0 && (kThreadsPerRow & (kThreadsPerRow - 1)) == 0,
      "the shfl_xor butterfly needs a power-of-two lane count per row");
  static_assert(
      kThreadsPerRow <= 32,
      "a row group must fit inside one warp for __shfl_xor_sync to reduce it");

  using AccumulatorFragment = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using ComputeFragment = cutlass::Array<ElementCompute, kElementsPerAccess>;
  using OutputVector = cutlass::Array<ElementOutput, kElementsPerAccess>;

  /// Host-facing arguments.
  ///
  /// The output iterator's Params is passed in ALREADY BUILT rather than being
  /// constructed from a TensorRef here. The conv path's output iterator is
  /// PredicatedTileIteratorConv, whose Params needs an NHWC layout plus the 4D
  /// conv extent (it decomposes a GEMM-M row into (n, p, q) with fast-divmods);
  /// a GEMM path's iterator wants a RowMajor stride instead. Keeping that
  /// construction in the caller lets one visitor serve both without knowing
  /// which iterator it is driving.
  struct Arguments {
    typename OutputTileIterator::Params params_D{};
    ElementOutput* ptr_D = nullptr;
    ElementCompute const* ptr_Bias = nullptr;   // [channels], pre-norm
    ElementCompute const* ptr_Gamma = nullptr;  // [channels], post-norm scale
    ElementCompute const* ptr_Beta = nullptr;   // [channels], post-norm shift
    ElementCompute epsilon = ElementCompute(1e-5f);

    Arguments() = default;

    CUTLASS_HOST_DEVICE
    Arguments(
        typename OutputTileIterator::Params const& params_D_,
        ElementOutput* ptr_D_,
        ElementCompute const* ptr_Bias_,
        ElementCompute const* ptr_Gamma_,
        ElementCompute const* ptr_Beta_,
        ElementCompute epsilon_)
        : params_D(params_D_),
          ptr_D(ptr_D_),
          ptr_Bias(ptr_Bias_),
          ptr_Gamma(ptr_Gamma_),
          ptr_Beta(ptr_Beta_),
          epsilon(epsilon_) {}
  };

  /// Device-side parameters. Structurally identical to Arguments here, kept
  /// separate to match the CUTLASS convention.
  struct Params {
    typename OutputTileIterator::Params params_D{};
    ElementOutput* ptr_D = nullptr;
    ElementCompute const* ptr_Bias = nullptr;
    ElementCompute const* ptr_Gamma = nullptr;
    ElementCompute const* ptr_Beta = nullptr;
    ElementCompute epsilon = ElementCompute(1e-5f);

    Params() = default;

    CUTLASS_HOST_DEVICE
    explicit Params(Arguments const& args)
        : params_D(args.params_D),
          ptr_D(args.ptr_D),
          ptr_Bias(args.ptr_Bias),
          ptr_Gamma(args.ptr_Gamma),
          ptr_Beta(args.ptr_Beta),
          epsilon(args.epsilon) {}
  };

  /// This visitor keeps no shared state; the reduction is entirely in-register.
  struct SharedStorage {};

 private:
  Params const& params_;
  cutlass::MatrixCoord extent_;
  OutputTileIterator iterator_D_;
  typename OutputTileIterator::Fragment fragment_D_;
  cutlass::MatrixCoord thread_offset_;

  // Row statistics, complete after the butterfly in visit().
  ElementCompute row_mean_ = ElementCompute(0);
  ElementCompute row_inv_std_ = ElementCompute(0);

 public:
  CUTLASS_DEVICE
  EpilogueVisitorLayerNorm(
      Params const& params,
      SharedStorage& shared_storage,
      cutlass::MatrixCoord const& problem_size,
      int thread_idx,
      int warp_idx,
      int lane_idx,
      cutlass::MatrixCoord const& threadblock_offset = cutlass::MatrixCoord(0, 0))
      : params_(params),
        extent_(problem_size),
        iterator_D_(
            params.params_D,
            params.ptr_D,
            problem_size,
            thread_idx,
            threadblock_offset) {}

  CUTLASS_DEVICE
  void set_k_partition(int split_k_index, int split_k_slices) {
    // Split-K would break the single-pass reduction: a CTA would only hold a
    // partial accumulator when the statistics are taken. The device layer
    // rejects split-K, so there is nothing to do here.
  }

  CUTLASS_DEVICE
  void set_batch_index(int batch_idx) {}

  CUTLASS_DEVICE
  void begin_epilogue() {}

  CUTLASS_DEVICE
  void begin_step(int step_idx) { fragment_D_.clear(); }

  CUTLASS_DEVICE
  void begin_row(int row_idx) {
    row_mean_ = ElementCompute(0);
    row_inv_std_ = ElementCompute(0);
  }

  /// Adds bias, reduces the row, then normalizes and applies gamma/beta.
  ///
  /// Because Iterations::kColumn == 1 this runs exactly once per (thread, row),
  /// so reduce-then-normalize inside a single call is correct.
  CUTLASS_DEVICE
  void visit(
      int iter_idx,
      int row_idx,
      int column_idx,
      int frag_idx,
      AccumulatorFragment const& accum) {
    thread_offset_ = iterator_D_.thread_start() +
                     OutputTileIterator::ThreadMap::iteration_offset(frag_idx);

    bool const column_guard = thread_offset_.column() < extent_.column();

    // Convert accumulators to compute precision and fold in the pre-norm bias.
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator,
                                  kElementsPerAccess>
        accum_converter;
    ComputeFragment value = accum_converter(accum);

    if (params_.ptr_Bias != nullptr && column_guard) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kElementsPerAccess; ++i) {
        value[i] += params_.ptr_Bias[thread_offset_.column() + i];
      }
    }

    // Partial sum / sum-of-squares. Out-of-bounds lanes contribute nothing;
    // at embed_dim % kElementsPerAccess == 0 a lane is either wholly in or
    // wholly out of bounds, so one guard covers the whole fragment.
    ElementCompute sum = ElementCompute(0);
    ElementCompute sum_sq = ElementCompute(0);
    if (column_guard) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kElementsPerAccess; ++i) {
        sum += value[i];
        sum_sq += value[i] * value[i];
      }
    }

    // Butterfly across the lanes owning this row. Every participating lane ends
    // up with the complete totals.
    CUTLASS_PRAGMA_UNROLL
    for (int i = kHalfThreadsPerRow; i > 0; i >>= 1) {
      sum += __shfl_xor_sync(0xFFFFFFFFu, sum, i);
      sum_sq += __shfl_xor_sync(0xFFFFFFFFu, sum_sq, i);
    }

    ElementCompute const inv_n =
        ElementCompute(1) / ElementCompute(extent_.column());
    ElementCompute const mean = sum * inv_n;
    // Biased variance, clamped at zero: E[x^2] - E[x]^2 can go slightly
    // negative from cancellation when a row is nearly constant.
    ElementCompute variance = sum_sq * inv_n - mean * mean;
    if (variance < ElementCompute(0)) {
      variance = ElementCompute(0);
    }

    row_mean_ = mean;
    row_inv_std_ = ElementCompute(1) /
                   cutlass::fast_sqrt(variance + params_.epsilon);

    // Normalize, then scale and shift.
    ComputeFragment result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute normalized = (value[i] - row_mean_) * row_inv_std_;
      if (column_guard) {
        int const column = thread_offset_.column() + i;
        if (params_.ptr_Gamma != nullptr) {
          normalized *= params_.ptr_Gamma[column];
        }
        if (params_.ptr_Beta != nullptr) {
          normalized += params_.ptr_Beta[column];
        }
      }
      result[i] = normalized;
    }

    cutlass::NumericArrayConverter<ElementOutput, ElementCompute,
                                  kElementsPerAccess>
        output_converter;
    OutputVector& output = reinterpret_cast<OutputVector*>(&fragment_D_)[frag_idx];
    output = output_converter(result);
  }

  CUTLASS_DEVICE
  void end_row(int row_idx) {}

  CUTLASS_DEVICE
  void end_step(int step_idx) {
    iterator_D_.store(fragment_D_);
    ++iterator_D_;
  }

  CUTLASS_DEVICE
  void end_epilogue() {}
};

}  // namespace tiny_cutlass::swin::patch_embed::epilogue
