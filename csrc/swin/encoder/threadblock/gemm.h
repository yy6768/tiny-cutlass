#pragma once

#include <cassert>
#include <type_traits>
#include "cutlass/aligned_buffer.h"
#include "cutlass/half.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h"

namespace tiny_cutlass::swin::encoder::threadblock {

// One warp owns the complete M/N/K traversal. The canonical tensors may live in
// global or shared memory; CUTLASS regular iterators use generic CUDA pointers.
// Shared operand staging follows DefaultMmaCore's ThreadMap -> SmemIterator
// organization. WarpMma and IteratorC retain CUTLASS's fragment ownership.
template <typename WarpMma_>
struct Gemm {
  using WarpMma = WarpMma_;
  using Shape = typename WarpMma::Shape;
  using InstructionShape = typename WarpMma::InstructionShape;
  using Element = typename WarpMma::ElementA;
  using ElementB = typename WarpMma::ElementB;
  using Accumulator = typename WarpMma::ElementC;
  using LayoutA = typename WarpMma::LayoutA;
  using LayoutB = typename WarpMma::LayoutB;
  using RowMajor = cutlass::layout::RowMajor;
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using ThreadMapA = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<Shape::kK, Shape::kM>, 32,
      cutlass::layout::PitchLinearShape<4, 8>, 8>;
  using ThreadMapB = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<Shape::kK, Shape::kN>, 32,
      cutlass::layout::PitchLinearShape<4, 8>, 8>;
  using SourceIteratorA = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<Shape::kM, Shape::kK>, Element, RowMajor, 1, ThreadMapA>;
  using SourceIteratorB = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<Shape::kK, Shape::kN>, Element, ColumnMajor, 0, ThreadMapB>;
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<Shape::kM, Shape::kK>, Element, LayoutA, 1, ThreadMapA>;
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<Shape::kK, Shape::kN>, Element, LayoutB, 0, ThreadMapB>;
  using StagingFragmentA = typename SmemIteratorA::Fragment;
  using StagingFragmentB = typename SmemIteratorB::Fragment;
  using IteratorA = typename WarpMma::IteratorA;
  using IteratorB = typename WarpMma::IteratorB;
  using IteratorC = typename WarpMma::IteratorC;
  using FragmentA = typename WarpMma::FragmentA;
  using FragmentB = typename WarpMma::FragmentB;
  using FragmentC = typename WarpMma::FragmentC;

  static_assert(Shape::kM == 16 && Shape::kN == 32 && Shape::kK == 32,
                "encoder GEMM requires a 16x32x32 warp tile");
  static_assert(InstructionShape::kM == 16 && InstructionShape::kN == 8 &&
                InstructionShape::kK == 8, "encoder GEMM requires m16n8k8");
  static_assert(std::is_same_v<Element, cutlass::half_t> &&
                std::is_same_v<Element, ElementB> &&
                std::is_same_v<Element, Accumulator>,
                "the source PTX uses half operands and half accumulators");

  struct SharedStorage {
    cutlass::AlignedBuffer<Element, Shape::kM * Shape::kK> a;
    cutlass::AlignedBuffer<Element, Shape::kK * Shape::kN> b;
  };

  // A[M,K] is row major. B is either column major [N,K] in physical
  // storage, or row major [K,N] (PV). C[M,N] is row major and already contains
  // its bias/residual seed. C must not alias A or B; all 32 lanes participate.
  CUTLASS_DEVICE
  void operator()(SharedStorage& shared, Element const* a, int lda,
                  Element const* b, int ldb, bool row_major_b,
                  Element* c, int ldc, int m, int n, int k) const {
    assert(m > 0 && m % Shape::kM == 0 && n > 0 && n % Shape::kN == 0 &&
           k > 0 && k % InstructionShape::kK == 0);
    assert(lda >= k && ldc >= n && ldb >= (row_major_b ? n : k));
    int lane = int(threadIdx.x) & 31;
    WarpMma mma;
    SmemIteratorA destination_a({shared.a.data(), LayoutA(Shape::kK)}, lane);
    SmemIteratorB destination_b({shared.b.data(), LayoutB(Shape::kK)}, lane);
    auto offset_a = ThreadMapA::initial_offset(lane);
    auto offset_b = ThreadMapB::initial_offset(lane);

    for (int row = 0; row < m; row += Shape::kM) {
      for (int column = 0; column < n; column += Shape::kN) {
        IteratorC destination_c({c + row * ldc + column, RowMajor(ldc)}, lane);
        FragmentC accum;
        destination_c.load(accum);
        for (int reduction = 0; reduction < k; reduction += Shape::kK) {
          StagingFragmentA staged_a;
          StagingFragmentB staged_b;
          int remaining = k - reduction;
          if (remaining >= Shape::kK) {
            SourceIteratorA source_a(
                {const_cast<Element*>(a + row * lda + reduction), RowMajor(lda)}, lane);
            source_a.load(staged_a);
          } else {
            // The boundary fragment uses the same official ThreadMap as its
            // RegularTileIterator. Zero padding protects the ldmatrix reads.
            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < ThreadMapA::Iterations::kStrided; ++s) {
              CUTLASS_PRAGMA_UNROLL
              for (int v = 0; v < ThreadMapA::kElementsPerAccess; ++v) {
                int kk = offset_a.contiguous() + v;
                int mm = offset_a.strided() + s * ThreadMapA::Delta::kStrided;
                staged_a[s * ThreadMapA::kElementsPerAccess + v] = kk < remaining
                    ? a[(row + mm) * lda + reduction + kk] : Element(0);
              }
            }
          }
          if (!row_major_b && remaining >= Shape::kK) {
            SourceIteratorB source_b(
                {const_cast<Element*>(b + column * ldb + reduction), ColumnMajor(ldb)}, lane);
            source_b.load(staged_b);
          } else {
            // PV's B has K-contiguous tensor coordinates but N-contiguous
            // physical storage. Gather into the official column-major fragment;
            // its shared swizzle is still entirely owned by SmemIteratorB.
            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < ThreadMapB::Iterations::kStrided; ++s) {
              CUTLASS_PRAGMA_UNROLL
              for (int v = 0; v < ThreadMapB::kElementsPerAccess; ++v) {
                int kk = offset_b.contiguous() + v;
                int nn = column + offset_b.strided() + s * ThreadMapB::Delta::kStrided;
                int index = row_major_b ? (reduction + kk) * ldb + nn
                                       : nn * ldb + reduction + kk;
                staged_b[s * ThreadMapB::kElementsPerAccess + v] =
                    kk < remaining ? b[index] : Element(0);
              }
            }
          }
          destination_a.store(staged_a);
          destination_b.store(staged_b);
          __syncwarp();
          IteratorA iterator_a({shared.a.data(), LayoutA(Shape::kK)}, lane);
          IteratorB iterator_b({shared.b.data(), LayoutB(Shape::kK)}, lane);
          FragmentA fragment_a;
          FragmentB fragment_b;
          CUTLASS_PRAGMA_UNROLL
          for (int inner = 0; inner < Shape::kK; inner += InstructionShape::kK) {
            if (inner < remaining) {
              iterator_a.load(fragment_a);
              iterator_b.load(fragment_b);
              ++iterator_a;
              ++iterator_b;
              mma(accum, fragment_a, fragment_b, accum);
            }
          }
          __syncwarp();  // Finish operand reads before reusing the scratch.
        }
        destination_c.store(accum);
        __syncwarp();
      }
    }
  }
};

}  // namespace tiny_cutlass::swin::encoder::threadblock
