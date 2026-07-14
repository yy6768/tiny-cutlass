#pragma once

/*
  Epilogue output tile iterator for conv1x1 -> nearest-neighbor SxS upsample.

  Key identity: a 1x1 convolution is pointwise in space, so

      upsample(conv1x1(x))[n, U*p+dh, U*q+dw, k] = conv1x1(x)[n, p, q, k]
      for all (dh, dw) in {0..U-1} x {0..U-1}

  (nearest-neighbor upsample by integer factor U just replicates each output
  pixel into a U x U block). So instead of computing the conv at the upsampled
  resolution, we compute it ONCE at the low resolution and broadcast each
  low-res output pixel to its U x U high-res block. This is a single ordinary
  implicit-GEMM conv1x1; all the upsample logic lives here, in the store path
  of the epilogue output iterator.

  This is a near-verbatim copy of CUTLASS
  cutlass/epilogue/threadblock/predicated_tile_iterator_conv.h
  (PredicatedTileIteratorConv, the StrideSupport::kStrided fprop output
  iterator that already decomposes a GEMM-M row into (n, p, q) conv
  coordinates via fast-divmods). Two changes:

    1. The Params fast-divmods are built from the LOW-resolution (P, Q) extent
       (the resolution the conv actually runs at), while tensor_stride carries
       the HIGH-resolution output tensor's NHWC strides. The host side passes a
       low-res Conv2dProblemSize but a high-res output TensorRef, so CUTLASS's
       ConvOutputIteratorParameter wires exactly this split for us -- see
       conv1x1_upsample/device/conv1x1_upsample.h.

    2. store_with_byte_offset() maps the decomposed low-res coordinate
       (q, p, n) to the base high-res address (n, U*p, U*q, k) and then writes
       the SAME accumulator fragment to all U x U corners
       (n, U*p+dh, U*q+dw, k). load_with_byte_offset() is unchanged: it is used
       to read the bias/source tensor C, which the host binds with zero spatial
       strides so every (p, q) reads bias[k] -- the low-res decomposition reads
       the right channel without any upsample-awareness.

  Because the conv is computed at low resolution, correctness requires the
  output extent to be exactly U * (low-res extent); the host enforces this.
*/

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/permute.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/fast_math.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/epilogue/threadblock/output_tile_thread_map.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_params.h"
#include "cutlass/conv/conv2d_problem_size.h"

////////////////////////////////////////////////////////////////////////////////

namespace tiny_cutlass::conv_fused::conv1x1_upsample::epilogue {

// This iterator is a near-verbatim copy of cutlass's PredicatedTileIteratorConv
// (which lives in namespace cutlass::epilogue::threadblock). Pull those two
// scopes in so the copied body's unqualified names (layout::TensorNHWC,
// FastDivmod, Array, PredicatedTileIteratorParams, make_OutputTileThreadMapDesc,
// CoordinateDecompositionLittleEndian, dot, ...) resolve as they did upstream.
using namespace cutlass;
using namespace cutlass::epilogue::threadblock;

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator that stores each low-resolution conv1x1 output pixel into a
/// UpsampleH x UpsampleW block of a higher-resolution NHWC tensor
/// (nearest-neighbor upsampling), and loads the source/bias tensor with the
/// ordinary conv fprop coordinate mapping.
///
/// Satisfies: ReadableTileIterator | ForwardTileIterator
template <
  typename ThreadMap_,       ///< Thread map (concept: OutputTileThreadMap)
  typename Element_,         ///< Element data type
  int UpsampleH,             ///< Nearest-neighbor upsample factor along H (rows / P)
  int UpsampleW              ///< Nearest-neighbor upsample factor along W (cols / Q)
>
class PredicatedTileIteratorUpsample {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;

  static int const kUpsampleH = UpsampleH;
  static int const kUpsampleW = UpsampleW;

  // Output tensor is NHWC (rank-4); the conv coordinate decomposition below
  // assumes the 3-element NHWC stride vector [stride_w=C, stride_h=W*C,
  // stride_n=H*W*C].
  using Layout = layout::TensorNHWC;

  using Stride = typename Layout::Stride;
  static int const kStrideRank = Layout::kStrideRank;

  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using MappedLayout = layout::RowMajor;
  using Index = typename MappedLayout::Index;
  using LongIndex = typename MappedLayout::LongIndex;
  using TensorCoord = typename MappedLayout::TensorCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  static_assert(UpsampleH >= 1, "UpsampleH must be >= 1");
  static_assert(UpsampleW >= 1, "UpsampleW must be >= 1");

  static_assert( ThreadMap::Iterations::kRow > 0,"ThreadMap::Iterations::kRow must be > 0");
  static_assert( ThreadMap::Iterations::kGroup > 0,"ThreadMap::Iterations::kGroup must be > 0");
  static_assert( ThreadMap::Iterations::kCluster > 0,"ThreadMap::Iterations::kCluster must be > 0");
  static_assert( ThreadMap::Iterations::kColumn > 0,"ThreadMap::Iterations::kColumn must be > 0");

  /// Fragment object
  using Fragment = Array<
    Element,
    ThreadMap::Iterations::kColumn *
    ThreadMap::Iterations::kRow *
    ThreadMap::Iterations::kGroup *
    ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

  /// Memory access size
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  //
  // Parameters struct
  //

  /// Uses a non-template class
  struct Params : PredicatedTileIteratorParams {
    using Base = PredicatedTileIteratorParams;

    /// Fast divmod objects divided by the LOW-resolution output extents.
    /// divmod[0] = Q (low-res width), divmod[1] = P (low-res height).
    FastDivmod divmod[kStrideRank - 1];

    /// Strides of the HIGH-resolution output tensor (NHWC:
    /// [stride_w, stride_h, stride_n]).
    Stride tensor_stride;

    CUTLASS_HOST_DEVICE
    Params() { }

    /// layout: high-resolution output tensor layout (its NHWC strides).
    /// low_res_extent: (m, n) = (N*P*Q, K) at the LOW resolution -- the same
    ///   coordinate the accumulator rows decompose against. tensor_extent[1]
    ///   is low-res P, tensor_extent[2] is low-res Q, following
    ///   ConvOutputIteratorParameter::extent which returns
    ///   implicit_gemm_problem_size(...).mn() with P, Q taken from the
    ///   (low-res) Conv2dProblemSize.
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout, cutlass::Tensor4DCoord const &low_res_extent):
      PredicatedTileIteratorParams(
        // Row stride between adjacent low-res GEMM-M rows is not used for
        // addressing here (store recomputes from divmods), but the base class
        // still wants a stride; use the high-res W-stride so packed-layout
        // helpers stay consistent.
        layout.stride()[0] * int(sizeof(AccessType)) / kElementsPerAccess,
        make_OutputTileThreadMapDesc<ThreadMap>()
      ) {
      divmod[0] = FastDivmod(low_res_extent[2] /* low-res Q */);
      divmod[1] = FastDivmod(low_res_extent[1] /* low-res P */);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kStrideRank; ++i) {
        tensor_stride[i] = layout.stride()[i];
      }
    }

    CUTLASS_HOST_DEVICE
    Params(Base const &base) :
      Base(base) { }
  };

  /// Mask object
  struct Mask {

    static int const kCount = ThreadMap::Iterations::kColumn;

    /// Predicate state
    bool predicates[kCount];

    CUTLASS_HOST_DEVICE
    Mask() {
      enable();
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_HOST_DEVICE void clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = false;
      }
    }

    ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
    CUTLASS_DEVICE void enable() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = true;
      }
    }
  };

private:

  //
  // Data members
  //

  /// Parameters structure containing reference and precomputed state.
  Params params_;

  /// Byte-level pointer.
  uint8_t *byte_pointer_;

  /// Array of boolean values to contain steady-state predicates
  Mask mask_;

  /// Extent of the matrix tile in rows (low-res GEMM-M = N*P*Q)
  Index extent_row_;

  /// Extent of the matrix tile in columns (K)
  Index extent_column_;

  /// A thread's starting row position (assuming steady-state predicates have been computed)
  Index thread_start_row_;

  /// A thread's starting column
  Index thread_start_column_;

  /// Internal state counter
  int state_[3];

  //
  // Static asserts about internal strides
  //

  static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(PredicatedTileIteratorParams::stride) == 8, "Expected 64b strides");

public:

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  PredicatedTileIteratorUpsample(
    Params const & params,
    Element *pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset = TensorCoord()
  ):
    params_(params)
  {

    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    extent_row_ = extent.row();
    extent_column_ = extent.column();

    thread_start_row_ = thread_offset.row();
    thread_start_column_ = thread_offset.column();

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {

      mask_.predicates[c] = ((thread_offset.column()
        + ThreadMap::Delta::kColumn * c) < extent.column());
    }

    // Null pointer performs no accesses
    if (!pointer) {
      mask_.clear();
    }

    // Initialize byte_pointer_
    byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
      LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;

    // Initialize internal state counter
    state_[0] = state_[1] = state_[2] = 0;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  /// Loads a fragment from memory (source/bias tensor C). Unchanged from the
  /// stock conv output iterator: the decomposed low-res coordinate maps
  /// through tensor_stride, which for the bias tensor the host binds with zero
  /// spatial strides so every (p, q) reads bias[k].
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, int64_t byte_offset) const {

    uint8_t *byte_pointer = byte_pointer_;
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx =
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow
            + group * ThreadMap::Delta::kGroup
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          Stride tensor_coord = CoordinateDecompositionLittleEndian<kStrideRank>(row_offset + thread_start_row_, params_.divmod);

          LongIndex tensor_offset = dot(tensor_coord, params_.tensor_stride);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            cutlass::arch::global_load<
              AccessType,
              sizeof(AccessType)
            >(
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn +
                         column],
                (void *)&memory_pointer[column * ThreadMap::Delta::kColumn /
                                        kElementsPerAccess + tensor_offset / kElementsPerAccess],
                guard);
          }
        }
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) const {

    load_with_byte_offset(frag, 0);
  }

  /// Stores a fragment to memory, broadcasting each low-res output pixel
  /// (n, p, q, k) to its UpsampleH x UpsampleW high-res block
  /// (n, U*p + dh, U*q + dw, k).
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, int64_t byte_offset) const {
    uint8_t *byte_pointer = byte_pointer_;
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    // High-res NHWC strides: tensor_stride = [stride_w, stride_h, stride_n].
    LongIndex const stride_w = params_.tensor_stride[0];
    LongIndex const stride_h = params_.tensor_stride[1];

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx =
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow
            + group * ThreadMap::Delta::kGroup
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          // Decompose the low-res GEMM-M row into (n, p, q). coord[0] = q
          // (low-res), coord[1] = p (low-res), coord[2] = n.
          Stride coord = CoordinateDecompositionLittleEndian<kStrideRank>(
              (row_offset + thread_start_row_), params_.divmod);

          // Base address of the high-res block top-left corner:
          // (n, U*p, U*q, k). tensor_stride already carries high-res strides,
          // so scale the spatial coordinates by the upsample factors.
          Stride hi_coord = coord;
          hi_coord[0] = coord[0] * kUpsampleW;  // U*q along W
          hi_coord[1] = coord[1] * kUpsampleH;  // U*p along H

          LongIndex base_offset = dot(hi_coord, params_.tensor_stride);

          CUTLASS_PRAGMA_UNROLL
          for (int dh = 0; dh < kUpsampleH; ++dh) {
            CUTLASS_PRAGMA_UNROLL
            for (int dw = 0; dw < kUpsampleW; ++dw) {

              LongIndex corner_offset = base_offset + dh * stride_h + dw * stride_w;

              AccessType *memory_pointer =
                  reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

              CUTLASS_PRAGMA_UNROLL
              for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

                bool guard = row_guard && mask_.predicates[column];

                cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                    frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                    (void *)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess
                                            + corner_offset / kElementsPerAccess],
                    guard);
              }
            }
          }
        }
      }
    }
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) const {

    store_with_byte_offset(frag, 0);
  }

  CUTLASS_DEVICE
  MatrixCoord thread_start() const {
    return MatrixCoord(thread_start_row_, thread_start_column_);
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_row() const {
    return thread_start_row_;
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_column() const {
    return thread_start_column_;
  }

  /// Extent of the matrix in rows
  CUTLASS_DEVICE
  Index extent_row() const {
    return extent_row_;
  }

  /// Extent of the matrix in columns
  CUTLASS_DEVICE
  Index extent_column() const {
    return extent_column_;
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorUpsample &operator++() {

    ++state_[0];

    thread_start_row_ += ThreadMap::Shape::kRow;

    if (state_[0] == ThreadMap::Count::kRow) {

      state_[0] = 0;
      ++state_[1];

      thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
        ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {

        state_[1] = 0;
        ++state_[2];

        thread_start_row_ += ThreadMap::Count::kGroup *
          ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;

          thread_start_row_ += ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow
            * ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile;
        }
      }
    }

    return *this;
  }

  ///< Efficiently disables all accesses guarded by mask
  CUTLASS_DEVICE void clear_mask() {
    mask_.clear();
  }

  ///< Efficiently enables all accesses guarded by mask
  CUTLASS_DEVICE void enable_mask() {
    mask_.enable();
  }

  ///< Sets the mask
  CUTLASS_DEVICE void get_mask(Mask &mask) const {
    mask = mask_;
  }

  ///< Sets the mask
  CUTLASS_DEVICE void set_mask(Mask const &mask) {
    mask_ = mask;
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace tiny_cutlass::conv_fused::conv1x1_upsample::epilogue
