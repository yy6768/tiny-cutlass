#pragma once

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"

// One collective asynchronous gmem->smem copy. Input is always
// [sequence, head-dimension]. K's shared view is transposed logically.
template <class Element, class SmemLayout, int Rows, int Columns,
          int Threads, bool Transpose = false>
struct FlashAttnTileLoader {
  using Shape = cutlass::layout::PitchLinearShape<Columns, Rows>;
  using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
      Shape, Threads, cutlass::layout::PitchLinearShape<8, 4>, 8>;
  using AccessType = cutlass::Array<Element, 8>;
  using Iterator = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      Shape, Element const, cutlass::layout::PitchLinear, 1, ThreadMap, AccessType>;
  using SmemShape = cutlass::MatrixShape<Transpose ? Columns : Rows, Transpose ? Rows : Columns>;
  using SmemIterator = cutlass::transform::threadblock::RegularTileAccessIterator<
      SmemShape, Element, SmemLayout, Transpose ? 1 : 0, ThreadMap>;

  CUTLASS_DEVICE
  static void copy(Iterator input, Element* destination, int thread) {
    // Copy a tile without advancing the caller's global iterator. The mainloop
    // advances K/V after issuing each copy; Q stays at its initial tile.
    SmemIterator output({destination, SmemLayout(Columns)}, thread);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ThreadMap::Iterations::kCount; ++i) {
      cutlass::arch::cp_async_zfill<sizeof(AccessType), cutlass::arch::CacheOperation::Global>(
          output.get(), input.get(), input.valid());
      ++input;
      ++output;
    }
  }
};
