#pragma once

/*
  Host-side table builders for the Swin window mechanics.

  ---------------------------------------------------------------------------
  WHY THESE ARE TABLES AND NOT KERNELS
  ---------------------------------------------------------------------------
  window_partition and window_reverse are pure data movement: memory bound, two
  wasted HBM round trips, and no arithmetic to hide them behind. CUTLASS 2.x
  already exposes exactly the hook needed to avoid them -- GemmUniversal takes
  GatherA / GatherB / ScatterD template flags (gemm_universal.h:130-136), and
  the gather is a row remap:

      predicated_tile_access_iterator.h:575
          if (Gather) coord_strided = indices_[coord_strided];

  For a row-major A operand the strided rank IS the GEMM M dimension, so one
  index per row. A Swin token's C channels are contiguous, so a token is exactly
  one row -- window partition becomes GatherA on the QKV GEMM and window reverse
  becomes ScatterD on the projection GEMM. The cyclic shift is baked into the
  table, so the kernel does zero index arithmetic.

  Everything here runs on the host, once per problem shape.

  ---------------------------------------------------------------------------
  COORDINATE CONVENTIONS (must match the reference exactly)
  ---------------------------------------------------------------------------
  Token layout      : [B, H, W, C] row-major, token row = (b * H + i) * W + j
  Window-major rows : row = window_id * L + l,  L = w * w
                      window_id = (b * (H/w) + wi) * (W/w) + wj
                      l         = li * w + lj
  Shift             : Swin rolls the feature map by -shift before partitioning,
                      so a window-major slot at rolled coord (i, j) reads the
                      token at original coord ((i + shift) % H, (j + shift) % W).
*/

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "swin/swin_problem.h"

namespace tiny_cutlass::swin {

// ---------------------------------------------------------------------------
// Row remap for window partition (GatherA) and window reverse (ScatterD).
//
// Returns a table of num_window_rows() entries mapping a window-major row to a
// token row in the [B, H, W] grid. The SAME table serves both directions:
// GatherA reads token[idx[m]] into GEMM row m, ScatterD writes GEMM row m out
// to token[idx[m]]. That symmetry is why residual1 can be read back through the
// epilogue's source-C path with this identical index array.
// ---------------------------------------------------------------------------
inline std::vector<int> build_window_row_index(SwinStageProblem const& problem) {
  int const w = problem.effective_window();
  int const shift = problem.effective_shift();
  int const H = problem.height;
  int const W = problem.width;

  assert(problem.window_tiles_exactly() && "window must tile the token grid");

  int const wins_col = problem.windows_per_col();  // H / w
  int const wins_row = problem.windows_per_row();  // W / w

  std::vector<int> index(size_t(problem.num_window_rows()));

  for (int b = 0; b < problem.batch; ++b) {
    for (int wi = 0; wi < wins_col; ++wi) {
      for (int wj = 0; wj < wins_row; ++wj) {
        int const window_id = (b * wins_col + wi) * wins_row + wj;
        for (int li = 0; li < w; ++li) {
          for (int lj = 0; lj < w; ++lj) {
            // Rolled (post-shift) coordinate this slot occupies.
            int const i = wi * w + li;
            int const j = wj * w + lj;
            // Original coordinate it reads from.
            int const h = (i + shift) % H;
            int const v = (j + shift) % W;

            int const row = window_id * (w * w) + (li * w + lj);
            index[size_t(row)] = (b * H + h) * W + v;
          }
        }
      }
    }
  }

  return index;
}

// ---------------------------------------------------------------------------
// Naive references for the two ops the tables replace. Step 0 checks the table
// against these element by element; later steps keep them as the parity oracle.
// ---------------------------------------------------------------------------
template <typename Element>
void window_partition_reference(
    SwinStageProblem const& problem,
    std::vector<Element> const& tokens,   // [B, H, W, C]
    std::vector<Element>& windows) {      // [num_windows * L, C]
  int const C = problem.channels;
  std::vector<int> const index = build_window_row_index(problem);
  windows.resize(index.size() * size_t(C));

  for (size_t row = 0; row < index.size(); ++row) {
    for (int c = 0; c < C; ++c) {
      windows[row * size_t(C) + size_t(c)] =
          tokens[size_t(index[row]) * size_t(C) + size_t(c)];
    }
  }
}

template <typename Element>
void window_reverse_reference(
    SwinStageProblem const& problem,
    std::vector<Element> const& windows,  // [num_windows * L, C]
    std::vector<Element>& tokens) {        // [B, H, W, C]
  int const C = problem.channels;
  std::vector<int> const index = build_window_row_index(problem);
  tokens.assign(size_t(problem.num_tokens()) * size_t(C), Element(0.0f));

  for (size_t row = 0; row < index.size(); ++row) {
    for (int c = 0; c < C; ++c) {
      tokens[size_t(index[row]) * size_t(C) + size_t(c)] =
          windows[row * size_t(C) + size_t(c)];
    }
  }
}

// ---------------------------------------------------------------------------
// Shifted-window attention mask classes.
//
// Only windows touching the wrapped edge carry a mask, so the mask depends on a
// window's (last-row?, last-col?) status -- four classes at most, one when
// shift == 0. Within a window, a slot is labelled by which of three ranges its
// ROLLED coordinate falls in:
//
//      i <  H - w        -> 0   (interior; always unwrapped)
//      H - w <= i < H-s  -> 1   (in the edge window, still unwrapped)
//      i >= H - s        -> 2   (in the edge window, wrapped around)
//
// Labelling by the rolled index with those boundaries is exactly equivalent to
// grouping by wrap status, which is the trick official Swin uses when it
// partitions an UNROLLED img_mask. Two slots may attend iff their labels match.
// ---------------------------------------------------------------------------
inline int window_mask_class_count(SwinStageProblem const& problem) {
  return problem.effective_shift() == 0 ? 1 : 4;
}

// Class of a given window: 0 when shift == 0, else 2 * (last row) + (last col).
inline int window_mask_class(SwinStageProblem const& problem, int window_id) {
  if (problem.effective_shift() == 0) {
    return 0;
  }
  int const wins_col = problem.windows_per_col();
  int const wins_row = problem.windows_per_row();
  int const wj = window_id % wins_row;
  int const wi = (window_id / wins_row) % wins_col;
  return 2 * int(wi == wins_col - 1) + int(wj == wins_row - 1);
}

namespace detail {

// Range label of a rolled coordinate within the last window along one axis.
inline int shift_range_label(int local, int window, int shift) {
  // `local` is the offset inside the window, i.e. rolled coord - (extent - w).
  return local < window - shift ? 0 : 1;
}

}  // namespace detail

// Builds the additive attention mask for every class:
//   [num_class, L, L], 0.0f where attention is allowed, kMaskValue where not.
//
// kMaskValue is -1e4 rather than -inf: the scores are accumulated in fp32 but
// may be stored as fp16, where -inf propagates NaN through the softmax's
// max-subtraction on a fully masked row.
inline std::vector<float> build_window_attention_mask(
    SwinStageProblem const& problem, float mask_value = -1.0e4f) {
  int const w = problem.effective_window();
  int const L = w * w;
  int const shift = problem.effective_shift();
  int const classes = window_mask_class_count(problem);

  std::vector<float> mask(size_t(classes) * size_t(L) * size_t(L), 0.0f);
  if (shift == 0) {
    return mask;
  }

  for (int cls = 0; cls < classes; ++cls) {
    bool const last_row = (cls & 2) != 0;
    bool const last_col = (cls & 1) != 0;

    // Region label of each slot in this window class.
    // (Braced init, not `label(size_t(L))` -- that is a most-vexing-parse and
    // declares a function.)
    std::vector<int> label(static_cast<size_t>(L), 0);
    for (int li = 0; li < w; ++li) {
      for (int lj = 0; lj < w; ++lj) {
        int const hl = last_row ? detail::shift_range_label(li, w, shift) : 0;
        int const wl = last_col ? detail::shift_range_label(lj, w, shift) : 0;
        label[size_t(li * w + lj)] = hl * 2 + wl;
      }
    }

    for (int a = 0; a < L; ++a) {
      for (int b = 0; b < L; ++b) {
        if (label[size_t(a)] != label[size_t(b)]) {
          mask[(size_t(cls) * size_t(L) + size_t(a)) * size_t(L) + size_t(b)] =
              mask_value;
        }
      }
    }
  }

  return mask;
}

// ---------------------------------------------------------------------------
// Relative position bias index: [L, L] into a (2w-1)^2 table.
//
//   index[a][b] = (dy + w - 1) * (2w - 1) + (dx + w - 1)
//   where a = ya * w + xa, b = yb * w + xb, dy = ya - yb, dx = xa - xb.
// ---------------------------------------------------------------------------
inline std::vector<int> build_relative_position_index(
    SwinStageProblem const& problem) {
  int const w = problem.effective_window();
  int const L = w * w;
  int const span = 2 * w - 1;

  std::vector<int> index(size_t(L) * size_t(L));
  for (int a = 0; a < L; ++a) {
    int const ya = a / w;
    int const xa = a % w;
    for (int b = 0; b < L; ++b) {
      int const yb = b / w;
      int const xb = b % w;
      index[size_t(a) * size_t(L) + size_t(b)] =
          (ya - yb + w - 1) * span + (xa - xb + w - 1);
    }
  }
  return index;
}

// ---------------------------------------------------------------------------
// The one table the kernel actually reads: relative position bias folded
// together with the shift mask, laid out [num_class, heads, L, L] in fp32.
//
// `bias_table` is the checkpoint's relative_position_bias_table, shaped
// [(2w-1)^2, num_heads] row-major (the official layout).
//
// Folding the two means the softmax adds a single contiguous [L, L] tile per
// (window class, head) and does no index arithmetic at all. At window = 4,
// heads = 3 that whole table is 4 * 3 * 16 * 16 * 4B = 12 KB: L2 resident.
// ---------------------------------------------------------------------------
inline std::vector<float> build_attention_bias_table(
    SwinStageProblem const& problem,
    std::vector<float> const& bias_table,
    float mask_value = -1.0e4f) {
  int const w = problem.effective_window();
  int const L = w * w;
  int const heads = problem.num_heads;
  int const classes = window_mask_class_count(problem);

  std::vector<int> const rel_index = build_relative_position_index(problem);
  std::vector<float> const mask = build_window_attention_mask(problem, mask_value);

  std::vector<float> table(
      size_t(classes) * size_t(heads) * size_t(L) * size_t(L), 0.0f);

  for (int cls = 0; cls < classes; ++cls) {
    for (int h = 0; h < heads; ++h) {
      for (int a = 0; a < L; ++a) {
        for (int b = 0; b < L; ++b) {
          size_t const ab = size_t(a) * size_t(L) + size_t(b);
          float bias = 0.0f;
          if (!bias_table.empty()) {
            bias = bias_table[size_t(rel_index[ab]) * size_t(heads) + size_t(h)];
          }
          float const m = mask[size_t(cls) * size_t(L) * size_t(L) + ab];
          table[((size_t(cls) * size_t(heads) + size_t(h)) * size_t(L) *
                 size_t(L)) +
                ab] = bias + m;
        }
      }
    }
  }

  return table;
}

// ---------------------------------------------------------------------------
// PatchMerging gather: one output token concatenates a 2x2 spatial neighbourhood
// along the channel axis. Official order is
//
//     cat([x[0::2, 0::2], x[1::2, 0::2], x[0::2, 1::2], x[1::2, 1::2]], -1)
//
// i.e. channel block k corresponds to (di, dj) = (0,0), (1,0), (0,1), (1,1).
// Returned table is [4][num_out_tokens]: block k's gather index for the
// k-th C-wide slice of the concatenated 4C row.
// ---------------------------------------------------------------------------
inline std::vector<int> build_patch_merging_index(
    PatchMergingProblem const& problem) {
  int const H = problem.height;
  int const W = problem.width;
  int const OH = problem.out_height();
  int const OW = problem.out_width();

  int const di[4] = {0, 1, 0, 1};
  int const dj[4] = {0, 0, 1, 1};

  std::vector<int> index(size_t(4) * size_t(problem.num_out_tokens()));

  for (int k = 0; k < 4; ++k) {
    for (int b = 0; b < problem.batch; ++b) {
      for (int i = 0; i < OH; ++i) {
        for (int j = 0; j < OW; ++j) {
          int const out_row = (b * OH + i) * OW + j;
          int const h = 2 * i + di[k];
          int const v = 2 * j + dj[k];
          index[size_t(k) * size_t(problem.num_out_tokens()) + size_t(out_row)] =
              (b * H + h) * W + v;
        }
      }
    }
  }

  return index;
}

}  // namespace tiny_cutlass::swin
